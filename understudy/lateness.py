"""How late the camera's pictures are, measured by moving the arm by hand and seeing when the picture follows.

With the arm limp and the camera looking at the desk, the swivel is swung back and forth
by hand. The joint's readings say when the arm moved; the pictures say when the view
moved. The time by which the second trails the first is the lateness of the whole chain a
policy driven by pictures would see: the camera, the program reading it, the pipe and our
own read. ``scripts/stream_lateness.py`` records the two; the arithmetic is here, where it
can be tested without a camera or an arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def view_shift(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    """How far, in pixels, the view moved between two grey pictures of the same size: sideways, and down."""
    import cv2

    window = cv2.createHanningWindow((before.shape[1], before.shape[0]), cv2.CV_32F)
    (across, down), _ = cv2.phaseCorrelate(before.astype(np.float32), after.astype(np.float32), window)
    return float(across), float(down)


def main_direction(across: Sequence[float], down: Sequence[float]) -> np.ndarray:
    """The view's position along the one direction it mostly moved in, whichever way up the camera is mounted."""
    track = np.column_stack([np.asarray(across, dtype=float), np.asarray(down, dtype=float)])
    steps = np.diff(track, axis=0)
    if len(steps) < 2:
        return track[:, 0]
    _, _, directions = np.linalg.svd(steps - steps.mean(axis=0), full_matrices=False)
    return track @ directions[0]


def rate(times: Sequence[float], values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """How fast ``values`` were changing, and the times those rates belong to (half way between readings)."""
    times, values = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    return (times[1:] + times[:-1]) / 2.0, np.diff(values) / np.maximum(np.diff(times), 1e-6)


@dataclass(frozen=True)
class Lag:
    seconds: float  # by how long the effect trails the cause
    alike: float  # how alike the two are when lined up that way, 0 to 1
    rival: float  # how alike they are at the best other way of lining them up
    sign: int  # +1 if the effect rises when the cause does, -1 if it falls
    sharp: float = 1.0  # how much less alike they are 50 ms either side of the best

    def clear(self, least: float = 0.6, margin: float = 0.15, sharp: float = 0.03) -> bool:
        """Whether this is an answer: the two line up well, no other way of lining them up comes
        close, and 50 ms either way is plainly worse.

        A steady swing back and forth lines up as well half a swing later, upside down, as it
        does at the true lag, and a whole swing later the right way up. Then there is no
        telling which is the lateness, and it must not be guessed. And a slow, smooth swing
        lines up nearly as well tens of milliseconds either side of the truth: measured, a
        swing every 2.6 s was up to 21 ms out and one every 3.5 s up to 37 ms, where quick
        turns with pauses were within 3 ms.
        """
        return self.alike >= least and self.alike - self.rival >= margin and self.sharp >= sharp


def lag(cause_times: Sequence[float], cause: Sequence[float], effect_times: Sequence[float], effect: Sequence[float],
        most_s: float = 1.0, grid_s: float = 0.005, sign: int | None = None) -> Lag:
    """By how long ``effect`` trails ``cause``, with how sure that is.

    Both are sampled at their own uneven times. The size of the effect need not be known,
    and its sign need not either: a view can move left or right for a given turn of the arm.
    If ``sign`` is given, only that way up counts. Lags from zero to ``most_s`` are tried on
    a grid of ``grid_s`` and the best refined between grid points. Check ``Lag.clear()``
    before believing the answer.
    """
    cause_times, effect_times = np.asarray(cause_times, dtype=float), np.asarray(effect_times, dtype=float)
    start = max(cause_times[0], effect_times[0]) + most_s
    end = min(cause_times[-1], effect_times[-1])
    if end - start < 1.0:
        raise ValueError("less than a second of the two overlap")
    grid = np.arange(start, end, grid_s)
    seen = np.interp(grid, effect_times, effect)
    seen = seen - seen.mean()
    lags = np.arange(0.0, most_s + grid_s / 2, grid_s)
    curve = []
    for by in lags:
        moved = np.interp(grid - by, cause_times, cause)
        moved = moved - moved.mean()
        curve.append(float(moved @ seen) / max(float(np.linalg.norm(moved) * np.linalg.norm(seen)), 1e-12))
    curve = np.asarray(curve)
    score = np.abs(curve) if sign is None else sign * curve
    best = int(score.argmax())
    by = float(lags[best])
    if 0 < best < len(lags) - 1:  # the top of a parabola through the best and its neighbours
        a, b, c = score[best - 1], score[best], score[best + 1]
        if a - 2 * b + c < 0:
            by += grid_s * 0.5 * (a - c) / (a - 2 * b + c)
    # Every other peak, the ends of the range included, more than 50 ms from the best.
    padded = np.concatenate([[-np.inf], score, [-np.inf]])
    peaks = np.flatnonzero((padded[1:-1] >= padded[:-2]) & (padded[1:-1] >= padded[2:]))
    others = [float(score[i]) for i in peaks if abs(lags[i] - lags[best]) > 0.05]
    aside = int(round(0.05 / grid_s))
    beside = [float(score[i]) for i in (best - aside, best + aside) if 0 <= i < len(score)]
    return Lag(by, float(score[best]), max(others, default=0.0), int(np.sign(curve[best])) or 1, float(score[best]) - max(beside))


def lag_by_stretch(cause_times, cause, effect_times, effect, sign: int, stretch_s: float = 4.0, most_s: float = 1.0) -> list[tuple[float, Lag]]:
    """``lag`` over successive stretches of the recording, the way up already known: for each, when it began and what was found."""
    cause_times, effect_times = np.asarray(cause_times, dtype=float), np.asarray(effect_times, dtype=float)
    cause, effect = np.asarray(cause, dtype=float), np.asarray(effect, dtype=float)
    out = []
    begin = max(cause_times[0], effect_times[0])
    while begin + stretch_s + most_s <= min(cause_times[-1], effect_times[-1]) + 1e-9:
        stop = begin + stretch_s + most_s
        a = (cause_times >= begin) & (cause_times <= stop)
        b = (effect_times >= begin) & (effect_times <= stop)
        try:
            out.append((float(begin), lag(cause_times[a], cause[a], effect_times[b], effect[b], most_s, sign=sign)))
        except ValueError:
            pass
        begin += stretch_s
    return out
