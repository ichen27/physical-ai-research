"""How late the camera's pictures are, measured by moving the arm by hand and seeing when the picture follows.

With the arm limp and the camera looking at the desk, the swivel is swung back and forth
by hand. The joint's readings say when the arm moved; the pictures say when the view
moved. The time by which the second trails the first is the lateness of the whole chain a
policy driven by pictures would see: the camera, the program reading it, the pipe and our
own read. ``scripts/stream_lateness.py`` records the two; the arithmetic is here, where it
can be tested without a camera or an arm.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def sideways_shift(before: np.ndarray, after: np.ndarray) -> float:
    """How far, in pixels, the view moved sideways between two grey pictures of the same size."""
    import cv2

    window = cv2.createHanningWindow((before.shape[1], before.shape[0]), cv2.CV_32F)
    (shift, _), _ = cv2.phaseCorrelate(before.astype(np.float32), after.astype(np.float32), window)
    return float(shift)


def rate(times: Sequence[float], values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """How fast ``values`` were changing, and the times those rates belong to (half way between readings)."""
    times, values = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    return (times[1:] + times[:-1]) / 2.0, np.diff(values) / np.maximum(np.diff(times), 1e-6)


def lag(cause_times: Sequence[float], cause: Sequence[float], effect_times: Sequence[float], effect: Sequence[float],
        most_s: float = 0.6, grid_s: float = 0.005) -> tuple[float, float]:
    """By how many seconds ``effect`` trails ``cause``, and how alike the two are when lined up (0 to 1).

    Both are sampled at their own uneven times. Neither the size nor the sign of the effect
    has to be known: a view can move left or right, and by any number of pixels, for a given
    turn of the arm. Lags from zero to ``most_s`` are tried on a grid of ``grid_s``, and the
    best is refined between grid points.
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
    alike = []
    for by in lags:
        moved = np.interp(grid - by, cause_times, cause)
        moved = moved - moved.mean()
        alike.append(abs(float(moved @ seen)) / max(float(np.linalg.norm(moved) * np.linalg.norm(seen)), 1e-12))
    alike = np.asarray(alike)
    best = int(alike.argmax())
    by = float(lags[best])
    if 0 < best < len(lags) - 1:  # the top of a parabola through the best and its neighbours
        a, b, c = alike[best - 1], alike[best], alike[best + 1]
        if a - 2 * b + c < 0:
            by += grid_s * 0.5 * (a - c) / (a - 2 * b + c)
    return by, float(alike[best])


def lag_by_stretch(cause_times, cause, effect_times, effect, stretch_s: float = 2.5, most_s: float = 0.6) -> list[tuple[float, float, float]]:
    """``lag`` over successive stretches of the recording: for each, when it began, the lag, and how alike the two were."""
    cause_times, effect_times = np.asarray(cause_times, dtype=float), np.asarray(effect_times, dtype=float)
    cause, effect = np.asarray(cause, dtype=float), np.asarray(effect, dtype=float)
    out = []
    begin = max(cause_times[0], effect_times[0])
    while begin + stretch_s + most_s <= min(cause_times[-1], effect_times[-1]) + 1e-9:
        stop = begin + stretch_s + most_s
        a = (cause_times >= begin) & (cause_times <= stop)
        b = (effect_times >= begin) & (effect_times <= stop)
        try:
            by, alike = lag(cause_times[a], cause[a], effect_times[b], effect[b], most_s)
            out.append((float(begin), by, alike))
        except ValueError:
            pass
        begin += stretch_s
    return out
