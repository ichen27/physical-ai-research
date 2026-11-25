"""Find a real arm's joint zero points from touches of the gripper tip on desk marks.

Posing an arm by eye in its zero pose is good to a few degrees, which is
centimetres at the gripper. Touching the closed gripper's tip to marks at
measured places on the desk, with the arm limp, gives two equations per touch
that do not depend on the swivel: the tip's distance from the swivel axis and
its height above the desk. The unknowns are the encoder readings at zero of the
shoulder, elbow and wrist-bend joints. Four or more touches at different
distances and in different arm shapes fix them by least squares, and what is
left over says how good the fit is.

Marks are measured from the swivel axis, the one physical reference that is
easy to find: ``forward`` along a ruler line laid outward from it, ``left``
across that line.

The swivel's zero point is taken from the direction of the marks, averaged over
the touches, and is only as good as the base stayed put: a base that turns a
little on the desk between touches moves it by as much. On the first real arm
two touches of one mark read 12 degrees apart. That is why the other three
joints are fitted without it, and why it need not be exact: a wrong swivel zero
turns the arm's whole frame about the swivel axis, so a ball position that was
itself found by touching the tip to the spot is turned by the same amount.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import arm
from .kinematics import Chain

# The swivel axis in the model's base frame: the vendor URDF's joint1 origin.
SWIVEL_AXIS_X_M = -0.01125
FITTED_JOINTS = (1, 2, 3)  # shoulder, elbow, wrist bend; wrist roll hardly moves the tip
MIN_TOUCHES = 4
DISTINCT_TICKS = 60.0  # two fits further apart than this (about 5 degrees) are different answers


@dataclass(frozen=True)
class Touch:
    """One reading of the five arm servos with the tip on a desk mark."""

    ticks: tuple[int, ...]
    forward_m: float
    left_m: float

    def mark(self) -> tuple[float, float, float]:
        """The mark in the model's base frame."""
        return (self.forward_m + SWIVEL_AXIS_X_M, self.left_m, 0.0)

    def distance_m(self) -> float:
        """How far the mark is from the swivel axis."""
        return math.hypot(self.forward_m, self.left_m)


@dataclass(frozen=True)
class Fit:
    zero_ticks: tuple[float, ...]  # all five arm joints; the last is passed through
    errors_m: tuple[tuple[float, float], ...]  # per touch: model minus mark, in distance and in height
    rms_m: float
    runner_up_rms_m: float | None  # best fit that is a different answer, if one was found
    swivel_spread_rad: float  # furthest any touch's direction lies from the swivel zero

    def ambiguous(self) -> bool:
        """True if a clearly different set of zero points fits almost as well."""
        return self.runner_up_rms_m is not None and self.runner_up_rms_m < max(2.0 * self.rms_m, 0.005)


def load_touches(path: str | Path) -> list[Touch]:
    path = Path(path)
    if not path.exists():
        return []
    return [Touch(tuple(row["ticks"]), row["forward_m"], row["left_m"]) for row in json.loads(path.read_text())]


def save_touches(path: str | Path, touches: Sequence[Touch]) -> None:
    Path(path).write_text(json.dumps([asdict(touch) for touch in touches], indent=2) + "\n")


def tip_errors(zero_ticks: Sequence[float], touches: Sequence[Touch], signs: Sequence[int], chain: Chain) -> np.ndarray:
    """Model tip minus mark, three numbers per touch."""
    out = []
    for touch in touches:
        angles = [arm.ticks_to_rad(tick, zero, sign) for tick, zero, sign in zip(touch.ticks, zero_ticks, signs)]
        out.extend(np.subtract(chain.tip_position(angles), touch.mark()))
    return np.array(out)


def reach_errors(zero_ticks: Sequence[float], touches: Sequence[Touch], signs: Sequence[int], chain: Chain) -> np.ndarray:
    """Model minus mark, two numbers per touch: distance from the swivel axis, and height."""
    out = []
    for touch in touches:
        angles = [arm.ticks_to_rad(tick, zero, sign) for tick, zero, sign in zip(touch.ticks, zero_ticks, signs)]
        x, y, z = chain.tip_position(angles)
        out.extend((math.hypot(x - SWIVEL_AXIS_X_M, y) - touch.distance_m(), z))
    return np.array(out)


def _refine(start: np.ndarray, residual) -> tuple[np.ndarray, float]:
    """Damped least squares from one starting guess, in ticks."""
    zero, damping = start.astype(float), 1e-3
    error = residual(zero)
    cost = float(error @ error)
    for _ in range(80):
        jacobian = np.column_stack([(residual(zero + step) - error) / 1.0 for step in np.eye(len(zero))])
        normal = jacobian.T @ jacobian
        move = np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-12), -jacobian.T @ error)
        trial = zero + move
        trial_error = residual(trial)
        trial_cost = float(trial_error @ trial_error)
        if trial_cost < cost:
            converged = cost - trial_cost < 1e-14
            zero, error, cost, damping = trial, trial_error, trial_cost, damping / 3
            if converged:
                break
        else:
            damping *= 10
            if damping > 1e9:
                break
    return zero, cost


def fit(
    touches: Sequence[Touch],
    signs: Sequence[int] = (1, 1, 1, 1, 1),
    roll_zero_tick: float = arm.STOCK_ZERO_TICK,
    chain: Chain | None = None,
) -> Fit:
    """Zero points of the shoulder, elbow and wrist bend that best put the model's tip on the marks."""
    if len(touches) < MIN_TOUCHES:
        raise ValueError(f"{MIN_TOUCHES} or more touches are needed; {len(touches)} given")
    chain = chain or Chain()
    turn = arm.TICKS_PER_REV

    def residual(fitted: np.ndarray) -> np.ndarray:
        return reach_errors([0.0, *fitted, roll_zero_tick], touches, signs, chain)

    # Started from every quarter turn, since a wrong start can settle on a wrong answer.
    quarter_turns = (1024.0, 2048.0, 3072.0, 4096.0)
    found = sorted(
        (
            (cost, tuple(zero))
            for zero, cost in (
                _refine(np.array(start), residual) for start in itertools.product(quarter_turns, repeat=len(FITTED_JOINTS))
            )
        ),
        key=lambda item: item[0],
    )

    def same_pose(a: Sequence[float], b: Sequence[float]) -> bool:
        return all(abs((x - y + turn / 2) % turn - turn / 2) < DISTINCT_TICKS for x, y in zip(a, b))

    best_cost, best = found[0]
    runner_up = next((cost for cost, zero in found[1:] if not same_pose(zero, best)), None)
    count = len(touches)
    # A zero point a whole turn away fits equally well but would report angles a
    # whole turn out. Keep the one that puts the touched poses within half a turn.
    middles = [np.mean([touch.ticks[joint] for touch in touches]) for joint in FITTED_JOINTS]
    best = tuple(zero + turn * round((middle - zero) / turn) for zero, middle in zip(best, middles))

    # The swivel's zero from which way each mark lies, allowing for the model's
    # tip sitting a little to one side of the swivel axis. It played no part above.
    def swivel_zero(touch: Touch) -> float:
        angles = [0.0] + [
            arm.ticks_to_rad(touch.ticks[j], zero, signs[j]) for j, zero in zip((1, 2, 3, 4), (*best, roll_zero_tick))
        ]
        x, y, _ = chain.tip_position(angles)
        turned = math.atan2(touch.left_m, touch.forward_m) - math.atan2(y, x - SWIVEL_AXIS_X_M)
        return touch.ticks[0] - signs[0] * turned / arm.RAD_PER_TICK

    bearings = [swivel_zero(touch) for touch in touches]
    swivel = float(np.mean(bearings))
    swivel_spread = max(abs(b - swivel) for b in bearings) * arm.RAD_PER_TICK
    zero_ticks = (swivel, *[float(v) for v in best], float(roll_zero_tick))
    per_touch = reach_errors(zero_ticks, touches, signs, chain).reshape(count, 2)
    return Fit(
        zero_ticks=zero_ticks,
        errors_m=tuple((float(row[0]), float(row[1])) for row in per_touch),
        rms_m=math.sqrt(best_cost / count),
        runner_up_rms_m=None if runner_up is None else math.sqrt(runner_up / count),
        swivel_spread_rad=float(swivel_spread),
    )
