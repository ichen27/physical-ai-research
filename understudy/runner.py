"""Run the pickup policy on an arm, real or simulated.

The loop here talks to the arm in the servos' own units (encoder ticks and
velocity counts) through a small interface, so the same code drives the real
DYNAMIXEL servos and, in tests, a simulated arm standing in for them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from . import arm, pickup, runtime
from .kinematics import Chain

# DYNAMIXEL X-series Present Velocity unit: 0.229 revolutions per minute.
VELOCITY_UNIT_RAD_S = 0.229 * 2.0 * math.pi / 60.0
SERVO_IDS = (*(j.dxl_id for j in arm.ARM_JOINTS), arm.GRIPPER_DXL_ID)


@dataclass(frozen=True)
class ArmReading:
    """One reading of the six servos, arm joints first, gripper last."""

    positions: tuple[int, ...]  # ticks
    velocities: tuple[int, ...]  # Present Velocity counts


class ArmIO(Protocol):
    def read(self) -> ArmReading: ...

    def write_goals(self, ticks: Sequence[int]) -> None:
        """Send six goal positions, arm joints first, gripper last."""


class BallSource(Protocol):
    def position(self, joint_angles: Sequence[float], finger_angle: float, gripper_open: bool) -> tuple[float, float, float]:
        """Where the ball is now, metres in the arm base frame."""


class PlacedBall:
    """Stand-in for a camera: the ball is where it was placed until the gripper holds it.

    Once the gripper has been commanded closed for a moment and its fingers have
    stopped short of closing, something is between them, and the ball is taken to
    be cupped between them. This cannot see the ball roll away.
    """

    HELD_FINGER_RAD = 0.15
    SETTLE_STEPS = 10

    def __init__(self, x: float, y: float, chain: Chain | None = None):
        self._placed = (float(x), float(y), pickup.BALL_RADIUS_M)
        self._chain = chain or Chain()
        self._closed_steps = 0

    def position(self, joint_angles, finger_angle, gripper_open):
        self._closed_steps = 0 if gripper_open else self._closed_steps + 1
        if self._closed_steps >= self.SETTLE_STEPS and finger_angle > self.HELD_FINGER_RAD:
            return self._chain.point_in_tip_frame(list(joint_angles), pickup.GRASP_POINT_FROM_TIP)
        return self._placed


@dataclass(frozen=True)
class Step:
    joint_angles: tuple[float, ...]  # rad, five arm joints then the finger
    ball_position: tuple[float, float, float]
    action: tuple[float, ...]
    command: runtime.Command


def sensed_from_reading(reading: ArmReading, calibration: arm.Calibration) -> tuple[list[float], list[float]]:
    """Joint angles and velocities the policy senses, from raw servo readings."""
    if len(reading.positions) != 6 or len(reading.velocities) != 6:
        raise ValueError("a reading needs six positions and six velocities")
    angles = calibration.to_rad(reading.positions[:5]) + [calibration.finger_to_rad(reading.positions[5])]
    signs = (*calibration.signs, calibration.gripper_sign)
    velocities = [sign * count * VELOCITY_UNIT_RAD_S for sign, count in zip(signs, reading.velocities)]
    return angles, velocities


def goals_from_command(command: runtime.Command, calibration: arm.Calibration) -> list[int]:
    """Six servo goal positions for a policy command."""
    return calibration.to_ticks(command.arm_targets) + [calibration.finger_to_ticks(command.finger_target)]


def check_safe(angles: Sequence[float], margin: float = 0.25) -> None:
    """Stop if an arm joint is well outside the range the policy was trained in."""
    for angle, joint in zip(angles, arm.ARM_JOINTS):
        if not joint.lower - margin <= angle <= joint.upper + margin:
            raise RuntimeError(f"{joint.name} at {angle:.2f} rad is outside its trained range; stopping")


def run_pickup(
    io: ArmIO,
    policy: Callable[[Sequence[float]], Sequence[float]],
    calibration: arm.Calibration,
    ball: BallSource,
    hold_target: Sequence[float],
    seconds: float,
    advance: Callable[[], None],
    send: bool = True,
    interface: runtime.Interface | None = None,
) -> list[Step]:
    """Run the policy for ``seconds`` at the control rate.

    ``advance`` is called once per control step after the command is sent: on a
    real arm it waits for the next tick of the clock, in a simulation it steps
    the physics. With ``send`` false nothing is written to the arm. ``interface``
    defaults to the one saved with the policy, if the policy carries one.
    """
    if interface is None:
        interface = getattr(policy, "interface", runtime.Interface())
    for axis, value in zip("xyz", hold_target):
        low, high = pickup.HOLD_TARGET[axis]
        if not low <= value <= high:
            raise ValueError(f"hold target {axis}={value} is outside the trained range {low}..{high}")

    steps: list[Step] = []
    previous_action = [0.0] * runtime.ACTION_SIZE
    gripper_open = True
    # The arm starts each attempt holding the start pose, so that is the first "previous target".
    targets = list(pickup.START_POSE)
    incremental = interface.arm_action == "incremental"
    for _ in range(int(round(seconds * arm.CONTROL_HZ))):
        angles, velocities = sensed_from_reading(io.read(), calibration)
        check_safe(angles[:5])
        ball_position = tuple(ball.position(angles[:5], angles[5], gripper_open))
        lead = [target - angle for target, angle in zip(targets, angles[:5])] if incremental else None
        observation = runtime.build_observation(angles, velocities, ball_position, hold_target, previous_action, lead)
        action = list(policy(observation))
        command = runtime.decode_action(action, angles[:5], interface, targets)
        if send:
            io.write_goals(goals_from_command(command, calibration))
        steps.append(Step(tuple(angles), ball_position, tuple(action), command))
        previous_action, gripper_open, targets = action, command.gripper_open, list(command.arm_targets)
        advance()
    return steps
