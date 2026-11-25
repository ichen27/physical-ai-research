"""Run the pickup policy on an arm, real or simulated.

The loop here talks to the arm in the servos' own units (encoder ticks and
velocity counts) through a small interface, so the same code drives the real
DYNAMIXEL servos and, in tests, a simulated arm standing in for them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
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


class _Grip:
    """Tells when the gripper is holding something, from its fingers alone.

    Once the gripper has been commanded closed for a moment and its fingers have
    stopped short of closing, something is between them.
    """

    # Fingers closed on nothing read about 0.00 rad. On the simulated rigid ball they stop
    # above 0.15; on the real foam ball, which squashes, they stopped at 0.13 (2026-11-25).
    HELD_FINGER_RAD = 0.07
    SETTLE_STEPS = 10

    def __init__(self) -> None:
        self._closed_steps = 0

    def holding(self, finger_angle: float, gripper_open: bool) -> bool:
        self._closed_steps = 0 if gripper_open else self._closed_steps + 1
        return self._closed_steps >= self.SETTLE_STEPS and finger_angle > self.HELD_FINGER_RAD


class PlacedBall:
    """Stand-in for a camera: the ball is where it was placed until the gripper holds it.

    Once the gripper is holding something (see ``_Grip``), the ball is taken to
    be cupped between the fingers. This cannot see the ball roll away.
    """

    HELD_FINGER_RAD = _Grip.HELD_FINGER_RAD
    SETTLE_STEPS = _Grip.SETTLE_STEPS

    def __init__(self, x: float, y: float, chain: Chain | None = None):
        self._placed = (float(x), float(y), pickup.BALL_RADIUS_M)
        self._chain = chain or Chain()
        self._grip = _Grip()

    def position(self, joint_angles, finger_angle, gripper_open):
        if self._grip.holding(finger_angle, gripper_open):
            return self._chain.point_in_tip_frame(list(joint_angles), pickup.GRASP_POINT_FROM_TIP)
        return self._placed


# A ball the arm could pick up lies in this band around the swivel axis, metres. Orange things
# nearer or further are something else.
BALL_DISTANCE_M = (0.09, 0.36)
# A patch is the ball only if it is about as big in the picture as a ball there must be.
BALL_SIZE_RATIO = (0.72, 1.35)
SWIVEL_AXIS_X_M = -0.01125  # the vendor URDF's joint1 origin


def pick_ball(camera, blobs, joint_angles: Sequence[float], chain: Chain,
              near: Sequence[float] | None = None) -> tuple[float, float, float] | None:
    """Where the ball is on the desk, from the patches of its colour found in one picture.

    ``camera`` is an ``understudy.camera_model.WristCamera`` and ``blobs`` come from
    ``understudy.vision.find_blobs``. A patch counts only if it is whole (round and not
    cut off by the edge of the picture), lies on the desk within the arm's reach, and is
    the size a ball there would appear. Of several, the one nearest ``near`` is taken,
    or the largest. None if no patch qualifies.
    """
    found = []
    for blob in blobs:
        if not blob.whole():
            continue
        point = camera.locate(blob.u, blob.v, joint_angles, chain)
        if point is None:
            continue
        distance = math.hypot(point[0] - SWIVEL_AXIS_X_M, point[1])
        if not BALL_DISTANCE_M[0] <= distance <= BALL_DISTANCE_M[1]:
            continue
        expected = camera.sight(point, joint_angles, chain)
        if expected is None or not BALL_SIZE_RATIO[0] <= blob.radius_px / expected.radius_px <= BALL_SIZE_RATIO[1]:
            continue
        found.append((point, blob.area_px))
    if not found:
        return None
    if near is not None:
        return min(found, key=lambda item: math.dist(item[0][:2], near[:2]))[0]
    return max(found, key=lambda item: item[1])[0]


class CameraBall:
    """The ball's position from the camera on the wrist.

    ``look`` returns the patches of the ball's colour in the newest picture
    (``understudy.vision.find_blobs``), or None when there is no new picture yet.

    While the gripper is open, each picture in which the ball is seen whole and
    where a ball can be moves the reported position. When it is not seen (out
    of the picture, cut off by its edge, or partly hidden by the fingers during
    the last of the approach) the last position seen is kept: a ball on a desk
    stays put unless something pushes it. Once the gripper is commanded closed
    the camera is ignored, because a ball in the gripper is no longer on the
    desk and that is what turns a pixel into a position; from then on the
    fingers say whether it is held, as they do for ``PlacedBall``.

    A picture is a little old when it arrives, and the wrist has moved since.
    ``latency_steps`` control steps back is the pose the picture is matched to.
    """

    def __init__(self, camera, look: Callable[[], Sequence | None], first: Sequence[float] | None = None,
                 chain: Chain | None = None, latency_steps: int = 2):
        if latency_steps < 0:
            raise ValueError("latency_steps cannot be negative")
        self._camera, self._look = camera, look
        self._chain = chain or Chain()
        self._grip = _Grip()
        self._last = None if first is None else (float(first[0]), float(first[1]), pickup.BALL_RADIUS_M)
        self._poses: list[tuple[float, ...]] = []
        self._latency = latency_steps
        self.steps_seen = 0  # control steps on which a picture moved the position
        self.steps_kept = 0  # steps on which the last position was kept
        self.steps_held = 0  # steps on which the gripper held the ball
        self.state = "kept"  # what the latest step was: "seen", "kept" or "held"

    def position(self, joint_angles, finger_angle, gripper_open):
        self._poses = [tuple(joint_angles), *self._poses[: self._latency]]
        if self._grip.holding(finger_angle, gripper_open):
            self.steps_held += 1
            self.state = "held"
            return self._chain.point_in_tip_frame(list(joint_angles), pickup.GRASP_POINT_FROM_TIP)
        seen = None
        if gripper_open:
            blobs = self._look()
            if blobs:
                seen = pick_ball(self._camera, blobs, self._poses[-1], self._chain, self._last)
        if seen is not None:
            self._last = seen
            self.steps_seen += 1
            self.state = "seen"
        else:
            self.steps_kept += 1
            self.state = "kept"
        if self._last is None:
            raise RuntimeError("the camera has not seen the ball; stopping")
        return self._last


# Facing the ball. The policy was trained on balls in a patch 6 cm either side of straight
# ahead, and with the speed limits it lifts reliably only from the middle of that patch to its
# left. The arm is the same whichever way its swivel points, so a ball anywhere around it can
# be brought to the patch's best spot by treating a turned swivel as straight ahead.
AIM_LEFT_M = 0.03  # where the ball is put, left of straight ahead, in the turned frame
MAX_TURN_RAD = 1.0  # the furthest the swivel's "straight ahead" is moved
# Distances from the swivel axis the policy lifts from once the arm faces the ball. Measured in
# MuJoCo with the speed limits on, 40 attempts per centimetre: 92.5% or better from 15 to 24 cm,
# 52.5% at 14 to 15, 77.5% at 24 to 25, 40% at 25 to 26 and none beyond
# (results/camera/2026-11-25-wrist-camera, the mujoco-distance files). Not measured on a real arm.
REACH_BAND_M = (0.15, 0.24)


def turn_to_face(ball_xy: Sequence[float]) -> float:
    """How far to turn "straight ahead" about the swivel axis (rad, positive to the left) so
    that a ball at ``ball_xy`` ends up ``AIM_LEFT_M`` to the left of it."""
    x, y = ball_xy[0] - SWIVEL_AXIS_X_M, ball_xy[1]
    distance = math.hypot(x, y)
    if distance < 2.0 * AIM_LEFT_M:
        raise ValueError("the ball is on top of the swivel axis")
    turn = math.atan2(y, x) - math.asin(AIM_LEFT_M / distance)
    if abs(turn) > MAX_TURN_RAD:
        raise ValueError(
            f"facing the ball means turning {math.degrees(turn):+.0f} degrees; "
            f"the arm turns at most {math.degrees(MAX_TURN_RAD):.0f} either way"
        )
    return turn


def turned(calibration: arm.Calibration, turn: float) -> arm.Calibration:
    """The calibration of the same arm with "straight ahead" turned by ``turn`` rad about the swivel axis.

    Everything computed from it (joint angles, the gripper's position, the camera's
    answers) is then in the turned frame, and commands computed in that frame reach the
    servos as the right ticks. Nothing else has to know.
    """
    zeros = list(calibration.zero_ticks)
    zeros[0] += calibration.signs[0] * turn / arm.RAD_PER_TICK
    return replace(calibration, zero_ticks=tuple(zeros))


def in_turned_frame(point: Sequence[float], turn: float) -> tuple[float, ...]:
    """A base-frame point as the frame turned by ``turn`` about the swivel axis sees it."""
    x, y = point[0] - SWIVEL_AXIS_X_M, point[1]
    c, s = math.cos(turn), math.sin(turn)
    return (SWIVEL_AXIS_X_M + c * x + s * y, -s * x + c * y, *point[2:])


class SwivelLimit:
    """An arm whose swivel goals are kept inside the joint's true limits.

    In a turned frame the runner's own limit on the swivel is turned too. This
    holds the servo to the limits of the frame the arm was calibrated in.
    """

    def __init__(self, io: ArmIO, calibration: arm.Calibration):
        joint = arm.ARM_JOINTS[0]
        ends = [arm.rad_to_ticks(angle, calibration.zero_ticks[0], calibration.signs[0]) for angle in (joint.lower, joint.upper)]
        self._io, self._low, self._high = io, min(ends), max(ends)

    def read(self) -> ArmReading:
        return self._io.read()

    def write_goals(self, ticks: Sequence[int]) -> None:
        self._io.write_goals([min(max(int(ticks[0]), self._low), self._high), *ticks[1:]])


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
    max_speed: float | None = None,
    max_lead: float | None = None,
    on_step: Callable[[Step], None] | None = None,
) -> list[Step]:
    """Run the policy for ``seconds`` at the control rate.

    ``advance`` is called once per control step after the command is sent: on a
    real arm it waits for the next tick of the clock, in a simulation it steps
    the physics. With ``send`` false nothing is written to the arm. ``interface``
    defaults to the one saved with the policy, if the policy carries one.

    Two limits make the arm gentler than the policy was trained to be, for a
    real arm. They take it outside its training: in MuJoCo the 2026-11-24
    policy lifts the ball in 99.5% of attempts without them and about 60% with. ``max_speed`` (rad/s) caps how fast each arm joint's target may
    move, and so how fast the joint is asked to go. ``max_lead`` (rad) caps how
    far a target may sit from its joint's present angle, and so how hard a
    servo pushes when something is in the way; it can only tighten the
    interface's own value. The policy sees the limited targets as its own.

    ``on_step`` is called with each step as it happens, so that a run which is
    cut short still leaves a record.
    """
    if interface is None:
        interface = getattr(policy, "interface", runtime.Interface())
    if max_speed is not None and max_speed <= 0:
        raise ValueError("max_speed must be positive")
    if max_lead is not None:
        if interface.arm_action != "incremental":
            raise ValueError("max_lead applies only to a policy with incremental targets")
        if not 0 < max_lead <= interface.max_lead:
            raise ValueError(f"max_lead must be above 0 and at most the policy's own {interface.max_lead}")
        interface = replace(interface, max_lead=max_lead)
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
        if max_speed is not None:
            step = max_speed / arm.CONTROL_HZ
            slowed = tuple(min(max(new, old - step), old + step) for new, old in zip(command.arm_targets, targets))
            command = replace(command, arm_targets=slowed)
        if send:
            io.write_goals(goals_from_command(command, calibration))
        steps.append(Step(tuple(angles), ball_position, tuple(action), command))
        if on_step is not None:
            on_step(steps[-1])
        previous_action, gripper_open, targets = action, command.gripper_open, list(command.arm_targets)
        advance()
    return steps
