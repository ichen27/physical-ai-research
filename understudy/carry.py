"""Carry a ball the gripper is holding to a place at the side and put it down, in a container or on the desk.

Not learnt. Once the pickup has the ball there is nothing left to find out:
the ball is in the gripper and the place is given, so the rest is a fixed,
slow move that can be read and checked. It is the same whoever picked the
ball up.

The move: raise the ball to a compact carrying pose (the look pose of
``understudy.seek``, gripper closed); swivel to the place; move out over it
with the ball well above the container's rim; lower straight down until the
ball is a little above the rim; open the gripper gradually; rise straight up
again, so the open fingers leave the way they came; and return. The fingers
never go below the rim: the ball is dropped the last few centimetres. On the
bare desk the same is done with a rim of no height.

Goals are sent every control step along eased paths, never faster on any
joint than ``understudy.camera_collect.MAX_JOINT_SPEED``, with the gripper
commanded closed, exactly as the policy holds it, until the release.

Checks: the carry does not start unless the fingers say something is held and
it has been lifted clear of the desk; it stops, with the arm left raised where
it is, the moment the fingers close fully, which means the ball was lost; and a
place the arm cannot reach, a container too close to the arm, or one further
round than the swivel goes is refused before anything moves. ``in_the_way``
says whether the container sits where the gripper must go to pick a ball up.

This moves the arm. On a real arm it is run only with the person's go-ahead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from . import arm, pickup
from .camera_collect import MAX_JOINT_SPEED, MIN_MOVE_SECONDS
from .kinematics import Chain
from .runner import AIM_LEFT_M, SWIVEL_AXIS_X_M, ArmIO, _Grip, sensed_from_reading
from .seek import LOOK_POSE

# The pose the ball is carried in: the look pose, which keeps the whole arm within 14 cm of the swivel axis.
CARRY_POSE = tuple(LOOK_POSE[1:4])  # shoulder, elbow and wrist; the swivel is wherever it needs to be, the roll level
# How far above the rim (or the desk, where there is no container) the bottom of the ball is when the gripper opens.
RELEASE_ABOVE_M = 0.01
# How far above that the ball is while it moves in over the place and before the arm swivels away again.
HOVER_ABOVE_M = 0.05
# The carry starts only if the ball has been lifted at least this far above where it rests.
MIN_LIFT_M = 0.03
# A container's near edge must be at least this far from the swivel axis, to be clear of the raised arm as it turns.
NEAREST_EDGE_M = 0.10
# What the pickup sweeps with the gripper still below 11 cm, measured over thousands of simulated pickups at the
# real arm's speed limits (``scripts/mujoco_carry.py`` reports it on every run): the arm turns up to 11 degrees to
# the left of where it faces to take the ball, and then, lifting, swings up to 42 degrees to the right. The gripper
# reaches a little beyond the ball, and its fingers and the ball are within 7 cm of the arm's line.
PICKUP_SWING_LEFT_RAD = math.radians(12.0)
PICKUP_SWING_RIGHT_RAD = math.radians(45.0)
# On the real arm more is allowed for. Its first pickup recorded with these limits swung 41 degrees to the right,
# where the same ball in simulation gave 34: one run, so the simulated worst case is not yet a safe bound for it.
REAL_ARM_SWING_RIGHT_RAD = math.radians(55.0)
PICKUP_HALF_WIDTH_M = 0.07
PICKUP_BEYOND_M = 0.08
# A container's place is read off the arm by hand and may be a little out. This much more room is asked for. With
# none, no pickup touched a bowl that stood within 2 cm of where it was said to be, and 2 in 100 touched one 3 cm out.
PLACE_SLACK_M = 0.02
# Gripper pitches tried for the set-down, radians below level: the start pose's first.
_PITCHES = (1.1, 1.3, 0.9, 1.5, 0.7)
RELEASE_SECONDS = 2.0  # the gripper opens over the first 1.4 s of this and the ball is left to settle for the rest


@dataclass(frozen=True)
class SetDown:
    """Where to put the ball, in the arm's own frame, not a turned one.

    ``bearing_rad`` is how far round from straight ahead the place is (positive to the
    arm's left) and ``distance_m`` how far from the swivel axis. For a container,
    ``rim_m`` is the height of its rim above the desk and ``opening_m`` the width of
    its opening; left at zero, the ball is put on the bare desk.
    """

    bearing_rad: float
    distance_m: float
    rim_m: float = 0.0
    opening_m: float = 0.0

    @property
    def xy(self) -> tuple[float, float]:
        return (SWIVEL_AXIS_X_M + self.distance_m * math.cos(self.bearing_rad), self.distance_m * math.sin(self.bearing_rad))

    @classmethod
    def at(cls, xy: Sequence[float], rim_m: float = 0.0, opening_m: float = 0.0) -> "SetDown":
        """The place whose middle is at ``xy`` in the arm's own frame."""
        x, y = xy[0] - SWIVEL_AXIS_X_M, xy[1]
        return cls(math.atan2(y, x), math.hypot(x, y), rim_m, opening_m)

    @classmethod
    def from_touches(cls, middle: Sequence[float], rim: Sequence[float]) -> "SetDown":
        """A round container, from two places the gripper's tip was held: its middle and a point on its rim.

        Each is x, y and height in the arm's own frame. Only the rim point's height is used:
        the tip does not have to reach the container's floor, only be over its middle.
        """
        return cls.at(middle, rim_m=rim[2], opening_m=2.0 * math.dist(middle[:2], rim[:2]))


@dataclass(frozen=True)
class Carried:
    done: bool  # the ball was released at the place
    said: str  # what happened, for the person at the arm
    released_at: tuple[float, float] | None = None  # where the gripper let go, x and y in the arm's frame


def in_the_way(place: SetDown, ball_xy: Sequence[float], facing_rad: float | None = None,
               swing_right_rad: float = PICKUP_SWING_RIGHT_RAD) -> str | None:
    """Why a ball at ``ball_xy`` should not be picked up with the container where it is, or None if it may be.

    A container is a real thing the gripper could strike. Picking a ball up, the arm
    faces it, comes down on it and then swings to its right as it lifts: the container
    must be clear of all of that. And a ball lying in the container is not on the desk.
    ``ball_xy`` is in the arm's own frame, not a turned one. ``facing_rad`` is how far
    round the arm faces for the pickup; left out, it is taken to turn to face the ball
    (``understudy.runner.turn_to_face``). ``swing_right_rad`` is how far to the right of that the lift is
    allowed to swing.
    """
    if place.opening_m <= 0:
        return None
    centre, half = place.xy, place.opening_m / 2.0
    if math.dist(ball_xy[:2], centre) < half:
        return "the ball is in the container already"
    reach = math.hypot(ball_xy[0] - SWIVEL_AXIS_X_M, ball_xy[1])
    facing = facing_rad
    if facing is None:
        facing = math.atan2(ball_xy[1], ball_xy[0] - SWIVEL_AXIS_X_M) - math.asin(min(1.0, AIM_LEFT_M / reach))
    out, round_ = math.hypot(centre[0] - SWIVEL_AXIS_X_M, centre[1]), math.atan2(centre[1], centre[0] - SWIVEL_AXIS_X_M)
    # How far the container's middle is from the nearest line the arm lies along during the pickup.
    aside = math.inf
    steps = 60
    for step in range(steps + 1):
        line = facing - swing_right_rad + (swing_right_rad + PICKUP_SWING_LEFT_RAD) * step / steps
        along = min(max(out * math.cos(round_ - line), 0.0), reach + PICKUP_BEYOND_M)
        aside = min(aside, math.sqrt(max(out * out - 2.0 * out * along * math.cos(round_ - line) + along * along, 0.0)))
    needed = PICKUP_HALF_WIDTH_M + PLACE_SLACK_M
    if aside < half + needed:
        return (f"the container is in the gripper's way: its edge is {max(0.0, aside - half) * 100:.0f} cm from where the arm "
                f"goes to pick this ball up, and {needed * 100:.0f} cm is needed")
    return None


def grasp_point(angles: Sequence[float], chain: Chain) -> tuple[float, float, float]:
    """Where a held ball's centre is, from the five arm joint angles."""
    return chain.point_in_tip_frame(list(angles), pickup.GRASP_POINT_FROM_TIP)


def pose_over(place: SetDown, ball_bottom_m: float, chain: Chain | None = None) -> tuple[float, ...]:
    """The five joint angles that hold the ball over ``place`` with its bottom ``ball_bottom_m`` above the desk.

    Raises ValueError, saying why, if the place is too far round for the swivel or the arm
    cannot hold the ball there with its fingers clear of the desk.
    """
    chain = chain or Chain()
    if not 0.05 < place.distance_m < 0.5:
        raise ValueError(f"a place {place.distance_m * 100:.0f} cm from the swivel axis is not one the arm can reach")
    # The ball sits a little to one side of the gripper's line, so the swivel is not quite the bearing.
    side = grasp_point((0.0, *CARRY_POSE, 0.0), chain)[1]
    swivel = place.bearing_rad - math.asin(max(-1.0, min(1.0, side / place.distance_m)))
    joint = arm.ARM_JOINTS[0]
    if joint.upper < abs(swivel) <= joint.upper + math.radians(1.5):
        swivel = math.copysign(joint.upper, swivel)  # a place just at the end of the swivel's travel: go as far as it does
    if not joint.lower <= swivel <= joint.upper:
        raise ValueError(
            f"a place {math.degrees(place.bearing_rad):+.0f} degrees round is further than the swivel goes "
            f"({math.degrees(joint.upper):.0f} degrees either way)"
        )
    want_reach = math.sqrt(max(place.distance_m**2 - side**2, 0.0))
    want_height = pickup.BALL_RADIUS_M + ball_bottom_m
    for pitch in _PITCHES:
        shoulder, elbow = pickup.START_POSE[1], pickup.START_POSE[2]

        def miss(s: float, e: float) -> tuple[float, float]:
            x, _, z = grasp_point((0.0, s, e, pitch - s - e, 0.0), chain)
            return x - SWIVEL_AXIS_X_M - want_reach, z - want_height

        for _ in range(60):  # Newton's method on the two joints that set reach and height
            fx, fz = miss(shoulder, elbow)
            if abs(fx) < 1e-6 and abs(fz) < 1e-6:
                break
            step = 1e-5
            ax, az = miss(shoulder + step, elbow)
            bx, bz = miss(shoulder, elbow + step)
            a, b, c, d = (ax - fx) / step, (bx - fx) / step, (az - fz) / step, (bz - fz) / step
            determinant = a * d - b * c
            if abs(determinant) < 1e-9:
                break
            shoulder -= max(-0.2, min(0.2, (d * fx - b * fz) / determinant))
            elbow -= max(-0.2, min(0.2, (a * fz - c * fx) / determinant))
        pose = (swivel, shoulder, elbow, pitch - shoulder - elbow, 0.0)
        fx, fz = miss(shoulder, elbow)
        inside = all(j.lower <= angle <= j.upper for angle, j in zip(pose, arm.ARM_JOINTS))
        # The finger tips reach below the ball's centre; they must stay off the desk.
        if abs(fx) < 1e-4 and abs(fz) < 1e-4 and inside and chain.tip_position(list(pose))[2] > 0.004:
            return pose
    raise ValueError(
        f"the arm cannot hold the ball {ball_bottom_m * 100:.0f} cm above the desk {place.distance_m * 100:.0f} cm from the swivel axis"
    )


def set_down_poses(place: SetDown, chain: Chain | None = None,
                   release_above_m: float = RELEASE_ABOVE_M) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The pose the ball hovers in above the place, and the pose it is let go from, straight below.

    Raises ValueError, saying why, if the place cannot be used: out of reach, too far
    round, or a container too close to the arm to be clear of it as it turns.
    """
    if place.rim_m < 0 or place.opening_m < 0 or (place.rim_m > 0) != (place.opening_m > 0):
        raise ValueError("a container needs both the height of its rim and the width of its opening")
    if place.opening_m > 0:
        if place.opening_m < 2 * pickup.BALL_RADIUS_M + 0.02:
            raise ValueError(f"an opening {place.opening_m * 100:.0f} cm wide is too narrow for the ball")
        near = place.distance_m - place.opening_m / 2.0
        if near < NEAREST_EDGE_M - 1e-9:
            raise ValueError(
                f"the container's near edge would be {near * 100:.0f} cm from the swivel axis; at least "
                f"{NEAREST_EDGE_M * 100:.0f} cm is needed for the raised arm to turn clear of it"
            )
    release = place.rim_m + release_above_m
    # Asked to let go below the rim, the ball still moves in, and the fingers leave, well above it.
    hover = place.rim_m + max(release_above_m, RELEASE_ABOVE_M) + HOVER_ABOVE_M
    return pose_over(place, hover, chain), pose_over(place, release, chain)


def carry(io: ArmIO, calibration: arm.Calibration, place: SetDown, advance: Callable[[], None], chain: Chain | None = None,
          max_speed: float = MAX_JOINT_SPEED, release_above_m: float = RELEASE_ABOVE_M,
          say: Callable[[str], None] = print) -> Carried:
    """Carry the held ball to ``place``, let it go there, and return to the carrying pose facing straight ahead.

    ``calibration`` is the arm's own, not a turned one: ``place`` and the swivel's limits
    are in the arm's own frame. ``advance`` waits for the next control step, as for
    ``understudy.runner.run_pickup``. ``max_speed`` is the fastest any joint is asked to turn.
    """
    if max_speed <= 0:
        raise ValueError("max_speed must be positive")
    chain = chain or Chain()
    over, down = set_down_poses(place, chain, release_above_m)  # refuses a place that cannot be used before anything moves
    closed, opened = pickup.GRIPPER_CLOSED_RAD, pickup.GRIPPER_OPEN_RAD

    def sensed() -> list[float]:
        return sensed_from_reading(io.read(), calibration)[0]

    def send(pose: Sequence[float], finger: float) -> None:
        io.write_goals(calibration.to_ticks(pose) + [calibration.finger_to_ticks(finger)])

    def glide(start: Sequence[float], end: Sequence[float], finger: float, holding: bool) -> bool:
        """Ease from one pose to another. False, stopping where it is, if the ball is lost on the way."""
        furthest = max(abs(b - a) for a, b in zip(start, end))
        # An eased move's fastest moment is pi/2 times its average speed.
        steps = max(1, math.ceil(max(MIN_MOVE_SECONDS, math.pi / 2 * furthest / max_speed) * arm.CONTROL_HZ))
        for step in range(1, steps + 1):
            part = 0.5 - 0.5 * math.cos(math.pi * step / steps)
            send([a + (b - a) * part for a, b in zip(start, end)], finger)
            advance()
            if holding and sensed()[5] <= _Grip.HELD_FINGER_RAD:
                return False
        return True

    now = sensed()
    if now[5] <= _Grip.HELD_FINGER_RAD:
        return Carried(False, "not carried: the fingers are closed on nothing")
    if grasp_point(now[:5], chain)[2] < pickup.BALL_RADIUS_M + MIN_LIFT_M:
        return Carried(False, "not carried: the ball has not been lifted clear of the desk")
    here = tuple(now[:5])
    raised = (here[0], *CARRY_POSE, 0.0)
    turned = (down[0], *CARRY_POSE, 0.0)
    legs = ((here, raised, "raising it"), (raised, turned, "swivelling to the place"),
            (turned, over, "moving out over the place"), (over, down, "lowering it"))
    for start, end, what in legs:
        if not glide(start, end, closed, holding=True):
            return Carried(False, f"stopped while {what}: the fingers closed fully, so the ball was lost. The arm is left raised where it was.")
    # Open gradually. Fingers that spring open nudge the ball as they leave it.
    steps = int(RELEASE_SECONDS * arm.CONTROL_HZ)
    for step in range(1, steps + 1):
        send(down, closed + (opened - closed) * min(1.0, step / (0.7 * steps)))
        advance()
    released = grasp_point(sensed()[:5], chain)
    glide(down, over, opened, holding=False)  # straight up, the way it came, so the open fingers clear the rim
    glide(over, turned, opened, holding=False)
    glide(turned, (0.0, *CARRY_POSE, 0.0), opened, holding=False)
    say(f"ball let go {math.hypot(released[0] - SWIVEL_AXIS_X_M, released[1]) * 100:.1f} cm from the swivel axis, "
        f"{math.degrees(math.atan2(released[1], released[0] - SWIVEL_AXIS_X_M)):+.0f} degrees round")
    return Carried(True, "let go over the container" if place.opening_m > 0 else "put down", (released[0], released[1]))
