"""Find the ball with the wrist camera and turn the arm to face it.

The ball is looked for from a raised pose, ``LOOK_POSE``, not from the start
pose the pickup begins in. On the first real arm the camera's lens is narrow
(67 degrees across) and from the start pose it looks almost straight down:
that picture shows, whole, fewer than half the places a ball can be picked up
from. The look pose lifts the camera to about 28 cm and shows all of them
within about 35 degrees either side. A ball further round is found by
swivelling the arm, still in the look pose, to look left and right (``SCAN``).

Once found, "straight ahead" is turned so that the ball sits where the pickup
policy is most reliable (see ``understudy.runner.turn_to_face``). The arm
swivels to face it, looks again from the look pose, where the ball is now near
the middle of the picture and best placed, and then comes down to the start
pose of that turned frame.

This moves the arm, slowly, with the gripper raised: no joint is asked to turn
faster than ``understudy.camera_collect.MAX_JOINT_SPEED``. Everything within
the reach of the raised arm, over the whole sweep, has to be clear; the README
gives the figures. On a real arm it is run only with the person's go-ahead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from . import arm, pickup
from .camera_collect import MovingArm, move_seconds
from .camera_model import WristCamera
from .kinematics import Chain
from .runner import REACH_BAND_M, SWIVEL_AXIS_X_M, in_turned_frame, pick_ball, turn_to_face, turned

# The pose the ball is looked for from: the start pose with the shoulder back 0.3 rad, the elbow
# in 0.4 and the wrist down 0.4. The gripper tip is 21 cm above the desk and 12 cm out; with the
# first real arm's camera the lens is 28 cm up and every place a ball can be picked up from is in
# the picture, whole, for bearings from about 35 degrees right to 40 left.
LOOK_POSE = (0.0, -0.9, 0.1, 1.6, 0.0)
# Swivel angles looked from, in order, until the ball is seen. Radians, positive to the left.
# Each look overlaps the next, and together they cover 95 degrees either side with that camera.
SCAN = (0.0, 0.5, -0.5, 1.0, -1.0)


@dataclass(frozen=True)
class Facing:
    calibration: arm.Calibration  # of the turned frame; use it for the run
    turn_rad: float
    ball: tuple[float, float]  # in the turned frame
    distance_m: float  # of the ball from the swivel axis

    def within_reach(self) -> bool:
        return REACH_BAND_M[0] <= self.distance_m <= REACH_BAND_M[1]


def _ticks(calibration: arm.Calibration, pose: Sequence[float], swivel: float) -> list[int]:
    """Servo goals for ``pose`` with the swivel at ``swivel`` and the gripper open."""
    return calibration.to_ticks((swivel, *pose[1:])) + [calibration.finger_to_ticks(pickup.GRIPPER_OPEN_RAD)]


def look_for_ball(io: MovingArm, calibration: arm.Calibration, camera: WristCamera, look: Callable[[], Sequence | None],
                  chain: Chain | None = None, say: Callable[[str], None] = print) -> tuple[float, float] | None:
    """Where the ball is, in ``calibration``'s frame, or None if it is nowhere the camera can look.

    The arm must be holding the start pose. It rises to the look pose and is left there,
    swivelled to wherever the ball was seen from (or to the last place looked from).
    ``look`` returns the patches of the ball's colour in a picture taken now.
    """
    chain = chain or Chain()
    here = tuple(pickup.START_POSE)
    for angle in SCAN:
        there = (angle, *LOOK_POSE[1:])
        io.move_slowly(_ticks(calibration, LOOK_POSE, angle), move_seconds(here, there))
        here = there
        blobs = look()
        point = pick_ball(camera, blobs or [], calibration.to_rad(io.read().positions[:5]), chain)
        if point is not None:
            return (point[0], point[1])
        say(f"  no ball seen looking {math.degrees(angle):+.0f} degrees round")
    return None


def face_ball(io: MovingArm, calibration: arm.Calibration, camera: WristCamera, look: Callable[[], Sequence | None],
              chain: Chain | None = None, say: Callable[[str], None] = print) -> Facing | None:
    """Find the ball, turn to face it, look again, and come down to the start pose. None if no ball is found.

    The arm must be holding the start pose of ``calibration``. If the ball is found it
    ends holding the start pose of the returned, turned calibration; if not, it is left
    in the look pose. Raises ValueError, with the arm in the look pose, if the ball is
    too far round to face.
    """
    chain = chain or Chain()
    first = look_for_ball(io, calibration, camera, look, chain, say)
    if first is None:
        return None
    turn = turn_to_face(first)
    seen_from = calibration.to_rad(io.read().positions[:5])[0]
    # Face it, still raised, and look again: the ball is now near the middle of the picture.
    io.move_slowly(_ticks(calibration, LOOK_POSE, turn), move_seconds([seen_from], [turn]))
    guess = in_turned_frame((*first, pickup.BALL_RADIUS_M), turn)
    again = pick_ball(camera, look() or [], turned(calibration, turn).to_rad(io.read().positions[:5]), chain, guess)
    raised_at = turn
    if again is None:
        say("  the ball was not seen again after turning to face it; using where it was first seen")
    else:
        # The second look is the better one. Turn "straight ahead" to suit it; the arm follows on its way down.
        turn = turn_to_face(in_turned_frame(again, -turn)[:2])
        guess = in_turned_frame(in_turned_frame(again, -raised_at), turn)
    facing = turned(calibration, turn)
    io.move_slowly(_ticks(facing, pickup.START_POSE, 0.0), move_seconds((raised_at, *LOOK_POSE[1:]), (turn, *pickup.START_POSE[1:])))
    distance = math.hypot(guess[0] - SWIVEL_AXIS_X_M, guess[1])
    return Facing(facing, turn, (guess[0], guess[1]), distance)
