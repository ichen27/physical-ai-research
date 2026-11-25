"""Find the ball with the wrist camera and turn the arm to face it.

From the start pose the camera sees the desk ahead of the gripper. A ball
further round is found by swivelling the raised arm to look left and right.
Once found, "straight ahead" is turned so that the ball sits where the pickup
policy is most reliable (see ``understudy.runner.turn_to_face``), the arm
swivels to the start pose of that turned frame, and looks again from there,
where the ball is near the middle of the picture.

This moves the arm, slowly, with the gripper raised. On a real arm it is run
only with the person's go-ahead.
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

# Swivel angles looked from, in order, until the ball is seen. Radians, positive to the left.
SCAN = (0.0, 0.5, -0.5, 1.0, -1.0)


@dataclass(frozen=True)
class Facing:
    calibration: arm.Calibration  # of the turned frame; use it for the run
    turn_rad: float
    ball: tuple[float, float]  # in the turned frame
    distance_m: float  # of the ball from the swivel axis

    def within_reach(self) -> bool:
        return REACH_BAND_M[0] <= self.distance_m <= REACH_BAND_M[1]


def _start_ticks(calibration: arm.Calibration, swivel: float = 0.0) -> list[int]:
    pose = (swivel, *pickup.START_POSE[1:])
    return calibration.to_ticks(pose) + [calibration.finger_to_ticks(pickup.GRIPPER_OPEN_RAD)]


def look_for_ball(io: MovingArm, calibration: arm.Calibration, camera: WristCamera, look: Callable[[], Sequence | None],
                  chain: Chain | None = None, say: Callable[[str], None] = print) -> tuple[float, float] | None:
    """Where the ball is, in ``calibration``'s frame, or None if it is nowhere the camera can look.

    The arm must be holding the start pose. It is left swivelled to wherever the ball was seen from.
    ``look`` returns the patches of the ball's colour in a picture taken now.
    """
    chain = chain or Chain()
    swivel = 0.0
    for angle in SCAN:
        if angle != swivel:
            io.move_slowly(_start_ticks(calibration, angle), move_seconds([swivel], [angle]))
            swivel = angle
        blobs = look()
        point = pick_ball(camera, blobs or [], calibration.to_rad(io.read().positions[:5]), chain)
        if point is not None:
            return (point[0], point[1])
        say(f"  no ball seen looking {math.degrees(angle):+.0f} degrees round")
    return None


def face_ball(io: MovingArm, calibration: arm.Calibration, camera: WristCamera, look: Callable[[], Sequence | None],
              chain: Chain | None = None, say: Callable[[str], None] = print) -> Facing | None:
    """Find the ball, turn to face it and look again. None if the ball is not found.

    The arm must be holding the start pose of ``calibration``. It ends holding the start
    pose of the returned, turned calibration. Raises ValueError if the ball is too far
    round to face.
    """
    chain = chain or Chain()
    first = look_for_ball(io, calibration, camera, look, chain, say)
    if first is None:
        return None
    turn = turn_to_face(first)
    facing = turned(calibration, turn)
    here = calibration.to_rad(io.read().positions[:5])[0]
    io.move_slowly(_start_ticks(facing), move_seconds([here], [turn]))
    guess = in_turned_frame((*first, pickup.BALL_RADIUS_M), turn)
    again = pick_ball(camera, look() or [], facing.to_rad(io.read().positions[:5]), chain, guess)
    ball = guess if again is None else again
    if again is None:
        say("  the ball was not seen again after turning to face it; using where it was first seen")
    distance = math.hypot(ball[0] - SWIVEL_AXIS_X_M, ball[1])
    return Facing(facing, turn, (ball[0], ball[1]), distance)
