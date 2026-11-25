"""Gather the pictures a camera calibration needs, with the arm doing the work.

The person does one thing: put the ball on the desk in front of the gripper.
The arm, holding its start pose, notices the ball appear (a picture from before
compared with pictures after), learns its colour from it, and then looks at it
from a round of small moves, works out a rough camera from those, and uses that
to choose wider views that keep the ball in the picture. Everything is checked
as it goes: a ball that never appears, a pose where it is not seen, a ball that
was nudged part way through.

The same code drives the real arm and the simulated one. It moves the arm, so
on a real arm it is run only with the person's go-ahead. Every pose keeps the
gripper tip well above the desk, and moves are slow (see ``MAX_JOINT_SPEED``).
Needs ``numpy`` and ``opencv-python-headless``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from . import arm, camera_calibration, pickup, vision
from .camera_calibration import Sighting
from .camera_model import WristCamera
from .kinematics import Chain
from .runner import ArmReading, pick_ball

# The fastest any joint is asked to turn while collecting, rad/s (about 14 degrees a second).
MAX_JOINT_SPEED = 0.25
MIN_MOVE_SECONDS = 1.5
# The gripper tip stays at least this far above the desk in every pose, and this far along the way between poses.
MIN_TIP_HEIGHT_M = 0.085
MIN_PATH_HEIGHT_M = 0.07
# Around the start pose, each of these changes the view a little. Radians added to the five arm joints.
SMALL_MOVES = (
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.12, 0.0, 0.0, 0.0, 0.0),
    (-0.12, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.12, 0.0),
    (0.0, 0.0, 0.0, -0.12, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.3),
    (0.0, 0.0, 0.0, 0.0, -0.3),
    (0.0, -0.12, 0.0, 0.0, 0.0),
    (0.0, 0.0, -0.12, 0.0, 0.0),
    (0.1, -0.06, 0.0, -0.08, 0.2),
    (-0.1, -0.06, 0.0, 0.08, -0.2),
)
# How far the wider views may stray from the start pose, radians per joint.
WIDE_RANGE = (0.4, 0.3, 0.3, 0.35, 0.5)
# The ball must be predicted this far inside the picture's edge (as a fraction of its size) for a view to be chosen.
VIEW_MARGIN = 0.12
# A ball that the fitted camera puts further than this from where it rested was probably moved.
MOVED_M = 0.012


class MovingArm(Protocol):
    def read(self) -> ArmReading: ...

    def move_slowly(self, ticks: Sequence[int], seconds: float) -> None: ...


@dataclass(frozen=True)
class Arrival:
    """The ball as first seen after it was put down."""

    blob: vision.Blob
    colour: vision.BallColour
    picture: np.ndarray


class NoBall(RuntimeError):
    """The ball did not appear, or could not be followed."""


def pose_is_clear(pose: Sequence[float], chain: Chain, height: float = MIN_TIP_HEIGHT_M) -> bool:
    inside = all(joint.lower <= angle <= joint.upper for angle, joint in zip(pose, arm.ARM_JOINTS))
    return inside and chain.tip_position(list(pose))[2] >= height


def path_is_clear(start: Sequence[float], end: Sequence[float], chain: Chain) -> bool:
    """Whether the gripper tip stays above the desk all the way between two poses."""
    for part in np.linspace(0.0, 1.0, 9):
        pose = [a + (b - a) * part for a, b in zip(start, end)]
        if not pose_is_clear(pose, chain, MIN_PATH_HEIGHT_M):
            return False
    return True


def move_seconds(start: Sequence[float], end: Sequence[float]) -> float:
    """How long a move between two poses takes at the collection speed."""
    furthest = max(abs(b - a) for a, b in zip(start, end))
    # A servo's profile peaks at about one and a half times its average speed.
    return max(MIN_MOVE_SECONDS, 1.5 * furthest / MAX_JOINT_SPEED)


def small_poses(chain: Chain) -> list[tuple[float, ...]]:
    poses = [tuple(s + d for s, d in zip(pickup.START_POSE, delta)) for delta in SMALL_MOVES]
    return [pose for pose in poses if pose_is_clear(pose, chain)]


def wide_poses(camera: WristCamera, ball_xy: Sequence[float], count: int, chain: Chain,
               rng: random.Random | None = None) -> list[tuple[float, ...]]:
    """Poses spread as widely as possible from which ``camera`` should see a ball at ``ball_xy`` whole."""
    rng = rng or random.Random(0)
    ball = (ball_xy[0], ball_xy[1], pickup.BALL_RADIUS_M)
    found: list[tuple[float, ...]] = []
    for _ in range(6000):
        pose = tuple(s + rng.uniform(-reach, reach) for s, reach in zip(pickup.START_POSE, WIDE_RANGE))
        if not pose_is_clear(pose, chain):
            continue
        sight = camera.sight(ball, pose, chain)
        if sight is None:
            continue
        margin_u = VIEW_MARGIN * camera.width + sight.radius_px
        margin_v = VIEW_MARGIN * camera.height + sight.radius_px
        if not (margin_u <= sight.u <= camera.width - margin_u and margin_v <= sight.v <= camera.height - margin_v):
            continue
        found.append(pose)
        if len(found) >= 400:
            break
    if not found:
        return []
    # Keep the ones furthest from each other, starting with the one furthest from the start pose.
    scaled = np.array(found) / np.array(WIDE_RANGE)
    home = np.array(pickup.START_POSE) / np.array(WIDE_RANGE)
    chosen = [int(np.argmax(np.linalg.norm(scaled - home, axis=1)))]
    nearest = np.linalg.norm(scaled - scaled[chosen[0]], axis=1)
    while len(chosen) < min(count, len(found)):
        chosen.append(int(np.argmax(nearest)))
        nearest = np.minimum(nearest, np.linalg.norm(scaled - scaled[chosen[-1]], axis=1))
    return [found[index] for index in chosen]


def in_visiting_order(poses: Sequence[Sequence[float]], start: Sequence[float]) -> list[tuple[float, ...]]:
    """The poses reordered so each is the nearest not yet visited: less travel, smaller moves."""
    left = [tuple(pose) for pose in poses]
    here, ordered = tuple(start), []
    while left:
        nearest = min(left, key=lambda pose: max(abs(a - b) for a, b in zip(pose, here)))
        left.remove(nearest)
        ordered.append(nearest)
        here = nearest
    return ordered


def wait_for_ball(take_picture: Callable[[], np.ndarray], empty: np.ndarray, attempts: int = 40,
                  say: Callable[[str], None] = print) -> Arrival:
    """Take pictures until a ball that was not in ``empty`` is seen in the same spot twice running.

    Twice running, so that a hand still holding the ball, or a ball still rolling, does not count.
    """
    previous = None
    for _ in range(attempts):
        picture = take_picture()
        blob = vision.find_new_ball(empty, picture)
        if blob is not None and previous is not None and math.hypot(blob.u - previous.u, blob.v - previous.v) < 6.0:
            try:
                colour = vision.learn_colour(picture, blob)
            except ValueError as error:
                say(f"something new is in the picture but it is not the ball: {error}")
                previous = None
                continue
            # The colour learnt must find the same ball again, or it is no use for the pictures that follow.
            again = next((b for b in vision.find_blobs(picture, colour) if b.whole()), None)
            if again is not None and math.hypot(again.u - blob.u, again.v - blob.v) < 0.25 * blob.radius_px:
                return Arrival(again, colour, picture)
            say("the ball's colour could not be told from its surroundings in this picture; trying again")
        previous = blob
    raise NoBall("no ball appeared in front of the gripper. Put it on the desk where the camera can see all of it.")


class Collector:
    """Visits poses and records where the ball appears from each."""

    def __init__(self, io: MovingArm, calibration: arm.Calibration, take_picture: Callable[[], np.ndarray],
                 colour: vision.BallColour, chain: Chain | None = None, say: Callable[[str], None] = print):
        self.io, self.calibration, self.take_picture, self.colour = io, calibration, take_picture, colour
        self.chain = chain or Chain()
        self.say = say
        self._pose = tuple(calibration.to_rad(io.read().positions[:5]))

    def go(self, pose: Sequence[float]) -> bool:
        """Move to a pose, gripper open. False, without moving, if the way there dips towards the desk."""
        if not pose_is_clear(pose, self.chain) or not path_is_clear(self._pose, pose, self.chain):
            return False
        ticks = self.calibration.to_ticks(pose) + [self.calibration.finger_to_ticks(pickup.GRIPPER_OPEN_RAD)]
        self.io.move_slowly(ticks, move_seconds(self._pose, pose))
        self._pose = tuple(pose)
        return True

    def see(self, place: int, camera: WristCamera | None = None, near: Sequence[float] | None = None) -> Sighting | None:
        """One picture from the present pose. None if the ball is not seen whole, or the arm was not still."""
        before = self.io.read().positions[:5]
        picture = self.take_picture()
        after = self.io.read().positions[:5]
        if max(abs(a - b) for a, b in zip(before, after)) > 3:
            return None
        blobs = [blob for blob in vision.find_blobs(picture, self.colour) if blob.whole()]
        if not blobs:
            return None
        chosen = blobs[0]
        if camera is not None:
            angles = self.calibration.to_rad(after)
            point = pick_ball(camera, blobs, angles, self.chain, near)
            if point is None:
                return None
            chosen = min(blobs, key=lambda blob: math.dist(camera.locate(blob.u, blob.v, angles, self.chain) or (9, 9, 9), point))
        return Sighting(tuple(int(t) for t in after), chosen.u, chosen.v, chosen.radius_px, place)

    def visit(self, poses: Sequence[Sequence[float]], place: int, camera: WristCamera | None = None,
              near: Sequence[float] | None = None) -> list[Sighting]:
        seen = []
        for pose in in_visiting_order(poses, self._pose):
            if not self.go(pose):
                continue
            sighting = self.see(place, camera, near)
            self.say(f"  pose {[round(a, 2) for a in pose]}: " + ("ball seen" if sighting else "ball not seen whole"))
            if sighting is not None:
                seen.append(sighting)
        return seen

    def collect_place(self, place: int, width: int, height: int, wide_count: int = 12,
                      known: WristCamera | None = None, rng: random.Random | None = None) -> list[Sighting]:
        """All the sightings of the ball where it now rests. Ends back at the start pose.

        ``known`` is a camera from an earlier fit, if there is one; without it a rough one
        is worked out from the round of small moves.
        """
        seen = self.visit(small_poses(self.chain), place)
        rough, where = known, None
        if rough is None:
            if len(seen) < camera_calibration.MIN_SIGHTINGS:
                self.go(pickup.START_POSE)
                raise NoBall(
                    f"the ball was seen whole from only {len(seen)} of the small moves. Put it nearer the middle "
                    "of the picture: on the desk straight ahead of the gripper."
                )
            rough = camera_calibration.fit(seen, self.calibration, width, height, self.chain, lens=False).camera
        try:
            where, _, _ = camera_calibration.spread(rough, seen, self.calibration, self.chain)
        except ValueError:
            where = None
        if where is not None:
            seen += self.visit(wide_poses(rough, where, wide_count, self.chain, rng), place, rough, (*where, 0.0))
        self.go(pickup.START_POSE)
        return seen


def moved_places(fit: camera_calibration.Fit, sightings: Sequence[Sighting]) -> list[int]:
    """Resting places whose sightings disagree so much that the ball was probably moved part way through."""
    worst: dict[int, float] = {}
    for sighting, error in zip(sightings, fit.desk_errors_m):
        worst[sighting.place] = max(worst.get(sighting.place, 0.0), error)
    return sorted(place for place, error in worst.items() if error > MOVED_M)
