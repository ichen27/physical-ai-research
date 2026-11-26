"""Find the wrist camera's focal length and its pose on the wrist from pictures of the ball.

Nothing has to be measured by hand. The ball is put on the desk and left
alone while the arm looks at it from several poses. Each picture gives the
ball's centre and size in pixels, and the arm's own joint readings say how the
wrist moved between pictures. The ball did not move, so there is one camera
(focal length, where the line of sight meets the picture, lens distortion, and
the camera's position and direction on the wrist) and one place on the desk
that explain all the pictures at once. Those are found together by least
squares. Putting the ball somewhere else and repeating adds two unknowns (the
new place) and many more equations.

The answer is in the arm's own frame, because the arm's joint readings are what
carried the camera about. A swivel zero that is a few degrees out turns the
camera's answers and the arm's idea of where its gripper is by the same amount.

What is left over after the fit, turned into millimetres on the desk, says how
good the camera is. Needs ``numpy``.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from . import arm, pickup
from .camera_model import WristCamera, sight_many
from .kinematics import Chain

MIN_SIGHTINGS = 8
# Lens terms beyond the focal length are fitted only with this many sightings in two or more places.
MIN_SIGHTINGS_FOR_LENS = 16
# A ball's size in the picture is a rougher measurement than its centre, so it counts for less.
RADIUS_WEIGHT = 0.3
# Two fits whose cameras point this far apart are different answers.
DISTINCT_RAD = math.radians(8.0)


@dataclass(frozen=True)
class Sighting:
    """One picture of the ball: the five arm servo readings and where the ball appeared."""

    ticks: tuple[int, ...]
    u: float
    v: float
    radius_px: float
    place: int  # sightings with the same number saw the ball resting in one place


@dataclass(frozen=True)
class Fit:
    camera: WristCamera
    places: tuple[tuple[float, float], ...]  # where the ball rested each time, metres on the desk
    pixel_rms: float
    desk_errors_m: tuple[float, ...]  # per sighting: how far the camera puts the ball from where it rested
    fitted_lens: bool  # whether the picture's centre and the distortion were fitted or left at their defaults
    runner_up_pixel_rms: float | None  # the best fit that is a different answer, if one was found

    @property
    def desk_rms_m(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.desk_errors_m))))

    def ambiguous(self) -> bool:
        """True if a clearly different camera explains the pictures almost as well."""
        return self.runner_up_pixel_rms is not None and self.runner_up_pixel_rms < max(2.0 * self.pixel_rms, 3.0)


def load_sightings(path: str | Path) -> list[Sighting]:
    path = Path(path)
    if not path.exists():
        return []
    rows = json.loads(path.read_text())["sightings"]
    return [
        Sighting(tuple(int(t) for t in row["ticks"]), float(row["u"]), float(row["v"]), float(row["radius_px"]), int(row["place"]))
        for row in rows
    ]


def load_notes(path: str | Path) -> dict:
    """Whatever was saved beside the sightings, such as the ball's colour."""
    path = Path(path)
    if not path.exists():
        return {}
    return {key: value for key, value in json.loads(path.read_text()).items() if key != "sightings"}


def save_sightings(path: str | Path, sightings: Sequence[Sighting], **notes) -> None:
    Path(path).write_text(json.dumps({**notes, "sightings": [asdict(s) for s in sightings]}, indent=2) + "\n")


def _least_squares(residual: Callable[[np.ndarray], np.ndarray], start: np.ndarray, steps: np.ndarray,
                   rounds: int = 60) -> tuple[np.ndarray, float]:
    """Damped least squares from one starting guess. ``steps`` are the sizes used to measure slopes."""
    best, damping = start.astype(float), 1e-3
    error = residual(best)
    cost = float(error @ error)
    for _ in range(rounds):
        jacobian = np.column_stack([
            (residual(best + step * unit) - error) / step for step, unit in zip(steps, np.eye(len(best)))
        ])
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ error
        improved = False
        for _ in range(8):
            try:
                move = np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-12), -gradient)
            except np.linalg.LinAlgError:
                damping *= 10
                continue
            trial = best + move
            trial_error = residual(trial)
            trial_cost = float(trial_error @ trial_error)
            if np.isfinite(trial_cost) and trial_cost < cost:
                converged = cost - trial_cost < 1e-10 * max(cost, 1e-12)
                best, error, cost, damping, improved = trial, trial_error, trial_cost, damping / 3, True
                break
            damping *= 10
        if not improved or converged:
            break
    return best, cost


def _tips(sightings: Sequence[Sighting], calibration: arm.Calibration, chain: Chain) -> tuple[np.ndarray, np.ndarray]:
    transforms = np.array([chain.tip_transform(calibration.to_rad(s.ticks)) for s in sightings])
    return transforms[:, :3, :3], transforms[:, :3, 3]


def located(camera: WristCamera, sightings: Sequence[Sighting], calibration: arm.Calibration,
            chain: Chain | None = None) -> list[tuple[float, float] | None]:
    """Where on the desk the camera puts the ball in each sighting; None where its line of sight misses the desk."""
    chain = chain or Chain()
    out = []
    for s in sightings:
        point = camera.locate(s.u, s.v, calibration.to_rad(s.ticks), chain)
        out.append(None if point is None else (point[0], point[1]))
    return out


def spread(camera: WristCamera, sightings: Sequence[Sighting], calibration: arm.Calibration,
           chain: Chain | None = None) -> tuple[tuple[float, float], float, float]:
    """For sightings of the ball in ONE place: where the camera puts it on average, and how far the
    sightings disagree with that (typical and worst, metres).

    A camera that is right puts a ball that did not move in the same place from every pose,
    so this checks a camera against pictures that played no part in fitting it.
    """
    points = [p for p in located(camera, sightings, calibration, chain) if p is not None]
    if len(points) < 2:
        raise ValueError("two or more sightings that reach the desk are needed")
    points = np.array(points)
    middle = points.mean(axis=0)
    off = np.linalg.norm(points - middle, axis=1)
    return (float(middle[0]), float(middle[1])), float(np.sqrt(np.mean(off**2))), float(off.max())


def fit(
    sightings: Sequence[Sighting],
    calibration: arm.Calibration,
    width: int = 1280,
    height: int = 720,
    chain: Chain | None = None,
    lens: bool | None = None,
    start: WristCamera | None = None,
    tick: Callable[[], object] | None = None,
    known: Mapping[int, Sequence[float]] | None = None,
) -> Fit:
    """The camera that best explains the sightings.

    ``lens`` says whether to fit the picture's centre and the lens distortion as well as
    the focal length and the mount. Left as None it is decided by how much there is to
    go on. ``start`` is a camera to try first, such as the last fit.

    ``known`` gives, for some of the resting places (by their number in the sightings),
    where the ball was: x, y in metres in the arm's frame, found some other way, such as
    by resting the gripper tip on the spot before the ball was put there. Those places are
    held where they are said to be and only the camera is fitted to them.

    With no place known, the answer is in the arm model's frame *with the arm raised*, and
    that is only as true as the arm's joint zeros; the fit's own error does not show
    whether they are. On the first real arm the zeros came from four touches of the desk,
    which hold them loosely (several degrees of change for a millimetre of touch error),
    and on a simulated arm with its wrist zero 4 degrees out the fitted camera agrees with
    itself to 2 mm while putting unseen balls 12 mm off. Whether the real arm has such an
    error has not been established. A known place ties the camera to where the gripper is
    when it is down at the desk, which is what a pickup needs, takes any such error into
    the camera's mount, and shows how large it was.

    A fit takes about half a second. ``tick`` is called thousands of times while it runs,
    so that a caller holding a powered arm can keep talking to its servos (see
    ``understudy.camera_collect.Pulse``).
    """
    sightings = list(sightings)
    if len(sightings) < MIN_SIGHTINGS:
        raise ValueError(f"{MIN_SIGHTINGS} or more sightings are needed; {len(sightings)} given")
    place_numbers = sorted({s.place for s in sightings})
    place_of = np.array([place_numbers.index(s.place) for s in sightings])
    known = {int(number): (float(xy[0]), float(xy[1])) for number, xy in (known or {}).items()}
    strangers = sorted(set(known) - set(place_numbers))
    if strangers:
        raise ValueError(f"no sightings were taken at the known place(s) {strangers}; the places are {place_numbers}")
    free = [index for index, number in enumerate(place_numbers) if number not in known]  # places still to be found
    if lens is None:
        lens = len(sightings) >= MIN_SIGHTINGS_FOR_LENS and len(place_numbers) >= 2
    chain = chain or Chain()
    tip_rotations, tip_positions = _tips(sightings, calibration, chain)
    seen = np.array([[s.u, s.v, s.radius_px] for s in sightings])
    radius = pickup.BALL_RADIUS_M
    centre = (width / 2.0, height / 2.0)
    count = len(sightings)

    # Unknowns: focal length (as a multiple of the picture's width), mount position, mount
    # rotation, then two per resting place not already known, then, if asked for, the picture
    # centre and distortion.
    held = np.array([known.get(number, (0.0, 0.0)) for number in place_numbers], dtype=float).reshape(-1, 2)

    def unpack(x: np.ndarray):
        places = held.copy()
        places[free] = x[7 : 7 + 2 * len(free)].reshape(-1, 2)
        if len(x) > 7 + 2 * len(free):
            cx, cy, distortion = x[-3] * width, x[-2] * width, x[-1]
        else:
            cx, cy, distortion = centre[0], centre[1], 0.0
        return x[0] * width, (cx, cy), distortion, x[1:4], x[4:7], places

    def residual(x: np.ndarray) -> np.ndarray:
        if tick is not None:
            tick()
        focal, middle, distortion, xyz, rotation, places = unpack(x)
        if focal <= 0:
            return np.full(3 * count, 1e6)
        balls = np.column_stack([places[place_of], np.full(count, radius)])
        u, v, r, ahead = sight_many(focal, middle, distortion, xyz, rotation, tip_rotations, tip_positions, balls)
        out = np.column_stack([u - seen[:, 0], v - seen[:, 1], RADIUS_WEIGHT * (r - seen[:, 2])])
        out[~ahead] = 2000.0  # a ball behind the camera cannot have been seen
        return np.clip(out, -5000.0, 5000.0).ravel()

    def vector(camera: WristCamera, with_lens: bool) -> np.ndarray | None:
        points = located(camera, sightings, calibration, chain)
        places = []
        for index in free:
            here = [p for p, which in zip(points, place_of) if which == index and p is not None]
            if not here:
                return None
            places.extend(np.median(np.array(here), axis=0))
        x = [camera.focal_px / width, *camera.mount_xyz, *camera.mount_rotation, *places]
        if with_lens:
            x += [camera.centre_px[0] / width, camera.centre_px[1] / width, camera.distortion]
        return np.array(x, dtype=float)

    def camera_from(x: np.ndarray) -> WristCamera:
        focal, middle, distortion, xyz, rotation, _ = unpack(x)
        return WristCamera(width, height, float(focal), (float(middle[0]), float(middle[1])), float(distortion),
                           tuple(float(v) for v in xyz), tuple(float(v) for v in rotation))

    def steps_for(x: np.ndarray) -> np.ndarray:
        steps = np.full(len(x), 1e-4)
        steps[0] = 1e-4
        return steps

    # Started from the camera looking along the gripper in each of four ways up, at several
    # tilts and focal lengths, since a wrong start can settle on a wrong answer.
    guesses = [] if start is None else [replace(start, width=width, height=height)]
    for quarter_turns, tilt, focal in itertools.product(range(4), (-0.4, 0.0, 0.4, 0.8), (0.35, 0.6, 1.0)):
        guesses.append(WristCamera.looking_along_gripper(width, height, focal * width, tilt_down_rad=tilt,
                                                         roll_quarter_turns=quarter_turns))
    found = []
    for guess in guesses:
        if tick is not None:
            tick()
        x0 = vector(guess, False)
        if x0 is None:
            continue
        x, cost = _least_squares(residual, x0, steps_for(x0))
        found.append((cost, x))
    if not found:
        raise ValueError("no starting guess puts the ball on the desk in front of the camera; are these sightings of the ball?")
    found.sort(key=lambda item: item[0])

    def sight_direction(x: np.ndarray) -> np.ndarray:
        from .camera_model import rotation_from_vector

        return rotation_from_vector(x[4:7])

    def same_answer(a: np.ndarray, b: np.ndarray) -> bool:
        turn = sight_direction(a).T @ sight_direction(b)
        angle = math.acos(min(1.0, max(-1.0, (np.trace(turn) - 1.0) / 2.0)))
        return angle < DISTINCT_RAD and abs(a[0] - b[0]) < 0.15 * a[0]

    def pixel_rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(residual(x).reshape(-1, 3)[:, :2]))))

    best_cost, best = found[0]
    runner_up = next((pixel_rms(x) for cost, x in found[1:] if not same_answer(x, best)), None)
    fitted_lens = False
    if lens:
        x0 = np.concatenate([best, [centre[0] / width, centre[1] / width, 0.0]])
        refined, cost = _least_squares(residual, x0, steps_for(x0), rounds=120)
        _, middle, distortion, _, _, _ = unpack(refined)
        believable = (
            abs(middle[0] - centre[0]) < 0.2 * width and abs(middle[1] - centre[1]) < 0.2 * width and abs(distortion) < 0.6
        )
        if believable and cost < best_cost:
            best, fitted_lens = refined, True

    camera = camera_from(best)
    _, _, _, _, _, places = unpack(best)
    points = located(camera, sightings, calibration, chain)
    errors = [
        float("inf") if point is None else math.dist(point, places[which]) for point, which in zip(points, place_of)
    ]
    return Fit(camera, tuple((float(p[0]), float(p[1])) for p in places), pixel_rms(best), tuple(errors), fitted_lens,
               runner_up)
