"""Three checks of a student against real pictures from the arm's camera, beyond ``student_on_real_pictures.py``.

They need a folder of real picture sets that is not in this repository (the pictures show
the author's desk): one sub-folder per session, and in one of them the files of a real
camera calibration (``sightings.local.json``, ``calibration.local.json``), which give the
arm's pose and the ball's true place for some of the pictures.

    python scripts/student_real_checks.py all     --student student.pt --samples FOLDER --calibrated cal4
    python scripts/student_real_checks.py colour  --student student.pt --samples FOLDER --calibrated cal4
    python scripts/student_real_checks.py changes --student student.pt --samples FOLDER --calibrated cal4

``all``: on every real picture that shows the ball whole, is the student sure the ball is
there; and where the ball's place is known, how far out is it. Where the arm's pose was not
recorded, the start pose is assumed (every such picture so far was taken from it or from
near it), and only "is it sure" is scored. The eye uses the pose: told the wrong one, it is
less sure of a ball it is sure of when told the right one.

``colour``: the ball's colour as the student sees it, in the real pictures against the
simulated pictures it is trained on.

``changes``: one thing about the ball's look is changed at a time in the real pictures, to
find what it is that leaves the student unsure.

Nothing is read from the arm and nothing moves. Needs torch, mujoco (for ``colour``), numpy and opencv.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from understudy import arm, camera_calibration, pickup, sim_look, student, vision  # noqa: E402

_spec = importlib.util.spec_from_file_location("student_on_real_pictures", REPO / "scripts" / "student_on_real_pictures.py")
real = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(real)

SMALL = sim_look.PICTURE_WIDTH / 1280.0


def disc(shape, u: float, v: float, radius: float) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    cv2.circle(mask, (int(round(u)), int(round(v))), max(1, int(round(radius))), 255, -1)
    return mask > 0


def whole_ball(image: np.ndarray, colour: vision.BallColour):
    blobs = [blob for blob in vision.find_blobs(image, colour) if blob.whole() and blob.radius_px > 40]
    return max(blobs, key=lambda blob: blob.area_px) if blobs else None


def ball_colour(small_bgr: np.ndarray, u: float, v: float, radius: float) -> dict:
    """The middle of the ball in a student-sized picture: its usual hue, saturation, brightness and colour."""
    inside = disc(small_bgr.shape, u, v, radius * 0.8)
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    blue, green, red = (float(np.median(small_bgr[..., channel][inside])) for channel in range(3))
    return {"hue_deg": float(np.median(hsv[..., 0][inside])) * 2, "saturation": float(np.median(hsv[..., 1][inside])) / 255,
            "brightness": float(np.median(hsv[..., 2][inside])) / 255, "red": red, "green": green, "blue": blue}


class Calibrated:
    """The calibrated set: the arm's calibration, the ball's fitted place and which pictures have a recorded pose."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.calibration = arm.Calibration.load(folder / "calibration.local.json")
        sightings = camera_calibration.load_sightings(folder / "sightings.local.json")
        notes = camera_calibration.load_notes(folder / "sightings.local.json")
        self.colour = vision.colour_from(notes.get("colour"))
        fit = camera_calibration.fit(sightings, self.calibration, int(notes.get("width", 1280)), int(notes.get("height", 720)))
        numbers = sorted({s.place for s in sightings})
        self.pairs = real.matched(sightings, folder, self.colour)
        self.known = {path.name: (s, fit.places[numbers.index(s.place)]) for path, s in self.pairs}


def folders(samples: Path, sets: list[str] | None) -> list[Path]:
    """The picture sets to use: those named, or every sub-folder."""
    return sorted(p for p in samples.iterdir() if p.is_dir() and (not sets or p.name in sets))


def check_all(brain: student.Brain, samples: Path, calibrated: Calibrated, sets: list[str] | None = None) -> dict:
    rows = []
    for folder in folders(samples, sets):
        for path in sorted(folder.glob("*.jpg")):
            image = vision.decode(path.read_bytes())
            if whole_ball(image, calibrated.colour) is None:
                continue
            known = calibrated.known.get(path.name) if folder == calibrated.folder else None
            if known is not None:
                pose = calibrated.calibration.to_rad(known[0].ticks)
            else:
                pose = list(pickup.START_POSE)
            (x, y, _), sure = brain.glance(sim_look.shrink(image), real.readings_for(pose))
            rows.append({"set": folder.name, "picture": path.name, "pose_recorded": known is not None, "sure": round(sure, 3),
                         "error_mm": round(1000 * math.dist((x, y), known[1]), 1) if known is not None else None})

    def share(some: list) -> str:
        return f"{sum(row['sure'] >= 0.5 for row in some)} of {len(some)}"

    errors = [row["error_mm"] for row in rows if row["error_mm"] is not None]
    return {
        "pictures_showing_the_ball_whole": len(rows),
        "sure_the_ball_is_there": share(rows),
        "by_set": {name: share([row for row in rows if row["set"] == name]) for name in sorted({row["set"] for row in rows})},
        "with_recorded_pose_and_known_ball_place": {
            "pictures": len(errors), "sure": share([row for row in rows if row["pose_recorded"]]),
            "error_mm_median": float(np.median(errors)), "error_mm_worst": max(errors), "within_10_mm": int(sum(e <= 10 for e in errors))},
        "median_sureness": round(float(np.median([row["sure"] for row in rows])), 2),
        "pictures": rows,
    }


def check_colour(notes: dict, samples: Path, calibrated: Calibrated, looks: int = 300, sets: list[str] | None = None) -> dict:
    from understudy import dagger
    from understudy.camera_model import WristCamera

    per_set, every = {}, []
    for folder in folders(samples, sets):
        found = []
        for path in sorted(folder.glob("*.jpg")):
            image = vision.decode(path.read_bytes())
            blob = whole_ball(image, calibrated.colour)
            if blob is not None:
                found.append(ball_colour(sim_look.shrink(image), blob.u * SMALL, blob.v * SMALL, blob.radius_px * SMALL))
        every += found
        if found:
            per_set[folder.name] = {"pictures": len(found), **{key: [min(f[key] for f in found), float(np.median([f[key] for f in found])), max(f[key] for f in found)]
                                                              for key in ("red", "green", "blue")}}
    nominal = WristCamera(**{key: tuple(value) if isinstance(value, list) else value for key, value in notes["nominal_camera"].items()})
    sim, scene = dagger.bench(nominal)
    rng, simulated = random.Random(4), []
    for index in range(looks):
        scene.held_out = index % 2 == 1
        ball = dagger.place_ball(rng)
        sim.reset(ball)
        look = scene.new_look(rng, ball)
        picture, visible = scene.picture_now()
        if visible < 0.95:
            continue
        sight = look.camera.sight(sim.ball(), sim.joint_angles()[:5], scene._chain)
        scale = sim_look.PICTURE_WIDTH / look.camera.width
        simulated.append(ball_colour(picture, sight.u * scale, sight.v * scale, sight.radius_px * scale))
    scene.close()
    out = {"real_pictures": len(every), "simulated_pictures": len(simulated), "real_by_set_least_median_most": per_set}
    inside = np.ones(len(simulated), bool)
    for key in ("hue_deg", "saturation", "brightness", "red", "green", "blue"):
        r, s = np.array([f[key] for f in every]), np.array([f[key] for f in simulated])
        within = (s >= r.min()) & (s <= r.max())
        if key in ("hue_deg", "saturation", "brightness"):
            inside &= within
        out[key] = {"real_median": round(float(np.median(r)), 2), "real_range": [round(float(r.min()), 2), round(float(r.max()), 2)],
                    "simulated_median": round(float(np.median(s)), 2),
                    "simulated_5th_to_95th": [round(float(np.percentile(s, 5)), 2), round(float(np.percentile(s, 95)), 2)],
                    "share_of_simulated_inside_the_real_range": round(float(within.mean()), 3)}
    out["simulated_balls_inside_the_real_range_on_hue_saturation_and_brightness_together"] = round(float(inside.mean()), 3)
    return out


def check_changes(brain: student.Brain, calibrated: Calibrated) -> dict:
    def ball(image, s):
        return disc(image.shape, s.u, s.v, s.radius_px * 1.02)

    def as_taken(image, s):
        return image

    def stronger(image, s):
        hsv, where = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32), ball(image, s)
        hsv[..., 1][where] = np.clip(hsv[..., 1][where] * 1.6, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def no_pale(image, s):
        out, where, hsv = image.copy(), ball(image, s), cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        pale = where & ((hsv[..., 1] < 110) | (hsv[..., 2] > 235))
        body = where & ~pale
        if body.sum() > 50:
            out[pale] = np.median(image[body], axis=0)
        return out

    def darker_ball(image, s):
        out = image.astype(np.float32)
        out[ball(image, s)] *= 0.8
        return np.clip(out, 0, 255).astype(np.uint8)

    def darker(image, s):
        return np.clip(image.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)

    def no_veil(image, s):
        grey = vision.veil(image)
        return np.clip((image.astype(np.float32) - grey) * 255.0 / (255.0 - grey), 0, 255).astype(np.uint8)

    def plain_orange(image, s):
        out, where = image.copy(), ball(image, s)
        light = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        shade = light / max(float(np.median(light[where])), 1.0)
        for channel, value in enumerate((30, 110, 235)):
            out[..., channel][where] = np.clip(value * shade[where], 0, 255)
        return out

    out = {}
    for name, change in (("as taken", as_taken), ("the ball's colour made 1.6 times stronger", stronger),
                         ("the pale and blown-out parts of the ball filled with its own colour", no_pale),
                         ("the ball 20% darker", darker_ball), ("the whole picture 20% darker", darker),
                         ("the veil taken off the whole picture", no_veil),
                         ("the ball repainted a plain orange, keeping its shading", plain_orange)):
        errors, sure = [], []
        for path, s in calibrated.pairs:
            image = change(vision.decode(path.read_bytes()), s)
            (x, y, _), p = brain.glance(sim_look.shrink(image), real.readings_for(calibrated.calibration.to_rad(s.ticks)))
            errors.append(1000 * math.dist((x, y), calibrated.known[path.name][1]))
            sure.append(p)
        out[name] = {"sure_the_ball_is_there": f"{sum(p >= 0.5 for p in sure)} of {len(sure)}", "median_sureness": round(float(np.median(sure)), 2),
                     "error_mm_median": round(float(np.median(errors)), 1), "error_mm_worst": round(max(errors), 1)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("check", choices=["all", "colour", "changes"])
    parser.add_argument("--student", required=True)
    parser.add_argument("--samples", required=True, help="The folder of real picture sets, one sub-folder per session.")
    parser.add_argument("--calibrated", required=True, help="The sub-folder that holds a real camera calibration's files.")
    parser.add_argument("--sets", nargs="+", help="Use only these sub-folders (default: all of them). Where a picture's pose was not "
                        "recorded the start pose is assumed, so leave out sets taken from another pose.")
    parser.add_argument("--output", help="Write the results to this JSON file.")
    args = parser.parse_args()
    net, notes = student.load(args.student)
    samples = Path(args.samples).expanduser()
    calibrated = Calibrated(samples / args.calibrated)
    if args.check == "all":
        result = check_all(student.Brain(net), samples, calibrated, args.sets)
    elif args.check == "colour":
        result = check_colour(notes, samples, calibrated, sets=args.sets)
    else:
        result = check_changes(student.Brain(net), calibrated)
    result = {"check": args.check, "student": Path(args.student).name, "student_input": getattr(net, "input", "colour"), **result}
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "pictures"}, indent=1))


if __name__ == "__main__":
    main()
