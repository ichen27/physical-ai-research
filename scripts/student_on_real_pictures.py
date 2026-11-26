"""Score the student's eye on real pictures from the arm's camera.

The student was trained on simulated pictures only. This shows it real ones, with the
arm's real joint readings, and compares where it says the ball is with where the ball was.

With the output of the real arm's camera calibration (``real_arm.py camera-collect``), where
the ball sat still while the arm looked at it from about twenty poses, twice:

    python scripts/student_on_real_pictures.py --student results/student/<name>/student.pt \
        --sightings sightings.local.json --calibration calibration.local.json --shots ~/understudy-camera/shots

Each saved picture is matched to its recorded sighting by finding the ball in it again with
the hand-written detector; the ball's true place is the one the calibration fitted for that
resting place. Nothing is read from the arm and nothing moves.

With pictures and no known ball position, it only prints what the student says:

    python scripts/student_on_real_pictures.py --student ... --pictures tests/fixtures/camera/hold1.jpg

Needs torch, numpy and opencv-python-headless.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from understudy import arm, camera_calibration, pickup, runtime, sim_look, student, vision  # noqa: E402

DEFAULT_HOLD_TARGET = (0.19, 0.0, 0.13)


def readings_for(angles: list[float], finger: float = pickup.GRIPPER_OPEN_RAD) -> np.ndarray:
    """The student's joint readings for an arm held still in a pose."""
    readings = np.zeros(student.PROPRIO_SIZE, dtype=np.float32)
    readings[:6] = np.subtract([*angles, finger], runtime.SENSED_START)
    readings[12:15] = DEFAULT_HOLD_TARGET
    return readings


def matched(sightings: list, shots: Path, colour: vision.BallColour) -> list[tuple[Path, object]]:
    """Each saved picture that a recorded sighting was taken from, with that sighting.

    A sighting records where the detector found the ball, to a fraction of a pixel, and the
    same detector on the same picture finds it in the same place. That is the match.
    """
    pairs, left = [], list(sightings)
    for path in sorted(shots.glob("*.jpg")):
        try:
            blobs = vision.find_blobs(vision.decode(path.read_bytes()), colour)
        except ValueError:
            continue
        for blob in blobs:
            hit = next((s for s in left if abs(s.u - blob.u) < 0.01 and abs(s.v - blob.v) < 0.01), None)
            if hit is not None:
                pairs.append((path, hit))
                left.remove(hit)
                break
    return pairs


def score_calibration_pictures(brain: student.Brain, args) -> dict:
    calibration = arm.Calibration.load(args.calibration)
    sightings = camera_calibration.load_sightings(args.sightings)
    notes = camera_calibration.load_notes(args.sightings)
    if not sightings:
        raise SystemExit(f"no sightings in {args.sightings}")
    fit = camera_calibration.fit(sightings, calibration, int(notes.get("width", 1280)), int(notes.get("height", 720)))
    place_numbers = sorted({s.place for s in sightings})
    pairs = matched(sightings, Path(args.shots).expanduser(), vision.colour_from(notes.get("colour")))
    rows = []
    for path, sighting in pairs:
        truth = fit.places[place_numbers.index(sighting.place)]
        picture = sim_look.shrink(vision.decode(path.read_bytes()))
        (x, y, _), sure = brain.glance(picture, readings_for(calibration.to_rad(sighting.ticks)))
        rows.append({"picture": path.name, "place": sighting.place, "ball": [round(v, 4) for v in truth],
                     "student_says": [round(x, 4), round(y, 4)], "in_view": round(sure, 3),
                     "error_mm": round(1000 * math.dist((x, y), truth), 1)})
    errors = [row["error_mm"] for row in rows]
    return {
        "pictures_matched_to_sightings": len(rows),
        "sightings": len(sightings),
        "resting_places": [[round(v, 4) for v in place] for place in fit.places],
        "the_calibration's_own_error_on_the_desk_mm": round(fit.desk_rms_m * 1000, 2),
        "student_error_mm": {"median": float(np.median(errors)), "worst": max(errors),
                             "share_within_10_mm": float(np.mean(np.array(errors) <= 10.0))} if errors else None,
        "student_judged_ball_in_view_share": float(np.mean([row["in_view"] >= 0.5 for row in rows])) if rows else None,
        "pictures": rows,
    }


def describe_pictures(brain: student.Brain, args) -> dict:
    angles = list(args.angles) if args.angles else list(pickup.START_POSE)
    rows = []
    for name in args.pictures:
        picture = sim_look.shrink(vision.decode(Path(name).expanduser().read_bytes()))
        (x, y, z), sure = brain.glance(picture, readings_for(angles, args.finger))
        rows.append({"picture": Path(name).name, "student_says": [round(x, 4), round(y, 4), round(z, 4)], "in_view": round(sure, 3)})
    return {"known_ball_positions": "none: these pictures only show what the student says",
            "joint_angles_assumed_rad": angles, "pictures": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--student", required=True)
    parser.add_argument("--sightings", help="The sightings file written by real_arm.py camera-collect.")
    parser.add_argument("--calibration", help="The arm's calibration file.")
    parser.add_argument("--shots", help="The folder of pictures the camera watcher saved during camera-collect.")
    parser.add_argument("--pictures", nargs="+", help="Pictures with no known ball position, to see what the student says.")
    parser.add_argument("--angles", type=float, nargs=5, help="For --pictures: the five arm joint angles, rad (default: the start pose).")
    parser.add_argument("--finger", type=float, default=pickup.GRIPPER_OPEN_RAD, help="For --pictures: the finger angle, rad.")
    parser.add_argument("--output", help="Write the results to this JSON file.")
    args = parser.parse_args()
    net, notes = student.load(args.student)
    brain = student.Brain(net)
    if args.sightings:
        if not (args.calibration and args.shots):
            parser.error("--sightings needs --calibration and --shots")
        result = score_calibration_pictures(brain, args)
    elif args.pictures:
        result = describe_pictures(brain, args)
    else:
        parser.error("give --sightings (with --calibration and --shots) or --pictures")
    result = {"student": str(args.student), "student_training": notes, **result}
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
