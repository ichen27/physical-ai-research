"""Measure the student searching for the ball over a wide sector and then picking it up, in MuJoCo.

The ball is put anywhere the arm can turn to face and lift from. The student, shown only
pictures, sweeps the raised arm left and right until it sees the ball, faces it, looks again,
comes down and picks it up (``understudy.dagger.run_attempt`` with ``facing="search"``). The
teacher, told where the ball is, is run on the same balls for comparison.

    python scripts/evaluate_search.py --student results/student/<name>/student.pt --part student --output out.json
    python scripts/evaluate_search.py --student ... --part teacher --output out.json
    python scripts/evaluate_search.py --student ... --part no-ball --output out.json     # does it see a ball that is not there?
    python scripts/evaluate_search.py --student ... --part outside --output out.json     # does it refuse a ball too near or too far?

Every attempt has a look the student was not trained on, the real arm's speed limits, and
goes through ``understudy.runner.run_pickup`` in servo ticks. Needs torch, mujoco,
onnxruntime and OpenGL. No GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from understudy import dagger, pickup, runner, runtime, seek, sim_look, student  # noqa: E402
from understudy.camera_model import WristCamera  # noqa: E402
from understudy.runner import AIM_LEFT_M, REACH_BAND_M, SWIVEL_AXIS_X_M  # noqa: E402

DISTANCE_BINS_CM = ((15, 18), (18, 21), (21, 24))


def ball_at(distance: float, turn: float) -> tuple[float, float]:
    """A ball ``distance`` from the swivel axis that the arm must turn ``turn`` rad to face."""
    bearing = turn + math.asin(AIM_LEFT_M / distance)
    return (SWIVEL_AXIS_X_M + distance * math.cos(bearing), distance * math.sin(bearing))


def share(rows: list[dict], key) -> float | None:
    return round(sum(bool(key(row)) for row in rows) / len(rows), 4) if rows else None


def table(rows: list[dict], width_deg: float) -> dict:
    """Find and pickup rates overall, by how far round the ball was, and by how far out."""
    def line(some: list[dict]) -> dict:
        found = [row for row in some if row["found"]]
        errors = [row["facing_error_m"] for row in found if row.get("facing_error_m") is not None]
        return {
            "attempts": len(some),
            "found": share(some, lambda row: row["found"]),
            "lifted_at_the_end": share(some, lambda row: row["lifted"]),
            "lifted_at_the_end_of_those_found": share(found, lambda row: row["lifted"]),
            "refused": share(some, lambda row: row["refused"]),
            "median_looks": float(np.median([row["looks"] for row in found])) if found else None,
            "first_guess_error_mm_median": round(1000 * float(np.median(errors)), 1) if errors else None,
        }

    edges = np.linspace(-width_deg, width_deg, 6)
    by_bearing = {
        f"{low:+.0f} to {high:+.0f} deg": line([row for row in rows if low <= row["turn_deg"] < high or (high == edges[-1] and row["turn_deg"] == high)])
        for low, high in zip(edges[:-1], edges[1:])
    }
    by_distance = {
        f"{low} to {high} cm": line([row for row in rows if low <= row["distance_cm"] < high or (high == 24 and row["distance_cm"] == 24)])
        for low, high in DISTANCE_BINS_CM
    }
    return {"all": line(rows), "by_turn_needed_to_face_the_ball": by_bearing, "by_distance_from_swivel_axis": by_distance}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", default=str(REPO / "results/pickup/2026-11-24-claw/policy.onnx"))
    parser.add_argument("--part", required=True, choices=["student", "teacher", "no-ball", "outside"])
    parser.add_argument("--attempts", type=int, default=200)
    parser.add_argument("--width_deg", type=float, default=math.degrees(runner.MAX_TURN_RAD),
                        help="Balls are put where facing them means turning up to this far either way.")
    parser.add_argument("--max_turn_deg", type=float,
                        help="Let the arm turn this far to face a ball, instead of the usual limit. For finding out what going wider costs.")
    parser.add_argument("--bare_desk", action="store_true", help="With --part no-ball: nothing at all on the desk, not even the other things.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", help="Write the results to this JSON file.")
    args = parser.parse_args()

    if args.max_turn_deg is not None:
        runner.MAX_TURN_RAD = math.radians(args.max_turn_deg)
    torch.set_num_threads(1)
    teacher = runtime.OnnxPolicy(args.teacher)
    net, notes = student.load(args.student)
    brain = student.Brain(net)
    nominal = WristCamera(**{key: tuple(value) if isinstance(value, list) else value
                             for key, value in notes["nominal_camera"].items()}) if "nominal_camera" in notes else None
    sim, scene = dagger.bench(nominal)
    scene.held_out = True
    scene.clutter = not args.bare_desk
    rng = random.Random(args.seed)
    seconds = pickup.EPISODE_SECONDS * 3
    started = time.monotonic()
    rows = []
    for _ in range(args.attempts):
        if args.part == "outside":
            distance = rng.choice((rng.uniform(0.10, REACH_BAND_M[0] - 0.005), rng.uniform(REACH_BAND_M[1] + 0.005, 0.31)))
        else:
            distance = rng.uniform(*REACH_BAND_M)
        turn = math.radians(rng.uniform(-args.width_deg, args.width_deg))
        ball = ball_at(distance, turn)
        if args.part == "teacher":
            outcome, _ = dagger.run_attempt(sim, scene, rng, teacher=teacher, drive="teacher", facing="exact", seconds=seconds, ball=ball)
        else:
            outcome, _ = dagger.run_attempt(sim, scene, rng, brain=brain, drive="student", facing="search", seconds=seconds,
                                            ball=ball, ball_present=args.part != "no-ball")
        rows.append({**outcome, "turn_deg": round(math.degrees(turn), 2), "distance_cm": round(100 * distance, 2)})

    result = {
        "part": args.part,
        "simulator": f"MuJoCo {mujoco.__version__}, vendor model, rendered wrist camera, looks the student was not trained on",
        "student": str(args.student),
        "student_training": notes,
        "attempts": len(rows),
        "seed": args.seed,
        "seconds": seconds,
        "max_speed_rad_s": student.SPEED_LIMIT_RAD_S,
        "max_lead_rad": student.LEAD_LIMIT_RAD,
        "sweep_of_looks_rad": list(seek.SCAN),
        "look_pose": list(seek.LOOK_POSE),
        "arm_turns_to_face_at_most_deg": round(math.degrees(runner.MAX_TURN_RAD), 1),
        "minutes": round((time.monotonic() - started) / 60, 1),
    }
    if args.part in ("student", "teacher"):
        result["balls"] = (f"{REACH_BAND_M[0] * 100:g} to {REACH_BAND_M[1] * 100:g} cm from the swivel axis, "
                           f"needing a turn of up to {args.width_deg:g} degrees either way to face")
        result["ball_position_given_to_the_driver"] = "the true position" if args.part == "teacher" else "none: pictures only"
        result.update(table(rows, args.width_deg))
        result["safety_stops"] = share(rows, lambda row: row["stopped"])
    elif args.part == "no-ball":
        result["desk"] = "bare" if args.bare_desk else "only the other things, red and orange ones among them"
        result["said_there_was_a_ball"] = share(rows, lambda row: row["found"])
        result["attempted_a_pickup"] = share(rows, lambda row: row["found"] and row["refused"] is None)
    else:
        near = [row for row in rows if row["distance_cm"] < REACH_BAND_M[0] * 100]
        far = [row for row in rows if row["distance_cm"] > REACH_BAND_M[1] * 100]
        result["balls"] = f"10 to {REACH_BAND_M[0] * 100 - 0.5:g} cm and {REACH_BAND_M[1] * 100 + 0.5:g} to 31 cm from the swivel axis"
        for label, some, word in (("too_near", near, "too near"), ("too_far", far, "too far")):
            result[label] = {
                "attempts": len(some),
                "found": share(some, lambda row: row["found"]),
                "refused_as_" + label: share(some, lambda row: row["refused"] == word),
                "attempted_a_pickup": share(some, lambda row: row["found"] and row["refused"] is None),
                "of_those_attempted_within_1_cm_of_the_band": share(
                    [row for row in some if row["found"] and row["refused"] is None],
                    lambda row: REACH_BAND_M[0] * 100 - 1 <= row["distance_cm"] <= REACH_BAND_M[1] * 100 + 1),
            }
    result["attempt_records"] = rows
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("attempt_records", "student_training")}, indent=1))
    scene.close()


if __name__ == "__main__":
    main()
