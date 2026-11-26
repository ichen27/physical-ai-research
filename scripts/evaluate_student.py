"""Measure the picture-driven student's pickup rate in MuJoCo, beside the teacher's.

Every attempt has a look the student was not trained on (new draws of light, desk, clutter
and camera, and desk patterns kept out of training), a ball somewhere ahead of the arm, the
real arm's speed limits, and goes through ``understudy.runner.run_pickup`` in servo ticks.

    python scripts/evaluate_student.py --student results/student/<name>/student.pt --output-dir results/student/<name>

Rows:
  teacher        the teacher, told where the ball is, the arm faced using the true position
  student        the student: faces the ball by its own guess, then drives from pictures alone
  eye-and-hand   the student's idea of where the ball is, fed to the teacher, which drives
  blank-0.5s     the student, with the picture blanked for half a second just before the grasp
  blank-1s       the same for a whole second

Needs torch, mujoco, onnxruntime and OpenGL. No GPU.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from understudy import arm, dagger, pickup, runtime, sim_look, student  # noqa: E402
from understudy.runner import REACH_BAND_M  # noqa: E402

# The teacher closes the gripper between steps 52 and 71 of an attempt (measured over 40 attempts).
ROWS = {
    "teacher": dict(drive="teacher", facing="exact", brain=False),
    "student": dict(drive="student", facing="student", teacher=False),
    "eye-and-hand": dict(drive="teacher", facing="student", ball_source="student"),
    "blank-0.5s": dict(drive="student", facing="student", teacher=False, blank=(40, 15)),
    "blank-1s": dict(drive="student", facing="student", teacher=False, blank=(30, 30)),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--student", required=True, help="The trained student (student.pt).")
    parser.add_argument("--teacher", default=str(REPO / "results/pickup/2026-11-24-claw/policy.onnx"))
    parser.add_argument("--attempts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rows", nargs="+", default=list(ROWS), choices=list(ROWS))
    parser.add_argument("--seen_looks", action="store_true", help="Use the desk patterns the student was trained on.")
    parser.add_argument("--output-dir", help="Write one JSON file per row here.")
    args = parser.parse_args()

    torch.set_num_threads(1)
    teacher = runtime.OnnxPolicy(args.teacher)
    net, notes = student.load(args.student)
    brain = student.Brain(net)
    sim, scene = dagger.bench()
    scene.held_out = not args.seen_looks
    for name in args.rows:
        row = dict(ROWS[name])
        use_teacher, use_brain = row.pop("teacher", True), row.pop("brain", True)
        rng = random.Random(args.seed)  # the same balls and looks for every row
        started = time.monotonic()
        outcomes = [
            dagger.run_attempt(sim, scene, rng, teacher=teacher if use_teacher else None, brain=brain if use_brain else None,
                               seconds=pickup.EPISODE_SECONDS * 3, **row)[0]
            for _ in range(args.attempts)
        ]
        facing = [o["facing_error_m"] for o in outcomes if o.get("facing_error_m") is not None]
        seen = [o["seen_error_m"] for o in outcomes if o.get("seen_error_m") is not None]
        result = {
            "row": name,
            "simulator": f"MuJoCo {mujoco.__version__}, vendor model, rendered wrist camera, varied looks",
            "student": str(args.student),
            "student_training": notes,
            "teacher": str(args.teacher),
            "attempts": len(outcomes),
            "seed": args.seed,
            "looks": "desk patterns the student was trained on" if args.seen_looks else
                     f"held out: new draws, and desk patterns {list(sim_look.HELD_OUT_PATTERNS)} never used in training",
            "ball_distance_from_swivel_axis_m": list(REACH_BAND_M),
            "ball_within_rad_of_straight_ahead": dagger.AHEAD_TURN_RAD,
            "seconds": pickup.EPISODE_SECONDS * 3,
            "max_speed_rad_s": student.SPEED_LIMIT_RAD_S,
            "max_lead_rad": student.LEAD_LIMIT_RAD,
            "arm_driven_by": row["drive"],
            "arm_faced_using": row["facing"],
            "ball_position_given_to_the_driver": (
                "none: pictures only" if row["drive"] == "student" else
                "the student's own estimate" if row.get("ball_source") == "student" else "the true position"),
            "picture_blanked_steps": list(row["blank"]) if "blank" in row else None,
            "ball_found_rate": sum(o["found"] for o in outcomes) / len(outcomes),
            "lifted_rate": sum(o["lifted"] for o in outcomes) / len(outcomes),
            "lifted_at_some_point_rate": sum(o["ever_lifted"] for o in outcomes) / len(outcomes),
            "safety_stop_rate": sum(o["stopped"] for o in outcomes) / len(outcomes),
            "facing_guess_error_mm": {"median": 1000 * float(np.median(facing)), "worst": 1000 * max(facing)} if facing else None,
            "ball_seen_error_before_lifting_mm_median": 1000 * float(np.median(seen)) if seen else None,
            "minutes": round((time.monotonic() - started) / 60, 1),
            "control_hz": arm.CONTROL_HZ,
        }
        text = json.dumps(result, indent=2)
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.output_dir) / f"mujoco-{name}.json").write_text(text + "\n")
        print(json.dumps({k: result[k] for k in ("row", "ball_found_rate", "lifted_rate", "lifted_at_some_point_rate",
                                                  "safety_stop_rate", "facing_guess_error_mm", "minutes")}), flush=True)
    scene.close()


if __name__ == "__main__":
    main()
