"""Measure the whole job in MuJoCo: find the ball, pick it up, carry it to a bowl and drop it in.

The student, shown only pictures, searches for the ball and picks it up
(``understudy.dagger.run_attempt`` with ``facing="search"``); the carry is the fixed move
of ``understudy.carry``, the same for the teacher's row. A bowl stands at the arm's left
(``understudy.sim_bowl``): the student was never shown one in training, so this is also a
test of whether a bowl in its pictures puts it off.

    python scripts/evaluate_student_carry.py --student results/student/<name>/student.pt --part student --output out.json
    python scripts/evaluate_student_carry.py --student ... --part teacher --output out.json

Balls the bowl is in the way of (``understudy.carry.in_the_way``) are left out, as the real
tool refuses them. Needs torch, mujoco, onnxruntime and OpenGL. No GPU.
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
import torch  # noqa: E402

from understudy import carry, dagger, pickup, runner, runtime, sim_bowl, student  # noqa: E402
from understudy.camera_model import WristCamera  # noqa: E402
from understudy.runner import AIM_LEFT_M, REACH_BAND_M, SWIVEL_AXIS_X_M  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", default="results/pickup/2026-11-24-claw/policy.onnx")
    parser.add_argument("--part", required=True, choices=["student", "teacher"])
    parser.add_argument("--attempts", type=int, default=100)
    parser.add_argument("--bounce", type=float, default=0.15, help="How the ball lands in the bowl: 1 does not bounce, 0.15 comes back up about half way.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", help="Write the results to this JSON file.")
    args = parser.parse_args()

    torch.set_num_threads(1)
    teacher = runtime.OnnxPolicy(str(REPO / args.teacher) if not Path(args.teacher).is_absolute() else args.teacher)
    net, notes = student.load(args.student)
    brain = student.Brain(net)
    nominal = WristCamera(**{key: tuple(value) if isinstance(value, list) else value
                             for key, value in notes["nominal_camera"].items()}) if "nominal_camera" in notes else None
    bowl = sim_bowl.Bowl(contact_damping=args.bounce)
    sim, scene = dagger.bench(nominal, also=lambda spec: sim_bowl.add_bowl(spec, bowl))
    scene.held_out = True
    rng = random.Random(args.seed)
    seconds = pickup.EPISODE_SECONDS * 3
    started = time.monotonic()
    rows, left_out = [], 0
    while len(rows) < args.attempts:
        distance = rng.uniform(*REACH_BAND_M)
        bearing = rng.uniform(-runner.MAX_TURN_RAD, runner.MAX_TURN_RAD) + math.asin(AIM_LEFT_M / distance)
        ball = (SWIVEL_AXIS_X_M + distance * math.cos(bearing), distance * math.sin(bearing))
        place = carry.SetDown(math.radians(rng.uniform(85.0, 90.0)), rng.uniform(0.23, 0.26), bowl.rim_m, bowl.opening_m)
        if carry.in_the_way(place, ball):
            left_out += 1
            continue
        stand = lambda place=place: sim_bowl.place_bowl(sim, place.xy)  # noqa: E731
        if args.part == "teacher":
            outcome, _ = dagger.run_attempt(sim, scene, rng, teacher=teacher, drive="teacher", facing="exact", seconds=seconds,
                                            ball=ball, after_reset=stand)
        else:
            outcome, _ = dagger.run_attempt(sim, scene, rng, brain=brain, drive="student", facing="search", seconds=seconds,
                                            ball=ball, after_reset=stand)
        row = {**outcome, "turn_deg": round(math.degrees(bearing - math.asin(AIM_LEFT_M / distance)), 1), "distance_cm": round(100 * distance, 1),
               "bowl_pushed_in_pickup_mm": round(1000 * math.dist(sim_bowl.bowl_xy(sim), place.xy), 1),
               "carried": False, "said": None, "in_bowl": False, "arm_touched_bowl_in_carry": False}
        if outcome["lifted"]:
            stood = sim_bowl.bowl_xy(sim)
            touched = {"arm": False}

            def watch(arm_sim, touched=touched) -> None:
                touched["arm"] |= sim_bowl.touching(arm_sim)[0]

            sim.after_step = watch
            done = carry.carry(sim, sim.calibration, place, sim.advance, say=lambda _: None)
            for _ in range(60):
                sim.advance()
            sim.after_step = None
            row.update(carried=done.done, said=done.said, in_bowl=sim_bowl.ball_is_in(sim, bowl), arm_touched_bowl_in_carry=touched["arm"],
                       bowl_pushed_in_carry_mm=round(1000 * math.dist(sim_bowl.bowl_xy(sim), stood), 1))
        rows.append(row)

    def share(some: list, of: list) -> float | None:
        return round(len(some) / len(of), 4) if of else None

    lifted = [row for row in rows if row["lifted"]]
    result = {
        "part": args.part,
        "simulator": f"MuJoCo {mujoco.__version__}, vendor model, rendered wrist camera, looks the student was not trained on",
        "student": str(args.student),
        "attempts": len(rows),
        "seed": args.seed,
        "balls": f"{REACH_BAND_M[0] * 100:g} to {REACH_BAND_M[1] * 100:g} cm from the swivel axis, wherever the bowl is not in the way",
        "balls_left_out_because_the_bowl_was_in_the_way": left_out,
        "bowl": {"opening_cm": bowl.opening_m * 100, "rim_cm": bowl.rim_m * 100, "mass_g": bowl.mass_kg * 1000,
                 "stands": "at the arm's left, 85 to 90 degrees round, 23 to 26 cm from the swivel axis",
                 "ball_rebounds_this_share_of_a_5_cm_drop": round(sim_bowl.rebound(bowl), 2), "shown_to_the_student_in_training": False},
        "ball_position_given_to_the_driver": "the true position" if args.part == "teacher" else "none: pictures only",
        "found": share([r for r in rows if r["found"]], rows),
        "refused": share([r for r in rows if r["refused"]], rows),
        "lifted_at_the_end": share(lifted, rows),
        "ended_in_the_bowl": share([r for r in rows if r["in_bowl"]], rows),
        "ended_in_the_bowl_of_those_lifted": share([r for r in lifted if r["in_bowl"]], lifted),
        "lost_on_the_way_of_those_lifted": share([r for r in lifted if (r["said"] or "").startswith("stopped")], lifted),
        "not_carried_of_those_lifted": share([r for r in lifted if (r["said"] or "").startswith("not carried")], lifted),
        "arm_touched_the_bowl_in_the_pickup": share([r for r in rows if r["bowl_pushed_in_pickup_mm"] >= 1.0], rows),
        "arm_touched_the_bowl_in_the_carry": share([r for r in lifted if r["arm_touched_bowl_in_carry"]], lifted),
        "minutes": round((time.monotonic() - started) / 60, 1),
        "attempt_records": rows,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "attempt_records"}, indent=1))
    scene.close()


if __name__ == "__main__":
    main()
