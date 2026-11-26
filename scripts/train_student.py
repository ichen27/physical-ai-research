"""Train the picture-driven student to copy the pickup teacher, in MuJoCo, on one computer.

Rounds of: gather attempts (the teacher drives the first round, the student a growing
share after that, the teacher labelling every step), then retrain the student on
everything gathered so far. See ``understudy/dagger.py`` and ``understudy/student.py``.

    pip install -r requirements-dev.txt torch
    python scripts/train_student.py --data /somewhere/with/15GB --out results/student/<name>

No GPU is needed; an Apple GPU is used if there is one. It is written for a small
machine: a few worker processes, pictures kept on disk, a fixed batch shape, a cap
on GPU memory, and it stops itself if the computer runs short of memory.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Before torch is imported: keep the Apple GPU's share of the computer's memory modest.
# The low mark must sit under the high one, or torch refuses to start on the GPU.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.4")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.3")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from understudy import dagger, sim_look, student  # noqa: E402
from understudy.camera_model import WristCamera  # noqa: E402

# The share of attempts the student drives in each round after the first, and the share of
# those in which the arm is turned to face the ball by the student's own guess.
STUDENT_DRIVES = (0.5, 0.7, 0.8, 0.9, 1.0)
STUDENT_FACES = 0.7


def memory_free_percent() -> float | None:
    """macOS's own figure for free memory, or None where there is no such tool."""
    try:
        text = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=20).stdout
        return float(text.rsplit("free percentage:", 1)[1].strip().rstrip("%"))
    except (OSError, IndexError, ValueError, subprocess.TimeoutExpired):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teacher", default=str(REPO / "results/pickup/2026-11-24-claw/policy.onnx"))
    parser.add_argument("--camera", help="A wrist camera file (from real_arm.py camera-fit) to take as the nominal camera. "
                        "Left out, the built-in rough fit is used (understudy.sim_look.student_camera).")
    parser.add_argument("--data", required=True, help="Folder for the gathered pictures (about 30 KB a step; not kept in Git).")
    parser.add_argument("--out", required=True, help="Folder for the student and the training record.")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--first", type=int, default=160, help="Attempts in the first round, all driven by the teacher.")
    parser.add_argument("--each", type=int, default=120, help="Attempts in each later round.")
    parser.add_argument("--views", type=int, default=40000, help="Single pictures from poses all round the start pose.")
    parser.add_argument("--steps", type=int, default=1500, help="Training steps after each round.")
    parser.add_argument("--windows", type=int, default=16, help="Runs of consecutive steps in each training batch.")
    parser.add_argument("--length", type=int, default=48, help="Steps in each run.")
    parser.add_argument("--view_batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--input", choices=list(student.INPUTS), default="colour",
                        help="What the network is shown of each picture: the colour picture, only which pixels are the "
                        "ball's colour (a fixed filter), or that beside the picture in grey.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--min_free_percent", type=float, default=12.0, help="Stop if the computer's free memory falls below this.")
    args = parser.parse_args()

    out, data = Path(args.out), Path(args.data)
    out.mkdir(parents=True, exist_ok=True)
    spool = data / "spool"
    spool.mkdir(parents=True, exist_ok=True)
    for stale in spool.glob("*.npz"):
        stale.unlink()
    steps_each = int(dagger.TRAINING_SECONDS * 30)
    capacity = args.views + (args.first + args.each * (args.rounds - 1)) * steps_each + 1000
    experience = dagger.Experience(data, capacity)
    nominal = WristCamera.load(args.camera) if args.camera else sim_look.student_camera()
    camera_notes = {"nominal_camera": asdict(nominal), "nominal_camera_from": args.camera or "the built-in rough fit",
                    "speed_limit_rad_s": student.SPEED_LIMIT_RAD_S, "lead_limit_rad": student.LEAD_LIMIT_RAD,
                    "look_pose": list(dagger.LOOK_POSE)}
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    net = student.StudentNet(input=args.input).to(device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sampler = np.random.default_rng(args.seed)
    record = out / "training.jsonl"
    record.write_text("")
    student_path = None
    started = time.monotonic()

    def note(entry: dict) -> None:
        entry["minutes"] = round((time.monotonic() - started) / 60, 1)
        with record.open("a") as file:
            file.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    def take_in() -> None:
        for path in sorted(spool.glob("*.npz")):
            with np.load(path) as saved:
                arrays = {name: saved[name] for name in saved.files}
            added = experience.add_views(arrays) if path.name.startswith("views") else experience.add_attempt(arrays)
            if not added and len(arrays["pictures"]) >= dagger.BURN_IN + 8:
                print(f"no room left for {path.name}; raise the capacity", flush=True)
            path.unlink()

    context = multiprocessing.get_context("spawn")
    with context.Pool(args.workers) as pool:
        for round_number in range(args.rounds):
            free = memory_free_percent()
            if free is not None and free < args.min_free_percent:
                note({"stopped": f"free memory is {free:.0f}%, under {args.min_free_percent:.0f}%", "round": round_number})
                break
            gathering = time.monotonic()
            attempts = args.first if round_number == 0 else args.each
            drives = 0.0 if round_number == 0 else STUDENT_DRIVES[min(round_number - 1, len(STUDENT_DRIVES) - 1)]
            share = -(-attempts // (args.workers * 4))  # small jobs, so no one process holds much
            jobs = [
                {"seed": args.seed * 100000 + round_number * 1000 + index, "spool": str(spool), "teacher": args.teacher,
                 "camera": args.camera, "student": student_path, "attempts": min(share, attempts - index * share), "student_drives": drives,
                 "student_faces": STUDENT_FACES}
                for index in range(-(-attempts // share)) if attempts - index * share > 0
            ]
            if round_number == 0:
                per = 2000
                jobs += [{"seed": args.seed * 100000 + 900000 + index, "spool": str(spool), "teacher": args.teacher,
                          "camera": args.camera, "views": per} for index in range(args.views // per)]
            outcomes = []
            for result in pool.imap_unordered(dagger.gather, jobs):
                outcomes += result
                take_in()
            take_in()
            driven = [o for o in outcomes if o["driver"] == "student"]
            taught = [o for o in outcomes if o["driver"] == "teacher"]
            faced = [o["facing_error_m"] for o in outcomes if o.get("facing_error_m") is not None]
            note({
                "round": round_number, "gathered_attempts": len(outcomes), "steps_held": experience.size,
                "gather_seconds": round(time.monotonic() - gathering),
                "teacher_drove": len(taught), "teacher_lifted": sum(o["lifted"] for o in taught),
                "student_drove": len(driven), "student_lifted": sum(o["lifted"] for o in driven),
                "student_did_not_find_ball": sum(not o["found"] for o in driven),
                "student_facing_error_mm_median": round(1000 * float(np.median(faced)), 1) if faced else None,
            })

            training = time.monotonic()
            net.train()
            for group in optimiser.param_groups:
                group["lr"] = 1e-3 if round_number < 2 else 5e-4
            sums: dict = {}
            for step in range(args.steps):
                batch = experience.batch(sampler, args.windows, args.length, args.view_batch)
                parts = dagger.losses(net, batch, device)
                optimiser.zero_grad(set_to_none=True)
                parts["total"].backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                optimiser.step()
                for name, value in parts.items():
                    sums[name] = sums.get(name, 0.0) + float(value.detach())
                if step % 200 == 199:
                    free = memory_free_percent()
                    if free is not None and free < args.min_free_percent:
                        note({"stopped": f"free memory is {free:.0f}%, under {args.min_free_percent:.0f}%", "round": round_number})
                        return
            seconds = time.monotonic() - training
            student_path = str(out / f"student-round{round_number}.pt")
            student.save(net.to("cpu"), student_path, round=round_number, steps_held=experience.size, seed=args.seed,
                         **camera_notes)
            net.to(device)
            note({"round": round_number, "trained_steps": args.steps, "train_seconds": round(seconds),
                  "pictures_a_second": round(args.steps * (args.windows * args.length + args.view_batch) / seconds),
                  "loss": {name: round(value / args.steps, 4) for name, value in sums.items()}})
    if student_path is not None:
        final = out / "student.pt"
        final.write_bytes(Path(student_path).read_bytes())
        note({"finished": str(final)})


if __name__ == "__main__":
    main()
