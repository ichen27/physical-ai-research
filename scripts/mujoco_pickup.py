"""Run a trained pickup policy in MuJoCo, on the vendor's own model of the arm.

The policy was trained in Isaac Lab (PhysX). This runs it in a different physics
engine, on the vendor's MuJoCo model with its servo gains, joint friction and
gripper coupling, using the same input and output code the real arm will use
(``understudy.runtime``). It needs no GPU.

    pip install -r requirements-dev.txt onnxruntime
    python scripts/mujoco_pickup.py --policy results/pickup/<run>/policy.onnx --attempts 200

A policy that only works in the simulator it was trained in will not work on
hardware, so this is a necessary check. It is not a sufficient one.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from understudy import arm, pickup, runtime, variation  # noqa: E402
from understudy.mujoco_arm import MujocoArm, SimulatedCamera, build_model  # noqa: E402
from understudy.runner import PlacedBall, run_pickup  # noqa: E402

FINAL_WINDOW_STEPS = int(0.5 * arm.CONTROL_HZ)


def run_attempt(sim: MujocoArm, policy, rng: random.Random, args, recorder=None) -> dict:
    ball_xy = (rng.uniform(*pickup.BALL_SPAWN["x"]), rng.uniform(*pickup.BALL_SPAWN["y"]))
    target = [rng.uniform(*pickup.HOLD_TARGET[axis]) for axis in "xyz"]
    sim.reset(ball_xy)
    if args.ball_source == "camera":
        ball = SimulatedCamera(sim, rng, args.camera_bias, args.camera_noise, args.camera_latency)
    else:
        ball = PlacedBall(*ball_xy)

    heights, distances = [], []

    def after_step(arm_sim: MujocoArm) -> None:
        position = arm_sim.ball()
        heights.append(float(position[2]))
        distances.append(float(np.linalg.norm(position - np.array(target))))
        if recorder is not None:
            recorder.capture(arm_sim)

    sim.after_step = after_step
    run_pickup(sim, policy, sim.calibration, ball, target, pickup.EPISODE_SECONDS, advance=sim.advance)

    height = float(np.mean(heights[-FINAL_WINDOW_STEPS:]))
    distance = float(np.mean(distances[-FINAL_WINDOW_STEPS:]))
    lifted = height > pickup.LIFTED_HEIGHT_M
    return {"lifted": lifted, "held": lifted and distance < pickup.HOLD_RADIUS_M, "height": height, "distance": distance}


class Recorder:
    """Writes the attempts to an MP4 through ffmpeg."""

    def __init__(self, model: mujoco.MjModel, path: Path, width: int = 960, height: int = 720):
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._camera = mujoco.MjvCamera()
        self._camera.lookat[:] = [0.14, 0.0, 0.08]
        self._camera.distance, self._camera.azimuth, self._camera.elevation = 0.75, 135.0, -25.0
        self._ffmpeg = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", str(int(arm.CONTROL_HZ)), "-i", "-", "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE,
        )

    def capture(self, sim: MujocoArm) -> None:
        self._renderer.update_scene(sim.data, camera=self._camera)
        self._ffmpeg.stdin.write(self._renderer.render().tobytes())

    def close(self) -> None:
        self._ffmpeg.stdin.close()
        self._ffmpeg.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", required=True, help="Exported policy (policy.onnx).")
    parser.add_argument("--attempts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stiffness_scale", type=float, default=1.0, help="Multiple of the vendor servo stiffness.")
    parser.add_argument("--ball_mass", type=float, default=pickup.BALL_MASS_KG)
    parser.add_argument("--ball_friction", type=float, default=pickup.BALL_FRICTION)
    parser.add_argument("--camera_bias", type=float, default=variation.BALL_POSITION_BIAS_M)
    parser.add_argument("--camera_noise", type=float, default=variation.BALL_POSITION_NOISE_M)
    parser.add_argument("--camera_latency", type=int, nargs=2, default=list(variation.CAMERA_LATENCY_STEPS))
    parser.add_argument("--command_delay", type=int, default=1, help="Command delay in control steps.")
    parser.add_argument(
        "--ball_source",
        choices=["camera", "placed"],
        default="camera",
        help="camera: a simulated camera tracks the ball. placed: no camera; the ball is assumed to be where it "
        "was put until the gripper holds it (understudy.runner.PlacedBall).",
    )
    parser.add_argument("--output", help="Write the results to this JSON file.")
    parser.add_argument("--video", help="Record the first attempts to this MP4 file (needs ffmpeg).")
    parser.add_argument("--video_attempts", type=int, default=3)
    args = parser.parse_args()
    policy = runtime.OnnxPolicy(args.policy)
    sim = MujocoArm(
        build_model(args.ball_mass, args.ball_friction, args.stiffness_scale), command_delay_steps=args.command_delay
    )
    rng = random.Random(args.seed)
    recorder = Recorder(sim.model, Path(args.video)) if args.video else None

    outcomes = []
    for attempt in range(args.attempts):
        recording = recorder if recorder is not None and attempt < args.video_attempts else None
        outcomes.append(run_attempt(sim, policy, rng, args, recording))
    if recorder is not None:
        recorder.close()

    lifted = [o for o in outcomes if o["lifted"]]
    result = {
        "simulator": f"MuJoCo {mujoco.__version__}, vendor model",
        "policy": str(args.policy),
        "attempts_scored": len(outcomes),
        "servo_stiffness_scale": args.stiffness_scale,
        "ball_mass_kg": args.ball_mass,
        "ball_friction": args.ball_friction,
        "camera_bias_m": args.camera_bias,
        "camera_noise_m": args.camera_noise,
        "camera_latency_steps": list(args.camera_latency),
        "command_delay_steps": args.command_delay,
        "ball_source": args.ball_source,
        "seed": args.seed,
        "success_rate": sum(o["held"] for o in outcomes) / len(outcomes),
        "lifted_rate": len(lifted) / len(outcomes),
        "median_final_ball_height_m": float(np.median([o["height"] for o in outcomes])),
        "median_distance_to_target_when_lifted_m": float(np.median([o["distance"] for o in lifted])) if lifted else None,
    }
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
