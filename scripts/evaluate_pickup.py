"""Measure how often a trained policy picks up the ball, and export it.

By default every simulated arm, ball and camera differs as in training
(see understudy/variation.py). The options pin one property to a single value
to show how the policy copes with it.

An attempt is scored over its final half second. It is a success only if,
averaged over that time, the ball is off the desk and within the hold radius of
the target. Touching the target briefly and dropping the ball does not count.

    $ISAACLAB_PATH/isaaclab.sh -p scripts/evaluate_pickup.py --headless \
        --checkpoint logs/rsl_rl/understudy_pickup_omx/<run>/model_1499.pt --output results/pickup/<name>

Writes ``policy.onnx``, ``policy.pt`` and ``evaluation.json`` to the output directory.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--checkpoint", required=True, help="Trained model checkpoint (.pt).")
parser.add_argument("--output", required=True, help="Directory for the exported policy and results.")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--attempts", type=int, default=5000, help="Number of scored pickup attempts.")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument(
    "--stiffness_scale",
    type=float,
    default=None,
    help="Give every arm this multiple of the vendor-model servo stiffness instead of a varied one.",
)
parser.add_argument("--ball_bias", type=float, default=None, help="Camera calibration error in metres, each axis.")
parser.add_argument("--ball_friction", type=float, default=None, help="Give every ball this friction coefficient.")
parser.add_argument("--ball_mass", type=float, default=None, help="Give every ball this mass in kilograms.")
parser.add_argument("--result_name", default="evaluation", help="File name (without .json) for the results.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import understudy_isaac  # noqa: E402
from understudy import arm, pickup, variation  # noqa: E402
from understudy_isaac.policy_io import load_policy  # noqa: E402

FINAL_WINDOW_STEPS = int(0.5 * arm.CONTROL_HZ)


def main() -> None:
    task = understudy_isaac.PICKUP_TASK_PLAY
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.observations.policy.enable_corruption = True
    if args.stiffness_scale is not None:
        if args.stiffness_scale <= 0:
            raise SystemExit("--stiffness_scale must be positive")
        env_cfg.events.servo_gains = None
        servos = env_cfg.scene.robot.actuators["arm"]
        servos.stiffness = {name: value * args.stiffness_scale for name, value in servos.stiffness.items()}
        servos.damping = {name: value * args.stiffness_scale**0.5 for name, value in servos.damping.items()}
    if args.ball_bias is not None:
        env_cfg.observations.policy.object_position.params["bias"] = args.ball_bias
    if args.ball_friction is not None:
        env_cfg.events.ball_friction = None
        material = env_cfg.scene.object.spawn.physics_material
        material.static_friction = material.dynamic_friction = args.ball_friction
    if args.ball_mass is not None:
        env_cfg.events.ball_mass = None
        env_cfg.scene.object.spawn.mass_props.mass = args.ball_mass

    output = Path(args.output)
    env, policy = load_policy(task, env_cfg, args.checkpoint, output)
    scene = env.unwrapped.scene
    robot, ball = scene["robot"], scene["object"]
    command = env.unwrapped.command_manager.get_term("object_pose")
    num_envs, device = env.unwrapped.num_envs, env.unwrapped.device

    heights = torch.zeros(num_envs, FINAL_WINDOW_STEPS, device=device)
    distances = torch.zeros(num_envs, FINAL_WINDOW_STEPS, device=device)
    final_heights, final_distances = [], []
    collected, step = 0, 0

    obs = env.get_observations()
    while collected < args.attempts:
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
            policy.reset(dones)
        # An attempt that just ended has already been reset, so score it from the
        # readings stored before this step.
        ended = dones.bool()
        if ended.any():
            final_heights.append(heights[ended].mean(dim=1).cpu())
            final_distances.append(distances[ended].mean(dim=1).cpu())
            collected += int(ended.sum())
        ball_position = ball.data.root_pos_w.torch[:, :3] - robot.data.root_pos_w.torch
        heights[:, step % FINAL_WINDOW_STEPS] = ball_position[:, 2]
        distances[:, step % FINAL_WINDOW_STEPS] = torch.linalg.norm(ball_position - command.command[:, :3], dim=1)
        step += 1

    height = torch.cat(final_heights)[: args.attempts]
    distance = torch.cat(final_distances)[: args.attempts]
    lifted = height > pickup.LIFTED_HEIGHT_M
    held = lifted & (distance < pickup.HOLD_RADIUS_M)
    result = {
        "task": task,
        "checkpoint": str(args.checkpoint),
        "attempts_scored": int(height.numel()),
        "attempt_seconds": pickup.EPISODE_SECONDS,
        "scored_over_final_seconds": FINAL_WINDOW_STEPS / arm.CONTROL_HZ,
        "servo_stiffness_scale": args.stiffness_scale if args.stiffness_scale is not None else "varied",
        "camera_bias_m": args.ball_bias if args.ball_bias is not None else variation.BALL_POSITION_BIAS_M,
        "ball_friction": args.ball_friction if args.ball_friction is not None else "varied",
        "ball_mass_kg": args.ball_mass if args.ball_mass is not None else "varied",
        "seed": args.seed,
        "lifted_height_m": pickup.LIFTED_HEIGHT_M,
        "hold_radius_m": pickup.HOLD_RADIUS_M,
        "success_rate": float(held.float().mean()),
        "lifted_rate": float(lifted.float().mean()),
        "median_final_ball_height_m": float(height.median()),
        "median_distance_to_target_when_lifted_m": float(distance[lifted].median()) if lifted.any() else None,
    }
    (output / f"{args.result_name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    main()
    app.close()
