"""Measure how well a trained reach policy reaches, and export it.

A reach is scored at the end of each target's hold: the mean gripper-tip error
over the final half second must be inside the success radius. This is stricter
than the training log's success rate, which counts an episode as a success if
the tip passes inside the radius at any moment.

Run with Isaac Lab's Python:

    $ISAACLAB_PATH/isaaclab.sh -p scripts/evaluate.py --headless \
        --checkpoint logs/rsl_rl/understudy_reach_omx/<run>/model_999.pt --output results/reach/<name>

Writes ``policy.onnx``, ``policy.pt`` and ``evaluation.json`` to the output directory.
Use ``--stiffness_scale`` and ``--noise`` with ``--result_name`` to record how the
same policy copes with servos that differ from the ones it was trained on.
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
parser.add_argument("--targets", type=int, default=5000, help="Number of scored reaches to collect.")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--noise", action="store_true", help="Keep training-time observation noise on.")
parser.add_argument(
    "--stiffness_scale",
    type=float,
    default=1.0,
    help="Multiply the simulated arm servo stiffness, keeping the damping ratio, to test sensitivity.",
)
parser.add_argument("--result_name", default="evaluation", help="File name (without .json) for the results.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import understudy_isaac  # noqa: E402
from understudy import arm  # noqa: E402
from understudy_isaac.policy_io import load_policy  # noqa: E402

FINAL_WINDOW_STEPS = int(0.5 * arm.CONTROL_HZ)


def main() -> None:
    task = understudy_isaac.REACH_TASK_PLAY
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.observations.policy.enable_corruption = args.noise
    if args.stiffness_scale <= 0:
        raise SystemExit("--stiffness_scale must be positive")
    for name in ("xl430", "xl330"):
        actuator = env_cfg.scene.robot.actuators[name]
        actuator.stiffness = actuator.stiffness * args.stiffness_scale
        actuator.damping = actuator.damping * args.stiffness_scale**0.5
    output = Path(args.output)
    env, policy = load_policy(task, env_cfg, args.checkpoint, output)

    command = env.unwrapped.command_manager.get_term("ee_pose")
    hold_steps = int(round(env_cfg.commands.ee_pose.resampling_time_range[0] * arm.CONTROL_HZ))
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    window = torch.zeros(num_envs, FINAL_WINDOW_STEPS, device=device)
    steps_on_target = torch.zeros(num_envs, dtype=torch.long, device=device)
    previous_target = command.command[:, :3].clone()
    final_errors: list[torch.Tensor] = []
    collected = 0

    obs = env.get_observations()
    step = 0
    while collected < args.targets:
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
            policy.reset(dones)
        # The error reported for this step is measured against the target that was
        # active during it, even when that target was replaced at the end of the step.
        window[:, step % FINAL_WINDOW_STEPS] = command.metrics["position_error"]
        steps_on_target += 1
        step += 1

        target = command.command[:, :3]
        changed = (target - previous_target).abs().amax(dim=1) > 1e-6
        # Score only targets that were held for their full time.
        scored = changed & (steps_on_target >= hold_steps)
        if scored.any():
            errors = window[scored].mean(dim=1)
            final_errors.append(errors.cpu())
            collected += errors.numel()
        steps_on_target[changed] = 0
        previous_target = target.clone()

    errors = torch.cat(final_errors)[: args.targets]
    result = {
        "task": task,
        "checkpoint": str(args.checkpoint),
        "reaches_scored": int(errors.numel()),
        "hold_seconds": hold_steps / arm.CONTROL_HZ,
        "scored_over_final_seconds": FINAL_WINDOW_STEPS / arm.CONTROL_HZ,
        "observation_noise": bool(args.noise),
        "servo_stiffness_scale": args.stiffness_scale,
        "seed": args.seed,
        "success_radius_m": arm.SUCCESS_RADIUS_M,
        "success_rate": float((errors < arm.SUCCESS_RADIUS_M).float().mean()),
        "success_rate_within_1cm": float((errors < 0.01).float().mean()),
        "success_rate_within_5cm": float((errors < 0.05).float().mean()),
        "median_error_m": float(errors.median()),
        "p95_error_m": float(errors.quantile(0.95)),
        "max_error_m": float(errors.max()),
    }
    (output / f"{args.result_name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    main()
    app.close()
