"""Record one pickup attempt from Isaac Lab: raw readings, policy input and policy output.

The recording is a fixture for tests/test_rollout_fixture.py, which checks that
``understudy.runtime`` builds the same policy input from the raw readings, and
that the exported ONNX policy gives the same output, as Isaac Lab did. Camera
error and sensor noise are switched off so the comparison is exact.

    $ISAACLAB_PATH/isaaclab.sh -p scripts/record_rollout.py --headless \
        --checkpoint <model.pt> --output tests/fixtures/pickup_rollout.json
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--seed", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import understudy_isaac  # noqa: E402
from understudy import arm  # noqa: E402
from understudy_isaac.policy_io import load_policy  # noqa: E402


def main() -> None:
    task = understudy_isaac.PICKUP_TASK_PLAY
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=2)
    env_cfg.seed = args.seed
    env_cfg.observations.policy.enable_corruption = False
    camera = env_cfg.observations.policy.object_position.params
    camera.update({"bias": 0.0, "noise": 0.0, "latency_steps": (0, 0), "occlusion_probability": 0.0})

    with tempfile.TemporaryDirectory() as scratch:
        env, policy = load_policy(task, env_cfg, args.checkpoint, scratch)
    base = env.unwrapped
    robot, ball = base.scene["robot"], base.scene["object"]
    names = list(robot.joint_names)
    sensed = [names.index(n) for n in (*arm.ARM_JOINT_NAMES, arm.GRIPPER_JOINT_NAMES[0])]

    def readings():
        return {
            "joint_angles": robot.data.joint_pos.torch[0, sensed].tolist(),
            "joint_velocities": robot.data.joint_vel.torch[0, sensed].tolist(),
            "ball_position": (ball.data.root_pos_w.torch[0, :3] - robot.data.root_pos_w.torch[0]).tolist(),
            "hold_target": base.command_manager.get_command("object_pose")[0, :3].tolist(),
        }

    steps = []
    obs = env.get_observations()
    previous_action = [0.0] * 6
    for _ in range(args.steps):
        with torch.inference_mode():
            action = policy(obs)
        steps.append(
            {
                **readings(),
                "previous_action": previous_action,
                "observation": obs["policy"][0].tolist(),
                "action": action[0].tolist(),
            }
        )
        previous_action = action[0].tolist()
        with torch.inference_mode():
            obs, _, dones, _ = env.step(action)
            policy.reset(dones)
        if bool(dones[0]):
            break

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"task": task, "checkpoint": str(args.checkpoint), "steps": steps}) + "\n")
    print(f"recorded {len(steps)} steps to {args.output}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    import os

    os._exit(0)
