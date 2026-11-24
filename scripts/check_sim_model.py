"""Check the simulated arm against the shared arm description.

Holds the simulated arm at a few poses and reports:

* body masses and total mass,
* joint order and limits as the simulator sees them,
* how far each joint settles from its commanded angle (servo sag),
* whether the simulator's gripper-tip position agrees with
  ``understudy.kinematics`` for the angles the arm actually reached.

Run with Isaac Lab's Python:

    $ISAACLAB_PATH/isaaclab.sh -p scripts/check_sim_model.py --headless

Exits non-zero if the two kinematic models disagree by more than 2 mm.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import understudy_isaac  # noqa: E402
from understudy import arm  # noqa: E402
from understudy.kinematics import Chain, distance  # noqa: E402

KINEMATICS_TOLERANCE_M = 0.002
SETTLE_STEPS = 90  # three seconds at the policy rate

POSES = [
    [j.default for j in arm.ARM_JOINTS],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.8, -0.6, 0.9, 0.5, 0.3],
    [-0.8, 0.4, 0.3, 1.2, -0.5],
]


def main() -> int:
    task = understudy_isaac.REACH_TASK_PLAY
    cfg = parse_env_cfg(task, device=args.device, num_envs=len(POSES))
    env = gym.make(task, cfg=cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]

    joint_names = list(robot.joint_names)
    body_names = list(robot.body_names)
    arm_ids = [joint_names.index(name) for name in arm.ARM_JOINT_NAMES]
    tip_id = body_names.index(arm.END_EFFECTOR_LINK)

    masses = robot.data.body_mass.torch[0].tolist()
    print("bodies and masses (kg):")
    for name, mass in zip(body_names, masses):
        print(f"  {name:20s} {mass:.5f}")
    print(f"  total                {sum(masses):.5f}")

    limits = robot.data.joint_pos_limits.torch[0]
    print("joints (simulator order) and limits (rad):")
    for index, name in enumerate(joint_names):
        print(f"  {name:18s} {limits[index, 0].item():+.3f} .. {limits[index, 1].item():+.3f}")

    term = cfg.actions.arm_action
    defaults = torch.tensor([j.default for j in arm.ARM_JOINTS], device=env.device)
    commanded = torch.tensor(POSES, device=env.device)
    action = (commanded - defaults) / term.scale
    for _ in range(SETTLE_STEPS):
        env.step(action)

    reached = robot.data.joint_pos.torch[:, arm_ids]
    speed = robot.data.joint_vel.torch[:, arm_ids].abs().max().item()
    tip = robot.data.body_pos_w.torch[:, tip_id] - robot.data.root_pos_w.torch
    chain = Chain()

    worst = 0.0
    print(f"poses after {SETTLE_STEPS} control steps (largest joint speed {speed:.4f} rad/s):")
    for row in range(len(POSES)):
        angles = reached[row].tolist()
        sag = [a - c for a, c in zip(angles, POSES[row])]
        sim_tip = tuple(tip[row].tolist())
        model_tip = chain.tip_position(angles)
        error = distance(sim_tip, model_tip)
        worst = max(worst, error)
        print(f"  commanded {[round(v, 3) for v in POSES[row]]}")
        print(f"    reached   {[round(v, 3) for v in angles]}")
        print(f"    sag (rad) {[round(v, 3) for v in sag]}")
        print(f"    tip: simulator {[round(v, 4) for v in sim_tip]}, model {[round(v, 4) for v in model_tip]}, gap {error * 1000:.2f} mm")

    ok = worst <= KINEMATICS_TOLERANCE_M
    print(f"kinematics {'agree' if ok else 'DISAGREE'}: worst gap {worst * 1000:.2f} mm (limit {KINEMATICS_TOLERANCE_M * 1000:.0f} mm)")
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    app.close()
    sys.exit(code)
