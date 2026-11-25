"""Check that the simulated gripper can physically pick up the ball.

Runs a fixed, hand-written sequence with no learning: open the gripper, move
the grasp point onto the ball, close, lift. If this cannot lift the ball, the
scene is wrong and training would be wasted.

    $ISAACLAB_PATH/isaaclab.sh -p scripts/check_pickup_scene.py --headless

The copies differ in servo gains, delays and ball properties as in training, and
the sequence is not tuned to cope with that, so a miss or two is expected. Exits
non-zero unless the ball ends more than 5 cm above the desk in at least six of
the eight copies.
"""

import argparse
import math
import random
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
from understudy import arm, pickup  # noqa: E402
from understudy.kinematics import Chain, distance  # noqa: E402

NUM_ENVS = 8
OPEN, CLOSE = 1.0, -1.0


def solve(chain: Chain, target: tuple[float, float, float], lift: float = 0.0) -> list[float]:
    """Joint angles that put the grasp point on ``target`` (raised by ``lift``), gripper pointing down."""
    goal = (target[0], target[1], target[2] + lift)
    pan = math.atan2(goal[1], goal[0])
    rng = random.Random(0)
    best, best_cost = None, float("inf")
    for trial in range(6000):
        if best is None or trial < 2000:
            q = [pan, rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), rng.uniform(-1.7, 1.7), 0.0]
        else:
            q = [pan] + [v + rng.gauss(0.0, 0.03) for v in best[1:4]] + [0.0]
        q = arm.clamp_to_limits(q)
        pitch = q[1] + q[2] + q[3]
        cost = distance(chain.point_in_tip_frame(q, pickup.GRASP_POINT_FROM_TIP), goal)
        cost += 0.05 * abs(pitch - math.pi / 2)  # gripper pointing straight down
        if cost < best_cost:
            best, best_cost = q, cost
    return best


def main() -> int:
    task = understudy_isaac.PICKUP_TASK_PLAY
    cfg = parse_env_cfg(task, device=args.device, num_envs=NUM_ENVS)
    # The fixed sequence is slower than one training attempt; do not reset in the middle of it.
    cfg.episode_length_s = 60.0
    cfg.commands.object_pose.resampling_time_range = (60.0, 60.0)
    env = gym.make(task, cfg=cfg).unwrapped
    env.reset()
    robot, ball = env.scene["robot"], env.scene["object"]
    joint_names = list(robot.joint_names)
    arm_ids = [joint_names.index(n) for n in arm.ARM_JOINT_NAMES]
    finger_id = joint_names.index(arm.GRIPPER_JOINT_NAMES[0])
    start = torch.tensor(pickup.START_POSE, device=env.device)
    scale = cfg.actions.arm_action.scale
    relative = getattr(cfg.actions.arm_action, "relative", False)
    chain = Chain()

    def ball_positions():
        return (ball.data.root_pos_w.torch[:, :3] - robot.data.root_pos_w.torch).tolist()

    def grasp_points():
        reached = robot.data.joint_pos.torch[:, arm_ids].tolist()
        return [chain.point_in_tip_frame(q, pickup.GRASP_POINT_FROM_TIP) for q in reached]

    def run(poses, gripper, steps, correct=False):
        """Hold ``poses``. With ``correct``, push the command past the pose to cancel servo sag."""
        wanted = torch.tensor(poses, device=env.device)
        command = wanted.clone()
        action = torch.zeros(NUM_ENVS, 6, device=env.device)
        action[:, 5] = gripper
        for step in range(steps):
            if correct and step % 10 == 9:
                command += 0.5 * (wanted - robot.data.joint_pos.torch[:, arm_ids])
            if relative:
                action[:, :5] = (command - robot.data.joint_pos.torch[:, arm_ids]) / scale
            else:
                action[:, :5] = (command - start) / scale
            env.step(action)

    def report(label):
        balls, points = ball_positions(), grasp_points()
        fingers = robot.data.joint_pos.torch[:, finger_id].tolist()
        print(label)
        for i in range(NUM_ENVS):
            gap = distance(points[i], tuple(balls[i]))
            print(
                f"  copy {i}: ball {[round(v, 3) for v in balls[i]]} grasp point {[round(v, 3) for v in points[i]]}"
                f" apart {gap * 1000:5.1f} mm, finger {fingers[i]:.3f} rad"
            )

    run([list(pickup.START_POSE)] * NUM_ENVS, OPEN, 30)
    balls = ball_positions()
    print("ball rest height (m):", round(balls[0][2], 4), "expected", pickup.BALL_RADIUS_M)

    above = [solve(chain, tuple(b), lift=0.06) for b in balls]
    at = [solve(chain, tuple(b)) for b in balls]
    up = [solve(chain, tuple(b), lift=0.08) for b in balls]
    print("pose at the ball, copy 0:", [round(v, 3) for v in at[0]], "pitch", round(sum(at[0][1:4]), 3))
    run(above, OPEN, 60, correct=True)
    report("above the ball, gripper open")
    run(at, OPEN, 90, correct=True)
    report("at the ball, gripper open")
    run(at, CLOSE, 45, correct=True)
    report("closed on the ball")
    run(up, CLOSE, 30, correct=True)
    report("one second into the lift")
    run(up, CLOSE, 90, correct=True)
    report("four seconds into the lift")
    held = [b[2] for b in ball_positions()]

    lifted = sum(h > pickup.LIFTED_HEIGHT_M for h in held)
    print(f"scripted grasp lifted the ball in {lifted} of {NUM_ENVS} copies")
    env.close()
    return 0 if lifted >= NUM_ENVS - 2 else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # Closing the simulator ends the process with its own status, so exit first.
    import os

    os._exit(code)
