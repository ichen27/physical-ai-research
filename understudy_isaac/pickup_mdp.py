"""Observation and reward terms specific to the ball pickup task."""

from __future__ import annotations

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

from understudy import pickup


def grasp_point_w(env, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """World position of the point between the fingers where the ball should sit."""
    robot = env.scene[robot_cfg.name]
    body = robot_cfg.body_ids[0]
    position = robot.data.body_pos_w.torch[:, body]
    orientation = robot.data.body_quat_w.torch[:, body]
    offset = torch.tensor(pickup.GRASP_POINT_FROM_TIP, device=position.device).expand(position.shape[0], 3)
    return position + quat_apply(orientation, offset)


def grasp_point_object_distance(
    env, std: float, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward for bringing the grasp point to the ball: 1 when touching, falling to 0 far away."""
    ball = env.scene[object_cfg.name].data.root_pos_w.torch[:, :3]
    return 1.0 - torch.tanh(torch.linalg.norm(ball - grasp_point_w(env, robot_cfg), dim=1) / std)


def closing_on_ball(
    env,
    near: float,
    tip_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for closing the gripper while the ball is between the fingers.

    The fingers cannot be both closed and within ``near`` of the ball's centre
    unless the ball is held between them, so this cannot be earned by closing on air.
    """
    robot = env.scene[tip_cfg.name]
    ball = env.scene[object_cfg.name].data.root_pos_w.torch[:, :3]
    is_near = torch.linalg.norm(ball - grasp_point_w(env, tip_cfg), dim=1) < near
    finger = robot.data.joint_pos.torch[:, finger_cfg.joint_ids[0]]
    closure = (1.0 - finger / pickup.GRIPPER_OPEN_RAD).clamp(0.0, 1.0)
    return is_near.float() * closure


def ball_height(env, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Reward that rises from 0 with the ball on the desk to 1 at the lifted height."""
    height = env.scene[object_cfg.name].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    span = pickup.LIFTED_HEIGHT_M - pickup.BALL_RADIUS_M
    return ((height - pickup.BALL_RADIUS_M) / span).clamp(0.0, 1.0)


def ball_position_b(env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), object_cfg: SceneEntityCfg = SceneEntityCfg("object")):
    """True ball position in the arm base frame."""
    robot = env.scene[robot_cfg.name]
    ball = env.scene[object_cfg.name].data.root_pos_w.torch[:, :3]
    position, _ = subtract_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, ball)
    return position


class seen_ball_position(ManagerTermBase):
    """Ball position in the arm base frame as a camera would report it.

    The report is late by a few control steps, carries a fixed error for the
    whole attempt (calibration) and a fresh error every frame, and sometimes
    freezes at its last value while the gripper is close enough to hide the ball.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        params = cfg.params
        self._bias_range = float(params["bias"])
        self._latency_range = tuple(int(v) for v in params["latency_steps"])
        count, device = env.num_envs, env.device
        self._bias = torch.zeros(count, 3, device=device)
        self._latency = torch.zeros(count, dtype=torch.long, device=device)
        self._history = torch.zeros(count, self._latency_range[1] + 1, 3, device=device)
        self._last_report = torch.zeros(count, 3, device=device)
        self._new_attempt = torch.ones(count, dtype=torch.bool, device=device)
        self._rows = torch.arange(count, device=device)
        self.reset()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        count = self._bias[ids].shape[0]
        device = self._bias.device
        self._bias[ids] = (2.0 * torch.rand(count, 3, device=device) - 1.0) * self._bias_range
        low, high = self._latency_range
        self._latency[ids] = torch.randint(low, high + 1, (count,), device=device)
        self._new_attempt[ids] = True

    def __call__(
        self,
        env,
        bias: float,
        noise: float,
        latency_steps: tuple[int, int],
        occlusion_distance: float,
        occlusion_probability: float,
        tip_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        true_position = ball_position_b(env)
        new = self._new_attempt
        self._history[new] = true_position[new].unsqueeze(1)
        self._history = torch.roll(self._history, shifts=1, dims=1)
        self._history[:, 0] = true_position
        report = self._history[self._rows, self._latency] + self._bias
        report = report + (2.0 * torch.rand_like(report) - 1.0) * noise

        ball_w = env.scene["object"].data.root_pos_w.torch[:, :3]
        hidden = torch.linalg.norm(ball_w - grasp_point_w(env, tip_cfg), dim=1) < occlusion_distance
        hidden &= torch.rand(report.shape[0], device=report.device) < occlusion_probability
        hidden &= ~new
        report = torch.where(hidden.unsqueeze(1), self._last_report, report)

        self._last_report = report.clone()
        self._new_attempt[:] = False
        return report


def randomize_gripper_force(env, env_ids, scale_range: tuple[float, float], asset_cfg: SceneEntityCfg):
    """Give each arm's gripper its own squeezing force, as a multiple of the configured limit."""
    robot = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    joint_ids = asset_cfg.joint_ids
    limits = robot.data.joint_effort_limits.torch[env_ids][:, joint_ids].clone()
    low, high = scale_range
    scale = low + (high - low) * torch.rand(len(env_ids), 1, device=limits.device)
    robot.write_joint_effort_limit_to_sim(limits * scale, joint_ids=joint_ids, env_ids=env_ids)


def arm_target_lead(env) -> torch.Tensor:
    """How far each arm joint's commanded target is ahead of its present angle.

    On a real arm this is the last goal sent minus the encoder reading. It is
    proportional to the torque the servo is applying, so it tells the policy about
    load and contact without a force sensor.
    """
    return env.action_manager.get_term("arm_action").target_lead
