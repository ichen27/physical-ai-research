"""Reward terms specific to the ball pickup task."""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

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
