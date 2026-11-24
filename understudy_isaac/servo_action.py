"""Joint position commands as a real DYNAMIXEL servo receives and follows them.

This module imports the simulator, so task configuration must not import it.
The configuration lives in ``servo_action_cfg`` and names this class by string.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

from .servo_action_cfg import ProfiledJointPositionActionCfg


class ProfiledJointPositionAction(JointPositionAction):
    """Joint position targets that arrive late and are reached over a short time.

    A command from the policy reaches the servo after a communication delay, and
    a servo with a time-based profile then moves its goal to the new value over
    the profile time. Each simulated arm draws its own delay and profile time at
    the start of every attempt.
    """

    cfg: ProfiledJointPositionActionCfg

    def __init__(self, cfg: ProfiledJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._physics_dt = env.physics_dt
        self._applied = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        self._start = self._applied.clone()
        self._elapsed = torch.zeros(self.num_envs, device=self.device)
        self._delay = torch.zeros(self.num_envs, device=self.device)
        self._profile_time = torch.zeros(self.num_envs, device=self.device)
        self._sample(slice(None))

    def _sample(self, ids) -> None:
        count = self._delay[ids].shape[0]
        low, high = self.cfg.delay_range_s
        self._delay[ids] = low + (high - low) * torch.rand(count, device=self.device)
        low, high = self.cfg.profile_time_range_s
        self._profile_time[ids] = low + (high - low) * torch.rand(count, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._start = self._applied.clone()
        self._elapsed[:] = 0.0

    def apply_actions(self):
        self._elapsed += self._physics_dt
        progress = (self._elapsed - self._delay) / self._profile_time.clamp(min=1.0e-6)
        progress = progress.clamp(0.0, 1.0).unsqueeze(1)
        self._applied = self._start + (self.processed_actions - self._start) * progress
        self._asset.set_joint_position_target_index(target=self._applied, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        default = self._asset.data.default_joint_pos.torch[:, self._joint_ids]
        self._applied[ids] = default[ids]
        self._start[ids] = default[ids]
        self._sample(ids)
