"""Joint position commands as a real DYNAMIXEL servo receives and follows them.

This module imports the simulator, so task configuration must not import it.
The configuration lives in ``servo_action_cfg`` and names this class by string.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

from .servo_action_cfg import ProfiledJointPositionActionCfg

# Age given to the commands an attempt starts with, so they count as long since carried out.
_SETTLED_S = 10.0


class ProfiledJointPositionAction(JointPositionAction):
    """Joint position targets that arrive late and are reached over a short time.

    A command from the policy reaches the servo after a communication delay, and
    a servo with a time-based profile then moves its goal to the new value over
    the profile time. Each simulated arm draws its own delay and profile time at
    the start of every attempt. The delay may be longer than one control step, in
    which case the servo is still acting on an older command when a new one is issued.
    """

    cfg: ProfiledJointPositionActionCfg

    def __init__(self, cfg: ProfiledJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._physics_dt = env.physics_dt
        # Enough past commands to cover the longest delay.
        self._depth = int(math.ceil(cfg.delay_range_s[1] / env.step_dt)) + 2
        default = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        self._applied = default.clone()
        self._start = default.clone()
        self._commands = default.unsqueeze(1).repeat(1, self._depth, 1)  # newest first
        self._age = torch.full((self.num_envs, self._depth), _SETTLED_S, device=self.device)
        self._delay = torch.zeros(self.num_envs, device=self.device)
        self._profile_time = torch.zeros(self.num_envs, device=self.device)
        self._rows = torch.arange(self.num_envs, device=self.device)
        self._sample(slice(None))

    def _sample(self, ids) -> None:
        count = self._delay[ids].shape[0]
        low, high = self.cfg.delay_range_s
        self._delay[ids] = low + (high - low) * torch.rand(count, device=self.device)
        low, high = self.cfg.profile_time_range_s
        self._profile_time[ids] = low + (high - low) * torch.rand(count, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        if self.cfg.relative:
            self._raw_actions[:] = actions
            present = self._asset.data.joint_pos.torch[:, self._joint_ids]
            self._processed_actions = present + self._raw_actions * self._scale
            if self.cfg.clip is not None:
                self._processed_actions = torch.clamp(
                    self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
                )
        else:
            super().process_actions(actions)
        self._commands = torch.roll(self._commands, shifts=1, dims=1)
        self._commands[:, 0] = self.processed_actions
        self._age = torch.roll(self._age, shifts=1, dims=1)
        self._age[:, 0] = 0.0

    def apply_actions(self):
        self._age += self._physics_dt
        arrived = self._age >= self._delay.unsqueeze(1)
        # The newest command that has reached the servo; before any has, the oldest one kept.
        newest = torch.where(arrived.any(dim=1), arrived.float().argmax(dim=1), self._depth - 1)
        goal = self._commands[self._rows, newest]
        since_arrival = self._age[self._rows, newest] - self._delay
        just_arrived = (since_arrival >= 0.0) & (since_arrival < self._physics_dt)
        # A new goal is approached from wherever the servo's goal had got to.
        self._start = torch.where(just_arrived.unsqueeze(1), self._applied, self._start)
        progress = (since_arrival / self._profile_time.clamp(min=1.0e-6)).clamp(0.0, 1.0).unsqueeze(1)
        self._applied = self._start + (goal - self._start) * progress
        self._asset.set_joint_position_target_index(target=self._applied, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        default = self._asset.data.default_joint_pos.torch[:, self._joint_ids]
        self._applied[ids] = default[ids]
        self._start[ids] = default[ids]
        self._commands[ids] = default[ids].unsqueeze(1)
        self._age[ids] = _SETTLED_S
        self._sample(ids)
