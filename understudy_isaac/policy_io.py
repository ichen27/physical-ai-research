"""Load a trained policy into an environment and export it. Imports Isaac Lab."""

from __future__ import annotations

import importlib.metadata as metadata
from pathlib import Path

import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner


def load_policy(task: str, env_cfg, checkpoint: str, export_dir: str | Path, render_mode: str | None = None):
    """Create the environment, load ``checkpoint`` and write ``policy.pt`` and ``policy.onnx``.

    Returns the wrapped environment and the deterministic policy.
    """
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env = RslRlVecEnvWrapper(gym.make(task, cfg=env_cfg, render_mode=render_mode), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    runner.export_policy_to_jit(path=str(export_dir), filename="policy.pt")
    runner.export_policy_to_onnx(path=str(export_dir), filename="policy.onnx")
    return env, policy
