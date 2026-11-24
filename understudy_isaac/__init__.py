"""Isaac Lab tasks for the OMX arm.

Importing this package only registers the task names. Nothing here imports
Isaac Lab, so it is safe to import before the simulator application starts.
"""

import gymnasium as gym

from . import agents

REACH_TASK = "Understudy-Reach-OMX-v0"
REACH_TASK_PLAY = "Understudy-Reach-OMX-Play-v0"

for _task, _cfg in ((REACH_TASK, "OmxReachEnvCfg"), (REACH_TASK_PLAY, "OmxReachEnvCfg_PLAY")):
    gym.register(
        id=_task,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.reach_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OmxReachPPORunnerCfg",
        },
    )
