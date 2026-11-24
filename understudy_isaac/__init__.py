"""Isaac Lab tasks for the OMX arm.

Importing this package only registers the task names. Nothing here imports
Isaac Lab, so it is safe to import before the simulator application starts.
"""

import gymnasium as gym

from . import agents

REACH_TASK = "Understudy-Reach-OMX-v0"
REACH_TASK_PLAY = "Understudy-Reach-OMX-Play-v0"
PICKUP_TASK = "Understudy-Pickup-Ball-OMX-v0"
PICKUP_TASK_PLAY = "Understudy-Pickup-Ball-OMX-Play-v0"

_TASKS = (
    (REACH_TASK, "reach_env_cfg:OmxReachEnvCfg", "rsl_rl_ppo_cfg:OmxReachPPORunnerCfg"),
    (REACH_TASK_PLAY, "reach_env_cfg:OmxReachEnvCfg_PLAY", "rsl_rl_ppo_cfg:OmxReachPPORunnerCfg"),
    (PICKUP_TASK, "pickup_env_cfg:OmxBallPickupEnvCfg", "pickup_rsl_rl_ppo_cfg:OmxBallPickupPPORunnerCfg"),
    (PICKUP_TASK_PLAY, "pickup_env_cfg:OmxBallPickupEnvCfg_PLAY", "pickup_rsl_rl_ppo_cfg:OmxBallPickupPPORunnerCfg"),
)

for _task, _env_cfg, _agent_cfg in _TASKS:
    gym.register(
        id=_task,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{_env_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.{_agent_cfg}",
        },
    )
