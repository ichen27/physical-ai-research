"""Settings for :class:`understudy_isaac.servo_action.ProfiledJointPositionAction`.

Kept free of simulator imports: task configuration is loaded before the simulator starts.
"""

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils.configclass import configclass


@configclass
class ProfiledJointPositionActionCfg(JointPositionActionCfg):
    class_type: type | str = "understudy_isaac.servo_action:ProfiledJointPositionAction"
    delay_range_s: tuple[float, float] = (0.0, 0.0)
    """Time before a new command starts to take effect."""
    profile_time_range_s: tuple[float, float] = (0.0, 0.0)
    """Time the servo takes to move its goal to a new command. Zero means at once."""
