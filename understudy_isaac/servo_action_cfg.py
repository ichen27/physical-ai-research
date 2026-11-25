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
    relative: bool = False
    """If True, the command is an offset from the joint's present angle instead of from the default pose.

    A command relative to where the joint is does not depend on how far the servo
    sags under load, so the same policy suits firm and soft servos.
    """
