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
    mode: str = "absolute"
    """How an output becomes a joint target.

    ``absolute``: default pose + scale * output.
    ``relative``: the joint's present angle + scale * output.
    ``incremental``: the previous target + scale * output. Holding still then needs an
    output of zero whatever the servo stiffness, because the target stays where it was put.
    """
    max_lead: float = 0.35
    """In incremental mode, how far (rad) the target may run ahead of the joint's present angle.

    Without this the target would keep moving while the arm is blocked, for example against the desk.
    """
