"""Reach task: move the OMX gripper tip to a target point.

Built on Isaac Lab's reach task. The policy sees the five arm joint angles and
velocities, the target position and its previous action, and outputs joint
position targets at 30 Hz. Everything it sees is available on the real arm.
"""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

from understudy import arm

from .omx import OMX_F_CFG


@configclass
class OmxReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # The arm is bolted to the ground plane; the larger arms' table is not used.
        self.scene.robot = OMX_F_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.table = None
        self.scene.ground.init_state.pos = (0.0, 0.0, 0.0)
        self.scene.env_spacing = 1.0

        # 60 Hz physics, one policy step every two physics steps.
        if abs(1.0 / (self.sim.dt * self.decimation) - arm.CONTROL_HZ) > 1e-6:
            raise ValueError("simulation control rate no longer matches the real arm's control rate")

        arm_joints = SceneEntityCfg("robot", joint_names=list(arm.ARM_JOINT_NAMES), preserve_order=True)
        self.observations.policy.joint_pos.params = {"asset_cfg": arm_joints}
        self.observations.policy.joint_vel.params = {"asset_cfg": arm_joints}

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(arm.ARM_JOINT_NAMES),
            preserve_order=True,
            scale=0.5,
            use_default_offset=True,
            clip={j.name: (j.lower, j.upper) for j in arm.ARM_JOINTS},
        )

        tip = [arm.END_EFFECTOR_LINK]
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = tip
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = tip
        self.rewards.end_effector_position_tracking_fine_grained.params["std"] = 0.05
        # Five joints cannot hold an arbitrary orientation, so only position is rewarded.
        self.rewards.end_effector_orientation_tracking = None

        command = self.commands.ee_pose
        command.body_name = arm.END_EFFECTOR_LINK
        command.position_success_threshold = arm.SUCCESS_RADIUS_M
        command.ranges.pos_x = arm.TARGET_BOX["x"]
        command.ranges.pos_y = arm.TARGET_BOX["y"]
        command.ranges.pos_z = arm.TARGET_BOX["z"]
        command.ranges.roll = (0.0, 0.0)
        command.ranges.pitch = (0.0, 0.0)
        command.ranges.yaw = (0.0, 0.0)

        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)


@configclass
class OmxReachEnvCfg_PLAY(OmxReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        # The arm is small; bring the camera in from the larger arms' default.
        self.viewer.eye = (0.55, 0.55, 0.45)
        self.viewer.lookat = (0.12, 0.0, 0.10)
        self.observations.policy.enable_corruption = False
