"""Pickup task: lift a foam golf ball off the desk and hold it at a target point.

Built on Isaac Lab's lift task. The policy sees the arm and gripper joint
angles and velocities, the ball position, the hold target and its previous
action. It outputs five joint position targets and an open-or-close gripper
command at 30 Hz.

The ball position is read from the simulator here. On a real arm it has to
come from a camera.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from understudy import arm, pickup

from . import pickup_mdp
from .omx import OMX_F_CFG

_FINGER_1, _FINGER_2 = arm.GRIPPER_JOINT_NAMES


@configclass
class OmxBallPickupEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 120 Hz physics for the finger contacts, one policy step every four physics steps.
        self.sim.dt = 1.0 / 120.0
        self.decimation = 4
        self.sim.render_interval = self.decimation
        if abs(1.0 / (self.sim.dt * self.decimation) - arm.CONTROL_HZ) > 1e-6:
            raise ValueError("simulation control rate no longer matches the real arm's control rate")
        self.episode_length_s = pickup.EPISODE_SECONDS

        # The arm and the ball sit on the ground plane, which stands in for the desk.
        self.scene.env_spacing = 1.0
        self.scene.table = None
        self.scene.plane.init_state.pos = [0.0, 0.0, 0.0]
        self.scene.ee_frame = None
        self.scene.robot = OMX_F_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    **dict(zip(arm.ARM_JOINT_NAMES, pickup.START_POSE)),
                    _FINGER_1: pickup.GRIPPER_OPEN_RAD,
                    _FINGER_2: -pickup.GRIPPER_OPEN_RAD,
                },
            ),
        )
        spawn_x, spawn_y = pickup.BALL_SPAWN["x"], pickup.BALL_SPAWN["y"]
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[sum(spawn_x) / 2, sum(spawn_y) / 2, pickup.BALL_RADIUS_M]
            ),
            spawn=sim_utils.SphereCfg(
                radius=pickup.BALL_RADIUS_M,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_depenetration_velocity=5.0,
                    linear_damping=pickup.BALL_LINEAR_DAMPING,
                    angular_damping=pickup.BALL_ANGULAR_DAMPING,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=pickup.BALL_MASS_KG),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=pickup.BALL_FRICTION, dynamic_friction=pickup.BALL_FRICTION, restitution=0.2
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.4, 0.0)),
            ),
        )
        self.events.reset_object_position.params["pose_range"] = {
            "x": (spawn_x[0] - sum(spawn_x) / 2, spawn_x[1] - sum(spawn_x) / 2),
            "y": (spawn_y[0] - sum(spawn_y) / 2, spawn_y[1] - sum(spawn_y) / 2),
            "z": (0.0, 0.0),
        }
        self.events.reset_object_position.params["asset_cfg"] = SceneEntityCfg("object")

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(arm.ARM_JOINT_NAMES),
            preserve_order=True,
            scale=0.5,
            use_default_offset=True,
            clip={j.name: (j.lower, j.upper) for j in arm.ARM_JOINTS},
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=list(arm.GRIPPER_JOINT_NAMES),
            open_command_expr={_FINGER_1: pickup.GRIPPER_OPEN_RAD, _FINGER_2: -pickup.GRIPPER_OPEN_RAD},
            close_command_expr={_FINGER_1: pickup.GRIPPER_CLOSED_RAD, _FINGER_2: -pickup.GRIPPER_CLOSED_RAD},
        )

        sensed_joints = SceneEntityCfg("robot", joint_names=[*arm.ARM_JOINT_NAMES, _FINGER_1], preserve_order=True)
        policy = self.observations.policy
        policy.joint_pos.params = {"asset_cfg": sensed_joints}
        policy.joint_pos.noise = Unoise(n_min=-0.01, n_max=0.01)
        policy.joint_vel.params = {"asset_cfg": sensed_joints}
        policy.joint_vel.noise = Unoise(n_min=-0.01, n_max=0.01)
        # A camera will not place the ball better than a few millimetres.
        policy.object_position.noise = Unoise(n_min=-0.005, n_max=0.005)

        target = self.commands.object_pose
        target.body_name = arm.END_EFFECTOR_LINK
        target.ranges.pos_x = pickup.HOLD_TARGET["x"]
        target.ranges.pos_y = pickup.HOLD_TARGET["y"]
        target.ranges.pos_z = pickup.HOLD_TARGET["z"]
        # One target per attempt.
        target.resampling_time_range = (pickup.EPISODE_SECONDS, pickup.EPISODE_SECONDS)

        tip = SceneEntityCfg("robot", body_names=[arm.END_EFFECTOR_LINK])
        self.rewards.reaching_object = RewTerm(
            func=pickup_mdp.grasp_point_object_distance, params={"std": 0.1, "robot_cfg": tip}, weight=1.0
        )
        # Without these two terms the arm learns to hover over the ball and never closes:
        # nothing else pays for closing the gripper or for the first millimetres of lift.
        self.rewards.closing_on_ball = RewTerm(
            func=pickup_mdp.closing_on_ball,
            params={"near": 0.02, "tip_cfg": tip, "finger_cfg": SceneEntityCfg("robot", joint_names=[_FINGER_1])},
            weight=2.0,
        )
        self.rewards.ball_height = RewTerm(func=pickup_mdp.ball_height, weight=5.0)
        self.rewards.lifting_object.params["minimal_height"] = pickup.LIFTED_HEIGHT_M
        self.rewards.object_goal_tracking.params.update(
            {"std": 0.1, "minimal_height": pickup.LIFTED_HEIGHT_M, "success_threshold": pickup.HOLD_RADIUS_M}
        )
        self.rewards.object_goal_tracking_fine_grained.params.update(
            {"std": 0.02, "minimal_height": pickup.LIFTED_HEIGHT_M}
        )

        # The ball cannot fall off the desk here; an attempt ends on time.
        self.terminations.object_dropping = None


@configclass
class OmxBallPickupEnvCfg_PLAY(OmxBallPickupEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (0.50, 0.45, 0.35)
        self.viewer.lookat = (0.15, 0.0, 0.08)
