"""Pickup task: lift a foam golf ball off the desk and hold it at a target point.

Built on Isaac Lab's lift task. The policy sees the arm and gripper joint
angles and velocities, the ball position as a camera would report it, the hold
target and its previous action. It outputs five joint position targets and an
open-or-close gripper command at 30 Hz.

Every simulated arm differs: servo gains, rotor inertia, command delay, servo
profile time, camera error and ball mass and friction are drawn from the ranges
in ``understudy.variation``, so the policy cannot depend on one exact value.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from understudy import arm, pickup, variation

from . import pickup_mdp
from .omx import OMX_F_CFG
from .servo_action_cfg import ProfiledJointPositionActionCfg

_FINGER_1, _FINGER_2 = arm.GRIPPER_JOINT_NAMES


@configclass
class CriticCfg(ObsGroup):
    """What the value estimate sees during training: the same quantities without camera error.

    Only the policy group is used once training is over.
    """

    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    object_position = ObsTerm(func=pickup_mdp.ball_position_b)
    target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


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

        self.actions.arm_action = ProfiledJointPositionActionCfg(
            asset_name="robot",
            joint_names=list(arm.ARM_JOINT_NAMES),
            preserve_order=True,
            scale=pickup.ARM_ACTION_SCALE,
            use_default_offset=pickup.ARM_ACTION_MODE == "absolute",
            mode=pickup.ARM_ACTION_MODE,
            max_lead=pickup.ARM_TARGET_MAX_LEAD,
            clip={j.name: (j.lower, j.upper) for j in arm.ARM_JOINTS},
            delay_range_s=variation.COMMAND_DELAY_S,
            profile_time_range_s=variation.PROFILE_TIME_S,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=list(arm.GRIPPER_JOINT_NAMES),
            open_command_expr={_FINGER_1: pickup.GRIPPER_OPEN_RAD, _FINGER_2: -pickup.GRIPPER_OPEN_RAD},
            close_command_expr={_FINGER_1: pickup.GRIPPER_CLOSED_RAD, _FINGER_2: -pickup.GRIPPER_CLOSED_RAD},
        )

        sensed_joints = SceneEntityCfg("robot", joint_names=[*arm.ARM_JOINT_NAMES, _FINGER_1], preserve_order=True)
        tip = SceneEntityCfg("robot", body_names=[arm.END_EFFECTOR_LINK])
        policy = self.observations.policy
        policy.joint_pos.params = {"asset_cfg": sensed_joints}
        policy.joint_pos.noise = Unoise(n_min=-0.01, n_max=0.01)
        policy.joint_vel.params = {"asset_cfg": sensed_joints}
        policy.joint_vel.noise = Unoise(n_min=-0.01, n_max=0.01)
        policy.object_position = ObsTerm(
            func=pickup_mdp.seen_ball_position,
            params={
                "bias": variation.BALL_POSITION_BIAS_M,
                "noise": variation.BALL_POSITION_NOISE_M,
                "latency_steps": variation.CAMERA_LATENCY_STEPS,
                "occlusion_distance": variation.OCCLUSION_DISTANCE_M,
                "occlusion_probability": variation.OCCLUSION_PROBABILITY,
                "tip_cfg": tip,
            },
        )
        if pickup.ARM_ACTION_MODE == "incremental":
            policy.target_lead = ObsTerm(func=pickup_mdp.arm_target_lead, noise=Unoise(n_min=-0.01, n_max=0.01))
        self.observations.critic = CriticCfg()
        if pickup.ARM_ACTION_MODE == "incremental":
            self.observations.critic.target_lead = ObsTerm(func=pickup_mdp.arm_target_lead)
        self.observations.critic.joint_pos.params = {"asset_cfg": sensed_joints}
        self.observations.critic.joint_vel.params = {"asset_cfg": sensed_joints}

        # Give every simulated arm and ball its own physical properties.
        arm_joints = SceneEntityCfg("robot", joint_names=list(arm.ARM_JOINT_NAMES))
        self.events.servo_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": arm_joints,
                "stiffness_distribution_params": variation.SERVO_STIFFNESS_SCALE,
                "damping_distribution_params": variation.SERVO_DAMPING_SCALE,
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )
        self.events.rotor_inertia = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": arm_joints,
                "armature_distribution_params": variation.ROTOR_INERTIA_SCALE,
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.events.gripper_force = EventTerm(
            func=pickup_mdp.randomize_gripper_force,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(arm.GRIPPER_JOINT_NAMES)),
                "scale_range": variation.GRIPPER_FORCE_SCALE,
            },
        )
        self.events.ball_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "mass_distribution_params": variation.BALL_MASS_KG,
                "operation": "abs",
                "distribution": "log_uniform",
            },
        )
        self.events.ball_friction = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "static_friction_range": variation.BALL_FRICTION,
                "dynamic_friction_range": variation.BALL_FRICTION,
                "restitution_range": (0.0, 0.3),
                "num_buckets": 64,
                "make_consistent": True,
            },
        )

        target = self.commands.object_pose
        target.body_name = arm.END_EFFECTOR_LINK
        target.ranges.pos_x = pickup.HOLD_TARGET["x"]
        target.ranges.pos_y = pickup.HOLD_TARGET["y"]
        target.ranges.pos_z = pickup.HOLD_TARGET["z"]
        # One target per attempt.
        target.resampling_time_range = (pickup.EPISODE_SECONDS, pickup.EPISODE_SECONDS)

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

        # Real servos shut down when held at their limit, so prefer gentle contact with the desk.
        self.rewards.joint_torque = RewTerm(
            func=mdp.joint_torques_l2, weight=-0.02, params={"asset_cfg": arm_joints}
        )

        # The ball cannot fall off the desk here; an attempt ends on time.
        self.terminations.object_dropping = None


@configclass
class OmxBallPickupEnvCfg_PLAY(OmxBallPickupEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        # Record with --num_envs 1: the camera is placed relative to the world origin.
        self.viewer.eye = (0.50, 0.45, 0.35)
        self.viewer.lookat = (0.15, 0.0, 0.08)
