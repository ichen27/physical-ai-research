"""The OMX follower arm as an Isaac Lab articulation.

Geometry and inertia come from the vendor URDF, through the corrected copy
that ``understudy.sim_urdf`` generates. Servo values come from ``understudy.arm``,
which takes them from the vendor's MuJoCo model.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from understudy import arm
from understudy.sim_urdf import GENERATED, GENERATED_PACKAGE, SIM_URDF

_DESCRIPTION = Path(__file__).resolve().parent.parent / "assets" / "open_manipulator_description"


OMX_F_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(SIM_URDF),
        ros_package_paths=[
            {"name": "open_manipulator_description", "path": str(_DESCRIPTION)},
            {"name": GENERATED_PACKAGE, "path": str(GENERATED)},
        ],
        fix_base=True,
        # Keep end_effector_link as its own body; the reach target is measured there.
        merge_fixed_joints=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={**{j.name: j.default for j in arm.ARM_JOINTS}, **{name: 0.0 for name in arm.GRIPPER_JOINT_NAMES}},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=list(arm.ARM_JOINT_NAMES),
            stiffness={j.name: j.stiffness for j in arm.ARM_JOINTS},
            damping={j.name: j.damping for j in arm.ARM_JOINTS},
            armature={j.name: j.armature for j in arm.ARM_JOINTS},
            effort_limit_sim={j.name: j.effort_limit for j in arm.ARM_JOINTS},
            velocity_limit_sim={j.name: j.max_velocity for j in arm.ARM_JOINTS},
        ),
        # One real servo drives both fingers, so each simulated finger gets half its torque.
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=list(arm.GRIPPER_JOINT_NAMES),
            stiffness=arm.GRIPPER_STIFFNESS,
            damping=arm.GRIPPER_DRIVE_DAMPING + arm.GRIPPER_PASSIVE_DAMPING,
            armature=arm.GRIPPER_ARMATURE,
            effort_limit_sim=arm.GRIPPER_EFFORT_LIMIT / 2,
            velocity_limit_sim=4.8,
        ),
    },
)
