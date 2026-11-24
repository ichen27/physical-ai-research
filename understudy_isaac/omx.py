"""The OMX follower arm as an Isaac Lab articulation.

Geometry and inertia come from the vendor URDF. Actuator values come from
``understudy.arm`` so the simulated servos and the real-arm runner share one
set of assumptions.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from understudy import arm

_DESCRIPTION = Path(__file__).resolve().parent.parent / "assets" / "open_manipulator_description"


def _actuator(joints: list[str], servo: arm.Servo) -> ImplicitActuatorCfg:
    return ImplicitActuatorCfg(
        joint_names_expr=joints,
        stiffness=servo.stiffness,
        damping=servo.damping,
        armature=servo.armature,
        effort_limit_sim=servo.stall_torque,
        velocity_limit_sim=servo.max_velocity,
    )


OMX_F_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(_DESCRIPTION / "urdf" / "omx_f" / "omx_f.urdf"),
        ros_package_paths=[{"name": "open_manipulator_description", "path": str(_DESCRIPTION)}],
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
        "xl430": _actuator([j.name for j in arm.ARM_JOINTS if j.servo is arm.XL430], arm.XL430),
        "xl330": _actuator([j.name for j in arm.ARM_JOINTS if j.servo is arm.XL330], arm.XL330),
        "gripper": _actuator(list(arm.GRIPPER_JOINT_NAMES), arm.XL330),
    },
)
