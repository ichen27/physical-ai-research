"""Single description of the ROBOTIS OMX follower arm.

The simulation task and the real-arm runner both import this module, so
joint order, limits, servo assumptions and unit conversions are defined once.

Angles are radians in the URDF joint frame. Servo positions are DYNAMIXEL
encoder ticks (4096 per revolution).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

TICKS_PER_REV = 4096
RAD_PER_TICK = 2.0 * math.pi / TICKS_PER_REV
STOCK_ZERO_TICK = 2048

# Policy and LeRobot control rate used on this arm.
CONTROL_HZ = 30.0


# Servo and joint values below are taken from the vendor's MuJoCo model
# (assets/open_manipulator_description/mujoco/omx/omx.xml). That file describes its own
# gains and joint dynamics as approximate and does not model the servos' motion
# profiles, so training varies these values instead of trusting them exactly
# (see understudy/variation.py). tests/test_vendor_spec.py checks them against the file.
VENDOR_POSITION_P_GAIN = 1000
VENDOR_POSITION_D_GAIN = 1000
# With Drive Mode 4 the vendor follower configuration uses a time-based profile:
# every new goal position is reached over this time.
VENDOR_PROFILE_TIME_S = 0.050


@dataclass(frozen=True)
class Joint:
    name: str  # URDF joint name
    dxl_id: int
    servo: str  # DYNAMIXEL model
    lower: float  # rad, limit used in simulation and enforced by the runner
    upper: float
    default: float  # rad, pose the reach task resets around
    stiffness: float  # N*m/rad, position gain of the vendor model
    drive_damping: float  # N*m*s/rad, velocity gain the vendor model compiles to in MuJoCo 3.3.7
    passive_damping: float  # N*m*s/rad, joint damping of the vendor model
    armature: float  # kg*m^2, reflected rotor inertia
    effort_limit: float  # N*m
    max_velocity: float = 4.8  # rad/s, vendor URDF limit

    @property
    def damping(self) -> float:
        """Total damping for a simulator that has one damping term per joint drive."""
        return self.drive_damping + self.passive_damping


# Limits are deliberately narrower than the vendor servo limits
# (ID 12: 830-3129 ticks, ID 13: 1024-3140 ticks, others unlimited) so a policy
# cannot command the arm into its own base or the table edge of travel.
ARM_JOINTS: tuple[Joint, ...] = (
    Joint("joint1", 11, "XL430-W250", -1.57, 1.57, 0.0, 50.0, 2.1047, 0.20, 0.005, 1.5),
    Joint("joint2", 12, "XL430-W250", -1.50, 1.50, -0.35, 50.0, 2.3606, 0.20, 0.005, 1.5),
    Joint("joint3", 13, "XL430-W250", -1.50, 1.50, 0.60, 50.0, 1.8746, 0.20, 0.005, 1.5),
    Joint("joint4", 14, "XL330-M288", -1.70, 1.70, 0.90, 50.0, 0.5185, 0.05, 0.001, 0.52),
    Joint("joint5", 15, "XL330-M288", -1.57, 1.57, 0.0, 50.0, 0.4559, 0.05, 0.001, 0.52),
)
# The vendor model applies 0.1 N*m of dry friction at each arm joint. PhysX has no
# equivalent torque-valued setting, so the Isaac Lab arm does not model it; the
# MuJoCo cross-check does.
VENDOR_JOINT_FRICTION_LOSS = 0.1

# One servo drives both fingers through gears.
GRIPPER_STIFFNESS = 50.0
GRIPPER_DRIVE_DAMPING = 0.4963
GRIPPER_PASSIVE_DAMPING = 0.01
GRIPPER_ARMATURE = 0.0003
GRIPPER_EFFORT_LIMIT = 0.2  # N*m at the configured current limit, shared by the two fingers
GRIPPER_JOINT_NAMES = ("gripper_joint_1", "gripper_joint_2")
# Gripper travel from the vendor MuJoCo model. The second finger mirrors the first.
GRIPPER_LIMITS = {"gripper_joint_1": (0.0, 1.745329), "gripper_joint_2": (-1.745329, 0.0)}
GRIPPER_DXL_ID = 16

ARM_JOINT_NAMES = tuple(j.name for j in ARM_JOINTS)
END_EFFECTOR_LINK = "end_effector_link"
BASE_LINK = "link0"

# Reach targets, metres in the arm base frame.
TARGET_BOX = {"x": (0.12, 0.26), "y": (-0.12, 0.12), "z": (0.04, 0.22)}
# A reach counts as a success inside this distance.
SUCCESS_RADIUS_M = 0.02


def ticks_to_rad(ticks: float, zero_tick: float = STOCK_ZERO_TICK, sign: int = 1) -> float:
    if sign not in (1, -1):
        raise ValueError("sign must be 1 or -1")
    return sign * (ticks - zero_tick) * RAD_PER_TICK


def rad_to_ticks(rad: float, zero_tick: float = STOCK_ZERO_TICK, sign: int = 1) -> int:
    if sign not in (1, -1):
        raise ValueError("sign must be 1 or -1")
    return int(round(zero_tick + sign * rad / RAD_PER_TICK))


def clamp_to_limits(angles: list[float] | tuple[float, ...]) -> list[float]:
    """Clamp arm joint angles to the limits the policy was trained with."""
    if len(angles) != len(ARM_JOINTS):
        raise ValueError(f"expected {len(ARM_JOINTS)} joint angles, got {len(angles)}")
    return [min(max(a, j.lower), j.upper) for a, j in zip(angles, ARM_JOINTS)]


@dataclass(frozen=True)
class Calibration:
    """Maps one real arm's encoder ticks to URDF joint angles.

    An arm set up through LeRobot carries its own homing offsets, so tick 2048
    is not guaranteed to be the URDF zero pose. ``zero_ticks`` is the reading
    of each arm servo with the arm held in the URDF zero pose, and ``signs``
    records whether increasing ticks increase the URDF angle.

    ``gripper_closed_tick`` is the gripper servo's reading with the fingers
    closed on nothing, and ``gripper_sign`` is 1 if opening increases its ticks.
    They are needed only for tasks that use the gripper.
    """

    zero_ticks: tuple[float, ...]
    signs: tuple[int, ...]
    gripper_closed_tick: float | None = None
    gripper_sign: int = 1

    def __post_init__(self) -> None:
        n = len(ARM_JOINTS)
        if len(self.zero_ticks) != n or len(self.signs) != n:
            raise ValueError(f"calibration needs {n} zero ticks and {n} signs")
        if any(s not in (1, -1) for s in self.signs) or self.gripper_sign not in (1, -1):
            raise ValueError("every sign must be 1 or -1")

    @classmethod
    def stock(cls) -> "Calibration":
        n = len(ARM_JOINTS)
        return cls((float(STOCK_ZERO_TICK),) * n, (1,) * n, float(STOCK_ZERO_TICK), 1)

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text())
        try:
            closed = data.get("gripper_closed_tick")
            return cls(
                tuple(float(t) for t in data["zero_ticks"]),
                tuple(int(s) for s in data["signs"]),
                float(closed) if closed is not None else None,
                int(data.get("gripper_sign", 1)),
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"{path}: expected keys 'zero_ticks' and 'signs'") from exc

    def save(self, path: str | Path) -> None:
        data = {
            "zero_ticks": list(self.zero_ticks),
            "signs": list(self.signs),
            "gripper_closed_tick": self.gripper_closed_tick,
            "gripper_sign": self.gripper_sign,
        }
        Path(path).write_text(json.dumps(data, indent=2) + "\n")

    def to_rad(self, ticks: list[float] | tuple[float, ...]) -> list[float]:
        if len(ticks) != len(ARM_JOINTS):
            raise ValueError(f"expected {len(ARM_JOINTS)} tick readings, got {len(ticks)}")
        return [ticks_to_rad(t, z, s) for t, z, s in zip(ticks, self.zero_ticks, self.signs)]

    def to_ticks(self, angles: list[float] | tuple[float, ...]) -> list[int]:
        return [rad_to_ticks(a, z, s) for a, z, s in zip(clamp_to_limits(angles), self.zero_ticks, self.signs)]

    def _gripper_zero(self) -> float:
        if self.gripper_closed_tick is None:
            raise ValueError("this calibration has no gripper reading; capture it before using the gripper")
        return self.gripper_closed_tick

    def finger_to_rad(self, ticks: float) -> float:
        """Finger angle: 0 closed, positive open."""
        return ticks_to_rad(ticks, self._gripper_zero(), self.gripper_sign)

    def finger_to_ticks(self, angle: float) -> int:
        low, high = GRIPPER_LIMITS[GRIPPER_JOINT_NAMES[0]]
        return rad_to_ticks(min(max(angle, low), high), self._gripper_zero(), self.gripper_sign)
