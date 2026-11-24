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


@dataclass(frozen=True)
class Servo:
    """Simulation assumptions for one DYNAMIXEL model.

    ``stiffness`` is a first-principles estimate, not a measurement. With the
    vendor Position P Gain of 1000 the controller output is ``1000 / 128`` PWM
    units per tick of error and saturates at 885 units, which is about 0.174 rad
    of error. Dividing stall torque by that error gives the value below. It must
    be replaced by a step-response measurement on the real arm.
    """

    model: str
    stall_torque: float  # N*m, vendor MuJoCo force range
    stiffness: float  # N*m/rad at Position P Gain 1000
    damping: float  # N*m*s/rad, near critical for the armature below
    armature: float  # kg*m^2, vendor MuJoCo model
    max_velocity: float  # rad/s, vendor URDF limit


XL430 = Servo("XL430-W250", stall_torque=1.5, stiffness=8.0, damping=0.7, armature=0.005, max_velocity=4.8)
XL330 = Servo("XL330-M288", stall_torque=0.52, stiffness=3.0, damping=0.15, armature=0.001, max_velocity=4.8)

VENDOR_POSITION_P_GAIN = 1000


@dataclass(frozen=True)
class Joint:
    name: str  # URDF joint name
    dxl_id: int
    servo: Servo
    lower: float  # rad, limit used in simulation and enforced by the runner
    upper: float
    default: float  # rad, pose the episode resets around


# Limits are deliberately narrower than the vendor servo limits
# (ID 12: 830-3129 ticks, ID 13: 1024-3140 ticks, others unlimited) so a policy
# cannot command the arm into its own base or the table edge of travel.
ARM_JOINTS: tuple[Joint, ...] = (
    Joint("joint1", 11, XL430, -1.57, 1.57, 0.0),
    Joint("joint2", 12, XL430, -1.50, 1.50, -0.35),
    Joint("joint3", 13, XL430, -1.50, 1.50, 0.60),
    Joint("joint4", 14, XL330, -1.70, 1.70, 0.90),
    Joint("joint5", 15, XL330, -1.57, 1.57, 0.0),
)
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
    """

    zero_ticks: tuple[float, ...]
    signs: tuple[int, ...]

    def __post_init__(self) -> None:
        n = len(ARM_JOINTS)
        if len(self.zero_ticks) != n or len(self.signs) != n:
            raise ValueError(f"calibration needs {n} zero ticks and {n} signs")
        if any(s not in (1, -1) for s in self.signs):
            raise ValueError("every sign must be 1 or -1")

    @classmethod
    def stock(cls) -> "Calibration":
        n = len(ARM_JOINTS)
        return cls((float(STOCK_ZERO_TICK),) * n, (1,) * n)

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text())
        try:
            return cls(tuple(float(t) for t in data["zero_ticks"]), tuple(int(s) for s in data["signs"]))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path}: expected keys 'zero_ticks' and 'signs'") from exc

    def to_rad(self, ticks: list[float] | tuple[float, ...]) -> list[float]:
        if len(ticks) != len(ARM_JOINTS):
            raise ValueError(f"expected {len(ARM_JOINTS)} tick readings, got {len(ticks)}")
        return [ticks_to_rad(t, z, s) for t, z, s in zip(ticks, self.zero_ticks, self.signs)]

    def to_ticks(self, angles: list[float] | tuple[float, ...]) -> list[int]:
        return [rad_to_ticks(a, z, s) for a, z, s in zip(clamp_to_limits(angles), self.zero_ticks, self.signs)]
