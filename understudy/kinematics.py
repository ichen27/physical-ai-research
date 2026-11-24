"""Forward kinematics read straight from the arm's URDF.

Used to score a reach on the real arm (encoder angles in, gripper-tip position
out) without trusting the simulator, and to check that reach targets are inside
the arm's workspace.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .arm import ARM_JOINT_NAMES, BASE_LINK, END_EFFECTOR_LINK

Matrix = list[list[float]]

DEFAULT_URDF = (
    Path(__file__).resolve().parent.parent / "assets" / "open_manipulator_description" / "urdf" / "omx_f" / "omx_f.urdf"
)


def _identity() -> Matrix:
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _matmul(a: Matrix, b: Matrix) -> Matrix:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _origin(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> Matrix:
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, xyz[0]],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, xyz[1]],
        [-sp, cp * sr, cp * cr, xyz[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _axis_rotation(axis: tuple[float, float, float], angle: float) -> Matrix:
    norm = math.sqrt(sum(a * a for a in axis))
    if norm == 0.0:
        raise ValueError("joint axis has zero length")
    x, y, z = (a / norm for a in axis)
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0.0],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0.0],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _floats(text: str | None, default: str) -> tuple[float, float, float]:
    values = tuple(float(v) for v in (text or default).split())
    if len(values) != 3:
        raise ValueError(f"expected three numbers, got {text!r}")
    return values  # type: ignore[return-value]


@dataclass(frozen=True)
class _ChainJoint:
    name: str
    kind: str
    origin: Matrix
    axis: tuple[float, float, float]


class Chain:
    """The joints between the arm base and the gripper tip."""

    def __init__(self, urdf_path: str | Path = DEFAULT_URDF, base: str = BASE_LINK, tip: str = END_EFFECTOR_LINK):
        text = Path(urdf_path).read_text()
        # A URDF never needs a document type or entities, and refusing them
        # closes the entity-expansion attacks the standard parser allows.
        if "<!DOCTYPE" in text or "<!ENTITY" in text:
            raise ValueError(f"{urdf_path}: document type and entity declarations are not accepted")
        root = ET.fromstring(text)
        by_child: dict[str, ET.Element] = {}
        for joint in root.findall("joint"):
            child = joint.find("child")
            if child is not None:
                by_child[child.get("link", "")] = joint

        joints: list[_ChainJoint] = []
        link = tip
        while link != base:
            joint = by_child.get(link)
            if joint is None:
                raise ValueError(f"no chain from {base!r} to {tip!r} in {urdf_path}")
            origin = joint.find("origin")
            axis = joint.find("axis")
            joints.append(
                _ChainJoint(
                    name=joint.get("name", ""),
                    kind=joint.get("type", "fixed"),
                    origin=_origin(
                        _floats(origin.get("xyz") if origin is not None else None, "0 0 0"),
                        _floats(origin.get("rpy") if origin is not None else None, "0 0 0"),
                    ),
                    axis=_floats(axis.get("xyz") if axis is not None else None, "1 0 0"),
                )
            )
            link = joint.find("parent").get("link", "")  # type: ignore[union-attr]
        self._joints = tuple(reversed(joints))

        moving = tuple(j.name for j in self._joints if j.kind != "fixed")
        if moving != ARM_JOINT_NAMES:
            raise ValueError(f"URDF chain joints {moving} do not match the arm description {ARM_JOINT_NAMES}")

    def tip_position(self, angles: list[float] | tuple[float, ...]) -> tuple[float, float, float]:
        """Gripper-tip position in the arm base frame, in metres."""
        if len(angles) != len(ARM_JOINT_NAMES):
            raise ValueError(f"expected {len(ARM_JOINT_NAMES)} joint angles, got {len(angles)}")
        remaining = iter(angles)
        pose = _identity()
        for joint in self._joints:
            pose = _matmul(pose, joint.origin)
            if joint.kind != "fixed":
                pose = _matmul(pose, _axis_rotation(joint.axis, next(remaining)))
        return (pose[0][3], pose[1][3], pose[2][3])


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))
