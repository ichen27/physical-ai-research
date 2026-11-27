"""The camera on the arm's wrist: where a ball on the desk appears in its picture, and back.

The camera rides on the last link, so where it is follows from the joint angles
and one fixed pose, the mount, which the vendor's arm description does not have
and which has to be found by calibration (``understudy.camera_calibration``).

One camera cannot measure distance, but the ball rests on the desk, so its
centre is one ball radius above it. The ball's place is where the line of sight
through its picture crosses that height.

Camera axes follow the usual convention: x to the right of the picture, y down
it, z out along the line of sight. Pixels are counted from the top left of the
full-size picture. Needs ``numpy``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from . import pickup
from .kinematics import Chain

FILE_KIND = "understudy wrist camera"


def rotation_from_vector(vector: Sequence[float]) -> np.ndarray:
    """Rotation matrix for a rotation vector (axis times angle in radians)."""
    v = np.asarray(vector, dtype=float)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = v / angle
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def vector_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Rotation vector for a rotation matrix."""
    r = np.asarray(rotation, dtype=float)
    cosine = min(1.0, max(-1.0, (np.trace(r) - 1.0) / 2.0))
    angle = math.acos(cosine)
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle < 1e-6:
        # Half a turn: the axis is the eigenvector of the matrix with eigenvalue one.
        values, vectors = np.linalg.eigh((r + r.T) / 2.0)
        return np.real(vectors[:, int(np.argmax(values))]) * angle
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2.0 * math.sin(angle))
    return axis * angle


def sight_many(
    focal_px: float,
    centre_px: Sequence[float],
    distortion: float,
    mount_xyz: Sequence[float],
    mount_rotation: Sequence[float],
    tip_rotations: np.ndarray,
    tip_positions: np.ndarray,
    balls: np.ndarray,
    radius_m: float = pickup.BALL_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Where balls appear, for many arm poses at once.

    ``tip_rotations`` (N x 3 x 3) and ``tip_positions`` (N x 3) are the gripper tip's
    pose for each picture and ``balls`` (N x 3) the ball's centre in each. Returns each
    ball's picture centre (u, v), its radius in pixels, and whether it is in front of the
    camera at all; the numbers mean nothing where that is false.

    A ball away from the middle of the picture appears as an ellipse whose centre is a
    little further out than the picture of the ball's own centre. The centre returned
    is the ellipse's, which is what a detector measures.
    """
    rotations = tip_rotations @ rotation_from_vector(mount_rotation)
    positions = tip_rotations @ np.asarray(mount_xyz, dtype=float) + tip_positions
    seen = np.einsum("nji,nj->ni", rotations, np.asarray(balls, dtype=float) - positions)
    x, y, z = seen[:, 0], seen[:, 1], seen[:, 2]
    clear = z * z - radius_m * radius_m
    ahead = (z > radius_m) & (clear > 1e-9)
    clear = np.where(ahead, clear, 1.0)
    distance_sq = x * x + y * y + z * z
    # Ellipse centre and half-axes on the plane one unit in front of the camera.
    cx, cy = x * z / clear, y * z / clear
    along = radius_m * np.sqrt(np.maximum(distance_sq - radius_m * radius_m, 0.0)) / clear
    across = radius_m / np.sqrt(clear)
    r2 = cx * cx + cy * cy
    factor = 1.0 + distortion * r2
    stretch = np.sqrt(np.abs(factor * (1.0 + 3.0 * distortion * r2)))
    u = centre_px[0] + focal_px * cx * factor
    v = centre_px[1] + focal_px * cy * factor
    return u, v, focal_px * np.sqrt(along * across) * stretch, ahead


@dataclass(frozen=True)
class Sight:
    """Where a ball should appear in the picture."""

    u: float
    v: float
    radius_px: float
    inside: bool  # the whole ball is within the picture


@dataclass(frozen=True)
class WristCamera:
    """A camera fixed to the arm's last link.

    ``focal_px`` is the focal length in pixels of the full-size picture and
    ``centre_px`` the point the line of sight passes through. ``distortion`` is
    the first radial term: a point at distance r from the centre (in units of
    the focal length) lands at r * (1 + distortion * r^2). It is negative for
    the barrel distortion of a wide lens.

    ``mount_xyz`` is the camera's position in the gripper-tip frame, metres
    (x along the gripper, y to its left, z up when the gripper is level), and
    ``mount_rotation`` turns camera axes into that frame, as a rotation vector.
    """

    width: int
    height: int
    focal_px: float
    centre_px: tuple[float, float]
    distortion: float
    mount_xyz: tuple[float, float, float]
    mount_rotation: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.focal_px <= 0:
            raise ValueError("a camera needs a positive picture size and focal length")

    # Saving ------------------------------------------------------------------

    def save(self, path: str | Path, **notes) -> None:
        data = {"kind": FILE_KIND, **asdict(self), **notes}
        Path(path).write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "WristCamera":
        data = json.loads(Path(path).read_text())
        try:
            return cls(
                int(data["width"]),
                int(data["height"]),
                float(data["focal_px"]),
                tuple(float(v) for v in data["centre_px"]),
                float(data["distortion"]),
                tuple(float(v) for v in data["mount_xyz"]),
                tuple(float(v) for v in data["mount_rotation"]),
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"{path}: not a wrist camera file ({error})") from error

    @classmethod
    def looking_along_gripper(
        cls,
        width: int = 1280,
        height: int = 720,
        focal_px: float = 520.0,
        back_m: float = 0.08,
        up_m: float = 0.05,
        tilt_down_rad: float = 0.0,
        roll_quarter_turns: int = 0,
    ) -> "WristCamera":
        """A guess to start from: above and behind the gripper tip, looking along the gripper.

        ``tilt_down_rad`` tips the line of sight towards the underside of the gripper and
        ``roll_quarter_turns`` turns the picture about it.
        """
        # Camera axes in the gripper-tip frame: sight along +x, picture right is the gripper's
        # right (-y), picture down is the gripper's underside (-z).
        level = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        tilt = rotation_from_vector([-tilt_down_rad, 0.0, 0.0])  # about the camera's own x
        roll = rotation_from_vector([0.0, 0.0, roll_quarter_turns * math.pi / 2.0])
        rotation = vector_from_rotation(level @ tilt @ roll)
        return cls(
            width, height, focal_px, (width / 2.0, height / 2.0), 0.0, (-back_m, 0.0, up_m), tuple(float(v) for v in rotation)
        )

    def resized(self, width: int, height: int) -> "WristCamera":
        """The same camera delivering a picture of another size with the same shape."""
        scale = width / self.width
        if abs(height / self.height - scale) > 1e-3:
            raise ValueError("the new size must have the same shape as the old one")
        centre = (self.centre_px[0] * scale, self.centre_px[1] * scale)
        return replace(self, width=width, height=height, focal_px=self.focal_px * scale, centre_px=centre)

    # Geometry ----------------------------------------------------------------

    def pose(self, joint_angles: Sequence[float], chain: Chain) -> tuple[np.ndarray, np.ndarray]:
        """Camera axes (columns) and position in the arm base frame."""
        tip = np.array(chain.tip_transform(list(joint_angles)))
        rotation = tip[:3, :3] @ rotation_from_vector(self.mount_rotation)
        position = tip[:3, :3] @ np.array(self.mount_xyz) + tip[:3, 3]
        return rotation, position

    def _undistort(self, x: float, y: float) -> tuple[float, float]:
        if self.distortion == 0.0:
            return x, y
        a, b = x, y
        for _ in range(12):
            factor = 1.0 + self.distortion * (a * a + b * b)
            if factor <= 0.2:  # beyond where this lens model means anything
                break
            a, b = x / factor, y / factor
        return a, b

    def sight(self, ball_centre: Sequence[float], joint_angles: Sequence[float], chain: Chain,
              radius_m: float = pickup.BALL_RADIUS_M) -> Sight | None:
        """Where a ball centred at ``ball_centre`` (base frame) appears, or None if it is behind the camera."""
        tip = np.array(chain.tip_transform(list(joint_angles)))
        u, v, radius_px, ahead = sight_many(
            self.focal_px, self.centre_px, self.distortion, self.mount_xyz, self.mount_rotation,
            tip[None, :3, :3], tip[None, :3, 3], np.asarray(ball_centre, dtype=float)[None, :], radius_m,
        )
        if not ahead[0]:
            return None
        u, v, radius_px = float(u[0]), float(v[0]), float(radius_px[0])
        inside = radius_px <= u <= self.width - radius_px and radius_px <= v <= self.height - radius_px
        return Sight(u, v, radius_px, bool(inside))

    def locate(self, u: float, v: float, joint_angles: Sequence[float], chain: Chain,
               radius_m: float = pickup.BALL_RADIUS_M) -> tuple[float, float, float] | None:
        """Centre of a ball resting on the desk whose picture is centred at pixel (u, v).

        None if the line of sight does not come down to the desk in front of the camera.
        """
        rotation, position = self.pose(joint_angles, chain)
        x, y = self._undistort((u - self.centre_px[0]) / self.focal_px, (v - self.centre_px[1]) / self.focal_px)
        shrink = 1.0  # undoes the ellipse centre's outward shift, found by repeating
        point = None
        for _ in range(4):
            ray = rotation @ np.array([x * shrink, y * shrink, 1.0])
            if ray[2] >= -1e-6:
                return None
            reach = (radius_m - position[2]) / ray[2]
            if reach <= 0.0:
                return None
            point = position + reach * ray
            depth = reach  # the ray has unit z in camera axes, so this is the ball centre's depth
            if depth <= radius_m:
                return None
            shrink = 1.0 - (radius_m / depth) ** 2
        return (float(point[0]), float(point[1]), float(radius_m))
