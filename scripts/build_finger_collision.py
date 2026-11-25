"""Build the gripper fingers' collision shapes as sets of convex pieces.

Physics engines replace a collision mesh with its convex hull. The real fingers
are thin curved claws that close into a hollow, and their hulls fill that hollow
in solid, so a simulated gripper can only pinch a ball between two flat faces
where the real one cups it. This cuts each finger into slabs along its length
and takes the hull of each slab, which keeps the hollow.

    pip install numpy scipy
    python scripts/build_finger_collision.py

Writes assets/generated/collision/finger_<n>_<k>.stl in metres. Run it again
only if the vendor finger meshes change; the output is committed.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from understudy.sim_urdf import FINGER_COLLISION_DIR, FINGER_MESHES, FINGER_SLABS  # noqa: E402

MESHES = REPO / "assets" / "open_manipulator_description" / "meshes" / "omx_f"
MILLIMETRES = 0.001
OVERLAP_M = 0.0005  # neighbouring slabs share this much, so no gap opens between pieces


def read_stl(path: Path) -> np.ndarray:
    """Triangles of a binary STL as an (n, 3, 3) array."""
    data = path.read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    triangles = np.empty((count, 3, 3))
    for i in range(count):
        offset = 84 + i * 50 + 12
        triangles[i] = np.array(struct.unpack("<9f", data[offset : offset + 36])).reshape(3, 3)
    return triangles


def write_stl(path: Path, points: np.ndarray, faces: np.ndarray) -> None:
    centre = points.mean(axis=0)
    with path.open("wb") as handle:
        handle.write(b"understudy finger collision piece".ljust(80, b" "))
        handle.write(struct.pack("<I", len(faces)))
        for face in faces:
            a, b, c = points[face]
            normal = np.cross(b - a, c - a)
            if np.dot(normal, a - centre) < 0:  # make the triangle face outwards
                b, c = c, b
                normal = -normal
            length = np.linalg.norm(normal)
            normal = normal / length if length > 0 else normal
            handle.write(struct.pack("<12fH", *normal, *a, *b, *c, 0))


def slab_points(triangles: np.ndarray, low: float, high: float) -> np.ndarray:
    """Mesh points inside a slab of x, including where mesh edges cross its two faces."""
    vertices = triangles.reshape(-1, 3)
    inside = vertices[(vertices[:, 0] >= low) & (vertices[:, 0] <= high)]
    crossings = []
    for a_index, b_index in ((0, 1), (1, 2), (2, 0)):
        a, b = triangles[:, a_index], triangles[:, b_index]
        for plane in (low, high):
            spans = (a[:, 0] - plane) * (b[:, 0] - plane) < 0
            t = (plane - a[spans, 0]) / (b[spans, 0] - a[spans, 0])
            crossings.append(a[spans] + t[:, None] * (b[spans] - a[spans]))
    return np.vstack([inside, *crossings])


def main() -> None:
    FINGER_COLLISION_DIR.mkdir(parents=True, exist_ok=True)
    for old in FINGER_COLLISION_DIR.glob("finger_*.stl"):
        old.unlink()
    for finger, mesh in FINGER_MESHES.items():
        triangles = read_stl(MESHES / mesh) * MILLIMETRES
        x = triangles[..., 0]
        edges = np.linspace(x.min(), x.max(), FINGER_SLABS + 1)
        hull_volume = ConvexHull(triangles.reshape(-1, 3)).volume
        total = 0.0
        for k in range(FINGER_SLABS):
            points = slab_points(triangles, edges[k] - OVERLAP_M, edges[k + 1] + OVERLAP_M)
            hull = ConvexHull(points)
            total += hull.volume
            used = np.unique(hull.simplices)
            remap = {old: new for new, old in enumerate(used)}
            faces = np.vectorize(remap.get)(hull.simplices)
            write_stl(FINGER_COLLISION_DIR / f"finger_{finger}_{k}.stl", points[used], faces)
        print(f"finger {finger}: {FINGER_SLABS} pieces, {total * 1e6:.1f} cm3 (single hull {hull_volume * 1e6:.1f} cm3)")


if __name__ == "__main__":
    main()
