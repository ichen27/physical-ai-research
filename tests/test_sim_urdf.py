import unittest
import xml.etree.ElementTree as ET

from understudy import arm, sim_urdf
from understudy.kinematics import DEFAULT_URDF, Chain


class SimUrdfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = sim_urdf.build(DEFAULT_URDF.read_text())
        cls.root = ET.fromstring(cls.text)

    def test_committed_file_is_up_to_date(self):
        self.assertEqual(
            sim_urdf.SIM_URDF.read_text(), self.text, "run `python -m understudy.sim_urdf` and commit the result"
        )

    def test_arm_joint_limits_match_the_arm_description(self):
        limits = {j.get("name"): j.find("limit") for j in self.root.findall("joint") if j.find("limit") is not None}
        for joint in arm.ARM_JOINTS:
            self.assertEqual(float(limits[joint.name].get("lower")), joint.lower)
            self.assertEqual(float(limits[joint.name].get("upper")), joint.upper)
        for name, (lower, upper) in arm.GRIPPER_LIMITS.items():
            self.assertEqual((float(limits[name].get("lower")), float(limits[name].get("upper"))), (lower, upper))

    def test_every_moving_link_has_mass(self):
        children = {j.find("child").get("link") for j in self.root.findall("joint")}
        for link in self.root.findall("link"):
            if link.get("name") in children:
                mass = link.find("inertial/mass")
                self.assertIsNotNone(mass, f"{link.get('name')} has no mass")
                self.assertGreater(float(mass.get("value")), 0.0)

    def test_marker_mass_is_negligible(self):
        masses = {link.get("name"): float(link.find("inertial/mass").get("value")) for link in self.root.findall("link") if link.find("inertial") is not None}
        self.assertEqual(masses[arm.END_EFFECTOR_LINK], sim_urdf.MARKER_MASS_KG)
        self.assertLess(masses[arm.END_EFFECTOR_LINK], 0.001 * sum(masses.values()))

    def test_geometry_is_unchanged(self):
        original, generated = Chain(DEFAULT_URDF), Chain(sim_urdf.SIM_URDF)
        for pose in ([0.0] * 5, [0.5, -0.4, 0.8, 0.3, 0.2]):
            self.assertEqual(original.tip_position(pose), generated.tip_position(pose))

    def test_fingers_collide_as_convex_pieces_that_exist(self):
        links = {link.get("name"): link for link in self.root.findall("link")}
        for finger, link_name in sim_urdf.FINGER_LINKS.items():
            pieces = sim_urdf.finger_collision_files(finger)
            self.assertEqual(len(pieces), sim_urdf.FINGER_SLABS)
            names = [c.find("geometry/mesh").get("filename") for c in links[link_name].findall("collision")]
            self.assertEqual(
                names, [f"package://{sim_urdf.GENERATED_PACKAGE}/collision/{piece.name}" for piece in pieces]
            )
            for piece in pieces:
                self.assertTrue(piece.is_file(), f"{piece} is missing; run scripts/build_finger_collision.py")
                self.assertFalse(piece.read_bytes().startswith(b"solid"))  # binary STL, not ASCII

    def test_finger_pieces_leave_the_hollow_between_the_claws(self):
        """A ball cupped between the closed-around claws must not overlap any collision piece."""
        import math
        import struct

        ball_centre = (0.0295 + 0.048, -0.0016)  # in the wrist link's frame, 48 mm out from the finger pivots
        pivots = {1: (0.0295, 0.0075, 1), 2: (0.0295, -0.0108, -1)}
        angle = 0.36
        clearance = float("inf")
        for finger, (px, py, sign) in pivots.items():
            cosine, sine = math.cos(sign * angle), math.sin(sign * angle)
            for piece in sim_urdf.finger_collision_files(finger):
                data = piece.read_bytes()
                for i in range(struct.unpack("<I", data[80:84])[0]):
                    for k in range(3):
                        x, y, _ = struct.unpack("<3f", data[84 + i * 50 + 12 + k * 12 : 84 + i * 50 + 24 + k * 12])
                        wx, wy = px + cosine * x - sine * y, py + sine * x + cosine * y
                        clearance = min(clearance, math.hypot(wx - ball_centre[0], wy - ball_centre[1]))
        # With solid convex-hull fingers a 21 mm ball could not be here at all.
        self.assertGreater(clearance, 0.018)

    def test_rejects_unknown_joint_layout(self):
        with self.assertRaises(ValueError):
            sim_urdf.build('<robot name="r"><link name="a"/></robot>')
        with self.assertRaises(ValueError):
            # The joints are present but the finger links are not.
            text = DEFAULT_URDF.read_text().replace('<link name="link6">', '<link name="other6">')
            sim_urdf.build(text)


if __name__ == "__main__":
    unittest.main()
