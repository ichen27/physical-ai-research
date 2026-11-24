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

    def test_rejects_unknown_joint_layout(self):
        with self.assertRaises(ValueError):
            sim_urdf.build('<robot name="r"><link name="a"/></robot>')


if __name__ == "__main__":
    unittest.main()
