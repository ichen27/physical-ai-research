"""The servo values in understudy.arm must match the vendor's MuJoCo model."""

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from understudy import arm

MJCF = Path(__file__).resolve().parent.parent / "assets" / "open_manipulator_description" / "mujoco" / "omx"


class VendorModelFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = ET.fromstring((MJCF / "omx.xml").read_text())
        cls.classes = {
            default.get("class"): default.find("position") for default in cls.root.find("default").findall("default")
        }
        cls.joints = {joint.get("name"): joint for joint in cls.root.iter("joint") if joint.get("axis")}

    def test_position_gain_and_force_limits(self):
        for joint in arm.ARM_JOINTS:
            servo = self.classes[f"{joint.servo}-T"]
            self.assertEqual(float(servo.get("kp")), joint.stiffness)
            self.assertEqual(float(servo.get("forcerange").split()[1]), joint.effort_limit)

    def test_passive_damping_and_rotor_inertia(self):
        for joint in arm.ARM_JOINTS:
            element = self.joints[joint.name]
            self.assertEqual(float(element.get("damping")), joint.passive_damping)
            self.assertEqual(float(element.get("armature")), joint.armature)
            self.assertEqual(float(element.get("frictionloss")), arm.VENDOR_JOINT_FRICTION_LOSS)

    def test_gripper(self):
        finger = self.joints[arm.GRIPPER_JOINT_NAMES[0]]
        self.assertEqual(float(finger.get("damping")), arm.GRIPPER_PASSIVE_DAMPING)
        self.assertEqual(float(finger.get("armature")), arm.GRIPPER_ARMATURE)
        low, high = (float(v) for v in finger.get("range").split())
        self.assertEqual((low, high), arm.GRIPPER_LIMITS[arm.GRIPPER_JOINT_NAMES[0]])
        actuator = next(a for a in self.root.find("actuator") if a.get("name") == arm.GRIPPER_JOINT_NAMES[0])
        self.assertEqual(float(actuator.get("forcerange").split()[1]), arm.GRIPPER_EFFORT_LIMIT)


class CompiledVendorModelTests(unittest.TestCase):
    """Values MuJoCo derives when it compiles the vendor model, such as the velocity gains."""

    @classmethod
    def setUpClass(cls):
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover
            raise unittest.SkipTest("mujoco is not installed") from exc
        cls.mujoco = mujoco
        cls.model = mujoco.MjModel.from_xml_path(str(MJCF / "scene.xml"))

    def _actuator(self, name):
        return self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def test_velocity_gains(self):
        # MuJoCo derives these from the damping ratio in the file, and the result differs between
        # MuJoCo releases by up to about 18%. requirements-dev.txt pins the release these came from.
        for joint in arm.ARM_JOINTS:
            index = self._actuator(joint.name)
            self.assertAlmostEqual(-self.model.actuator_biasprm[index, 2], joint.drive_damping, delta=0.02)
            self.assertAlmostEqual(self.model.actuator_gainprm[index, 0], joint.stiffness, places=6)
        index = self._actuator(arm.GRIPPER_JOINT_NAMES[0])
        self.assertAlmostEqual(-self.model.actuator_biasprm[index, 2], arm.GRIPPER_DRIVE_DAMPING, delta=0.02)
        self.assertAlmostEqual(self.model.actuator_gainprm[index, 0], arm.GRIPPER_STIFFNESS, places=6)

    def test_arm_mass(self):
        self.assertAlmostEqual(float(self.model.body_mass.sum()), 0.5588, places=3)


if __name__ == "__main__":
    unittest.main()
