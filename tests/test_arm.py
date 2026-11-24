import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from understudy import arm
from understudy.kinematics import Chain, distance


class TickConversionTests(unittest.TestCase):
    def test_stock_zero_is_zero_radians(self):
        self.assertEqual(arm.ticks_to_rad(2048), 0.0)

    def test_quarter_turn(self):
        self.assertAlmostEqual(arm.ticks_to_rad(3072), math.pi / 2)
        self.assertAlmostEqual(arm.ticks_to_rad(3072, sign=-1), -math.pi / 2)

    def test_round_trip_with_offset_and_reversed_direction(self):
        for ticks in (0, 707, 2048, 3358, 4095):
            angle = arm.ticks_to_rad(ticks, zero_tick=1900, sign=-1)
            self.assertEqual(arm.rad_to_ticks(angle, zero_tick=1900, sign=-1), ticks)

    def test_rejects_invalid_sign(self):
        with self.assertRaises(ValueError):
            arm.ticks_to_rad(2048, sign=0)


class LimitTests(unittest.TestCase):
    def test_defaults_are_inside_limits(self):
        for joint in arm.ARM_JOINTS:
            self.assertLess(joint.lower, joint.default)
            self.assertLess(joint.default, joint.upper)

    def test_limits_stay_inside_vendor_servo_limits(self):
        vendor_ticks = {12: (830, 3129), 13: (1024, 3140)}
        for joint in arm.ARM_JOINTS:
            if joint.dxl_id in vendor_ticks:
                low, high = (arm.ticks_to_rad(t) for t in vendor_ticks[joint.dxl_id])
                self.assertGreaterEqual(joint.lower, low)
                self.assertLessEqual(joint.upper, high)

    def test_clamp(self):
        clamped = arm.clamp_to_limits([9.0, -9.0, 0.0, 0.0, 0.0])
        self.assertEqual(clamped[0], arm.ARM_JOINTS[0].upper)
        self.assertEqual(clamped[1], arm.ARM_JOINTS[1].lower)

    def test_clamp_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            arm.clamp_to_limits([0.0, 0.0])


class CalibrationTests(unittest.TestCase):
    def test_load_and_convert(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps({"zero_ticks": [2000, 2048, 2100, 2048, 2048], "signs": [1, -1, 1, 1, -1]}))
            calibration = arm.Calibration.load(path)
        angles = calibration.to_rad([2000, 2048 - 652, 2100, 2048, 2048])
        self.assertAlmostEqual(angles[0], 0.0)
        self.assertAlmostEqual(angles[1], 652 * arm.RAD_PER_TICK)
        self.assertEqual(calibration.to_ticks(angles), [2000, 2048 - 652, 2100, 2048, 2048])

    def test_to_ticks_never_leaves_trained_limits(self):
        ticks = arm.Calibration.stock().to_ticks([10.0, 10.0, 10.0, 10.0, 10.0])
        for tick, joint in zip(ticks, arm.ARM_JOINTS):
            self.assertEqual(tick, arm.rad_to_ticks(joint.upper))

    def test_rejects_bad_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps({"zero_ticks": [2048] * 5}))
            with self.assertRaises(ValueError):
                arm.Calibration.load(path)
        with self.assertRaises(ValueError):
            arm.Calibration((2048.0,) * 5, (1, 1, 1, 1, 2))
        with self.assertRaises(ValueError):
            arm.Calibration((2048.0,) * 4, (1,) * 4)


class KinematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chain = Chain()

    def test_zero_pose_matches_urdf_offsets(self):
        # Sum of the joint origins along the chain with every joint at zero.
        x, y, z = self.chain.tip_position([0.0] * 5)
        self.assertAlmostEqual(x, -0.01125 + 0.0415 + 0.162 + 0.0287 + 0.09193, places=6)
        self.assertAlmostEqual(y, -0.0016, places=6)
        self.assertAlmostEqual(z, 0.034 + 0.0635 + 0.11315, places=6)

    def test_base_rotation_swings_the_tip_sideways(self):
        x0, _, z0 = self.chain.tip_position([0.0] * 5)
        x, y, z = self.chain.tip_position([math.pi / 2, 0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(z, z0, places=6)
        self.assertGreater(y, 0.25)
        self.assertLess(abs(x), 0.05)
        self.assertGreater(x0, 0.25)

    def test_wrist_roll_does_not_move_the_tip_far(self):
        a = self.chain.tip_position([0.0, 0.0, 0.0, 0.0, 0.0])
        b = self.chain.tip_position([0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertLess(distance(a, b), 0.005)

    def test_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            self.chain.tip_position([0.0] * 6)

    def test_rejects_urdf_with_entity_declarations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.urdf"
            path.write_text('<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "b">]><robot name="r"/>')
            with self.assertRaises(ValueError):
                Chain(path)

    def test_default_pose_is_inside_the_target_box(self):
        tip = self.chain.tip_position([j.default for j in arm.ARM_JOINTS])
        for value, axis in zip(tip, "xyz"):
            low, high = arm.TARGET_BOX[axis]
            self.assertTrue(low <= value <= high, f"{axis}={value:.3f} outside {low}..{high}")

    def test_target_box_is_reachable_within_joint_limits(self):
        rng = random.Random(0)
        tips = [
            self.chain.tip_position([rng.uniform(j.lower, j.upper) for j in arm.ARM_JOINTS]) for _ in range(40000)
        ]
        box = arm.TARGET_BOX
        targets = [(x, y, z) for x in box["x"] for y in box["y"] for z in box["z"]]
        targets.append(tuple(sum(box[a]) / 2 for a in "xyz"))
        for target in targets:
            nearest = min(distance(tip, target) for tip in tips)
            self.assertLess(nearest, arm.SUCCESS_RADIUS_M, f"target {target} is {nearest:.3f} m from any sampled pose")


if __name__ == "__main__":
    unittest.main()
