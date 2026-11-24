import math
import random
import unittest

from understudy import arm, pickup
from understudy.kinematics import Chain, distance


class PickupDescriptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chain = Chain()

    def test_start_pose_is_inside_limits_and_clear_of_the_ball(self):
        for angle, joint in zip(pickup.START_POSE, arm.ARM_JOINTS):
            self.assertTrue(joint.lower < angle < joint.upper)
        tip_height = self.chain.tip_position(pickup.START_POSE)[2]
        self.assertGreater(tip_height, 2 * pickup.BALL_RADIUS_M + 0.05)

    def test_gripper_commands_are_inside_finger_travel(self):
        low, high = arm.GRIPPER_LIMITS[arm.GRIPPER_JOINT_NAMES[0]]
        self.assertTrue(low <= pickup.GRIPPER_CLOSED_RAD < pickup.GRIPPER_OPEN_RAD <= high)

    def test_open_gripper_is_wider_than_the_ball(self):
        # Gap between the fingers' inner faces 45 mm out from the pivots (see the URDF finger joints and meshes).
        reach, half_thickness, pivot_gap = 0.045, 0.010, 0.0075 + 0.0108
        angle = pickup.GRIPPER_OPEN_RAD
        gap = pivot_gap + 2 * reach * math.sin(angle) - 2 * half_thickness * math.cos(angle)
        self.assertGreater(gap, 2 * pickup.BALL_RADIUS_M + 0.01)

    def test_success_is_stricter_than_lifting(self):
        self.assertGreater(pickup.LIFTED_HEIGHT_M, 2 * pickup.BALL_RADIUS_M)
        self.assertGreater(pickup.HOLD_TARGET["z"][0], pickup.LIFTED_HEIGHT_M + pickup.HOLD_RADIUS_M)

    def test_point_in_tip_frame(self):
        pose = [0.3, -0.2, 0.4, 0.5, 0.1]
        self.assertEqual(self.chain.point_in_tip_frame(pose, (0.0, 0.0, 0.0)), self.chain.tip_position(pose))
        zero = [0.0] * 5
        tip = self.chain.tip_position(zero)
        shifted = self.chain.point_in_tip_frame(zero, (-0.02, 0.0, 0.0))
        self.assertAlmostEqual(shifted[0], tip[0] - 0.02)
        self.assertAlmostEqual(shifted[2], tip[2])

    def test_ball_positions_can_be_grasped_from_above(self):
        """Every corner of the ball area has a gripper-down pose that puts the grasp point on the ball."""
        rng = random.Random(1)
        shoulder, elbow, wrist = arm.ARM_JOINTS[1:4]
        for x in pickup.BALL_SPAWN["x"]:
            for y in pickup.BALL_SPAWN["y"]:
                ball = (x, y, pickup.BALL_RADIUS_M)
                pan = math.atan2(y, x)
                nearest = float("inf")
                for _ in range(8000):
                    q2 = rng.uniform(shoulder.lower, shoulder.upper)
                    q3 = rng.uniform(elbow.lower, elbow.upper)
                    q4 = math.pi / 2 - q2 - q3  # gripper pointing straight down
                    if not wrist.lower <= q4 <= wrist.upper:
                        continue
                    point = self.chain.point_in_tip_frame([pan, q2, q3, q4, 0.0], pickup.GRASP_POINT_FROM_TIP)
                    nearest = min(nearest, distance(point, ball))
                self.assertLess(nearest, 0.01, f"ball at {ball} is {nearest:.3f} m from any downward grasp")


if __name__ == "__main__":
    unittest.main()
