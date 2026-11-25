import unittest

from understudy import arm, pickup, runtime


class ObservationTests(unittest.TestCase):
    def test_layout(self):
        observation = runtime.build_observation(
            joint_angles=list(runtime.SENSED_START),
            joint_velocities=[0.1] * 6,
            ball_position=[0.2, 0.01, 0.021],
            hold_target=[0.18, 0.0, 0.12],
            previous_action=[1, 2, 3, 4, 5, -1],
        )
        self.assertEqual(len(observation), runtime.Interface().observation_size)
        self.assertEqual(observation[0:6], [0.0] * 6)  # at the start pose
        self.assertEqual(observation[6:12], [0.1] * 6)
        self.assertEqual(observation[12:15], [0.2, 0.01, 0.021])
        self.assertEqual(observation[15:22], [0.18, 0.0, 0.12, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(observation[22:28], [1.0, 2.0, 3.0, 4.0, 5.0, -1.0])

    def test_joint_angles_are_relative_to_the_start_pose(self):
        angles = [a + 0.1 for a in runtime.SENSED_START]
        observation = runtime.build_observation(angles, [0] * 6, [0] * 3, [0] * 3, [0] * 6)
        for value in observation[0:6]:
            self.assertAlmostEqual(value, 0.1)

    def test_rejects_wrong_sizes(self):
        with self.assertRaises(ValueError):
            runtime.build_observation([0] * 5, [0] * 6, [0] * 3, [0] * 3, [0] * 6)
        with self.assertRaises(ValueError):
            runtime.build_observation([0] * 6, [0] * 6, [0] * 2, [0] * 3, [0] * 6)


class ActionTests(unittest.TestCase):
    def test_zero_action_holds_the_start_pose_with_the_gripper_open(self):
        command = runtime.decode_action([0.0] * 6)
        self.assertEqual(command.arm_targets, pickup.START_POSE)
        self.assertTrue(command.gripper_open)
        self.assertEqual(command.finger_target, pickup.GRIPPER_OPEN_RAD)

    def test_scale_and_gripper_sign(self):
        command = runtime.decode_action([0.2, 0.0, 0.0, 0.0, 0.0, -0.01])
        self.assertAlmostEqual(command.arm_targets[0], pickup.START_POSE[0] + 0.1)
        self.assertFalse(command.gripper_open)
        self.assertEqual(command.finger_target, pickup.GRIPPER_CLOSED_RAD)

    def test_targets_never_leave_the_trained_limits(self):
        for sign in (1.0, -1.0):
            command = runtime.decode_action([sign * 100.0] * 5 + [1.0])
            for target, joint in zip(command.arm_targets, arm.ARM_JOINTS):
                self.assertTrue(joint.lower <= target <= joint.upper)

    def test_relative_interface_moves_from_the_present_angles(self):
        relative = runtime.Interface("relative", 0.2)
        present = [0.1, -0.5, 0.4, 1.0, 0.0]
        command = runtime.decode_action([1.0, 0.0, -1.0, 0.0, 0.0, 1.0], present, relative)
        self.assertAlmostEqual(command.arm_targets[0], 0.3)
        self.assertAlmostEqual(command.arm_targets[1], -0.5)
        self.assertAlmostEqual(command.arm_targets[2], 0.2)
        # A zero output holds the arm where it is, wherever that is.
        hold = runtime.decode_action([0.0] * 6, present, relative)
        for target, angle in zip(hold.arm_targets, present):
            self.assertAlmostEqual(target, angle)

    def test_relative_interface_needs_the_present_angles_and_respects_limits(self):
        relative = runtime.Interface("relative", 0.2)
        with self.assertRaises(ValueError):
            runtime.decode_action([0.0] * 6, None, relative)
        command = runtime.decode_action([100.0] * 5 + [1.0], [0.0] * 5, relative)
        for target, joint in zip(command.arm_targets, arm.ARM_JOINTS):
            self.assertEqual(target, joint.upper)

    def test_incremental_interface_moves_the_previous_target(self):
        incremental = runtime.Interface("incremental", 0.1, 0.35)
        self.assertEqual(incremental.observation_size, 33)
        present = [0.0, -0.6, 0.5, 1.2, 0.0]
        previous = [0.05, -0.55, 0.5, 1.2, 0.0]
        command = runtime.decode_action([1.0, 0.0, 0.0, -2.0, 0.0, 1.0], present, incremental, previous)
        self.assertAlmostEqual(command.arm_targets[0], 0.15)
        self.assertAlmostEqual(command.arm_targets[1], -0.55)  # zero output leaves the target where it was
        self.assertAlmostEqual(command.arm_targets[3], 1.0)
        # The target may not run more than max_lead ahead of the joint.
        far = runtime.decode_action([10.0, 0.0, 0.0, 0.0, 0.0, 1.0], present, incremental, previous)
        self.assertAlmostEqual(far.arm_targets[0], 0.35)
        with self.assertRaises(ValueError):
            runtime.decode_action([0.0] * 6, present, incremental)
        with self.assertRaises(ValueError):
            runtime.Interface("incremental", 0.1)

    def test_observation_with_target_lead(self):
        observation = runtime.build_observation(
            list(runtime.SENSED_START), [0] * 6, [0.2, 0, 0.02], [0.19, 0, 0.13], [0] * 6, [0.01, 0.02, 0.03, 0.04, 0.05]
        )
        self.assertEqual(len(observation), 33)
        self.assertEqual(observation[28:33], [0.01, 0.02, 0.03, 0.04, 0.05])
        with self.assertRaises(ValueError):
            runtime.build_observation(list(runtime.SENSED_START), [0] * 6, [0] * 3, [0] * 3, [0] * 6, [0.0] * 4)

    def test_interface_file_round_trip_and_default(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            policy = Path(tmp) / "policy.onnx"
            # No file beside the policy: an older, absolute policy.
            self.assertEqual(runtime.Interface.beside(policy), runtime.Interface("absolute", 0.5))
            runtime.Interface("relative", 0.2).save(tmp)
            self.assertEqual(runtime.Interface.beside(policy), runtime.Interface("relative", 0.2))
            runtime.Interface("incremental", 0.1, 0.35).save(tmp)
            self.assertEqual(runtime.Interface.beside(policy), runtime.Interface("incremental", 0.1, 0.35))
        with self.assertRaises(ValueError):
            runtime.Interface("sideways", 0.2)

    def test_start_command(self):
        command = runtime.start_command()
        self.assertEqual(command.arm_targets, pickup.START_POSE)
        self.assertTrue(command.gripper_open)

    def test_rejects_bad_actions(self):
        with self.assertRaises(ValueError):
            runtime.decode_action([0.0] * 5)
        with self.assertRaises(ValueError):
            runtime.decode_action([float("nan")] + [0.0] * 5)


if __name__ == "__main__":
    unittest.main()
