import unittest

from understudy import arm, pickup, runtime
from understudy.runner import (
    VELOCITY_UNIT_RAD_S,
    ArmReading,
    PlacedBall,
    check_safe,
    goals_from_command,
    run_pickup,
    sensed_from_reading,
)

# A deliberately awkward arm: offset zeros, two reversed joints, reversed gripper.
ODD = arm.Calibration((3012.0, 2031.0, 3042.0, 3013.0, 2050.0), (1, -1, 1, -1, 1), 1740.0, -1)


class FakeArm:
    """Servos that jump straight to their goals."""

    def __init__(self, calibration):
        self.calibration = calibration
        start = runtime.start_command()
        self.positions = goals_from_command(start, calibration)
        self.written = []

    def read(self):
        return ArmReading(tuple(self.positions), (0,) * 6)

    def write_goals(self, ticks):
        self.written.append(list(ticks))
        self.positions = list(ticks)

    def advance(self):
        pass


class ConversionTests(unittest.TestCase):
    def test_start_pose_round_trips_through_ticks(self):
        for calibration in (arm.Calibration.stock(), ODD):
            goals = goals_from_command(runtime.start_command(), calibration)
            angles, velocities = sensed_from_reading(ArmReading(tuple(goals), (0,) * 6), calibration)
            for angle, wanted in zip(angles, runtime.SENSED_START):
                self.assertAlmostEqual(angle, wanted, delta=arm.RAD_PER_TICK)
            self.assertEqual(velocities, [0.0] * 6)

    def test_reversed_joints_reverse_velocity(self):
        reading = ArmReading((2048,) * 5 + (1740,), (10, 10, 10, 10, 10, 10))
        _, velocities = sensed_from_reading(reading, ODD)
        expected = [10 * VELOCITY_UNIT_RAD_S * s for s in (1, -1, 1, -1, 1, -1)]
        for value, wanted in zip(velocities, expected):
            self.assertAlmostEqual(value, wanted)

    def test_velocity_unit(self):
        # 0.229 rev/min per count: 262 counts is about one revolution per second.
        self.assertAlmostEqual(262 * VELOCITY_UNIT_RAD_S, 2 * 3.14159265, delta=0.01)

    def test_gripper_commands(self):
        open_goal = goals_from_command(runtime.decode_action([0, 0, 0, 0, 0, 1.0]), ODD)[5]
        closed_goal = goals_from_command(runtime.decode_action([0, 0, 0, 0, 0, -1.0]), ODD)[5]
        self.assertEqual(closed_goal, 1740)
        self.assertEqual(open_goal, arm.rad_to_ticks(pickup.GRIPPER_OPEN_RAD, 1740.0, -1))
        self.assertLess(open_goal, closed_goal)  # reversed gripper opens towards lower ticks

    def test_gripper_needs_calibration(self):
        without = arm.Calibration((2048.0,) * 5, (1,) * 5)
        with self.assertRaises(ValueError):
            goals_from_command(runtime.start_command(), without)

    def test_calibration_file_round_trip(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            ODD.save(path)
            self.assertEqual(arm.Calibration.load(path), ODD)


class SafetyTests(unittest.TestCase):
    def test_stops_outside_trained_range(self):
        check_safe(list(pickup.START_POSE))
        with self.assertRaises(RuntimeError):
            check_safe([0.0, 2.5, 0.0, 0.0, 0.0])

    def test_rejects_untrained_hold_target(self):
        fake = FakeArm(ODD)
        with self.assertRaises(ValueError):
            run_pickup(fake, lambda obs: [0.0] * 6, ODD, PlacedBall(0.2, 0.0), [0.19, 0.0, 0.5], 1.0, fake.advance)


class LoopTests(unittest.TestCase):
    def test_policy_sees_start_pose_and_commands_reach_the_servos(self):
        fake = FakeArm(ODD)
        seen = []

        def policy(observation):
            seen.append(list(observation))
            return [0.1, 0.0, 0.0, 0.0, 0.0, -1.0]

        steps = run_pickup(fake, policy, ODD, PlacedBall(0.2, 0.01), [0.19, 0.0, 0.13], 1.0, fake.advance)
        self.assertEqual(len(steps), 30)
        self.assertEqual(len(fake.written), 30)
        for value in seen[0][0:6]:  # first observation: at the start pose, within encoder resolution
            self.assertAlmostEqual(value, 0.0, delta=arm.RAD_PER_TICK)
        self.assertEqual(seen[0][22:28], [0.0] * 6)
        self.assertEqual(seen[1][22:28], [0.1, 0.0, 0.0, 0.0, 0.0, -1.0])
        self.assertAlmostEqual(seen[0][12], 0.2)
        wanted = goals_from_command(runtime.decode_action([0.1, 0, 0, 0, 0, -1.0]), ODD)
        self.assertEqual(fake.written[-1], wanted)

    def test_incremental_policy_gets_the_target_lead_and_accumulates_targets(self):
        fake = FakeArm(ODD)
        incremental = runtime.Interface("incremental", 0.1, 0.35)
        seen = []

        def policy(observation):
            seen.append(list(observation))
            return [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        steps = run_pickup(fake, policy, ODD, PlacedBall(0.2, 0.0), [0.19, 0.0, 0.13], 0.1, fake.advance, interface=incremental)
        self.assertEqual(len(seen[0]), 33)
        for value in seen[0][28:33]:  # at the start pose the target is where the joints are
            self.assertAlmostEqual(value, 0.0, delta=arm.RAD_PER_TICK)
        # Each step moves the first joint's target another 0.1 rad on from the last target.
        self.assertAlmostEqual(steps[0].command.arm_targets[0], pickup.START_POSE[0] + 0.1)
        self.assertAlmostEqual(steps[1].command.arm_targets[0], pickup.START_POSE[0] + 0.2)
        self.assertAlmostEqual(steps[2].command.arm_targets[0], pickup.START_POSE[0] + 0.3)

    def test_a_speed_limit_caps_how_fast_targets_move(self):
        fake = FakeArm(ODD)
        incremental = runtime.Interface("incremental", 0.1, 0.35)
        seen = []
        steps = run_pickup(
            fake, lambda obs: [1.0, -1.0, 0.0, 0.0, 0.0, 1.0], ODD, PlacedBall(0.2, 0.0), [0.19, 0.0, 0.13], 0.2,
            fake.advance, interface=incremental, max_speed=0.6, on_step=seen.append,
        )
        per_step = 0.6 / arm.CONTROL_HZ  # far less than the 0.1 rad the policy asked for
        self.assertAlmostEqual(steps[0].command.arm_targets[0], pickup.START_POSE[0] + per_step)
        self.assertAlmostEqual(steps[2].command.arm_targets[0], pickup.START_POSE[0] + 3 * per_step)
        self.assertAlmostEqual(steps[2].command.arm_targets[1], pickup.START_POSE[1] - 3 * per_step)
        self.assertEqual(seen, steps)  # every step is handed over as it happens

    def test_a_lead_limit_can_only_tighten_the_policys_own(self):
        fake = FakeArm(ODD)
        incremental = runtime.Interface("incremental", 0.1, 0.35)

        def run(**limits):
            return run_pickup(fake, lambda obs: [5.0, 0, 0, 0, 0, 1.0], ODD, PlacedBall(0.2, 0.0), [0.19, 0.0, 0.13],
                              0.1, fake.advance, interface=incremental, **limits)

        def furthest_lead(steps):
            return max(step.command.arm_targets[0] - step.joint_angles[0] for step in steps)

        # The policy asks for half a radian a step. Its own limit lets 0.35 through; the tighter one 0.1.
        self.assertAlmostEqual(furthest_lead(run()), 0.35, delta=0.002)
        self.assertAlmostEqual(furthest_lead(run(max_lead=0.1)), 0.1, delta=0.002)
        with self.assertRaises(ValueError):
            run(max_lead=0.5)
        with self.assertRaises(ValueError):
            run(max_speed=0.0)

    def test_dry_run_writes_nothing(self):
        fake = FakeArm(ODD)
        run_pickup(fake, lambda obs: [0.5] * 6, ODD, PlacedBall(0.2, 0.0), [0.19, 0.0, 0.13], 0.5, fake.advance, send=False)
        self.assertEqual(fake.written, [])


class PlacedBallTests(unittest.TestCase):
    def test_follows_the_gripper_only_once_it_holds_something(self):
        ball = PlacedBall(0.2, 0.0)
        pose = list(pickup.START_POSE)
        self.assertEqual(ball.position(pose, 1.0, True), (0.2, 0.0, pickup.BALL_RADIUS_M))
        for _ in range(PlacedBall.SETTLE_STEPS - 1):
            self.assertEqual(ball.position(pose, 0.4, False)[2], pickup.BALL_RADIUS_M)
        held = ball.position(pose, 0.4, False)
        self.assertGreater(held[2], 0.08)  # now reported at the gripper tip
        # Fingers closed all the way: nothing is held, so the ball is still on the desk.
        empty = PlacedBall(0.2, 0.0)
        for _ in range(PlacedBall.SETTLE_STEPS + 5):
            position = empty.position(pose, 0.0, False)
        self.assertEqual(position, (0.2, 0.0, pickup.BALL_RADIUS_M))


if __name__ == "__main__":
    unittest.main()
