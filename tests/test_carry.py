import math
import unittest

import sim_rig
from understudy import arm, carry, pickup, runtime
from understudy.camera_collect import MAX_JOINT_SPEED
from understudy.carry import SetDown
from understudy.runner import AIM_LEFT_M, SWIVEL_AXIS_X_M, ArmReading, sensed_from_reading

CHAIN = sim_rig.CHAIN
ODD = arm.Calibration((3012.0, 2031.0, 3042.0, 3013.0, 2050.0), (1, -1, 1, -1, 1), 1740.0, -1)
HELD = 0.13  # where the fingers stop on the real foam ball, rad
LIFTED = (0.2, -0.45, 0.45, 1.15, 0.0)  # the ball held about 13 cm up, the arm turned a little to the left


class HoldingArm:
    """Servos that jump to their goals, with a ball in the gripper: told to close, the fingers stop on it."""

    def __init__(self, calibration=ODD, pose=LIFTED, lose_ball_at: int | None = None):
        self.calibration = calibration
        self.positions = calibration.to_ticks(pose) + [calibration.finger_to_ticks(HELD)]
        self.written: list[list[int]] = []
        self.lose_ball_at = lose_ball_at

    def read(self) -> ArmReading:
        return ArmReading(tuple(self.positions), (0,) * 6)

    def write_goals(self, ticks) -> None:
        self.written.append(list(ticks))
        finger = self.calibration.finger_to_rad(ticks[5])
        lost = self.lose_ball_at is not None and len(self.written) >= self.lose_ball_at
        held = 0.0 if lost else HELD
        self.positions = [*ticks[:5], self.calibration.finger_to_ticks(max(finger, held))]

    def poses(self) -> list[list[float]]:
        return [self.calibration.to_rad(ticks[:5]) for ticks in self.written]

    def fingers(self) -> list[float]:
        return [self.calibration.finger_to_rad(ticks[5]) for ticks in self.written]


BOWL = SetDown(math.radians(88), 0.24, rim_m=0.065, opening_m=0.15)  # a cereal bowl at the arm's left


class SetDownPoseTests(unittest.TestCase):
    def test_the_ball_is_held_just_above_the_desk_at_the_place(self):
        for bearing in (-80.0, -30.0, 45.0, 89.0):
            for distance in (0.14, 0.2, 0.26):
                place = SetDown(math.radians(bearing), distance)
                over, pose = carry.set_down_poses(place, CHAIN)
                ball = carry.grasp_point(pose, CHAIN)
                with self.subTest(bearing=bearing, distance=distance):
                    self.assertLess(math.dist(ball[:2], place.xy), 2e-4)
                    self.assertAlmostEqual(ball[2], pickup.BALL_RADIUS_M + carry.RELEASE_ABOVE_M, delta=2e-4)
                    self.assertGreater(CHAIN.tip_position(list(pose))[2], 0.004)  # finger tips off the desk
                    for angle, joint in zip(pose, arm.ARM_JOINTS):
                        self.assertTrue(joint.lower <= angle <= joint.upper)
                    # It gets there from straight above.
                    above = carry.grasp_point(over, CHAIN)
                    self.assertLess(math.dist(above[:2], ball[:2]), 3e-4)
                    self.assertAlmostEqual(above[2] - ball[2], carry.HOVER_ABOVE_M, delta=3e-4)

    def test_over_a_bowl_the_ball_and_the_fingers_stay_above_the_rim(self):
        for rim in (0.03, 0.065, 0.09):
            for distance in (0.2, 0.26):
                place = SetDown(math.radians(88), distance, rim_m=rim, opening_m=0.15)
                over, pose = carry.set_down_poses(place, CHAIN)
                ball = carry.grasp_point(pose, CHAIN)
                with self.subTest(rim=rim, distance=distance):
                    self.assertLess(math.dist(ball[:2], place.xy), 2e-4)
                    self.assertAlmostEqual(ball[2] - pickup.BALL_RADIUS_M, rim + carry.RELEASE_ABOVE_M, delta=2e-4)
                    self.assertGreater(CHAIN.tip_position(list(pose))[2], rim + 0.01)
                    self.assertGreater(CHAIN.tip_position(list(over))[2], rim + 0.05)

    def test_a_higher_release_is_a_higher_pose(self):
        place = SetDown(0.5, 0.2)
        low = carry.grasp_point(carry.set_down_poses(place, CHAIN, 0.0)[1], CHAIN)[2]
        high = carry.grasp_point(carry.set_down_poses(place, CHAIN, 0.03)[1], CHAIN)[2]
        self.assertAlmostEqual(high - low, 0.03, delta=3e-4)

    def test_let_go_below_the_rim_the_ball_still_comes_in_well_above_it(self):
        over, pose = carry.set_down_poses(BOWL, CHAIN, -0.02)
        self.assertAlmostEqual(carry.grasp_point(pose, CHAIN)[2] - pickup.BALL_RADIUS_M, BOWL.rim_m - 0.02, delta=3e-4)
        self.assertGreater(carry.grasp_point(over, CHAIN)[2] - pickup.BALL_RADIUS_M, BOWL.rim_m + carry.HOVER_ABOVE_M)

    def test_a_place_at_the_end_of_the_swivels_travel_is_allowed_and_one_beyond_it_is_not(self):
        pose = carry.set_down_poses(SetDown(math.radians(90), 0.2), CHAIN)[1]
        self.assertAlmostEqual(pose[0], arm.ARM_JOINTS[0].upper, places=6)
        for bearing in (95.0, -120.0):
            with self.assertRaises(ValueError) as refused:
                carry.set_down_poses(SetDown(math.radians(bearing), 0.2), CHAIN)
            self.assertIn("further than the swivel goes", str(refused.exception))

    def test_a_place_out_of_reach_is_refused(self):
        for distance in (0.03, 0.42, 0.6):
            with self.assertRaises(ValueError):
                carry.set_down_poses(SetDown(0.6, distance), CHAIN)

    def test_a_bowl_too_close_to_the_arm_or_too_narrow_or_half_described_is_refused(self):
        for place, why in (
            (SetDown(1.2, 0.16, rim_m=0.065, opening_m=0.15), "near edge"),
            (SetDown(1.2, 0.22, rim_m=0.065, opening_m=0.05), "too narrow"),
            (SetDown(1.2, 0.22, rim_m=0.065), "both"),
            (SetDown(1.2, 0.22, opening_m=0.15), "both"),
        ):
            with self.subTest(place=place), self.assertRaises(ValueError) as refused:
                carry.set_down_poses(place, CHAIN)
            self.assertIn(why, str(refused.exception))
        carry.set_down_poses(SetDown(1.2, 0.175, rim_m=0.065, opening_m=0.15), CHAIN)  # exactly as near as allowed


class PlaceTests(unittest.TestCase):
    def test_a_place_can_be_given_by_its_middle(self):
        place = SetDown.at((0.05, 0.2), rim_m=0.06, opening_m=0.14)
        self.assertAlmostEqual(place.xy[0], 0.05, places=9)
        self.assertAlmostEqual(place.xy[1], 0.2, places=9)
        self.assertEqual((place.rim_m, place.opening_m), (0.06, 0.14))

    def test_a_bowl_is_described_by_touching_its_middle_and_its_rim(self):
        # The tip held over the middle, at any height, then rested on the rim 7.5 cm from it.
        place = SetDown.from_touches((0.0, 0.24, 0.11), (0.0, 0.165, 0.065))
        self.assertAlmostEqual(place.opening_m, 0.15, places=9)
        self.assertAlmostEqual(place.rim_m, 0.065, places=9)
        self.assertLess(math.dist(place.xy, (0.0, 0.24)), 1e-9)
        self.assertAlmostEqual(math.degrees(place.bearing_rad), 87.3, delta=0.1)  # the swivel axis is a little behind the origin


def ball_needing_turn(degrees: float, distance: float = 0.2) -> tuple[float, float]:
    bearing = math.radians(degrees) + math.asin(AIM_LEFT_M / distance)
    return (SWIVEL_AXIS_X_M + distance * math.cos(bearing), distance * math.sin(bearing))


class InTheWayTests(unittest.TestCase):
    def test_no_container_is_never_in_the_way(self):
        self.assertIsNone(carry.in_the_way(SetDown(0.1, 0.2), (0.2, 0.0)))

    def test_a_ball_in_the_bowl_is_not_picked_up(self):
        self.assertIn("in the container already", carry.in_the_way(BOWL, BOWL.xy))
        self.assertIn("in the container already", carry.in_the_way(BOWL, (BOWL.xy[0] + 0.05, BOWL.xy[1])))

    def test_a_bowl_at_the_left_leaves_most_of_the_sector_free(self):
        for turn in (-57, -30, 0, 30):
            for distance in (0.15, 0.2, 0.24):
                with self.subTest(turn=turn, distance=distance):
                    self.assertIsNone(carry.in_the_way(BOWL, ball_needing_turn(turn, distance)))
        self.assertIn("in the gripper's way", carry.in_the_way(BOWL, ball_needing_turn(50)))

    def test_a_bowl_at_the_right_is_in_the_way_of_the_lift(self):
        # Lifting, the arm swings up to 42 degrees to the right of where it faces.
        right = SetDown(math.radians(-70), 0.22, rim_m=0.065, opening_m=0.15)
        self.assertIn("in the gripper's way", carry.in_the_way(right, ball_needing_turn(-20)))
        self.assertIn("in the gripper's way", carry.in_the_way(right, ball_needing_turn(0)))
        self.assertIsNone(carry.in_the_way(right, ball_needing_turn(40)))
        # The same bowl mirrored to the left is clear of both of those.
        left = SetDown(math.radians(70), 0.22, rim_m=0.065, opening_m=0.15)
        self.assertIsNone(carry.in_the_way(left, ball_needing_turn(0)))
        self.assertIsNone(carry.in_the_way(left, ball_needing_turn(-20)))

    def test_the_facing_can_be_given_when_the_arm_does_not_turn_to_the_ball(self):
        right = SetDown(math.radians(-60), 0.22, rim_m=0.065, opening_m=0.15)
        self.assertIsNotNone(carry.in_the_way(right, (0.2, 0.03), facing_rad=0.0))
        self.assertIsNone(carry.in_the_way(BOWL, (0.2, 0.03), facing_rad=0.0))
        self.assertIsNotNone(carry.in_the_way(BOWL, (0.2, 0.03), facing_rad=math.radians(50)))


class CarryTests(unittest.TestCase):
    PLACE = SetDown(math.radians(-70), 0.2)

    def carried(self, **kwargs):
        held = HoldingArm(**kwargs)
        said = []
        result = carry.carry(held, held.calibration, self.PLACE, advance=lambda: None, chain=CHAIN, say=said.append)
        return held, result, said

    def test_the_ball_is_let_go_at_the_place_and_the_arm_ends_raised_facing_ahead(self):
        held, result, said = self.carried()
        self.assertTrue(result.done)
        self.assertLess(math.dist(result.released_at, self.PLACE.xy), 0.001)
        self.assertIn("ball let go 20.0 cm from the swivel axis, -70 degrees round", said[0])
        self.assertEqual(result.said, "put down")
        end = held.poses()[-1]
        for angle, wanted in zip(end, (0.0, *carry.CARRY_POSE, 0.0)):
            self.assertAlmostEqual(angle, wanted, delta=0.002)
        self.assertAlmostEqual(held.fingers()[-1], pickup.GRIPPER_OPEN_RAD, delta=0.01)

    def test_the_gripper_stays_closed_until_the_ball_is_down_and_then_opens_gradually(self):
        held, _, _ = self.carried()
        fingers, poses = held.fingers(), held.poses()
        down = carry.set_down_poses(self.PLACE, CHAIN)[1]
        first_open = next(i for i, f in enumerate(fingers) if f > 0.02)
        # Not opened before the arm is in the set-down pose...
        for angle, wanted in zip(poses[first_open], down):
            self.assertAlmostEqual(angle, wanted, delta=0.002)
        self.assertTrue(all(f < 0.002 for f in fingers[:first_open]))
        # ...and then over more than a second, not at once.
        fully = next(i for i, f in enumerate(fingers) if f > 0.99 * pickup.GRIPPER_OPEN_RAD)
        self.assertGreater(fully - first_open, arm.CONTROL_HZ)
        self.assertLess(max(b - a for a, b in zip(fingers, fingers[1:])), 0.05)

    def test_no_joint_is_asked_to_turn_faster_than_the_limit(self):
        held, _, _ = self.carried()
        # Goals are whole encoder ticks, so speed is taken over ten steps: one tick in one step is 0.05 rad/s.
        poses = [list(LIFTED)] + held.poses()
        over = 10
        fastest = max(abs(b - a) for before, after in zip(poses, poses[over:]) for a, b in zip(before, after)) * arm.CONTROL_HZ / over
        self.assertLessEqual(fastest, MAX_JOINT_SPEED + 0.006)
        self.assertGreater(fastest, 0.8 * MAX_JOINT_SPEED)  # and it is not needlessly slow either

    def test_the_ball_is_raised_before_it_is_swung_and_stays_clear_of_the_desk_until_lowered(self):
        held, _, _ = self.carried()
        poses = held.poses()
        swinging = [pose for pose in poses if abs(pose[0] - LIFTED[0]) > 0.05 and abs(pose[0] - carry.set_down_poses(self.PLACE, CHAIN)[1][0]) > 0.05]
        self.assertTrue(swinging)
        for pose in swinging:  # while the swivel is on its way, the arm is in the carrying pose
            for angle, wanted in zip(pose[1:4], carry.CARRY_POSE):
                self.assertAlmostEqual(angle, wanted, delta=0.002)
            self.assertGreater(carry.grasp_point(pose, CHAIN)[2], 0.18)
        self.assertGreaterEqual(min(carry.grasp_point(pose, CHAIN)[2] for pose in poses), pickup.BALL_RADIUS_M + carry.RELEASE_ABOVE_M - 0.001)

    def test_nothing_moves_if_the_fingers_hold_nothing(self):
        empty = HoldingArm()
        empty.positions[5] = ODD.finger_to_ticks(0.0)
        result = carry.carry(empty, ODD, self.PLACE, advance=lambda: None, chain=CHAIN, say=lambda _: None)
        self.assertFalse(result.done)
        self.assertIn("closed on nothing", result.said)
        self.assertEqual(empty.written, [])

    def test_nothing_moves_if_the_ball_has_not_been_lifted(self):
        low = carry.set_down_poses(SetDown(0.2, 0.2), CHAIN)[1]  # gripper down at the desk, ball in it
        result = carry.carry(HoldingArm(pose=low), ODD, self.PLACE, advance=lambda: None, chain=CHAIN, say=lambda _: None)
        self.assertFalse(result.done)
        self.assertIn("not been lifted", result.said)

    def test_it_stops_at_once_if_the_ball_is_lost_on_the_way(self):
        held, result, _ = self.carried(lose_ball_at=120)
        self.assertFalse(result.done)
        self.assertIn("the ball was lost", result.said)
        self.assertEqual(len(held.written), 120)  # not one more goal after the fingers closed fully
        self.assertLess(held.fingers()[-1], 0.002)  # and the gripper was not opened
        self.assertGreater(carry.grasp_point(held.poses()[-1], CHAIN)[2], 0.1)  # left raised, not dropped

    def test_into_a_bowl_the_gripper_comes_down_from_above_never_below_the_rim_and_leaves_the_way_it_came(self):
        held = HoldingArm()
        result = carry.carry(held, held.calibration, BOWL, advance=lambda: None, chain=CHAIN, say=lambda _: None)
        self.assertTrue(result.done)
        self.assertEqual(result.said, "let go over the container")
        self.assertLess(math.dist(result.released_at, BOWL.xy), 0.001)
        poses, fingers = held.poses(), held.fingers()
        near = BOWL.opening_m / 2 + 0.06  # anywhere the fingers could be over the bowl or its rim
        over_bowl = [i for i, pose in enumerate(poses) if math.dist(carry.grasp_point(pose, CHAIN)[:2], BOWL.xy) < near]
        self.assertTrue(over_bowl)
        for i in over_bowl:
            self.assertGreater(CHAIN.tip_position(poses[i])[2], BOWL.rim_m + 0.01)
            self.assertGreater(carry.grasp_point(poses[i], CHAIN)[2] - pickup.BALL_RADIUS_M, BOWL.rim_m + carry.RELEASE_ABOVE_M - 0.001)
        # Once it has swivelled to the bowl, the gripper goes down and comes up only straight above the middle.
        swivel = carry.set_down_poses(BOWL, CHAIN)[1][0]
        low = [pose for pose in poses if abs(pose[0] - swivel) < 0.01
               and carry.grasp_point(pose, CHAIN)[2] - pickup.BALL_RADIUS_M < BOWL.rim_m + carry.HOVER_ABOVE_M]
        self.assertTrue(low)
        for pose in low:
            self.assertLess(math.dist(carry.grasp_point(pose, CHAIN)[:2], BOWL.xy), 0.004)  # the joints are eased, so not quite a straight line
        first_open = next(i for i, f in enumerate(fingers) if f > 0.02)
        self.assertAlmostEqual(carry.grasp_point(poses[first_open], CHAIN)[2] - pickup.BALL_RADIUS_M, BOWL.rim_m + carry.RELEASE_ABOVE_M, delta=0.001)

    def test_a_place_out_of_reach_is_refused_before_anything_moves(self):
        held = HoldingArm()
        with self.assertRaises(ValueError):
            carry.carry(held, ODD, SetDown(math.radians(120), 0.2), advance=lambda: None, chain=CHAIN)
        self.assertEqual(held.written, [])

    def test_a_faster_carry_takes_fewer_steps_and_a_bad_speed_is_refused(self):
        slow = HoldingArm()
        carry.carry(slow, ODD, self.PLACE, advance=lambda: None, chain=CHAIN, say=lambda _: None)
        fast = HoldingArm()
        carry.carry(fast, ODD, self.PLACE, advance=lambda: None, chain=CHAIN, max_speed=1.0, say=lambda _: None)
        self.assertLess(len(fast.written), 0.6 * len(slow.written))
        with self.assertRaises(ValueError):
            carry.carry(HoldingArm(), ODD, self.PLACE, advance=lambda: None, chain=CHAIN, max_speed=0.0)


class SimulatedCarryTests(unittest.TestCase):
    """The trained policy lifts the ball in MuJoCo and the carry takes it to the side. No rendering."""

    @classmethod
    def setUpClass(cls):
        try:
            from pathlib import Path

            from understudy import mujoco_arm, sim_bowl

            policy = Path(__file__).resolve().parent.parent / "results" / "pickup" / "2026-11-24-claw" / "policy.onnx"
            cls.policy = runtime.OnnxPolicy(policy)
            cls.bowl = sim_bowl.Bowl(opening_m=BOWL.opening_m, rim_m=BOWL.rim_m, contact_damping=0.15)  # the ball bounces in it
            cls.sim = mujoco_arm.MujocoArm(mujoco_arm.build_model(extras=lambda spec: sim_bowl.add_bowl(spec, cls.bowl)),
                                           command_delay_s=0.015, profile_time_s=arm.VENDOR_PROFILE_TIME_S)
        except ImportError as error:
            raise unittest.SkipTest(f"needs mujoco and onnxruntime: {error}") from error

    def lift(self):
        from understudy.runner import PlacedBall, run_pickup

        self.sim.reset((0.2, 0.03))
        run_pickup(self.sim, self.policy, self.sim.calibration, PlacedBall(0.2, 0.03), [0.19, 0.0, 0.13], 15.0,
                   advance=self.sim.advance, max_speed=0.3, max_lead=0.05)
        self.assertGreater(float(self.sim.ball()[2]), pickup.LIFTED_HEIGHT_M)

    def test_a_lifted_ball_is_carried_and_put_down_on_the_desk(self):
        sim = self.sim
        for bearing in (60.0, -75.0):
            self.lift()
            place = SetDown(math.radians(bearing), 0.2)
            seen = {"lowest_while_held": 9.0, "at_release": None}

            def watch(arm_sim, seen=seen):
                opening = float(arm_sim.data.ctrl[arm_sim._ctrl[5]]) > 0.02
                if opening and seen["at_release"] is None:
                    seen["at_release"] = arm_sim.ball().copy()
                if not opening:
                    seen["lowest_while_held"] = min(seen["lowest_while_held"], float(arm_sim.ball()[2]))

            sim.after_step = watch
            result = carry.carry(sim, sim.calibration, place, sim.advance, say=lambda _: None)
            sim.after_step = None
            with self.subTest(bearing=bearing):
                self.assertTrue(result.done, result.said)
                self.assertLess(math.dist(seen["at_release"][:2], place.xy), 0.015)
                self.assertLess(float(seen["at_release"][2]), pickup.BALL_RADIUS_M + 0.02)  # let go close to the desk
                self.assertGreater(seen["lowest_while_held"], pickup.BALL_RADIUS_M + 0.002)  # never pressed into it
                self.assertLess(float(sim.ball()[2]), pickup.BALL_RADIUS_M + 0.003)  # the ball is on the desk again
                angles, _ = sensed_from_reading(sim.read(), sim.calibration)
                self.assertGreater(angles[5], 0.8)  # gripper open
                self.assertAlmostEqual(angles[0], 0.0, delta=0.05)  # facing ahead again

    def test_a_lifted_ball_is_dropped_into_a_bowl_that_nothing_touches(self):
        from understudy import sim_bowl

        sim = self.sim
        for place, stands in ((BOWL, BOWL.xy), (BOWL, (BOWL.xy[0] + 0.02, BOWL.xy[1] - 0.02))):  # where it was told, and 3 cm off
            self.lift()
            sim_bowl.place_bowl(sim, stands)
            touched = {"arm": False, "held_ball": False}

            def watch(arm_sim, touched=touched):
                arm_touches, ball_touches = sim_bowl.touching(arm_sim)
                touched["arm"] |= arm_touches
                if float(arm_sim.data.ctrl[arm_sim._ctrl[5]]) < 0.02:  # still held
                    touched["held_ball"] |= ball_touches

            sim.after_step = watch
            result = carry.carry(sim, sim.calibration, place, sim.advance, say=lambda _: None)
            sim.after_step = None
            with self.subTest(stands=stands):
                self.assertTrue(result.done, result.said)
                self.assertTrue(sim_bowl.ball_is_in(sim, self.bowl))
                self.assertFalse(touched["arm"])
                self.assertFalse(touched["held_ball"])
                self.assertLess(math.dist(sim_bowl.bowl_xy(sim), stands), 0.001)  # the bowl was not pushed


if __name__ == "__main__":
    unittest.main()
