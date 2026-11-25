import dataclasses
import math
import random
import unittest

import sim_rig
from sim_rig import JumpArm, painted
from understudy import arm, camera_calibration, camera_collect, pickup, vision
from understudy.camera_model import WristCamera

# The camera to be found. Not the one the collection starts from.
TRUE = dataclasses.replace(
    WristCamera.looking_along_gripper(focal_px=540.0, back_m=0.06, up_m=0.035, tilt_down_rad=0.45),
    mount_xyz=(-0.06, 0.004, 0.035),
)
ODD = arm.Calibration((3012.0, 2031.0, 3042.0, 3013.0, 2050.0), (1, -1, 1, -1, 1), 1740.0, -1)


class Desk:
    """A simulated arm, a ball and the camera's pictures of it."""

    def __init__(self, ball=None):
        self.arm = JumpArm(ODD)
        self.ball = ball
        self.pictures = 0

    def picture(self):
        self.pictures += 1
        return painted(TRUE, self.arm.angles(), [self.ball] if self.ball else [])


class PoseTests(unittest.TestCase):
    def test_every_small_move_keeps_the_gripper_well_above_the_desk(self):
        poses = camera_collect.small_poses(sim_rig.CHAIN)
        self.assertGreaterEqual(len(poses), camera_calibration.MIN_SIGHTINGS + 2)
        self.assertEqual(poses[0], tuple(pickup.START_POSE))
        for pose in poses:
            self.assertGreaterEqual(sim_rig.CHAIN.tip_position(list(pose))[2], camera_collect.MIN_TIP_HEIGHT_M)

    def test_wide_views_see_the_ball_and_stay_above_the_desk(self):
        poses = camera_collect.wide_poses(TRUE, (0.21, 0.02), 12, sim_rig.CHAIN, random.Random(1))
        self.assertEqual(len(poses), 12)
        for pose in poses:
            self.assertTrue(camera_collect.pose_is_clear(pose, sim_rig.CHAIN))
            self.assertTrue(TRUE.sight((0.21, 0.02, pickup.BALL_RADIUS_M), pose, sim_rig.CHAIN).inside)
        swivels = [pose[0] for pose in poses]
        self.assertGreater(max(swivels) - min(swivels), 0.3)  # spread out, not twelve of the same view

    def test_a_pose_or_path_near_the_desk_is_refused(self):
        low = (0.0, 0.3, 0.4, 0.9, 0.0)
        self.assertFalse(camera_collect.pose_is_clear(low, sim_rig.CHAIN))
        self.assertFalse(camera_collect.path_is_clear(pickup.START_POSE, low, sim_rig.CHAIN))
        beyond = (2.0, *pickup.START_POSE[1:])
        self.assertFalse(camera_collect.pose_is_clear(beyond, sim_rig.CHAIN))  # past the swivel's limit

    def test_moves_are_slow(self):
        seconds = camera_collect.move_seconds(pickup.START_POSE, (0.4, *pickup.START_POSE[1:]))
        self.assertGreaterEqual(0.4 / seconds * 1.5, 0.0)
        self.assertLessEqual(1.5 * 0.4 / seconds, camera_collect.MAX_JOINT_SPEED + 1e-9)
        self.assertEqual(camera_collect.move_seconds(pickup.START_POSE, pickup.START_POSE), camera_collect.MIN_MOVE_SECONDS)

    def test_visiting_order_takes_the_nearest_next(self):
        poses = [(0.3, 0, 0, 0, 0), (0.1, 0, 0, 0, 0), (0.2, 0, 0, 0, 0)]
        self.assertEqual([p[0] for p in camera_collect.in_visiting_order(poses, (0.0,) * 5)], [0.1, 0.2, 0.3])


class ArrivalTests(unittest.TestCase):
    def test_sees_the_ball_arrive_and_learns_its_colour(self):
        desk = Desk()
        empty = desk.picture()
        desk.ball = (0.21, 0.01)
        arrival = camera_collect.wait_for_ball(desk.picture, empty, say=lambda _: None)
        expected = TRUE.sight((0.21, 0.01, pickup.BALL_RADIUS_M), desk.arm.angles(), sim_rig.CHAIN)
        self.assertLess(math.hypot(arrival.blob.u - expected.u, arrival.blob.v - expected.v), 2.0)
        self.assertLess(arrival.colour.hue_low, 24.7)
        self.assertGreater(arrival.colour.hue_high, 24.7)

    def test_waits_while_the_ball_is_still_moving(self):
        desk = Desk()
        empty = desk.picture()
        spots = iter([(0.25, 0.05), (0.23, 0.03), (0.21, 0.01), (0.21, 0.01), (0.21, 0.01)])

        def picture():
            desk.ball = next(spots)
            return desk.picture()

        arrival = camera_collect.wait_for_ball(picture, empty, say=lambda _: None)
        expected = TRUE.sight((0.21, 0.01, pickup.BALL_RADIUS_M), desk.arm.angles(), sim_rig.CHAIN)
        self.assertLess(math.hypot(arrival.blob.u - expected.u, arrival.blob.v - expected.v), 2.0)

    def test_gives_up_when_no_ball_is_put_down(self):
        desk = Desk()
        with self.assertRaises(camera_collect.NoBall):
            camera_collect.wait_for_ball(desk.picture, desk.picture(), attempts=4, say=lambda _: None)
        self.assertEqual(desk.pictures, 5)

    def test_a_ball_moved_to_a_new_spot_counts_as_arriving(self):
        desk = Desk((0.2, -0.04))
        before = desk.picture()
        desk.ball = (0.22, 0.05)
        arrival = camera_collect.wait_for_ball(desk.picture, before, say=lambda _: None)
        expected = TRUE.sight((0.22, 0.05, pickup.BALL_RADIUS_M), desk.arm.angles(), sim_rig.CHAIN)
        self.assertLess(math.hypot(arrival.blob.u - expected.u, arrival.blob.v - expected.v), 2.0)


class PulseTests(unittest.TestCase):
    """A powered arm drops if the computer is silent for half a second, so slow working-out reads the arm as it goes."""

    def test_reads_the_arm_at_intervals_however_often_it_is_called(self):
        import time

        heard = []
        pulse = camera_collect.Pulse(lambda: heard.append(time.monotonic()), every_s=0.02)
        until = time.monotonic() + 0.3
        calls = 0
        while time.monotonic() < until:
            pulse()
            calls += 1
        self.assertGreater(calls, 1000)
        self.assertGreaterEqual(len(heard), 8)
        self.assertLessEqual(len(heard), 16)
        self.assertEqual(pulse.beats, len(heard))
        self.assertLess(max(b - a for a, b in zip(heard, heard[1:])), 0.1)

    def test_without_an_arm_it_does_nothing(self):
        pulse = camera_collect.Pulse(None)
        pulse()
        self.assertEqual(pulse.beats, 0)

    def test_a_fit_never_leaves_the_arm_unheard_for_long(self):
        import time

        desk = Desk((0.21, 0.01))
        collector = camera_collect.Collector(desk.arm, ODD, desk.picture, vision.BallColour(), say=lambda _: None)
        seen = collector.visit(camera_collect.small_poses(sim_rig.CHAIN), 0)
        heard = [time.monotonic()]
        pulse = camera_collect.Pulse(lambda: heard.append(time.monotonic()))
        started = time.monotonic()
        camera_calibration.fit(seen, ODD, TRUE.width, TRUE.height, lens=False, tick=pulse)
        heard.append(time.monotonic())
        took = heard[-1] - started
        self.assertGreater(len(heard), 2 + took / 0.1)  # it did read the arm throughout, not just at the ends
        self.assertLess(max(b - a for a, b in zip(heard, heard[1:])), 0.2)

    def test_choosing_views_never_leaves_the_arm_unheard_for_long(self):
        import time

        heard = [time.monotonic()]
        pulse = camera_collect.Pulse(lambda: heard.append(time.monotonic()), every_s=0.01)
        camera_collect.wide_poses(TRUE, (0.21, 0.02), 12, sim_rig.CHAIN, random.Random(1), tick=pulse)
        heard.append(time.monotonic())
        self.assertLess(max(b - a for a, b in zip(heard, heard[1:])), 0.2)

    def test_collecting_reads_the_arm_while_it_fits_and_chooses_views(self):
        desk = Desk((0.21, 0.01))
        collector = camera_collect.Collector(desk.arm, ODD, desk.picture, vision.BallColour(), say=lambda _: None)
        collector.collect_place(0, TRUE.width, TRUE.height, rng=random.Random(0))
        self.assertGreaterEqual(collector.pulse.beats, 3)


class CollectionTests(unittest.TestCase):
    def collect(self, places, say=lambda _: None):
        desk = Desk()
        before = desk.picture()
        sightings, rough, collector = [], None, None
        for number, ball in enumerate(places):
            desk.ball = ball
            arrival = camera_collect.wait_for_ball(desk.picture, before, say=say)
            collector = collector or camera_collect.Collector(desk.arm, ODD, desk.picture, arrival.colour, say=say)
            sightings += collector.collect_place(number, TRUE.width, TRUE.height, known=rough, rng=random.Random(number))
            rough = camera_calibration.fit(sightings, ODD, lens=False).camera
            before = desk.picture()
        return desk, sightings

    def test_two_places_give_a_camera_good_to_a_millimetre(self):
        desk, sightings = self.collect([(0.21, 0.01), (0.19, 0.06)])
        self.assertGreaterEqual(len(sightings), 30)
        fit = camera_calibration.fit(sightings, ODD)
        self.assertLess(fit.desk_rms_m, 0.001)
        self.assertLess(math.dist(fit.places[0], (0.21, 0.01)), 0.0015)
        self.assertLess(math.dist(fit.places[1], (0.19, 0.06)), 0.0015)
        self.assertLess(math.dist(fit.camera.mount_xyz, TRUE.mount_xyz), 0.004)
        self.assertEqual(camera_collect.moved_places(fit, sightings), [])
        # It ends where it began, and never asked for a move faster than the collection speed.
        for angle, start in zip(desk.arm.angles(), pickup.START_POSE):
            self.assertAlmostEqual(angle, start, delta=0.002)
        previous = list(pickup.START_POSE)
        for ticks, seconds in desk.arm.moves:
            pose = ODD.to_rad(ticks[:5])
            self.assertLessEqual(1.5 * max(abs(a - b) for a, b in zip(pose, previous)) / seconds,
                                 camera_collect.MAX_JOINT_SPEED + 0.01)
            self.assertGreaterEqual(sim_rig.CHAIN.tip_position(pose)[2], camera_collect.MIN_TIP_HEIGHT_M - 0.002)
            self.assertEqual(ticks[5], ODD.finger_to_ticks(pickup.GRIPPER_OPEN_RAD))
            previous = pose

    def test_a_ball_at_the_edge_of_the_picture_is_asked_to_be_moved(self):
        desk = Desk((0.22, 0.17))  # at the picture's left edge from the start pose, out of it after some small moves
        collector = camera_collect.Collector(desk.arm, ODD, desk.picture, vision.BallColour(), say=lambda _: None)
        with self.assertRaises(camera_collect.NoBall) as raised:
            collector.collect_place(0, TRUE.width, TRUE.height)
        self.assertIn("nearer the middle", str(raised.exception))
        for angle, start in zip(desk.arm.angles(), pickup.START_POSE):
            self.assertAlmostEqual(angle, start, delta=0.002)  # back at the start pose all the same

    def test_a_ball_nudged_part_way_through_is_noticed(self):
        desk = Desk((0.21, 0.01))
        collector = camera_collect.Collector(desk.arm, ODD, desk.picture, vision.BallColour(), say=lambda _: None)
        first = collector.visit(camera_collect.small_poses(sim_rig.CHAIN), 0)
        desk.ball = (0.21, 0.05)
        second = collector.visit(camera_collect.small_poses(sim_rig.CHAIN), 0)
        sightings = first + second
        fit = camera_calibration.fit(sightings, ODD)
        self.assertEqual(camera_collect.moved_places(fit, sightings), [0])

    def test_a_picture_taken_while_the_arm_moved_is_dropped(self):
        desk = Desk((0.21, 0.01))

        def picture_that_bumps_the_arm():
            image = desk.picture()
            desk.arm.positions = [desk.arm.positions[0] + 9, *desk.arm.positions[1:]]
            return image

        collector = camera_collect.Collector(desk.arm, ODD, picture_that_bumps_the_arm, vision.BallColour())
        self.assertIsNone(collector.see(0))


if __name__ == "__main__":
    unittest.main()
