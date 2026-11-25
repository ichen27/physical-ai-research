import dataclasses
import math
import unittest

import sim_rig
from sim_rig import JumpArm, painted
from understudy import arm, pickup, runner, seek, vision
from understudy.camera_model import WristCamera
from understudy.runner import CameraBall, PlacedBall, SwivelLimit, pick_ball

CAMERA = dataclasses.replace(
    WristCamera.looking_along_gripper(focal_px=540.0, back_m=0.06, up_m=0.035, tilt_down_rad=0.45),
    mount_xyz=(-0.06, 0.004, 0.035),
)
START = list(pickup.START_POSE)
R = pickup.BALL_RADIUS_M
ODD = arm.Calibration((3012.0, 2031.0, 3042.0, 3013.0, 2050.0), (1, -1, 1, -1, 1), 1740.0, -1)


def blob_for(ball, angles=START, scale=1.0, whole=True) -> vision.Blob:
    """The patch a ball at ``ball`` makes in the picture."""
    sight = CAMERA.sight((ball[0], ball[1], R), angles, sim_rig.CHAIN)
    radius = sight.radius_px * scale
    return vision.Blob(sight.u, sight.v, radius, 1.0 if whole else 0.5, not whole, math.pi * radius * radius)


class PickBallTests(unittest.TestCase):
    def test_places_a_whole_ball(self):
        point = pick_ball(CAMERA, [blob_for((0.21, 0.02))], START, sim_rig.CHAIN)
        self.assertLess(math.dist(point[:2], (0.21, 0.02)), 1e-4)
        self.assertEqual(point[2], R)

    def test_a_patch_the_wrong_size_is_not_the_ball(self):
        # Something small and orange lying on the desk, or something large and orange further off.
        for scale in (0.5, 1.6):
            self.assertIsNone(pick_ball(CAMERA, [blob_for((0.21, 0.02), scale=scale)], START, sim_rig.CHAIN))

    def test_a_cut_off_or_half_hidden_patch_is_not_used(self):
        self.assertIsNone(pick_ball(CAMERA, [blob_for((0.21, 0.02), whole=False)], START, sim_rig.CHAIN))

    def test_a_patch_too_near_the_base_is_not_the_ball(self):
        near = (0.05, 0.0)
        sight = CAMERA.sight((*near, R), START, sim_rig.CHAIN)
        blob = vision.Blob(sight.u, sight.v, sight.radius_px, 1.0, False, 1.0)
        self.assertIsNone(pick_ball(CAMERA, [blob], START, sim_rig.CHAIN))

    def test_of_two_balls_the_one_nearest_the_last_position_is_taken(self):
        blobs = [blob_for((0.24, -0.05)), blob_for((0.19, 0.04))]
        point = pick_ball(CAMERA, blobs, START, sim_rig.CHAIN, near=(0.19, 0.05))
        self.assertLess(math.dist(point[:2], (0.19, 0.04)), 1e-4)

    def test_found_in_a_painted_picture_by_the_detector(self):
        picture = painted(CAMERA, START, [(0.2, -0.03)], extras=[(1064, 600, 14, (121, 106, 196)), (530, 716, 10, (30, 118, 240))])
        point = pick_ball(CAMERA, vision.find_blobs(picture), START, sim_rig.CHAIN)
        self.assertLess(math.dist(point[:2], (0.2, -0.03)), 0.0015)


class CameraBallTests(unittest.TestCase):
    def source(self, looks, first=None, latency_steps=0):
        looks = iter(looks)
        return CameraBall(CAMERA, lambda: next(looks), first, latency_steps=latency_steps)

    def test_follows_the_ball_while_it_is_seen(self):
        source = self.source([[blob_for((0.21, 0.0))], [blob_for((0.22, 0.01))]])
        self.assertLess(math.dist(source.position(START, 1.0, True)[:2], (0.21, 0.0)), 1e-4)
        self.assertLess(math.dist(source.position(START, 1.0, True)[:2], (0.22, 0.01)), 1e-4)
        self.assertEqual((source.steps_seen, source.steps_kept, source.state), (2, 0, "seen"))

    def test_keeps_the_last_position_when_the_ball_is_not_seen(self):
        # No new picture, an empty picture, a ball cut off by the edge, and one hidden by the fingers.
        looks = [[blob_for((0.21, 0.0))], None, [], [blob_for((0.25, 0.05), whole=False)], [blob_for((0.21, 0.0), scale=0.4)]]
        source = self.source(looks)
        positions = [source.position(START, 1.0, True) for _ in looks]
        for position in positions:
            self.assertLess(math.dist(position[:2], (0.21, 0.0)), 1e-4)
        self.assertEqual((source.steps_seen, source.steps_kept, source.state), (1, 4, "kept"))

    def test_starts_from_the_first_look_when_the_ball_is_out_of_view(self):
        source = self.source([[], []], first=(0.2, 0.03))
        self.assertEqual(source.position(START, 1.0, True), (0.2, 0.03, R))

    def test_stops_if_the_ball_was_never_seen(self):
        with self.assertRaises(RuntimeError):
            self.source([[]]).position(START, 1.0, True)

    def test_ignores_the_camera_once_the_gripper_closes_and_reports_the_ball_held(self):
        elsewhere = [blob_for((0.25, 0.06))]
        source = self.source([[blob_for((0.21, 0.0))]] + [elsewhere] * 40)
        source.position(START, 1.0, True)
        held = None
        for _ in range(CameraBall(CAMERA, lambda: None, (0, 0))._grip.SETTLE_STEPS + 2):
            held = source.position(START, 0.13, False)  # closed on the foam ball: fingers stop at 0.13 rad
        self.assertEqual(source.state, "held")
        grasp = sim_rig.CHAIN.point_in_tip_frame(START, pickup.GRASP_POINT_FROM_TIP)
        self.assertLess(math.dist(held, grasp), 1e-9)
        self.assertGreater(held[2], 0.08)  # in the raised gripper, not on the desk

    def test_closed_on_nothing_keeps_the_last_position_and_resumes_when_reopened(self):
        source = self.source([[blob_for((0.21, 0.0))], [blob_for((0.23, 0.02))]])
        source.position(START, 1.0, True)
        for _ in range(20):
            self.assertLess(math.dist(source.position(START, 0.0, False)[:2], (0.21, 0.0)), 1e-4)
        self.assertEqual(source.state, "kept")
        self.assertLess(math.dist(source.position(START, 1.0, True)[:2], (0.23, 0.02)), 1e-4)

    def test_matches_a_late_picture_to_the_pose_it_was_taken_from(self):
        # The picture arriving now was taken two steps ago, when the wrist was somewhere else.
        earlier = [0.2, -0.6, 0.5, 1.2, 0.0]
        source = self.source([None, None, [blob_for((0.21, 0.02), earlier)]], first=(0.0, 0.3), latency_steps=2)
        source.position(earlier, 1.0, True)
        source.position([0.1, -0.6, 0.5, 1.2, 0.0], 1.0, True)
        position = source.position(START, 1.0, True)
        self.assertLess(math.dist(position[:2], (0.21, 0.02)), 1e-4)

    def test_placed_ball_still_reports_a_held_ball(self):
        placed = PlacedBall(0.21, 0.0)
        for _ in range(PlacedBall.SETTLE_STEPS + 1):
            position = placed.position(START, 0.13, False)
        self.assertGreater(position[2], 0.08)


class TurningTests(unittest.TestCase):
    def test_turning_puts_the_ball_three_centimetres_left_of_straight_ahead(self):
        for ball in ((0.21, 0.0), (0.1, 0.18), (0.15, -0.14), (0.2, 0.06)):
            turn = runner.turn_to_face(ball)
            x, y = runner.in_turned_frame(ball, turn)
            self.assertAlmostEqual(y, runner.AIM_LEFT_M, places=9)
            self.assertAlmostEqual(math.hypot(x - runner.SWIVEL_AXIS_X_M, y),
                                   math.hypot(ball[0] - runner.SWIVEL_AXIS_X_M, ball[1]), places=9)

    def test_a_ball_behind_the_arm_is_refused(self):
        with self.assertRaises(ValueError) as raised:
            runner.turn_to_face((-0.15, 0.1))
        self.assertIn("at most 57", str(raised.exception))
        with self.assertRaises(ValueError):
            runner.turn_to_face((runner.SWIVEL_AXIS_X_M, 0.01))

    def test_a_turned_calibration_reads_the_swivel_from_the_new_straight_ahead(self):
        for calibration in (arm.Calibration.stock(), ODD):
            turned = runner.turned(calibration, 0.4)
            ticks = calibration.to_ticks([0.4, -0.6, 0.5, 1.2, 0.0])
            for got, want in zip(turned.to_rad(ticks), [0.0, -0.6, 0.5, 1.2, 0.0]):
                self.assertAlmostEqual(got, want, delta=0.002)
            self.assertEqual(turned.to_ticks(pickup.START_POSE)[1:], calibration.to_ticks(pickup.START_POSE)[1:])

    def test_the_gripper_is_in_the_same_place_in_both_frames(self):
        turn = -0.6
        pose = [turn + 0.1, -0.5, 0.4, 1.1, 0.2]
        there = sim_rig.CHAIN.tip_position(pose)
        seen = sim_rig.CHAIN.tip_position([pose[0] - turn, *pose[1:]])
        for a, b in zip(runner.in_turned_frame(there, turn), seen):
            self.assertAlmostEqual(a, b, places=9)

    def test_swivel_goals_are_held_to_the_true_limits(self):
        for calibration in (arm.Calibration.stock(), ODD):
            jump = JumpArm(calibration)
            limited = SwivelLimit(jump, calibration)
            beyond = arm.rad_to_ticks(2.2, calibration.zero_ticks[0], calibration.signs[0])
            limited.write_goals([beyond, 2000, 2000, 2000, 2000, 2000])
            self.assertAlmostEqual(calibration.to_rad(jump.positions[:5])[0], arm.ARM_JOINTS[0].upper, delta=0.002)
            self.assertEqual(jump.positions[1:], [2000] * 5)
            self.assertEqual(limited.read(), jump.read())


class SeekTests(unittest.TestCase):
    def seek(self, ball, calibration=ODD, camera=CAMERA):
        jump = JumpArm(calibration)

        def look():
            return vision.find_blobs(painted(camera, jump.angles(), [ball] if ball else []))

        return jump, seek.face_ball(jump, calibration, camera, look, say=lambda _: None)

    def test_faces_a_ball_seen_from_the_start_pose(self):
        jump, facing = self.seek((0.21, -0.04))
        self.assertAlmostEqual(facing.ball[1], runner.AIM_LEFT_M, delta=0.002)
        self.assertTrue(facing.within_reach())
        self.assertLess(facing.turn_rad, 0.0)
        # The arm ends in the start pose of the turned frame, gripper open.
        for angle, start in zip(facing.calibration.to_rad(jump.positions[:5]), pickup.START_POSE):
            self.assertAlmostEqual(angle, start, delta=0.002)
        self.assertEqual(jump.positions[5], ODD.finger_to_ticks(pickup.GRIPPER_OPEN_RAD))

    def test_finds_a_ball_far_round_by_looking_left_and_right(self):
        for bearing in (-0.8, 0.95):
            ball = (runner.SWIVEL_AXIS_X_M + 0.21 * math.cos(bearing), 0.21 * math.sin(bearing))
            # A camera with a narrow view, which cannot see that far round from the start pose.
            narrow = dataclasses.replace(CAMERA, focal_px=1300.0)
            jump, facing = self.seek(ball, camera=narrow)
            with self.subTest(bearing=bearing):
                self.assertIsNotNone(facing)
                self.assertAlmostEqual(facing.distance_m, 0.21, delta=0.003)
                self.assertAlmostEqual(facing.ball[1], runner.AIM_LEFT_M, delta=0.003)
                self.assertGreater(len(jump.moves), 2)  # it had to swivel to look
                # ... and no swivel was faster than the collection speed.
                swivel = 0.0
                for ticks, seconds in jump.moves:
                    angle = ODD.to_rad(ticks[:5])[0]
                    self.assertLessEqual(1.5 * abs(angle - swivel) / seconds, 0.26)
                    swivel = angle

    def test_reports_no_ball_when_there_is_none(self):
        jump, facing = self.seek(None)
        self.assertIsNone(facing)
        self.assertEqual(len(jump.moves), len(seek.SCAN) - 1)

    def test_says_when_the_ball_is_out_of_reach(self):
        _, facing = self.seek((0.14, 0.0))
        self.assertFalse(facing.within_reach())
        self.assertAlmostEqual(facing.distance_m, 0.151, delta=0.004)


if __name__ == "__main__":
    unittest.main()
