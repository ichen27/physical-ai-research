import math
import unittest

import numpy as np

import sim_rig
from understudy import arm, eye, pickup, seek, vision
from understudy.runner import in_turned_frame


def desk_with_ball(u: int = 70, v: int = 30, radius: int = 9) -> np.ndarray:
    """A student-sized picture: a grey desk, something black, and a ball the real camera's pale peach."""
    import cv2

    picture = np.full((eye.PICTURE_HEIGHT, eye.PICTURE_WIDTH, 3), 120, np.uint8)
    picture[60:72, 100:128] = 10
    cv2.circle(picture, (u, v), radius, (131, 186, 255), -1, cv2.LINE_AA)  # blue, green, red
    return picture


class ShownTests(unittest.TestCase):
    def test_colour_is_the_picture_itself(self):
        picture = desk_with_ball()
        layers = eye.shown(picture, "colour")
        self.assertEqual(layers.shape, (3, 72, 128))
        self.assertTrue((layers[2] == picture[..., 2]).all())

    def test_the_mask_marks_the_ball_and_the_grey_layer_keeps_the_rest(self):
        picture = desk_with_ball()
        mask = eye.shown(picture, "mask")
        self.assertEqual(mask.shape, (1, 72, 128))
        self.assertGreater(mask[0, 25:35, 65:75].mean(), 250)  # the real camera's peach passes the filter
        self.assertEqual(mask[0, :10, :10].max(), 0)  # the grey desk does not
        both = eye.shown(picture, "mask+grey")
        self.assertEqual(both.shape, (2, 72, 128))
        self.assertTrue((both[0] == mask[0]).all())
        self.assertAlmostEqual(float(both[1, 0, 0]), 120.0)
        self.assertLess(float(both[1, 65, 110]), 20)

    def test_the_filter_is_the_hand_written_detectors_colour_test(self):
        picture = desk_with_ball()
        self.assertTrue((eye.shown(picture, "mask")[0] > 0).tolist() == vision.orange_mask(picture, eye.FILTER).tolist())

    def test_an_unknown_input_is_refused(self):
        with self.assertRaises(ValueError):
            eye.shown(desk_with_ball(), "infrared")

    def test_a_full_size_picture_is_shrunk_and_another_shape_refused(self):
        self.assertEqual(eye.shrink(np.zeros((720, 1280, 3), np.uint8)).shape, (72, 128, 3))
        with self.assertRaises(ValueError):
            eye.shrink(np.zeros((480, 640, 3), np.uint8))


class Looker:
    """Stands in for an eye: always the same answer, and remembers what it was asked."""

    def __init__(self, point=(0.2, 0.03, 0.02), sure=0.9):
        self.point, self.sure, self.asked = point, sure, []

    def glance(self, picture, joint_angles, *more):
        self.asked.append(tuple(joint_angles))
        return self.point, self.sure


class FinderTests(unittest.TestCase):
    def test_a_look_from_a_swivelled_arm_is_shown_as_one_from_straight_ahead_and_the_answer_turned_back(self):
        looker = Looker()
        find = eye.finder(looker, lambda: None, say=lambda _: None)
        angles = (0.5, *seek.LOOK_POSE[1:])
        point = find(angles)
        self.assertEqual(looker.asked[0][0], 0.0)  # the eye is told the arm faces straight ahead
        self.assertEqual(looker.asked[0][1:], tuple(seek.LOOK_POSE[1:]))
        wanted = in_turned_frame((0.2, 0.03, pickup.BALL_RADIUS_M), -0.5)  # what it saw, turned back by the swivel
        for got, want in zip(point, wanted):
            self.assertAlmostEqual(got, want, places=9)
        self.assertGreater(point[1], 0.1)  # a ball seen ahead of an arm turned to the left is to the left

    def test_checked_against_a_second_finder_the_two_must_agree(self):
        looks, said = [], []
        seen = {}

        def second(picture, joint_angles, near):
            seen.update(picture=picture, joint_angles=tuple(joint_angles), near=near)
            return seen["answer"]

        find = eye.finder(Looker(), lambda: "the picture", say=said.append, second=second, looks=looks)
        angles = (0.5, *seek.LOOK_POSE[1:])
        there = in_turned_frame((0.2, 0.03, pickup.BALL_RADIUS_M), -0.5)
        seen["answer"] = (there[0] + 0.006, there[1], there[2])
        self.assertIsNotNone(find(angles))
        self.assertEqual(seen["picture"], "the picture")  # the very picture the eye was shown
        self.assertEqual(seen["joint_angles"], angles)  # in the caller's frame, swivel and all
        self.assertIn("agrees to 6 mm", said[-1])
        seen["answer"] = (there[0] + 0.04, there[1], there[2])
        self.assertIsNone(find(angles))
        self.assertIn("4.0 cm apart", said[-1])
        seen["answer"] = None
        self.assertIsNone(find(angles))
        self.assertIn("hand-written finder sees none", said[-1])
        self.assertEqual(len(looks), 3)
        self.assertEqual(looks[0]["joint_angles"][0], 0.5)
        self.assertEqual(looks[0]["sure"], 0.9)
        self.assertIsNone(looks[2]["other_finder"])
        self.assertAlmostEqual(looks[1]["other_finder"][0] - looks[1]["eye"][0], 0.04, places=4)

    def test_an_eye_that_is_not_sure_finds_nothing_and_says_so(self):
        said = []
        find = eye.finder(Looker(sure=0.3), lambda: None, say=said.append)
        self.assertIsNone(find(seek.LOOK_POSE))
        self.assertIn("not sure", said[0])


class Thinker:
    """Stands in for an exported student: says what it is told to say, step by step."""

    def __init__(self, sure, opens):
        self.sure, self.opens, self.step_number, self.resets = list(sure), list(opens), 0, 0

    def reset(self) -> None:
        self.resets += 1

    def step(self, picture, readings, guess):
        i = min(self.step_number, len(self.sure) - 1)
        self.step_number += 1
        return {"moves": np.full(5, 0.5, np.float32), "open": self.opens[min(i, len(self.opens) - 1)], "ball": (0.2, 0.03, 0.02),
                "seen": (0.2, 0.03, 0.02), "in_view": self.sure[i]}


class StreamDriverTests(unittest.TestCase):
    INTERFACE = __import__("understudy.runtime", fromlist=["Interface"]).Interface("incremental", 0.1, 0.35)
    FRAME = np.zeros((360, 640, 3), np.uint8)

    def driven(self, thinker, frames, **options):
        frames = iter(frames)
        driver = eye.StreamDriver(thinker, lambda: next(frames), (0.2, 0.03), self.INTERFACE, **options)
        return driver

    def run_steps(self, driver, count):
        for _ in range(count):
            self.assertEqual(driver.position([0.0] * 5, 0.9, True), (0.0, 0.0, 0.0))  # the student is never told where the ball is
            action = driver([0.0] * 33)
        return action

    def test_each_step_is_one_picture_one_thought_and_an_action_under_the_speed_limit(self):
        thinker = Thinker([0.9], [True])
        driver = self.driven(thinker, [self.FRAME] * 5)
        action = self.run_steps(driver, 5)
        self.assertEqual(thinker.resets, 1)  # memory cleared at the start of the attempt
        self.assertEqual(len(action), 6)
        self.assertAlmostEqual(action[0] * self.INTERFACE.arm_action_scale, 0.5 * eye.SPEED_LIMIT_RAD_S / arm.CONTROL_HZ, places=7)
        self.assertEqual(action[5], 1.0)
        self.assertEqual(len(driver.thoughts), 5)
        self.assertEqual(driver.pictures[0].shape, (72, 128, 3))
        self.assertEqual(driver.thoughts[0]["sure"], 0.9)

    def test_a_late_picture_is_used_again_and_a_dead_stream_stops_the_run(self):
        driver = self.driven(Thinker([0.9], [True]), [self.FRAME, None, None, self.FRAME] + [None] * 30)
        self.run_steps(driver, 4)
        self.assertEqual([t["picture_steps_old"] for t in driver.thoughts], [0, 1, 2, 0])
        with self.assertRaises(eye.NotSure) as stopped:
            self.run_steps(driver, 30)
        self.assertIn("no new picture", str(stopped.exception))
        self.assertEqual(len(driver.thoughts), 4 + eye.STALE_STEPS)  # ten steps on the old picture, then it stops

    def test_no_picture_at_all_stops_it_at_once(self):
        driver = self.driven(Thinker([0.9], [True]), [None])
        with self.assertRaises(eye.NotSure):
            driver.position([0.0] * 5, 0.9, True)

    def test_it_stops_after_a_run_of_unsure_steps_with_the_gripper_open(self):
        sure = [0.9] * 5 + [0.2] * 60
        driver = self.driven(Thinker(sure, [True]), [self.FRAME] * 100)
        with self.assertRaises(eye.NotSure) as stopped:
            self.run_steps(driver, 100)
        self.assertEqual(len(driver.thoughts), 5 + eye.UNSURE_STEPS)
        self.assertIn("not been sure of the ball for 20 control steps", str(stopped.exception))

    def test_a_short_spell_of_doubt_does_not_stop_it_nor_does_doubt_once_the_gripper_has_closed(self):
        sure = ([0.9] * 5 + [0.2] * (eye.UNSURE_STEPS - 1)) * 3
        self.run_steps(self.driven(Thinker(sure, [True]), [self.FRAME] * 100), len(sure))
        # Closing on the ball hides it: the student is not expected to see it then.
        closed = self.driven(Thinker([0.9] * 5 + [0.1] * 80, [True] * 5 + [False] * 80), [self.FRAME] * 100)
        self.run_steps(closed, 85)


class SeekWithAFinderTests(unittest.TestCase):
    def test_the_arm_sweeps_until_the_finder_sees_the_ball_and_reports_it_in_its_own_frame(self):
        calibration = arm.Calibration.stock()
        robot = sim_rig.JumpArm(calibration)
        seen_from = []

        def find(joint_angles, near=None):
            seen_from.append(round(joint_angles[0], 2))
            return (0.18, 0.09, pickup.BALL_RADIUS_M) if len(seen_from) == 3 else None

        where = seek.look_for_ball(robot, calibration, None, None, say=lambda _: None, find=find)
        self.assertEqual(where, (0.18, 0.09))
        self.assertEqual(seen_from, [round(angle, 2) for angle in seek.SCAN[:3]])

    def test_asked_to_confirm_it_stops_if_the_ball_is_not_seen_again_facing_it(self):
        calibration = arm.Calibration.stock()
        answers = iter([(0.18, 0.09, pickup.BALL_RADIUS_M), None])
        robot = sim_rig.JumpArm(calibration)
        with self.assertRaises(ValueError) as stopped:
            seek.face_ball(robot, calibration, None, None, say=lambda _: None, find=lambda angles, near=None: next(answers), confirm=True)
        self.assertIn("not again", str(stopped.exception))
        # Left raised in the look pose, facing where the ball was first seen, not brought down towards it.
        pose = calibration.to_rad(robot.read().positions[:5])
        for angle, wanted in zip(pose[1:4], seek.LOOK_POSE[1:4]):
            self.assertAlmostEqual(angle, wanted, delta=0.01)
        # Without it, the first sighting is used, as the hand-written finder's has always been.
        answers = iter([(0.18, 0.09, pickup.BALL_RADIUS_M), None])
        facing = seek.face_ball(sim_rig.JumpArm(calibration), calibration, None, None, say=lambda _: None,
                                find=lambda angles, near=None: next(answers))
        self.assertIsNotNone(facing)

    def test_without_a_camera_or_a_finder_it_refuses(self):
        calibration = arm.Calibration.stock()
        with self.assertRaises(ValueError):
            seek.look_for_ball(sim_rig.JumpArm(calibration), calibration, None, None)


class TorchAgreementTests(unittest.TestCase):
    """The filter here and the one the student is trained through are the same."""

    def test_the_numpy_and_torch_filters_agree(self):
        try:
            import torch

            from understudy import student
        except ImportError as error:
            raise unittest.SkipTest(f"needs torch: {error}") from error
        rng = np.random.default_rng(2)
        for kind in ("mask", "mask+grey", "colour"):
            net = student.StudentNet(input=kind).eval()
            for _ in range(5):
                picture = desk_with_ball(int(rng.integers(20, 100)), int(rng.integers(15, 55)))
                picture = np.clip(picture.astype(np.int16) + rng.integers(-12, 12, picture.shape), 0, 255).astype(np.uint8)
                theirs = net.shown(torch.from_numpy(picture).permute(2, 0, 1)[None])[0].numpy()
                ours = eye.shown(picture, kind)
                self.assertLess(float(np.abs(ours - theirs).max()), 1e-3, kind)


if __name__ == "__main__":
    unittest.main()
