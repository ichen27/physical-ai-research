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

    def test_an_eye_that_is_not_sure_finds_nothing_and_says_so(self):
        said = []
        find = eye.finder(Looker(sure=0.3), lambda: None, say=said.append)
        self.assertIsNone(find(seek.LOOK_POSE))
        self.assertIn("not sure", said[0])


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
