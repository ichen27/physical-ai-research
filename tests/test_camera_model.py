import dataclasses
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from understudy import pickup
from understudy.camera_model import WristCamera, rotation_from_vector, vector_from_rotation
from understudy.kinematics import Chain

CHAIN = Chain()
PLAIN = WristCamera.looking_along_gripper(tilt_down_rad=0.2)
# A wide lens that bends straight lines, a little off-centre, mounted a little to one side.
WIDE = dataclasses.replace(PLAIN, focal_px=540.0, centre_px=(655.0, 348.0), distortion=-0.18, mount_xyz=(-0.095, 0.006, 0.043))
R = pickup.BALL_RADIUS_M


class RotationTests(unittest.TestCase):
    def test_vector_and_matrix_round_trip(self):
        for vector in ((0.0, 0.0, 0.0), (0.3, -0.2, 0.9), (1.2, 1.2, -1.2), (math.pi, 0.0, 0.0), (0.0, 2.2, 2.2)):
            matrix = rotation_from_vector(vector)
            np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(rotation_from_vector(vector_from_rotation(matrix)), matrix, atol=1e-9)


class GeometryTests(unittest.TestCase):
    def test_guess_looks_along_the_gripper(self):
        level = WristCamera.looking_along_gripper()
        rotation, position = level.pose([0.0] * 5, CHAIN)  # arm straight out, gripper level
        np.testing.assert_allclose(rotation[:, 2], [1, 0, 0], atol=1e-9)  # sight forward
        np.testing.assert_allclose(rotation[:, 0], [0, -1, 0], atol=1e-9)  # picture right is the arm's right
        np.testing.assert_allclose(rotation[:, 1], [0, 0, -1], atol=1e-9)  # picture down is down
        tip = CHAIN.tip_position([0.0] * 5)
        np.testing.assert_allclose(position, [tip[0] - 0.08, tip[1], tip[2] + 0.05], atol=1e-9)

    def test_a_ball_ahead_of_the_gripper_is_in_view_from_the_start_pose(self):
        sight = PLAIN.sight((0.21, 0.0, R), pickup.START_POSE, CHAIN)
        self.assertTrue(sight.inside)
        self.assertGreater(sight.radius_px, 30)

    def test_a_ball_to_the_arms_left_is_on_the_left_of_the_picture(self):
        middle = PLAIN.sight((0.21, 0.0, R), pickup.START_POSE, CHAIN)
        left = PLAIN.sight((0.21, 0.05, R), pickup.START_POSE, CHAIN)
        further = PLAIN.sight((0.25, 0.0, R), pickup.START_POSE, CHAIN)
        self.assertLess(left.u, middle.u - 50)
        self.assertLess(further.v, middle.v - 30)  # further away is higher in the picture
        self.assertLess(further.radius_px, middle.radius_px)

    def test_locate_undoes_sight(self):
        poses = (pickup.START_POSE, (0.3, -0.8, 0.6, 1.0, 0.4), (-0.25, -0.45, 0.4, 1.4, -0.5))
        for camera in (PLAIN, WIDE):
            for pose in poses:
                for ball in ((0.21, 0.0), (0.17, -0.07), (0.26, 0.08), (0.14, 0.02)):
                    sight = camera.sight((*ball, R), pose, CHAIN)
                    if sight is None or not sight.inside:
                        continue
                    back = camera.locate(sight.u, sight.v, pose, CHAIN)
                    with self.subTest(pose=pose, ball=ball):
                        self.assertLess(math.dist(back[:2], ball), 2e-5)
                        self.assertEqual(back[2], R)

    def test_the_ball_appears_as_wide_as_geometry_says(self):
        # Straight ahead of the camera at distance d, a ball's outline is at the angle asin(R / d).
        rotation, position = PLAIN.pose(pickup.START_POSE, CHAIN)
        for distance in (0.12, 0.2, 0.3):
            ball = position + distance * rotation[:, 2]
            sight = PLAIN.sight(ball, pickup.START_POSE, CHAIN)
            self.assertAlmostEqual(sight.u, PLAIN.centre_px[0], places=6)
            self.assertAlmostEqual(sight.radius_px, PLAIN.focal_px * math.tan(math.asin(R / distance)), places=6)

    def test_nothing_is_seen_behind_the_camera_or_above_the_horizon(self):
        rotation, position = PLAIN.pose(pickup.START_POSE, CHAIN)
        self.assertIsNone(PLAIN.sight(position - 0.2 * rotation[:, 2], pickup.START_POSE, CHAIN))
        level = WristCamera.looking_along_gripper()
        self.assertIsNone(level.locate(640, 100, [0.0] * 5, CHAIN))  # above the middle of a level camera: sky
        self.assertIsNotNone(level.locate(640, 700, [0.0] * 5, CHAIN))

    def test_resized_camera_sees_the_same_place(self):
        half = WIDE.resized(640, 360)
        sight = WIDE.sight((0.2, 0.03, R), pickup.START_POSE, CHAIN)
        small = half.sight((0.2, 0.03, R), pickup.START_POSE, CHAIN)
        self.assertAlmostEqual(small.u * 2, sight.u, places=6)
        self.assertAlmostEqual(small.radius_px * 2, sight.radius_px, places=6)
        with self.assertRaises(ValueError):
            WIDE.resized(640, 480)


class FileTests(unittest.TestCase):
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "camera.json"
            WIDE.save(path, desk_rms_mm=1.2)
            self.assertEqual(WristCamera.load(path), WIDE)
            path.write_text("{}")
            with self.assertRaises(ValueError):
                WristCamera.load(path)

    def test_refuses_a_camera_with_no_size(self):
        with self.assertRaises(ValueError):
            dataclasses.replace(PLAIN, focal_px=0.0)


if __name__ == "__main__":
    unittest.main()
