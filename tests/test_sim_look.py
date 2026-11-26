"""Varied wrist-camera pictures from the simulator. These render, so they need OpenGL and are skipped without it."""

import math
import random
import unittest

import numpy as np

from understudy import pickup, sim_look
from understudy.kinematics import Chain


class TextureTests(unittest.TestCase):
    def test_every_desk_pattern_makes_a_picture(self):
        for pattern in sim_look.DESK_PATTERNS:
            texture = sim_look.desk_texture(random.Random(1), pattern)
            self.assertEqual(texture.dtype, np.uint8)
            self.assertEqual(texture.shape[2], 3)
        with self.assertRaises(ValueError):
            sim_look.desk_texture(random.Random(1), "marble")

    def test_some_patterns_are_kept_out_of_training(self):
        self.assertTrue(set(sim_look.HELD_OUT_PATTERNS) < set(sim_look.DESK_PATTERNS))
        self.assertGreaterEqual(len(set(sim_look.DESK_PATTERNS) - set(sim_look.HELD_OUT_PATTERNS)), 3)


class ClutterTests(unittest.TestCase):
    def test_clutter_is_kept_out_of_the_arms_way(self):
        ball = (0.2, 0.05)
        self.assertFalse(sim_look._clear_of_reach(0.1, 0.02, ball))  # between the base and the ball
        self.assertFalse(sim_look._clear_of_reach(0.24, 0.06, ball))  # just past the ball
        self.assertFalse(sim_look._clear_of_reach(0.2, 0.12, ball))  # close beside it
        self.assertTrue(sim_look._clear_of_reach(0.2, -0.2, ball))  # well off to the side
        self.assertTrue(sim_look._clear_of_reach(0.45, 0.1, ball))  # well beyond

    def test_things_not_in_use_are_parked_close_by_and_under_the_desk(self):
        for index in range(sim_look.DISTRACTORS):
            x, y, z = sim_look._parked(index)
            self.assertLess(z, -0.05)
            self.assertLess(math.hypot(x, y), 1.0)


class ShrinkTests(unittest.TestCase):
    def test_a_real_picture_is_shrunk_to_the_students_size(self):
        self.assertEqual(sim_look.shrink(np.zeros((720, 1280, 3), np.uint8)).shape, (72, 128, 3))
        self.assertEqual(sim_look.shrink(np.zeros((72, 128, 3), np.uint8)).shape, (72, 128, 3))
        with self.assertRaises(ValueError):
            sim_look.shrink(np.zeros((480, 640, 3), np.uint8))


class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from understudy.mujoco_arm import MujocoArm, build_model

            cls.nominal = sim_look.student_camera()
            cls.sim = MujocoArm(build_model(wrist_camera=cls.nominal.resized(256, 144), extras=sim_look.dress))
            cls.scene = sim_look.Scene(cls.sim, cls.nominal)
        except ImportError as error:
            raise unittest.SkipTest(f"needs mujoco: {error}") from error
        except Exception as error:  # no OpenGL on this machine
            raise unittest.SkipTest(f"cannot render: {error}") from error

    @classmethod
    def tearDownClass(cls):
        cls.scene.close()

    def test_the_camera_sees_things_a_few_centimetres_away(self):
        # A scene made large by things parked far away once pushed this to 15 cm, and the desk
        # vanished from the picture whenever the gripper came down to the ball.
        model = self.sim.model
        self.assertLess(model.vis.map.znear * model.stat.extent, 0.03)

    def test_a_picture_is_the_students_size_and_shows_the_ball(self):
        self.sim.reset((0.21, 0.02))
        self.scene.new_look(random.Random(3), (0.21, 0.02))
        picture, visible = self.scene.picture_now()
        self.assertEqual(picture.shape, (sim_look.PICTURE_HEIGHT, sim_look.PICTURE_WIDTH, 3))
        self.assertEqual(picture.dtype, np.uint8)
        self.assertGreater(visible, 0.9)
        self.assertGreater(picture.std(), 2.0)  # not a blank frame

    def test_the_desk_stays_in_the_picture_with_the_gripper_down_at_the_ball(self):
        import mujoco

        self.sim.reset((0.3, 0.2))
        self.scene.new_look(random.Random(4), (0.3, 0.2))
        low = [0.0, 0.2, 0.35, 1.0, 0.0, pickup.GRIPPER_OPEN_RAD]  # gripper a few centimetres above the desk
        for address, angle in zip(self.sim._qpos, low):
            self.sim.data.qpos[address] = angle
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.assertLess(Chain().tip_position(low[:5])[2], 0.06)
        self.scene._segments.update_scene(self.sim.data, camera=self.scene._camera)
        drawn = self.scene._segments.render()[..., 0]
        self.assertLess(float((drawn < 0).mean()), 0.2)  # most of the picture is something, not empty space

    def test_a_ball_out_of_the_picture_counts_as_not_visible(self):
        self.sim.reset((0.2, 0.4))
        self.scene.new_look(random.Random(5), (0.2, 0.4))
        self.assertLess(self.scene.picture_now()[1], 0.05)

    def test_the_same_seed_gives_the_same_look_and_another_gives_another(self):
        def take(seed):
            self.sim.reset((0.2, 0.0))
            look = self.scene.new_look(random.Random(seed), (0.2, 0.0))
            return look, self.scene.picture_now()[0]

        look_a, picture_a = take(11)
        look_b, picture_b = take(11)
        look_c, picture_c = take(12)
        self.assertEqual(look_a, look_b)
        np.testing.assert_array_equal(picture_a, picture_b)
        self.assertNotEqual(look_a, look_c)
        self.assertGreater(np.abs(picture_a.astype(int) - picture_c.astype(int)).mean(), 2.0)

    def test_the_camera_is_moved_only_a_little_about_its_nominal_mount(self):
        rng = random.Random(6)
        for _ in range(20):
            self.sim.reset((0.2, 0.0))
            camera = self.scene.new_look(rng, (0.2, 0.0)).camera
            self.assertLess(max(abs(a - b) for a, b in zip(camera.mount_xyz, self.nominal.mount_xyz)), sim_look.MOUNT_SHIFT_M + 1e-9)
            self.assertLess(abs(camera.focal_px / self.nominal.focal_px - 1), sim_look.FOCAL_SPREAD + 1e-9)
            self.assertTrue(sim_look.DISTORTION_RANGE[0] <= camera.distortion <= sim_look.DISTORTION_RANGE[1])

    def test_held_out_looks_use_only_the_patterns_kept_back(self):
        rng = random.Random(7)
        seen = set()
        for _ in range(12):
            self.sim.reset((0.2, 0.0))
            seen.add(self.scene.new_look(rng, (0.2, 0.0)).pattern)
        self.assertFalse(seen & set(sim_look.HELD_OUT_PATTERNS))
        self.scene.held_out = True
        try:
            held = {self.scene.new_look(rng, (0.2, 0.0)).pattern for _ in range(12)}
        finally:
            self.scene.held_out = False
        self.assertTrue(held <= set(sim_look.HELD_OUT_PATTERNS))

    def test_late_pictures_arrive_the_looks_number_of_steps_late(self):
        self.sim.reset((0.2, 0.0))
        rng = random.Random(8)
        look = self.scene.new_look(rng, (0.2, 0.0))
        while look.latency_steps != 2:
            look = self.scene.new_look(rng, (0.2, 0.0))
        first = self.scene.picture()[1]
        self.sim.place_ball((0.2, 0.4))  # the ball is taken away
        still_there = [self.scene.picture()[1] for _ in range(2)]
        gone = self.scene.picture()[1]
        self.assertGreater(first, 0.9)
        self.assertGreater(min(still_there), 0.9)  # two steps later the picture arriving still shows it
        self.assertLess(gone, 0.05)


if __name__ == "__main__":
    unittest.main()
