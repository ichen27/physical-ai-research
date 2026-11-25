import unittest

import numpy as np

import ball_pictures
from ball_pictures import CORAL, ORANGE, paint_ball, picture
from understudy import vision


def colour_for(tint) -> vision.BallColour:
    """What a first look at a newly placed ball of this colour teaches the detector."""
    empty = picture("hold1")
    placed = paint_ball(empty, 640, 300, 60, tint)
    return vision.learn_colour(placed, vision.find_new_ball(empty, placed))


class RealPictureTests(unittest.TestCase):
    """Pictures from the first arm's camera, none with a ball in front of the gripper."""

    def test_no_ball_is_reported_where_there_is_none(self):
        for name in ("hold1", "hold4", "arm-camera-1", "rest1"):
            for colour in (vision.BallColour(), colour_for(CORAL), colour_for(ORANGE)):
                with self.subTest(picture=name, colour=colour):
                    self.assertIsNone(vision.find_ball(picture(name), colour))

    def test_the_veil_is_measured(self):
        # The start-pose pictures are hazier than the ones with the gripper on the desk.
        self.assertGreater(vision.veil(picture("hold1")), 30)
        self.assertLess(vision.veil(picture("rest1")), 25)

    def test_red_wires_are_not_the_ball(self):
        # The red wires in the lower right of the start-pose pictures are close to the ball in hue.
        for colour in (colour_for(CORAL), colour_for(ORANGE)):
            for blob in vision.find_blobs(picture("hold1"), colour):
                self.assertFalse(1040 < blob.u < 1090 and 585 < blob.v < 615, blob)


class PaintedBallTests(unittest.TestCase):
    def test_finds_the_ball_across_the_picture(self):
        for tint in (CORAL, ORANGE):
            colour = colour_for(tint)
            for name in ("hold1", "hold4"):
                for u, v, radius in ((640, 300, 60), (300, 200, 45), (1000, 250, 38), (200, 560, 70), (760, 420, 85)):
                    with self.subTest(tint=tint, picture=name, at=(u, v)):
                        found = vision.find_ball(paint_ball(picture(name), u, v, radius, tint), colour)
                        self.assertIsNotNone(found)
                        self.assertLess(abs(found.u - u), 2.5)
                        self.assertLess(abs(found.v - v), 2.5)
                        self.assertLess(abs(found.radius_px - radius), 0.06 * radius)
                        self.assertTrue(found.whole())

    def test_finds_the_ball_in_the_arms_shadow(self):
        # The gripper's shadow lies across the middle of the start-pose pictures.
        for tint in (CORAL, ORANGE):
            colour = colour_for(tint)
            for u, v in ((560, 290), (680, 360), (760, 300)):
                with self.subTest(tint=tint, at=(u, v)):
                    found = vision.find_ball(paint_ball(picture("hold1"), u, v, 55, tint, shade=0.45), colour)
                    self.assertIsNotNone(found)
                    self.assertLess(abs(found.u - u), 3.0)
                    self.assertLess(abs(found.v - v), 3.0)

    def test_a_ball_cut_off_by_the_edge_is_not_whole(self):
        colour = colour_for(CORAL)
        image = paint_ball(picture("hold1"), 640, 700, 60)
        self.assertIsNone(vision.find_ball(image, colour))
        blob = vision.find_blobs(image, colour)[0]
        self.assertTrue(blob.cut_off)
        self.assertFalse(blob.whole())

    def test_a_half_hidden_ball_is_not_whole(self):
        colour = colour_for(CORAL)
        image = paint_ball(picture("hold1"), 640, 300, 60, cover=(640, 200, 760, 400))
        self.assertIsNone(vision.find_ball(image, colour))
        self.assertFalse(vision.find_blobs(image, colour)[0].whole())

    def test_positions_do_not_depend_on_picture_size(self):
        import cv2

        colour = colour_for(CORAL)
        full = paint_ball(picture("hold1"), 500, 260, 64)
        half = cv2.resize(full, (640, 360), interpolation=cv2.INTER_AREA)
        found = vision.find_ball(half, colour).scaled(2.0)
        self.assertLess(abs(found.u - 500), 3.0)
        self.assertLess(abs(found.v - 260), 3.0)
        self.assertLess(abs(found.radius_px - 64), 4.0)


class NewBallTests(unittest.TestCase):
    def test_finds_a_ball_that_was_put_down_without_knowing_its_colour(self):
        empty = picture("hold1")
        for tint in (CORAL, ORANGE):
            for u, v, radius, shade in ((640, 300, 60, 1.0), (350, 220, 48, 1.0), (700, 350, 55, 0.45)):
                with self.subTest(tint=tint, at=(u, v)):
                    found = vision.find_new_ball(empty, paint_ball(empty, u, v, radius, tint, shade))
                    self.assertIsNotNone(found)
                    self.assertLess(abs(found.u - u), 3.0)
                    self.assertLess(abs(found.v - v), 3.0)
                    self.assertLess(abs(found.radius_px - radius), 0.08 * radius)

    def test_nothing_new_when_no_ball_was_put_down(self):
        # Two pictures of the same scene taken seconds apart differ only by camera noise.
        self.assertIsNone(vision.find_new_ball(picture("hold1"), picture("hold4")))

    def test_a_hand_in_the_way_is_not_a_ball(self):
        empty = picture("hold1")
        image = empty.copy()
        image[150:520, 500:1280] = (150, 170, 215)  # a forearm across the picture
        self.assertIsNone(vision.find_new_ball(empty, image))

    def test_learned_colour_covers_the_ball_it_was_learned_from(self):
        empty = picture("hold1")
        placed = paint_ball(empty, 640, 300, 60, ORANGE)
        colour = vision.learn_colour(placed, vision.find_new_ball(empty, placed))
        self.assertLess(colour.hue_low, 20)
        self.assertGreater(colour.hue_high, 30)
        mask = vision.orange_mask(placed, colour)
        self.assertGreater(mask[260:340, 600:680].mean(), 0.98)

    def test_learning_refuses_a_patch_with_no_one_colour(self):
        rng = np.random.default_rng(3)
        noise = rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            vision.learn_colour(noise, vision.Blob(640, 360, 60, 1.0, False, 11310))


class FileTests(unittest.TestCase):
    def test_decodes_a_picture_file(self):
        data = (ball_pictures.FIXTURES / "hold1.jpg").read_bytes()
        self.assertEqual(vision.decode(data).shape, (720, 1280, 3))
        with self.assertRaises(ValueError):
            vision.decode(b"not a picture")


if __name__ == "__main__":
    unittest.main()
