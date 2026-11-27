import dataclasses
import math
import random
import tempfile
import unittest
from pathlib import Path

from understudy import arm, camera_calibration, pickup
from understudy.camera_calibration import Sighting
from understudy.camera_model import WristCamera
from understudy.kinematics import Chain

CHAIN = Chain()
# The camera to be found: a wide lens that bends lines, off-centre, mounted to one side and tipped down.
TRUE = dataclasses.replace(
    WristCamera.looking_along_gripper(focal_px=540.0, tilt_down_rad=0.25),
    centre_px=(655.0, 348.0), distortion=-0.18, mount_xyz=(-0.095, 0.006, 0.043),
)
# An arm whose servo zeros are nowhere near stock, with two joints running backwards.
ODD = arm.Calibration((3012.0, 2031.0, 3042.0, 3013.0, 2050.0), (1, -1, 1, -1, 1), 1740.0, -1)


def sightings(rng: random.Random, ball, count: int, place: int = 0, calibration=ODD, camera=TRUE,
              pixel_noise: float = 1.0, size_noise: float = 0.03, reach: float = 1.0) -> list[Sighting]:
    """Pictures of a ball resting at ``ball``, from random poses that keep the gripper clear of the desk."""
    out = []
    while len(out) < count:
        pose = [
            rng.uniform(-0.35, 0.35) * reach,
            -0.6 + rng.uniform(-0.3, 0.3) * reach,
            0.5 + rng.uniform(-0.3, 0.3) * reach,
            1.2 + rng.uniform(-0.3, 0.3) * reach,
            rng.uniform(-0.5, 0.5) * reach,
        ]
        if CHAIN.tip_position(pose)[2] < 0.085:
            continue
        ticks = tuple(calibration.to_ticks(pose))
        sight = camera.sight((*ball, pickup.BALL_RADIUS_M), calibration.to_rad(ticks), CHAIN)
        if sight is None or not sight.inside:
            continue
        out.append(Sighting(ticks, sight.u + rng.gauss(0, pixel_noise), sight.v + rng.gauss(0, pixel_noise),
                            sight.radius_px * (1 + rng.gauss(0, size_noise)), place))
    return out


class FitTests(unittest.TestCase):
    def test_finds_the_camera_and_the_balls_from_two_placements(self):
        rng = random.Random(1)
        seen = sightings(rng, (0.21, 0.0), 10, 0) + sightings(rng, (0.19, 0.07), 10, 1)
        fit = camera_calibration.fit(seen, ODD)
        self.assertTrue(fit.fitted_lens)
        self.assertFalse(fit.ambiguous())
        self.assertLess(fit.pixel_rms, 2.0)
        self.assertLess(fit.desk_rms_m, 0.002)
        self.assertLess(math.dist(fit.places[0], (0.21, 0.0)), 0.003)
        self.assertLess(math.dist(fit.places[1], (0.19, 0.07)), 0.003)
        self.assertLess(abs(fit.camera.focal_px - TRUE.focal_px), 25)
        self.assertLess(math.dist(fit.camera.mount_xyz, TRUE.mount_xyz), 0.008)

    def test_the_fitted_camera_places_a_ball_it_never_saw(self):
        rng = random.Random(2)
        seen = sightings(rng, (0.21, 0.0), 10, 0) + sightings(rng, (0.19, 0.07), 10, 1)
        camera = camera_calibration.fit(seen, ODD).camera
        for ball in ((0.23, 0.04), (0.17, -0.08), (0.26, -0.03)):
            where, typical, worst = camera_calibration.spread(camera, sightings(rng, ball, 10), ODD)
            with self.subTest(ball=ball):
                self.assertLess(math.dist(where, ball), 0.004)
                self.assertLess(typical, 0.003)
                self.assertLess(worst, 0.006)

    def test_one_placement_is_enough_for_a_usable_camera(self):
        rng = random.Random(3)
        fit = camera_calibration.fit(sightings(rng, (0.21, 0.0), 12), ODD)
        self.assertFalse(fit.fitted_lens)  # too little to go on for the lens, so it is left alone
        where, typical, _ = camera_calibration.spread(fit.camera, sightings(rng, (0.23, 0.04), 10), ODD)
        self.assertLess(math.dist(where, (0.23, 0.04)), 0.008)
        self.assertLess(typical, 0.008)

    def test_small_moves_give_a_camera_good_enough_to_plan_wider_ones(self):
        rng = random.Random(4)
        fit = camera_calibration.fit(sightings(rng, (0.21, 0.01), 8, reach=0.4), ODD)
        where, _, _ = camera_calibration.spread(fit.camera, sightings(rng, (0.19, -0.05), 12), ODD)
        self.assertLess(math.dist(where, (0.19, -0.05)), 0.02)

    def test_a_camera_mounted_another_way_up_is_found(self):
        rng = random.Random(5)
        turned = dataclasses.replace(
            WristCamera.looking_along_gripper(focal_px=700.0, tilt_down_rad=0.5, roll_quarter_turns=2),
            mount_xyz=(-0.06, -0.01, 0.06),
        )
        seen = sightings(rng, (0.21, 0.0), 10, 0, camera=turned) + sightings(rng, (0.2, -0.06), 10, 1, camera=turned)
        fit = camera_calibration.fit(seen, ODD)
        self.assertLess(fit.desk_rms_m, 0.002)
        self.assertLess(abs(fit.camera.focal_px - 700.0), 35)
        self.assertLess(math.dist(fit.camera.mount_xyz, turned.mount_xyz), 0.008)

    def test_a_ball_that_moved_between_pictures_shows_in_the_error(self):
        rng = random.Random(6)
        seen = sightings(rng, (0.21, 0.0), 6) + sightings(rng, (0.21, 0.05), 6)  # nudged 5 cm, recorded as one place
        fit = camera_calibration.fit(seen, ODD)
        self.assertGreater(fit.desk_rms_m, 0.006)

    def test_too_few_sightings_are_refused(self):
        rng = random.Random(7)
        with self.assertRaises(ValueError):
            camera_calibration.fit(sightings(rng, (0.21, 0.0), camera_calibration.MIN_SIGHTINGS - 1), ODD)

    def test_sightings_that_cannot_be_a_ball_on_the_desk_are_refused_or_fit_badly(self):
        rng = random.Random(8)
        junk = [Sighting(s.ticks, rng.uniform(0, 1280), rng.uniform(0, 720), rng.uniform(20, 90), 0)
                for s in sightings(rng, (0.21, 0.0), 12)]
        try:
            fit = camera_calibration.fit(junk, ODD)
        except ValueError:
            return
        self.assertGreater(fit.pixel_rms, 20)


class KnownPlaceTests(unittest.TestCase):
    """An arm whose joint zeros are not quite what its calibration says, which a camera fit alone cannot tell."""

    # Where the joints really are: the wrist's zero is 4 degrees from what the arm's calibration
    # says. Touches of the desk at a few distances barely tell that apart from a true one.
    WRONG = dataclasses.replace(ODD, zero_ticks=tuple(
        zero + sign * math.radians(degrees) / arm.RAD_PER_TICK
        for zero, sign, degrees in zip(ODD.zero_ticks, ODD.signs, (0.0, 0.0, 0.0, 4.0, 0.0))))
    PLACES = ((0.16, 0.0), (0.19, 0.05))

    def seen_through_the_real_arm(self, rng, ball, count, place=0):
        """Sightings recorded as the arm reports them, taken from where the arm really is."""
        out = []
        while len(out) < count:
            pose = [p + rng.uniform(-r, r) for p, r in zip(pickup.START_POSE, (0.35, 0.25, 0.25, 0.25, 0.4))]
            if CHAIN.tip_position(pose)[2] < 0.085:
                continue
            ticks = tuple(ODD.to_ticks(pose))
            sight = TRUE.sight((*ball, pickup.BALL_RADIUS_M), self.WRONG.to_rad(ticks), CHAIN)
            if sight is not None and sight.inside:
                out.append(Sighting(ticks, sight.u + rng.gauss(0, 1), sight.v + rng.gauss(0, 1), sight.radius_px, place))
        return out

    def error_on_new_balls(self, camera, rng):
        errors = []
        for ball in ((0.15, 0.03), (0.2, -0.04), (0.22, 0.02), (0.18, 0.07)):
            for s in self.seen_through_the_real_arm(rng, ball, 3):
                where = camera.locate(s.u, s.v, ODD.to_rad(s.ticks), CHAIN)
                errors.append(math.dist(where[:2], ball))
        return sorted(errors)[len(errors) // 2]

    def setUp(self):
        self.rng = random.Random(12)
        self.seen = [s for place, ball in enumerate(self.PLACES) for s in self.seen_through_the_real_arm(self.rng, ball, 12, place)]

    def test_without_a_known_place_the_fit_agrees_with_itself_and_is_wrong(self):
        fit = camera_calibration.fit(self.seen, ODD, lens=False)
        self.assertLess(fit.desk_rms_m, 0.003)  # nothing in the fit's own error gives it away
        self.assertGreater(math.dist(fit.places[0], self.PLACES[0]), 0.008)
        self.assertGreater(self.error_on_new_balls(fit.camera, self.rng), 0.008)

    def test_known_places_tie_the_camera_to_where_the_ball_really_is(self):
        plain = self.error_on_new_balls(camera_calibration.fit(self.seen, ODD, lens=False).camera, random.Random(1))
        one = camera_calibration.fit(self.seen, ODD, lens=False, known={0: self.PLACES[0]})
        both = camera_calibration.fit(self.seen, ODD, lens=False, known=dict(enumerate(self.PLACES)))
        self.assertEqual(both.places, tuple(self.PLACES))
        self.assertEqual(one.places[0], self.PLACES[0])
        self.assertLess(math.dist(one.places[1], self.PLACES[1]), 0.004)  # the place not told is now found near where it was
        # A wrong joint zero is not something a camera's mount can take up exactly: what is left
        # depends on the pose. Over poses all round the start pose it is under half of what it was
        # (about 12 mm down to about 5); from the start pose itself it is smaller still.
        for fit in (one, both):
            error = self.error_on_new_balls(fit.camera, random.Random(1))
            self.assertLess(error, 0.0065)
            self.assertLess(error, 0.55 * plain)

    def test_on_an_arm_that_is_right_knowing_a_place_changes_little(self):
        rng = random.Random(13)
        seen = sightings(rng, (0.21, 0.0), 10, 0) + sightings(rng, (0.19, 0.07), 10, 1)
        plain = camera_calibration.fit(seen, ODD)
        told = camera_calibration.fit(seen, ODD, known={0: (0.21, 0.0), 1: (0.19, 0.07)})
        self.assertLess(math.dist(plain.camera.mount_xyz, told.camera.mount_xyz), 0.004)
        self.assertLess(told.desk_rms_m, 0.003)

    def test_a_known_place_with_no_sightings_is_refused(self):
        with self.assertRaises(ValueError) as raised:
            camera_calibration.fit(self.seen, ODD, known={5: (0.2, 0.0)})
        self.assertIn("known place", str(raised.exception))


class SpreadTests(unittest.TestCase):
    def test_the_true_camera_agrees_with_itself(self):
        rng = random.Random(9)
        where, typical, worst = camera_calibration.spread(
            TRUE, sightings(rng, (0.2, 0.03), 10, pixel_noise=0.0, size_noise=0.0), ODD)
        self.assertLess(math.dist(where, (0.2, 0.03)), 1e-4)
        self.assertLess(worst, 1e-4)

    def test_a_wrong_camera_disagrees_with_itself(self):
        rng = random.Random(10)
        wrong = dataclasses.replace(TRUE, mount_xyz=(-0.06, 0.03, 0.043))
        _, typical, _ = camera_calibration.spread(wrong, sightings(rng, (0.2, 0.03), 10), ODD)
        self.assertGreater(typical, 0.008)


class FileTests(unittest.TestCase):
    def test_sightings_save_and_load(self):
        rng = random.Random(11)
        seen = sightings(rng, (0.21, 0.0), 3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sightings.json"
            self.assertEqual(camera_calibration.load_sightings(path), [])
            camera_calibration.save_sightings(path, seen)
            self.assertEqual(camera_calibration.load_sightings(path), seen)


if __name__ == "__main__":
    unittest.main()
