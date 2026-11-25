import math
import random
import tempfile
import unittest
from pathlib import Path

from understudy import arm, touch_calibration
from understudy.kinematics import Chain
from understudy.touch_calibration import Touch

# Zero points of an arm set up through LeRobot: nowhere near the stock 2048 on the elbow.
TRUE_ZERO = (1943.0, 2031.0, 3020.0, 2075.0, 2048.0)
CHAIN = Chain()


def touch_on_desk(rng: random.Random, zero=TRUE_ZERO, signs=(1, 1, 1, 1, 1)) -> Touch:
    """A pose with the tip exactly on the desk, read through an arm with the given zero points."""
    while True:
        swivel = rng.uniform(-0.5, 0.5)
        shoulder = rng.uniform(-0.3, 1.0)
        wrist = rng.uniform(0.0, 1.4)

        def height(elbow: float) -> float:
            return CHAIN.tip_position([swivel, shoulder, elbow, wrist, 0.0])[2]

        low, high = -1.2, 1.4
        if height(low) * height(high) > 0:
            continue
        for _ in range(60):
            middle = (low + high) / 2
            low, high = (middle, high) if height(low) * height(middle) > 0 else (low, middle)
        angles = [swivel, shoulder, (low + high) / 2, wrist, 0.0]
        x, y, _ = CHAIN.tip_position(angles)
        if x < 0.10:
            continue
        ticks = tuple(arm.rad_to_ticks(a, z, s) for a, z, s in zip(angles, zero, signs))
        return Touch(ticks, x - touch_calibration.SWIVEL_AXIS_X_M, y)


class FittingZeroPoints(unittest.TestCase):
    def test_recovers_the_zero_points_from_six_touches(self):
        rng = random.Random(7)
        touches = [touch_on_desk(rng) for _ in range(6)]
        fit = touch_calibration.fit(touches)
        for found, true in zip(fit.zero_ticks[:4], TRUE_ZERO[:4]):
            self.assertAlmostEqual(found, true, delta=3)  # ticks are whole numbers, so not exact
        self.assertLess(fit.rms_m, 0.001)
        self.assertLess(max(abs(e) for pair in fit.errors_m for e in pair), 0.002)
        self.assertFalse(fit.ambiguous())
        self.assertLess(fit.swivel_spread_rad, 0.01)

    def test_a_base_that_turned_between_touches_spoils_only_the_swivel(self):
        rng = random.Random(7)
        turned = []
        for touch in (touch_on_desk(rng) for _ in range(6)):
            slip = round(rng.uniform(-0.15, 0.15) / arm.RAD_PER_TICK)  # the base turned up to 9 degrees
            turned.append(Touch((touch.ticks[0] + slip, *touch.ticks[1:]), touch.forward_m, touch.left_m))
        fit = touch_calibration.fit(turned)
        for found, true in zip(fit.zero_ticks[1:4], TRUE_ZERO[1:4]):
            self.assertAlmostEqual(found, true, delta=3)
        self.assertLess(fit.rms_m, 0.001)
        self.assertGreater(fit.swivel_spread_rad, 0.05)

    def test_touches_along_one_line_are_enough(self):
        """What the first real arm was calibrated with: every mark on the ruler line."""
        rng = random.Random(21)
        touches = []
        while len(touches) < 5:
            touch = touch_on_desk(rng)
            distance = touch.distance_m()
            touches.append(Touch((round(TRUE_ZERO[0]), *touch.ticks[1:]), distance, 0.0))
        fit = touch_calibration.fit(touches)
        for found, true in zip(fit.zero_ticks[1:4], TRUE_ZERO[1:4]):
            self.assertAlmostEqual(found, true, delta=3)
        # The model's tip sits 1.6 mm to one side, so a mark dead ahead is not quite at swivel zero.
        self.assertAlmostEqual(fit.zero_ticks[0], TRUE_ZERO[0], delta=10)

    def test_copes_with_marks_measured_a_few_millimetres_out(self):
        rng = random.Random(11)
        touches = [touch_on_desk(rng) for _ in range(8)]
        noisy = [Touch(t.ticks, t.forward_m + rng.gauss(0, 0.003), t.left_m + rng.gauss(0, 0.003)) for t in touches]
        fit = touch_calibration.fit(noisy)
        clean = touch_calibration.tip_errors(fit.zero_ticks, touches, (1,) * 5, CHAIN).reshape(-1, 3)
        self.assertLess(max(math.hypot(*row) for row in clean), 0.01)
        self.assertLess(fit.rms_m, 0.01)

    def test_a_reversed_joint_is_handled_through_its_sign(self):
        rng = random.Random(3)
        signs = (1, -1, 1, 1, 1)
        zero = (2048.0, 2100.0, 1990.0, 2048.0, 2048.0)
        touches = [touch_on_desk(rng, zero, signs) for _ in range(6)]
        fit = touch_calibration.fit(touches, signs)
        for found, true in zip(fit.zero_ticks[:4], zero[:4]):
            self.assertAlmostEqual(found, true, delta=3)

    def test_the_wrong_signs_leave_a_large_error(self):
        rng = random.Random(3)
        touches = [touch_on_desk(rng, (2048.0,) * 5, (1, -1, 1, 1, 1)) for _ in range(6)]
        self.assertGreater(touch_calibration.fit(touches).rms_m, 0.01)

    def test_too_few_touches_are_refused(self):
        rng = random.Random(1)
        with self.assertRaises(ValueError):
            touch_calibration.fit([touch_on_desk(rng) for _ in range(3)])

    def test_touches_survive_saving_and_loading(self):
        rng = random.Random(5)
        touches = [touch_on_desk(rng) for _ in range(2)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "touches.json"
            self.assertEqual(touch_calibration.load_touches(path), [])
            touch_calibration.save_touches(path, touches)
            self.assertEqual(touch_calibration.load_touches(path), touches)


if __name__ == "__main__":
    unittest.main()
