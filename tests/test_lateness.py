import math
import unittest

import numpy as np

from understudy import lateness


def steady(period_s: float):
    """A swivel swung back and forth at one steady pace."""
    return lambda times: 0.08 * np.sin(2 * math.pi * np.asarray(times) / period_s)


def as_cued(seed: int = 0, seconds: float = 24.0):
    """A swivel turned as spoken cues ask: a little left, a little right, held, at uneven intervals."""
    rng = np.random.default_rng(seed)
    knots, values, at, side, angle = [0.0], [0.0], 0.0, 1.0, 0.0
    while at < seconds:
        at += rng.uniform(0.6, 2.0)
        if rng.random() < 0.3:
            pass  # hold
        else:
            angle = side * rng.uniform(0.04, 0.1)
            side = -side
        knots.append(at)
        values.append(angle)
    knots, values = np.array(knots), np.array(values)

    def at_times(times):
        # A hand takes about a third of a second to make each turn, and eases into and out of it.
        times = np.asarray(times, dtype=float)
        out = np.zeros_like(times)
        for k in range(1, len(knots)):
            part = np.clip((times - knots[k]) / 0.35, 0.0, 1.0)
            out += (values[k] - values[k - 1]) * (0.5 - 0.5 * np.cos(math.pi * part))
        return out

    return at_times


def recording(motion, late_s: float, seconds: float = 20.0, seed: int = 0, drop: float = 0.1):
    """Joint readings at about the control rate, and the view's position in pictures that arrive ``late_s`` later."""
    rng = np.random.default_rng(seed)
    read_times = np.cumsum(rng.uniform(0.030, 0.037, int(seconds * 30)))
    angles = motion(read_times) + rng.normal(0, 0.0005, len(read_times))  # an encoder's worth of noise
    frame_times = np.arange(0.5, seconds, 1 / 30.0) + rng.uniform(-0.004, 0.004, int((seconds - 0.5) * 30))
    frame_times = frame_times[rng.random(len(frame_times)) > drop]  # some pictures never arrive
    view = -420.0 * motion(frame_times - late_s) + rng.normal(0, 0.4, len(frame_times))  # pixels, opposite in sign
    return read_times, angles, frame_times, view


def found_in(motion, late_s: float, seed: int = 0) -> lateness.Lag:
    reads, angles, frames, view = recording(motion, late_s, seed=seed)
    return lateness.lag(*lateness.rate(reads, angles), *lateness.rate(frames, view))


class LagTests(unittest.TestCase):
    def test_with_uneven_motion_the_lateness_is_found_whatever_it_is(self):
        for late in (0.03, 0.08, 0.15, 0.2, 0.32, 0.4, 0.55):
            for seed in range(4):
                found = found_in(as_cued(seed), late, seed)
                with self.subTest(late=late, seed=seed):
                    self.assertTrue(found.clear(), found)
                    self.assertAlmostEqual(found.seconds, late, delta=0.012)  # well inside the 50 ms that decides anything
                    self.assertEqual(found.sign, -1)  # this view moves against the swivel, and that is found too

    def test_a_steady_swing_is_refused_and_never_answered_wrongly(self):
        # A steady swing lines up as well half a swing later, upside down. The answer must be
        # the true lateness or none at all, in every trial: 80 ms at one swing a second read
        # 580 ms in half the trials before this was caught.
        for late, period in ((0.08, 1.0), (0.4, 0.7), (0.08, 0.7), (0.3, 1.2), (0.15, 0.5)):
            for seed in range(8):
                found = found_in(steady(period), late, seed)
                with self.subTest(late=late, period=period, seed=seed):
                    if found.clear():
                        self.assertAlmostEqual(found.seconds, late, delta=0.015)
        # And for the two cases named above, it is in fact a refusal.
        self.assertFalse(found_in(steady(1.0), 0.08).clear())
        self.assertFalse(found_in(steady(0.7), 0.4).clear())

    def test_a_slow_smooth_swing_is_refused_too(self):
        # Its other match is out of range, but 50 ms either side lines up nearly as well: answers
        # from swings like these were up to 37 ms out.
        for period in (2.6, 3.5):
            for late in (0.08, 0.4):
                for seed in range(6):
                    found = found_in(steady(period), late, seed)
                    with self.subTest(period=period, late=late, seed=seed):
                        self.assertFalse(found.clear(), found)
                        self.assertLess(found.sharp, 0.03)

    def test_turns_made_slowly_but_with_pauses_are_still_answered(self):
        slow = as_cued(2)
        found = found_in(lambda times: (slow(np.asarray(times)) + slow(np.asarray(times) - 0.25)) / 2.0, 0.15)  # each turn drawn out
        self.assertTrue(found.clear(), found)
        self.assertAlmostEqual(found.seconds, 0.15, delta=0.015)

    def test_once_the_way_up_is_known_each_stretch_is_judged_with_it(self):
        reads, angles, frames, view = recording(as_cued(3, 30.0), 0.12, seconds=30.0, seed=5)
        cause, effect = lateness.rate(reads, angles), lateness.rate(frames, view)
        whole = lateness.lag(*cause, *effect)
        stretches = [found for _, found in lateness.lag_by_stretch(*cause, *effect, sign=whole.sign) if found.clear()]
        self.assertGreaterEqual(len(stretches), 3)
        for found in stretches:
            self.assertAlmostEqual(found.seconds, 0.12, delta=0.02)

    def test_an_arm_that_was_not_moved_gives_nothing_to_go_on(self):
        rng = np.random.default_rng(1)
        times = np.arange(0, 10, 1 / 30.0)
        found = lateness.lag(*lateness.rate(times, rng.normal(0, 0.0005, len(times))), *lateness.rate(times, rng.normal(0, 0.4, len(times))))
        self.assertFalse(found.clear())  # noise against noise

    def test_too_short_a_recording_is_refused(self):
        times = np.arange(0, 1.6, 1 / 30.0)
        with self.assertRaises(ValueError):
            lateness.lag(times, np.sin(times), times, np.sin(times))


class ViewTests(unittest.TestCase):
    def desk(self) -> np.ndarray:
        import cv2

        rng = np.random.default_rng(2)
        return cv2.GaussianBlur(rng.integers(0, 255, (300, 400)).astype(np.float32), (0, 0), 2.0)  # something textured, as a desk is

    def test_a_view_that_moved_is_measured_in_pixels_both_ways(self):
        desk = self.desk()
        for across, down in ((-9, 0), (3, 0), (12, 0), (0, 5), (4, -6)):
            before = desk[100:190, 100:260]
            after = desk[100 - down : 190 - down, 100 - across : 260 - across]
            got = lateness.view_shift(before, after)
            self.assertAlmostEqual(got[0], across, delta=0.6)
            self.assertAlmostEqual(got[1], down, delta=0.6)

    def test_the_direction_the_view_mostly_moved_in_is_found_however_the_camera_is_mounted(self):
        swing = 30.0 * np.sin(np.linspace(0, 12, 200))
        rng = np.random.default_rng(4)
        for turn in (0.0, math.pi / 2, 0.6):  # level, on its side, askew
            across = swing * math.cos(turn) + rng.normal(0, 0.3, 200)
            down = swing * math.sin(turn) + rng.normal(0, 0.3, 200)
            along = lateness.main_direction(across, down)
            self.assertGreater(abs(np.corrcoef(along, swing)[0, 1]), 0.99)


class ScriptTests(unittest.TestCase):
    """The recording loop and the verdict of scripts/stream_lateness.py, against a stand-in arm and stream on a made-up clock."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "scripts" / "stream_lateness.py"
        spec = importlib.util.spec_from_file_location("stream_lateness", path)
        cls.script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.script)

    def taken(self, late_s: float, motion, seconds: float = 20.0, said=None) -> dict:
        import cv2

        from understudy import arm
        from understudy.runner import ArmReading

        now = [100.0]
        rng = np.random.default_rng(3)
        desk = cv2.GaussianBlur(rng.integers(0, 255, (360, 1400)).astype(np.float32), (0, 0), 6.0)
        desk = cv2.cvtColor(np.clip(desk, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        def angle(at: float) -> float:
            return float(motion(np.array([at - 100.0]))[0])

        class Arm:
            def read(self):
                now[0] += 0.002  # a read takes a couple of milliseconds
                return ArmReading((int(round(2048 + angle(now[0]) / arm.RAD_PER_TICK)), 0, 0, 0, 0, 0), (0,) * 6)

        class Stream:
            """A picture every thirtieth of a second, showing the view as it was ``late_s`` before it arrives."""

            def arrival(self):
                number = int((now[0] - 100.0) * 30)
                return number, 100.0 + number / 30.0

            def newest(self):
                at = self.arrival()[1] - late_s
                left = int(round(380 + 2400 * angle(at)))  # the view slides as the swivel turns
                return desk[:, left : left + 640]

        def sleep(seconds: float) -> None:
            now[0] += seconds

        script = self.script.cues(seconds)
        return self.script.record(Arm(), Stream(), seconds, clock=lambda: now[0], sleep=sleep,
                                  say=(said.append if said is not None else None), script=script)

    def test_a_known_lateness_is_measured_through_the_whole_script(self):
        for late in (0.05, 0.18, 0.4):
            result = self.script.measure(self.taken(late, as_cued(1)))
            with self.subTest(late=late):
                self.assertTrue(result["told"], result)
                self.assertAlmostEqual(result["lateness_ms"], late * 1000, delta=20)
                self.assertGreater(result["pictures_a_second"], 25)
                for stretch in result["by_stretch_ms"]:
                    self.assertAlmostEqual(stretch, late * 1000, delta=30)

    def test_a_steady_swing_gets_no_number_and_is_told_why(self):
        for late, period, why in ((0.08, 1.0, "too regular"), (0.4, 0.7, "too regular"), (0.08, 3.0, "too slow and smooth")):
            result = self.script.measure(self.taken(late, steady(period)))
            with self.subTest(late=late, period=period):
                self.assertFalse(result["told"], result)
                self.assertIn(why, result["why"])
                self.assertNotIn("lateness_ms", result)

    def test_an_arm_that_was_not_swung_is_reported_and_no_number_given(self):
        result = self.script.measure(self.taken(0.1, lambda times: 0.004 * np.sin(np.asarray(times))))
        self.assertFalse(result["told"])
        self.assertIn("swing it", result["why"])
        self.assertNotIn("lateness_ms", result)

    def test_the_cues_are_spoken_one_word_at_a_time_at_uneven_intervals(self):
        said = []
        self.taken(0.1, as_cued(1), seconds=20.0, said=said)
        script = self.script.cues(20.0)
        self.assertEqual(said, [word for _, word in script])
        self.assertTrue(set(said) <= {"left", "right", "hold"})
        self.assertGreater(len(said), 8)
        gaps = np.diff([at for at, _ in script])
        self.assertGreater(gaps.max() / gaps.min(), 2.0)  # no rhythm to fall into
        turns = [word for word in said if word != "hold"]
        self.assertTrue(all(a != b for a, b in zip(turns, turns[1:])))  # left and right take turns


if __name__ == "__main__":
    unittest.main()
