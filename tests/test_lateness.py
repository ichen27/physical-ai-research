import math
import unittest

import numpy as np

from understudy import lateness


def swing(times: np.ndarray) -> np.ndarray:
    """A swivel swung back and forth by hand: not quite regular, a few degrees each way."""
    return 0.08 * np.sin(2 * math.pi * 0.9 * times) + 0.03 * np.sin(2 * math.pi * 2.3 * times + 1.0)


def recording(late_s: float, seconds: float = 10.0, seed: int = 0, drop: float = 0.1):
    """Joint readings at about the control rate, and the view's sideways position in pictures that arrive ``late_s`` later."""
    rng = np.random.default_rng(seed)
    read_times = np.cumsum(rng.uniform(0.030, 0.037, int(seconds * 30)))
    angles = swing(read_times) + rng.normal(0, 0.0005, len(read_times))  # an encoder's worth of noise
    frame_times = np.arange(0.5, seconds, 1 / 30.0) + rng.uniform(-0.004, 0.004, int((seconds - 0.5) * 30))
    frame_times = frame_times[rng.random(len(frame_times)) > drop]  # some pictures never arrive
    view = -420.0 * swing(frame_times - late_s) + rng.normal(0, 0.4, len(frame_times))  # pixels, opposite in sign
    return read_times, angles, frame_times, view


class LagTests(unittest.TestCase):
    def test_the_lateness_is_found_whatever_it_is(self):
        for late in (0.03, 0.08, 0.15, 0.2, 0.32):
            reads, angles, frames, view = recording(late, seed=int(late * 1000))
            found, alike = lateness.lag(*lateness.rate(reads, angles), *lateness.rate(frames, view))
            with self.subTest(late=late):
                self.assertAlmostEqual(found, late, delta=0.012)  # well inside the 50 ms that decides anything
                self.assertGreater(alike, 0.8)

    def test_it_is_found_in_each_stretch_of_the_recording(self):
        reads, angles, frames, view = recording(0.12, seconds=12.0, seed=5)
        stretches = lateness.lag_by_stretch(*lateness.rate(reads, angles), *lateness.rate(frames, view))
        self.assertGreaterEqual(len(stretches), 3)
        for _, found, alike in stretches:
            self.assertAlmostEqual(found, 0.12, delta=0.02)
            self.assertGreater(alike, 0.7)

    def test_an_arm_that_was_not_moved_gives_nothing_to_go_on(self):
        rng = np.random.default_rng(1)
        times = np.arange(0, 10, 1 / 30.0)
        _, alike = lateness.lag(*lateness.rate(times, rng.normal(0, 0.0005, len(times))), *lateness.rate(times, rng.normal(0, 0.4, len(times))))
        self.assertLess(alike, 0.5)  # noise against noise: the script says so instead of reporting a number

    def test_too_short_a_recording_is_refused(self):
        times = np.arange(0, 1.2, 1 / 30.0)
        with self.assertRaises(ValueError):
            lateness.lag(times, np.sin(times), times, np.sin(times))


class ShiftTests(unittest.TestCase):
    def test_a_view_moved_sideways_is_measured_in_pixels(self):
        rng = np.random.default_rng(2)
        desk = rng.integers(0, 255, (90, 400)).astype(np.float32)
        import cv2

        desk = cv2.GaussianBlur(desk, (0, 0), 2.0)  # something textured, as a desk is
        for moved in (-9, -2, 3, 12):
            before, after = desk[:, 100:260], desk[:, 100 - moved : 260 - moved]
            self.assertAlmostEqual(lateness.sideways_shift(before, after), moved, delta=0.6)



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

    def taken(self, late_s: float, swing_rad: float = 0.1, seconds: float = 10.0) -> dict:
        import cv2

        from understudy import arm
        from understudy.runner import ArmReading

        now = [100.0]
        rng = np.random.default_rng(3)
        desk = cv2.GaussianBlur(rng.integers(0, 255, (360, 1400)).astype(np.float32), (0, 0), 6.0)
        desk = cv2.cvtColor(np.clip(desk, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        def angle(at: float) -> float:
            return swing_rad * math.sin(2 * math.pi * 0.8 * at) + 0.3 * swing_rad * math.sin(2 * math.pi * 2.1 * at)

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

        return self.script.record(Arm(), Stream(), seconds, clock=lambda: now[0], sleep=sleep)

    def test_a_known_lateness_is_measured_through_the_whole_script(self):
        for late in (0.05, 0.18):
            result = self.script.measure(self.taken(late))
            with self.subTest(late=late):
                self.assertTrue(result["told"], result)
                self.assertAlmostEqual(result["lateness_ms"], late * 1000, delta=20)
                self.assertAlmostEqual(result["median_stretch_ms"], late * 1000, delta=25)
                self.assertGreater(result["pictures_a_second"], 25)

    def test_an_arm_that_was_not_swung_is_reported_and_no_number_given(self):
        result = self.script.measure(self.taken(0.1, swing_rad=0.005))
        self.assertFalse(result["told"])
        self.assertIn("swing it", result["why"])
        self.assertNotIn("lateness_ms", result)


if __name__ == "__main__":
    unittest.main()
