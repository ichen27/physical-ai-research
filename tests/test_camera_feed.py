import stat
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

import ball_pictures
from understudy import vision
from understudy.camera_feed import CameraError, LiveCamera, WatcherCamera, find_camera

LISTING = """\
[AVFoundation indev @ 0x1] AVFoundation video devices:
[AVFoundation indev @ 0x1] [0] Innomaker-U20CAM-720P  
[AVFoundation indev @ 0x1] [1] MacBook Air Camera
[AVFoundation indev @ 0x1] [2] Somebody's iPhone Camera
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] Innomaker-U20CAM-720P Microphone
"""


def fake_ffmpeg(folder: Path, listing: str) -> str:
    """A program that answers a request for the list of cameras the way ffmpeg does."""
    (folder / "listing.txt").write_text(listing)
    path = folder / "ffmpeg"
    path.write_text(f"#!{sys.executable}\nimport sys\nsys.stderr.write(open({str(folder / 'listing.txt')!r}).read())\nsys.exit(1)\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


class WatcherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        (self.folder / "shots").mkdir()
        self.asked: list[str] = []
        self.stop = threading.Event()

    def tearDown(self):
        self.stop.set()
        self.temp.cleanup()

    def start_watcher(self, answer=True):
        """Does what scripts/camera_watch.sh does, with a saved picture instead of a camera."""
        picture = (ball_pictures.FIXTURES / "hold1.jpg").read_bytes()

        def watch():
            while not self.stop.is_set():
                note = self.folder / "take"
                if note.exists():
                    name = note.read_text()
                    note.unlink()
                    self.asked.append(name)
                    if answer:
                        (self.folder / "shots" / f"{name}.jpg").write_bytes(picture)
                    (self.folder / "shots" / f"{name}.done").write_text("done\n")
                time.sleep(0.01)

        threading.Thread(target=watch, daemon=True).start()

    def test_asks_for_a_picture_and_reads_it(self):
        self.start_watcher()
        fed = []
        camera = WatcherCamera(self.folder, keep_alive=lambda: fed.append(1), prefix="cal")
        first = camera.picture()
        camera.picture()
        self.assertEqual(first.shape, (720, 1280, 3))
        self.assertEqual(len(self.asked), 2)
        self.assertNotEqual(self.asked[0], self.asked[1])  # an old picture can never be mistaken for a new one
        self.assertRegex(self.asked[0], r"^cal-\d+-001$")
        self.assertTrue(fed)  # the arm was kept alive while waiting
        self.assertEqual(camera.last_path.name, f"{self.asked[1]}.jpg")

    def test_says_how_to_start_the_watcher_when_nothing_answers(self):
        camera = WatcherCamera(self.folder, timeout_s=0.2)
        with self.assertRaises(CameraError) as raised:
            camera.picture()
        self.assertIn("camera-watch.sh", str(raised.exception))
        self.assertFalse((self.folder / "take").exists())  # the request is withdrawn, not left for later

    def test_says_when_the_watcher_could_not_take_the_picture(self):
        self.start_watcher(answer=False)
        with self.assertRaises(CameraError) as raised:
            WatcherCamera(self.folder).picture()
        self.assertIn("plugged in", str(raised.exception))

    def test_a_missing_folder_is_explained(self):
        with self.assertRaises(CameraError):
            WatcherCamera(self.folder / "absent").picture()


class FindCameraTests(unittest.TestCase):
    def test_picks_the_arms_camera_by_name_and_no_other(self):
        with tempfile.TemporaryDirectory() as folder:
            name = find_camera(fake_ffmpeg(Path(folder), LISTING))
        self.assertEqual(name, "Innomaker-U20CAM-720P  ")  # exactly as listed, trailing spaces and all

    def test_refuses_when_the_arms_camera_is_not_there(self):
        without = "\n".join(line for line in LISTING.splitlines() if "[0] Innomaker" not in line)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(CameraError) as raised:
                find_camera(fake_ffmpeg(Path(folder), without))
        self.assertIn("MacBook Air Camera", str(raised.exception))  # says what is there, opens none of it

    def test_refuses_when_two_cameras_match(self):
        twice = LISTING.replace("[1] MacBook Air Camera", "[1] Innomaker-U20CAM-720P #2")
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(CameraError):
                find_camera(fake_ffmpeg(Path(folder), twice))

    def test_a_missing_ffmpeg_is_explained(self):
        with self.assertRaises(CameraError):
            find_camera("/nonexistent/ffmpeg")


class LiveCameraTests(unittest.TestCase):
    def stream(self, body: str) -> LiveCamera:
        return LiveCamera(width=64, height=36, full_width=1280, command=[sys.executable, "-c", textwrap.dedent(body)])

    def test_keeps_the_newest_picture_and_reports_positions_at_full_size(self):
        live = self.stream("""
            import sys, time
            frame = bytearray([118, 121, 119] * (64 * 36))
            for row in range(10, 26):
                for column in range(24, 40):
                    if (row - 18) ** 2 + (column - 32) ** 2 <= 49:
                        frame[(row * 64 + column) * 3:(row * 64 + column) * 3 + 3] = bytes([30, 118, 240])
            for _ in range(200):
                sys.stdout.buffer.write(frame); sys.stdout.buffer.flush(); time.sleep(0.01)
        """)
        try:
            live.wait_for_first(5.0)
            blobs = None
            for _ in range(100):
                blobs = live.look(vision.BallColour())
                if blobs is not None:
                    break
                time.sleep(0.01)
            self.assertEqual(len(blobs), 1)
            self.assertAlmostEqual(blobs[0].u, 32.5 * 20, delta=12)
            self.assertAlmostEqual(blobs[0].v, 18.5 * 20, delta=12)
            self.assertAlmostEqual(blobs[0].radius_px, 7 * 20, delta=16)
            self.assertEqual(live.newest().shape, (36, 64, 3))
        finally:
            live.close()

    def test_no_new_picture_means_no_answer(self):
        live = self.stream("""
            import sys, time
            sys.stdout.buffer.write(bytes(64 * 36 * 3)); sys.stdout.buffer.flush(); time.sleep(3)
        """)
        try:
            live.wait_for_first(5.0)
            self.assertIsNotNone(live.picture())
            self.assertIsNone(live.picture())
            self.assertIsNone(live.look(vision.BallColour()))
        finally:
            live.close()

    def test_the_camera_is_asked_for_its_full_size_unless_a_smaller_one_of_the_same_shape_is_named(self):
        with mock.patch("understudy.camera_feed.find_camera", lambda ffmpeg: "the arm's camera"):
            for capture, wanted in ((None, "1280x720"), ((640, 360), "640x360")):
                with mock.patch("understudy.camera_feed.subprocess.Popen") as started:
                    started.return_value.stdout.read.return_value = b""
                    LiveCamera(capture=capture)
                command = started.call_args[0][0]
                self.assertEqual(command[command.index("-video_size") + 1], wanted)
                self.assertIn("scale=640:360", command[command.index("-vf") + 1])  # what the tool is given does not change
            with self.assertRaises(CameraError) as refused:
                LiveCamera(capture=(640, 480))  # four by three: the camera would show a different view
            self.assertIn("different view", str(refused.exception))

    def test_a_camera_that_will_not_open_is_explained(self):
        live = self.stream("""
            import sys
            sys.stderr.write("Input/output error"); sys.exit(1)
        """)
        with self.assertRaises(CameraError) as raised:
            live.wait_for_first(5.0)
        self.assertIn("Terminal", str(raised.exception))
        self.assertIn("Input/output error", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
