"""The real-arm tool's camera commands, run against the simulated arm.

The arm is MuJoCo's, behind the same interface as the real servos, and the
pictures are rendered from a camera on its wrist, so these need OpenGL and are
skipped where there is none. Nothing here touches hardware.
"""

import argparse
import contextlib
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sim_rig
from understudy import arm, pickup

REPO = Path(__file__).resolve().parent.parent
POLICY = REPO / "results" / "pickup" / "2026-11-24-claw" / "policy.onnx"
ODD = arm.Calibration((3012.0, 2031.0, 3042.0, 3013.0, 2050.0), (1, -1, 1, -1, 1), 1740.0, -1)


def load_tool():
    spec = importlib.util.spec_from_file_location("real_arm_tool", REPO / "scripts" / "real_arm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Bench:
    """A simulated arm with a wrist camera, and what the tool needs patched to use it."""

    def __init__(self, folder: Path):
        from understudy import mujoco_arm

        self.true_camera = mujoco_arm.sim_wrist_camera()
        self.sim = mujoco_arm.MujocoArm(
            mujoco_arm.build_model(wrist_camera=self.true_camera), ODD, command_delay_s=0.015,
            profile_time_s=arm.VENDOR_PROFILE_TIME_S,
        )
        self.view = mujoco_arm.RenderedCamera(self.sim, self.true_camera)
        self.robot = sim_rig.SimRobot(self.sim)
        self.folder = folder
        ODD.save(folder / "calibration.json")
        self.on_picture = None
        self.highest = 0.0
        self.sim.after_step = self._after_step
        bench = self

        class Watcher:
            def __init__(self, *args, **kwargs):
                pass

            def picture(self):
                if bench.on_picture is not None:
                    bench.on_picture()
                return bench.view.picture()

        self.watcher = Watcher

    def _after_step(self, sim) -> None:
        self.highest = max(self.highest, float(sim.ball()[2]))

    def rest(self, ball) -> None:
        """The arm in the start pose with the ball somewhere, torque off, as if just set down."""
        self.sim.reset(ball)
        self.highest = 0.0

    def args(self, **extra) -> argparse.Namespace:
        base = dict(
            port="sim", calibration=str(self.folder / "calibration.json"), camera_file=str(self.folder / "camera.json"),
            sightings=str(self.folder / "sightings.json"), watch_folder=str(self.folder), move_seconds=1.0,
        )
        return argparse.Namespace(**{**base, **extra})

    def run(self, tool, command, **extra) -> str:
        said = io.StringIO()
        with mock.patch.object(tool, "open_arm", lambda args: self.robot), \
                mock.patch("understudy.camera_feed.WatcherCamera", self.watcher), \
                mock.patch("builtins.input", lambda prompt="": "yes"), \
                mock.patch.object(tool.time, "sleep", lambda seconds: None), \
                contextlib.redirect_stdout(said):
            try:
                command(self.args(**extra))
            except SystemExit as stop:
                said.write(f"\nSTOPPED: {stop}")
        return said.getvalue()


class ToolWithCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import onnxruntime  # noqa: F401

            cls.temp = tempfile.TemporaryDirectory()
            cls.bench = Bench(Path(cls.temp.name))
        except ImportError as error:
            raise unittest.SkipTest(f"needs mujoco and onnxruntime: {error}") from error
        except Exception as error:  # no OpenGL on this machine
            raise unittest.SkipTest(f"cannot render: {error}") from error
        cls.tool = load_tool()
        # Calibrate the camera once, the way the real procedure does, for all the tests.
        bench = cls.bench
        bench.rest((0.6, 0.5))  # ball off the desk in front of the arm
        spots = iter([(0.21, 0.01), (0.19, 0.06)])
        pictures = [0]

        def put_ball_down_when_asked():
            # The tool takes one picture of the empty desk, then waits for the ball: put it down on the
            # next picture, and move it when the arm is back at the start pose asking again.
            pictures[0] += 1
            at_start = all(abs(a - s) < 0.01 for a, s in zip(ODD.to_rad(bench.sim.read().positions[:5]), pickup.START_POSE))
            if at_start and pictures[0] > 1 and bench.waiting:
                bench.sim.place_ball(next(spots))
                bench.waiting = False

        bench.waiting = True
        bench.on_picture = put_ball_down_when_asked
        original = cls.tool.hold_still

        def hold_still(robot, seconds):
            original(robot, 0.0)

        with mock.patch.object(cls.tool, "hold_still", hold_still):
            real_wait = __import__("understudy.camera_collect", fromlist=["x"]).wait_for_ball

            def wait(take_picture, before, attempts=40, say=print):
                bench.waiting = True
                return real_wait(take_picture, before, attempts, say)

            with mock.patch("understudy.camera_collect.wait_for_ball", wait):
                cls.collect_output = bench.run(cls.tool, cls.tool.cmd_camera_collect, places=2, wait_pictures=6)
        bench.on_picture = None
        cls.fit_output = bench.run(cls.tool, cls.tool.cmd_camera_fit, max_error_mm=8.0)

    @classmethod
    def tearDownClass(cls):
        cls.bench.view.close()
        cls.temp.cleanup()

    def test_collection_saw_the_ball_in_both_places_and_left_the_arm_limp(self):
        self.assertIn("place 1: the ball was seen from", self.collect_output)
        self.assertIn("place 2: the ball was seen from", self.collect_output)
        self.assertIn("torque off, settings restored", self.collect_output)
        self.assertFalse(self.bench.robot.torque_on)
        saved = json.loads((self.bench.folder / "sightings.json").read_text())
        self.assertGreaterEqual(len(saved["sightings"]), 30)
        self.assertIn("hue_low", saved["colour"])

    def test_the_fitted_camera_matches_the_simulated_one(self):
        self.assertIn("saved", self.fit_output, self.fit_output)
        from understudy.camera_model import WristCamera

        camera = WristCamera.load(self.bench.folder / "camera.json")
        self.assertLess(math.dist(camera.mount_xyz, self.bench.true_camera.mount_xyz), 0.004)
        self.assertLess(abs(camera.focal_px - self.bench.true_camera.focal_px), 8)
        notes = json.loads((self.bench.folder / "camera.json").read_text())
        self.assertLess(notes["desk_rms_mm"], 1.5)
        self.assertEqual(notes["places"], 2)

    def run_pickup(self, ball, **extra):
        self.bench.rest(ball)
        out = self.bench.run(
            self.tool, self.tool.cmd_run, policy=str(POLICY), ball=None, camera="look", target=[0.19, 0.0, 0.13],
            seconds=15.0, max_speed=0.3, max_lead=0.05, hold_seconds=0.0, log=str(self.bench.folder / "run.json"),
            dry_run=False, any_distance=False, camera_latency_steps=3, ffmpeg="ffmpeg", **extra,
        )
        return out, json.loads((self.bench.folder / "run.json").read_text()) if "steps recorded" in out else None

    def test_picks_up_a_ball_well_off_to_the_right_with_the_gentle_limits(self):
        # 40 degrees to the right: outside the patch the policy was trained on, and on the side
        # where it fails with these limits unless the arm turns to face the ball.
        distance, bearing = 0.21, math.radians(-40)
        ball = (-0.01125 + distance * math.cos(bearing), distance * math.sin(bearing))
        out, log = self.run_pickup(ball)
        self.assertIn("the arm has turned to face it", out, out)
        self.assertIn("torque off, settings restored", out)
        self.assertGreater(self.bench.highest, pickup.LIFTED_HEIGHT_M, out)
        self.assertLess(log["turn_rad"], -0.5)
        self.assertAlmostEqual(log["ball"][1], 0.03, delta=0.006)  # in the turned frame the ball is 3 cm to the left
        self.assertFalse(self.bench.robot.torque_on)

    def test_stops_without_a_pickup_when_there_is_no_ball(self):
        out, log = self.run_pickup((0.6, 0.5))
        self.assertIn("did not see the ball", out)
        self.assertIn("torque off, settings restored", out)
        self.assertEqual(log["steps"], [])
        self.assertFalse(self.bench.robot.torque_on)

    def test_stops_when_the_ball_is_too_close_to_pick_up(self):
        out, log = self.run_pickup((0.13, 0.0))
        self.assertIn("the policy picks up between", out, out)
        self.assertEqual(log["steps"], [])
        self.assertFalse(self.bench.robot.torque_on)

    def test_look_reports_where_the_ball_is(self):
        self.bench.rest((0.2, -0.03))
        out = self.bench.run(self.tool, self.tool.cmd_look, save=str(self.bench.folder / "look.jpg"))
        self.assertRegex(out, r"ball at x=0\.(19\d|20\d) y=-0\.0(2\d|3\d)")
        self.assertTrue((self.bench.folder / "look.jpg").exists())


class ArgumentTests(unittest.TestCase):
    def test_run_needs_a_ball_position_or_a_camera_and_not_both(self):
        tool = load_tool()
        for ball, camera in ((None, None), ([0.2, 0.0], "look")):
            with self.assertRaises(SystemExit) as stop:
                tool.cmd_run(argparse.Namespace(ball=ball, camera=camera, dry_run=False))
            self.assertIn("--ball", str(stop.exception))

    def test_a_dry_run_cannot_use_the_camera(self):
        tool = load_tool()
        with self.assertRaises(SystemExit) as stop:
            tool.cmd_run(argparse.Namespace(ball=None, camera="look", dry_run=True))
        self.assertIn("limp", str(stop.exception))


if __name__ == "__main__":
    unittest.main()
