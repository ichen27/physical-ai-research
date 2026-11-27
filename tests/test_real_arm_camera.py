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
# The real servos cut their power, and the raised arm drops, after 500 ms without hearing from the
# computer. While powered, the tool must never be silent for half that.
QUIET_LIMIT_S = 0.25
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
            def __init__(self, folder=None, keep_alive=None, **kwargs):
                self.keep_alive = keep_alive

            def picture(self):
                if self.keep_alive is not None:
                    self.keep_alive()  # as the real one does, many times, while it waits for the picture
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
        cls.collect_silence_s = bench.robot.longest_silence_s
        cls.collect_silence_at = bench.robot.longest_silence_at
        cls.fit_output = bench.run(cls.tool, cls.tool.cmd_camera_fit, max_error_mm=8.0, known=None)

    @classmethod
    def tearDownClass(cls):
        cls.bench.view.close()
        cls.temp.cleanup()

    def test_collection_saw_the_ball_in_both_places_and_left_the_arm_limp(self):
        self.assertIn("place 1: the ball was seen from", self.collect_output)
        self.assertIn("place 2: the ball was seen from", self.collect_output)
        self.assertIn("torque off, settings restored", self.collect_output)
        self.assertFalse(self.bench.robot.torque_on)
        self.assertLess(self.collect_silence_s, QUIET_LIMIT_S, f"longest silence ended in: {self.collect_silence_at}")
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

    def test_the_fit_can_be_told_where_the_ball_was(self):
        # The simulated arm's zeros are right, so telling the fit where the ball was changes little,
        # and it says how far the pictures alone were from what it was told.
        other = self.bench.folder / "camera-known.json"
        out = self.bench.run(self.tool, self.tool.cmd_camera_fit, max_error_mm=8.0, camera_file=str(other),
                             known=[[1.0, 0.21, 0.01], [2.0, 0.19, 0.06]])
        self.assertIn("place 1: ball known to be x=0.210 y=0.010", out, out)
        self.assertIn("place 2: ball known to be x=0.190 y=0.060", out)
        self.assertRegex(out, r"the pictures alone would have put it at x=0\.2\d\d y=0\.0\d\d, [0-3] mm away")
        self.assertNotIn("No place's position was given", out)
        notes = json.loads(other.read_text())
        self.assertEqual(notes["known_places"], {"1": [0.21, 0.01], "2": [0.19, 0.06]})
        self.assertLess(notes["desk_rms_mm"], 2.5)
        self.assertIn("No place's position was given", self.fit_output)  # and without it, the fit says what that costs

    def test_a_known_place_that_does_not_exist_is_refused(self):
        out = self.bench.run(self.tool, self.tool.cmd_camera_fit, max_error_mm=8.0,
                             camera_file=str(self.bench.folder / "never.json"), known=[[3.0, 0.2, 0.0]])
        self.assertIn("the places are numbered 1 to 2", out)
        self.assertFalse((self.bench.folder / "never.json").exists())

    def run_pickup(self, ball, **extra):
        self.bench.rest(ball)
        out = self.bench.run(
            self.tool, self.tool.cmd_run, policy=str(POLICY), ball=None, camera="look", target=[0.19, 0.0, 0.13],
            seconds=15.0, max_speed=0.3, max_lead=0.05, hold_seconds=0.0, log=str(self.bench.folder / "run.json"),
            dry_run=False, camera_latency_steps=3, ffmpeg="ffmpeg",
            **{"any_distance": False, "find_only": False, "set_down": None, "bowl": None, **extra},
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
        self.assertLess(self.bench.robot.longest_silence_s, QUIET_LIMIT_S,
                        f"longest silence ended in: {self.bench.robot.longest_silence_at}")
        self.assertLess(log["turn_rad"], -0.5)
        self.assertAlmostEqual(log["ball"][1], 0.03, delta=0.006)  # in the turned frame the ball is 3 cm to the left
        self.assertFalse(self.bench.robot.torque_on)

    def test_the_ball_is_picked_up_and_put_down_on_the_desk_at_the_side(self):
        ball = (-0.01125 + 0.2 * math.cos(0.3), 0.2 * math.sin(0.3))
        out, log = self.run_pickup(ball, set_down=[80.0, 20.0])
        self.assertIn("carrying the ball to +80 degrees round, 20 cm from the swivel axis\n", out, out)
        self.assertRegex(out, r"ball let go (19\.\d|20\.\d) cm from the swivel axis, \+(7\d|8\d) degrees round")
        self.assertIn("torque off, settings restored", out)
        self.assertTrue(log["carry"]["done"])
        spot = (-0.01125 + 0.2 * math.cos(math.radians(80)), 0.2 * math.sin(math.radians(80)))
        self.assertLess(math.dist(log["carry"]["released_at"], spot), 0.01)
        self.assertGreater(self.bench.highest, 0.15)  # it was carried high
        self.assertLess(float(self.bench.sim.ball()[2]), pickup.BALL_RADIUS_M + 0.003)  # and is on the desk again, not held
        self.assertLess(math.dist(self.bench.sim.ball()[:2], ball), 0.5)
        self.assertGreater(math.dist(self.bench.sim.ball()[:2], ball), 0.1)  # away from where it was picked up
        self.assertFalse(self.bench.robot.torque_on)
        self.assertLess(self.bench.robot.longest_silence_s, QUIET_LIMIT_S,
                        f"longest silence ended in: {self.bench.robot.longest_silence_at}")

    # A cereal bowl at the arm's left, as two readings of `where`: the tip over its middle, then resting on its rim.
    BOWL = [0.0, 0.24, 0.0, 0.165, 0.065]

    def test_the_ball_is_carried_over_a_bowl_and_let_go_above_its_rim(self):
        ball = (-0.01125 + 0.2 * math.cos(0.1), 0.2 * math.sin(0.1))
        heights = []
        watch = self.bench.sim.after_step

        def also(sim):
            watch(sim)
            heights.append((float(sim.ball()[2]), float(sim.data.ctrl[sim._ctrl[5]])))

        self.bench.sim.after_step = also
        try:
            out, log = self.run_pickup(ball, bowl=self.BOWL)
        finally:
            self.bench.sim.after_step = watch
        self.assertIn("carrying the ball to +87 degrees round, 24 cm from the swivel axis, "
                      "into a bowl 15.0 cm across with its rim 6.5 cm up", out, out)
        self.assertIn("let go over the container", out)
        self.assertTrue(log["carry"]["done"])
        self.assertLess(math.dist(log["carry"]["released_at"], (0.0, 0.24)), 0.01)
        # The bench has no bowl in it: what is checked is that the ball was still held, and above
        # where the rim would be, when the gripper began to open over the bowl's middle.
        carried = heights[next(i for i, (height, _) in enumerate(heights) if height > 0.18):]  # from when it was raised
        let_go = next(height for height, finger in carried if finger > 0.02)
        self.assertGreater(let_go - pickup.BALL_RADIUS_M, 0.065 + 0.005)
        self.assertLess(let_go - pickup.BALL_RADIUS_M, 0.065 + 0.02)
        self.assertFalse(self.bench.robot.torque_on)
        self.assertLess(self.bench.robot.longest_silence_s, QUIET_LIMIT_S,
                        f"longest silence ended in: {self.bench.robot.longest_silence_at}")

    def test_a_ball_the_bowl_is_in_the_way_of_is_not_picked_up(self):
        ball = (-0.01125 + 0.2 * math.cos(math.radians(55)), 0.2 * math.sin(math.radians(55)))  # round towards the bowl
        out, log = self.run_pickup(ball, bowl=self.BOWL)
        self.assertIn("stopped before the pickup: the container is in the gripper's way", out, out)
        self.assertNotIn("carrying the ball", out)
        self.assertLess(float(self.bench.sim.ball()[2]), pickup.BALL_RADIUS_M + 0.003)  # still on the desk
        self.assertLess(math.dist(self.bench.sim.ball()[:2], ball), 0.005)  # and not knocked
        self.assertFalse(self.bench.robot.torque_on)

    def test_a_place_the_arm_cannot_use_is_refused_before_the_arm_is_powered(self):
        for extra, why in (
            (dict(set_down=[120.0, 20.0]), "further than the swivel goes"),
            (dict(bowl=[0.0, 0.16, 0.0, 0.085, 0.065]), "near edge"),
            (dict(bowl=self.BOWL, set_down=[80.0, 20.0]), "not both"),
        ):
            self.bench.rest((0.2, 0.03))
            self.bench.robot.log.clear()
            out, log = self.run_pickup((0.2, 0.03), **extra)
            with self.subTest(extra=extra):
                self.assertIn(why, out)
                self.assertIsNone(log)
                self.assertEqual(self.bench.robot.log, [])  # never configured, never moved

    def test_find_only_says_where_the_ball_is_and_leaves_it_on_the_desk(self):
        distance, bearing = 0.2, math.radians(25)
        ball = (-0.01125 + distance * math.cos(bearing), distance * math.sin(bearing))
        out, log = self.run_pickup(ball, find_only=True)
        self.assertRegex(out, r"ball seen (19\.\d|20\.\d) cm from the swivel axis, \+1\d degrees round")
        self.assertIn("no pickup was attempted", out)
        self.assertIn("torque off, settings restored", out)
        self.assertEqual(log["steps"], [])
        self.assertAlmostEqual(log["ball"][1], 0.03, delta=0.006)
        self.assertLess(self.bench.highest, pickup.BALL_RADIUS_M + 0.002)  # the ball never left the desk
        self.assertFalse(self.bench.robot.torque_on)
        self.assertLess(self.bench.robot.longest_silence_s, QUIET_LIMIT_S,
                        f"longest silence ended in: {self.bench.robot.longest_silence_at}")

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
