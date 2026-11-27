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
import re
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

    def __init__(self, folder: Path, camera=None):
        from understudy import mujoco_arm

        self.true_camera = camera or mujoco_arm.sim_wrist_camera()
        self.sim = mujoco_arm.MujocoArm(
            mujoco_arm.build_model(wrist_camera=self.true_camera), ODD, command_delay_s=0.015,
            profile_time_s=arm.VENDOR_PROFILE_TIME_S,
        )
        self.view = mujoco_arm.RenderedCamera(self.sim, self.true_camera)
        self.robot = sim_rig.SimRobot(self.sim)
        self.folder = folder
        ODD.save(folder / "calibration.json")
        self.on_picture = None
        self.stream = None  # a class to stand in for the live camera, if a test wants one
        self.shots = 0
        self.highest = 0.0
        self.sim.after_step = self._after_step
        bench = self

        class Watcher:
            def __init__(self, folder=None, keep_alive=None, **kwargs):
                self.keep_alive = keep_alive

            def picture(self):
                bench.shots += 1
                self.last_path = Path(f"shot-{bench.shots:03d}.jpg")
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
        from understudy import camera_feed

        with mock.patch.object(tool, "open_arm", lambda args: self.robot), \
                mock.patch("understudy.camera_feed.WatcherCamera", self.watcher), \
                mock.patch("understudy.camera_feed.LiveCamera", self.stream or camera_feed.LiveCamera), \
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
            **{"any_distance": False, "find_only": False, "set_down": None, "bowl": None, "eye": None, "driver": None, "sweep_deg": 57.0, "eye_must_agree": False, "compare_finders": False, **extra},
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



EYE = Path(__file__).resolve().parent.parent / "results" / "student" / "2026-11-26-mask-and-grey" / "eye.onnx"


class ToolWithTrainedEyeTests(unittest.TestCase):
    """``run --camera look --eye``: a trained eye says where the ball is; the pickup is the same. No PyTorch needed."""

    @classmethod
    def setUpClass(cls):
        if not EYE.exists():
            raise unittest.SkipTest("no exported eye in this checkout")
        try:
            import onnxruntime  # noqa: F401

            from understudy.camera_model import WristCamera

            trained = json.loads(EYE.with_suffix(".json").read_text())["student_training"]["nominal_camera"]
            cls.camera = WristCamera(**{key: tuple(value) if isinstance(value, list) else value for key, value in trained.items()})
            cls.temp = tempfile.TemporaryDirectory()
            cls.bench = Bench(Path(cls.temp.name), cls.camera.resized(640, 360))  # the camera the eye was trained for
        except ImportError as error:
            raise unittest.SkipTest(f"needs mujoco and onnxruntime: {error}") from error
        except Exception as error:  # no OpenGL on this machine
            raise unittest.SkipTest(f"cannot render: {error}") from error
        cls.tool = load_tool()

    @classmethod
    def tearDownClass(cls):
        cls.bench.view.close()
        cls.temp.cleanup()

    def run_pickup(self, ball, **extra):
        self.bench.rest(ball)
        self.bench.robot.log.clear()
        out = self.bench.run(
            self.tool, self.tool.cmd_run, policy=str(POLICY), ball=None, target=[0.19, 0.0, 0.13],
            seconds=15.0, max_speed=0.3, max_lead=0.05, hold_seconds=0.0, log=str(self.bench.folder / "run.json"),
            dry_run=False, camera_latency_steps=3, ffmpeg="ffmpeg",
            **{"camera": "look", "any_distance": False, "find_only": False, "set_down": None, "bowl": None, "eye": str(EYE), "driver": None, "sweep_deg": 57.0, "eye_must_agree": False, "compare_finders": False, **extra},
        )
        return out

    def test_the_eye_finds_the_ball_off_to_one_side_and_the_arm_picks_it_up(self):
        distance, bearing = 0.2, math.radians(-30)
        ball = (-0.01125 + distance * math.cos(bearing), distance * math.sin(bearing))
        out = self.run_pickup(ball)
        self.assertIn("cannot be checked against this arm's camera", out, out)  # there is no camera file on this bench
        self.assertRegex(out, r"the eye sees the ball \(0\.\d\d sure\)")
        said = re.search(r"ball seen ([\d.]+) cm from the swivel axis, ([+-]\d+) degrees round", out)
        self.assertIsNotNone(said, out)
        self.assertAlmostEqual(float(said.group(1)), 20.0, delta=1.5)
        self.assertGreater(self.bench.highest, pickup.LIFTED_HEIGHT_M)  # lifted, from the eye's answer alone
        self.assertFalse(self.bench.robot.torque_on)
        self.assertLess(self.bench.robot.longest_silence_s, QUIET_LIMIT_S,
                        f"longest silence ended in: {self.bench.robot.longest_silence_at}")

    def test_with_no_ball_the_eye_is_not_sure_and_nothing_is_picked_up(self):
        out = self.run_pickup((0.6, 0.5))  # the ball is nowhere the camera can look
        self.assertIn("the eye is not sure there is a ball in this picture", out, out)
        self.assertIn("stopped before the pickup: the camera did not see the ball anywhere it looked", out)
        self.assertNotIn("finished. gripper", out)
        self.assertFalse(self.bench.robot.torque_on)
        self.assertLess(self.bench.robot.longest_silence_s, QUIET_LIMIT_S)

    def test_a_narrower_sweep_does_not_look_where_it_was_told_not_to(self):
        distance, bearing = 0.2, math.radians(62)  # too far round to show in the look straight ahead
        ball = (-0.01125 + distance * math.cos(bearing), distance * math.sin(bearing))
        out = self.run_pickup(ball, sweep_deg=0.0, find_only=True)
        self.assertIn("no ball seen looking +0 degrees round", out, out)
        self.assertNotIn("29 degrees round", out)
        self.assertNotIn("57 degrees round", out)
        self.assertIn("stopped before the pickup: the camera did not see the ball anywhere it looked", out)
        out = self.run_pickup(ball, sweep_deg=29.0, find_only=True)  # one look to each side is enough for this one
        self.assertNotIn("57 degrees round", out)
        self.assertRegex(out, r"ball seen ([\d.]+) cm from the swivel axis, \+5\d degrees round")
        self.assertIn("no pickup was attempted because --find_only was given", out)

    def test_an_eye_for_another_camera_is_refused_before_the_arm_is_powered(self):
        import dataclasses

        other = dataclasses.replace(self.camera, focal_px=self.camera.focal_px * 1.3)
        other.save(self.bench.folder / "camera.json")
        try:
            out = self.run_pickup((0.2, 0.03))
        finally:
            (self.bench.folder / "camera.json").unlink()
        self.assertIn("too far for its answers to be trusted", out, out)
        self.assertEqual(self.bench.robot.log, [])  # never configured, never moved

    def with_camera_file(self, **colour):
        """This bench's camera written as the arm's camera file, with the ball's colour as the hand-written finder is to take it."""
        import dataclasses

        path = self.bench.folder / "camera.json"
        self.camera.save(path)
        if colour:
            path.write_text(json.dumps({**json.loads(path.read_text()), "colour": colour}))
        return path

    def test_asked_to_agree_the_two_finders_are_compared_on_every_look_and_recorded(self):
        distance, bearing = 0.2, math.radians(20)
        ball = (-0.01125 + distance * math.cos(bearing), distance * math.sin(bearing))
        path = self.with_camera_file()
        try:
            out = self.run_pickup(ball, eye_must_agree=True)
        finally:
            path.unlink()
        self.assertEqual(len(re.findall(r"the eye sees the ball \(\d\.\d\d sure\) and the hand-written finder agrees to \d+ mm", out)), 2, out)
        self.assertGreater(self.bench.highest, pickup.LIFTED_HEIGHT_M)
        looks = json.loads((self.bench.folder / "run.json").read_text())["looks"]
        self.assertEqual(len(looks), 2)
        for look in looks:  # the joints as read at each look, and what each finder said
            self.assertEqual(len(look["joint_angles"]), 5)
            self.assertLess(math.dist((look["eye"]["x"], look["eye"]["y"]), (look["other_finder"]["x"], look["other_finder"]["y"])), 0.025)
            self.assertRegex(look["picture"], r"shot-\d+\.jpg")
        self.assertAlmostEqual(looks[0]["joint_angles"][1], -0.9, delta=0.05)  # the look pose

    def test_asked_to_agree_a_ball_only_the_eye_sees_is_refused_and_the_reason_is_said(self):
        # The hand-written finder told that the ball is green sees nothing: the eye alone is not enough.
        path = self.with_camera_file(hue_low=100.0, hue_high=140.0, saturation_min=0.3, value_min=0.2, veil_max=90.0)
        try:
            out = self.run_pickup((0.2, 0.03), eye_must_agree=True)
        finally:
            path.unlink()
        self.assertIn("but the hand-written finder sees none in the same picture: not taken as the ball", out, out)
        self.assertIn("stopped before the pickup: the camera did not see the ball anywhere it looked", out)
        self.assertNotIn("finished. gripper", out)
        self.assertLess(float(self.bench.sim.ball()[2]), pickup.BALL_RADIUS_M + 0.003)  # the ball was left alone
        self.assertFalse(self.bench.robot.torque_on)

    def test_comparing_the_finders_prints_and_records_every_place_and_decides_nothing(self):
        distance, bearing = 0.17, math.radians(-15)  # near enough to be whole in the start-pose picture
        ball = (-0.01125 + distance * math.cos(bearing), distance * math.sin(bearing))
        path = self.with_camera_file()
        try:
            out = self.run_pickup(ball, compare_finders=True, find_only=True)
        finally:
            path.unlink()
        line = r"place {} {}: (\d+\.\d) cm from the swivel axis, ([+-]\d+\.\d) degrees round, x=(-?\d\.\d+) y=(-?\d\.\d+)"
        places = {}
        for pose in ("look-pose look 1", "look-pose look 2", "start-pose"):
            for who in ("eye", "finder"):
                found = re.search(line.format(pose, who), out)
                self.assertIsNotNone(found, f"{pose} {who}\n{out}")
                places[pose, who] = (float(found.group(3)), float(found.group(4)))
                self.assertAlmostEqual(float(found.group(1)), 17.0, delta=2.0)
        self.assertRegex(out, r"place start-pose eye: .* sure=\d\.\d\d")
        self.assertRegex(out, r"place start-pose finder: .* whole=yes")
        # All six are in the arm's own frame, so they can be set beside one another and beside a touch.
        for key, place in places.items():
            self.assertLess(math.dist(place, ball), 0.025, key)
        self.assertIn("no pickup was attempted because --find_only was given", out)
        log = json.loads((self.bench.folder / "run.json").read_text())
        self.assertEqual(len(log["looks"]), 2)
        self.assertEqual(log["looks"][0]["picture"][:5], "shot-")
        start = log["start_pose_look"]
        self.assertTrue(start["ball_whole_in_the_picture"])
        self.assertAlmostEqual(start["joint_angles"][1], pickup.START_POSE[1], delta=0.05)
        self.assertAlmostEqual(start["joint_angles"][0], math.radians(-15) - math.asin(0.03 / 0.17), delta=0.08)  # facing the ball
        self.assertLess(math.dist((start["eye"]["x"], start["eye"]["y"]), ball), 0.025)
        self.assertLess(self.bench.robot.longest_silence_s, QUIET_LIMIT_S)
        self.assertFalse(self.bench.robot.torque_on)

    def test_comparing_does_not_veto_a_ball_the_hand_written_finder_cannot_see(self):
        path = self.with_camera_file(hue_low=100.0, hue_high=140.0, saturation_min=0.3, value_min=0.2, veil_max=90.0)
        try:
            out = self.run_pickup((0.19, 0.03), compare_finders=True, find_only=True)
        finally:
            path.unlink()
        self.assertIn("the hand-written finder sees none in the same picture\n", out, out)
        self.assertIn("place look-pose look 1 finder: none", out)
        self.assertIn("place start-pose finder: none whole=no", out)
        self.assertRegex(out, r"ball seen (19|20|21)\.\d cm from the swivel axis")  # the eye alone still decided

    def test_asking_them_to_agree_needs_a_trained_eye_and_the_camera_file(self):
        out = self.run_pickup((0.2, 0.03), eye_must_agree=True)  # no camera file on this bench
        self.assertIn("does not exist", out)
        self.assertEqual(self.bench.robot.log, [])

    def test_the_eye_works_only_with_still_pictures(self):
        out = self.run_pickup((0.2, 0.03), camera="live")
        self.assertIn("use it with --camera look", out)
        self.assertEqual(self.bench.robot.log, [])



DRIVER = EYE.with_name("driver.onnx")


class ToolWithStudentDrivingTests(unittest.TestCase):
    """``run --camera live --driver``: a trained student moves the arm itself from a stream of pictures. No PyTorch needed."""

    @classmethod
    def setUpClass(cls):
        if not DRIVER.exists():
            raise unittest.SkipTest("no exported student in this checkout")
        try:
            import onnxruntime  # noqa: F401

            from understudy.camera_model import WristCamera

            trained = json.loads(DRIVER.with_suffix(".json").read_text())["student_training"]["nominal_camera"]
            camera = WristCamera(**{key: tuple(value) if isinstance(value, list) else value for key, value in trained.items()})
            cls.temp = tempfile.TemporaryDirectory()
            cls.bench = Bench(Path(cls.temp.name), camera.resized(640, 360))
        except ImportError as error:
            raise unittest.SkipTest(f"needs mujoco and onnxruntime: {error}") from error
        except Exception as error:  # no OpenGL on this machine
            raise unittest.SkipTest(f"cannot render: {error}") from error
        cls.tool = load_tool()
        bench = cls.bench

        class Stream:
            """Stands in for the live camera: the simulated wrist camera's picture, a few steps late, some steps with none."""

            width, height = 640, 360

            def __init__(self, **kwargs):
                self.steps, self.late = 0, []

            def wait_for_first(self, timeout_s: float = 10.0) -> None:
                pass

            def newest(self):
                return bench.view.picture()

            def picture(self):
                self.steps += 1
                if bench.at_step is not None:
                    bench.at_step(self.steps)
                if bench.stops_at is not None and self.steps >= bench.stops_at:
                    return None  # the stream has died
                self.late = [bench.view.picture(), *self.late[:2]]
                if self.steps % 7 == 0:
                    return None  # no new picture this step
                return self.late[-1]  # two control steps old

            def close(self) -> None:
                pass

        bench.stream = Stream

    @classmethod
    def tearDownClass(cls):
        cls.bench.view.close()
        cls.temp.cleanup()

    def setUp(self):
        self.bench.at_step = self.bench.stops_at = None

    def drive(self, ball, **extra):
        self.bench.rest(ball)
        self.bench.robot.log.clear()
        log = self.bench.folder / "run.json"
        for old in self.bench.folder.glob("run*"):
            old.unlink()
        out = self.bench.run(
            self.tool, self.tool.cmd_run, policy=str(POLICY), ball=None, target=[0.19, 0.0, 0.13],
            seconds=15.0, max_speed=0.3, max_lead=0.05, hold_seconds=0.0, log=str(log),
            dry_run=False, camera_latency_steps=3, ffmpeg="ffmpeg",
            **{"camera": "live", "any_distance": False, "find_only": False, "set_down": None, "bowl": None, "eye": None,
               "driver": str(DRIVER), "sweep_deg": 57.0, "eye_must_agree": False, "compare_finders": False, **extra},
        )
        return out, json.loads(log.read_text()) if log.exists() else None

    def test_the_student_drives_from_the_stream_and_lifts_the_ball_and_every_step_is_recorded(self):
        import numpy as np

        distance, bearing = 0.19, math.radians(-25)
        ball = (-0.01125 + distance * math.cos(bearing), distance * math.sin(bearing))
        out, log = self.drive(ball)
        self.assertIn("A trained student, shown the pictures, moves the arm itself", out, out)
        self.assertRegex(out, r"the eye sees the ball \(0\.\d\d sure\)")
        self.assertIn("finished. gripper closed", out, out)
        self.assertGreater(self.bench.highest, pickup.LIFTED_HEIGHT_M)  # lifted, from pictures alone
        self.assertEqual(log["driven_by"], "driver.onnx")
        steps = log["steps"]
        self.assertEqual(len(steps), 450)
        self.assertEqual(steps[0]["ball"], [0.0, 0.0, 0.0])  # the student was never told where the ball is
        for key in ("seen", "sure", "ball_from_memory", "picture_steps_old", "thinking_ms"):
            self.assertIn(key, steps[10])
        self.assertGreater(steps[10]["sure"], 0.5)
        self.assertEqual(max(step["picture_steps_old"] for step in steps), 1)  # every seventh step reused the last picture
        self.assertLess(max(step["thinking_ms"] for step in steps), 33.0)  # well inside a control step
        # No joint was asked to move faster than the limit.
        fastest = max(abs(b - a) for before, after in zip(steps, steps[1:]) for a, b in zip(before["targets"], after["targets"]))
        self.assertLessEqual(fastest * arm.CONTROL_HZ, 0.3 + 1e-3)
        shown = np.load(self.bench.folder / "run.pictures.npz")["pictures"]
        self.assertEqual(shown.shape, (450, 72, 128, 3))
        self.assertIn("pictures the student was shown are in", out)
        self.assertFalse(self.bench.robot.torque_on)
        self.assertLess(self.bench.robot.longest_silence_s, QUIET_LIMIT_S,
                        f"longest silence ended in: {self.bench.robot.longest_silence_at}")

    def test_it_stops_and_the_arm_is_lowered_when_the_ball_is_taken_away(self):
        ball = (0.19, 0.03)

        def take_it_away(step: int) -> None:
            if step == 30:  # a second into the approach
                self.bench.sim.place_ball((0.6, 0.5))

        self.bench.at_step = take_it_away
        out, log = self.drive(ball)
        self.assertRegex(out, r"stopped during the pickup: the student has not been sure of the ball for 20 control steps in a row")
        self.assertIn("The arm is being lowered", out)
        self.assertNotIn("finished. gripper", out)
        self.assertLess(len(log["steps"]), 120)  # stopped within about three seconds of losing it, not after the whole attempt
        self.assertTrue(all(step["gripper_open"] for step in log["steps"]))  # and never closed on nothing
        self.assertIn("torque off, settings restored", out)
        self.assertFalse(self.bench.robot.torque_on)
        self.assertLess(self.bench.robot.longest_silence_s, QUIET_LIMIT_S)

    def test_it_stops_when_the_pictures_stop_coming(self):
        self.bench.stops_at = 40
        out, log = self.drive((0.19, 0.03))
        self.assertRegex(out, r"stopped during the pickup: no new picture has arrived from the camera for 11 control steps")
        self.assertLess(len(log["steps"]), 60)
        self.assertIn("torque off, settings restored", out)
        self.assertFalse(self.bench.robot.torque_on)

    def test_with_no_ball_it_never_starts(self):
        out, log = self.drive((0.6, 0.5))
        self.assertIn("stopped before the pickup: the camera did not see the ball anywhere it looked", out, out)
        self.assertEqual(log["steps"], [])
        self.assertFalse(self.bench.robot.torque_on)

    def test_the_student_drives_only_from_a_stream(self):
        out, log = self.drive((0.19, 0.03), camera="look")
        self.assertIn("use it with --camera live", out)
        self.assertIsNone(log)
        self.assertEqual(self.bench.robot.log, [])


if __name__ == "__main__":
    unittest.main()
