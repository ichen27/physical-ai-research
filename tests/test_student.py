"""The picture-driven student and the code that teaches it. Needs torch; skipped without it."""

import math
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch

    from understudy import dagger, student
except ImportError as error:  # torch is not among the packages CI installs
    raise unittest.SkipTest(f"needs torch: {error}") from error

from understudy import arm, pickup, runtime
from understudy.runner import ArmReading, goals_from_command, run_pickup
from understudy.sim_look import PICTURE_HEIGHT, PICTURE_WIDTH

SCALE = 0.1  # the teacher's action scale
GUESS = (0.2, 0.03)  # where the ball was taken to be before the attempt
LIMITS = dict(max_speed=student.SPEED_LIMIT_RAD_S, max_lead=student.LEAD_LIMIT_RAD)
INTERFACE = runtime.Interface("incremental", SCALE, 0.35)


def picture(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (PICTURE_HEIGHT, PICTURE_WIDTH, 3), dtype=np.uint8)


class JumpArm:
    """Servos that jump straight to their goals."""

    def __init__(self):
        self.calibration = arm.Calibration.stock()
        self.positions = goals_from_command(runtime.start_command(), self.calibration)
        self.written = []

    def read(self):
        return ArmReading(tuple(self.positions), (0,) * 6)

    def write_goals(self, ticks):
        self.written.append(list(ticks))
        self.positions = list(ticks)


class Still:
    def position(self, joint_angles, finger_angle, gripper_open):
        return (0.2, 0.03, pickup.BALL_RADIUS_M)


class ReadingsTests(unittest.TestCase):
    def test_the_student_is_not_given_the_ball(self):
        observation = [float(i) for i in range(33)]
        elsewhere = list(observation)
        elsewhere[12:15] = [9.0, 9.0, 9.0]  # the ball's position
        elsewhere[22:28] = [7.0] * 6  # and the previous output
        np.testing.assert_array_equal(student.proprio_from_observation(observation), student.proprio_from_observation(elsewhere))
        self.assertEqual(len(student.proprio_from_observation(observation)), student.PROPRIO_SIZE)
        with self.assertRaises(ValueError):
            student.proprio_from_observation([0.0] * 28)

    def test_the_readings_are_the_ones_the_teacher_gets(self):
        observation = runtime.build_observation(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1, 2, 3, 4, 5, 6], [0.2, 0.0, 0.02], [0.19, 0.01, 0.13], [0.0] * 6, [0.01] * 5)
        readings = student.proprio_from_observation(observation)
        np.testing.assert_allclose(readings[:6], np.subtract([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], runtime.SENSED_START), atol=1e-6)
        np.testing.assert_allclose(readings[6:12], [1, 2, 3, 4, 5, 6])
        np.testing.assert_allclose(readings[12:15], [0.19, 0.01, 0.13], atol=1e-6)
        np.testing.assert_allclose(readings[15:], [0.01] * 5, atol=1e-6)


class MovesTests(unittest.TestCase):
    def test_what_the_student_learns_moves_the_arm_exactly_as_the_teacher_would_under_the_limits(self):
        # The teacher's raw outputs are mostly far beyond what the speed limit lets through. The
        # student learns their limited effect; sent through the same runner, that must come to
        # the same servo goals, step after step.
        rng = random.Random(4)
        raw = [[rng.choice((rng.uniform(-0.12, 0.12), rng.uniform(-30, 30))) for _ in range(5)] + [rng.uniform(-20, 5)]
               for _ in range(90)]

        def played(convert):
            actions = iter(raw)

            def policy(observation):
                action = next(actions)
                return convert(action) if convert else action

            policy.interface = INTERFACE
            jump = JumpArm()
            run_pickup(jump, policy, jump.calibration, Still(), [0.19, 0.0, 0.13], 3.0, advance=lambda: None, **LIMITS)
            return jump.written

        as_student = played(lambda a: student.action_from_moves(*student.moves_from_teacher(a, SCALE), SCALE))
        self.assertEqual(played(None), as_student)
        self.assertEqual(len(as_student), 90)

    def test_moves_are_shares_of_the_allowed_step(self):
        moves, is_open = student.moves_from_teacher([0.05, -0.05, 24.0, -11.0, 0.0, -13.0], SCALE)
        np.testing.assert_allclose(moves, [0.5, -0.5, 1.0, -1.0, 0.0], atol=1e-6)
        self.assertFalse(is_open)
        self.assertTrue(student.moves_from_teacher([0, 0, 0, 0, 0, 0.0], SCALE)[1])


class NetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = student.StudentNet()

    def test_shapes(self):
        out = self.net(torch.zeros(2, 5, 3, PICTURE_HEIGHT, PICTURE_WIDTH), torch.zeros(2, 5, student.PROPRIO_SIZE),
                       torch.zeros(2, 5, student.GUESS_SIZE))
        self.assertEqual(out["moves"].shape, (2, 5, 5))
        self.assertEqual(out["open"].shape, (2, 5))
        self.assertEqual(out["ball"].shape, (2, 5, 3))
        self.assertEqual(out["seen"].shape, (2, 5, 4))
        self.assertTrue(bool((out["moves"].abs() <= 1).all()))

    def test_memory_carries_from_step_to_step_and_reset_clears_it(self):
        brain = student.Brain(self.net)
        readings = np.zeros(student.PROPRIO_SIZE, dtype=np.float32)
        first = brain.step(picture(1), readings, GUESS)
        after_other = brain.step(picture(2), readings, GUESS)
        brain.reset()
        fresh = brain.step(picture(2), readings, GUESS)
        self.assertFalse(np.allclose(after_other["moves"], fresh["moves"]))  # what came before matters
        brain.reset()
        np.testing.assert_allclose(brain.step(picture(1), readings, GUESS)["moves"], first["moves"], atol=1e-6)
        # What one picture alone says does not depend on what came before.
        np.testing.assert_allclose(after_other["seen"], fresh["seen"], atol=1e-6)

    def test_a_glance_leaves_the_memory_alone(self):
        brain = student.Brain(self.net)
        readings = np.zeros(student.PROPRIO_SIZE, dtype=np.float32)
        brain.step(picture(1), readings, GUESS)
        position, sure = brain.glance(picture(3), readings)
        self.assertEqual(len(position), 3)
        self.assertTrue(0.0 <= sure <= 1.0)
        with_glance = brain.step(picture(2), readings, GUESS)
        brain.reset()
        brain.step(picture(1), readings, GUESS)
        np.testing.assert_allclose(brain.step(picture(2), readings, GUESS)["moves"], with_glance["moves"], atol=1e-6)

    def test_the_first_guess_steers_the_outputs_but_never_what_one_picture_says(self):
        readings = np.zeros(student.PROPRIO_SIZE, dtype=np.float32)
        near, far = student.Brain(self.net), student.Brain(self.net)
        a, b = near.step(picture(1), readings, (0.15, 0.03)), far.step(picture(1), readings, (0.24, 0.03))
        self.assertFalse(np.allclose(a["moves"], b["moves"]))
        np.testing.assert_allclose(a["seen"], b["seen"], atol=1e-7)
        self.assertEqual(a["in_view"], b["in_view"])

    def test_a_picture_of_the_wrong_size_is_refused(self):
        with self.assertRaises(ValueError):
            student.Brain(self.net).step(np.zeros((36, 64, 3), np.uint8), np.zeros(student.PROPRIO_SIZE, np.float32), GUESS)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "student.pt"
            student.save(self.net, path, round=3)
            loaded, notes = student.load(path)
        self.assertEqual(notes, {"round": 3})
        readings = np.zeros(student.PROPRIO_SIZE, dtype=np.float32)
        np.testing.assert_allclose(student.Brain(loaded).step(picture(5), readings, GUESS)["moves"],
                                   student.Brain(self.net).step(picture(5), readings, GUESS)["moves"], atol=1e-6)


class StubScene:
    """Stands in for the simulated camera: a fixed picture, and a ball that is where it is said to be."""

    class _Sim:
        def __init__(self, ball):
            self._ball = ball

        def ball(self):
            return np.array(self._ball)

    def __init__(self, ball=(0.2, 0.05, pickup.BALL_RADIUS_M)):
        self.sim = self._Sim(ball)
        self.pictures = 0

    def picture(self):
        self.pictures += 1
        return picture(self.pictures), 1.0


class DriverTests(unittest.TestCase):
    def run_with(self, steps=12, **kwargs):
        jump = JumpArm()
        scene = StubScene()
        driver = dagger.PictureDriver(scene, INTERFACE, guess=GUESS, **kwargs)
        taken = run_pickup(jump, driver, jump.calibration, driver, [0.19, 0.0, 0.13], steps / arm.CONTROL_HZ,
                           advance=lambda: None, **LIMITS)
        return driver, taken, scene

    def test_a_student_driving_alone_is_never_told_where_the_ball_is(self):
        torch.manual_seed(1)
        driver, taken, scene = self.run_with(turn=0.0, brain=student.Brain(student.StudentNet()), drive="student")
        self.assertTrue(all(step.ball_position == (0.0, 0.0, 0.0) for step in taken))
        self.assertEqual(scene.pictures, len(taken))
        self.assertEqual(len(driver.thoughts), len(taken))

    def test_the_teacher_is_told_the_truth_in_the_turned_frame_and_shown_its_own_last_output(self):
        asked = []

        def teacher(observation):
            asked.append(list(observation))
            return [0.05 * len(asked), -30.0, 0.0, 0.0, 0.0, 1.0]

        record = dagger.Episode()
        turn = 0.3
        torch.manual_seed(1)
        driver, taken, _ = self.run_with(turn=turn, brain=student.Brain(student.StudentNet()), teacher=teacher,
                                         drive="student", record=record)
        from understudy.runner import in_turned_frame

        truth = in_turned_frame((0.2, 0.05, pickup.BALL_RADIUS_M), turn)
        np.testing.assert_allclose(asked[0][12:15], truth, atol=1e-6)
        # The arm was moved by the student, but the teacher sees its own previous output, not the student's.
        np.testing.assert_allclose(asked[3][22:28], [0.05 * 3, -30.0, 0.0, 0.0, 0.0, 1.0], atol=1e-6)
        self.assertEqual(len(record), len(taken))
        arrays = record.arrays()
        np.testing.assert_allclose(arrays["moves"][0], [0.5, -1.0, 0.0, 0.0, 0.0], atol=1e-6)
        self.assertTrue(bool(arrays["open"].all()))
        np.testing.assert_allclose(arrays["ball"][0], truth, atol=1e-6)
        self.assertEqual(arrays["pictures"].shape, (len(taken), PICTURE_HEIGHT, PICTURE_WIDTH, 3))
        np.testing.assert_allclose(arrays["guess"], np.tile(GUESS, (len(taken), 1)), atol=1e-6)

    def test_a_blanked_stretch_reaches_the_student_as_grey(self):
        seen = []

        class Watching(student.Brain):
            def step(self, image, readings, guess):
                seen.append(image.copy())
                return super().step(image, readings, guess)

        torch.manual_seed(1)
        self.run_with(turn=0.0, brain=Watching(student.StudentNet()), drive="student", blank=(4, 3))
        blanked = [bool((image == 128).all()) for image in seen]
        self.assertEqual(blanked, [False] * 4 + [True] * 3 + [False] * 5)

    def test_someone_must_be_there_to_drive(self):
        with self.assertRaises(ValueError):
            dagger.PictureDriver(StubScene(), INTERFACE, 0.0, GUESS, drive="student")
        with self.assertRaises(ValueError):
            dagger.PictureDriver(StubScene(), INTERFACE, 0.0, GUESS, brain=student.Brain(student.StudentNet()), drive="teacher")


def attempt(rng: np.random.Generator, steps: int, ball=(0.2, 0.03, 0.02)) -> dict:
    return {
        "pictures": rng.integers(0, 255, (steps, PICTURE_HEIGHT, PICTURE_WIDTH, 3), dtype=np.uint8),
        "proprio": rng.normal(0, 0.1, (steps, student.PROPRIO_SIZE)).astype(np.float32),
        "moves": rng.uniform(-1, 1, (steps, 5)).astype(np.float32),
        "open": (np.arange(steps) < steps // 3).astype(np.float32),
        "ball": np.tile(np.asarray(ball, dtype=np.float32), (steps, 1)),
        "visible": np.ones(steps, dtype=np.float32),
        "guess": np.tile(np.asarray(ball[:2], dtype=np.float32), (steps, 1)),
    }


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.rng = np.random.default_rng(0)
        self.experience = dagger.Experience(self.temp.name, capacity=400)

    def tearDown(self):
        del self.experience
        self.temp.cleanup()

    def test_runs_are_consecutive_steps_of_one_attempt(self):
        first, second = attempt(self.rng, 60), attempt(self.rng, 80)
        first["proprio"][:, 0] = np.arange(60)  # mark each step with its number
        second["proprio"][:, 0] = 1000 + np.arange(80)
        self.assertTrue(self.experience.add_attempt(first))
        self.assertTrue(self.experience.add_attempt(second))
        batch = self.experience.batch(self.rng, windows=12, length=40, views=0, blank_share=0.0)
        self.assertEqual(batch["pictures"].shape, (12, 40, PICTURE_HEIGHT, PICTURE_WIDTH, 3))
        marks = batch["proprio"][:, :, 0]
        np.testing.assert_array_equal(np.diff(marks, axis=1), np.ones((12, 39)))  # never across two attempts
        # The pictures drawn are the ones that belong to those steps.
        row = int(marks[0, 0])
        source = first if row < 1000 else second
        np.testing.assert_array_equal(batch["pictures"][0, 0], source["pictures"][row % 1000])

    def test_a_blanked_stretch_is_grey_and_marked_out_of_view(self):
        self.experience.add_attempt(attempt(self.rng, 100))
        batch = self.experience.batch(self.rng, windows=8, length=48, views=0, blank_share=1.0)
        for row in range(8):
            grey = (batch["pictures"][row].reshape(48, -1) == 128).all(axis=1)
            self.assertTrue(5 <= int(grey.sum()) <= 30)
            np.testing.assert_array_equal(batch["visible"][row][grey], 0.0)
            np.testing.assert_array_equal(batch["visible"][row][~grey], 1.0)
            self.assertEqual(int(np.abs(np.diff(grey.astype(int))).sum()), 2 - int(grey[0]) - int(grey[-1]))  # one stretch
        np.testing.assert_array_equal(self.experience.visible[:100], 1.0)  # what is stored is left alone

    def test_single_pictures_are_drawn_from_the_views(self):
        self.experience.add_attempt(attempt(self.rng, 60))
        views = attempt(self.rng, 50)
        views["proprio"][:, 0] = 5000 + np.arange(50)
        self.assertTrue(self.experience.add_views(views))
        batch = self.experience.batch(self.rng, windows=2, length=40, views=30)
        self.assertEqual(batch["view_pictures"].shape, (30, PICTURE_HEIGHT, PICTURE_WIDTH, 3))
        self.assertTrue(bool((batch["view_proprio"][:, 0] >= 5000).all()))

    def test_refuses_what_does_not_fit_or_is_too_short(self):
        self.assertFalse(self.experience.add_attempt(attempt(self.rng, 5)))
        self.assertTrue(self.experience.add_attempt(attempt(self.rng, 300)))
        self.assertFalse(self.experience.add_attempt(attempt(self.rng, 200)))  # only 100 steps of room left
        self.assertEqual(self.experience.size, 300)
        with self.assertRaises(ValueError):
            dagger.Experience(Path(self.temp.name) / "other", 100).batch(self.rng, 2, 40, 0)


class LearningTests(unittest.TestCase):
    def test_the_student_learns_a_small_lesson(self):
        # Twelve attempts in which the right moves and the ball's place can be read off where a
        # bright square sits in the picture. Training must bring every part of the loss down. The
        # ball's place is learnt first, since it weighs most, and the moves after it: by step 160.
        torch.manual_seed(0)
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as folder:
            experience = dagger.Experience(folder, capacity=1000)
            for _ in range(12):
                level = float(rng.uniform(-1, 1))
                data = attempt(rng, 60, ball=(0.2 + 0.03 * level, 0.03 * level, 0.02))
                column = int(58 + 40 * level)
                data["pictures"][:] = 30
                data["pictures"][:, 30:42, column : column + 12] = 230
                data["moves"][:] = level
                data["open"][:] = float(level > 0)
                experience.add_attempt(data)
            experience.add_views(attempt(rng, 40))
            net = student.StudentNet()
            optimiser = torch.optim.Adam(net.parameters(), lr=2e-3)
            device = torch.device("cpu")
            history = []
            for _ in range(160):
                parts = dagger.losses(net, experience.batch(rng, 4, 24, 16, blank_share=0.0), device)
                optimiser.zero_grad()
                parts["total"].backward()
                optimiser.step()
                history.append({name: float(value.detach()) for name, value in parts.items()})
            del experience
        self.assertTrue(all(math.isfinite(v) for step in history for v in step.values()))
        early = {name: np.mean([h[name] for h in history[:5]]) for name in history[0]}
        late = {name: np.mean([h[name] for h in history[-5:]]) for name in history[0]}
        for name in ("moves", "ball", "seen", "total"):
            self.assertLess(late[name], 0.6 * early[name], name)


class PlacementTests(unittest.TestCase):
    def test_balls_are_put_ahead_of_the_arm_where_it_can_lift_them(self):
        from understudy.runner import REACH_BAND_M, SWIVEL_AXIS_X_M, turn_to_face

        rng = random.Random(0)
        for _ in range(300):
            x, y = dagger.place_ball(rng)
            distance = math.hypot(x - SWIVEL_AXIS_X_M, y)
            self.assertTrue(REACH_BAND_M[0] - 1e-9 <= distance <= REACH_BAND_M[1] + 1e-9)
            self.assertLessEqual(abs(turn_to_face((x, y))), dagger.AHEAD_TURN_RAD + 1e-9)


if __name__ == "__main__":
    unittest.main()
