"""The scripts around the student: scoring it on real pictures, and gathering attempts in the simulator."""

import argparse
import importlib.util
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

try:
    import torch

    from understudy import dagger, student
except ImportError as error:  # torch is not among the packages CI installs
    raise unittest.SkipTest(f"needs torch: {error}") from error

import sim_rig
from understudy import arm, camera_calibration, pickup, runtime, vision
from understudy.camera_calibration import Sighting
from understudy.camera_model import WristCamera

REPO = Path(__file__).resolve().parent.parent
POLICY = REPO / "results" / "pickup" / "2026-11-24-claw" / "policy.onnx"
ODD = arm.Calibration((3012.0, 2031.0, 3042.0, 3013.0, 2050.0), (1, -1, 1, -1, 1), 1740.0, -1)
CAMERA = WristCamera.looking_along_gripper(focal_px=540.0, back_m=0.06, up_m=0.035, tilt_down_rad=0.45)


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RealPictureScoringTests(unittest.TestCase):
    """A made-up calibration session, in the files the real one leaves behind."""

    @classmethod
    def setUpClass(cls):
        cls.tool = load_script("student_on_real_pictures")
        cls.temp = tempfile.TemporaryDirectory()
        folder = Path(cls.temp.name)
        (folder / "shots").mkdir()
        rng = random.Random(2)
        cls.places = [(0.21, 0.01), (0.19, 0.06)]
        sightings, number = [], 0
        for place, ball in enumerate(cls.places):
            while sum(s.place == place for s in sightings) < 9:
                pose = [p + rng.uniform(-r, r) for p, r in zip(pickup.START_POSE, (0.3, 0.2, 0.2, 0.25, 0.4))]
                if sim_rig.CHAIN.tip_position(pose)[2] < 0.085:
                    continue
                ticks = tuple(ODD.to_ticks(pose))
                number += 1
                path = folder / "shots" / f"cal-1-{number:03d}.jpg"
                cv2.imwrite(str(path), sim_rig.painted(CAMERA, ODD.to_rad(ticks), [ball]))
                # As on the real arm: the sighting is what the detector finds in the saved picture.
                whole = [b for b in vision.find_blobs(vision.decode(path.read_bytes())) if b.whole()]
                if whole:
                    sightings.append(Sighting(ticks, whole[0].u, whole[0].v, whole[0].radius_px, place))
        cv2.imwrite(str(folder / "shots" / "cal-1-900.jpg"), sim_rig.painted(CAMERA, list(pickup.START_POSE), []))  # empty desk
        cls.sightings = sightings
        camera_calibration.save_sightings(folder / "sightings.json", sightings, colour=vars(vision.BallColour()), width=1280, height=720)
        ODD.save(folder / "calibration.json")
        torch.manual_seed(0)
        student.save(student.StudentNet(), folder / "student.pt", round=0)
        cls.folder = folder
        cls.brain = student.Brain(student.load(folder / "student.pt")[0])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def arguments(self, **extra):
        base = dict(sightings=str(self.folder / "sightings.json"), calibration=str(self.folder / "calibration.json"),
                    shots=str(self.folder / "shots"), pictures=None, angles=None, finger=pickup.GRIPPER_OPEN_RAD)
        return argparse.Namespace(**{**base, **extra})

    def test_every_sighting_is_matched_to_its_picture_and_no_other(self):
        result = self.tool.score_calibration_pictures(self.brain, self.arguments())
        self.assertEqual(result["pictures_matched_to_sightings"], len(self.sightings))
        names = [row["picture"] for row in result["pictures"]]
        self.assertEqual(len(set(names)), len(names))
        self.assertNotIn("cal-1-900.jpg", names)  # the empty desk has no sighting

    def test_the_balls_true_place_is_the_one_the_calibration_fitted(self):
        result = self.tool.score_calibration_pictures(self.brain, self.arguments())
        for fitted, true in zip(result["resting_places"], self.places):
            self.assertLess(math.dist(fitted, true), 0.004)
        for row in result["pictures"]:
            self.assertLess(math.dist(row["ball"], self.places[row["place"]]), 0.004)
            self.assertAlmostEqual(row["error_mm"], 1000 * math.dist(row["student_says"], row["ball"]), delta=0.2)
        self.assertGreaterEqual(result["student_error_mm"]["worst"], result["student_error_mm"]["median"])
        json.dumps(result)  # it can be written out as it is

    def test_the_student_is_shown_the_pose_each_picture_was_taken_from(self):
        asked = []

        class Recording(student.Brain):
            def glance(self, image, readings):
                asked.append((image.shape, readings.copy()))
                return super().glance(image, readings)

        self.tool.score_calibration_pictures(Recording(self.brain.net), self.arguments())
        self.assertEqual(len(asked), len(self.sightings))
        poses = np.array([np.subtract([*ODD.to_rad(s.ticks), pickup.GRIPPER_OPEN_RAD], runtime.SENSED_START)
                          for s in self.sightings])
        used = set()
        for shape, readings in asked:
            self.assertEqual(shape, (72, 128, 3))
            apart = np.abs(poses - readings[:6]).max(axis=1)
            self.assertLess(float(apart.min()), 1e-5)
            used.add(int(apart.argmin()))
            np.testing.assert_allclose(readings[6:12], 0.0)  # the arm was still
        self.assertEqual(len(used), len(self.sightings))  # each picture with its own pose

    def test_pictures_with_no_known_ball_are_only_described(self):
        result = self.tool.describe_pictures(self.brain, self.arguments(pictures=[str(self.folder / "shots" / "cal-1-900.jpg")]))
        self.assertIn("none", result["known_ball_positions"])
        self.assertEqual(result["joint_angles_assumed_rad"], list(pickup.START_POSE))
        self.assertEqual(len(result["pictures"][0]["student_says"]), 3)
        self.assertTrue(0.0 <= result["pictures"][0]["in_view"] <= 1.0)

    def test_a_picture_that_is_not_the_cameras_shape_is_refused(self):
        odd = self.folder / "square.jpg"
        cv2.imwrite(str(odd), np.zeros((480, 480, 3), np.uint8))
        with self.assertRaises(ValueError):
            self.tool.describe_pictures(self.brain, self.arguments(pictures=[str(odd)]))


class SearchTableTests(unittest.TestCase):
    def test_rates_are_given_overall_by_turn_and_by_distance(self):
        tool = load_script("evaluate_search")
        rows = []
        for index in range(40):
            turn = -57 + 114 * index / 39
            rows.append({"found": index % 4 != 0, "lifted": index % 2 == 1, "refused": None, "looks": 1 + index % 3,
                         "facing_error_m": 0.004, "turn_deg": turn, "distance_cm": 15 + 9 * (index % 10) / 9, "stopped": False})
        out = tool.table(rows, 57.0)
        self.assertEqual(out["all"]["attempts"], 40)
        self.assertEqual(out["all"]["found"], 0.75)
        self.assertEqual(out["all"]["lifted_at_the_end"], 0.5)
        self.assertEqual(sum(line["attempts"] for line in out["by_turn_needed_to_face_the_ball"].values()), 40)
        self.assertEqual(sum(line["attempts"] for line in out["by_distance_from_swivel_axis"].values()), 40)
        self.assertEqual(len(out["by_turn_needed_to_face_the_ball"]), 5)
        self.assertEqual(out["all"]["first_guess_error_mm_median"], 4.0)

    def test_a_ball_is_put_where_the_turn_to_face_it_is_the_one_asked_for(self):
        from understudy.runner import SWIVEL_AXIS_X_M, turn_to_face

        tool = load_script("evaluate_search")
        for distance, turn in ((0.15, -0.9), (0.2, 0.0), (0.24, 0.8)):
            x, y = tool.ball_at(distance, turn)
            self.assertAlmostEqual(turn_to_face((x, y)), turn, places=9)
            self.assertAlmostEqual(math.hypot(x - SWIVEL_AXIS_X_M, y), distance, places=9)


class GatheringTests(unittest.TestCase):
    """Attempts in the simulator with rendered pictures. Skipped where nothing can be rendered."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.teacher = runtime.OnnxPolicy(POLICY)
            cls.sim, cls.scene = dagger.bench()
        except ImportError as error:
            raise unittest.SkipTest(f"needs mujoco and onnxruntime: {error}") from error
        except Exception as error:  # no OpenGL on this machine
            raise unittest.SkipTest(f"cannot render: {error}") from error
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.scene.close()

    def test_the_teacher_lifts_the_ball_and_every_step_is_recorded(self):
        rng = random.Random(0)
        lifted = 0
        for _ in range(3):
            outcome, episode = dagger.run_attempt(self.sim, self.scene, rng, teacher=self.teacher, drive="teacher",
                                                  facing="truth", seconds=dagger.TRAINING_SECONDS, record=True)
            lifted += outcome["lifted"]
            arrays = episode.arrays()
            steps = int(dagger.TRAINING_SECONDS * arm.CONTROL_HZ)
            self.assertEqual(arrays["pictures"].shape, (steps, 72, 128, 3))
            self.assertEqual(arrays["proprio"].shape, (steps, student.PROPRIO_SIZE))
            self.assertTrue(bool((np.abs(arrays["moves"]) <= 1.0).all()))
            # No blank frames: a camera that lost the desk as the gripper came down once passed every other check.
            self.assertGreater(float(arrays["pictures"].reshape(steps, -1).std(axis=1).min()), 2.0)
            # With this camera the ball is in the picture while the gripper comes down on it, and
            # mostly out of it once it is held.
            reaching = arrays["open"] > 0.5
            self.assertGreater(float((arrays["visible"][reaching] >= dagger.IN_VIEW).mean()), 0.8)
            # In the turned frame the ball starts about 3 cm left of straight ahead, on the desk,
            # and the first guess the student is given is near it.
            self.assertAlmostEqual(float(arrays["ball"][0, 1]), 0.03, delta=0.03)
            self.assertAlmostEqual(float(arrays["ball"][0, 2]), pickup.BALL_RADIUS_M, delta=0.002)
            self.assertLess(math.dist(arrays["guess"][0], arrays["ball"][0, :2]), 0.04)
            self.assertAlmostEqual(float(arrays["guess"][0, 1]), 0.03, delta=1e-4)
            self.assertTrue(bool(arrays["open"][0]) and not bool(arrays["open"][-1]))
            self.assertGreater(outcome["in_view_from_look_pose"], 0.3)
        self.assertGreaterEqual(lifted, 2)

    def test_the_look_pose_shows_nearly_every_place_a_ball_can_be(self):
        from understudy.kinematics import Chain

        chain, rng, whole = Chain(), random.Random(5), {"look": 0, "start": 0}
        for _ in range(400):
            ball = (*dagger.place_ball(rng), pickup.BALL_RADIUS_M)
            whole["look"] += self.scene.nominal.sight(ball, dagger.LOOK_POSE, chain).inside
            whole["start"] += self.scene.nominal.sight(ball, pickup.START_POSE, chain).inside
        self.assertGreater(whole["look"], 0.9 * 400)
        self.assertLess(whole["start"], 0.75 * 400)  # which is why the ball is not looked for from the start pose

    def test_an_untrained_student_can_drive_while_the_teacher_labels(self):
        torch.manual_seed(0)
        brain = student.Brain(student.StudentNet())
        outcome, episode = dagger.run_attempt(self.sim, self.scene, random.Random(1), teacher=self.teacher, brain=brain,
                                              drive="student", facing="truth", seconds=2.0, record=True)
        self.assertEqual(len(episode), 60)
        self.assertFalse(outcome["lifted"])  # it knows nothing yet
        self.assertGreater(float(np.abs(episode.arrays()["moves"]).mean()), 0.2)  # the teacher's wishes, not the student's

    def test_a_student_that_sees_no_ball_does_not_start(self):
        class Blind(student.Brain):
            def glance(self, image, readings):
                return (0.2, 0.0, 0.02), 0.1

        outcome, episode = dagger.run_attempt(self.sim, self.scene, random.Random(2), brain=Blind(student.StudentNet()),
                                              drive="student", facing="student", seconds=2.0)
        self.assertFalse(outcome["found"])
        self.assertIsNone(episode)

    def test_a_gathering_job_leaves_only_finished_files_for_the_trainer(self):
        with tempfile.TemporaryDirectory() as folder:
            job = {"seed": 77, "spool": folder, "teacher": str(POLICY), "student": None, "attempts": 1,
                   "student_drives": 0.0, "student_faces": 0.0}
            dagger._BENCH.clear()
            dagger._BENCH.update(sim=self.sim, scene=self.scene, teacher=self.teacher, student_path=None, brain=None)
            try:
                outcomes = dagger.gather(job)
                dagger.gather({**job, "seed": 78, "views": 8})
            finally:
                dagger._BENCH.clear()
            names = sorted(path.name for path in Path(folder).iterdir())
            self.assertEqual(names, ["attempt-77-0.npz", "views-78.npz"])  # nothing half written, nothing left over
            self.assertEqual(outcomes[0]["driver"], "teacher")
            with np.load(Path(folder) / "attempt-77-0.npz") as saved:
                self.assertEqual(saved["pictures"].shape[1:], (72, 128, 3))
                self.assertEqual(saved["guess"].shape[1], 2)

    def test_a_half_written_file_is_never_seen_under_its_final_name(self):
        seen = []
        real_savez = np.savez

        def watching(file, **arrays):
            seen.append(sorted(p.name for p in Path(file.name).parent.glob("*.npz")))
            return real_savez(file, **arrays)

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "attempt-1-0.npz"
            np.savez, kept = watching, np.savez
            try:
                dagger._save_whole(target, {"pictures": np.zeros((3, 72, 128, 3), np.uint8)})
            finally:
                np.savez = kept
            self.assertEqual(seen, [[]])  # while it was being written, no .npz file was there to be opened
            self.assertTrue(target.exists())

    def honest_eyes(self):
        """A stand-in for a trained student's eye: says where the ball truly is whenever all of it is in the picture.

        Like the real one it is shown a swivelled look as a look from straight ahead, so it
        answers in the frame turned by the swivel.
        """
        from understudy.runner import in_turned_frame

        sim, scene = self.sim, self.scene

        class Honest(student.Brain):
            def glance(self, image, readings):
                swivel = sim.calibration.to_rad(sim.read().positions[:5])[0]
                where = in_turned_frame(tuple(float(v) for v in sim.ball()), swivel)
                return where, 1.0 if scene.picture_now()[1] > 0.95 else 0.0

        return Honest(student.StudentNet())

    def ball_round(self, distance, turn_deg):
        from understudy.runner import AIM_LEFT_M, SWIVEL_AXIS_X_M

        bearing = math.radians(turn_deg) + math.asin(AIM_LEFT_M / distance)
        return (SWIVEL_AXIS_X_M + distance * math.cos(bearing), distance * math.sin(bearing))

    def test_a_ball_far_round_is_found_by_sweeping_and_faced(self):
        from understudy import seek
        from understudy.kinematics import Chain

        looks = {}
        for turn_deg in (-50.0, 48.0):
            ball = self.ball_round(0.2, turn_deg)
            outcome, _ = dagger.run_attempt(self.sim, self.scene, random.Random(4), brain=self.honest_eyes(), drive="student",
                                            facing="search", seconds=0.5, ball=ball)
            looks[turn_deg] = outcome["looks"]
            with self.subTest(turn=turn_deg):
                self.assertTrue(outcome["found"])
                self.assertIsNone(outcome["refused"])
                # One look is enough only if the ball is in the picture from straight ahead.
                straight = self.scene.nominal.sight((*ball, pickup.BALL_RADIUS_M), seek.LOOK_POSE, Chain())
                if straight is None or not (-100 < straight.u < self.scene.nominal.width + 100):
                    self.assertGreaterEqual(outcome["looks"], 2)
                self.assertLess(outcome["facing_error_m"], 0.002)
                self.assertAlmostEqual(outcome["said_distance_m"], 0.2, delta=0.002)
                # The arm ended up turned to face it.
                swivel = self.sim.calibration.to_rad(self.sim.read().positions[:5])[0]
                self.assertAlmostEqual(math.degrees(swivel), turn_deg, delta=6.0)
        self.assertGreaterEqual(max(looks.values()), 2)  # at least one of the two was out of the first picture

    def test_a_ball_straight_ahead_takes_one_look(self):
        outcome, _ = dagger.run_attempt(self.sim, self.scene, random.Random(5), brain=self.honest_eyes(), drive="student",
                                        facing="search", seconds=0.5, ball=self.ball_round(0.19, 5.0))
        self.assertEqual(outcome["looks"], 1)

    def test_with_no_ball_it_looks_everywhere_and_attempts_nothing(self):
        outcome, episode = dagger.run_attempt(self.sim, self.scene, random.Random(6), brain=self.honest_eyes(), drive="student",
                                              facing="search", seconds=0.5, ball=self.ball_round(0.2, 0.0), ball_present=False)
        from understudy import seek

        self.assertFalse(outcome["found"])
        self.assertEqual(outcome["looks"], len(seek.SCAN))
        self.assertIsNone(episode)
        pose = self.sim.calibration.to_rad(self.sim.read().positions[:5])
        for angle, wanted in zip(pose, (seek.SCAN[-1], *seek.LOOK_POSE[1:])):
            self.assertAlmostEqual(angle, wanted, delta=0.03)  # left raised where it last looked from

    def test_a_ball_well_beyond_reach_is_never_attempted(self):
        # Far enough out, the ball is not whole in any look, and is then simply not found.
        outcome, _ = dagger.run_attempt(self.sim, self.scene, random.Random(9), brain=self.honest_eyes(), drive="student",
                                        facing="search", seconds=0.5, ball=self.ball_round(0.3, 10.0))
        self.assertTrue(not outcome["found"] or outcome["refused"] == "too far")
        self.assertAlmostEqual(float(self.sim.ball()[2]), pickup.BALL_RADIUS_M, delta=0.002)

    def test_a_ball_too_near_or_too_far_is_found_and_refused(self):
        for distance, word in ((0.12, "too near"), (0.255, "too far")):
            outcome, _ = dagger.run_attempt(self.sim, self.scene, random.Random(7), brain=self.honest_eyes(), drive="student",
                                            facing="search", seconds=0.5, ball=self.ball_round(distance, 10.0))
            with self.subTest(distance=distance):
                self.assertTrue(outcome["found"])
                self.assertEqual(outcome["refused"], word)
                self.assertFalse(outcome["lifted"])
                self.assertAlmostEqual(float(self.sim.ball()[2]), pickup.BALL_RADIUS_M, delta=0.002)  # not touched

    def test_a_ball_further_round_than_the_arm_may_turn_is_refused(self):
        outcome, _ = dagger.run_attempt(self.sim, self.scene, random.Random(8), brain=self.honest_eyes(), drive="student",
                                        facing="search", seconds=0.5, ball=self.ball_round(0.2, 75.0))
        self.assertTrue(outcome["found"])  # the sweep looks further round than the arm may turn to face
        self.assertEqual(outcome["refused"], "too far round")

    def test_single_pictures_come_with_the_ball_and_the_pose(self):
        views = dagger.random_views(self.sim, self.scene, random.Random(3), 24)
        self.assertEqual(views["pictures"].shape, (24, 72, 128, 3))
        seen = views["visible"] >= dagger.IN_VIEW
        self.assertTrue(4 <= int(seen.sum()) <= 24)
        np.testing.assert_allclose(views["ball"][:, 2], pickup.BALL_RADIUS_M, atol=1e-6)
        np.testing.assert_allclose(views["proprio"][:, 6:12], 0.0)
        self.assertGreater(float(np.abs(views["proprio"][:, 0]).max()), 0.05)  # poses away from the start pose


class WholeJobTests(unittest.TestCase):
    """A bowl in the scene and in the pictures: the teacher lifts the ball and the fixed carry drops it in."""

    @classmethod
    def setUpClass(cls):
        try:
            from understudy import sim_bowl

            cls.sim_bowl = sim_bowl
            cls.bowl = sim_bowl.Bowl()
            cls.teacher = runtime.OnnxPolicy(POLICY)
            cls.sim, cls.scene = dagger.bench(also=lambda spec: sim_bowl.add_bowl(spec, cls.bowl))
        except ImportError as error:
            raise unittest.SkipTest(f"needs mujoco and onnxruntime: {error}") from error
        except Exception as error:  # no OpenGL on this machine
            raise unittest.SkipTest(f"cannot render: {error}") from error

    @classmethod
    def tearDownClass(cls):
        cls.scene.close()

    def test_the_bowl_is_stood_on_the_desk_after_the_scene_is_put_back_and_the_ball_ends_in_it(self):
        from understudy import carry

        place = carry.SetDown(math.radians(88), 0.24, self.bowl.rim_m, self.bowl.opening_m)
        ball = (0.19, 0.03)
        self.assertIsNone(carry.in_the_way(place, ball))
        outcome, _ = dagger.run_attempt(self.sim, self.scene, random.Random(3), teacher=self.teacher, drive="teacher", facing="exact",
                                        seconds=pickup.EPISODE_SECONDS * 3, ball=ball,
                                        after_reset=lambda: self.sim_bowl.place_bowl(self.sim, place.xy))
        self.assertLess(math.dist(self.sim_bowl.bowl_xy(self.sim), place.xy), 0.001)  # there, and not pushed by the pickup
        self.assertTrue(outcome["lifted"])
        done = carry.carry(self.sim, self.sim.calibration, place, self.sim.advance, say=lambda _: None)
        self.assertTrue(done.done, done.said)
        self.assertTrue(self.sim_bowl.ball_is_in(self.sim, self.bowl))


if __name__ == "__main__":
    unittest.main()
