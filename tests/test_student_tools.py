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

    def test_single_pictures_come_with_the_ball_and_the_pose(self):
        views = dagger.random_views(self.sim, self.scene, random.Random(3), 24)
        self.assertEqual(views["pictures"].shape, (24, 72, 128, 3))
        seen = views["visible"] >= dagger.IN_VIEW
        self.assertTrue(4 <= int(seen.sum()) <= 24)
        np.testing.assert_allclose(views["ball"][:, 2], pickup.BALL_RADIUS_M, atol=1e-6)
        np.testing.assert_allclose(views["proprio"][:, 6:12], 0.0)
        self.assertGreater(float(np.abs(views["proprio"][:, 0]).max()), 0.05)  # poses away from the start pose


if __name__ == "__main__":
    unittest.main()
