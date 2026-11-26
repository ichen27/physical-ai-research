"""Teach the student by having it drive while the teacher says what it would have done.

Copying a teacher only from the teacher's own attempts does not work for long:
the student makes small mistakes, ends up somewhere the teacher never was, and
has never been shown what to do there. So after a first round of the teacher's
attempts, the student drives a growing share of them, the teacher (which is
still told where the ball is) labels every step with what it would do from
there, and the student is retrained on everything gathered so far. This is the
DAgger procedure.

An attempt here is what an attempt on the real arm would be: a ball is put down
in front of the arm; the arm rises to a pose that shows everywhere a ball can
be (``LOOK_POSE``) and the ball is found in that one picture; the arm turns to
face it (``understudy.runner.turn_to_face``) and comes down to the start pose;
and the policy runs through ``understudy.runner.run_pickup`` under the real
arm's speed limits, in servo ticks. Only the pictures are simulated, with a new
look each attempt (``understudy.sim_look``).

Needs ``torch``, ``mujoco``, ``numpy``, ``opencv-python-headless`` and OpenGL.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from . import arm, pickup, runtime, student
from .camera_collect import WIDE_RANGE, move_seconds, pose_is_clear
from .kinematics import Chain
from .mujoco_arm import MujocoArm, build_model
from .runner import AIM_LEFT_M, REACH_BAND_M, SWIVEL_AXIS_X_M, in_turned_frame, run_pickup, turn_to_face, turned
from .sim_look import PICTURE_HEIGHT, PICTURE_WIDTH, Scene, dress, student_camera
from .student import BALL_SCALE, GUESS_SIZE, LEAD_LIMIT_RAD, PROPRIO_SIZE, SPEED_LIMIT_RAD_S, Brain, StudentNet

# Balls are put where facing them means turning at most this far.
AHEAD_TURN_RAD = 0.45
# The pose the ball is looked for from: the start pose with the shoulder back 0.3 rad, the elbow
# in 0.4 and the wrist down 0.4, which lifts the camera to about 28 cm. With the first real arm's
# calibrated camera it shows every place a ball can be, whole; the start pose shows 43% of them.
LOOK_POSE = (0.0, -0.9, 0.1, 1.6, 0.0)
# When the arm is faced using the ball's true place instead of the student's guess, the place is
# put this far out at random, about what the student's guess is out by. The student then also
# learns from first guesses that are wrong, and from balls not exactly where facing aims to put them.
FACING_NOISE_M = 0.008
TRAINING_SECONDS = 10.0
IN_VIEW = 0.5  # the share of the ball's disc that must be seen for it to count as in the picture
BURN_IN = 8  # steps at the start of each training window that only warm the memory up


def bench(nominal=None) -> tuple[MujocoArm, Scene]:
    """A simulated arm with a dressed scene and wrist camera, as every attempt here uses.

    ``nominal`` is the camera as it is believed to be (an ``understudy.camera_model.WristCamera``
    at the real picture's full size); left out, it is ``understudy.sim_look.student_camera()``.
    """
    nominal = nominal or student_camera()
    model = build_model(wrist_camera=nominal.resized(2 * PICTURE_WIDTH, 2 * PICTURE_HEIGHT), extras=dress)
    sim = MujocoArm(model, command_delay_s=0.015, profile_time_s=arm.VENDOR_PROFILE_TIME_S)
    return sim, Scene(sim, nominal)


def place_ball(rng: random.Random) -> tuple[float, float]:
    """A ball ahead of the arm: anywhere it lifts from, no further round than ``AHEAD_TURN_RAD``."""
    distance, turn = rng.uniform(*REACH_BAND_M), rng.uniform(-AHEAD_TURN_RAD, AHEAD_TURN_RAD)
    bearing = turn + math.asin(AIM_LEFT_M / distance)
    return (SWIVEL_AXIS_X_M + distance * math.cos(bearing), distance * math.sin(bearing))


@dataclass
class Episode:
    """What was seen and what the teacher would have done, step by step."""

    pictures: list = field(default_factory=list)
    proprio: list = field(default_factory=list)
    moves: list = field(default_factory=list)
    open: list = field(default_factory=list)
    ball: list = field(default_factory=list)
    visible: list = field(default_factory=list)
    guess: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.pictures)

    def arrays(self) -> dict:
        return {
            "pictures": np.asarray(self.pictures, dtype=np.uint8),
            "proprio": np.asarray(self.proprio, dtype=np.float32),
            "moves": np.asarray(self.moves, dtype=np.float32),
            "open": np.asarray(self.open, dtype=np.float32),
            "ball": np.asarray(self.ball, dtype=np.float32),
            "visible": np.asarray(self.visible, dtype=np.float32),
            "guess": np.asarray(self.guess, dtype=np.float32).reshape(len(self.pictures), GUESS_SIZE),
        }


class PictureDriver:
    """Sits where ``run_pickup`` expects both the ball's position and the policy.

    ``run_pickup`` asks for the ball's position and then for the policy's output, every
    step. This takes the picture at the first call and decides the output at the second,
    so the one path from sensors to servo commands is used unchanged.

    With a ``teacher`` it also asks the teacher what it would do, for the record, telling
    it where the ball truly is. ``drive`` says whose output moves the arm. The student is
    never given the ball's position: with no teacher, the position reported is zeros. What
    the student is given, on every step, is ``guess``: where it was taken to be when the
    arm turned to face it (x, y in the turned frame).
    """

    def __init__(self, scene: Scene, interface: runtime.Interface, turn: float, guess: Sequence[float],
                 brain: Brain | None = None,
                 teacher: Callable[[Sequence[float]], Sequence[float]] | None = None, drive: str = "student",
                 record: Episode | None = None, blank: tuple[int, int] | None = None,
                 ball_for_teacher: Callable[[], Sequence[float]] | None = None):
        if drive not in ("student", "teacher"):
            raise ValueError("drive must be 'student' or 'teacher'")
        if drive == "student" and brain is None or drive == "teacher" and teacher is None:
            raise ValueError(f"no {drive} was given to drive")
        self.interface = interface
        self._scene, self._turn, self._brain, self._teacher = scene, turn, brain, teacher
        self._drive, self._record, self._blank = drive, record, blank
        self._ball_for_teacher = ball_for_teacher
        self._guess = (float(guess[0]), float(guess[1]))
        self._teacher_last = [0.0] * runtime.ACTION_SIZE
        self._step = 0
        self._picture, self._visible, self._ball = None, 0.0, (0.0, 0.0, 0.0)
        self.thoughts: list[dict] = []  # the student's outputs, step by step
        if brain is not None:
            brain.reset()

    def position(self, joint_angles, finger_angle, gripper_open):
        self._picture, self._visible = self._scene.picture()
        if self._blank is not None and self._blank[0] <= self._step < self._blank[0] + self._blank[1]:
            self._picture, self._visible = np.full_like(self._picture, 128), 0.0
        self._ball = in_turned_frame(tuple(float(v) for v in self._scene.sim.ball()), self._turn)
        if self._teacher is None:
            return (0.0, 0.0, 0.0)
        return self._ball if self._ball_for_teacher is None else tuple(self._ball_for_teacher())

    def __call__(self, observation: Sequence[float]) -> list[float]:
        readings = student.proprio_from_observation(observation)
        thought = self._brain.step(self._picture, readings, self._guess) if self._brain is not None else None
        if thought is not None:
            thought["picture_showed_ball"] = self._visible
            self.thoughts.append(thought)
        wanted = None
        if self._teacher is not None:
            # The teacher is shown its own last output as the previous one, whoever moved the arm.
            asked = list(observation)
            asked[22:28] = self._teacher_last
            wanted = [float(v) for v in self._teacher(asked)]
            self._teacher_last = wanted
            if self._record is not None:
                moves, is_open = student.moves_from_teacher(wanted, self.interface.arm_action_scale)
                self._record.pictures.append(self._picture)
                self._record.proprio.append(readings)
                self._record.moves.append(moves)
                self._record.open.append(float(is_open))
                self._record.ball.append(self._ball)
                self._record.visible.append(self._visible)
                self._record.guess.append(self._guess)
        self._step += 1
        if self._drive == "teacher":
            return wanted
        return student.action_from_moves(thought["moves"], thought["open"], self.interface.arm_action_scale)


def find_ball(scene: Scene, brain: Brain, hold_target: Sequence[float], pose: Sequence[float] = LOOK_POSE,
              looks: int = 3) -> tuple[float, float] | None:
    """Where the student says the ball is, with the arm still in ``pose``, or None if it does not see one."""
    readings = np.zeros(PROPRIO_SIZE, dtype=np.float32)
    readings[:6] = np.subtract([*pose, pickup.GRIPPER_OPEN_RAD], runtime.SENSED_START)
    readings[12:15] = hold_target
    seen = [brain.glance(scene.picture_now()[0], readings) for _ in range(looks)]
    if float(np.median([sure for _, sure in seen])) < 0.5:
        return None
    return (float(np.median([p[0] for p, _ in seen])), float(np.median([p[1] for p, _ in seen])))


def run_attempt(sim: MujocoArm, scene: Scene, rng: random.Random, teacher=None, brain: Brain | None = None,
                drive: str = "student", facing: str = "student", seconds: float = pickup.EPISODE_SECONDS,
                record: bool = False, blank: tuple[int, int] | None = None, ball_source=None) -> tuple[dict, Episode | None]:
    """One attempt: a ball ahead of the arm, a new look; rise and look for it, face it, come down to the
    start pose, and run the policy with the speed limits.

    ``facing`` is "student" (the arm turns to where the student says the ball is), "truth"
    (to where it is, a few millimetres out at random) or "exact". ``ball_source`` "student"
    feeds a teacher that drives the student's own idea of where the ball is instead of the
    truth: the student's eye with the teacher's hand. Returns what happened and, if asked, the record.
    """
    ball = place_ball(rng)
    target = [rng.uniform(*pickup.HOLD_TARGET[axis]) for axis in "xyz"]
    sim.reset(ball)
    sim.after_step = None
    scene.new_look(rng, ball)
    outcome = {"found": True, "lifted": False, "ever_lifted": False, "stopped": False, "facing_error_m": None}
    calibration = sim.calibration
    gripper = [calibration.finger_to_ticks(pickup.GRIPPER_OPEN_RAD)]
    sim.move_slowly(calibration.to_ticks(LOOK_POSE) + gripper, move_seconds(pickup.START_POSE, LOOK_POSE))
    outcome["in_view_from_look_pose"] = scene.picture_now()[1]
    if facing == "student":
        guess = find_ball(scene, brain, target)
        if guess is None:
            return {**outcome, "found": False}, None
        outcome["facing_error_m"] = math.dist(guess, ball)
    elif facing == "exact":
        guess = ball
    elif facing == "truth":
        guess = (ball[0] + rng.gauss(0, FACING_NOISE_M), ball[1] + rng.gauss(0, FACING_NOISE_M))
    else:
        raise ValueError("facing must be 'student', 'truth' or 'exact'")
    try:
        turn = turn_to_face(guess)
    except ValueError:
        return {**outcome, "found": False}, None
    frame = turned(calibration, turn)
    sim.move_slowly(frame.to_ticks(pickup.START_POSE) + gripper, move_seconds(LOOK_POSE, (turn, *pickup.START_POSE[1:])))
    first = in_turned_frame((*guess, pickup.BALL_RADIUS_M), turn)
    outcome["in_view_from_start_pose"] = scene.picture_now()[1]

    episode = Episode() if record else None
    interface = getattr(teacher, "interface", None) or runtime.Interface.current()
    told = None
    if ball_source == "student":
        # One step late: the student's idea of the ball after the last picture. Before the first
        # picture, where it said the ball was when the arm turned to face it.
        told = lambda: driver.thoughts[-1]["ball"] if driver.thoughts else first  # noqa: E731
    elif ball_source is not None:
        raise ValueError("ball_source must be None or 'student'")
    driver = PictureDriver(scene, interface, turn, first[:2], brain, teacher, drive, episode, blank, told)
    heights: list[float] = []
    sim.after_step = lambda arm_sim: heights.append(float(arm_sim.ball()[2]))
    try:
        run_pickup(sim, driver, frame, driver, target, seconds, advance=sim.advance,
                   max_speed=SPEED_LIMIT_RAD_S, max_lead=LEAD_LIMIT_RAD)
    except RuntimeError:
        outcome["stopped"] = True  # the runner's safety check stopped the arm, as it would on hardware
    sim.after_step = None
    final = int(0.5 * arm.CONTROL_HZ)
    outcome["ever_lifted"] = bool(heights) and max(heights) > pickup.LIFTED_HEIGHT_M
    outcome["lifted"] = not outcome["stopped"] and len(heights) >= final and float(np.mean(heights[-final:])) > pickup.LIFTED_HEIGHT_M
    if driver.thoughts:
        # Where the student put the ball from single pictures, while it lay on the desk in view.
        on_desk = [i for i, h in enumerate(heights[: len(driver.thoughts)])
                   if h < pickup.BALL_RADIUS_M + 0.004 and driver.thoughts[i]["picture_showed_ball"] >= IN_VIEW]
        truth = in_turned_frame((*ball, pickup.BALL_RADIUS_M), turn)
        errors = [math.dist(driver.thoughts[i]["seen"][:2], truth[:2]) for i in on_desk[:30]]
        outcome["seen_error_m"] = float(np.median(errors)) if errors else None
    return outcome, episode


def random_views(sim: MujocoArm, scene: Scene, rng: random.Random, count: int, per_look: int = 4) -> dict:
    """Single pictures from poses all round the start pose and the look pose, each with where the ball is.

    The attempts only show the ball from the poses a pickup passes through. These teach
    the student's eye to place the ball from the pose it is first looked for from, and
    from anywhere the camera calibration looks from, which is what real pictures with
    known ball positions come from.
    """
    import mujoco

    chain = Chain()
    out = Episode()
    while len(out) < count:
        ball = (rng.uniform(0.09, 0.33), rng.uniform(-0.17, 0.17))
        sim.reset(ball)
        scene.new_look(rng, ball)
        for _ in range(per_look):
            about = pickup.START_POSE if rng.random() < 0.5 else LOOK_POSE
            for _ in range(100):
                pose = [s + rng.uniform(-reach, reach) for s, reach in zip(about, WIDE_RANGE)]
                if pose_is_clear(pose, chain):
                    break
            finger = pickup.GRIPPER_OPEN_RAD if rng.random() < 0.7 else rng.uniform(0.0, pickup.GRIPPER_OPEN_RAD)
            for address, angle in zip(sim._qpos, [*pose, finger]):
                sim.data.qpos[address] = angle
            sim.data.qpos[sim._second_finger] = -finger
            mujoco.mj_forward(sim.model, sim.data)
            picture, visible = scene.picture_now()
            readings = np.zeros(PROPRIO_SIZE, dtype=np.float32)
            readings[:6] = np.subtract([*pose, finger], runtime.SENSED_START)
            readings[12:15] = [rng.uniform(*pickup.HOLD_TARGET[axis]) for axis in "xyz"]
            out.pictures.append(picture)
            out.proprio.append(readings)
            out.moves.append(np.zeros(5, dtype=np.float32))
            out.open.append(1.0)
            out.ball.append((*ball, pickup.BALL_RADIUS_M))
            out.visible.append(visible)
            out.guess.append((0.0, 0.0))  # not used: a single picture is read without it
    return out.arrays()


class Experience:
    """Everything gathered so far, with the pictures kept on disk.

    ``folder`` holds one large file of pictures; the small arrays stay in memory. Attempts
    are kept whole and in order, so that training can draw runs of consecutive steps.
    """

    def __init__(self, folder: str | Path, capacity: int):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.pictures = np.lib.format.open_memmap(
            str(self.folder / "pictures.npy"), mode="w+", dtype=np.uint8, shape=(capacity, PICTURE_HEIGHT, PICTURE_WIDTH, 3))
        self.proprio = np.zeros((capacity, PROPRIO_SIZE), dtype=np.float32)
        self.moves = np.zeros((capacity, 5), dtype=np.float32)
        self.open = np.zeros(capacity, dtype=np.float32)
        self.ball = np.zeros((capacity, 3), dtype=np.float32)
        self.visible = np.zeros(capacity, dtype=np.float32)
        self.guess = np.zeros((capacity, GUESS_SIZE), dtype=np.float32)
        self.size = 0
        self.runs: list[tuple[int, int]] = []  # attempts: first step and length
        self.views: list[tuple[int, int]] = []  # blocks of single pictures

    def _add(self, arrays: dict) -> tuple[int, int] | None:
        count = len(arrays["pictures"])
        if count == 0 or self.size + count > self.capacity:
            return None
        where = slice(self.size, self.size + count)
        self.pictures[where] = arrays["pictures"]
        for name in ("proprio", "moves", "open", "ball", "visible", "guess"):
            getattr(self, name)[where] = arrays[name]
        self.size += count
        return (where.start, count)

    def add_attempt(self, arrays: dict) -> bool:
        """Add one attempt. False if it is too short to learn from or there is no room left."""
        if len(arrays["pictures"]) < BURN_IN + 8:
            return False
        added = self._add(arrays)
        if added is not None:
            self.runs.append(added)
        return added is not None

    def add_views(self, arrays: dict) -> bool:
        added = self._add(arrays)
        if added is not None:
            self.views.append(added)
        return added is not None

    def batch(self, rng: np.random.Generator, windows: int, length: int, views: int, blank_share: float = 0.35) -> dict:
        """Training data: ``windows`` runs of ``length`` consecutive steps, and ``views`` single pictures.

        In a share of the runs the picture is blanked for a stretch, with everything else left
        as it was, so that what to do has to come from memory.
        """
        usable = [(start, count) for start, count in self.runs if count >= length]
        if not usable:
            raise ValueError(f"no attempt of {length} steps or more has been gathered")
        index = np.empty((windows, length), dtype=np.int64)
        for row in range(windows):
            start, count = usable[int(rng.integers(len(usable)))]
            index[row] = start + int(rng.integers(count - length + 1)) + np.arange(length)
        flat = index.ravel()
        order = np.argsort(flat)  # read the disk in order
        pictures = np.empty((flat.size, PICTURE_HEIGHT, PICTURE_WIDTH, 3), dtype=np.uint8)
        pictures[order] = self.pictures[flat[order]]
        pictures = pictures.reshape(windows, length, PICTURE_HEIGHT, PICTURE_WIDTH, 3)
        visible = self.visible[index].copy()
        for row in range(windows):
            if rng.random() < blank_share:
                stretch = int(rng.integers(5, 31))
                first = int(rng.integers(0, length - stretch + 1))
                pictures[row, first : first + stretch] = 128
                visible[row, first : first + stretch] = 0.0
        out = {
            "pictures": pictures, "proprio": self.proprio[index], "moves": self.moves[index], "open": self.open[index],
            "ball": self.ball[index], "visible": visible, "guess": self.guess[index],
        }
        if views and self.views:
            total = sum(count for _, count in self.views)
            picks = np.sort(rng.integers(total, size=views))
            starts = np.cumsum([0] + [count for _, count in self.views])
            block = np.searchsorted(starts, picks, side="right") - 1
            where = np.array([self.views[b][0] for b in block]) + picks - starts[block]
            out["view_pictures"] = self.pictures[where]
            out["view_proprio"] = self.proprio[where]
            out["view_ball"] = self.ball[where]
            out["view_visible"] = self.visible[where]
        return out


def _huber(predicted: torch.Tensor, wanted: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    loss = torch.nn.functional.smooth_l1_loss(predicted, wanted, reduction="none", beta=0.5)
    if weight is None:
        return loss.mean()
    weight = weight.expand_as(loss)
    return (loss * weight).sum() / weight.sum().clamp(min=1.0)


def losses(net: StudentNet, batch: dict, device: torch.device) -> dict:
    """How far the student's outputs are from the teacher's and from the truth, on one batch."""
    def tensor(name: str) -> torch.Tensor:
        return torch.from_numpy(batch[name]).to(device)

    pictures = tensor("pictures").permute(0, 1, 4, 2, 3)
    out = net(pictures, tensor("proprio"), tensor("guess"))
    visible, ball = tensor("visible"), tensor("ball")
    in_view = (visible >= IN_VIEW).float()
    after = slice(BURN_IN, None)
    parts = {
        "moves": _huber(out["moves"][:, after], tensor("moves")[:, after]),
        "open": torch.nn.functional.binary_cross_entropy_with_logits(out["open"][:, after], tensor("open")[:, after]),
        "ball": _huber(out["ball"][:, after] * BALL_SCALE, ball[:, after] * BALL_SCALE),
        "seen": _huber(out["seen"][..., :3] * BALL_SCALE, ball * BALL_SCALE, in_view[..., None]),
        "in_view": torch.nn.functional.binary_cross_entropy_with_logits(out["seen"][..., 3], in_view),
    }
    if "view_pictures" in batch:
        _, seen = net.look(tensor("view_pictures").permute(0, 3, 1, 2), tensor("view_proprio"))
        view_in = (tensor("view_visible") >= IN_VIEW).float()
        parts["view_seen"] = _huber(seen[:, :3] * BALL_SCALE, tensor("view_ball") * BALL_SCALE, view_in[:, None])
        parts["view_in_view"] = torch.nn.functional.binary_cross_entropy_with_logits(seen[:, 3], view_in)
    weights = {"moves": 1.0, "open": 0.5, "ball": 4.0, "seen": 4.0, "in_view": 0.5, "view_seen": 4.0, "view_in_view": 0.5}
    parts["total"] = sum(weights[name] * value for name, value in parts.items())
    return parts


# Gathering in several processes -------------------------------------------------

_BENCH: dict = {}


def gather(job: dict) -> list[dict]:
    """Run one batch of attempts (or take single pictures) in this process and save them for the trainer.

    ``job`` has: ``seed``; ``spool`` (folder to save into); ``teacher`` (path of the policy);
    ``camera`` (a wrist camera file to take as nominal, or None for the built-in one);
    ``student`` (path of the student so far, or None); ``attempts``; ``student_drives`` (the
    share of attempts the student drives); ``student_faces`` (the share of those in which
    the arm is faced by the student's own guess); and ``views`` (single pictures to take
    instead of attempts). Returns what happened in each attempt.
    """
    torch.set_num_threads(1)
    if "sim" not in _BENCH:
        from .camera_model import WristCamera

        _BENCH["sim"], _BENCH["scene"] = bench(WristCamera.load(job["camera"]) if job.get("camera") else None)
        _BENCH["teacher"] = runtime.OnnxPolicy(job["teacher"])
        _BENCH["student_path"], _BENCH["brain"] = None, None
    if job.get("student") != _BENCH["student_path"]:
        _BENCH["student_path"] = job.get("student")
        _BENCH["brain"] = Brain(student.load(job["student"])[0]) if job.get("student") else None
    sim, scene, teacher, brain = _BENCH["sim"], _BENCH["scene"], _BENCH["teacher"], _BENCH["brain"]
    rng = random.Random(job["seed"])
    spool = Path(job["spool"])
    if job.get("views"):
        _save_whole(spool / f"views-{job['seed']}.npz", random_views(sim, scene, rng, job["views"]))
        return []
    outcomes = []
    for number in range(job["attempts"]):
        student_drives = brain is not None and rng.random() < job["student_drives"]
        faces = "student" if student_drives and rng.random() < job["student_faces"] else "truth"
        outcome, episode = run_attempt(
            sim, scene, rng, teacher=teacher, brain=brain if student_drives else None,
            drive="student" if student_drives else "teacher", facing=faces, seconds=TRAINING_SECONDS, record=True,
        )
        outcome["driver"] = "student" if student_drives else "teacher"
        outcome["faced_by"] = faces
        outcomes.append(outcome)
        if episode is not None and len(episode):
            _save_whole(spool / f"attempt-{job['seed']}-{number}.npz", episode.arrays())
    return outcomes


def _save_whole(path: Path, arrays: dict) -> None:
    """Save under another name and rename when finished, so the trainer never opens half a file.

    The trainer takes in whatever ``.npz`` files it finds whenever any worker finishes a job;
    a file another worker was still writing once stopped a run a few minutes in.
    """
    part = path.with_suffix(".part")
    with part.open("wb") as file:
        np.savez(file, **arrays)
    part.replace(path)
