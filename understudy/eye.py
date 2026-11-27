"""The student's eye on its own: where the ball is, from one still picture and the arm's joint readings.

A trained student (``understudy.student``) says, from each picture alone, where the ball
is and how sure it is that the ball is in the picture. That part is exported to a small
ONNX file by ``scripts/export_eye.py`` and run here with onnxruntime, so the computer at
the arm needs neither PyTorch nor a simulator: numpy, OpenCV and onnxruntime are enough.

    eye = Eye("results/student/<name>/eye.onnx")
    (x, y, z), sure = eye.glance(picture, joint_angles)

The picture may be the camera's full size; it is shrunk to what the student was trained
on. A student trained on a filtered picture is given the same filter here, the one
``understudy.vision.orange_mask`` applies, so simulated and real pictures go through
identical code. ``finder`` wraps an eye for ``understudy.seek``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import arm, pickup, runtime, vision
from .runner import in_turned_frame

# The picture the student network sees: the real camera's 1280 x 720 shrunk ten times.
PICTURE_WIDTH, PICTURE_HEIGHT = 128, 72
PROPRIO_SIZE = 20
# What a student is shown of each picture. "colour" is the picture as it is. The other two put it
# through a fixed filter first, the same for simulated and real pictures, so that what the ball's
# colour happens to be is dealt with once, in the filter: "mask" is only which pixels are the ball's
# colour, and "mask+grey" is that beside the picture in grey, in which the fingers and the desk still show.
INPUTS = {"colour": 3, "mask": 1, "mask+grey": 2}
# The filter's colour test, after the grey veil is taken off. Wide: red through orange to yellow, and
# pale. Other things pass it too; telling them from the ball is the network's job.
FILTER = vision.BallColour(hue_low=0.0, hue_high=55.0, saturation_min=0.25, value_min=0.22, veil_max=90.0)
# The student counts the ball as seen when it is at least this sure.
SURE = 0.5
DEFAULT_HOLD_TARGET = (0.19, 0.0, 0.13)
GUESS_SIZE = 2
# The joint readings the student gets, taken from the teacher's 33 inputs: joint angles relative
# to the start pose (6), joint speeds (6), the hold target (3), and how far each arm joint's target
# leads the joint (5). Left out: the ball's position (3), a constant (4) and the previous output (6).
_FROM_OBSERVATION = (*range(0, 12), *range(15, 18), *range(28, 33))
SPEED_LIMIT_RAD_S = 0.3  # the limit the student's arm outputs are a share of
LEAD_LIMIT_RAD = 0.05
_STEP_RAD = SPEED_LIMIT_RAD_S / arm.CONTROL_HZ
# A student driving from a stream of pictures is stopped when it has not been sure of the ball for this
# many control steps in a row with the gripper still open. Measured in simulation: with the ball taken
# away a second into the approach the student carries on from memory and closes on nothing; this stops
# 19 such attempts in 20, and 1 in 35 of the attempts that would have lifted the ball.
UNSURE_STEPS = 20
# And when no new picture has arrived for this many control steps (a third of a second).
STALE_STEPS = 10
# Asked to check the eye against the hand-written finder, the two must put the ball within this of each other.
# In simulation they are 6 mm apart at the median and 10 mm at the 90th percentile on a real ball.
AGREE_M = 0.025


def proprio_from_observation(observation: Sequence[float]) -> np.ndarray:
    """The student's joint readings from the teacher's 33 inputs. The ball's position is not among them."""
    if len(observation) != 33:
        raise ValueError(f"the teacher's observation has 33 numbers, got {len(observation)}")
    return np.asarray([observation[i] for i in _FROM_OBSERVATION], dtype=np.float32)


def action_from_moves(moves: Sequence[float], gripper_open: bool, scale: float) -> list[float]:
    """A raw output, in the teacher's terms, that produces these moves under the speed limit."""
    return [float(np.clip(m, -1.0, 1.0)) * _STEP_RAD / scale for m in moves] + [1.0 if gripper_open else -1.0]


def shrink(image_bgr: np.ndarray) -> np.ndarray:
    """A real picture from the camera at the size the student sees."""
    import cv2

    if image_bgr.shape[1] * PICTURE_HEIGHT != image_bgr.shape[0] * PICTURE_WIDTH:
        raise ValueError(f"a picture of the camera's 16 by 9 shape is needed, got {image_bgr.shape[1]} x {image_bgr.shape[0]}")
    if image_bgr.shape[1] == PICTURE_WIDTH:
        return image_bgr
    return cv2.resize(image_bgr, (PICTURE_WIDTH, PICTURE_HEIGHT), interpolation=cv2.INTER_AREA)


def shown(picture_bgr: np.ndarray, kind: str) -> np.ndarray:
    """What a student with input ``kind`` is given of a student-sized picture: layers x height x width, 0 to 255."""
    if kind not in INPUTS:
        raise ValueError(f"input must be one of {', '.join(INPUTS)}")
    if kind == "colour":
        return np.ascontiguousarray(picture_bgr.transpose(2, 0, 1), dtype=np.float32)
    layers = [vision.orange_mask(picture_bgr, FILTER).astype(np.float32) * 255.0]
    if kind == "mask+grey":
        layers.append(picture_bgr.astype(np.float32).mean(axis=2))
    return np.stack(layers)


def readings_for(joint_angles: Sequence[float], finger: float = pickup.GRIPPER_OPEN_RAD,
                 hold_target: Sequence[float] = DEFAULT_HOLD_TARGET) -> np.ndarray:
    """The student's joint readings for an arm held still in a pose."""
    readings = np.zeros(PROPRIO_SIZE, dtype=np.float32)
    readings[:6] = np.subtract([*joint_angles, finger], runtime.SENSED_START)
    readings[12:15] = hold_target
    return readings


class Eye:
    """An exported eye. ``path`` is the ``.onnx`` file; the ``.json`` file beside it says what it was trained for."""

    def __init__(self, path: str | Path):
        import onnxruntime

        path = Path(path)
        notes = path.with_suffix(".json")
        if not notes.exists():
            raise ValueError(f"{notes.name} should be beside {path.name}: it says what input the eye takes")
        self.notes = json.loads(notes.read_text())
        self.input = self.notes["input"]
        if self.input not in INPUTS:
            raise ValueError(f"{notes.name} names an input this code does not know: {self.input}")
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def glance(self, picture_bgr: np.ndarray, joint_angles: Sequence[float], finger: float = pickup.GRIPPER_OPEN_RAD,
               hold_target: Sequence[float] = DEFAULT_HOLD_TARGET) -> tuple[tuple[float, float, float], float]:
        """Where the ball is (x, y, z in the frame the joint angles are in) and how sure the eye is that it is in the picture, 0 to 1."""
        layers = shown(shrink(picture_bgr), self.input)
        seen = self._session.run(None, {"shown": layers[None], "readings": readings_for(joint_angles, finger, hold_target)[None]})[0][0]
        return (float(seen[0]), float(seen[1]), float(seen[2])), 1.0 / (1.0 + math.exp(-float(seen[3])))


def finder(eye: Eye, picture: Callable[[], np.ndarray], sure: float = SURE,
           say: Callable[[str], None] = print, second: Callable | None = None, agree_m: float = AGREE_M,
           looks: list | None = None, must_agree: bool = True) -> Callable[..., tuple[float, float, float] | None]:
    """A ball finder for ``understudy.seek``: takes a picture now and asks the eye.

    The returned function is given the arm's joint angles in whatever frame the caller is
    working in and returns the ball's place in that frame, or None if the eye is not sure
    the ball is in the picture. The eye was taught looks from an arm facing straight ahead,
    so a look from a swivelled arm is shown to it as one from straight ahead in a frame
    turned by the swivel, and its answer is turned back.

    ``second`` is another finder to check the eye against: it is given the same picture,
    the same joint angles and the eye's answer, and returns where it puts the ball, or None.
    With it, a ball counts only if both find one and they are within ``agree_m`` of each
    other; what is said tells the two ways of failing apart. With ``must_agree`` false the
    second finder is only reported and recorded, and the eye alone decides. ``looks``, a
    list, is given a record of every look: the joint angles, how sure the eye was, and what
    each finder said.
    """
    def find(joint_angles: Sequence[float], near: Sequence[float] | None = None) -> tuple[float, float, float] | None:
        swivel = float(joint_angles[0])
        frame = picture()
        point, how_sure = eye.glance(frame, (0.0, *joint_angles[1:5]))
        there = in_turned_frame((point[0], point[1], pickup.BALL_RADIUS_M), -swivel)
        look = {"joint_angles": [round(float(a), 4) for a in joint_angles[:5]], "sure": round(how_sure, 3),
                "eye": [round(v, 4) for v in there[:2]]}
        if looks is not None:
            looks.append(look)
        if how_sure < sure:
            say(f"  the eye is not sure there is a ball in this picture ({how_sure:.2f}; it needs {sure:.2f})")
            return None
        if second is None:
            say(f"  the eye sees the ball ({how_sure:.2f} sure)")
            return there
        other = second(frame, joint_angles, there)
        look["other_finder"] = None if other is None else [round(float(v), 4) for v in other[:2]]
        if other is None:
            say(f"  the eye sees a ball ({how_sure:.2f} sure) but the hand-written finder sees none in the same picture"
                + (": not taken as the ball" if must_agree else ""))
            return None if must_agree else there
        apart = math.dist(other[:2], there[:2])
        if apart > agree_m:
            say(f"  the eye ({how_sure:.2f} sure) and the hand-written finder put the ball {apart * 100:.1f} cm apart in the same picture"
                + (f"; at most {agree_m * 100:.1f} cm is allowed: not taken as the ball" if must_agree else ""))
            return None if must_agree else there
        say(f"  the eye sees the ball ({how_sure:.2f} sure) and the hand-written finder agrees to {apart * 1000:.0f} mm")
        return there

    return find


class Driver:
    """An exported whole student: its eye, its memory and its moves. ``path`` is the ``.onnx`` file, with its ``.json`` beside it.

    ``step`` is one control step and keeps the memory; ``glance`` is the eye alone, as
    ``Eye.glance``, and leaves the memory as it is; ``reset`` forgets everything.
    """

    def __init__(self, path: str | Path):
        import onnxruntime

        path = Path(path)
        notes = path.with_suffix(".json")
        if not notes.exists():
            raise ValueError(f"{notes.name} should be beside {path.name}: it says what input the student takes")
        self.notes = json.loads(notes.read_text())
        self.input = self.notes["input"]
        if self.input not in INPUTS:
            raise ValueError(f"{notes.name} names an input this code does not know: {self.input}")
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self._width = int(self.notes["memory"])
        self._state = np.zeros((1, 1, self._width), dtype=np.float32)

    def reset(self) -> None:
        self._state = np.zeros((1, 1, self._width), dtype=np.float32)

    def _run(self, picture_bgr: np.ndarray, readings: np.ndarray, guess: Sequence[float], state: np.ndarray) -> list:
        return self._session.run(None, {
            "shown": shown(shrink(picture_bgr), self.input)[None], "readings": np.asarray(readings, dtype=np.float32)[None],
            "guess": np.asarray([guess[:2]], dtype=np.float32), "state": state})

    def glance(self, picture_bgr: np.ndarray, joint_angles: Sequence[float], finger: float = pickup.GRIPPER_OPEN_RAD,
               hold_target: Sequence[float] = DEFAULT_HOLD_TARGET) -> tuple[tuple[float, float, float], float]:
        seen = self._run(picture_bgr, readings_for(joint_angles, finger, hold_target), (0.0, 0.0), np.zeros_like(self._state))[3][0]
        return (float(seen[0]), float(seen[1]), float(seen[2])), 1.0 / (1.0 + math.exp(-float(seen[3])))

    def step(self, picture_bgr: np.ndarray, readings: np.ndarray, guess: Sequence[float]) -> dict:
        """One control step. ``guess`` is where the ball was taken to be before the attempt (x, y, metres)."""
        moves, is_open, ball, seen, self._state = self._run(picture_bgr, readings, guess, self._state)
        return {
            "moves": moves[0], "open": bool(is_open[0] >= 0.0), "ball": tuple(float(v) for v in ball[0]),
            "seen": tuple(float(v) for v in seen[0, :3]), "in_view": 1.0 / (1.0 + math.exp(-float(seen[0, 3]))),
        }


class NotSure(RuntimeError):
    """The student driving the arm cannot see what it is doing: the run must stop."""


class StreamDriver:
    """Drives the arm from a stream of pictures. Sits where ``understudy.runner.run_pickup`` expects both the
    ball's position and the policy, as ``understudy.dagger.PictureDriver`` does in simulation.

    ``picture`` returns the newest picture, or None if no new one has arrived since it was
    last asked (``understudy.camera_feed.LiveCamera.picture``). The student is never told
    where the ball is: the position reported to the runner is zeros. ``guess`` is where the
    ball was taken to be when the arm turned to face it (x, y in the turned frame).

    It raises ``NotSure``, which stops the run, when the student has not been sure of the
    ball for ``unsure_steps`` control steps in a row with the gripper still open, or when no
    new picture has arrived for ``stale_steps`` steps. A picture that is a step or two late
    is used again; the student was trained on pictures up to three steps old.

    ``thoughts`` records, for every step, what the student saw and how sure it was, and
    ``pictures`` the picture it was given, at the student's size.
    """

    def __init__(self, driver: Driver, picture: Callable[[], np.ndarray | None], guess: Sequence[float],
                 interface: runtime.Interface, sure: float = SURE, unsure_steps: int = UNSURE_STEPS,
                 stale_steps: int = STALE_STEPS, clock: Callable[[], float] | None = None):
        import time

        self.interface = interface
        self._driver, self._picture = driver, picture
        self._guess = (float(guess[0]), float(guess[1]))
        self._sure, self._unsure_steps, self._stale_steps = sure, unsure_steps, stale_steps
        self._clock = clock or time.perf_counter
        self._frame: np.ndarray | None = None
        self._stale = self._unsure = 0
        self.thoughts: list[dict] = []
        self.pictures: list[np.ndarray] = []
        driver.reset()

    def position(self, joint_angles, finger_angle, gripper_open) -> tuple[float, float, float]:
        frame = self._picture()
        if frame is None:
            self._stale += 1
            if self._frame is None or self._stale > self._stale_steps:
                raise NotSure(f"no new picture has arrived from the camera for {self._stale} control steps")
        else:
            self._frame, self._stale = shrink(frame), 0
        return (0.0, 0.0, 0.0)

    def __call__(self, observation: Sequence[float]) -> list[float]:
        started = self._clock()
        thought = self._driver.step(self._frame, proprio_from_observation(observation), self._guess)
        self._unsure = self._unsure + 1 if (thought["open"] and thought["in_view"] < self._sure) else 0
        self.thoughts.append({
            "seen": [round(v, 4) for v in thought["seen"]], "sure": round(thought["in_view"], 3),
            "ball_from_memory": [round(v, 4) for v in thought["ball"]], "picture_steps_old": self._stale,
            "thinking_ms": round(1000.0 * (self._clock() - started), 2),
        })
        self.pictures.append(self._frame)
        if self._unsure >= self._unsure_steps:
            raise NotSure(f"the student has not been sure of the ball for {self._unsure} control steps in a row "
                          f"({self._unsure / arm.CONTROL_HZ:.1f} s) with the gripper still open")
        return action_from_moves(thought["moves"], thought["open"], self.interface.arm_action_scale)

