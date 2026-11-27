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

from . import pickup, runtime, vision
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
           say: Callable[[str], None] = print) -> Callable[..., tuple[float, float, float] | None]:
    """A ball finder for ``understudy.seek``: takes a picture now and asks the eye.

    The returned function is given the arm's joint angles in whatever frame the caller is
    working in and returns the ball's place in that frame, or None if the eye is not sure
    the ball is in the picture. The eye was taught looks from an arm facing straight ahead,
    so a look from a swivelled arm is shown to it as one from straight ahead in a frame
    turned by the swivel, and its answer is turned back.
    """
    def find(joint_angles: Sequence[float], near: Sequence[float] | None = None) -> tuple[float, float, float] | None:
        swivel = float(joint_angles[0])
        point, how_sure = eye.glance(picture(), (0.0, *joint_angles[1:5]))
        if how_sure < sure:
            say(f"  the eye is not sure there is a ball in this picture ({how_sure:.2f}; it needs {sure:.2f})")
            return None
        say(f"  the eye sees the ball ({how_sure:.2f} sure)")
        return in_turned_frame((point[0], point[1], pickup.BALL_RADIUS_M), -swivel)

    return find
