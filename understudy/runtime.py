"""The pickup policy's inputs and outputs, outside the training simulator.

The same code turns sensor readings into the policy's input and its output into
servo commands, whether the readings come from a second simulator or the real
arm. Keeping it in one place is what makes a cross-check in another simulator
meaningful for the real arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import arm, pickup

OBSERVATION_SIZE = 28
ACTION_SIZE = 6
ARM_ACTION_SCALE = 0.5
# Orientation part of the hold target. The policy was trained with a constant zero rotation (x, y, z, w).
TARGET_QUATERNION = (0.0, 0.0, 0.0, 1.0)

# The six joints the policy senses: the five arm joints, then the first gripper finger.
SENSED_START = (*pickup.START_POSE, pickup.GRIPPER_OPEN_RAD)


@dataclass(frozen=True)
class Command:
    arm_targets: tuple[float, ...]  # rad, five arm joints, inside the trained limits
    gripper_open: bool

    @property
    def finger_target(self) -> float:
        return pickup.GRIPPER_OPEN_RAD if self.gripper_open else pickup.GRIPPER_CLOSED_RAD


def build_observation(
    joint_angles: Sequence[float],
    joint_velocities: Sequence[float],
    ball_position: Sequence[float],
    hold_target: Sequence[float],
    previous_action: Sequence[float],
) -> list[float]:
    """Assemble the policy input.

    ``joint_angles`` and ``joint_velocities`` are the five arm joints followed by the
    first gripper finger, in radians and radians per second, in the model's joint frame.
    ``ball_position`` and ``hold_target`` are metres in the arm base frame.
    ``previous_action`` is the policy's last raw output, or zeros on the first step.
    """
    for name, values, size in (
        ("joint_angles", joint_angles, 6),
        ("joint_velocities", joint_velocities, 6),
        ("ball_position", ball_position, 3),
        ("hold_target", hold_target, 3),
        ("previous_action", previous_action, ACTION_SIZE),
    ):
        if len(values) != size:
            raise ValueError(f"{name} needs {size} values, got {len(values)}")
    observation = [angle - start for angle, start in zip(joint_angles, SENSED_START)]
    observation += list(joint_velocities)
    observation += list(ball_position)
    observation += list(hold_target) + list(TARGET_QUATERNION)
    observation += list(previous_action)
    return [float(v) for v in observation]


def decode_action(action: Sequence[float]) -> Command:
    """Turn the policy's raw output into joint targets and a gripper command."""
    if len(action) != ACTION_SIZE:
        raise ValueError(f"action needs {ACTION_SIZE} values, got {len(action)}")
    if any(v != v for v in action):
        raise ValueError("action contains NaN")
    targets = [start + ARM_ACTION_SCALE * a for start, a in zip(pickup.START_POSE, action[:5])]
    return Command(tuple(arm.clamp_to_limits(targets)), gripper_open=action[5] >= 0.0)


class OnnxPolicy:
    """The exported policy. Needs ``onnxruntime`` and ``numpy``."""

    def __init__(self, path: str | Path):
        import numpy
        import onnxruntime

        self._numpy = numpy
        self._session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        (model_input,) = self._session.get_inputs()
        self._input_name = model_input.name
        if model_input.shape[-1] != OBSERVATION_SIZE:
            raise ValueError(f"{path}: policy expects {model_input.shape[-1]} inputs, not {OBSERVATION_SIZE}")
        (model_output,) = self._session.get_outputs()
        if model_output.shape[-1] != ACTION_SIZE:
            raise ValueError(f"{path}: policy produces {model_output.shape[-1]} outputs, not {ACTION_SIZE}")

    def __call__(self, observation: Sequence[float]) -> list[float]:
        batch = self._numpy.asarray([observation], dtype=self._numpy.float32)
        return [float(v) for v in self._session.run(None, {self._input_name: batch})[0][0]]
