"""The pickup policy's inputs and outputs, outside the training simulator.

The same code turns sensor readings into the policy's input and its output into
servo commands, whether the readings come from a second simulator or the real
arm. Keeping it in one place is what makes a cross-check in another simulator
meaningful for the real arm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import arm, pickup

ACTION_SIZE = 6
INTERFACE_FILE = "interface.json"
_MODES = ("absolute", "relative", "incremental")


@dataclass(frozen=True)
class Interface:
    """How one trained policy's arm outputs become joint targets.

    ``absolute``: target = start pose + scale * output.
    ``relative``: target = the joint's present angle + scale * output.
    ``incremental``: target = the previous target + scale * output, kept within
    ``max_lead`` of the joint's present angle. Such a policy also takes, as five
    extra inputs, how far each previous target is ahead of its joint.

    Saved as ``interface.json`` beside the exported policy.
    """

    arm_action: str = "absolute"
    arm_action_scale: float = 0.5
    max_lead: float = 0.0

    def __post_init__(self) -> None:
        if self.arm_action not in _MODES:
            raise ValueError(f"unknown arm action mode {self.arm_action!r}")
        if self.arm_action_scale <= 0:
            raise ValueError("arm action scale must be positive")
        if self.arm_action == "incremental" and self.max_lead <= 0:
            raise ValueError("an incremental interface needs a positive max_lead")

    @property
    def observation_size(self) -> int:
        return 33 if self.arm_action == "incremental" else 28

    @classmethod
    def current(cls) -> "Interface":
        """The interface policies are trained with now."""
        lead = pickup.ARM_TARGET_MAX_LEAD if pickup.ARM_ACTION_MODE == "incremental" else 0.0
        return cls(pickup.ARM_ACTION_MODE, pickup.ARM_ACTION_SCALE, lead)

    @classmethod
    def beside(cls, policy_path: str | Path) -> "Interface":
        """Read the interface saved with a policy. Policies exported before the file existed were absolute."""
        path = Path(policy_path).parent / INTERFACE_FILE
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(str(data["arm_action"]), float(data["arm_action_scale"]), float(data.get("max_lead", 0.0)))

    def save(self, directory: str | Path) -> None:
        data = {"arm_action": self.arm_action, "arm_action_scale": self.arm_action_scale}
        if self.arm_action == "incremental":
            data["max_lead"] = self.max_lead
        (Path(directory) / INTERFACE_FILE).write_text(json.dumps(data, indent=2) + "\n")

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
    target_lead: Sequence[float] | None = None,
) -> list[float]:
    """Assemble the policy input.

    ``joint_angles`` and ``joint_velocities`` are the five arm joints followed by the
    first gripper finger, in radians and radians per second, in the model's joint frame.
    ``ball_position`` and ``hold_target`` are metres in the arm base frame.
    ``previous_action`` is the policy's last raw output, or zeros on the first step.
    ``target_lead`` is for an incremental interface only: each arm joint's previous
    target minus its present angle.
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
    if target_lead is not None:
        if len(target_lead) != 5:
            raise ValueError(f"target_lead needs 5 values, got {len(target_lead)}")
        observation += list(target_lead)
    return [float(v) for v in observation]


def decode_action(
    action: Sequence[float],
    joint_angles: Sequence[float] | None = None,
    interface: Interface = Interface(),
    previous_targets: Sequence[float] | None = None,
) -> Command:
    """Turn the policy's raw output into joint targets and a gripper command.

    ``joint_angles`` are the present arm joint angles, needed for a relative or
    incremental interface. ``previous_targets`` are the arm targets of the last
    command, needed for an incremental one.
    """
    if len(action) != ACTION_SIZE:
        raise ValueError(f"action needs {ACTION_SIZE} values, got {len(action)}")
    if any(v != v for v in action):
        raise ValueError("action contains NaN")
    scale = interface.arm_action_scale
    if interface.arm_action == "absolute":
        targets = [start + scale * a for start, a in zip(pickup.START_POSE, action[:5])]
    else:
        if joint_angles is None or len(joint_angles) < 5:
            raise ValueError(f"a {interface.arm_action} interface needs the five present arm joint angles")
        present = joint_angles[:5]
        if interface.arm_action == "relative":
            targets = [angle + scale * a for angle, a in zip(present, action[:5])]
        else:
            if previous_targets is None or len(previous_targets) != 5:
                raise ValueError("an incremental interface needs the five previous arm targets")
            lead = interface.max_lead
            targets = [
                min(max(previous + scale * a, angle - lead), angle + lead)
                for previous, a, angle in zip(previous_targets, action[:5], present)
            ]
    return Command(tuple(arm.clamp_to_limits(targets)), gripper_open=action[5] >= 0.0)


def start_command() -> Command:
    """Hold the start pose with the gripper open."""
    return Command(tuple(pickup.START_POSE), gripper_open=True)


class OnnxPolicy:
    """The exported policy. Needs ``onnxruntime`` and ``numpy``."""

    def __init__(self, path: str | Path):
        import numpy
        import onnxruntime

        self.interface = Interface.beside(path)
        self._numpy = numpy
        self._session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        (model_input,) = self._session.get_inputs()
        self._input_name = model_input.name
        expected = self.interface.observation_size
        if model_input.shape[-1] != expected:
            raise ValueError(
                f"{path}: policy expects {model_input.shape[-1]} inputs, but its interface describes {expected}"
            )
        (model_output,) = self._session.get_outputs()
        if model_output.shape[-1] != ACTION_SIZE:
            raise ValueError(f"{path}: policy produces {model_output.shape[-1]} outputs, not {ACTION_SIZE}")

    def __call__(self, observation: Sequence[float]) -> list[float]:
        batch = self._numpy.asarray([observation], dtype=self._numpy.float32)
        return [float(v) for v in self._session.run(None, {self._input_name: batch})[0][0]]
