"""The vendor's MuJoCo model of the arm, driven through the same interface as the real servos.

Needs the ``mujoco`` and ``numpy`` packages. Readings come back as encoder ticks
and velocity counts, rounded as a real servo reports them, and goals are taken in
ticks, so everything between the policy and the servo bus is exercised.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

from . import arm, pickup, runtime
from .runner import VELOCITY_UNIT_RAD_S, ArmReading

VENDOR_MODEL = (
    Path(__file__).resolve().parent.parent / "assets" / "open_manipulator_description" / "mujoco" / "omx" / "omx.xml"
)
SUBSTEPS = 16
TIMESTEP = 1.0 / (arm.CONTROL_HZ * SUBSTEPS)


def build_model(
    ball_mass: float = pickup.BALL_MASS_KG, ball_friction: float = pickup.BALL_FRICTION, stiffness_scale: float = 1.0
) -> mujoco.MjModel:
    """The vendor arm on a desk with a ball. ``stiffness_scale`` multiplies the arm servo gains."""
    if stiffness_scale <= 0:
        raise ValueError("stiffness_scale must be positive")
    spec = mujoco.MjSpec.from_file(str(VENDOR_MODEL))
    spec.option.timestep = TIMESTEP
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 960, 720
    world = spec.worldbody
    world.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1])
    world.add_geom(name="desk", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], rgba=[0.35, 0.35, 0.38, 1])
    ball = world.add_body(name="ball", pos=[0.21, 0.0, pickup.BALL_RADIUS_M])
    ball.add_freejoint()
    ball.add_geom(
        name="ball",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[pickup.BALL_RADIUS_M, 0, 0],
        mass=ball_mass,
        friction=[ball_friction, 0.005, 0.0001],
        rgba=[1.0, 0.4, 0.0, 1.0],
    )
    model = spec.compile()
    for joint in arm.ARM_JOINTS:
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint.name)
        model.actuator_gainprm[index, 0] *= stiffness_scale
        model.actuator_biasprm[index, 1] *= stiffness_scale
        model.actuator_biasprm[index, 2] *= math.sqrt(stiffness_scale)  # keeps the damping ratio
    return model


class MujocoArm:
    """Implements ``understudy.runner.ArmIO`` on the simulated arm."""

    def __init__(self, model: mujoco.MjModel, calibration: arm.Calibration | None = None, command_delay_steps: int = 0):
        self.model = model
        self.data = mujoco.MjData(model)
        self.calibration = calibration or arm.Calibration.stock()
        names = [*arm.ARM_JOINT_NAMES, arm.GRIPPER_JOINT_NAMES[0]]
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
        self._qpos = [int(model.jnt_qposadr[j]) for j in joints]
        self._qvel = [int(model.jnt_dofadr[j]) for j in joints]
        second = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, arm.GRIPPER_JOINT_NAMES[1])
        self._second_finger = int(model.jnt_qposadr[second])
        self._ctrl = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names]
        self._ball_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self._ball_qpos = int(model.jnt_qposadr[int(model.body_jntadr[self._ball_body])])
        self._delay = command_delay_steps
        self._queue: list[list[float]] = []
        self.after_step = None  # optional callback, called after each control step

    def reset(self, ball_xy: Sequence[float]) -> None:
        mujoco.mj_resetData(self.model, self.data)
        for address, angle in zip(self._qpos, runtime.SENSED_START):
            self.data.qpos[address] = angle
        self.data.qpos[self._second_finger] = -pickup.GRIPPER_OPEN_RAD
        self.data.qpos[self._ball_qpos : self._ball_qpos + 3] = [ball_xy[0], ball_xy[1], pickup.BALL_RADIUS_M]
        self.data.qpos[self._ball_qpos + 3 : self._ball_qpos + 7] = [1, 0, 0, 0]
        start = list(runtime.SENSED_START)
        for index, angle in zip(self._ctrl, start):
            self.data.ctrl[index] = angle
        self._queue = [start] * self._delay
        mujoco.mj_forward(self.model, self.data)

    def ball(self) -> np.ndarray:
        return self.data.xpos[self._ball_body].copy()

    def joint_angles(self) -> list[float]:
        return [float(self.data.qpos[a]) for a in self._qpos]

    # ArmIO -----------------------------------------------------------------

    def read(self) -> ArmReading:
        c = self.calibration
        angles = self.joint_angles()
        speeds = [float(self.data.qvel[a]) for a in self._qvel]
        positions = [arm.rad_to_ticks(a, z, s) for a, z, s in zip(angles[:5], c.zero_ticks, c.signs)]
        positions.append(arm.rad_to_ticks(angles[5], c.gripper_closed_tick, c.gripper_sign))
        signs = (*c.signs, c.gripper_sign)
        velocities = [int(round(sign * speed / VELOCITY_UNIT_RAD_S)) for sign, speed in zip(signs, speeds)]
        return ArmReading(tuple(positions), tuple(velocities))

    def write_goals(self, ticks: Sequence[int]) -> None:
        if len(ticks) != 6:
            raise ValueError("six goal positions are needed")
        c = self.calibration
        targets = c.to_rad(ticks[:5]) + [c.finger_to_rad(ticks[5])]
        self._queue.append(targets)
        for index, target in zip(self._ctrl, self._queue.pop(0)):
            self.data.ctrl[index] = target

    def advance(self) -> None:
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        if self.after_step is not None:
            self.after_step(self)


class SimulatedCamera:
    """Reports the simulated ball as a camera would: late, with a fixed and a per-frame error."""

    def __init__(self, sim: MujocoArm, rng: random.Random, bias: float, noise: float, latency_steps: Sequence[int]):
        self._sim, self._rng, self._noise = sim, rng, noise
        self._bias = np.array([rng.uniform(-bias, bias) for _ in range(3)])
        self._reports = [sim.ball()] * (rng.randint(latency_steps[0], latency_steps[1]) + 1)

    def position(self, joint_angles, finger_angle, gripper_open):
        self._reports = [self._sim.ball(), *self._reports[:-1]]
        noise = np.array([self._rng.uniform(-self._noise, self._noise) for _ in range(3)])
        return tuple(float(v) for v in self._reports[-1] + self._bias + noise)
