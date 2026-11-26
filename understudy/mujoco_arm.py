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
from .sim_urdf import FINGER_LINKS, finger_collision_files

VENDOR_MODEL = (
    Path(__file__).resolve().parent.parent / "assets" / "open_manipulator_description" / "mujoco" / "omx" / "omx.xml"
)
SUBSTEPS = 16
TIMESTEP = 1.0 / (arm.CONTROL_HZ * SUBSTEPS)
# Time since arrival given to the pose an attempt starts in, so it counts as long since reached.
_SETTLED_S = 10.0
WRIST_CAMERA = "wrist"


def sim_wrist_camera():
    """The simulated arm's wrist camera: a stand-in mounted where the first real arm's seems to be.

    Just above the gripper's palm, looking along the gripper and tipped a little towards
    its underside, so that the open fingers are out of the picture and the closed ones fill
    the bottom of it, as in the real camera's pictures. Half the real picture's size, to
    render quickly, with the same field of view as a first estimate of the real lens.
    """
    from dataclasses import replace

    from .camera_model import WristCamera

    camera = WristCamera.looking_along_gripper(640, 360, 260.0, back_m=0.06, up_m=0.035, tilt_down_rad=0.45)
    return replace(camera, mount_xyz=(-0.06, 0.004, 0.035))


def build_model(
    ball_mass: float = pickup.BALL_MASS_KG,
    ball_friction: float = pickup.BALL_FRICTION,
    stiffness_scale: float = 1.0,
    claw_fingers: bool = True,
    wrist_camera=None,
    extras=None,
) -> mujoco.MjModel:
    """The vendor arm on a desk with a ball. ``stiffness_scale`` multiplies the arm servo gains.

    With ``claw_fingers`` the fingers collide as the curved claws they are, using the same
    convex pieces as the Isaac Lab arm. Without it they collide as the vendor model has them,
    as the convex hull of each finger, which fills the hollow between the claws.

    ``wrist_camera`` (an ``understudy.camera_model.WristCamera``) adds a camera named
    ``wrist`` on the gripper, for ``RenderedCamera``. MuJoCo's camera has no lens
    distortion and looks through the middle of its picture, so only the size, focal
    length and mount of the one given are used.

    ``extras`` is called with the model's description just before it is compiled, to add
    things to the scene (``understudy.sim_look.dress`` adds what varied pictures need).
    """
    if stiffness_scale <= 0:
        raise ValueError("stiffness_scale must be positive")
    spec = mujoco.MjSpec.from_file(str(VENDOR_MODEL))
    if claw_fingers:
        for finger, link in FINGER_LINKS.items():
            body = spec.body(link)
            for geom in body.geoms:
                geom.contype, geom.conaffinity = 0, 0
            for piece in finger_collision_files(finger):
                spec.add_mesh(name=piece.stem, file=str(piece))
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=piece.stem, group=3, rgba=[0.2, 0.2, 0.2, 1])
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
    if wrist_camera is not None:
        from .camera_model import rotation_from_vector

        # MuJoCo cameras look along their own -z with +y up; the camera model looks along +z with +y down.
        axes = rotation_from_vector(wrist_camera.mount_rotation) @ np.diag([1.0, -1.0, -1.0])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, axes.flatten())
        spec.body(arm.END_EFFECTOR_LINK).add_camera(
            name=WRIST_CAMERA, pos=list(wrist_camera.mount_xyz), quat=quat,
            fovy=math.degrees(2.0 * math.atan(wrist_camera.height / (2.0 * wrist_camera.focal_px))),
        )
        for geom in spec.body(arm.END_EFFECTOR_LINK).geoms:
            geom.rgba[3] = 0.0  # the vendor model marks the gripper tip with a red box that no real arm has
        spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, wrist_camera.width)
        spec.visual.global_.offheight = max(spec.visual.global_.offheight, wrist_camera.height)
    if extras is not None:
        extras(spec)
    model = spec.compile()
    for joint in arm.ARM_JOINTS:
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint.name)
        model.actuator_gainprm[index, 0] *= stiffness_scale
        model.actuator_biasprm[index, 1] *= stiffness_scale
        model.actuator_biasprm[index, 2] *= math.sqrt(stiffness_scale)  # keeps the damping ratio
    return model


class MujocoArm:
    """Implements ``understudy.runner.ArmIO`` on the simulated arm."""

    def __init__(
        self,
        model: mujoco.MjModel,
        calibration: arm.Calibration | None = None,
        command_delay_s: float = 0.0,
        profile_time_s: float = 0.0,
        extra_delay_steps: int = 0,
    ):
        """``command_delay_s`` is the time before a new command starts to act and
        ``profile_time_s`` the time the servo then takes to reach it, as in training.
        ``extra_delay_steps`` holds every command back by whole control steps on top."""
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
        self._delay_s = command_delay_s + extra_delay_steps / arm.CONTROL_HZ
        self._profile_s = profile_time_s
        # Commands on their way to the servo, newest first, each with the time since it was sent.
        self._sent: list[tuple[list[float], float]] = []
        self._start = self._goal = list(runtime.SENSED_START)
        self._since_arrival = _SETTLED_S
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
        self._sent = []
        self._start = self._goal = start
        self._since_arrival = _SETTLED_S
        mujoco.mj_forward(self.model, self.data)

    def ball(self) -> np.ndarray:
        return self.data.xpos[self._ball_body].copy()

    def place_ball(self, ball_xy: Sequence[float]) -> None:
        """Put the ball down at rest somewhere else, leaving the arm as it is."""
        self.data.qpos[self._ball_qpos : self._ball_qpos + 3] = [ball_xy[0], ball_xy[1], pickup.BALL_RADIUS_M]
        self.data.qpos[self._ball_qpos + 3 : self._ball_qpos + 7] = [1, 0, 0, 0]
        ball_dof = int(self.model.body_dofadr[self._ball_body])
        self.data.qvel[ball_dof : ball_dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

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
        self._sent.insert(0, (targets, 0.0))

    def move_slowly(self, ticks: Sequence[int], seconds: float) -> None:
        """Move to a pose over ``seconds`` and let it settle, as the real arm's driver does."""
        start = self.read().positions
        steps = max(1, int(round(seconds * arm.CONTROL_HZ)))
        for step in range(1, steps + 1):
            # Ease in and out, like the servos' own profile.
            part = 0.5 - 0.5 * math.cos(math.pi * step / steps)
            self.write_goals([int(round(a + (b - a) * part)) for a, b in zip(start, ticks)])
            self.advance()
        for _ in range(int(0.5 * arm.CONTROL_HZ)):
            self.advance()

    def advance(self) -> None:
        for _ in range(SUBSTEPS):
            # A command reaches the servo after the delay. Until then the servo keeps
            # following the previous one; when it arrives, the servo heads for it from
            # wherever its goal has got to, taking the profile time.
            self._sent = [(targets, age + TIMESTEP) for targets, age in self._sent]
            self._since_arrival += TIMESTEP
            arrived = [item for item in self._sent if item[1] >= self._delay_s]
            if arrived:
                targets, age = arrived[0]  # the newest one to have arrived
                self._start = [float(self.data.ctrl[index]) for index in self._ctrl]
                self._goal = targets
                self._since_arrival = age - self._delay_s
                self._sent = [item for item in self._sent if item[1] < self._delay_s]
            if self._profile_s > 0.0:
                progress = min(1.0, self._since_arrival / self._profile_s)
            else:
                progress = 1.0
            for index, start, goal in zip(self._ctrl, self._start, self._goal):
                self.data.ctrl[index] = start + (goal - start) * progress
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


class RenderedCamera:
    """The simulated wrist camera: renders its picture and finds the ball in it with the real detector.

    The model must have been built with ``wrist_camera``. Rendering needs OpenGL;
    where there is none, creating this raises and ``ProjectedCamera`` can stand in.
    ``latency_steps`` is how many control steps old each picture is when it is used.
    """

    def __init__(self, sim: MujocoArm, camera, colour=None, latency_steps: int = 0):
        from . import vision

        self._vision = vision
        self._sim = sim
        self._renderer = mujoco.Renderer(sim.model, height=camera.height, width=camera.width)
        self.colour = colour or vision.BallColour()
        self._latency = latency_steps
        self._waiting: list = []

    def picture(self) -> np.ndarray:
        """The camera's picture now, blue-green-red."""
        self._renderer.update_scene(self._sim.data, camera=WRIST_CAMERA)
        return np.ascontiguousarray(self._renderer.render()[:, :, ::-1])

    def look(self):
        """Patches of the ball's colour in the picture from ``latency_steps`` ago."""
        self._waiting = [self._vision.find_blobs(self.picture(), self.colour), *self._waiting[: self._latency]]
        return self._waiting[-1]

    def close(self) -> None:
        self._renderer.close()


class ProjectedCamera:
    """The simulated wrist camera without rendering: where geometry says the ball appears.

    For machines with no OpenGL. It knows the edges of the picture but not the
    fingers: a ball between them is reported as seen whole, which a rendered or
    real picture would not show. ``pixel_noise`` jitters the centre each frame.
    """

    def __init__(self, sim: MujocoArm, camera, rng: random.Random | None = None, pixel_noise: float = 0.0,
                 latency_steps: int = 0):
        from . import vision
        from .kinematics import Chain

        self._vision, self._sim, self._camera = vision, sim, camera
        self._chain = Chain()
        self._rng, self._noise = rng or random.Random(0), pixel_noise
        self._latency = latency_steps
        self._waiting: list = []

    def look(self):
        sight = self._camera.sight(self._sim.ball(), self._sim.joint_angles()[:5], self._chain)
        blobs = []
        if sight is not None:
            c = self._camera
            r = sight.radius_px
            if -r < sight.u < c.width + r and -r < sight.v < c.height + r:
                u = sight.u + self._rng.gauss(0.0, self._noise)
                v = sight.v + self._rng.gauss(0.0, self._noise)
                blobs = [self._vision.Blob(u, v, r, 1.0 if sight.inside else 0.5, not sight.inside, math.pi * r * r)]
        self._waiting = [blobs, *self._waiting[: self._latency]]
        return self._waiting[-1]
