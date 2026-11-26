"""A bowl for the simulated desk: something the ball can be dropped into and the gripper can strike.

Made for measuring ``understudy.carry``. The bowl is round, with a flat middle and a wall
that curves up from it to the rim, steep at the top, as a cereal bowl's does. It is a free
body with the weight of a ceramic one, resting on the desk under its own weight: an arm
that strikes it pushes it, and that is measured too.

    model = build_model(extras=lambda spec: add_bowl(spec, Bowl()))
    sim = MujocoArm(model)
    sim.reset(ball_xy)
    place_bowl(sim, (0.0, 0.2))

Needs the ``mujoco`` and ``numpy`` packages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np

BOWL = "bowl"
_RINGS, _AROUND = 5, 24  # the wall is this many rings of this many flat pieces
_PARKED = (0.0, 0.6)  # where the bowl is until it is placed: on the desk, out of the arm's reach


@dataclass(frozen=True)
class Bowl:
    opening_m: float = 0.15  # across the inside of the rim
    rim_m: float = 0.065  # the rim's height above the desk
    flat_m: float = 0.06  # across the flat middle; made smaller if the opening is too narrow for it
    wall_m: float = 0.004  # how thick the wall is
    base_m: float = 0.008  # how thick the bowl is under its flat middle
    mass_kg: float = 0.4
    friction: float = 0.6  # against the desk and the arm
    # How a ball lands in it: MuJoCo's contact "damping ratio". At 1 the ball does not bounce;
    # at 0.3 it comes back up about a fifth of the way it fell and at 0.15 about a half
    # (``rebound`` measures it). Below about 0.12 the simulated contact is unstable.
    contact_damping: float = 1.0
    contact_time_s: float = 0.02  # MuJoCo's usual contact "time constant"

    @property
    def flat_radius_m(self) -> float:
        return min(self.flat_m, 0.6 * self.opening_m) / 2.0

    def inside_radius_at(self, height_m: float) -> float:
        """How far from its middle the inside of the bowl is, ``height_m`` above the desk."""
        up = (height_m - self.base_m) / (self.rim_m - self.base_m)
        if up <= 0:
            return self.flat_radius_m
        up = min(up, 1.0)
        return self.flat_radius_m + (self.opening_m / 2.0 - self.flat_radius_m) * math.sqrt(1.0 - (1.0 - up) ** 2)


def add_bowl(spec: mujoco.MjSpec, bowl: Bowl) -> None:
    """Add the bowl to a model's description, for ``understudy.mujoco_arm.build_model(extras=...)``."""
    if bowl.rim_m <= bowl.base_m + bowl.wall_m or bowl.opening_m <= 2 * bowl.wall_m:
        raise ValueError("the bowl's rim must be above its base and its opening wider than its walls")
    body = spec.worldbody.add_body(name=BOWL, pos=[*_PARKED, 0.0])
    body.add_freejoint()
    common = dict(friction=[bowl.friction, 0.005, 0.0001], rgba=[0.92, 0.92, 0.88, 1.0])
    names = [f"{BOWL}_floor"]
    flat, top = bowl.flat_radius_m, bowl.opening_m / 2.0
    # The flat middle, which is also all of the bowl that touches the desk.
    body.add_geom(name=f"{BOWL}_floor", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[flat + 0.002, bowl.base_m / 2.0, 0],
                  pos=[0, 0, bowl.base_m / 2.0], mass=bowl.mass_kg / 2.0, **common)
    # The wall: a quarter of an ellipse from the edge of the flat middle up to the rim, turned about the middle.
    across, up = top - flat, bowl.rim_m - bowl.base_m
    piece_mass = bowl.mass_kg / 2.0 / (_RINGS * _AROUND)
    for ring in range(_RINGS):
        a, b = (math.pi / 2.0 * (ring + part) / _RINGS for part in (0, 1))
        low = (flat + across * math.sin(a), bowl.rim_m - up * math.cos(a))
        high = (flat + across * math.sin(b), bowl.rim_m - up * math.cos(b))
        length = math.dist(low, high)
        slope = math.atan2(high[1] - low[1], high[0] - low[0])
        # The piece's inner face lies on the line from ``low`` to ``high``; the piece is on the outside of it.
        out, down = math.sin(slope) * bowl.wall_m / 2.0, -math.cos(slope) * bowl.wall_m / 2.0
        middle = ((low[0] + high[0]) / 2.0 + out, (low[1] + high[1]) / 2.0 + down)
        half_width = math.pi * (high[0] + bowl.wall_m) / _AROUND * 1.05
        for piece in range(_AROUND):
            turn = 2.0 * math.pi * piece / _AROUND
            outward = np.array([math.cos(turn), math.sin(turn), 0.0])
            along = math.cos(slope) * outward + np.array([0.0, 0.0, math.sin(slope)])
            round_ = np.array([-math.sin(turn), math.cos(turn), 0.0])
            axes = np.column_stack([along, round_, np.cross(along, round_)])
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, axes.flatten())
            names.append(f"{BOWL}_{ring}_{piece}")
            body.add_geom(name=names[-1], type=mujoco.mjtGeom.mjGEOM_BOX, size=[length / 2.0 + 0.001, half_width, bowl.wall_m / 2.0],
                          pos=[middle[0] * outward[0], middle[0] * outward[1], middle[1]], quat=quat, mass=piece_mass, **common)
    # How the ball lands in the bowl is set for those two alone: the bowl meets the desk and the arm as anything else does.
    grip = max(bowl.friction, float(spec.geom("ball").friction[0]))
    for name in names:
        spec.add_pair(geomname1="ball", geomname2=name, condim=3, friction=[grip, grip, 0.005, 0.0001, 0.0001],
                      solref=[bowl.contact_time_s, bowl.contact_damping])


def _address(sim) -> int:
    body = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, BOWL)
    return int(sim.model.jnt_qposadr[int(sim.model.body_jntadr[body])])


def place_bowl(sim, xy: Sequence[float]) -> None:
    """Stand the bowl on the desk with its middle at ``xy``, at rest."""
    body = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, BOWL)
    at = _address(sim)
    sim.data.qpos[at : at + 7] = [xy[0], xy[1], 0.0, 1, 0, 0, 0]
    dof = int(sim.model.body_dofadr[body])
    sim.data.qvel[dof : dof + 6] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


def bowl_xy(sim) -> tuple[float, float]:
    at = _address(sim)
    return (float(sim.data.qpos[at]), float(sim.data.qpos[at + 1]))


def touching(sim) -> tuple[bool, bool]:
    """Whether the arm is touching the bowl now, and whether the ball is."""
    model, data = sim.model, sim.data
    bowl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BOWL)
    ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    arm_touches = ball_touches = False
    for contact in data.contact[: data.ncon]:
        bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
        if bowl not in bodies or len(bodies) == 1:
            continue
        other = (bodies - {bowl}).pop()
        if other == ball:
            ball_touches = True
        elif other != 0:  # body 0 is the world: the desk
            arm_touches = True
    return arm_touches, ball_touches


def ball_is_in(sim, bowl: Bowl) -> bool:
    """Whether the ball's middle is below the rim and within the bowl's inside there."""
    x, y, z = (float(v) for v in sim.ball())
    return z < bowl.rim_m and math.dist((x, y), bowl_xy(sim)) < bowl.inside_radius_at(z)


def rebound(bowl: Bowl, drop_m: float = 0.05) -> float:
    """How high a ball dropped ``drop_m`` onto the bowl's flat middle bounces, as a share of the drop."""
    from . import pickup
    from .mujoco_arm import MujocoArm, build_model

    sim = MujocoArm(build_model(extras=lambda spec: add_bowl(spec, bowl)))
    sim.reset((0.0, 0.6))
    place_bowl(sim, (0.0, 0.6))
    rest = bowl.base_m + pickup.BALL_RADIUS_M
    at = sim._ball_qpos
    sim.data.qpos[at : at + 3] = [0.0, 0.6, rest + drop_m]
    mujoco.mj_forward(sim.model, sim.data)
    fell, highest = False, 0.0
    for _ in range(int(0.5 / sim.model.opt.timestep)):
        mujoco.mj_step(sim.model, sim.data)
        height = float(sim.ball()[2]) - rest
        if height < 0.001:
            if fell and highest > 0.001:
                break  # landed a second time: the first bounce is over
            fell = True
        elif fell:
            highest = max(highest, height)
    return highest / drop_m
