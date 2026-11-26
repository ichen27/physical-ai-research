"""Wrist-camera pictures from the simulator that look different on every attempt.

A policy that is to work from the real camera's pictures after seeing only
simulated ones must not be able to rely on anything the simulator gets wrong,
and it gets most things wrong: the light, the desk, the haze over this
camera's pictures, what else lies about. So every attempt is given a new
*look*: where the light comes from, how strong and what colour it is, whether
it casts the arm's shadow, the desk's colour and pattern, the ball's shade,
other things on the desk including red and orange ones, the arm's own shade,
and, after rendering, lens bend, glare, a grey veil, exposure, colour cast,
blur and sensor noise. The camera itself is moved a few millimetres and
degrees about its nominal mount, and its focal length a few percent, since
the real mount is only ever known that well.

What this does not do is make the pictures realistic. MuJoCo draws flat-shaded
shapes with hard shadows; there are no real textures, reflections, depth of
field or rolling shutter. The bet is on variety, not fidelity.

Pictures come out at the size the student network sees, blue-green-red like
the real camera's through OpenCV. Needs ``mujoco``, ``numpy`` and
``opencv-python-headless``, and OpenGL to render.
"""

from __future__ import annotations

import colorsys
import math
import random
from dataclasses import dataclass, replace

import cv2
import mujoco
import numpy as np

from .camera_model import WristCamera, rotation_from_vector, vector_from_rotation
from .kinematics import Chain
from .mujoco_arm import WRIST_CAMERA, MujocoArm

# The picture the student network sees. The real camera's 1280 x 720 shrunk ten times.
PICTURE_WIDTH, PICTURE_HEIGHT = 128, 72
_RENDER_SCALE = 2  # rendered at twice that and shrunk, as the real picture is shrunk from far larger
# Rendered with a wider view than the camera's, so that a lens that bends lines inward has something to show at its edges.
_WIDE = 1.3
_TEXTURE = 128
DISTRACTORS = 16
DESK_PATTERNS = ("flat", "speckle", "blotch", "grain", "tiles", "weave")
# Kept out of training, so that a measurement on held-out looks includes desks never seen.
HELD_OUT_PATTERNS = ("tiles", "weave")
# How far the camera is moved about its nominal mount, and its lens about the nominal lens: about
# what a calibration leaves unknown. Each of these also blurs where the student can say the ball is
# (1.5 degrees is 5 mm at 20 cm), and the pickup tolerates about 1 cm, so they are not made wider.
MOUNT_SHIFT_M = 0.003
MOUNT_TURN_RAD = math.radians(1.5)
FOCAL_SPREAD = 0.04
CENTRE_SPREAD = 0.02  # of the picture's width
DISTORTION_RANGE = (-0.05, 0.01)


@dataclass(frozen=True)
class Look:
    """One attempt's appearance. ``camera`` is the camera as it truly is for this attempt."""

    camera: WristCamera
    pattern: str
    veil: float
    blur: float
    noise: float
    glare: float
    latency_steps: int


def dress(spec: mujoco.MjSpec) -> None:
    """Add to the scene what varied looks need. Pass as ``extras`` to ``understudy.mujoco_arm.build_model``."""
    spec.add_texture(name="desk", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     width=_TEXTURE, height=_TEXTURE, rgb1=[0.5, 0.5, 0.5], rgb2=[0.5, 0.5, 0.5])
    desk = spec.add_material(name="desk", rgba=[1, 1, 1, 1], specular=0.05, shininess=0.05, texrepeat=[4, 4], texuniform=True)
    desk.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "desk"
    spec.geom("desk").material = "desk"
    spec.add_material(name="ball", rgba=[1.0, 0.4, 0.0, 1.0], specular=0.08, shininess=0.1)
    spec.geom("ball").material = "ball"
    world = spec.worldbody
    world.add_light(name="fill", pos=[0.5, 0.5, 1.0], dir=[0, 0, -1], castshadow=False)
    for index in range(DISTRACTORS):
        # Seen but not felt: the arm passes through them, so the teacher's behaviour is not disturbed.
        world.add_geom(name=f"thing{index}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.01, 0.01], pos=_parked(index),
                       contype=0, conaffinity=0, rgba=[0.5, 0.5, 0.5, 1.0])


def _parked(index: int) -> list[float]:
    """Where a thing not in use is kept: just under the desk, out of sight.

    Not far away. MuJoCo cuts off whatever is nearer the camera than a fraction of the
    whole scene's size, and things parked metres off made that 15 cm: the desk vanished
    from the wrist camera's picture as soon as the gripper came down to the ball.
    """
    return [0.1 + 0.03 * index, 0.0, -0.2]


def _colour(rng: random.Random, hue: tuple[float, float], saturation: tuple[float, float], value: tuple[float, float]) -> list[float]:
    """A colour by hue (degrees), saturation and brightness ranges, as red-green-blue 0 to 1."""
    return list(colorsys.hsv_to_rgb((rng.uniform(*hue) % 360.0) / 360.0, rng.uniform(*saturation), rng.uniform(*value)))


def _clear_of_reach(x: float, y: float, ball_xy: tuple[float, float], margin_m: float = 0.11) -> bool:
    """Whether a spot on the desk is clear of where the arm goes: the strip from its base to just past the ball.

    The other things on the desk are seen but not felt. One in the gripper's way would be
    passed through, and the camera would look out from inside it, which no real camera does.
    """
    ax, ay = -0.011, 0.0  # the swivel axis
    bx, by = ball_xy[0] - ax, ball_xy[1] - ay
    reach = math.hypot(bx, by)
    ux, uy = bx / reach, by / reach
    along = min(max((x - ax) * ux + (y - ay) * uy, 0.0), reach + 0.06)
    return math.hypot(x - (ax + along * ux), y - (ay + along * uy)) > margin_m


def desk_texture(rng: random.Random, pattern: str) -> np.ndarray:
    """A desk surface: a base colour with one of several patterns over it. Red-green-blue bytes."""
    if rng.random() < 0.6:  # greys and off-whites, light to dark
        base = np.array(_colour(rng, (0, 360), (0.0, 0.15), (0.35, 0.95)))
    else:  # woods, beiges, blues, greens: anything not strongly red or orange
        base = np.array(_colour(rng, (25, 290), (0.1, 0.45), (0.2, 0.9)))
    contrast = rng.uniform(0.02, 0.3)
    grid = np.random.default_rng(rng.getrandbits(32))
    n = _TEXTURE
    yy, xx = np.mgrid[0:n, 0:n] / n
    if pattern == "flat":
        field = np.zeros((n, n))
    elif pattern == "speckle":  # carpet, felt, a desk mat
        field = grid.normal(0, 1, (n, n))
    elif pattern == "blotch":  # stains, marble, worn paint
        field = cv2.GaussianBlur(grid.normal(0, 1, (n, n)), (0, 0), rng.uniform(4, 12)) * 8
    elif pattern == "grain":  # wood
        waves = rng.uniform(4, 14)
        field = np.sin(2 * math.pi * waves * yy + 3 * cv2.GaussianBlur(grid.normal(0, 1, (n, n)), (0, 0), 8))
    elif pattern == "tiles":
        count = rng.choice((2, 4, 8))
        field = ((np.floor(xx * count) + np.floor(yy * count)) % 2) * 2 - 1
    elif pattern == "weave":
        count = rng.choice((8, 16, 32))
        field = np.sin(2 * math.pi * count * xx) * np.sin(2 * math.pi * count * yy)
    else:
        raise ValueError(f"unknown desk pattern {pattern!r}")
    image = base[None, None, :] * (1.0 + contrast * np.clip(field, -2.5, 2.5)[..., None])
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


class Scene:
    """The simulated arm's wrist camera with a look that can be changed between attempts.

    The simulated arm's model must have been built with ``wrist_camera`` and with
    ``extras=dress``. ``nominal`` is the camera as it is believed to be mounted; each look
    moves the true camera a little about it.
    """

    def __init__(self, sim: MujocoArm, nominal: WristCamera, held_out: bool = False):
        self.sim, self.nominal, self.held_out = sim, nominal, held_out
        model = sim.model
        self._width, self._height = PICTURE_WIDTH * _RENDER_SCALE, PICTURE_HEIGHT * _RENDER_SCALE
        self._renderer = mujoco.Renderer(model, height=self._height, width=self._width)
        self._segments = mujoco.Renderer(model, height=self._height, width=self._width)
        self._segments.enable_segmentation_rendering()
        name = mujoco.mj_name2id
        self._camera = name(model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA)
        self._desk_texture = name(model, mujoco.mjtObj.mjOBJ_TEXTURE, "desk")
        self._desk_material = name(model, mujoco.mjtObj.mjOBJ_MATERIAL, "desk")
        self._ball_material = name(model, mujoco.mjtObj.mjOBJ_MATERIAL, "ball")
        self._ball_geom = name(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        self._ball_body = name(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self._fill = name(model, mujoco.mjtObj.mjOBJ_LIGHT, "fill")
        self._main = next(index for index in range(model.nlight) if index != self._fill)
        self._things = [name(model, mujoco.mjtObj.mjOBJ_GEOM, f"thing{index}") for index in range(DISTRACTORS)]
        if min(self._camera, self._desk_texture, self._fill, *self._things) < 0:
            raise ValueError("the model was not built with a wrist camera and extras=dress")
        skip = {self._ball_geom, name(model, mujoco.mjtObj.mjOBJ_GEOM, "desk"), *self._things}
        self._arm_geoms = [g for g in range(model.ngeom) if g not in skip and model.geom_rgba[g][3] > 0]
        self._arm_rgba = model.geom_rgba[self._arm_geoms].copy()
        self._chain = Chain()
        self.look: Look | None = None
        self._maps = None
        self._glare = None
        self._gains = np.ones(3, dtype=np.float32)
        self._veil_colour = np.full(3, 200.0, dtype=np.float32)
        self._noise = np.random.default_rng(0)
        self._late: list[tuple[np.ndarray, float]] = []

    def close(self) -> None:
        self._renderer.close()
        self._segments.close()

    # A new look ---------------------------------------------------------------

    def new_look(self, rng: random.Random, ball_xy: tuple[float, float] | None = None) -> Look:
        """Give the scene a new appearance. ``ball_xy`` keeps the other things on the desk clear of the ball."""
        model = self.sim.model
        self._noise = np.random.default_rng(rng.getrandbits(32))

        # The camera: a little off its nominal mount, with a lens a little off the nominal one.
        nominal = self.nominal
        xyz = tuple(v + rng.uniform(-MOUNT_SHIFT_M, MOUNT_SHIFT_M) for v in nominal.mount_xyz)
        axis = np.array([rng.gauss(0, 1) for _ in range(3)])
        wobble = rotation_from_vector(axis / np.linalg.norm(axis) * rng.uniform(0, MOUNT_TURN_RAD))
        rotation = vector_from_rotation(wobble @ rotation_from_vector(nominal.mount_rotation))
        camera = replace(
            nominal,
            focal_px=nominal.focal_px * (1 + rng.uniform(-FOCAL_SPREAD, FOCAL_SPREAD)),
            centre_px=(nominal.centre_px[0] + rng.uniform(-CENTRE_SPREAD, CENTRE_SPREAD) * nominal.width,
                       nominal.centre_px[1] + rng.uniform(-CENTRE_SPREAD, CENTRE_SPREAD) * nominal.width),
            distortion=rng.uniform(*DISTORTION_RANGE),
            mount_xyz=xyz,
            mount_rotation=tuple(float(v) for v in rotation),
        )
        axes = rotation_from_vector(camera.mount_rotation) @ np.diag([1.0, -1.0, -1.0])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, axes.flatten())
        model.cam_pos[self._camera] = camera.mount_xyz
        model.cam_quat[self._camera] = quat
        scale = self._width / camera.width
        focal = camera.focal_px * scale
        model.cam_fovy[self._camera] = math.degrees(2.0 * math.atan(self._height / (2.0 * focal / _WIDE)))
        self._maps = self._lens_maps(camera, scale)

        # Light: one main lamp somewhere above, a fill from elsewhere, and the room's own glow.
        aim = np.array([rng.uniform(0.05, 0.35), rng.uniform(-0.15, 0.15), 0.0])
        bearing, height, away = rng.uniform(0, 2 * math.pi), rng.uniform(math.radians(30), math.radians(88)), rng.uniform(0.8, 2.5)
        lamp = aim + away * np.array([math.cos(height) * math.cos(bearing), math.cos(height) * math.sin(bearing), math.sin(height)])
        warmth = rng.uniform(-0.15, 0.15)  # towards yellow or towards blue
        tint = np.array([1 + warmth, 1.0, 1 - warmth])
        strength = rng.uniform(0.45, 1.0)
        model.light_pos[self._main] = lamp
        model.light_dir[self._main] = (aim - lamp) / np.linalg.norm(aim - lamp)
        model.light_diffuse[self._main] = np.clip(strength * tint, 0, 1)
        model.light_specular[self._main] = rng.uniform(0.0, 0.3)
        model.light_castshadow[self._main] = rng.random() < 0.9
        model.light_pos[self._fill] = [rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.5, 2.0)]
        model.light_dir[self._fill] = [0, 0, -1]
        model.light_diffuse[self._fill] = rng.uniform(0.0, 0.4) * tint
        model.vis.headlight.ambient[:] = rng.uniform(0.1, 0.7) * tint
        model.vis.headlight.diffuse[:] = rng.uniform(0.0, 0.3)

        # Desk.
        patterns = HELD_OUT_PATTERNS if self.held_out else tuple(p for p in DESK_PATTERNS if p not in HELD_OUT_PATTERNS)
        pattern = rng.choice(patterns)
        texture = desk_texture(rng, pattern)
        start = int(model.tex_adr[self._desk_texture])
        model.tex_data[start : start + texture.size] = texture.ravel()
        mujoco.mjr_uploadTexture(model, self._renderer._mjr_context, self._desk_texture)
        repeat = rng.uniform(1.0, 12.0)
        model.mat_texrepeat[self._desk_material] = [repeat, repeat]
        model.mat_specular[self._desk_material] = rng.uniform(0.0, 0.3)

        # Ball: orange through coral, deep to pale. On the real camera the foam ball reads pale and
        # bright, almost glowing, even in the arm's shadow, so it is given some light of its own.
        model.mat_rgba[self._ball_material] = [*_colour(rng, (0, 36), (0.3, 1.0), (0.75, 1.0)), 1.0]
        model.mat_emission[self._ball_material] = rng.choice((0.0, rng.uniform(0.0, 0.45)))
        model.geom_friction[self._ball_geom][0] = rng.uniform(0.7, 1.2)
        model.body_mass[self._ball_body] = rng.uniform(0.004, 0.008)

        # The arm's own shade.
        shade = rng.uniform(0.5, 1.8)
        model.geom_rgba[self._arm_geoms] = np.clip(self._arm_rgba * [shade, shade, shade, 1.0], 0, 1)

        # Other things on the desk, red and orange ones among them, clear of where the arm goes.
        shown = rng.randint(2, DISTRACTORS)
        for index, geom in enumerate(self._things):
            if index >= shown:
                model.geom_pos[geom] = _parked(index)
                continue
            kind = rng.choice(("box", "cylinder", "ball", "wire", "wire"))
            for _ in range(40):
                x, y = rng.uniform(0.0, 0.42), rng.uniform(-0.28, 0.28)  # where the camera looks
                if ball_xy is None or _clear_of_reach(x, y, ball_xy):
                    break
            else:  # nowhere clear was found: leave it out
                model.geom_pos[geom] = _parked(index)
                continue
            turn = rng.uniform(0, math.pi)
            if kind == "box":
                size = [rng.uniform(0.005, 0.07), rng.uniform(0.005, 0.07), rng.uniform(0.003, 0.03)]
                model.geom_type[geom], height, quat = mujoco.mjtGeom.mjGEOM_BOX, size[2], [math.cos(turn / 2), 0, 0, math.sin(turn / 2)]
            elif kind == "cylinder":
                size = [rng.uniform(0.005, 0.03), rng.uniform(0.003, 0.03), 0]
                model.geom_type[geom], height, quat = mujoco.mjtGeom.mjGEOM_CYLINDER, size[1], [1, 0, 0, 0]
            elif kind == "ball":  # never the ball's own size
                radius = rng.choice((rng.uniform(0.006, 0.014), rng.uniform(0.032, 0.045)))
                size = [radius, 0, 0]
                model.geom_type[geom], height, quat = mujoco.mjtGeom.mjGEOM_SPHERE, radius, [1, 0, 0, 0]
            else:  # a wire lying on the desk
                size = [rng.uniform(0.001, 0.003), rng.uniform(0.03, 0.12), 0]
                model.geom_type[geom], height = mujoco.mjtGeom.mjGEOM_CAPSULE, size[0]
                c, s = math.cos(turn / 2), math.sin(turn / 2)
                quat = [c * math.sqrt(0.5), -s * math.sqrt(0.5), c * math.sqrt(0.5), s * math.sqrt(0.5)]  # lying flat, turned
            model.geom_size[geom] = size
            model.geom_pos[geom] = [x, y, height]
            model.geom_quat[geom] = quat
            shade = rng.random()
            if shade < 0.35:  # red through orange: what the ball must be told from
                rgb = _colour(rng, (340, 400), (0.5, 1.0), (0.5, 1.0))
            elif shade < 0.6:  # black, grey and white things: boxes, bags, cables, a laptop
                rgb = _colour(rng, (0, 360), (0.0, 0.1), (0.03, 1.0))
            else:
                rgb = _colour(rng, (0, 360), (0.0, 0.8), (0.1, 1.0))
            model.geom_rgba[geom] = [*rgb, 1.0]
        mujoco.mj_forward(model, self.sim.data)

        # What the camera and its lens do to the picture afterwards.
        veil = rng.choice((0.0, rng.uniform(0.0, 0.45)))
        grey = rng.uniform(150, 235)
        self._veil_colour = np.array([grey + rng.uniform(-12, 12) for _ in range(3)], dtype=np.float32)
        # Exposure runs from dim to well over: the real camera blows out the ball's lit side and
        # whatever is under the lamp, which here is the top of the picture more often than not.
        self._gains = np.array([rng.uniform(0.85, 1.15) for _ in range(3)], dtype=np.float32) * rng.uniform(0.7, 2.0)
        glare = rng.choice((0.0, rng.uniform(0.0, 0.6), rng.uniform(0.3, 1.0)))
        rows, columns = np.mgrid[0 : self._height, 0 : self._width].astype(np.float32)
        gu, gv = rng.uniform(0, self._width), rng.uniform(-0.2, rng.choice((0.2, 0.6))) * self._height
        su, sv = rng.uniform(0.15, 0.6) * self._width, rng.uniform(0.15, 0.6) * self._height
        self._glare = (glare * 255.0 * np.exp(-(((columns - gu) / su) ** 2 + ((rows - gv) / sv) ** 2)))[..., None]
        self.look = Look(camera, pattern, veil, rng.choice((0.0, rng.uniform(0.0, 1.6))), rng.uniform(0.0, 7.0), glare,
                         rng.randint(0, 3))
        self._late = []
        return self.look

    def _lens_maps(self, camera: WristCamera, scale: float) -> tuple[np.ndarray, np.ndarray]:
        """For each pixel of the picture, where to take it from in the wider, unbent rendering."""
        focal = camera.focal_px * scale
        cu, cv = camera.centre_px[0] * scale, camera.centre_px[1] * scale
        rows, columns = np.mgrid[0 : self._height, 0 : self._width].astype(np.float64)
        xd, yd = (columns + 0.5 - cu) / focal, (rows + 0.5 - cv) / focal
        xu, yu = xd.copy(), yd.copy()
        for _ in range(10):
            factor = np.maximum(1.0 + camera.distortion * (xu * xu + yu * yu), 0.3)
            xu, yu = xd / factor, yd / factor
        wide = focal / _WIDE
        return ((self._width / 2.0 + wide * xu - 0.5).astype(np.float32), (self._height / 2.0 + wide * yu - 0.5).astype(np.float32))

    # Pictures -------------------------------------------------------------------

    def picture_now(self) -> tuple[np.ndarray, float]:
        """The picture at this instant (blue-green-red, the student's size) and how much of the ball is in it.

        The second number is the share of the ball's disc that can actually be seen: 1 when
        it is whole and in the open, less when the fingers, the arm or the picture's edge hide it.
        """
        if self.look is None:
            raise RuntimeError("call new_look first")
        look, data = self.look, self.sim.data
        self._renderer.update_scene(data, camera=self._camera)
        image = cv2.remap(self._renderer.render(), self._maps[0], self._maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        image = image.astype(np.float32) * self._gains + self._glare
        image = image * (1.0 - look.veil) + self._veil_colour * look.veil
        if look.blur > 0.05:
            image = cv2.GaussianBlur(image, (0, 0), look.blur)
        if look.noise > 0.05:
            image += self._noise.normal(0.0, look.noise, image.shape).astype(np.float32)
        image = np.clip(image, 0, 255).astype(np.uint8)
        small = cv2.resize(image, (PICTURE_WIDTH, PICTURE_HEIGHT), interpolation=cv2.INTER_AREA)

        self._segments.update_scene(data, camera=self._camera)
        segments = self._segments.render()
        ball = ((segments[..., 0] == self._ball_geom) & (segments[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))).astype(np.uint8)
        seen = float(cv2.remap(ball, self._maps[0], self._maps[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).sum())
        sight = look.camera.sight(self.sim.ball(), self.sim.joint_angles()[:5], self._chain)
        whole = math.pi * (sight.radius_px * self._width / look.camera.width) ** 2 if sight is not None else 0.0
        visible = min(1.0, seen / whole) if whole > 1.0 else 0.0
        return np.ascontiguousarray(small[:, :, ::-1]), visible

    def picture(self) -> tuple[np.ndarray, float]:
        """The picture as it reaches the computer: the look's number of control steps late. Call once a step."""
        self._late = [self.picture_now(), *self._late[: self.look.latency_steps]]
        return self._late[-1]


def shrink(image_bgr: np.ndarray) -> np.ndarray:
    """A real picture from the camera at the size the student sees."""
    if image_bgr.shape[1] * PICTURE_HEIGHT != image_bgr.shape[0] * PICTURE_WIDTH:
        raise ValueError(f"a picture of the camera's 16 by 9 shape is needed, got {image_bgr.shape[1]} x {image_bgr.shape[0]}")
    if image_bgr.shape[1] == PICTURE_WIDTH:
        return image_bgr
    return cv2.resize(image_bgr, (PICTURE_WIDTH, PICTURE_HEIGHT), interpolation=cv2.INTER_AREA)


def student_camera() -> WristCamera:
    """The nominal wrist camera, at the real camera's full picture size.

    A rough fit to the first real arm's camera, not its calibration. It comes from the
    eight whole sightings of the ball in that arm's first calibration attempt
    (2026-11-26), which stopped early, fitted with the poses the arm was commanded to and
    not the ones it reported. A point not used in the fit was then placed consistently from
    all eleven pictures, to within 6 to 9 mm. It is only a stand-in for people without the arm:
    that arm's real calibration, made later the same day, has a 975-pixel lens on the gripper's
    centre line, 7 degrees and 4 cm away from this. Train with the real calibration when there
    is one (``scripts/train_student.py --camera``).

    It differs greatly from the first guess used for ``understudy.mujoco_arm.sim_wrist_camera``:
    the lens is about half as wide (62 degrees across, not 102), and from the start pose the
    camera looks almost straight down, seeing the desk from about 6 to 22 cm ahead.
    """
    return WristCamera(1280, 720, 1055.1, (640.0, 360.0), 0.0, (-0.066, 0.0369, 0.0405), (-1.4751, 1.7807, -0.8959))


