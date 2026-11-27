"""A pickup policy that sees the wrist camera's picture instead of being told where the ball is.

The *teacher* is the trained pickup policy, which is given the ball's position.
The *student* is given the camera's picture and the same joint readings, and
learns to do what the teacher would do. It never receives the ball's position.

What the student puts out each control step:

- how far to move each of the five arm joints' targets, as a share of the most
  the real arm's speed limit allows in one step (-1 to 1);
- whether the gripper should be open;
- where it thinks the ball is, twice: from this picture alone (with whether the
  ball is in the picture at all), and from everything it has seen so far.

The teacher's own outputs are not copied as they are. With the speed limits the
real arm runs under, most of them are far beyond what the limit lets through
(a typical wrist output asks for a hundred times the allowed step), so only
their limited effect carries meaning. The student learns that.

**When the ball is not in the picture.** The real camera's lens is narrow and
from the start pose it looks almost straight down: about half the places a
ball can be are cut off or out of the picture when an attempt starts, and come
into it only as the gripper moves out over them. Two things cover that.

- Before the attempt the arm looks from a raised pose that shows everywhere a
  ball can be (``understudy.dagger.LOOK_POSE``), and the student says where the
  ball is from that one picture. That first guess is used to turn the arm to
  face the ball, and is then given back to the student on every step. It is the
  student's own reading of its own picture, not the ball's true position.
- Between the picture and its outputs the student keeps a small running state
  (a GRU), so that what it has seen stays with it when glare or the fingers
  hide the ball. So that this is actually learnt, training blanks the picture
  for stretches at random, and ``scripts/evaluate_student.py`` tests the same.

Needs ``torch`` and ``numpy``. Pictures are blue-green-red bytes at
``understudy.sim_look.PICTURE_WIDTH`` by ``PICTURE_HEIGHT``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from . import arm
from .eye import FILTER, INPUTS, PICTURE_HEIGHT, PICTURE_WIDTH, PROPRIO_SIZE  # noqa: F401

GUESS_SIZE = 2
# The joint readings the student gets, taken from the teacher's 33 inputs: joint angles relative
# to the start pose (6), joint speeds (6), the hold target (3), and how far each arm joint's target
# leads the joint (5). Left out: the ball's position (3), a constant (4) and the previous output (6).
_FROM_OBSERVATION = (*range(0, 12), *range(15, 18), *range(28, 33))
_PROPRIO_SCALE = np.array([1.0] * 6 + [2.0] * 6 + [5.0] * 3 + [20.0] * 5, dtype=np.float32)
BALL_SCALE = 10.0  # the network works in tenths of a metre
SPEED_LIMIT_RAD_S = 0.3  # the limit the student's arm outputs are a share of
LEAD_LIMIT_RAD = 0.05
_STEP_RAD = SPEED_LIMIT_RAD_S / arm.CONTROL_HZ


def proprio_from_observation(observation: Sequence[float]) -> np.ndarray:
    """The student's joint readings from the teacher's 33 inputs. The ball's position is not among them."""
    if len(observation) != 33:
        raise ValueError(f"the teacher's observation has 33 numbers, got {len(observation)}")
    return np.asarray([observation[i] for i in _FROM_OBSERVATION], dtype=np.float32)


def moves_from_teacher(action: Sequence[float], scale: float) -> tuple[np.ndarray, bool]:
    """What a teacher's raw output comes to under the speed limit: each arm target's move as a
    share of the allowed step, and whether the gripper is open. ``scale`` is the policy's action scale."""
    moves = np.clip(np.asarray(action[:5], dtype=np.float32) * scale / _STEP_RAD, -1.0, 1.0)
    return moves, bool(action[5] >= 0.0)


def action_from_moves(moves: Sequence[float], gripper_open: bool, scale: float) -> list[float]:
    """A raw output, in the teacher's terms, that produces these moves under the speed limit."""
    return [float(np.clip(m, -1.0, 1.0)) * _STEP_RAD / scale for m in moves] + [1.0 if gripper_open else -1.0]


# The filter's colour test, as ``understudy.eye`` and ``understudy.vision.orange_mask`` make it.
FILTER_HUE_DEG = (FILTER.hue_low, FILTER.hue_high)
FILTER_SATURATION_MIN = FILTER.saturation_min
FILTER_VALUE_MIN = FILTER.value_min
FILTER_VEIL_MAX = FILTER.veil_max


def ball_coloured(pictures: torch.Tensor) -> torch.Tensor:
    """Which pixels are the ball's colour: N x height x width, 1 or 0, from N x 3 x height x width pictures
    (blue, green, red, 0 to 255). The same test as ``understudy.vision.orange_mask`` with the constants above."""
    pixels = pictures.float()
    darkest = pixels.amin(dim=1)[:, ::4, ::4].flatten(1).sort(dim=1).values
    # The level of the darkest half percent, as numpy's percentile gives it.
    at = 0.005 * (darkest.shape[1] - 1)
    low, part = int(math.floor(at)), at - math.floor(at)
    veil = (darkest[:, low] * (1.0 - part) + darkest[:, min(low + 1, darkest.shape[1] - 1)] * part).clamp(max=FILTER_VEIL_MAX)
    pixels = (pixels - veil[:, None, None, None]).clamp(min=0.0)
    blue, green, red = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    high = pixels.amax(dim=1)
    spread = high - pixels.amin(dim=1)
    hue = torch.rad2deg(torch.atan2(math.sqrt(3.0) * (green - blue), 2.0 * red - green - blue))
    saturation = spread / high.clamp(min=1.0)
    value = high / (255.0 - veil).clamp(min=1.0)[:, None, None]
    return ((hue >= FILTER_HUE_DEG[0]) & (hue <= FILTER_HUE_DEG[1])
            & (saturation >= FILTER_SATURATION_MIN) & (value >= FILTER_VALUE_MIN)).float()


def spoiled(mask: torch.Tensor) -> torch.Tensor:
    """A mask as a real one comes: with specks that are not the ball, a hole in the ball, a bite out of it.

    Used in training only. In real pictures a speckled desk passes the colour test in
    patches of specks (3 to 4% of the picture, 8% at most, in 65 pictures), the ball's
    brightest spot can fail it and leave a hole, and its blown-out side a crescent.
    """
    count, rows, columns = mask.shape
    device = mask.device
    # Specks, in patches: thick in some parts of the picture and absent from others.
    patches = torch.nn.functional.interpolate(torch.rand(count, 1, 5, 8, device=device), size=(rows, columns),
                                              mode="bilinear", align_corners=False)[:, 0] ** 3
    amount = torch.rand(count, 1, 1, device=device) ** 2 * 0.6  # most pictures have few, some have many
    out = torch.maximum(mask, (torch.rand_like(mask) < patches * amount).float())
    v, u = torch.meshgrid(torch.arange(rows, device=device), torch.arange(columns, device=device), indexing="ij")
    area = mask.flatten(1).sum(1)
    weight = area.clamp(min=1.0)
    middle_u, middle_v = (mask * u).flatten(1).sum(1) / weight, (mask * v).flatten(1).sum(1) / weight

    def without_disc(image: torch.Tensor, share: float, off: float, radius: tuple[float, float]) -> torch.Tensor:
        centre_u = middle_u + torch.empty(count, device=device).uniform_(-off, off)
        centre_v = middle_v + torch.empty(count, device=device).uniform_(-off, off)
        size = torch.empty(count, device=device).uniform_(*radius)
        chosen = (torch.rand(count, device=device) < share) & (area > 0)
        disc = ((u[None] - centre_u[:, None, None]) ** 2 + (v[None] - centre_v[:, None, None]) ** 2) < (size**2)[:, None, None]
        return torch.where(disc & chosen[:, None, None], torch.zeros_like(image), image)

    out = without_disc(out, 0.3, 4.0, (1.0, 4.0))  # a hole
    return without_disc(out, 0.3, 12.0, (3.0, 12.0))  # a bite


class StudentNet(nn.Module):
    """Picture and joint readings in; arm moves, gripper and ball estimates out."""

    def __init__(self, width: int = 128, input: str = "colour"):
        super().__init__()
        if input not in INPUTS:
            raise ValueError(f"input must be one of {', '.join(INPUTS)}")
        self.width, self.input = width, input
        self.eye = nn.Sequential(
            nn.Conv2d(INPUTS[input], 24, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(),
            nn.Conv2d(48, 64, 3, padding=1),
        )
        rows, columns = math.ceil(PICTURE_HEIGHT / 4), math.ceil(PICTURE_WIDTH / 4)
        grid_v, grid_u = torch.meshgrid(torch.linspace(-1, 1, rows), torch.linspace(-1, 1, columns), indexing="ij")
        self.register_buffer("grid_u", grid_u.reshape(-1), persistent=False)
        self.register_buffer("grid_v", grid_v.reshape(-1), persistent=False)
        self.register_buffer("proprio_scale", torch.from_numpy(_PROPRIO_SCALE), persistent=False)
        self.fuse = nn.Sequential(nn.Linear(64 * 3 + PROPRIO_SIZE, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.seen = nn.Linear(width, 4)  # the ball from this picture alone: x, y, z, and whether it is in view
        # The first guess joins after "seen" is read off, so that what one picture says can never lean on it.
        self.memory = nn.GRU(width + GUESS_SIZE, width, batch_first=True)
        self.act = nn.Sequential(nn.Linear(2 * width + GUESS_SIZE, width), nn.ReLU(), nn.Linear(width, 9))  # 5 moves, gripper, ball x y z

    def shown(self, pictures: torch.Tensor) -> torch.Tensor:
        """What the network is given of N x 3 x height x width colour pictures: 0 to 255, in as many layers as its input has."""
        if self.input == "colour":
            return pictures.float()
        mask = ball_coloured(pictures)
        if self.training:
            mask = spoiled(mask)
        layers = [mask * 255.0]
        if self.input == "mask+grey":
            layers.append(pictures.float().mean(dim=1))
        return torch.stack(layers, dim=1)

    def look(self, pictures: torch.Tensor, proprio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """What one picture and one set of joint readings say, without memory.

        ``pictures`` is N x 3 x height x width, bytes or floats 0 to 255; ``proprio`` N x 20.
        Returns the features passed on to the memory and the ball seen (N x 4: x, y, z in
        metres, then how sure it is that the ball is in view, before the sigmoid).
        """
        return self.look_at(self.shown(pictures), proprio)

    def look_at(self, shown: torch.Tensor, proprio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``look`` on pictures already put through ``shown``. This is the part that is exported as the eye."""
        features = self.eye(shown / 255.0 - 0.5)
        flat = features.flatten(2)
        # Where each feature fires in the picture: a soft arg-max, which suits "where is the ball".
        weights = torch.softmax(flat, dim=2)
        where = torch.cat([(weights * self.grid_u).sum(2), (weights * self.grid_v).sum(2), flat.mean(2)], dim=1)
        fused = self.fuse(torch.cat([where, proprio * self.proprio_scale], dim=1))
        seen = self.seen(fused)
        return fused, torch.cat([seen[:, :3] / BALL_SCALE, seen[:, 3:]], dim=1)

    def forward(self, pictures: torch.Tensor, proprio: torch.Tensor, guess: torch.Tensor,
                state: torch.Tensor | None = None) -> dict:
        """A run of steps. ``pictures`` is B x T x 3 x height x width, ``proprio`` B x T x 20, and
        ``guess`` B x T x 2: where the student said the ball was before the attempt (x, y, metres).

        Returns ``moves`` (B x T x 5, -1 to 1), ``open`` (B x T, before the sigmoid), ``ball``
        (B x T x 3, metres, from memory), ``seen`` (B x T x 4, from each picture alone) and ``state``.
        """
        batch, steps = pictures.shape[:2]
        fused, seen = self.look(pictures.flatten(0, 1), proprio.flatten(0, 1))
        fused = torch.cat([fused.view(batch, steps, -1), guess * BALL_SCALE], dim=2)
        remembered, state = self.memory(fused, state)
        out = self.act(torch.cat([remembered, fused], dim=2))
        return {
            "moves": torch.tanh(out[..., :5]),
            "open": out[..., 5],
            "ball": out[..., 6:9] / BALL_SCALE,
            "seen": seen.view(batch, steps, 4),
            "state": state,
        }


def save(net: StudentNet, path: str | Path, **notes) -> None:
    torch.save({"width": net.width, "input": net.input, "weights": net.state_dict(), "notes": notes}, str(path))


def load(path: str | Path) -> tuple[StudentNet, dict]:
    saved = torch.load(str(path), map_location="cpu", weights_only=True)
    net = StudentNet(saved["width"], saved.get("input", "colour"))
    net.load_state_dict(saved["weights"])
    return net.eval(), saved.get("notes", {})


class Brain:
    """The student, one control step at a time, on the processor. Keeps its memory between steps."""

    def __init__(self, net: StudentNet):
        self.net = net.eval()
        self._state = None

    def reset(self) -> None:
        """Forget everything: call at the start of each attempt."""
        self._state = None

    @staticmethod
    def _tensors(picture: np.ndarray, proprio: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if picture.shape != (PICTURE_HEIGHT, PICTURE_WIDTH, 3):
            raise ValueError(f"a {PICTURE_WIDTH} x {PICTURE_HEIGHT} colour picture is needed, got {picture.shape}")
        image = torch.from_numpy(np.ascontiguousarray(picture)).permute(2, 0, 1)
        return image, torch.from_numpy(np.asarray(proprio, dtype=np.float32))

    @torch.no_grad()
    def glance(self, picture: np.ndarray, proprio: np.ndarray) -> tuple[tuple[float, float, float], float]:
        """Where the ball is from this one picture, and how sure the student is that it is in view (0 to 1).
        Does not touch the memory."""
        image, readings = self._tensors(picture, proprio)
        _, seen = self.net.look(image[None], readings[None])
        x, y, z, sure = (float(v) for v in seen[0])
        return (x, y, z), 1.0 / (1.0 + math.exp(-sure))

    @torch.no_grad()
    def step(self, picture: np.ndarray, proprio: np.ndarray, guess: Sequence[float]) -> dict:
        """One control step: ``moves`` (5), ``open``, ``ball`` (from memory), ``seen`` and ``in_view``.

        ``guess`` is where the student said the ball was before the attempt (x, y, metres)."""
        image, readings = self._tensors(picture, proprio)
        first = torch.tensor([[[float(guess[0]), float(guess[1])]]], dtype=torch.float32)
        out = self.net(image[None, None], readings[None, None], first, self._state)
        self._state = out["state"]
        seen = out["seen"][0, 0]
        return {
            "moves": out["moves"][0, 0].numpy(),
            "open": bool(out["open"][0, 0] >= 0.0),
            "ball": tuple(float(v) for v in out["ball"][0, 0]),
            "seen": tuple(float(v) for v in seen[:3]),
            "in_view": 1.0 / (1.0 + math.exp(-float(seen[3]))),
        }
