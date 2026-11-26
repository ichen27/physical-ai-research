"""Find the orange ball in a picture from the arm's camera.

The camera on the first arm gives a hazy, low-contrast picture: a grey veil
lies over everything, so the ball's orange is washed out, and the arm's own
shadow falls where the ball sits. A veil adds the same grey to every colour,
which leaves the hue alone and only weakens the saturation. So the veil is
estimated from the darkest things in the picture and taken off, and the ball is
then found by hue first, with only loose demands on saturation and brightness.

Colour alone cannot tell the ball from another orange thing. What is returned
therefore carries each orange patch's size, roundness and whether it is cut off
by the edge of the picture, and ``understudy.runner.CameraBall`` checks the
size against what a ball at that place on the desk must look like.

Needs ``numpy`` and ``opencv-python-headless``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np

WORKING_WIDTH = 640  # pictures are shrunk to this width before the colour test
# How far a pixel's colour must move, and how colourful it must end up, to count as part of a
# newly placed ball (0 to 255, on a picture shrunk to the working width).
NEW_BALL_TINT_CHANGE = 18.0
# A patch is a whole ball if at least this share of it lies inside the circle of its own area.
# On the first real arm's pictures a whole ball scores 0.96 to 0.99; a painted ball with a tenth
# of it hidden 0.94, and half hidden 0.79.
WHOLE_ROUNDNESS = 0.93
BLOWN_OUT = 250  # a pixel with any colour channel at or above this (of 255) tells nothing about colour


@dataclass(frozen=True)
class BallColour:
    """What counts as the ball's orange, after the grey veil is taken off.

    Hue is in degrees (red 0, orange about 25, yellow 60). The defaults come
    from the first arm's camera, where the ball read hue 24 to 32 degrees; red
    wires on the same desk read 350 and are left out.
    """

    hue_low: float = 10.0
    hue_high: float = 42.0
    saturation_min: float = 0.30  # 0 to 1
    value_min: float = 0.22  # 0 to 1
    veil_max: float = 90.0  # most grey that may be taken off, 0 to 255


def colour_from(data: dict | None) -> BallColour:
    """A saved colour test, or the default if there is none."""
    if not data:
        return BallColour()
    return BallColour(**{key: float(data[key]) for key in asdict(BallColour()) if key in data})


@dataclass(frozen=True)
class Blob:
    """One orange patch. Positions and sizes are in pixels of the picture it was found in."""

    u: float  # centre
    v: float
    radius_px: float  # of the circle with the patch's area
    roundness: float  # the share of the patch that lies inside that circle about its centre: 1 for a disc
    cut_off: bool  # touches the edge of the picture
    area_px: float

    def whole(self, min_roundness: float = WHOLE_ROUNDNESS) -> bool:
        """A ball seen whole: round, and not cut off by the picture's edge."""
        return self.roundness >= min_roundness and not self.cut_off

    def scaled(self, factor: float) -> "Blob":
        """The same patch in a picture ``factor`` times the size."""
        return replace(self, u=self.u * factor, v=self.v * factor, radius_px=self.radius_px * factor,
                       area_px=self.area_px * factor * factor)


def veil(image_bgr: np.ndarray, colour: BallColour = BallColour()) -> float:
    """How much grey lies over the picture, 0 to 255: the level of its darkest half percent."""
    darkest = np.min(image_bgr, axis=2)
    return float(min(np.percentile(darkest[::4, ::4], 0.5), colour.veil_max))


def colour_parts(image_bgr: np.ndarray, colour: BallColour = BallColour()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each pixel's hue (degrees, -180 to 180), saturation and brightness (both 0 to 1), with the veil taken off."""
    grey = veil(image_bgr, colour)
    pixels = image_bgr.astype(np.float32) - grey
    np.clip(pixels, 0.0, None, out=pixels)
    blue, green, red = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    high = np.max(pixels, axis=2)
    spread = high - np.min(pixels, axis=2)
    hue = np.degrees(np.arctan2(math.sqrt(3.0) * (green - blue), 2.0 * red - green - blue))
    saturation = spread / np.maximum(high, 1.0)
    return hue, saturation, high / max(255.0 - grey, 1.0)


def orange_mask(image_bgr: np.ndarray, colour: BallColour = BallColour()) -> np.ndarray:
    """True where the picture is the ball's colour."""
    hue, saturation, value = colour_parts(image_bgr, colour)
    return (
        (hue >= colour.hue_low) & (hue <= colour.hue_high)
        & (saturation >= colour.saturation_min) & (value >= colour.value_min)
    )


def _patches(mask: np.ndarray, floor_area: float, scale: float) -> list[Blob]:
    """Patches of a mask as blobs, largest first, in pixels ``scale`` times the mask's.

    Each patch is taken as its outline filled in, so a hole inside it does not count
    against it: the lit side of the real ball is blown out to white, which is not the
    ball's colour, and leaves one.

    Roundness is the share of the patch inside the circle of the patch's own area about
    its centre. An earlier version compared the area with the smallest circle around the
    patch. A few specks of the desk touching the ball were enough to stretch that circle,
    and on the real arm it scored the same resting ball anywhere from 0.59 to 0.86.
    """
    height, width = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blobs = []
    for contour in contours:
        left, top, wide, tall = cv2.boundingRect(contour)
        if wide * tall < floor_area:
            continue
        patch = np.zeros((tall, wide), dtype=np.uint8)
        cv2.drawContours(patch, [contour], -1, 1, thickness=cv2.FILLED, offset=(-left, -top))
        area = float(patch.sum())
        if area < floor_area:
            continue
        rows, columns = np.nonzero(patch)
        u, v = float(columns.mean()) + 0.5, float(rows.mean()) + 0.5  # within the patch's own box
        radius = math.sqrt(area / math.pi)
        inside = ((columns + 0.5 - u) ** 2 + (rows + 0.5 - v) ** 2 <= radius * radius).sum()
        cut_off = left <= 0 or top <= 0 or left + wide >= width or top + tall >= height
        blob = Blob(left + u, top + v, radius, float(inside / area), bool(cut_off), area)
        blobs.append(blob.scaled(scale))
    return sorted(blobs, key=lambda blob: -blob.area_px)


def _shrunk(image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("a colour picture is needed (height x width x 3, blue-green-red)")
    height, width = image_bgr.shape[:2]
    shrink = min(1.0, WORKING_WIDTH / width)
    if shrink == 1.0:
        return image_bgr, 1.0
    size = (int(round(width * shrink)), int(round(height * shrink)))
    return cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA), shrink


def find_blobs(image_bgr: np.ndarray, colour: BallColour = BallColour(), min_radius_px: float = 6.0) -> list[Blob]:
    """Patches of the ball's colour in the picture, largest first.

    ``min_radius_px`` is in pixels of a full-size (1280 wide) picture and scales with the
    picture; a ball anywhere the arm can reach is far larger than the default.
    """
    small, shrink = _shrunk(image_bgr)
    mask = orange_mask(small, colour).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    floor = math.pi * (min_radius_px * mask.shape[1] / 1280.0) ** 2
    return _patches(mask, floor, 1.0 / shrink)


def find_ball(image_bgr: np.ndarray, colour: BallColour = BallColour(), min_radius_px: float = 12.0) -> Blob | None:
    """The largest orange patch that looks like a whole ball, or None.

    For use before the camera is calibrated. With a calibrated camera, take
    ``find_blobs`` and let ``understudy.runner.CameraBall`` choose by size and place.
    """
    for blob in find_blobs(image_bgr, colour, min_radius_px):
        if blob.whole():
            return blob
    return None


def find_new_ball(before_bgr: np.ndarray, after_bgr: np.ndarray, min_radius_px: float = 12.0) -> Blob | None:
    """A ball that is in the second picture and was not in the first, or None.

    Needs no knowledge of the ball's colour: it looks for a round patch that
    became more colourful. The ball's own shadow changes brightness but hardly
    colour, so it is left out. The two pictures must be taken from the same place.
    """
    if before_bgr.shape != after_bgr.shape:
        raise ValueError("the two pictures must be the same size")
    before, shrink = _shrunk(before_bgr)
    after, _ = _shrunk(after_bgr)
    before = cv2.GaussianBlur(before, (5, 5), 0).astype(np.float32)
    after = cv2.GaussianBlur(after, (5, 5), 0).astype(np.float32)

    def tint(pixels: np.ndarray) -> np.ndarray:  # colour with brightness taken out: two differences of channels
        return np.stack([pixels[..., 2] - pixels[..., 1], pixels[..., 1] - pixels[..., 0]], axis=2)

    change = np.linalg.norm(tint(after) - tint(before), axis=2)
    colourful = np.linalg.norm(tint(after), axis=2)
    mask = ((change > NEW_BALL_TINT_CHANGE) & (colourful > NEW_BALL_TINT_CHANGE)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    floor = math.pi * (min_radius_px * mask.shape[1] / 1280.0) ** 2
    for blob in _patches(mask, floor, 1.0 / shrink):
        if blob.whole():
            return blob
    return None


def learn_colour(image_bgr: np.ndarray, ball: Blob, base: BallColour = BallColour()) -> BallColour:
    """The colour test that fits a ball already found in a picture (by ``find_new_ball``).

    Takes the hue, saturation and brightness of the pixels well inside the ball
    and allows a margin around them, so the ball's shaded side is still its colour.

    Pixels the camera has blown out are left aside. The lit side of the real ball goes
    pale yellow to white, and a colour test wide enough to take that in also takes in a
    beige desk; on the real arm that is what stopped the ball being recognised after it
    was moved. The blown-out part is still found, as the inside of the ball's outline.
    """
    hue, saturation, value = colour_parts(image_bgr, base)
    rows, columns = np.mgrid[0 : image_bgr.shape[0], 0 : image_bgr.shape[1]]
    inside = (columns + 0.5 - ball.u) ** 2 + (rows + 0.5 - ball.v) ** 2 <= (0.85 * ball.radius_px) ** 2
    if int(inside.sum()) < 50:
        raise ValueError("the ball is too small in the picture to learn its colour from")
    exposed = inside & (np.max(image_bgr, axis=2) < BLOWN_OUT)
    if int(exposed.sum()) < 0.2 * int(inside.sum()):
        exposed = inside  # nearly all of it is blown out: there is nothing better to learn from
    hues = hue[exposed]
    middle = float(np.median(hues))
    low, high = (float(v) for v in np.percentile(hues, [5, 95]))
    if high - low > 70.0:
        raise ValueError("the patch has no one colour; it is probably not the ball")
    return replace(
        base,
        hue_low=min(low, middle - 6.0) - 5.0,
        hue_high=max(high, middle + 6.0) + 5.0,
        saturation_min=max(0.12, 0.7 * float(np.percentile(saturation[exposed], 10))),
        # The floor on brightness is kept low: a ball in the arm's shadow is far darker than a lit one.
        value_min=max(0.05, 0.25 * float(np.percentile(value[exposed], 3))),
    )


def decode(data: bytes) -> np.ndarray:
    """A picture file's bytes as a blue-green-red array."""
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("not a picture this can read")
    return image


def draw(image_bgr: np.ndarray, blobs: list[Blob], chosen: Blob | None = None, note: str = "") -> np.ndarray:
    """A copy of the picture with the patches outlined: the chosen one in green, the others in blue."""
    out = image_bgr.copy()
    for blob in blobs:
        picked = chosen is not None and blob == chosen
        centre = (int(round(blob.u)), int(round(blob.v)))
        cv2.circle(out, centre, int(round(blob.radius_px)), (0, 255, 0) if picked else (255, 128, 0), 2)
        cv2.drawMarker(out, centre, (0, 255, 0) if picked else (255, 128, 0), cv2.MARKER_CROSS, 12, 2)
    if note:
        cv2.putText(out, note, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(out, note, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out
