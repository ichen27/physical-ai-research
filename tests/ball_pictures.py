"""Paints a ball into pictures from the arm's camera, for the tests.

The pictures in ``fixtures/camera`` are real ones from the first arm's wrist
camera with no ball in front of the gripper. A painted ball stands in for the
real one: a shaded sphere of the ball's colour, put through the same grey veil
and blur as the rest of the picture.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from understudy import vision

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "camera"
# The ball's colour as the first arm's camera shows it once the veil is off (blue, green, red).
CORAL = (88.0, 104.0, 232.0)
ORANGE = (30.0, 118.0, 240.0)


def picture(name: str) -> np.ndarray:
    image = cv2.imread(str(FIXTURES / f"{name}.jpg"))
    if image is None:
        raise FileNotFoundError(name)
    return image


def paint_ball(image: np.ndarray, u: float, v: float, radius: float, colour=CORAL, shade: float = 1.0,
               cover: tuple[float, float, float, float] | None = None) -> np.ndarray:
    """A copy of ``image`` with a ball centred at (u, v).

    ``shade`` darkens the whole ball, as the arm's shadow does. ``cover`` is a box
    (left, top, right, bottom) of the ball that something dark hides.
    """
    height, width = image.shape[:2]
    grey = vision.veil(image)
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float32)
    dx, dy = (columns + 0.5 - u) / radius, (rows + 0.5 - v) / radius
    inside = dx * dx + dy * dy
    # Lit from above and a little to the left; the lower right of the ball is in its own shade.
    depth = np.sqrt(np.clip(1.0 - inside, 0.0, 1.0))
    light = np.clip(0.55 + 0.45 * (-0.35 * dx - 0.55 * dy + 0.75 * depth), 0.25, 1.0) * shade
    ball = np.stack([light * c for c in colour], axis=2)
    if cover is not None:
        left, top, right, bottom = cover
        hidden = (columns >= left) & (columns < right) & (rows >= top) & (rows < bottom)
        ball[hidden] = 12.0
    # Through the veil: scaled into the range left above the grey, as everything else in the picture is.
    veiled = ball * (255.0 - grey) / 255.0 + grey
    edge = np.clip((1.0 - np.sqrt(inside)) * radius + 0.5, 0.0, 1.0)[..., None]  # soft one-pixel rim
    out = image.astype(np.float32) * (1.0 - edge) + veiled * edge
    box = (slice(max(0, int(v - radius - 6)), int(v + radius + 6)), slice(max(0, int(u - radius - 6)), int(u + radius + 6)))
    out[box] = cv2.GaussianBlur(out[box], (5, 5), 1.2)  # the camera is a little out of focus
    return np.clip(out, 0, 255).astype(np.uint8)
