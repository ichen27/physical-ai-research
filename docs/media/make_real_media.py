"""Draw the two pieces of real wrist-camera media in the README, each beside the colour filter's mask.

The source pictures are not in the repository (camera pictures are kept out by default), so this
needs them passed in:

    python docs/media/make_real_media.py --look run-569-002.jpg --live live-run-20261127-011102.local.pictures.npz

--look  one 1280 x 720 picture taken from the raised look pose (a JPEG the arm tool saved)
--live  the pictures a student was shown during a live run (the ``.pictures.npz`` file beside its record)

Needs numpy, opencv-python and pillow. Run from the repository's top folder.
"""

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from understudy import eye, vision  # noqa: E402

SCALE = 3  # the student's 128 x 72 picture drawn three times its size
WIDE, HIGH = eye.PICTURE_WIDTH * SCALE, eye.PICTURE_HEIGHT * SCALE
GAP, BAND = 12, 26
PAPER = (252, 252, 251)
INK = (30, 30, 30)
NEW_PICTURE_EVERY = 3  # the live stream gave a new picture on every third control step
CONTROL_HZ = 30


def font(size: int = 12):
    for path in ("/System/Library/Fonts/Supplemental/Arial Narrow.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def mask_of(small_bgr: np.ndarray) -> np.ndarray:
    """The filter's mask of a student-sized picture, as a black and white picture three times the size."""
    mask = vision.orange_mask(small_bgr, eye.FILTER).astype(np.uint8) * 255
    return cv2.cvtColor(cv2.resize(mask, (WIDE, HIGH), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2RGB)


def pair(left_rgb: np.ndarray, right_rgb: np.ndarray, left_label: str, right_label: str) -> Image.Image:
    sheet = Image.new("RGB", (WIDE * 2 + GAP, HIGH + BAND), PAPER)
    sheet.paste(Image.fromarray(left_rgb), (0, BAND))
    sheet.paste(Image.fromarray(right_rgb), (WIDE + GAP, BAND))
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 6), left_label, fill=INK, font=font())
    draw.text((WIDE + GAP + 4, 6), right_label, fill=INK, font=font())
    return sheet


def look(path: str, out: str) -> None:
    picture = cv2.imread(path)
    if picture is None:
        raise SystemExit(f"cannot read {path}")
    small = eye.shrink(picture)
    left = cv2.cvtColor(cv2.resize(picture, (WIDE, HIGH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    pair(left, mask_of(small), "Real wrist camera, from the raised look pose", "After the colour filter (the mask the student gets, 128 x 72)").save(out, optimize=True)
    print("wrote", out)


def live(path: str, out: str) -> None:
    pictures = np.load(path)["pictures"]
    frames = []
    for step in range(0, len(pictures), NEW_PICTURE_EVERY):
        small = pictures[step]
        left = cv2.cvtColor(cv2.resize(small, (WIDE, HIGH), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)
        frames.append(pair(left, mask_of(small), "Real wrist camera, as the student got it (128 x 72)",
                           f"After the colour filter (the mask)     t = {step / CONTROL_HZ:4.1f} s"))
    # One set of 256 colours for the whole animation, chosen from frames spread across the run: the
    # picture's colours change a lot once the ball is held up, and one frame's colours would wreck the rest.
    spread = [np.asarray(frames[i]) for i in np.linspace(0, len(frames) - 1, 12).astype(int)]
    palette = Image.fromarray(np.vstack(spread)).quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    frames[0].save(out, save_all=True, append_images=frames[1:], loop=0, optimize=True,
                   duration=int(1000 * NEW_PICTURE_EVERY / CONTROL_HZ))
    print("wrote", out, len(frames), "frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--look", help="a 1280 x 720 picture from the look pose")
    parser.add_argument("--live", help="the .pictures.npz file of a live run")
    args = parser.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    if args.look:
        look(args.look, os.path.join(here, "real-look-and-filter.png"))
    if args.live:
        live(args.live, os.path.join(here, "real-student-view.gif"))
    if not (args.look or args.live):
        parser.error("give --look, --live or both")
