"""Pictures from the real camera on the arm, and from no other camera.

macOS lets a program use a camera only if the person at the computer started
it (from Terminal, say); a program started over a remote login is refused. So
there are two ways to get pictures:

``WatcherCamera`` asks a small script that the person left running in Terminal
(``scripts/camera_watch.sh``) for one picture at a time, by leaving a note in a
folder. It works from a remote login, and takes a second or two per picture:
enough to look before a pickup and to calibrate, not to follow the ball.

``LiveCamera`` reads a stream of pictures itself, through ``ffmpeg``, and keeps
only the newest. It follows the ball during a pickup, and needs the whole
program to have been started from Terminal on that computer.

Both name the camera they use. Neither falls back to another one: a laptop
also has its own camera and may see a phone's, and those are never opened.
Needs ``numpy`` and ``opencv-python-headless``; ``LiveCamera`` needs ``ffmpeg``.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import vision

ARM_CAMERA = "Innomaker-U20CAM-720P"
WATCH_FOLDER = Path.home() / "understudy-camera"


class CameraError(RuntimeError):
    """The arm's camera could not give a picture."""


class WatcherCamera:
    """One picture at a time from the watcher script running in Terminal.

    ``keep_alive`` is called several times a second while waiting. Give it the
    arm's ``read`` when the arm is powered, so its servos keep hearing from the
    computer and do not switch themselves off.
    """

    def __init__(self, folder: str | Path = WATCH_FOLDER, keep_alive: Callable[[], object] | None = None,
                 timeout_s: float = 12.0, prefix: str = "look"):
        self.folder = Path(folder)
        self._keep_alive = keep_alive
        self._timeout = timeout_s
        self._prefix = re.sub(r"[^A-Za-z0-9_-]", "", prefix)[:20] or "look"
        self._count = 0
        self.last_path: Path | None = None

    def picture(self) -> np.ndarray:
        """A picture taken now, blue-green-red. The arm should be still while this waits."""
        self._count += 1
        name = f"{self._prefix}-{os.getpid()}-{self._count:03d}"
        shots = self.folder / "shots"
        done, shot = shots / f"{name}.done", shots / f"{name}.jpg"
        if not self.folder.is_dir():
            raise CameraError(f"{self.folder} does not exist; set up the camera watcher first (scripts/camera_watch.sh)")
        note = self.folder / "take.writing"
        note.write_text(name)
        os.replace(note, self.folder / "take")
        deadline = time.monotonic() + self._timeout
        while not done.exists():
            if time.monotonic() > deadline:
                (self.folder / "take").unlink(missing_ok=True)
                raise CameraError(
                    "the camera watcher is not answering. In Terminal on the Mac the arm is plugged into, run:\n"
                    f"  {self.folder}/camera-watch.sh"
                )
            if self._keep_alive is not None:
                self._keep_alive()
            time.sleep(0.05)
        if not shot.exists() or shot.stat().st_size == 0:
            raise CameraError("the watcher could not take a picture. Is the arm's camera plugged in?")
        self.last_path = shot
        return vision.decode(shot.read_bytes())


def find_camera(ffmpeg: str = "ffmpeg", wanted: str = ARM_CAMERA) -> str:
    """The exact name macOS gives the arm's camera. Raises unless exactly one camera matches."""
    try:
        listing = subprocess.run(
            [ffmpeg, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=20,
        ).stderr
    except FileNotFoundError:
        raise CameraError(f"{ffmpeg} was not found; install ffmpeg or give its path") from None
    video = listing.split("AVFoundation audio devices")[0]
    names = [match.group(1) for match in re.finditer(r"\[\d+\] (.*)", video)]
    matches = [name for name in names if wanted in name]
    if len(matches) != 1:
        seen = ", ".join(repr(n.strip()) for n in names) or "none"
        raise CameraError(f"expected exactly one camera named {wanted!r}; this computer shows: {seen}")
    return matches[0]


class LiveCamera:
    """The newest picture from a stream, read in the background.

    ``command`` produces raw blue-green-red pictures of ``width`` by ``height``
    on its standard output; by default it is ``ffmpeg`` reading the arm's
    camera by name. ``full_width`` is the camera's own picture width, which
    positions are reported in whatever size the stream is.
    """

    # Options that ask ffmpeg not to hold pictures back. Not yet the default: their effect on this camera is unmeasured.
    LOW_DELAY = ("-fflags", "nobuffer", "-flags", "low_delay", "-probesize", "32", "-analyzeduration", "0")

    def __init__(self, width: int = 640, height: int = 360, full_width: int = 1280, ffmpeg: str = "ffmpeg",
                 command: list[str] | None = None, input_options: Sequence[str] = ()):
        if command is None:
            name = find_camera(ffmpeg)
            command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", *input_options, "-f", "avfoundation", "-pixel_format", "uyvy422",
                "-framerate", "30", "-video_size", f"{full_width}x{full_width * height // width}", "-i", f"{name}:none",
                "-vf", f"scale={width}:{height}", "-pix_fmt", "bgr24", "-f", "rawvideo", "-",
            ]
        self.width, self.height = width, height
        self._scale = full_width / width
        self._process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._number = 0
        self._arrived = 0.0
        self._taken = 0
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        size = self.width * self.height * 3
        stream = self._process.stdout
        while True:
            data = bytearray()
            while len(data) < size:
                chunk = stream.read(size - len(data))
                if not chunk:
                    return
                data += chunk
            frame = np.frombuffer(bytes(data), dtype=np.uint8).reshape(self.height, self.width, 3)
            with self._lock:
                self._frame, self._number, self._arrived = frame, self._number + 1, time.monotonic()

    def wait_for_first(self, timeout_s: float = 10.0) -> None:
        """Wait until pictures are arriving. Raises with the camera program's own complaint if they never do."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._number > 0:
                    return
            if self._process.poll() is not None:
                break
            time.sleep(0.05)
        complaint = ""
        if self._process.poll() is not None and self._process.stderr is not None:
            complaint = self._process.stderr.read().decode(errors="replace").strip()
        self.close()
        raise CameraError(
            "no pictures are arriving from the arm's camera. macOS only allows the camera to programs started "
            "from Terminal on this computer, not over a remote login."
            + (f"\nThe camera program said: {complaint}" if complaint else "")
        )

    def picture(self) -> np.ndarray | None:
        """The newest picture, or None if there has been no new one since the last call."""
        with self._lock:
            if self._number == self._taken:
                return None
            self._taken = self._number
            return self._frame

    def arrival(self) -> tuple[int, float]:
        """The newest picture's number, counting from one, and when it arrived (``time.monotonic``). (0, 0.0) before the first."""
        with self._lock:
            return self._number, self._arrived

    def newest(self) -> np.ndarray:
        """The newest picture whether or not it was already taken. Raises if none has arrived."""
        with self._lock:
            if self._frame is None:
                raise CameraError("no picture has arrived yet")
            self._taken = self._number
            return self._frame

    def look(self, colour: vision.BallColour) -> list[vision.Blob] | None:
        """Patches of the ball's colour in the newest picture, in full-size pixels; None if there is no new picture."""
        frame = self.picture()
        if frame is None:
            return None
        return [blob.scaled(self._scale) for blob in vision.find_blobs(frame, colour)]

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
        for stream in (self._process.stdout, self._process.stderr):
            if stream is not None:
                stream.close()
