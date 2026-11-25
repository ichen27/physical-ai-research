"""Stand-ins for the real arm and its camera, for the tests.

``JumpArm`` is an arm whose servos jump straight to their goals. ``painted``
draws what a wrist camera would see of a ball on a plain desk, from geometry
alone, so the detector, the calibration and the pickup can be exercised
together on machines that cannot render.
"""

from __future__ import annotations

import cv2
import numpy as np

from understudy import pickup, runtime
from understudy.kinematics import Chain
from understudy.runner import ArmReading, goals_from_command

CHAIN = Chain()
DESK = (118, 121, 119)  # blue, green, red
BALL = (30, 118, 240)


class JumpArm:
    """Servos that jump straight to their goals. Remembers every slow move it was asked for."""

    def __init__(self, calibration):
        self.calibration = calibration
        self.positions = goals_from_command(runtime.start_command(), calibration)
        self.moves: list[tuple[list[int], float]] = []
        self.written: list[list[int]] = []
        self.reads = 0

    def read(self) -> ArmReading:
        self.reads += 1
        return ArmReading(tuple(self.positions), (0,) * 6)

    def write_goals(self, ticks) -> None:
        self.written.append(list(ticks))
        self.positions = list(ticks)

    def move_slowly(self, ticks, seconds: float) -> None:
        self.moves.append((list(ticks), seconds))
        self.positions = list(ticks)

    def advance(self) -> None:
        pass

    def angles(self) -> list[float]:
        return self.calibration.to_rad(self.positions[:5])


def painted(camera, angles, balls, extras=()) -> np.ndarray:
    """The picture ``camera`` takes with the arm at ``angles``: each ball in ``balls`` (x, y) as a disc on a plain desk.

    ``extras`` are (u, v, radius, colour) discs painted as they are, for things that are not the ball.
    """
    image = np.empty((camera.height, camera.width, 3), dtype=np.uint8)
    image[:] = DESK
    for ball in balls:
        sight = camera.sight((ball[0], ball[1], pickup.BALL_RADIUS_M), angles, CHAIN)
        if sight is None or abs(sight.u) > 4 * camera.width or abs(sight.v) > 4 * camera.width:
            continue
        cv2.circle(image, (int(round(sight.u * 16)), int(round(sight.v * 16))), int(round(sight.radius_px * 16)), BALL, -1,
                   cv2.LINE_AA, shift=4)
    for u, v, radius, colour in extras:
        cv2.circle(image, (int(u), int(v)), int(radius), colour, -1, cv2.LINE_AA)
    return image


class SimRobot:
    """The simulated MuJoCo arm dressed as ``understudy.dynamixel.DynamixelArm``, for testing the real-arm tool."""

    class _Port:
        def closePort(self) -> None:
            pass

    def __init__(self, sim):
        self.sim = sim
        self._port = self._Port()
        self.torque_on = False
        self.log: list[str] = []
        # The real servos cut their own power if the computer is silent for 500 ms, and the
        # raised arm drops. This remembers the longest silence while powered, on the real clock.
        self.longest_silence_s = 0.0
        self._heard = None

    def _hearing(self) -> None:
        """Call at the start of anything that talks to the servos."""
        import time

        if self.torque_on and self._heard is not None:
            self.longest_silence_s = max(self.longest_silence_s, time.monotonic() - self._heard)

    def _heard_now(self) -> None:
        import time

        self._heard = time.monotonic()

    def status(self):
        import types

        return [types.SimpleNamespace(dxl_id=i, torque_on=self.torque_on) for i in (11, 12, 13, 14, 15, 16)]

    def configure(self) -> None:
        self.torque_on = True
        self.log.append("configure")
        self._heard_now()

    def shutdown(self) -> list[str]:
        self._hearing()
        self.torque_on = False
        self.log.append("shutdown")
        return []

    def read(self):
        self._hearing()
        reading = self.sim.read()
        self._heard_now()
        return reading

    def write_goals(self, ticks) -> None:
        if not self.torque_on:
            raise AssertionError("goals written with torque off")
        self._hearing()
        self.sim.write_goals(ticks)
        self._heard_now()

    def move_slowly(self, ticks, seconds: float) -> None:
        if not self.torque_on:
            raise AssertionError("moved with torque off")
        self._hearing()
        self.log.append(f"move {seconds:.1f}")
        self.sim.move_slowly(ticks, seconds)  # the real driver reads throughout a slow move
        self._heard_now()

    def advance(self) -> None:
        self.sim.advance()  # simulated time passing, not talk on the bus
