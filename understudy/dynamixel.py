"""The real OMX follower arm's servos, behind ``understudy.runner.ArmIO``.

Needs the ``dynamixel_sdk`` package and the arm's USB controller. Only the reads
behind ``status`` and ``check_ready`` have been run on a real arm; nothing here
has driven its motors yet. See the README before using it.

Only volatile (RAM) settings are changed, and they are put back on shutdown.
Settings that live in the servos' permanent memory (operating mode, drive mode,
limits, homing offsets) are checked and never written.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from . import arm
from .runner import SERVO_IDS, ArmReading

BAUD_RATE = 1_000_000
MODEL_NUMBERS = {1060: "XL430-W250", 1200: "XL330-M288"}

# Control table addresses and sizes shared by the XL430 and XL330.
MODEL_NUMBER = (0, 2)
DRIVE_MODE = (10, 1)
OPERATING_MODE = (11, 1)
TORQUE_ENABLE = (64, 1)
HARDWARE_ERROR = (70, 1)
POSITION_D_GAIN = (80, 2)
POSITION_I_GAIN = (82, 2)
POSITION_P_GAIN = (84, 2)
BUS_WATCHDOG = (98, 1)
GOAL_CURRENT = (102, 2)
PROFILE_ACCELERATION = (108, 4)
PROFILE_VELOCITY = (112, 4)
GOAL_POSITION = (116, 4)
PRESENT_VELOCITY = (128, 4)
PRESENT_POSITION = (132, 4)
PRESENT_INPUT_VOLTAGE = (144, 2)  # units of 0.1 V
PRESENT_TEMPERATURE = (146, 1)
MIN_VOLTAGE_LIMIT = (34, 2)  # units of 0.1 V
SIGNED = {GOAL_POSITION[0], PRESENT_VELOCITY[0], PRESENT_POSITION[0]}

# The error byte of every reply. Its top bit is an alert: the servo has a
# hardware error on record, but the instruction itself was carried out. Only the
# lower seven bits say the instruction failed.
ALERT = 0x80
HARDWARE_ERRORS = {0: "input voltage", 2: "overheating", 3: "motor encoder", 4: "electrical shock", 5: "overload"}


def hardware_error_names(value: int) -> list[str]:
    """The Hardware Error Status register as words."""
    return [HARDWARE_ERRORS.get(bit, f"unknown bit {bit}") for bit in range(8) if value >> bit & 1]

TIME_BASED_PROFILE = 0x04  # Drive Mode bit 2
POSITION_MODES = (3, 4)  # position, extended position
CURRENT_BASED_POSITION_MODE = 5
WATCHDOG_500_MS = 25  # units of 20 ms: the servos stop if the program stops talking to them
MAX_TEMPERATURE_C = 55
GRIPPER_GOAL_CURRENT_MA = 600  # vendor follower configuration

# Volatile settings changed for a run and restored afterwards.
_RESTORED = (POSITION_P_GAIN, POSITION_I_GAIN, POSITION_D_GAIN, PROFILE_ACCELERATION, PROFILE_VELOCITY)


@dataclass(frozen=True)
class ServoStatus:
    dxl_id: int
    model: str
    drive_mode: int
    operating_mode: int
    torque_on: bool
    hardware_error: int
    position: int
    temperature_c: int
    position_p_gain: int
    voltage: float
    min_voltage: float


class DynamixelArm:
    """Implements ``understudy.runner.ArmIO`` on the real servos."""

    def __init__(self, port: str):
        import dynamixel_sdk as sdk

        self._sdk = sdk
        self._port = sdk.PortHandler(port)
        self._packet = sdk.PacketHandler(2.0)
        if not self._port.openPort() or not self._port.setBaudRate(BAUD_RATE):
            raise RuntimeError(f"cannot open {port} at {BAUD_RATE} baud")
        self._reader = sdk.GroupSyncRead(self._port, self._packet, PRESENT_VELOCITY[0], 8)
        for dxl_id in SERVO_IDS:
            if not self._reader.addParam(dxl_id):
                raise RuntimeError(f"cannot add servo {dxl_id} to the group read")
        self._writer = sdk.GroupSyncWrite(self._port, self._packet, GOAL_POSITION[0], GOAL_POSITION[1])
        self._saved: dict[tuple[int, int], int] = {}
        self._configured = False
        self._next_tick: float | None = None

    # Register access ---------------------------------------------------------

    def _read(self, dxl_id: int, register: tuple[int, int]) -> int:
        address, size = register
        value, result, error = getattr(self._packet, f"read{size}ByteTxRx")(self._port, dxl_id, address)
        if result != self._sdk.COMM_SUCCESS or error & ~ALERT:
            raise RuntimeError(f"reading servo {dxl_id} address {address}: communication {result}, device error {error}")
        if address in SIGNED and value >= 2 ** (8 * size - 1):
            value -= 2 ** (8 * size)
        return value

    def _write(self, dxl_id: int, register: tuple[int, int], value: int) -> None:
        address, size = register
        raw = value & ((1 << (8 * size)) - 1)
        result, error = getattr(self._packet, f"write{size}ByteTxRx")(self._port, dxl_id, address, raw)
        if result != self._sdk.COMM_SUCCESS or error & ~ALERT:
            raise RuntimeError(f"writing servo {dxl_id} address {address}: communication {result}, device error {error}")

    # Inspection (reads only) -------------------------------------------------

    def status(self) -> list[ServoStatus]:
        rows = []
        for dxl_id in SERVO_IDS:
            model = self._read(dxl_id, MODEL_NUMBER)
            rows.append(
                ServoStatus(
                    dxl_id=dxl_id,
                    model=MODEL_NUMBERS.get(model, f"unknown ({model})"),
                    drive_mode=self._read(dxl_id, DRIVE_MODE),
                    operating_mode=self._read(dxl_id, OPERATING_MODE),
                    torque_on=bool(self._read(dxl_id, TORQUE_ENABLE)),
                    hardware_error=self._read(dxl_id, HARDWARE_ERROR),
                    position=self._read(dxl_id, PRESENT_POSITION),
                    temperature_c=self._read(dxl_id, PRESENT_TEMPERATURE),
                    position_p_gain=self._read(dxl_id, POSITION_P_GAIN),
                    voltage=self._read(dxl_id, PRESENT_INPUT_VOLTAGE) / 10,
                    min_voltage=self._read(dxl_id, MIN_VOLTAGE_LIMIT) / 10,
                )
            )
        return rows

    def check_ready(self) -> list[str]:
        """Reasons the arm must not be driven, empty if there are none. Reads only."""
        problems = []
        for joint_model, row in zip([j.servo for j in arm.ARM_JOINTS] + ["XL330-M288"], self.status()):
            name = f"servo {row.dxl_id}"
            if row.model != joint_model:
                problems.append(f"{name} is a {row.model}, expected {joint_model}")
            if row.hardware_error:
                problems.append(f"{name} reports a hardware error: {', '.join(hardware_error_names(row.hardware_error))}")
            if row.voltage < row.min_voltage:
                problems.append(
                    f"{name} is getting {row.voltage:.1f} V and needs at least {row.min_voltage:.1f} V: "
                    "is the arm's power supply plugged in and switched on?"
                )
            if row.temperature_c >= MAX_TEMPERATURE_C:
                problems.append(f"{name} is at {row.temperature_c} C")
            if not row.drive_mode & TIME_BASED_PROFILE:
                problems.append(f"{name} drive mode {row.drive_mode} is not a time-based profile (vendor setting is 4)")
            is_gripper = row.dxl_id == arm.GRIPPER_DXL_ID
            if is_gripper and row.operating_mode != CURRENT_BASED_POSITION_MODE:
                problems.append(f"{name} (gripper) operating mode {row.operating_mode}, expected current-based position (5)")
            if not is_gripper and row.operating_mode not in POSITION_MODES:
                problems.append(f"{name} operating mode {row.operating_mode}, expected position (3) or extended position (4)")
        return problems

    # Driving -----------------------------------------------------------------

    def configure(self) -> None:
        """Apply the vendor follower settings for a run and switch torque on, holding the present pose."""
        problems = self.check_ready()
        if problems:
            raise RuntimeError("arm is not ready:\n  " + "\n  ".join(problems))
        self._configured = True
        for dxl_id in SERVO_IDS:
            for register in _RESTORED:
                self._saved[(dxl_id, register[0])] = self._read(dxl_id, register)
            self._write(dxl_id, POSITION_P_GAIN, arm.VENDOR_POSITION_P_GAIN)
            self._write(dxl_id, POSITION_I_GAIN, 0)
            self._write(dxl_id, POSITION_D_GAIN, arm.VENDOR_POSITION_D_GAIN)
            self._write(dxl_id, GOAL_POSITION, self._read(dxl_id, PRESENT_POSITION))
        gripper = arm.GRIPPER_DXL_ID
        self._saved[(gripper, GOAL_CURRENT[0])] = self._read(gripper, GOAL_CURRENT)
        self._write(gripper, GOAL_CURRENT, min(self._saved[(gripper, GOAL_CURRENT[0])], GRIPPER_GOAL_CURRENT_MA))
        self.set_profile_time(arm.VENDOR_PROFILE_TIME_S)
        for dxl_id in SERVO_IDS:
            self._write(dxl_id, BUS_WATCHDOG, WATCHDOG_500_MS)
            self._write(dxl_id, TORQUE_ENABLE, 1)

    def set_profile_time(self, seconds: float) -> None:
        """Time every servo takes to reach a new goal (time-based profile)."""
        milliseconds = max(1, int(round(seconds * 1000)))
        for dxl_id in SERVO_IDS:
            self._write(dxl_id, PROFILE_ACCELERATION, milliseconds // 2)
            self._write(dxl_id, PROFILE_VELOCITY, milliseconds)

    def move_slowly(self, ticks: Sequence[int], seconds: float) -> None:
        """Move to a pose over ``seconds``, then return to the vendor profile time."""
        self.set_profile_time(seconds)
        self.write_goals(ticks)
        deadline = time.monotonic() + seconds + 0.5
        while time.monotonic() < deadline:
            self.read()  # keeps the bus watchdog fed
            time.sleep(0.05)
        self.set_profile_time(arm.VENDOR_PROFILE_TIME_S)

    def read(self) -> ArmReading:
        if self._reader.txRxPacket() != self._sdk.COMM_SUCCESS:
            raise RuntimeError("group read of the servos failed")
        positions, velocities = [], []
        for dxl_id in SERVO_IDS:
            if not self._reader.isAvailable(dxl_id, PRESENT_VELOCITY[0], 8):
                raise RuntimeError(f"servo {dxl_id} did not answer")
            values = []
            for address in (PRESENT_VELOCITY[0], PRESENT_POSITION[0]):
                value = self._reader.getData(dxl_id, address, 4)
                values.append(value - 2**32 if value >= 2**31 else value)
            velocities.append(values[0])
            positions.append(values[1])
        return ArmReading(tuple(positions), tuple(velocities))

    def write_goals(self, ticks: Sequence[int]) -> None:
        if len(ticks) != len(SERVO_IDS):
            raise ValueError("six goal positions are needed")
        self._writer.clearParam()
        for dxl_id, tick in zip(SERVO_IDS, ticks):
            raw = int(tick) & 0xFFFFFFFF
            if not self._writer.addParam(dxl_id, list(raw.to_bytes(4, "little"))):
                raise RuntimeError(f"cannot add servo {dxl_id} to the group write")
        if self._writer.txPacket() != self._sdk.COMM_SUCCESS:
            raise RuntimeError("group write of the goal positions failed")

    def advance(self) -> None:
        """Wait for the next control tick."""
        period = 1.0 / arm.CONTROL_HZ
        now = time.monotonic()
        if self._next_tick is None or now - self._next_tick > period:
            self._next_tick = now
        self._next_tick += period
        time.sleep(max(0.0, self._next_tick - time.monotonic()))

    def shutdown(self) -> list[str]:
        """Switch torque off, restore the changed settings and close the port.

        Returns a list of problems. If it reports that torque could not be
        switched off, disconnect the arm's power.
        """
        problems = []
        for dxl_id in SERVO_IDS:
            for _ in range(3):
                try:
                    self._write(dxl_id, TORQUE_ENABLE, 0)
                    break
                except RuntimeError as error:
                    last = str(error)
            else:
                problems.append(f"TORQUE NOT CONFIRMED OFF on servo {dxl_id}: {last}. Disconnect the arm's power.")
        if self._configured:
            for dxl_id in SERVO_IDS:
                try:
                    self._write(dxl_id, BUS_WATCHDOG, 0)
                except RuntimeError as error:
                    problems.append(str(error))
            sizes = {register[0]: register for register in (*_RESTORED, GOAL_CURRENT)}
            for (dxl_id, address), value in self._saved.items():
                try:
                    self._write(dxl_id, sizes[address], value)
                except RuntimeError as error:
                    problems.append(str(error))
        try:
            for dxl_id in SERVO_IDS:
                if self._read(dxl_id, TORQUE_ENABLE):
                    problems.append(f"servo {dxl_id} still reports torque on. Disconnect the arm's power.")
        except RuntimeError as error:
            problems.append(f"could not confirm torque is off: {error}. Disconnect the arm's power.")
        self._port.closePort()
        return problems
