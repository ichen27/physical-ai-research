"""The servo driver against a stand-in for the servo library, so no arm is needed.

The values in ``unpowered_arm`` are what the author's arm reported the first
time it was read, with its 12 V supply off and only USB power: three XL430
servos flagging an input-voltage error, and every reply from them carrying the
protocol's alert flag.
"""

import sys
import types
import unittest
from unittest import mock

from understudy import dynamixel
from understudy.runner import SERVO_IDS

ALERT = 0x80
XL430, XL330 = 1060, 1200


class FakeBus:
    def __init__(self):
        self.registers: dict[tuple[int, int], int] = {}
        self.instruction_error: dict[int, int] = {}
        self.writes: list[tuple[int, int, int]] = []

    def set(self, dxl_id: int, register: tuple[int, int], value: int) -> None:
        self.registers[(dxl_id, register[0])] = value

    def error_byte(self, dxl_id: int) -> int:
        alert = ALERT if self.registers.get((dxl_id, dynamixel.HARDWARE_ERROR[0]), 0) else 0
        return alert | self.instruction_error.get(dxl_id, 0)

    def library(self) -> types.ModuleType:
        bus = self

        class PortHandler:
            def __init__(self, port):
                self.closed = False

            def openPort(self):
                return True

            def setBaudRate(self, baud):
                return True

            def closePort(self):
                self.closed = True

        class PacketHandler:
            def __init__(self, version):
                pass

            def _read(self, port, dxl_id, address):
                return bus.registers.get((dxl_id, address), 0), 0, bus.error_byte(dxl_id)

            def _write(self, port, dxl_id, address, value):
                bus.writes.append((dxl_id, address, value))
                if not bus.instruction_error.get(dxl_id):
                    bus.registers[(dxl_id, address)] = value
                return 0, bus.error_byte(dxl_id)

            read1ByteTxRx = read2ByteTxRx = read4ByteTxRx = _read
            write1ByteTxRx = write2ByteTxRx = write4ByteTxRx = _write

        class Group:
            def __init__(self, *args):
                pass

            def addParam(self, *args):
                return True

            def clearParam(self):
                pass

        module = types.ModuleType("dynamixel_sdk")
        module.COMM_SUCCESS = 0
        module.PortHandler = PortHandler
        module.PacketHandler = PacketHandler
        module.GroupSyncRead = module.GroupSyncWrite = Group
        return module


def healthy_arm() -> FakeBus:
    bus = FakeBus()
    for dxl_id in SERVO_IDS:
        big = dxl_id <= 13
        bus.set(dxl_id, dynamixel.MODEL_NUMBER, XL430 if big else XL330)
        bus.set(dxl_id, dynamixel.DRIVE_MODE, 4)
        bus.set(dxl_id, dynamixel.OPERATING_MODE, 5 if dxl_id == 16 else 3)
        bus.set(dxl_id, dynamixel.MIN_VOLTAGE_LIMIT, 60 if big else 35)
        bus.set(dxl_id, dynamixel.PRESENT_INPUT_VOLTAGE, 120 if big else 50)
        bus.set(dxl_id, dynamixel.PRESENT_TEMPERATURE, 25)
        bus.set(dxl_id, dynamixel.PRESENT_POSITION, 2048)
    return bus


def unpowered_arm() -> FakeBus:
    bus = healthy_arm()
    for dxl_id in SERVO_IDS:
        bus.set(dxl_id, dynamixel.PRESENT_INPUT_VOLTAGE, 46)
    for dxl_id in (11, 12, 13):
        bus.set(dxl_id, dynamixel.HARDWARE_ERROR, 0b1)
    return bus


def open_arm(bus: FakeBus) -> dynamixel.DynamixelArm:
    with mock.patch.dict(sys.modules, {"dynamixel_sdk": bus.library()}):
        return dynamixel.DynamixelArm("/dev/fake")


class AlertFlag(unittest.TestCase):
    def test_a_servo_with_a_hardware_error_can_still_be_read(self):
        rows = open_arm(unpowered_arm()).status()
        self.assertEqual([row.dxl_id for row in rows], list(SERVO_IDS))
        self.assertEqual(rows[0].hardware_error, 1)
        self.assertAlmostEqual(rows[0].voltage, 4.6)

    def test_an_instruction_error_still_fails(self):
        bus = healthy_arm()
        bus.instruction_error[12] = 0x03  # the servo rejected the instruction
        with self.assertRaises(RuntimeError) as raised:
            open_arm(bus).status()
        self.assertIn("servo 12", str(raised.exception))

    def test_torque_is_confirmed_off_on_a_servo_that_raised_an_alert_mid_run(self):
        bus = healthy_arm()
        robot = open_arm(bus)
        robot.configure()
        self.assertEqual(bus.registers[(11, dynamixel.TORQUE_ENABLE[0])], 1)
        bus.set(11, dynamixel.HARDWARE_ERROR, 0b100000)  # overload
        problems = robot.shutdown()
        self.assertEqual(problems, [])
        self.assertTrue(all(bus.registers[(i, dynamixel.TORQUE_ENABLE[0])] == 0 for i in SERVO_IDS))


class Readiness(unittest.TestCase):
    def test_a_healthy_arm_is_ready(self):
        self.assertEqual(open_arm(healthy_arm()).check_ready(), [])

    def test_an_arm_on_usb_power_alone_is_reported_as_unpowered(self):
        problems = open_arm(unpowered_arm()).check_ready()
        text = "\n".join(problems)
        self.assertIn("servo 11 reports a hardware error: input voltage", text)
        self.assertIn("4.6 V", text)
        self.assertIn("power supply", text)
        self.assertNotIn("servo 14", text)

    def test_an_unpowered_arm_is_never_given_torque(self):
        bus = unpowered_arm()
        with self.assertRaises(RuntimeError):
            open_arm(bus).configure()
        self.assertEqual(bus.writes, [])

    def test_hardware_errors_are_named(self):
        self.assertEqual(dynamixel.hardware_error_names(0), [])
        self.assertEqual(dynamixel.hardware_error_names(0b100101), ["input voltage", "overheating", "overload"])
        self.assertEqual(dynamixel.hardware_error_names(0b10), ["unknown bit 1"])


if __name__ == "__main__":
    unittest.main()
