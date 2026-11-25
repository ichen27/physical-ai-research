import argparse
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "real_arm.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("real_arm_tool", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def driver_that_raises(error: Exception) -> types.ModuleType:
    """Stands in for understudy.dynamixel, so the test needs neither the servo library nor an arm."""

    class DynamixelArm:
        def __init__(self, port: str):
            raise error

    module = types.ModuleType("understudy.dynamixel")
    module.DynamixelArm = DynamixelArm
    return module


class OpeningTheArm(unittest.TestCase):
    def open_with(self, error: Exception) -> str:
        tool = load_tool()
        args = argparse.Namespace(port="/dev/cu.usbmodem-absent")
        with mock.patch.dict(sys.modules, {"understudy.dynamixel": driver_that_raises(error)}):
            with self.assertRaises(SystemExit) as raised:
                tool.open_arm(args)
        return str(raised.exception)

    def test_a_missing_port_is_explained_without_a_traceback(self):
        message = self.open_with(FileNotFoundError(2, "could not open port"))
        self.assertIn("/dev/cu.usbmodem-absent", message)
        self.assertIn("--port", message)
        self.assertIn("USB serial ports on this computer", message)

    def test_a_port_that_opens_at_the_wrong_speed_is_explained(self):
        message = self.open_with(RuntimeError("cannot open /dev/cu.usbmodem-absent at 1000000 baud"))
        self.assertIn("1000000 baud", message)

    def test_a_missing_servo_library_says_how_to_install_it(self):
        message = self.open_with(ImportError("No module named 'dynamixel_sdk'"))
        self.assertIn("pip install dynamixel_sdk", message)


if __name__ == "__main__":
    unittest.main()
