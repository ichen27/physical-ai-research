"""Use a trained pickup policy on the real OMX follower arm.

NOT YET RUN ON HARDWARE. Read the "Real arm" section of the README first, keep
a hand on the arm's power switch, and go through the steps in order:

    python scripts/real_arm.py status                      # reads only
    python scripts/real_arm.py calibrate                   # reads only; you pose the arm by hand
    python scripts/real_arm.py check                       # reads only; you touch the gripper to a mark
    python scripts/real_arm.py start-pose                  # first powered motion: slow move to the start pose
    python scripts/real_arm.py run --policy <policy.onnx> --ball 0.21 0.0 --dry-run
    python scripts/real_arm.py run --policy <policy.onnx> --ball 0.21 0.0

Needs: pip install dynamixel_sdk onnxruntime numpy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from understudy import arm, pickup, runtime  # noqa: E402
from understudy.kinematics import Chain  # noqa: E402
from understudy.runner import PlacedBall, goals_from_command, run_pickup, sensed_from_reading  # noqa: E402

DEFAULT_CALIBRATION = REPO / "calibration.local.json"
BASE_NOTE = (
    "Positions are metres from the centre of the arm's base plate, on the desk: "
    "x forward, y to the arm's left, z up."
)


def usb_serial_ports() -> list[str]:
    patterns = ("cu.usb*", "ttyUSB*", "ttyACM*")
    return sorted(str(path) for pattern in patterns for path in Path("/dev").glob(pattern))


def open_arm(args):
    from understudy.dynamixel import DynamixelArm

    try:
        return DynamixelArm(args.port)
    except ImportError:
        raise SystemExit("the dynamixel_sdk package is missing. Install it with: pip install dynamixel_sdk") from None
    except (OSError, RuntimeError) as error:
        found = ", ".join(usb_serial_ports()) or "none"
        raise SystemExit(
            f"cannot open the arm on {args.port}: {error}\n"
            f"USB serial ports on this computer: {found}\n"
            "Check the arm's USB cable and power, close any other program using the port, "
            "and give the port with --port before the command, for example:\n"
            "  python scripts/real_arm.py --port /dev/cu.usbmodemXXXX status"
        ) from None


def require_torque_off(robot) -> None:
    powered = [row.dxl_id for row in robot.status() if row.torque_on]
    if powered:
        raise SystemExit(f"torque is on for servos {powered}; this step needs the arm limp. Nothing was changed.")


def cmd_status(args) -> None:
    robot = open_arm(args)
    try:
        print("id  model        drive  mode  torque  error  position  temp  P gain")
        for row in robot.status():
            print(
                f"{row.dxl_id:<3} {row.model:<12} {row.drive_mode:<6} {row.operating_mode:<5} "
                f"{'on' if row.torque_on else 'off':<7} {row.hardware_error:<6} {row.position:<9} "
                f"{row.temperature_c:<5} {row.position_p_gain}"
            )
        problems = robot.check_ready()
        print("ready to drive" if not problems else "not ready:\n  " + "\n  ".join(problems))
    finally:
        robot._port.closePort()


def cmd_calibrate(args) -> None:
    robot = open_arm(args)
    try:
        require_torque_off(robot)
        print("The arm must be limp. Nothing is written to the servos.\n")
        print("1. Hold the arm in the model's zero pose:")
        print("   - base pointing straight ahead,")
        print("   - upper arm straight up,")
        print("   - forearm and gripper pointing straight forward, level with the desk,")
        print("   - gripper not rolled.")
        print("   In this pose the gripper tip is 31.3 cm in front of the base centre and 21.1 cm above the desk.")
        input("   Press Enter while holding it there. ")
        zero = robot.read().positions
        print("2. Close the gripper fully by hand, with nothing in it.")
        input("   Press Enter. ")
        closed = robot.read().positions[5]
        print("3. Open the gripper wide by hand.")
        input("   Press Enter. ")
        opened = robot.read().positions[5]
        if abs(opened - closed) < 200:
            raise SystemExit("the gripper reading barely changed; nothing was saved.")
        calibration = arm.Calibration(
            tuple(float(t) for t in zero[:5]), tuple(args.signs), float(closed), 1 if opened > closed else -1
        )
        calibration.save(args.calibration)
        print(f"\nSaved {args.calibration}. Run the 'check' step next: it is what tells you whether this is right.")
    finally:
        robot._port.closePort()


def cmd_check(args) -> None:
    calibration = arm.Calibration.load(args.calibration)
    chain = Chain()
    robot = open_arm(args)
    try:
        require_torque_off(robot)
        print("The arm must be limp. Nothing is written to the servos.")
        print(BASE_NOTE)
        print("Mark a point on the desk with a ruler, for example 20 cm straight in front of the base centre.")
        while True:
            answer = input("Touch the gripper tip to the mark, type its x y in metres (blank to finish): ").split()
            if not answer:
                break
            x, y = float(answer[0]), float(answer[1])
            angles, _ = sensed_from_reading(robot.read(), calibration)
            tip = chain.tip_position(angles[:5])
            error = ((tip[0] - x) ** 2 + (tip[1] - y) ** 2 + tip[2] ** 2) ** 0.5
            print(f"  model says the tip is at x={tip[0]:.3f} y={tip[1]:.3f} z={tip[2]:.3f}; off by {error * 100:.1f} cm")
            print("  good" if error < 0.01 else "  more than 1 cm off: recalibrate, or a sign may be wrong (--signs)")
    finally:
        robot._port.closePort()


def cmd_start_pose(args) -> None:
    calibration = arm.Calibration.load(args.calibration)
    robot = open_arm(args)
    problems = []
    try:
        print("This moves the arm slowly to the start pose (gripper about 12 cm above the desk), holds, and returns.")
        if input("Arm clear of people and objects? Type yes: ").strip().lower() != "yes":
            raise SystemExit("stopped; nothing was changed.")
        rest = robot.read().positions
        robot.configure()
        start = goals_from_command(runtime.start_command(), calibration)
        robot.move_slowly(start, args.move_seconds)
        angles, _ = sensed_from_reading(robot.read(), calibration)
        print("reached (rad):", [round(a, 3) for a in angles], "wanted:", [round(a, 3) for a in runtime.SENSED_START])
        robot.move_slowly(rest, args.move_seconds)
    finally:
        problems = robot.shutdown()
    print("torque off, settings restored" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))


def cmd_run(args) -> None:
    calibration = arm.Calibration.load(args.calibration)
    policy = runtime.OnnxPolicy(args.policy)
    ball = PlacedBall(*args.ball)
    print(BASE_NOTE)
    print(f"Ball taken to be at x={args.ball[0]} y={args.ball[1]}. Hold target {args.target}.")
    print("There is no camera yet: the ball is assumed to stay where it was put until the gripper holds it.")
    robot = open_arm(args)
    problems = []
    try:
        if args.dry_run:
            print("Dry run: the arm stays limp. Showing what the policy would command from the present pose.")
            require_torque_off(robot)
            steps = run_pickup(robot, policy, calibration, ball, args.target, 1.0, advance=robot.advance, send=False)
            for step in steps[:: max(1, len(steps) // 6)]:
                print("  arm targets (rad)", [round(t, 3) for t in step.command.arm_targets],
                      "gripper", "open" if step.command.gripper_open else "closed")
            robot._port.closePort()
            return
        if input("Arm clear of people and objects, hand on the power switch? Type yes: ").strip().lower() != "yes":
            raise SystemExit("stopped; nothing was changed.")
        rest = robot.read().positions
        robot.configure()
        start = goals_from_command(runtime.start_command(), calibration)
        robot.move_slowly(start, args.move_seconds)
        steps = run_pickup(robot, policy, calibration, ball, args.target, args.seconds, advance=robot.advance)
        last = steps[-1]
        print("finished. gripper", "open" if last.command.gripper_open else "closed",
              f"finger angle {last.joint_angles[5]:.2f} rad (above {PlacedBall.HELD_FINGER_RAD} with the gripper closed means it is holding something)")
        input("Press Enter to open the gripper and return to rest. ")
        release = list(robot.read().positions)
        release[5] = calibration.finger_to_ticks(pickup.GRIPPER_OPEN_RAD)
        robot.move_slowly(release, 1.0)
        robot.move_slowly(rest, args.move_seconds)
    finally:
        if not args.dry_run:
            problems = robot.shutdown()
            print("torque off, settings restored" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", default="/dev/cu.usbmodem21401", help="Serial port of the arm's USB controller.")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show the servos' state. Reads only.").set_defaults(run=cmd_status)
    calibrate = commands.add_parser("calibrate", help="Capture this arm's encoder offsets by hand. Reads only.")
    calibrate.add_argument("--signs", type=int, nargs=5, default=[1, 1, 1, 1, 1], choices=[1, -1])
    calibrate.set_defaults(run=cmd_calibrate)
    commands.add_parser("check", help="Test the calibration against marks on the desk. Reads only.").set_defaults(run=cmd_check)
    start = commands.add_parser("start-pose", help="Slow powered move to the start pose and back.")
    start.add_argument("--move_seconds", type=float, default=4.0)
    start.set_defaults(run=cmd_start_pose)
    run = commands.add_parser("run", help="Run the pickup policy.")
    run.add_argument("--policy", required=True)
    run.add_argument("--ball", type=float, nargs=2, required=True, metavar=("X", "Y"))
    run.add_argument("--target", type=float, nargs=3, default=[0.19, 0.0, 0.13], metavar=("X", "Y", "Z"))
    run.add_argument("--seconds", type=float, default=pickup.EPISODE_SECONDS)
    run.add_argument("--move_seconds", type=float, default=4.0)
    run.add_argument("--dry-run", action="store_true", help="Compute commands without powering the arm.")
    run.set_defaults(run=cmd_run)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
