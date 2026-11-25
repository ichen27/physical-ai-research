"""Use a trained pickup policy on the real OMX follower arm.

ONLY THE STEPS THAT READ HAVE RUN ON A REAL ARM; nothing here has driven its motors yet. Read the "Real arm" section of the README first, keep
a hand on the arm's power switch, and go through the steps in order:

    python scripts/real_arm.py status                      # reads only
    python scripts/real_arm.py clear-errors                # only if status reports a hardware error
    python scripts/real_arm.py touch 20 0                  # reads only; tip on a desk mark, four or more times
    python scripts/real_arm.py fit-touches                 # works out and saves the calibration
    python scripts/real_arm.py where                       # reads only; tip on the spot where the ball will sit
    python scripts/real_arm.py start-pose                  # first powered motion: slow move to the start pose
    python scripts/real_arm.py run --policy <policy.onnx> --ball 0.21 0.0 --dry-run
    python scripts/real_arm.py run --policy <policy.onnx> --ball 0.21 0.0

Needs: pip install dynamixel_sdk onnxruntime numpy
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from understudy import arm, pickup, runtime  # noqa: E402
from understudy.kinematics import Chain  # noqa: E402
from understudy.runner import PlacedBall, goals_from_command, run_pickup, sensed_from_reading  # noqa: E402

DEFAULT_CALIBRATION = REPO / "calibration.local.json"
DEFAULT_TOUCHES = REPO / "touches.local.json"
BASE_NOTE = (
    "Positions are metres in the arm model's frame, on the desk: x forward, y to the arm's left, z up. "
    "Its origin is 1.1 cm in front of the swivel axis. The 'where' step reads a position off the arm itself."
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


def cmd_clear_errors(args) -> None:
    robot = open_arm(args)
    try:
        restarted = robot.restart_faulted()
    except RuntimeError as error:
        robot._port.closePort()
        raise SystemExit(f"nothing was restarted: {error}") from None
    print(f"restarted servos {restarted}" if restarted else "no servo has a hardware error; nothing was restarted")
    robot._port.closePort()
    cmd_status(args)


def cmd_status(args) -> None:
    from understudy.dynamixel import hardware_error_names

    robot = open_arm(args)
    try:
        print("id  model        drive  mode  torque  volts  position  temp  P gain  hardware error")
        for row in robot.status():
            print(
                f"{row.dxl_id:<3} {row.model:<12} {row.drive_mode:<6} {row.operating_mode:<5} "
                f"{'on' if row.torque_on else 'off':<7} {row.voltage:<6.1f} {row.position:<9} "
                f"{row.temperature_c:<5} {row.position_p_gain:<7} "
                f"{', '.join(hardware_error_names(row.hardware_error)) or 'none'}"
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


def cmd_touch(args) -> None:
    import time

    from understudy import touch_calibration

    robot = open_arm(args)
    try:
        require_torque_off(robot)
        reads = []
        for _ in range(5):
            reads.append(robot.read().positions[:5])
            time.sleep(0.1)
    finally:
        robot._port.closePort()
    if max(max(column) - min(column) for column in zip(*reads)) > 3:
        raise SystemExit("the arm moved while it was being read. Hold it still and try again; nothing was recorded.")
    touches = touch_calibration.load_touches(args.touches)
    touches.append(touch_calibration.Touch(tuple(int(t) for t in reads[-1]), args.forward_cm / 100, args.left_cm / 100))
    touch_calibration.save_touches(args.touches, touches)
    print(
        f"recorded touch {len(touches)}: tip on the mark {args.forward_cm:g} cm forward and {args.left_cm:g} cm left "
        f"of the swivel axis, servo readings {list(touches[-1].ticks)}"
    )
    missing = touch_calibration.MIN_TOUCHES - len(touches)
    print(f"{missing} more needed before a fit" if missing > 0 else "enough for a fit: run fit-touches (more touches make it better)")


def cmd_clear_touches(args) -> None:
    from understudy import touch_calibration

    touch_calibration.save_touches(args.touches, [])
    print(f"cleared {args.touches}")


def cmd_fit_touches(args) -> None:
    from understudy import touch_calibration

    touches = touch_calibration.load_touches(args.touches)
    try:
        fit = touch_calibration.fit(touches, args.signs)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    for number, (touch, (reach, height)) in enumerate(zip(touches, fit.errors_m), 1):
        print(
            f"  touch {number}: mark {touch.distance_m() * 100:5.1f} cm from the swivel axis; "
            f"model tip {reach * 1000:+5.1f} mm in distance, {height * 1000:+5.1f} mm in height"
        )
    print(f"zero points (ticks): {[round(z, 1) for z in fit.zero_ticks]}   typical error {fit.rms_m * 1000:.1f} mm")
    spread = math.degrees(fit.swivel_spread_rad)
    print(f"swivel zero: the touches disagree by up to {spread:.1f} degrees about which way is straight ahead")
    if spread > 3:
        print(
            "  The base probably turned between touches. Distance and height are not affected. Give the ball's\n"
            "  position by touching the tip to its spot (the 'where' step), so the swivel's zero cancels out."
        )
    if fit.ambiguous():
        print(
            f"WARNING: a different set of zero points fits nearly as well ({fit.runner_up_rms_m * 1000:.1f} mm). "
            "Add touches with the arm in different shapes before trusting this."
        )
    if fit.rms_m > args.max_error_mm / 1000:
        raise SystemExit(
            f"not saved: the typical error is over {args.max_error_mm:g} mm. Check the marks, add touches in "
            "different arm shapes, or try --signs if a joint may run backwards."
        )
    closed, gripper_sign = None, 1
    if Path(args.calibration).exists():
        previous = arm.Calibration.load(args.calibration)
        closed, gripper_sign = previous.gripper_closed_tick, previous.gripper_sign
    arm.Calibration(tuple(fit.zero_ticks), tuple(args.signs), closed, gripper_sign).save(args.calibration)
    print(f"saved {args.calibration} (gripper settings kept as they were)")


def cmd_where(args) -> None:
    calibration = arm.Calibration.load(args.calibration)
    robot = open_arm(args)
    try:
        require_torque_off(robot)
        angles, _ = sensed_from_reading(robot.read(), calibration)
    finally:
        robot._port.closePort()
    x, y, z = Chain().tip_position(angles[:5])
    print(f"gripper tip: x={x:.3f} y={y:.3f} z={z:.3f} (metres, arm model's frame)")
    print(f"if the tip is resting on the desk where the ball will sit, the height should read near zero: it is {z * 1000:+.0f} mm")
    print(f"to put the ball there: run ... --ball {x:.3f} {y:.3f}")


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
    parser.add_argument("--touches", default=str(DEFAULT_TOUCHES))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show the servos' state. Reads only.").set_defaults(run=cmd_status)
    commands.add_parser(
        "clear-errors", help="Restart servos that report a hardware error. The arm stays limp; nothing moves."
    ).set_defaults(run=cmd_clear_errors)
    calibrate = commands.add_parser("calibrate", help="Capture this arm's encoder offsets by hand. Reads only.")
    calibrate.add_argument("--signs", type=int, nargs=5, default=[1, 1, 1, 1, 1], choices=[1, -1])
    calibrate.set_defaults(run=cmd_calibrate)
    touch = commands.add_parser(
        "touch", help="Record the limp arm's pose with the gripper tip on a desk mark. Reads only."
    )
    touch.add_argument("forward_cm", type=float, help="Mark's distance from the swivel axis along the ruler line.")
    touch.add_argument("left_cm", type=float, help="Mark's distance to the left of the ruler line (negative for right).")
    touch.set_defaults(run=cmd_touch)
    commands.add_parser("clear-touches", help="Forget the recorded touches.").set_defaults(run=cmd_clear_touches)
    fit = commands.add_parser("fit-touches", help="Work out the calibration from the recorded touches.")
    fit.add_argument("--signs", type=int, nargs=5, default=[1, 1, 1, 1, 1], choices=[1, -1])
    fit.add_argument("--max_error_mm", type=float, default=10.0)
    fit.set_defaults(run=cmd_fit_touches)
    commands.add_parser(
        "where", help="Show where the calibration says the limp arm's gripper tip is. Reads only."
    ).set_defaults(run=cmd_where)
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
