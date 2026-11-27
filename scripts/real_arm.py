"""Use a trained pickup policy on the real OMX follower arm.

ON A REAL ARM this has picked the ball up once, with the speed limits on; without them the arm moved at full speed and hit the desk. Read the "Real arm" section of the README first, keep
a hand on the arm's power switch, and go through the steps in order:

    python scripts/real_arm.py status                      # reads only
    python scripts/real_arm.py clear-errors                # only if status reports a hardware error
    python scripts/real_arm.py touch 20 0                  # reads only; tip on a desk mark, four or more times
    python scripts/real_arm.py fit-touches                 # works out and saves the calibration
    python scripts/real_arm.py where                       # reads only; tip on the spot where the ball will sit
    python scripts/real_arm.py start-pose                  # first powered motion: slow move to the start pose
    python scripts/real_arm.py run --policy <policy.onnx> --ball 0.21 0.0 --dry-run
    python scripts/real_arm.py run --policy <policy.onnx> --ball 0.21 0.0

With the camera on the wrist, the ball's position need not be typed in. Once, to calibrate the camera:

    python scripts/real_arm.py camera-collect              # powered, slow: looks at the ball from many poses
    python scripts/real_arm.py camera-fit                  # works out and saves the camera
    python scripts/real_arm.py look                        # reads only; one picture, says where the ball is

and then:

    python scripts/real_arm.py run --policy <policy.onnx> --camera look --find_only   # find it and say where; no pickup
    python scripts/real_arm.py run --policy <policy.onnx> --camera look   # one look, then pick it up
    python scripts/real_arm.py run --policy <policy.onnx> --camera look --eye <eye.onnx>   # the same, a trained eye finding the ball
    python scripts/real_arm.py run --policy <policy.onnx> --camera look --bowl 0.0 0.24 0.0 0.165 0.065   # ... and drop it in a bowl
    python scripts/real_arm.py run --policy <policy.onnx> --camera look --set_down 80 22   # ... or put it down on the desk
    python scripts/real_arm.py run --policy <policy.onnx> --camera live   # follow the ball; start from Terminal
    python scripts/real_arm.py run --policy <policy.onnx> --camera live --driver <driver.onnx>   # a trained student drives from the pictures

Needs: pip install dynamixel_sdk onnxruntime numpy opencv-python-headless
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from understudy import arm, pickup, runtime  # noqa: E402
from understudy.kinematics import Chain  # noqa: E402
from understudy.runner import (  # noqa: E402
    REACH_BAND_M,
    CameraBall,
    PlacedBall,
    SwivelLimit,
    goals_from_command,
    in_turned_frame,
    run_pickup,
    sensed_from_reading,
)

DEFAULT_CALIBRATION = REPO / "calibration.local.json"
DEFAULT_TOUCHES = REPO / "touches.local.json"
DEFAULT_CAMERA = REPO / "camera.local.json"
DEFAULT_SIGHTINGS = REPO / "sightings.local.json"
# A camera whose answers for a ball that did not move are spread wider than this is not saved.
MAX_CAMERA_ERROR_MM = 8.0
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
        try:
            start = goals_from_command(runtime.start_command(), calibration)
            robot.move_slowly(start, args.move_seconds)
            angles, _ = sensed_from_reading(robot.read(), calibration)
            print("reached (rad):", [round(a, 3) for a in angles], "wanted:", [round(a, 3) for a in runtime.SENSED_START])
            if args.hold_seconds > 0:
                print(f"holding the start pose for {args.hold_seconds:g} s", flush=True)
            deadline = time.monotonic() + args.hold_seconds
            while time.monotonic() < deadline:  # reading keeps the servos' watchdog fed
                robot.read()
                time.sleep(0.05)
        finally:
            # Whatever happened, lower the arm to where it started instead of letting it drop.
            try:
                robot.move_slowly(rest, args.move_seconds)
            except Exception as error:
                print(f"could not lower the arm to rest: {error}")
    finally:
        problems = robot.shutdown()
    print("torque off, settings restored" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))


def load_camera(args):
    """The calibrated wrist camera and the ball's colour as it sees it."""
    from understudy import vision
    from understudy.camera_model import WristCamera

    path = Path(args.camera_file)
    if not path.exists():
        raise SystemExit(f"{path} does not exist. Calibrate the camera first: camera-collect, then camera-fit.")
    return WristCamera.load(path), vision.colour_from(json.loads(path.read_text()).get("colour"))


def load_eye(args, driver: bool = False):
    """The trained eye named by --eye, or the whole student named by --driver, checked against this arm's camera file if there is one."""
    from understudy.eye import Driver, Eye

    option, file = ("--driver", args.driver) if driver else ("--eye", args.eye)
    try:
        eye = Driver(file) if driver else Eye(file)
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"{option}: {error}") from None
    trained = eye.notes.get("student_training", {}).get("nominal_camera")
    path = Path(args.camera_file)
    if trained is None or not path.exists():
        print(f"The eye {Path(file).name} cannot be checked against this arm's camera: "
              + ("it does not say what camera it was trained for." if trained is None else f"{path} does not exist."))
        return eye
    from understudy.camera_model import WristCamera, rotation_from_vector

    here = WristCamera.load(path)
    focal = abs(here.focal_px / here.width * trained["width"] / trained["focal_px"] - 1.0)
    shift = math.dist(here.mount_xyz, trained["mount_xyz"])
    between = rotation_from_vector(here.mount_rotation).T @ rotation_from_vector(trained["mount_rotation"])
    turn = math.degrees(math.acos(max(-1.0, min(1.0, (float(between[0][0] + between[1][1] + between[2][2]) - 1.0) / 2.0))))
    said = (f"The eye {Path(file).name} was trained for a camera whose lens differs from this arm's by {focal * 100:.1f}% "
            f"and whose mount differs by {shift * 1000:.1f} mm and {turn:.1f} degrees.")
    # It was trained with the camera up to 4%, 3 mm and 1.5 degrees from where it was taken to be.
    if focal > 0.10 or shift > 0.010 or turn > 4.0:
        raise SystemExit(f"{option}: {said} That is too far for its answers to be trusted; train or export one for this camera.")
    print(said + ("" if focal <= 0.04 and shift <= 0.003 and turn <= 1.5 else " That is more than it was trained to allow for: expect it to be less exact."))
    return eye


def watcher(args, robot=None, prefix="look"):
    from understudy.camera_feed import WatcherCamera

    return WatcherCamera(args.watch_folder, keep_alive=robot.read if robot is not None else None, prefix=prefix)


def hold_still(robot, seconds: float) -> None:
    """Stay put, powered, for a while. Reading keeps the servos' watchdog fed."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        robot.read()
        time.sleep(0.05)


def cmd_look(args) -> None:
    from understudy import vision
    from understudy.camera_feed import CameraError
    from understudy.runner import pick_ball

    camera = colour = None
    if Path(args.camera_file).exists():
        camera, colour = load_camera(args)
    colour = colour or vision.BallColour()
    try:
        picture = watcher(args).picture()
    except CameraError as error:
        raise SystemExit(str(error)) from None
    blobs = vision.find_blobs(picture, colour)
    print(f"{len(blobs)} patch(es) of the ball's colour in the picture")
    for blob in blobs:
        print(f"  at pixel ({blob.u:.0f}, {blob.v:.0f}), radius {blob.radius_px:.0f}, roundness {blob.roundness:.2f}"
              + (", cut off by the edge" if blob.cut_off else ""))
    chosen, note = None, "camera not calibrated"
    if camera is None:
        chosen = next((blob for blob in blobs if blob.whole()), None)
        print("The camera is not calibrated yet (camera-collect, camera-fit), so there is no position to give.")
    else:
        calibration = arm.Calibration.load(args.calibration)
        robot = open_arm(args)
        try:
            angles, _ = sensed_from_reading(robot.read(), calibration)
        finally:
            robot._port.closePort()
        chain = Chain()
        point = pick_ball(camera, blobs, angles[:5], chain)
        if point is None:
            note = "no ball seen whole on the desk within reach"
            print("No ball: no patch is whole, on the desk within the arm's reach, and the size a ball there would be.")
        else:
            chosen = min(blobs, key=lambda blob: math.dist(camera.locate(blob.u, blob.v, angles[:5], chain) or (9, 9, 9), point))
            note = f"ball at x={point[0]:.3f} y={point[1]:.3f}"
            print(f"ball at x={point[0]:.3f} y={point[1]:.3f} (metres, arm model's frame)")
    if args.save:
        import cv2

        cv2.imwrite(args.save, vision.draw(picture, blobs, chosen, note))
        print(f"picture saved to {args.save}")


def cmd_camera_collect(args) -> None:
    from understudy import camera_calibration, camera_collect, vision
    from understudy.camera_feed import CameraError

    calibration = arm.Calibration.load(args.calibration)
    robot = open_arm(args)
    problems = []
    sightings, colour, rough = [], None, None
    try:
        print(
            "This raises the arm to the start pose and holds it there. Each time it is holding still there, put the\n"
            "ball on the desk in front of the gripper (the first time) or move it to another spot (after that).\n"
            "The arm sees the ball arrive and then looks at it from about twenty poses, all with the gripper\n"
            f"raised, no joint turning faster than {math.degrees(camera_collect.MAX_JOINT_SPEED):.0f} degrees a second. "
            "Keep hands clear while it moves."
        )
        if input("Arm clear of people and objects, hand on the power switch? Type yes: ").strip().lower() != "yes":
            raise SystemExit("stopped; nothing was changed.")
        rest = robot.read().positions
        camera = watcher(args, robot, "cal")
        robot.configure()
        try:
            robot.move_slowly(goals_from_command(runtime.start_command(), calibration), args.move_seconds)
            before = camera.picture()
            height, width = before.shape[:2]
            collector = None
            for place in range(args.places):
                print(
                    ("PUT THE BALL on the desk in front of the gripper now." if place == 0
                     else "MOVE THE BALL to a different spot in front of the gripper now.")
                    + f" Waiting up to about {args.wait_pictures * 2} s.", flush=True)
                try:
                    arrival = camera_collect.wait_for_ball(camera.picture, before, args.wait_pictures)
                except camera_collect.NoBall as error:
                    print(f"{error}" if place == 0 else "The ball was not moved; going on with what there is.")
                    break
                print(f"ball seen at pixel ({arrival.blob.u:.0f}, {arrival.blob.v:.0f}); starting in 3 s", flush=True)
                if colour is None:
                    colour = arrival.colour
                    collector = camera_collect.Collector(robot, calibration, camera.picture, colour)
                hold_still(robot, 3.0)
                try:
                    seen = collector.collect_place(place, width, height, known=rough)
                except camera_collect.NoBall as error:
                    print(str(error))
                    break
                sightings += seen
                print(f"place {place + 1}: the ball was seen from {len(seen)} poses", flush=True)
                if len(sightings) >= camera_calibration.MIN_SIGHTINGS:
                    # The arm is powered and must keep hearing from the computer while this is worked out.
                    rough = camera_calibration.fit(sightings, calibration, width, height, lens=False,
                                                   tick=collector.pulse).camera
                before = camera.picture()
        except CameraError as error:
            print(str(error))
        finally:
            if sightings:
                camera_calibration.save_sightings(
                    args.sightings, sightings, colour=vars(colour), width=width, height=height,
                    taken=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                print(f"{len(sightings)} sightings saved to {args.sightings}. Next: camera-fit")
            # Whatever happened, lower the arm to where it started instead of letting it drop.
            try:
                robot.move_slowly(rest, args.move_seconds)
            except Exception as error:
                print(f"could not lower the arm to rest: {error}")
    finally:
        problems = robot.shutdown()
    print("torque off, settings restored" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))


def cmd_camera_fit(args) -> None:
    from understudy import camera_calibration, camera_collect

    calibration = arm.Calibration.load(args.calibration)
    sightings = camera_calibration.load_sightings(args.sightings)
    notes = camera_calibration.load_notes(args.sightings)
    if not sightings:
        raise SystemExit(f"no sightings in {args.sightings}; run camera-collect first")
    size = (int(notes.get("width", 1280)), int(notes.get("height", 720)))
    numbers = sorted({s.place for s in sightings})
    known = {}
    for printed, x, y in args.known or []:
        if not 1 <= int(printed) <= len(numbers) or printed != int(printed):
            raise SystemExit(f"--known {printed:g}: the places are numbered 1 to {len(numbers)}")
        known[numbers[int(printed) - 1]] = (x, y)
    try:
        fit = camera_calibration.fit(sightings, calibration, *size, known=known)
        alone = camera_calibration.fit(sightings, calibration, *size) if known else fit
    except ValueError as error:
        raise SystemExit(str(error)) from None
    camera = fit.camera
    for place, (x, y) in enumerate(fit.places):
        errors = [e for s, e in zip(sightings, fit.desk_errors_m) if s.place == numbers[place]]
        told = numbers[place] in known
        print(f"  place {place + 1}: ball {'known to be' if told else 'at'} x={x:.3f} y={y:.3f}, seen from {len(errors)} poses; "
              f"they {'put it' if told else 'disagree by'} {1000 * (sum(e * e for e in errors) / len(errors)) ** 0.5:.1f} mm "
              f"{'from there ' if told else ''}typically, {1000 * max(errors):.1f} mm at worst")
        if told:
            guess = alone.places[place]
            print(f"      the pictures alone would have put it at x={guess[0]:.3f} y={guess[1]:.3f}, "
                  f"{1000 * math.dist(guess, (x, y)):.0f} mm away: that is how far the pictures and the known position disagree")
    if not known:
        print("  No place's position was given (--known). The camera then agrees with itself but is only as true as the arm's\n"
              "  joint zeros with the arm raised, which this fit cannot check. See the README.")
    print(f"focal length {camera.focal_px:.0f} pixels (field of view {2 * math.degrees(math.atan(camera.width / 2 / camera.focal_px)):.0f} "
          f"degrees across); picture centre {camera.centre_px[0]:.0f}, {camera.centre_px[1]:.0f}; distortion {camera.distortion:+.3f}"
          + ("" if fit.fitted_lens else " (centre and distortion left at their defaults)"))
    print(f"camera sits {-camera.mount_xyz[0] * 100:.1f} cm behind the gripper tip, {camera.mount_xyz[1] * 100:+.1f} cm to its left, "
          f"{camera.mount_xyz[2] * 100:+.1f} cm above")
    print(f"typical error: {fit.pixel_rms:.1f} pixels in the picture, {fit.desk_rms_m * 1000:.1f} mm on the desk")
    moved = camera_collect.moved_places(fit, sightings)
    if moved:
        print(f"WARNING: at place(s) {[m + 1 for m in moved]} the ball was probably nudged while the arm was looking.")
    if fit.ambiguous():
        print(f"WARNING: a different camera fits nearly as well ({fit.runner_up_pixel_rms:.1f} pixels). "
              "Collect again with the ball in two places.")
    if fit.desk_rms_m * 1000 > args.max_error_mm or fit.ambiguous():
        raise SystemExit(f"not saved: the typical error must be under {args.max_error_mm:g} mm and the answer unambiguous.")
    camera.save(args.camera_file, colour=notes.get("colour"), desk_rms_mm=round(fit.desk_rms_m * 1000, 2),
                pixel_rms=round(fit.pixel_rms, 2), sightings=len(sightings), places=len(fit.places),
                known_places={str(numbers.index(number) + 1): list(xy) for number, xy in known.items()},
                fitted=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"saved {args.camera_file}")


def cmd_run(args) -> None:
    if (args.ball is None) == (args.camera is None):
        raise SystemExit("give the ball's position with --ball X Y, or let the camera find it with --camera look|live")
    if args.camera and args.dry_run:
        raise SystemExit("a dry run leaves the arm limp, and the camera sees the desk only from the raised start pose; use --ball")
    place = None
    if args.set_down and args.bowl:
        raise SystemExit("say where the ball goes with --bowl (into a bowl) or --set_down (onto the desk), not both")
    if args.set_down or args.bowl:
        from understudy import carry

        option = "--bowl" if args.bowl else "--set_down"
        if args.bowl:
            place = carry.SetDown.from_touches(args.bowl[:2], args.bowl[2:5])
        else:
            place = carry.SetDown(math.radians(args.set_down[0]), args.set_down[1] / 100.0)
        try:
            carry.set_down_poses(place)  # refuse a place the arm cannot use before anything is powered
        except ValueError as error:
            raise SystemExit(f"{option}: {error}") from None
        if args.dry_run:
            raise SystemExit(f"a dry run does not pick anything up, so there is nothing to carry; leave out {option}")
        if args.ball:
            # With --ball the arm faces straight ahead.
            blocked = carry.in_the_way(place, args.ball, facing_rad=0.0, swing_right_rad=carry.REAL_ARM_SWING_RIGHT_RAD)
            if blocked:
                raise SystemExit(f"{option}: {blocked}. Move the bowl or the ball; nothing was powered.")
    calibration = arm.Calibration.load(args.calibration)
    policy = runtime.OnnxPolicy(args.policy)
    print(BASE_NOTE)
    wrist = colour = live = trained_eye = trained_driver = None
    if args.eye and args.camera != "look":
        raise SystemExit("--eye finds the ball in still pictures: use it with --camera look")
    if args.driver and args.camera != "live":
        raise SystemExit("--driver drives the arm from the camera's stream of pictures: use it with --camera live")
    if args.camera:
        from understudy import seek, vision
        from understudy.camera_feed import CameraError, LiveCamera

        if args.eye:
            trained_eye = load_eye(args)
        elif args.driver:
            trained_driver = load_eye(args, driver=True)
        else:
            wrist, colour = load_camera(args)
        if args.camera == "live":
            # Before the arm is powered, so that a camera macOS will not open costs nothing.
            try:
                live = LiveCamera(ffmpeg=args.ffmpeg, full_width=wrist.width if wrist is not None else 1280)
                live.wait_for_first()
            except CameraError as error:
                raise SystemExit(str(error)) from None
        print("The camera on the wrist will look for the ball"
              + (" and follow it." if live else " once; after that the ball is assumed to stay where it was seen.")
              + (" A trained eye, not the hand-written finder, says where the ball is; the arm does not move towards a ball it is not sure of." if trained_eye else "")
              + (" A trained student, shown the pictures, moves the arm itself; --policy is used only for how its commands are scaled. "
                 "It stops, and the arm is lowered, if it loses sight of the ball or the pictures stop coming." if trained_driver else ""))
    else:
        print(f"Ball taken to be at x={args.ball[0]} y={args.ball[1]}.")
        print("No camera: the ball is assumed to stay where it was put until the gripper holds it.")
    print(f"Hold target {args.target}.")
    robot = open_arm(args)
    problems = []
    try:
        if args.dry_run:
            print("Dry run: the arm stays limp. Showing what the policy would command from the present pose.")
            require_torque_off(robot)
            steps = run_pickup(robot, policy, calibration, PlacedBall(*args.ball), args.target, 1.0,
                               advance=robot.advance, send=False)
            for step in steps[:: max(1, len(steps) // 6)]:
                print("  arm targets (rad)", [round(t, 3) for t in step.command.arm_targets],
                      "gripper", "open" if step.command.gripper_open else "closed")
            robot._port.closePort()
            return
        print(
            f"Limits: joint targets move at most {args.max_speed} rad/s and sit at most {args.max_lead} rad from "
            "the joint. They make the arm gentler than the policy was trained to be, and less likely to succeed."
        )
        if input("Arm clear of people and objects, hand on the power switch? Type yes: ").strip().lower() != "yes":
            raise SystemExit("stopped; nothing was changed.")
        rest = robot.read().positions
        log_path = Path(args.log or REPO / time.strftime("run-%Y%m%d-%H%M%S.local.json"))
        chain, log = Chain(), []
        record_of_run = {"ball": None, "turn_rad": 0.0, "ball_source": args.camera or "placed"}
        ball = stream = None

        def record(step) -> None:
            log.append({
                "angles": [round(a, 4) for a in step.joint_angles],
                "targets": [round(t, 4) for t in step.command.arm_targets],
                "gripper_open": step.command.gripper_open,
                "tip": [round(v, 4) for v in chain.tip_position(step.joint_angles[:5])],
                "ball": [round(v, 4) for v in step.ball_position],
                "action": [round(a, 3) for a in step.action],
                **({"ball_state": ball.state} if isinstance(ball, CameraBall) else {}),
                **(stream.thoughts[-1] if stream is not None and stream.thoughts else {}),
            })

        robot.configure()
        try:
            start = goals_from_command(runtime.start_command(), calibration)
            robot.move_slowly(start, args.move_seconds)
            frame, io = calibration, robot
            if args.camera:
                if live:
                    def picture_now():
                        hold_still(robot, 0.4)  # let the picture catch up with the arm, which has just stopped
                        return live.newest()

                    def look_now():
                        return [blob.scaled(wrist.width / live.width) for blob in vision.find_blobs(picture_now(), colour)]
                else:
                    eye = watcher(args, robot, "run")

                    def look_now():
                        return vision.find_blobs(eye.picture(), colour)

                # Looks further round than --sweep_deg are left out; straight ahead is always looked from.
                sweep = [angle for angle in seek.SCAN if abs(math.degrees(angle)) <= args.sweep_deg + 0.5]
                try:
                    if trained_eye is not None or trained_driver is not None:
                        from understudy.eye import finder

                        # A trained eye has to be sure twice: from where it first sees the ball and again facing it.
                        find = finder(trained_driver, picture_now) if trained_driver is not None else finder(trained_eye, eye.picture)
                        facing = seek.face_ball(robot, calibration, None, None, find=find, sweep=sweep, confirm=True)
                    else:
                        facing = seek.face_ball(robot, calibration, wrist, look_now, sweep=sweep)
                except (ValueError, CameraError) as error:
                    raise SystemExit(f"stopped before the pickup: {error}") from None
                if facing is None:
                    raise SystemExit("stopped before the pickup: the camera did not see the ball anywhere it looked.")
                print(f"ball seen {facing.distance_m * 100:.1f} cm from the swivel axis, "
                      f"{math.degrees(facing.turn_rad):+.0f} degrees round; the arm has turned to face it.", flush=True)
                if not facing.within_reach() and not args.any_distance:
                    raise SystemExit(
                        f"stopped before the pickup: the policy picks up between {REACH_BAND_M[0] * 100:g} and "
                        f"{REACH_BAND_M[1] * 100:g} cm from the swivel axis. Move the ball, or pass --any_distance to try anyway."
                    )
                frame, io = facing.calibration, SwivelLimit(robot, calibration)
                record_of_run.update(ball=list(facing.ball), turn_rad=facing.turn_rad)
                if args.find_only:
                    raise SystemExit("found the ball; no pickup was attempted because --find_only was given.")
                if place is not None:
                    seen_at = in_turned_frame((*facing.ball, 0.0), -facing.turn_rad)[:2]  # back in the arm's own frame
                    blocked = carry.in_the_way(place, seen_at, facing.turn_rad, carry.REAL_ARM_SWING_RIGHT_RAD)
                    if blocked:
                        raise SystemExit(f"stopped before the pickup: {blocked}. Move the bowl or the ball.")
                if trained_driver is not None:
                    from understudy.eye import StreamDriver

                    # The student stands in for both the ball's position, which it is never given, and the policy.
                    record_of_run["driven_by"] = Path(args.driver).name
                    ball = stream = StreamDriver(trained_driver, live.picture, facing.ball, policy.interface)
                elif live:
                    ball = CameraBall(wrist, lambda: live.look(colour), facing.ball, latency_steps=args.camera_latency_steps)
                else:
                    ball = PlacedBall(*facing.ball)
            else:
                ball = PlacedBall(*args.ball)
                record_of_run["ball"] = list(args.ball)
            from understudy.eye import NotSure

            try:
                steps = run_pickup(
                    io, stream or policy, frame, ball, args.target, args.seconds, advance=robot.advance,
                    max_speed=args.max_speed, max_lead=args.max_lead, on_step=record,
                )
            except NotSure as error:
                raise SystemExit(f"stopped during the pickup: {error}. The arm is being lowered.") from None
            last = steps[-1]
            print("finished. gripper", "open" if last.command.gripper_open else "closed",
                  f"finger angle {last.joint_angles[5]:.2f} rad (above {PlacedBall.HELD_FINGER_RAD} with the gripper closed means it is holding something)")
            if isinstance(ball, CameraBall):
                print(f"the camera placed the ball on {ball.steps_seen} steps and kept its last position on {ball.steps_kept}")
            carried = None
            if place is not None:
                # The carry sends a goal and reads the servos every control step, which keeps their watchdog fed.
                into = (f", into a bowl {place.opening_m * 100:.1f} cm across with its rim {place.rim_m * 100:.1f} cm up"
                        if place.opening_m > 0 else "")
                print(f"carrying the ball to {math.degrees(place.bearing_rad):+.0f} degrees round, "
                      f"{place.distance_m * 100:.0f} cm from the swivel axis{into}", flush=True)
                carried = carry.carry(robot, calibration, place, robot.advance)
                print(carried.said)
                record_of_run["carry"] = {"bearing_deg": math.degrees(place.bearing_rad), "distance_cm": place.distance_m * 100,
                                          "bowl_opening_cm": place.opening_m * 100, "bowl_rim_cm": place.rim_m * 100,
                                          "done": carried.done, "said": carried.said,
                                          "released_at": None if carried.released_at is None else list(carried.released_at)}
            if carried is None or (not carried.done and carried.said.startswith("not carried")):
                # Nothing waits on the keyboard while the arm is powered: the servos'
                # watchdog stops them if the bus goes quiet for half a second.
                # So the hold keeps reading, and sends no new goal: the gripper keeps squeezing.
                hold_still(robot, args.hold_seconds)
                release = list(robot.read().positions)
                release[5] = calibration.finger_to_ticks(pickup.GRIPPER_OPEN_RAD)
                robot.move_slowly(release, 1.0)
        finally:
            log_path.write_text(json.dumps({**record_of_run, "target": list(args.target),
                                            "max_speed": args.max_speed, "max_lead": args.max_lead,
                                            "control_hz": arm.CONTROL_HZ, "steps": log}) + "\n")
            print(f"{len(log)} steps recorded in {log_path}")
            if stream is not None and stream.pictures:
                import numpy as np

                seen_path = log_path.with_name(log_path.name.replace(".json", "") + ".pictures.npz")
                np.savez_compressed(seen_path, pictures=np.asarray(stream.pictures, dtype=np.uint8))
                slowest = max(thought["thinking_ms"] for thought in stream.thoughts)
                print(f"the {len(stream.pictures)} pictures the student was shown are in {seen_path}; its slowest step took {slowest:.1f} ms")
            # Whatever happened, lower the arm to where it started instead of letting it drop.
            try:
                robot.move_slowly(rest, args.move_seconds)
            except Exception as error:  # the servos may not be answering; shutdown below still runs
                print(f"could not lower the arm to rest: {error}")
    finally:
        if live is not None:
            live.close()
        if not args.dry_run:
            problems = robot.shutdown()
            print("torque off, settings restored" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", default="/dev/cu.usbmodem21401", help="Serial port of the arm's USB controller.")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--touches", default=str(DEFAULT_TOUCHES))
    parser.add_argument("--camera_file", default=str(DEFAULT_CAMERA), help="The calibrated wrist camera.")
    parser.add_argument("--sightings", default=str(DEFAULT_SIGHTINGS))
    parser.add_argument("--watch_folder", default=str(Path.home() / "understudy-camera"),
                        help="Folder of the camera watcher script (scripts/camera_watch.sh).")
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
    start.add_argument("--hold_seconds", type=float, default=0.0, help="Stay in the start pose this long before returning.")
    start.set_defaults(run=cmd_start_pose)
    look = commands.add_parser("look", help="Take one picture with the wrist camera and say where the ball is. Reads only.")
    look.add_argument("--save", help="Save the picture, with what was found drawn on it, to this file.")
    look.set_defaults(run=cmd_look)
    collect = commands.add_parser(
        "camera-collect", help="Powered, slow: look at the ball from many poses, to calibrate the wrist camera.")
    collect.add_argument("--places", type=int, default=2, help="How many spots the ball is looked at in.")
    collect.add_argument("--wait_pictures", type=int, default=30, help="Pictures to wait for the ball to appear, about 2 s each.")
    collect.add_argument("--move_seconds", type=float, default=4.0)
    collect.set_defaults(run=cmd_camera_collect)
    camera_fit = commands.add_parser("camera-fit", help="Work out the wrist camera from the collected sightings.")
    camera_fit.add_argument("--max_error_mm", type=float, default=MAX_CAMERA_ERROR_MM)
    camera_fit.add_argument("--known", type=float, nargs=3, action="append", metavar=("PLACE", "X", "Y"),
                            help="Where the ball was at one of the places (1 for the first), metres in the arm's frame: "
                            "rest the closed gripper tip on the spot and run 'where' before putting the ball there. "
                            "May be given for each place. Ties the camera to where the gripper is at the desk.")
    camera_fit.set_defaults(run=cmd_camera_fit)
    run = commands.add_parser("run", help="Run the pickup policy.")
    run.add_argument("--policy", required=True)
    run.add_argument("--ball", type=float, nargs=2, metavar=("X", "Y"), help="Where the ball is, if no camera is used.")
    run.add_argument("--camera", choices=["look", "live"],
                     help="Find the ball with the wrist camera and turn to face it. look: one picture through the "
                     "watcher, then the ball is assumed to stay put. live: follow the ball during the pickup; the "
                     "command must be started from Terminal on the Mac the camera is plugged into.")
    run.add_argument("--camera_latency_steps", type=int, default=3,
                     help="For --camera live: how many control steps old a picture is taken to be.")
    run.add_argument("--ffmpeg", default=str(Path.home() / "understudy-camera" / "ffmpeg"), help="For --camera live.")
    run.add_argument("--any_distance", action="store_true", help="Try the pickup even if the ball is outside the distances the policy handles.")
    run.add_argument("--bowl", type=float, nargs=5, metavar=("MIDDLE_X", "MIDDLE_Y", "RIM_X", "RIM_Y", "RIM_Z"),
                     help="After the pickup, carry the ball to a bowl and drop it in. Say where the bowl is with two readings "
                     "of the `where` command, in metres as it prints them: x and y with the gripper's tip over the middle of "
                     "the bowl, then x, y and z with the tip resting on its rim. Slow, with the gripper raised; everything "
                     "the README lists has to be clear. Off unless given.")
    run.add_argument("--set_down", type=float, nargs=2, metavar=("BEARING_DEG", "DISTANCE_CM"),
                     help="After the pickup, carry the ball to this place on the bare desk and put it down: degrees round "
                     "from straight ahead (positive to the arm's left, up to 90 either way) and centimetres from the swivel "
                     "axis. Off unless given.")
    run.add_argument("--driver", help="With --camera live: a trained student (an .onnx file from scripts/export_eye.py --driver, with "
                     "its .json beside it) that moves the arm itself from the camera's pictures, thirty times a second, in place "
                     "of the policy that is told where the ball is. The same speed and lead limits apply. It finds the ball "
                     "first, as --eye does, and stops if it loses sight of it.")
    run.add_argument("--sweep_deg", type=float, default=57.0,
                     help="With --camera: how far round to either side the raised arm swivels to look for the ball. The looks are "
                     "at 0, 29 and 57 degrees; 29 leaves the two widest out, 0 looks straight ahead only.")
    run.add_argument("--eye", help="With --camera look: a trained eye (an .onnx file from scripts/export_eye.py, with its .json "
                     "beside it) that says where the ball is in each still picture, in place of the hand-written finder. The "
                     "pickup itself is unchanged. If the eye is not sure there is a ball, the arm does not go for one.")
    run.add_argument("--find_only", action="store_true",
                     help="With --camera: find the ball, turn to face it, say where it is, and lower the arm without a pickup.")
    run.add_argument("--target", type=float, nargs=3, default=[0.19, 0.0, 0.13], metavar=("X", "Y", "Z"))
    run.add_argument("--seconds", type=float, default=10.0, help="How long the policy runs.")
    run.add_argument("--max_speed", type=float, default=0.6, help="Cap on how fast each joint's target moves, rad/s.")
    run.add_argument("--max_lead", type=float, default=0.1, help="Cap on how far a target sits from its joint, rad.")
    run.add_argument("--hold_seconds", type=float, default=3.0, help="Hold after the policy finishes, before letting go.")
    run.add_argument("--log", help="Where to record every step (default: run-<time>.local.json in the repository).")
    run.add_argument("--move_seconds", type=float, default=4.0)
    run.add_argument("--dry-run", action="store_true", help="Compute commands without powering the arm.")
    run.set_defaults(run=cmd_run)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
