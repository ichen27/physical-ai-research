"""Measure how late the live camera's pictures are, with the arm limp. Reads only: nothing is written to the servos.

A policy driven by pictures sees each one some time after it was true. This measures that
time for the whole chain: the camera, the program reading it, the pipe, and our own read.
It opens the camera's stream exactly as ``real_arm.py run --camera live`` does and reads
the arm's joints at the control rate, stamping both with one clock.

    python scripts/stream_lateness.py --port /dev/cu.usbmodemXXXX

Start it from Terminal on the Mac the camera is plugged into (macOS refuses the camera to
programs started over a remote login). Then, when it says so: hold the limp arm so the
camera looks at the desk, with the ball or anything textured in view, and swing the swivel
gently left and right, a few centimetres each way, until it says stop. The swivel's
readings say when the arm turned and the pictures say when the view moved; the time by
which the view trails the arm is the lateness.

It refuses to run if any servo's torque is on.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from understudy import arm, lateness  # noqa: E402

SMALL = (160, 90)  # pictures are compared at this size: enough to see the view move, quick enough to do every step


def speak(words: str) -> None:
    """Say a cue aloud without waiting for it to finish, and print it."""
    print(words, flush=True)
    try:
        subprocess.Popen(["say", words], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass  # no voice on this computer: the printed line is the cue


def record(robot, live, seconds: float, clock=time.monotonic, sleep=time.sleep) -> dict:
    """Read the swivel and watch the stream for ``seconds``. Returns the two records, each with its own times."""
    import cv2

    read_times, swivel, frame_times, view = [], [], [], []
    last_number, previous, position = live.arrival()[0], None, 0.0
    step = 1.0 / arm.CONTROL_HZ
    began = clock()
    tick = began
    while clock() - began < seconds:
        before = clock()
        reading = robot.read()
        read_times.append((before + clock()) / 2.0)
        swivel.append(reading.positions[0] * arm.RAD_PER_TICK)
        number, arrived = live.arrival()
        if number != last_number:
            last_number = number
            grey = cv2.cvtColor(cv2.resize(live.newest(), SMALL, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            if previous is not None:
                position += lateness.sideways_shift(previous, grey)
            previous = grey
            frame_times.append(arrived)
            view.append(position)
        tick += step
        sleep(max(0.0, tick - clock()))
    return {"read_times": read_times, "swivel_rad": swivel, "frame_times": frame_times, "view_px": view}


def measure(taken: dict) -> dict:
    """The lateness from a record, or why it cannot be told."""
    reads, frames = np.asarray(taken["read_times"]), np.asarray(taken["frame_times"])
    swung = float(np.degrees(np.ptp(taken["swivel_rad"]))) if len(reads) else 0.0
    out = {"joint_readings": len(reads), "pictures": len(frames), "swivel_swung_through_deg": round(swung, 1)}
    if len(frames) > 1:
        gaps = np.diff(frames)
        out["pictures_a_second"] = round(float(1.0 / np.median(gaps)), 1)
        out["longest_wait_for_a_picture_ms"] = round(1000 * float(gaps.max()))
    if swung < 3.0:
        return {**out, "told": False, "why": "the swivel moved less than 3 degrees: swing it left and right while this runs"}
    if len(frames) < 60:
        return {**out, "told": False, "why": "too few pictures arrived"}
    cause, effect = lateness.rate(reads, taken["swivel_rad"]), lateness.rate(frames, taken["view_px"])
    try:
        whole, alike = lateness.lag(*cause, *effect)
    except ValueError as error:
        return {**out, "told": False, "why": str(error)}
    stretches = [(by, how) for _, by, how in lateness.lag_by_stretch(*cause, *effect) if how >= 0.6]
    out.update(alike=round(alike, 2), lateness_ms=round(1000 * whole),
               by_stretch_ms=[round(1000 * by) for by, _ in stretches])
    if alike < 0.6 or not stretches:
        return {**out, "told": False, "why": "the view did not follow the swivel clearly enough: is the camera looking at something with texture, and was only the swivel moved?"}
    out.update(told=True, median_stretch_ms=round(float(np.median(out["by_stretch_ms"]))), worst_stretch_ms=max(out["by_stretch_ms"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", required=True, help="The arm's serial port.")
    parser.add_argument("--ffmpeg", default=str(Path.home() / "understudy-camera" / "ffmpeg"), help="As for real_arm.py run --camera live.")
    parser.add_argument("--seconds", type=float, default=12.0, help="How long to swing the swivel for.")
    parser.add_argument("--low_delay", action="store_true",
                        help="Ask ffmpeg not to hold pictures back (no input buffering, no probing), to see what that is worth. "
                        "run --camera live does not do this yet.")
    parser.add_argument("--output", help="Write the measurement and the record to this JSON file.")
    args = parser.parse_args()

    from understudy.camera_feed import CameraError, LiveCamera
    from understudy.dynamixel import DynamixelArm

    try:
        robot = DynamixelArm(args.port)
    except (OSError, RuntimeError) as error:
        raise SystemExit(f"cannot open the arm on {args.port}: {error}") from None
    live = None
    try:
        powered = [row.dxl_id for row in robot.status() if row.torque_on]
        if powered:
            raise SystemExit(f"torque is on for servos {powered}; this needs the arm limp. Nothing was changed.")
        try:
            # The same stream, with the same settings, as run --camera live, unless --low_delay is given.
            live = LiveCamera(ffmpeg=args.ffmpeg, input_options=LiveCamera.LOW_DELAY if args.low_delay else ())
            live.wait_for_first()
        except CameraError as error:
            raise SystemExit(str(error)) from None
        speak("Hold the arm so the camera looks at the desk. Swing the swivel gently left and right until I say stop.")
        time.sleep(4.0)
        taken = record(robot, live, args.seconds)
        speak("Stop. Thank you.")
    finally:
        if live is not None:
            live.close()
        robot._port.closePort()
    result = measure(taken)
    if result["told"]:
        print(f"The pictures are {result['lateness_ms']} ms late over the whole recording; by stretches of 2.5 s, "
              f"{result['median_stretch_ms']} ms at the median and {result['worst_stretch_ms']} ms at the worst "
              f"({result['by_stretch_ms']}).")
        print("That is when a picture reaches this program. A policy shown the newest picture at each control step "
              f"sees it up to {round(1000 / arm.CONTROL_HZ)} ms later still.")
    else:
        print(f"The lateness could not be told: {result['why']}")
    print(json.dumps({key: value for key, value in result.items() if key != "why"}))
    if args.output:
        Path(args.output).write_text(json.dumps({**result, "record": taken}) + "\n")
        print(f"record written to {args.output}")


if __name__ == "__main__":
    main()
