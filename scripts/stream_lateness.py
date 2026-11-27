"""Measure how late the live camera's pictures are, with the arm limp. Reads only: nothing is written to the servos.

A policy driven by pictures sees each one some time after it was true. This measures that
time for the whole chain: the camera, the program reading it, the pipe, and our own read.
It opens the camera's stream exactly as ``real_arm.py run --camera live`` does and reads
the arm's joints at the control rate, stamping both with one clock.

    python scripts/stream_lateness.py --port /dev/cu.usbmodemXXXX

Start it from Terminal on the Mac the camera is plugged into (macOS refuses the camera to
programs started over a remote login). Hold the limp arm so the camera looks at the desk,
with the ball or anything textured in view, and do what it says, one word at a time: turn
the swivel a few centimetres left, or right, or hold it still. The words come at uneven
intervals on purpose. The swivel's readings say when the arm turned and the pictures say
when the view moved; the time by which the view trails the arm is the lateness.

A steady swing back and forth would not do: it matches itself half a swing later, upside
down, as well as it does at the true lateness. If the recording cannot tell the two apart,
no number is given.

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


def cues(seconds: float, seed: int = 1) -> list[tuple[float, str]]:
    """What to say and when: left, right and hold, at uneven intervals, so that the motion never settles into a rhythm."""
    import random

    rng = random.Random(seed)
    out, at, side = [], 0.0, "left"
    while at < seconds - 0.5:
        word = "hold" if out and out[-1][1] != "hold" and rng.random() < 0.3 else side
        out.append((round(at, 2), word))
        if word != "hold":
            side = "right" if side == "left" else "left"
        at += rng.uniform(0.6, 2.0)
    return out


def record(robot, live, seconds: float, clock=time.monotonic, sleep=time.sleep, say=None, script=()) -> dict:
    """Read the swivel and watch the stream for ``seconds``, saying each cue of ``script`` when its time comes.
    Returns the two records, each with its own times."""
    import cv2

    read_times, swivel, frame_times, across, down = [], [], [], [], []
    last_number, previous, position = live.arrival()[0], None, [0.0, 0.0]
    script = list(script)
    step = 1.0 / arm.CONTROL_HZ
    began = clock()
    tick = began
    while clock() - began < seconds:
        if script and say is not None and clock() - began >= script[0][0]:
            say(script.pop(0)[1])
        before = clock()
        reading = robot.read()
        read_times.append((before + clock()) / 2.0)
        swivel.append(reading.positions[0] * arm.RAD_PER_TICK)
        number, arrived = live.arrival()
        if number != last_number:
            last_number = number
            grey = cv2.cvtColor(cv2.resize(live.newest(), SMALL, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            if previous is not None:
                moved = lateness.view_shift(previous, grey)
                position = [position[0] + moved[0], position[1] + moved[1]]
            previous = grey
            frame_times.append(arrived)
            across.append(position[0])
            down.append(position[1])
        tick += step
        sleep(max(0.0, tick - clock()))
    return {"read_times": read_times, "swivel_rad": swivel, "frame_times": frame_times, "view_across_px": across, "view_down_px": down}


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
    view = lateness.main_direction(taken["view_across_px"], taken["view_down_px"])
    cause, effect = lateness.rate(reads, taken["swivel_rad"]), lateness.rate(frames, view)
    try:
        whole = lateness.lag(*cause, *effect)
    except ValueError as error:
        return {**out, "told": False, "why": str(error)}
    out.update(alike=round(whole.alike, 2), next_best_alike=round(whole.rival, 2))
    if whole.alike < 0.6:
        return {**out, "told": False, "why": "the view did not follow the swivel clearly enough: is the camera looking at something with texture, and was only the swivel moved?"}
    out["falls_off_by_at_50_ms"] = round(whole.sharp, 3)
    if whole.alike - whole.rival >= 0.15 and not whole.clear():
        return {**out, "told": False, "why": "the motion was too slow and smooth: 50 ms either way lines up nearly as well, so the "
                "lateness cannot be pinned down. Make each turn quick, and stop between turns, as the spoken words ask"}
    if not whole.clear():
        return {**out, "told": False, "why": "the motion was too regular: it lines up nearly as well another way "
                f"({whole.rival:.2f} against {whole.alike:.2f}), so the lateness cannot be told from half a swing later. "
                "Follow the spoken words, with their pauses, and do not settle into a rhythm"}
    stretches = [found for _, found in lateness.lag_by_stretch(*cause, *effect, sign=whole.sign) if found.clear()]
    out.update(lateness_ms=round(1000 * whole.seconds), by_stretch_ms=[round(1000 * found.seconds) for found in stretches])
    if not stretches:
        return {**out, "told": True, "median_stretch_ms": None, "worst_stretch_ms": None}
    out.update(told=True, median_stretch_ms=round(float(np.median(out["by_stretch_ms"]))), worst_stretch_ms=max(out["by_stretch_ms"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", required=True, help="The arm's serial port.")
    parser.add_argument("--ffmpeg", default=str(Path.home() / "understudy-camera" / "ffmpeg"), help="As for real_arm.py run --camera live.")
    parser.add_argument("--seconds", type=float, default=20.0, help="How long the recording lasts.")
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
        speak("Hold the arm so the camera looks at the desk. Turn the swivel a little, as I say: left, right, or hold.")
        time.sleep(6.0)
        taken = record(robot, live, args.seconds, say=speak, script=cues(args.seconds))
        speak("Stop. Thank you.")
    finally:
        if live is not None:
            live.close()
        robot._port.closePort()
    result = measure(taken)
    if result["told"]:
        print(f"The pictures are {result['lateness_ms']} ms late over the whole recording"
              + (f"; by stretches of 4 s, {result['median_stretch_ms']} ms at the median and {result['worst_stretch_ms']} ms at the "
                 f"worst ({result['by_stretch_ms']})." if result["by_stretch_ms"] else "; no single stretch of 4 s was clear enough by itself."))
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
