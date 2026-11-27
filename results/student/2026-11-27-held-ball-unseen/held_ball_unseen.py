"""The exported student driving itself, as in student_driving.py, but with the ball's colour taken out of every
picture from a number of control steps after the gripper closes: what the real camera did once the ball filled
its view. Asks whether the lift and the hold need the picture.

    held_ball_unseen.py DRIVER SEED BALLS OUTPUT AFTER_STEPS     (AFTER_STEPS -1: pictures left alone)
"""
import dataclasses, json, math, random, sys
import numpy as np
sys.path.insert(0, ".")
from understudy import dagger, eye, pickup, runtime
from understudy.camera_collect import move_seconds
from understudy.camera_model import WristCamera
from understudy.runner import AIM_LEFT_M, SWIVEL_AXIS_X_M, in_turned_frame, run_pickup, turn_to_face, turned
from understudy.seek import LOOK_POSE

driver = eye.Driver(sys.argv[1])
interface = runtime.OnnxPolicy("results/pickup/2026-11-24-claw/policy.onnx").interface
nominal = WristCamera(**{k: tuple(v) if isinstance(v, list) else v for k, v in driver.notes["student_training"]["nominal_camera"].items()})
sim, scene = dagger.bench(nominal); scene.held_out = True
cal = sim.calibration; grip = [cal.finger_to_ticks(pickup.GRIPPER_OPEN_RAD)]
rng = random.Random(int(sys.argv[2])); after = int(sys.argv[5]); rows = []
state = {"closed": 0}


class Watched(eye.StreamDriver):
    def position(self, joint_angles, finger_angle, gripper_open):
        state["closed"] = 0 if gripper_open else state["closed"] + 1
        return super().position(joint_angles, finger_angle, gripper_open)


def picture():
    shown = scene.picture()[0]
    if shown is not None and after >= 0 and state["closed"] > after:
        shown = np.repeat(shown.mean(axis=2, keepdims=True), 3, axis=2).astype(np.uint8)   # no colour: the filter finds nothing
    return shown


for _ in range(int(sys.argv[3])):
    d = rng.uniform(0.14, 0.25); turn0 = rng.uniform(-0.5, 0.5)
    b = turn0 + math.asin(AIM_LEFT_M / d); ball = (SWIVEL_AXIS_X_M + d * math.cos(b), d * math.sin(b))
    sim.reset(ball); sim.after_step = None; scene.new_look(rng, ball)
    scene.look = dataclasses.replace(scene.look, latency_steps=0); scene._late = []
    sim.move_slowly(cal.to_ticks(LOOK_POSE) + grip, move_seconds(pickup.START_POSE, LOOK_POSE))
    first, sure1 = driver.glance(scene.picture_now()[0], LOOK_POSE)
    row = {"distance": d}
    if sure1 < 0.5: rows.append(row); continue
    try: turn = turn_to_face(first[:2])
    except ValueError: rows.append(row); continue
    sim.move_slowly(cal.to_ticks((turn, *LOOK_POSE[1:])) + grip, move_seconds([0.0], [turn]))
    again, sure2 = driver.glance(scene.picture_now()[0], LOOK_POSE)
    if sure2 < 0.5: rows.append(row); continue
    there = in_turned_frame((again[0], again[1], 0.0), -turn)
    turn2 = turn_to_face(there[:2]); guess = in_turned_frame(there, turn2)
    frame = turned(cal, turn2)
    sim.move_slowly(frame.to_ticks(pickup.START_POSE) + grip, move_seconds((turn, *LOOK_POSE[1:]), (turn2, *pickup.START_POSE[1:])))
    scene._late = []; state["closed"] = 0
    stream = Watched(driver, picture, guess[:2], interface)
    heights = []; closed = []
    def note(s):
        heights.append(float(s.ball()[2])); closed.append(state["closed"])
    sim.after_step = note
    row["stopped"] = None
    try:
        run_pickup(sim, stream, frame, stream, [0.19, 0.0, 0.13], 15.0, advance=sim.advance, max_speed=0.3, max_lead=0.05)
        row["lifted"] = float(np.mean(heights[-15:])) > pickup.LIFTED_HEIGHT_M
    except eye.NotSure as error:
        row["lifted"], row["stopped"] = False, str(error)[:40]
    except RuntimeError:
        row["lifted"], row["stopped"] = False, "the runner's safety check"
    sim.after_step = None
    row["highest"] = max(heights) if heights else None
    row["final"] = float(np.mean(heights[-15:])) if heights else None
    row["was_up_then_lost"] = bool(heights and max(heights) > pickup.LIFTED_HEIGHT_M and not row["lifted"])
    sure = [t["sure"] for t in stream.thoughts]
    n = min(len(sure), len(closed))
    held = [sure[i] for i in range(n) if closed[i] > 100]
    row["sure_while_held"] = float(np.mean(held)) if held else None
    row["reopened"] = bool(any(closed[i] == 0 and closed[i - 1] > 0 for i in range(1, len(closed))))
    rows.append(row)
scene.close()
json.dump({"driver": sys.argv[1], "seed": int(sys.argv[2]), "colour_removed_after_closed_steps": after, "balls": rows}, open(sys.argv[4], "w"))
done = [r for r in rows if "lifted" in r]; band = [r for r in done if 0.15 <= r["distance"] <= 0.24]
held = [r["sure_while_held"] for r in band if r["sure_while_held"] is not None]
print(f"{sys.argv[1]} colour removed {after} steps after closing: {len(rows)} balls, {len(done)} attempted; in the band lifted "
      f"{sum(r['lifted'] for r in band)} of {len(band)} ({np.mean([r['lifted'] for r in band]):.1%}); up then lost {sum(r['was_up_then_lost'] for r in band)}; "
      f"reopened {sum(r['reopened'] for r in band)}; stopped {sum(bool(r['stopped']) for r in band)}; sure while held, mean {np.mean(held):.2f}")
