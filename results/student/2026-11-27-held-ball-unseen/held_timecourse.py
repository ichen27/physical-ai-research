"""Baseline (pictures left alone): how sure the simulated student is, and how much of its picture the filter marks,
step by step after the gripper closes.

    held_timecourse.py DRIVER BALLS OUTPUT
"""
import dataclasses, json, math, random, sys
import numpy as np
sys.path.insert(0, ".")
from understudy import dagger, eye, pickup, runtime, vision
from understudy.camera_collect import move_seconds
from understudy.camera_model import WristCamera
from understudy.runner import AIM_LEFT_M, SWIVEL_AXIS_X_M, in_turned_frame, run_pickup, turn_to_face, turned
from understudy.seek import LOOK_POSE
driver = eye.Driver(sys.argv[1])
interface = runtime.OnnxPolicy("results/pickup/2026-11-24-claw/policy.onnx").interface
nominal = WristCamera(**{k: tuple(v) if isinstance(v, list) else v for k, v in driver.notes["student_training"]["nominal_camera"].items()})
sim, scene = dagger.bench(nominal); scene.held_out = True
cal = sim.calibration; grip = [cal.finger_to_ticks(pickup.GRIPPER_OPEN_RAD)]
rng = random.Random(8); out = []
class Watched(eye.StreamDriver):
    def position(self, ja, fa, go):
        opens.append(bool(go)); return super().position(ja, fa, go)
for _ in range(int(sys.argv[2])):
    d = rng.uniform(0.14, 0.25); turn0 = rng.uniform(-0.5, 0.5)
    b = turn0 + math.asin(AIM_LEFT_M / d); ball = (SWIVEL_AXIS_X_M + d * math.cos(b), d * math.sin(b))
    sim.reset(ball); sim.after_step = None; scene.new_look(rng, ball)
    scene.look = dataclasses.replace(scene.look, latency_steps=0); scene._late = []
    sim.move_slowly(cal.to_ticks(LOOK_POSE) + grip, move_seconds(pickup.START_POSE, LOOK_POSE))
    first, sure1 = driver.glance(scene.picture_now()[0], LOOK_POSE)
    if sure1 < 0.5: continue
    try: turn = turn_to_face(first[:2])
    except ValueError: continue
    sim.move_slowly(cal.to_ticks((turn, *LOOK_POSE[1:])) + grip, move_seconds([0.0], [turn]))
    again, sure2 = driver.glance(scene.picture_now()[0], LOOK_POSE)
    if sure2 < 0.5: continue
    there = in_turned_frame((again[0], again[1], 0.0), -turn)
    turn2 = turn_to_face(there[:2]); guess = in_turned_frame(there, turn2); frame = turned(cal, turn2)
    sim.move_slowly(frame.to_ticks(pickup.START_POSE) + grip, move_seconds((turn, *LOOK_POSE[1:]), (turn2, *pickup.START_POSE[1:])))
    scene._late = []; opens = []
    stream = Watched(driver, lambda: scene.picture()[0], guess[:2], interface)
    heights = []; sim.after_step = lambda s: heights.append(float(s.ball()[2]))
    try: run_pickup(sim, stream, frame, stream, [0.19, 0.0, 0.13], 15.0, advance=sim.advance, max_speed=0.3, max_lead=0.05)
    except Exception as e: sim.after_step = None; continue
    sim.after_step = None
    if not (0.15 <= d <= 0.24) or np.mean(heights[-15:]) < pickup.LIFTED_HEIGHT_M: continue
    n = min(len(stream.thoughts), len(opens), len(heights))
    close = next(i for i in range(n) if not opens[i])
    out.append({"distance": d, "closed_at_step": close,
                "sure": [round(stream.thoughts[i]["sure"], 3) for i in range(n)],
                "mask": [round(float(vision.orange_mask(stream.pictures[i], eye.FILTER).mean()), 3) for i in range(n)],
                "ball_height": [round(h, 3) for h in heights[:n]]})
scene.close()
json.dump({"driver": sys.argv[1], "seed": 8, "balls_drawn": int(sys.argv[2]), "lifted_in_the_band": out}, open(sys.argv[3], "w"), separators=(",", ":"))
print(len(out), "lifted balls")
print("steps after closing | sure, median (least..most) | filter's share, median | ball's height, median cm")
for k in (-15, 0, 15, 30, 45, 60, 75, 90, 120, 150, 200, 300):
    rows = [(r["sure"][r["closed_at_step"]+k], r["mask"][r["closed_at_step"]+k], r["ball_height"][r["closed_at_step"]+k]) for r in out if 0 <= r["closed_at_step"]+k < len(r["sure"])]
    if rows:
        s, m, h = map(np.array, zip(*rows))
        print(f"{k:5d} ({k/30:4.1f} s) | {np.median(s):.2f} ({s.min():.2f}..{s.max():.2f}) | {np.median(m):.2f} | {np.median(h)*100:5.1f}   n={len(rows)}")
