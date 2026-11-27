"""The exported student driving itself, as run --camera live --driver wires it, on the same balls and looks as
eye-by-distance.py, with every picture a fixed number of control steps late."""
import dataclasses, json, math, random, sys
import numpy as np
sys.path.insert(0, ".")
from understudy import dagger, eye, pickup, runtime
from understudy.camera_collect import move_seconds
from understudy.camera_model import WristCamera
from understudy.runner import AIM_LEFT_M, SWIVEL_AXIS_X_M, in_turned_frame, run_pickup, turn_to_face, turned
from understudy.seek import LOOK_POSE
driver = eye.Driver("results/student/2026-11-26-mask-and-grey/driver.onnx")
interface = runtime.OnnxPolicy("results/pickup/2026-11-24-claw/policy.onnx").interface
nominal = WristCamera(**{k: tuple(v) if isinstance(v, list) else v for k, v in driver.notes["student_training"]["nominal_camera"].items()})
sim, scene = dagger.bench(nominal); scene.held_out = True
cal = sim.calibration; grip = [cal.finger_to_ticks(pickup.GRIPPER_OPEN_RAD)]
rng = random.Random(int(sys.argv[1])); late = int(sys.argv[4]); rows = []
for _ in range(int(sys.argv[2])):
    d = rng.uniform(0.14, 0.25); turn0 = rng.uniform(-0.5, 0.5)
    b = turn0 + math.asin(AIM_LEFT_M / d); ball = (SWIVEL_AXIS_X_M + d * math.cos(b), d * math.sin(b))
    sim.reset(ball); sim.after_step = None; scene.new_look(rng, ball)
    scene.look = dataclasses.replace(scene.look, latency_steps=late); scene._late = []
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
    scene._late = []
    stream = eye.StreamDriver(driver, lambda: scene.picture()[0], guess[:2], interface)
    heights = []; sim.after_step = lambda s: heights.append(float(s.ball()[2]))
    row["stopped"] = None
    try:
        run_pickup(sim, stream, frame, stream, [0.19, 0.0, 0.13], 15.0, advance=sim.advance, max_speed=0.3, max_lead=0.05)
        row["lifted"] = float(np.mean(heights[-15:])) > pickup.LIFTED_HEIGHT_M
    except eye.NotSure as error:
        row["lifted"], row["stopped"] = False, str(error)[:40]
    except RuntimeError:
        row["lifted"], row["stopped"] = False, "the runner's safety check"
    row["would_have_lifted_when_stopped"] = None
    sim.after_step = None
    rows.append(row)
scene.close()
json.dump(rows, open(sys.argv[3], "w"))
done = [r for r in rows if "lifted" in r]; band = [r for r in done if 0.15 <= r["distance"] <= 0.24]
print(f"pictures {late} steps ({late * 1000 // 30} ms) late: {len(rows)} balls, {len(done)} attempted; lifted {np.mean([r['lifted'] for r in done]):.1%} of all attempted, {sum(r['lifted'] for r in band)} of {len(band)} ({np.mean([r['lifted'] for r in band]):.1%}) in the band; stopped by the not-sure rule {sum(bool(r['stopped']) for r in band)}")
for lo, hi in ((0.15, 0.18), (0.18, 0.21), (0.21, 0.24)):
    s = [r for r in done if lo <= r["distance"] < hi]
    print(f"  {lo*100:.0f}-{hi*100:.0f} cm: {sum(r['lifted'] for r in s)} of {len(s)} ({np.mean([r['lifted'] for r in s]):.0%}); stopped {sum(bool(r['stopped']) for r in s)}")
