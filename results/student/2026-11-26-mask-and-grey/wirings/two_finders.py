"""The eye feeding the teacher through the real search code (understudy.seek.face_ball, sure twice), with and without
the hand-written finder having to agree. Usage: agree.py SEED COUNT ball|noball 0|1 OUT.json"""
import json, math, random, sys
import numpy as np
sys.path.insert(0, ".")
from understudy import dagger, eye, pickup, runtime, seek, vision
from understudy.camera_collect import move_seconds
from understudy.camera_model import WristCamera
from understudy.kinematics import Chain
from understudy.runner import AIM_LEFT_M, REACH_BAND_M, SWIVEL_AXIS_X_M, PlacedBall, pick_ball, run_pickup
e = eye.Eye("results/student/2026-11-26-mask-and-grey/eye.onnx")
teacher = runtime.OnnxPolicy("results/pickup/2026-11-24-claw/policy.onnx")
nominal = WristCamera(**{k: tuple(v) if isinstance(v, list) else v for k, v in e.notes["student_training"]["nominal_camera"].items()})
sim, scene = dagger.bench(nominal); scene.held_out = True; chain = Chain()
cal = sim.calibration
mode, agree = sys.argv[3], sys.argv[4] == "1"
rng = random.Random(int(sys.argv[1])); rows = []
def picture():
    scene.picture_now()
    return scene.last_render   # 256 x 144: the eye shrinks it, the hand-written finder uses it as it is
def second(frame, joint_angles, near):
    camera = nominal.resized(frame.shape[1], frame.shape[0])
    return pick_ball(camera, vision.find_blobs(frame, eye.FILTER), joint_angles, chain, near)
for _ in range(int(sys.argv[2])):
    d = rng.uniform(0.14, 0.25); turn0 = rng.uniform(-0.5, 0.5)
    b = turn0 + math.asin(AIM_LEFT_M / d); ball = (SWIVEL_AXIS_X_M + d * math.cos(b), d * math.sin(b))
    sim.reset(ball if mode == "ball" else (3.0, 3.0)); sim.after_step = None; scene.new_look(rng, ball)
    said, looks = [], []
    find = eye.finder(e, picture, say=said.append, second=second if agree else None, looks=looks)
    row = {"distance": d, "looks": len(looks), "attempted": False, "lifted": False, "why": None}
    try:
        facing = seek.face_ball(sim, cal, None, None, say=said.append, find=find, confirm=True)
    except ValueError as error:
        facing, row["why"] = None, str(error)[:60]
    row["looks"] = len(looks)
    row["said"] = [s.strip()[:70] for s in said if "not taken" in s]
    if facing is None:
        row["why"] = row["why"] or "no ball seen"
    elif not facing.within_reach():
        row["why"] = "outside the band"
    else:
        row["attempted"] = True
        if mode == "ball":
            heights = []; sim.after_step = lambda s: heights.append(float(s.ball()[2]))
            try:
                run_pickup(sim, teacher, facing.calibration, PlacedBall(*facing.ball), [0.19, 0.0, 0.13], 15.0, advance=sim.advance, max_speed=0.3, max_lead=0.05)
                row["lifted"] = float(np.mean(heights[-15:])) > pickup.LIFTED_HEIGHT_M
            except RuntimeError:
                pass
            sim.after_step = None
    rows.append(row)
scene.close()
json.dump(rows, open(sys.argv[5], "w"))
import collections
print(f"{mode}, the two must agree: {agree}; {len(rows)} searches; pickup attempted {sum(r['attempted'] for r in rows)}; why not: {dict(collections.Counter(r['why'] for r in rows if not r['attempted']))}")
if mode == "ball":
    band = [r for r in rows if 0.15 <= r["distance"] <= 0.24]
    print(f"  in the band: {len(band)} balls; attempted {sum(r['attempted'] for r in band)}; lifted {sum(r['lifted'] for r in band)} ({np.mean([r['lifted'] for r in band]):.1%} of all, {sum(r['lifted'] for r in band) / max(1, sum(r['attempted'] for r in band)):.1%} of those attempted)")
    print("  refusals by the check:", dict(collections.Counter(("finder saw none" if "sees none" in s else "apart") for r in band for s in r["said"])))
else:
    print("  looks in which the eye was sure but the check refused:", sum(len(r["said"]) for r in rows))
