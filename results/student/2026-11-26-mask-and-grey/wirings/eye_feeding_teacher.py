"""The trained eye feeding the teacher, as run --camera look --eye wires them, by how far out the ball is."""
import json, math, random, sys
import numpy as np
sys.path.insert(0, ".")
from understudy import dagger, eye, pickup, runtime
from understudy.camera_collect import move_seconds
from understudy.camera_model import WristCamera
from understudy.runner import AIM_LEFT_M, MAX_TURN_RAD, SWIVEL_AXIS_X_M, PlacedBall, in_turned_frame, run_pickup, turn_to_face, turned
from understudy.seek import LOOK_POSE
e = eye.Eye("results/student/2026-11-26-mask-and-grey/eye.onnx")
teacher = runtime.OnnxPolicy("results/pickup/2026-11-24-claw/policy.onnx")
nominal = WristCamera(**{k: tuple(v) if isinstance(v, list) else v for k, v in e.notes["student_training"]["nominal_camera"].items()})
sim, scene = dagger.bench(nominal); scene.held_out = True
cal = sim.calibration; grip = [cal.finger_to_ticks(pickup.GRIPPER_OPEN_RAD)]
rng = random.Random(int(sys.argv[1])); rows = []
BIAS = float(sys.argv[4]) / 1000.0 if len(sys.argv) > 4 else 0.0
def radial(p): return math.hypot(p[0] - SWIVEL_AXIS_X_M, p[1])
for _ in range(int(sys.argv[2])):
    d = rng.uniform(0.14, 0.25); turn0 = rng.uniform(-0.5, 0.5)
    b = turn0 + math.asin(AIM_LEFT_M / d); ball = (SWIVEL_AXIS_X_M + d * math.cos(b), d * math.sin(b))
    sim.reset(ball); sim.after_step = None; scene.new_look(rng, ball)
    sim.move_slowly(cal.to_ticks(LOOK_POSE) + grip, move_seconds(pickup.START_POSE, LOOK_POSE))
    first, sure1 = e.glance(scene.picture_now()[0], LOOK_POSE)
    row = {"distance": d, "sure_first": sure1}
    if sure1 < 0.5: rows.append(row); continue
    try: turn = turn_to_face(first[:2])
    except ValueError: rows.append({**row, "refused": True}); continue
    sim.move_slowly(cal.to_ticks((turn, *LOOK_POSE[1:])) + grip, move_seconds([0.0], [turn]))
    again, sure2 = e.glance(scene.picture_now()[0], LOOK_POSE)   # a look from straight ahead in the turned frame
    row["sure_again"] = sure2
    if sure2 < 0.5: rows.append(row); continue
    there = in_turned_frame((again[0], again[1], 0.0), -turn)
    if BIAS:
        # tell the teacher the ball is a little further out (or nearer) than the eye says, along the line from the swivel axis
        r0 = radial(there); k = (r0 + BIAS) / r0
        there = (SWIVEL_AXIS_X_M + (there[0] - SWIVEL_AXIS_X_M) * k, there[1] * k, 0.0)
    turn2 = turn_to_face(there[:2]); guess = in_turned_frame(there, turn2)
    row.update(first_radial_error=radial(first) - d, radial_error=radial(there) - d, error=math.dist(there[:2], ball),
               sideways_error=math.atan2(there[1], there[0] - SWIVEL_AXIS_X_M) - math.atan2(ball[1], ball[0] - SWIVEL_AXIS_X_M))
    frame = turned(cal, turn2)
    sim.move_slowly(frame.to_ticks(pickup.START_POSE) + grip, move_seconds((turn, *LOOK_POSE[1:]), (turn2, *pickup.START_POSE[1:])))
    heights = []; sim.after_step = lambda s: heights.append(float(s.ball()[2]))
    try:
        run_pickup(sim, teacher, frame, PlacedBall(guess[0], guess[1]), [0.19, 0.0, 0.13], 15.0, advance=sim.advance, max_speed=0.3, max_lead=0.05)
        row["lifted"] = float(np.mean(heights[-15:])) > pickup.LIFTED_HEIGHT_M
    except RuntimeError:
        row["lifted"] = False
    sim.after_step = None
    if BIAS:
        rows.append(row); continue
    # and the teacher told the truth, same ball and look, for comparison
    sim.reset(ball); turn3 = turn_to_face(ball); frame = turned(cal, turn3)
    sim.move_slowly(frame.to_ticks(pickup.START_POSE) + grip, 1.5)
    heights = []; sim.after_step = lambda s: heights.append(float(s.ball()[2]))
    try:
        run_pickup(sim, teacher, frame, PlacedBall(*in_turned_frame((*ball, 0.0), turn3)[:2]), [0.19, 0.0, 0.13], 15.0, advance=sim.advance, max_speed=0.3, max_lead=0.05)
        row["lifted_told_the_truth"] = float(np.mean(heights[-15:])) > pickup.LIFTED_HEIGHT_M
    except RuntimeError:
        row["lifted_told_the_truth"] = False
    sim.after_step = None
    rows.append(row)
scene.close()
json.dump(rows, open(sys.argv[3], "w"))
done = [r for r in rows if "lifted" in r]
print(len(rows), "balls;", len(done), "seen twice and attempted; told", BIAS * 1000, "mm further than the eye says; lifted overall", f"{np.mean([r['lifted'] for r in done]):.1%}", "; in the band 15 to 24 cm:", f"{np.mean([r['lifted'] for r in done if 0.15 <= r['distance'] <= 0.24]):.1%}")
print("| True distance | Balls | Eye says, against the truth: median | 10th to 90th percentile | Error, median | Lifted, eye's place | Lifted, told the truth |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for lo, hi in ((0.14, 0.16), (0.16, 0.18), (0.18, 0.20), (0.20, 0.22), (0.22, 0.25)):
    s = [r for r in done if lo <= r["distance"] < hi]
    if not s: continue
    re = np.array([r["radial_error"] for r in s]) * 1000
    print(f"| {lo*100:.0f} to {hi*100:.0f} cm | {len(s)} | {np.median(re):+.1f} mm | {np.percentile(re, 10):+.1f} to {np.percentile(re, 90):+.1f} mm | {np.median([r['error'] for r in s])*1000:.1f} mm | {np.mean([r['lifted'] for r in s]):.0%} | {np.mean([r.get('lifted_told_the_truth', float('nan')) for r in s]):.0%} |")
# does a wrong distance, by itself, make the teacher miss?  lifted against the signed error
for lo, hi, name in ((-99, -8, "eye 8 mm or more short"), (-8, -3, "3 to 8 mm short"), (-3, 3, "within 3 mm"), (3, 8, "3 to 8 mm long"), (8, 99, "8 mm or more long")):
    s = [r for r in done if lo <= r["radial_error"] * 1000 < hi]
    if s: print(f"{name}: {len(s)} balls, lifted {np.mean([r['lifted'] for r in s]):.0%}")
