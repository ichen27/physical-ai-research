"""Does a large patch of the ball's colour elsewhere in the picture move where an eye puts a ball it does see?

Looks from the look pose with the arm facing the ball, as the second look of a search is. Each picture is
shown to each eye as it is, and again with a patch painted into its top-left corner, cut off by the edge:
a bag or a wrapper lying at the edge of the view. Run from the repository's root.

    python results/student/2026-11-26-mask-and-grey/wirings/second_orange_patch.py 300 out.json
"""
import json, math, random, sys
import cv2, mujoco, numpy as np
sys.path.insert(0, ".")
from understudy import dagger, eye, pickup
from understudy.camera_model import WristCamera
from understudy.runner import AIM_LEFT_M, SWIVEL_AXIS_X_M
from understudy.seek import LOOK_POSE
EYES = {"colour, ball as the real camera shows it": "results/student/2026-11-26-real-ball-colour/eye.onnx",
        "mask beside grey": "results/student/2026-11-26-mask-and-grey/eye.onnx", "mask alone": "results/student/2026-11-26-mask-alone/eye.onnx"}
eyes = {name: eye.Eye(path) for name, path in EYES.items()}
first = next(iter(eyes.values()))
nominal = WristCamera(**{k: tuple(v) if isinstance(v, list) else v for k, v in first.notes["student_training"]["nominal_camera"].items()})
sim, scene = dagger.bench(nominal); scene.held_out = True
rng = random.Random(4)
PATCHES = {"the real camera's ball colour": (131, 186, 255), "a deep orange": (30, 110, 240)}  # blue, green, red
rows = []
def radial(p): return math.hypot(p[0] - SWIVEL_AXIS_X_M, p[1])
for _ in range(int(sys.argv[1])):
    d = rng.uniform(0.15, 0.24)
    ball = (SWIVEL_AXIS_X_M + d * math.cos(math.asin(AIM_LEFT_M / d)), AIM_LEFT_M)  # as it sits once the arm has turned to face it
    sim.reset(ball); scene.clutter = False; scene.new_look(rng, ball)
    for address, angle in zip(sim._qpos, [*LOOK_POSE, pickup.GRIPPER_OPEN_RAD]): sim.data.qpos[address] = angle
    sim.data.qpos[sim._second_finger] = -pickup.GRIPPER_OPEN_RAD; mujoco.mj_forward(sim.model, sim.data)
    picture, visible = scene.picture_now()
    if visible < 0.95: continue
    row = {"distance": d}
    for name, e in eyes.items():
        (x, y, _), sure = e.glance(picture, LOOK_POSE)
        row[name] = {"plain": [radial((x, y)) - d, sure]}
        for patch, colour in PATCHES.items():
            painted = picture.copy()
            cv2.circle(painted, (8, 12), 8, colour, -1, cv2.LINE_AA)   # 80 px across at the real picture's size, cut off by the corner
            (x2, y2, _), sure2 = e.glance(painted, LOOK_POSE)
            row[name][patch] = [radial((x2, y2)) - d, sure2, math.dist((x, y), (x2, y2))]
    rows.append(row)
scene.close()
json.dump(rows, open(sys.argv[2], "w"))
print(len(rows), "looks, the ball whole and near the middle, no other things on the desk")
for name in eyes:
    plain = np.array([r[name]["plain"][0] for r in rows]) * 1000
    print(f"{name}: as it is, distance against the truth median {np.median(plain):+.1f} mm")
    for patch in PATCHES:
        with_it = np.array([r[name][patch][0] for r in rows]) * 1000
        moved = np.array([r[name][patch][2] for r in rows]) * 1000
        shift = with_it - plain
        near = [s for s, r in zip(shift, rows) if r["distance"] < 0.18]
        print(f"   with a patch of {patch} in the corner: the answer moves {np.median(moved):.1f} mm at the median, {np.percentile(moved, 90):.1f} mm at the 90th percentile; "
              f"its distance changes by {np.median(shift):+.1f} mm at the median ({np.percentile(shift, 10):+.1f} to {np.percentile(shift, 90):+.1f}); "
              f"balls nearer than 18 cm: {np.median(near):+.1f} mm; shorter by 5 mm or more in {np.mean(shift <= -5):.0%}; still sure in {np.mean([r[name][patch][1] >= 0.5 for r in rows]):.0%}")
