"""Calibrate the wrist camera in MuJoCo, with the same procedure the real arm uses.

The simulated arm carries a camera whose mount the procedure is not told. A
ball is put down, the arm notices it in its rendered pictures, learns its
colour, looks at it from poses it chooses, and the camera is fitted from what
it saw (``understudy.camera_collect`` and ``understudy.camera_calibration``).
The fitted camera is then checked against the true one, and against balls in
places it never saw.

    pip install -r requirements-dev.txt
    python scripts/mujoco_camera_calibration.py --output calibration.json --picture arrival.png

Needs OpenGL to render. ``--zero_error_ticks`` makes the arm's own calibration
wrong by up to that many encoder ticks on each joint (11 ticks is one degree),
as a real arm's is, to see how much of that ends up in the camera's answers.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from understudy import arm, camera_calibration, camera_collect, vision  # noqa: E402
from understudy.kinematics import Chain  # noqa: E402
from understudy.mujoco_arm import MujocoArm, RenderedCamera, build_model, sim_wrist_camera  # noqa: E402
from understudy.runner import pick_ball  # noqa: E402

PLACES = ((0.21, 0.01), (0.19, 0.06), (0.24, -0.05))
OFF_THE_DESK = (0.6, 0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--places", type=int, default=2, choices=(1, 2, 3), help="How many spots the ball is looked at in.")
    parser.add_argument("--zero_error_ticks", type=float, default=0.0,
                        help="The arm's joint zero points are taken to be out by up to this many ticks each.")
    parser.add_argument("--checks", type=int, default=100, help="Balls in new places to check the fitted camera on.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", help="Write the results to this JSON file.")
    parser.add_argument("--picture", help="Save the picture in which the ball was first seen, with the ball marked.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    true_camera = sim_wrist_camera()
    true_calibration = arm.Calibration.stock()
    zeros = tuple(z + rng.uniform(-args.zero_error_ticks, args.zero_error_ticks) for z in true_calibration.zero_ticks)
    believed = replace(true_calibration, zero_ticks=zeros)  # what the procedure is given
    sim = MujocoArm(build_model(wrist_camera=true_camera), true_calibration)
    view = RenderedCamera(sim, true_camera)
    chain = Chain()

    desk = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "desk")
    touches = [0]

    def count_touches(arm_sim: MujocoArm) -> None:
        for index in range(arm_sim.data.ncon):
            contact = arm_sim.data.contact[index]
            if desk in (contact.geom1, contact.geom2):
                other = contact.geom2 if contact.geom1 == desk else contact.geom1
                if arm_sim.model.geom_bodyid[other] != arm_sim._ball_body:
                    touches[0] += 1

    sim.reset(OFF_THE_DESK)
    sim.after_step = count_touches
    before = view.picture()
    sightings, rough, collector, colour = [], None, None, None
    for number, place in enumerate(PLACES[: args.places]):
        sim.place_ball(place)
        arrival = camera_collect.wait_for_ball(view.picture, before, say=lambda _: None)
        if collector is None:
            colour = arrival.colour
            collector = camera_collect.Collector(sim, believed, view.picture, colour, chain, say=lambda _: None)
            if args.picture:
                import cv2

                cv2.imwrite(args.picture, vision.draw(arrival.picture, [arrival.blob], arrival.blob, "ball seen arriving"))
        sightings += collector.collect_place(number, true_camera.width, true_camera.height, known=rough,
                                             rng=random.Random(args.seed + number))
        rough = camera_calibration.fit(sightings, believed, true_camera.width, true_camera.height, chain, lens=False).camera
        before = view.picture()
    sim.after_step = None
    fit = camera_calibration.fit(sightings, believed, true_camera.width, true_camera.height, chain)

    # Balls in places the fit never saw, looked at once from the start pose, as before a pickup.
    errors, missed = [], 0
    for _ in range(args.checks):
        ball = (rng.uniform(0.15, 0.27), rng.uniform(-0.09, 0.09))
        sim.reset(ball)
        point = pick_ball(fit.camera, vision.find_blobs(view.picture(), colour),
                          believed.to_rad(sim.read().positions[:5]), chain)
        if point is None:
            missed += 1
        else:
            errors.append(math.dist(point[:2], ball))
    view.close()

    result = {
        "simulator": f"MuJoCo {mujoco.__version__}, vendor model, rendered wrist camera",
        "seed": args.seed,
        "places": args.places,
        "sightings": len(sightings),
        "arm_zero_error_ticks": args.zero_error_ticks,
        "arm_touched_the_desk_steps": touches[0],
        "fit": {
            "pixel_rms": fit.pixel_rms,
            "desk_rms_mm": fit.desk_rms_m * 1000,
            "desk_worst_mm": max(fit.desk_errors_m) * 1000,
            "fitted_lens": fit.fitted_lens,
            "ambiguous": fit.ambiguous(),
            "places_moved": camera_collect.moved_places(fit, sightings),
        },
        "camera": {
            "focal_px": {"fitted": fit.camera.focal_px, "true": true_camera.focal_px},
            "centre_px": {"fitted": list(fit.camera.centre_px), "true": list(true_camera.centre_px)},
            "distortion": {"fitted": fit.camera.distortion, "true": true_camera.distortion},
            "mount_xyz_m": {"fitted": list(fit.camera.mount_xyz), "true": list(true_camera.mount_xyz)},
            "mount_position_error_mm": 1000 * math.dist(fit.camera.mount_xyz, true_camera.mount_xyz),
        },
        "balls_in_new_places_from_the_start_pose": {
            "tried": args.checks,
            "not_seen": missed,
            "median_error_mm": 1000 * float(np.median(errors)) if errors else None,
            "worst_error_mm": 1000 * max(errors) if errors else None,
        },
    }
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
