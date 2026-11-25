"""Run a trained pickup policy in MuJoCo, on the vendor's own model of the arm.

The policy was trained in Isaac Lab (PhysX). This runs it in a different physics
engine, on the vendor's MuJoCo model with its servo gains, joint friction and
gripper coupling, using the same input and output code the real arm will use
(``understudy.runtime``). It needs no GPU.

    pip install -r requirements-dev.txt onnxruntime
    python scripts/mujoco_pickup.py --policy results/pickup/<run>/policy.onnx --attempts 200

A policy that only works in the simulator it was trained in will not work on
hardware, so this is a necessary check. It is not a sufficient one.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from understudy import arm, pickup, runtime, seek, variation  # noqa: E402
from understudy.camera_model import rotation_from_vector, vector_from_rotation  # noqa: E402
from understudy.mujoco_arm import (  # noqa: E402
    MujocoArm,
    ProjectedCamera,
    RenderedCamera,
    SimulatedCamera,
    build_model,
    sim_wrist_camera,
)
from understudy.runner import (  # noqa: E402
    AIM_LEFT_M,
    MAX_TURN_RAD,
    REACH_BAND_M,
    SWIVEL_AXIS_X_M,
    CameraBall,
    PlacedBall,
    in_turned_frame,
    pick_ball,
    run_pickup,
)

FINAL_WINDOW_STEPS = int(0.5 * arm.CONTROL_HZ)  # control ticks, so half a second on the clock
WRIST_SOURCES = ("wrist-camera", "wrist-look")
speeds: list[float] = []  # each finished attempt's fastest arm joint speed, rad/s


def place_for(rng: random.Random, area: str, distance_m: Sequence[float]) -> tuple[float, float]:
    """Where the ball starts: in the patch the policy was trained on, or anywhere around the arm it can turn to face."""
    if area == "trained":
        return (rng.uniform(*pickup.BALL_SPAWN["x"]), rng.uniform(*pickup.BALL_SPAWN["y"]))
    distance, turn = rng.uniform(*distance_m), rng.uniform(-MAX_TURN_RAD, MAX_TURN_RAD)
    bearing = turn + math.asin(AIM_LEFT_M / distance)
    return (SWIVEL_AXIS_X_M + distance * math.cos(bearing), distance * math.sin(bearing))


def believed_camera(true_camera, rng: random.Random, mount_error_m: float, angle_error_rad: float):
    """The camera as a calibration would give it: the true one, a little out in position and direction."""
    xyz = tuple(v + rng.uniform(-mount_error_m, mount_error_m) for v in true_camera.mount_xyz)
    axis = np.array([rng.gauss(0, 1) for _ in range(3)])
    wobble = rotation_from_vector(axis / np.linalg.norm(axis) * rng.uniform(0, angle_error_rad))
    rotation = vector_from_rotation(wobble @ rotation_from_vector(true_camera.mount_rotation))
    return replace(true_camera, mount_xyz=xyz, mount_rotation=tuple(float(v) for v in rotation))


def run_attempt(sim: MujocoArm, policy, rng: random.Random, args, recorder=None, eye=None) -> dict:
    ball_xy = place_for(rng, args.area, args.distance)
    target = [rng.uniform(*pickup.HOLD_TARGET[axis]) for axis in "xyz"]
    sim.reset(ball_xy)
    sim.after_step = None
    calibration, turn, source = sim.calibration, 0.0, None
    if args.ball_source in WRIST_SOURCES:
        camera = believed_camera(eye.camera, rng, args.camera_mount_error_mm / 1000, math.radians(args.camera_angle_error_deg))
        eye.reset()
        if args.face:
            try:
                facing = seek.face_ball(sim, calibration, camera, eye.look_now, say=lambda _: None)
            except ValueError:
                facing = None
            if facing is None:
                return {"lifted": False, "held": False, "ever_lifted": False, "height": float(sim.ball()[2]),
                        "distance": float("nan"), "stopped": False, "found": False}
            calibration, turn, first = facing.calibration, facing.turn_rad, facing.ball
        else:
            first = pick_ball(camera, eye.look_now(), calibration.to_rad(sim.read().positions[:5]), eye.chain)
            if first is None:
                return {"lifted": False, "held": False, "ever_lifted": False, "height": float(sim.ball()[2]),
                        "distance": float("nan"), "stopped": False, "found": False}
        if args.ball_source == "wrist-look":
            ball = PlacedBall(first[0], first[1])
        else:
            ball = source = CameraBall(camera, eye.look, first, latency_steps=args.camera_latency[1])
    elif args.ball_source == "camera":
        ball = SimulatedCamera(sim, rng, args.camera_bias, args.camera_noise, args.camera_latency)
    else:
        ball = PlacedBall(*ball_xy)
    # The hold target is given in the frame the policy works in; the simulator's is not turned.
    world_target = np.array(in_turned_frame(target, -turn))

    heights, distances = [], []
    peak_speed = [0.0]

    def after_step(arm_sim: MujocoArm) -> None:
        peak_speed[0] = max(peak_speed[0], float(np.max(np.abs([arm_sim.data.qvel[a] for a in arm_sim._qvel[:5]]))))
        position = arm_sim.ball()
        heights.append(float(position[2]))
        distances.append(float(np.linalg.norm(position - world_target)))
        if recorder is not None:
            recorder.capture(arm_sim)

    sim.after_step = after_step
    stopped = False
    try:
        run_pickup(sim, policy, calibration, ball, target, args.seconds, advance=sim.advance,
                   max_speed=args.max_speed, max_lead=args.max_lead)
        speeds.append(peak_speed[0])
    except RuntimeError:
        # The runner's safety check stopped the arm, as it would on hardware. The attempt counts as failed.
        stopped = True
    sim.after_step = None
    ever_lifted = bool(heights) and max(heights) > pickup.LIFTED_HEIGHT_M
    outcome = {"found": True}
    if source is not None:
        open_steps = max(1, source.steps_seen + source.steps_kept)
        outcome["seen_share"] = source.steps_seen / open_steps
    if stopped or len(heights) < FINAL_WINDOW_STEPS:
        return {**outcome, "lifted": False, "held": False, "ever_lifted": ever_lifted, "height": float(sim.ball()[2]),
                "distance": float("nan"), "stopped": True}

    height = float(np.mean(heights[-FINAL_WINDOW_STEPS:]))
    distance = float(np.mean(distances[-FINAL_WINDOW_STEPS:]))
    lifted = height > pickup.LIFTED_HEIGHT_M
    return {
        **outcome,
        "lifted": lifted,
        "held": lifted and distance < pickup.HOLD_RADIUS_M,
        "height": height,
        "distance": distance,
        "ever_lifted": ever_lifted,
        "stopped": False,
    }


class Eye:
    """The simulated wrist camera as the pickup uses it: one look on request, or a stream with latency."""

    def __init__(self, sim: MujocoArm, camera, geometric: bool, latency_steps: int, rng: random.Random):
        from understudy import camera_collect, vision
        from understudy.kinematics import Chain

        self.camera, self.chain, self._sim = camera, Chain(), sim
        self._geometric, self._latency, self._rng = geometric, latency_steps, rng
        self.colour = None
        if not geometric:
            # Learn the ball's colour the way the real procedure does: a picture before and after it is put down.
            view = RenderedCamera(sim, camera)
            sim.reset((0.6, 0.5))
            empty = view.picture()
            sim.place_ball((0.21, 0.0))
            self.colour = camera_collect.wait_for_ball(view.picture, empty, say=lambda _: None).colour
            view.close()
            del vision
        self.reset()

    def reset(self) -> None:
        if self._geometric:
            self._now = ProjectedCamera(self._sim, self.camera, self._rng, pixel_noise=1.0)
            self._stream = ProjectedCamera(self._sim, self.camera, self._rng, pixel_noise=1.0, latency_steps=self._latency)
        else:
            if getattr(self, "_now", None) is not None:
                self._now.close()
                self._stream.close()
            self._now = RenderedCamera(self._sim, self.camera, self.colour)
            self._stream = RenderedCamera(self._sim, self.camera, self.colour, latency_steps=self._latency)

    def look_now(self):
        return self._now.look()

    def look(self):
        return self._stream.look()


class Recorder:
    """Writes the attempts to an MP4 through ffmpeg."""

    def __init__(self, model: mujoco.MjModel, path: Path, width: int = 960, height: int = 720):
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._camera = mujoco.MjvCamera()
        self._camera.lookat[:] = [0.14, 0.0, 0.08]
        self._camera.distance, self._camera.azimuth, self._camera.elevation = 0.75, 135.0, -25.0
        self._ffmpeg = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", str(int(arm.CONTROL_HZ)), "-i", "-", "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE,
        )

    def capture(self, sim: MujocoArm) -> None:
        self._renderer.update_scene(sim.data, camera=self._camera)
        self._ffmpeg.stdin.write(self._renderer.render().tobytes())

    def close(self) -> None:
        self._ffmpeg.stdin.close()
        self._ffmpeg.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", required=True, help="Exported policy (policy.onnx).")
    parser.add_argument("--attempts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stiffness_scale", type=float, default=1.0, help="Multiple of the vendor servo stiffness.")
    parser.add_argument("--ball_mass", type=float, default=pickup.BALL_MASS_KG)
    parser.add_argument("--ball_friction", type=float, default=pickup.BALL_FRICTION)
    parser.add_argument("--camera_bias", type=float, default=variation.BALL_POSITION_BIAS_M)
    parser.add_argument("--camera_noise", type=float, default=variation.BALL_POSITION_NOISE_M)
    parser.add_argument("--camera_latency", type=int, nargs=2, default=list(variation.CAMERA_LATENCY_STEPS))
    parser.add_argument("--command_delay", type=float, default=0.015, help="Seconds before a command takes effect.")
    parser.add_argument(
        "--profile_time", type=float, default=arm.VENDOR_PROFILE_TIME_S, help="Seconds a servo takes to reach a new goal."
    )
    parser.add_argument("--extra_delay_steps", type=int, default=0, help="Whole control steps of extra command delay.")
    parser.add_argument(
        "--ball_source",
        choices=["camera", "placed", *WRIST_SOURCES],
        default="camera",
        help="camera: the ball's true position with errors added, as in training. placed: no camera; the ball is "
        "assumed to be where it was put until the gripper holds it (understudy.runner.PlacedBall). wrist-camera: "
        "the wrist camera is rendered and the real detector follows the ball in it (understudy.runner.CameraBall). "
        "wrist-look: one look with the wrist camera before the attempt, then as placed.",
    )
    parser.add_argument("--geometric_camera", action="store_true",
                        help="For the wrist sources: no rendering; the ball appears where geometry says, fingers ignored.")
    parser.add_argument("--camera_mount_error_mm", type=float, default=0.0,
                        help="Wrist camera's position taken to be out by up to this much on each axis, new each attempt.")
    parser.add_argument("--camera_angle_error_deg", type=float, default=0.0,
                        help="Wrist camera's direction taken to be out by up to this much, new each attempt.")
    parser.add_argument("--area", choices=["trained", "around"], default="trained",
                        help="trained: balls in the patch the policy was trained on. around: anywhere the arm can turn "
                        f"to face, up to {math.degrees(MAX_TURN_RAD):.0f} degrees either way, at --distance from the swivel axis.")
    parser.add_argument("--distance", type=float, nargs=2, default=list(REACH_BAND_M), metavar=("NEAR", "FAR"),
                        help="For --area around: the ball's distance from the swivel axis, metres.")
    parser.add_argument("--face", action="store_true",
                        help="For the wrist sources: look for the ball, swivelling if need be, and turn to face it first.")
    parser.add_argument(
        "--hull_fingers",
        action="store_true",
        help="Collide the fingers as convex hulls, as the vendor model does, instead of as curved claws.",
    )
    parser.add_argument("--gripper_force_scale", type=float, default=1.0, help="Multiple of the vendor gripper force.")
    parser.add_argument("--seconds", type=float, default=pickup.EPISODE_SECONDS, help="Length of each attempt.")
    parser.add_argument("--max_speed", type=float, help="Cap on how fast each arm joint's target moves, rad/s.")
    parser.add_argument("--max_lead", type=float, help="Cap on how far a target may sit from its joint, rad.")
    parser.add_argument("--output", help="Write the results to this JSON file.")
    parser.add_argument("--video", help="Record the first attempts to this MP4 file (needs ffmpeg).")
    parser.add_argument("--video_attempts", type=int, default=3)
    args = parser.parse_args()
    policy = runtime.OnnxPolicy(args.policy)
    wrist = sim_wrist_camera() if args.ball_source in WRIST_SOURCES else None
    if args.area == "around" and not (wrist and args.face):
        parser.error("--area around needs a wrist ball source and --face: the policy cannot reach round by itself")
    sim = MujocoArm(
        build_model(args.ball_mass, args.ball_friction, args.stiffness_scale, claw_fingers=not args.hull_fingers,
                    wrist_camera=wrist),
        command_delay_s=args.command_delay,
        profile_time_s=args.profile_time,
        extra_delay_steps=args.extra_delay_steps,
    )
    gripper = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, arm.GRIPPER_JOINT_NAMES[0])
    sim.model.actuator_forcerange[gripper] *= args.gripper_force_scale
    rng = random.Random(args.seed)
    eye = Eye(sim, wrist, args.geometric_camera, args.camera_latency[1], random.Random(args.seed + 1)) if wrist else None
    recorder = Recorder(sim.model, Path(args.video)) if args.video else None

    outcomes = []
    for attempt in range(args.attempts):
        recording = recorder if recorder is not None and attempt < args.video_attempts else None
        outcomes.append(run_attempt(sim, policy, rng, args, recording, eye))
    if recorder is not None:
        recorder.close()

    lifted = [o for o in outcomes if o["lifted"]]
    result = {
        "simulator": f"MuJoCo {mujoco.__version__}, vendor model",
        "policy": str(args.policy),
        "attempts_scored": len(outcomes),
        "servo_stiffness_scale": args.stiffness_scale,
        "ball_mass_kg": args.ball_mass,
        "ball_friction": args.ball_friction,
        "gripper_force_scale": args.gripper_force_scale,
        "finger_collision": "convex hull" if args.hull_fingers else "claw pieces",
        "camera_bias_m": args.camera_bias,
        "camera_noise_m": args.camera_noise,
        "camera_latency_steps": list(args.camera_latency),
        "command_delay_s": args.command_delay,
        "servo_profile_time_s": args.profile_time,
        "extra_delay_steps": args.extra_delay_steps,
        "ball_source": args.ball_source,
        "ball_area": args.area,
        "ball_distance_from_swivel_axis_m": list(args.distance) if args.area == "around" else None,
        "faced_the_ball": args.face,
        "wrist_camera": None if wrist is None else {
            "pictures": "geometry only, fingers ignored" if args.geometric_camera else "rendered, found by understudy.vision",
            "mount_error_mm": args.camera_mount_error_mm,
            "angle_error_deg": args.camera_angle_error_deg,
            "ball_found_rate": sum(o["found"] for o in outcomes) / len(outcomes),
            "median_share_of_open_gripper_steps_ball_seen": (
                float(np.median([o["seen_share"] for o in outcomes if "seen_share" in o]))
                if any("seen_share" in o for o in outcomes) else None
            ),
        },
        "seed": args.seed,
        "seconds": args.seconds,
        "max_speed_rad_s": args.max_speed,
        "max_lead_rad": args.max_lead,
        "peak_joint_speed_rad_s": {"median": float(np.median(speeds)), "highest": float(max(speeds))} if speeds else None,
        "success_rate": sum(o["held"] for o in outcomes) / len(outcomes),
        "lifted_rate": len(lifted) / len(outcomes),
        "lifted_at_some_point_rate": sum(o["ever_lifted"] for o in outcomes) / len(outcomes),
        "safety_stop_rate": sum(o["stopped"] for o in outcomes) / len(outcomes),
        "median_final_ball_height_m": float(np.median([o["height"] for o in outcomes])),
        "median_distance_to_target_when_lifted_m": float(np.median([o["distance"] for o in lifted])) if lifted else None,
    }
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
