"""Measure the carry in MuJoCo: pick the ball up, carry it to a bowl at the side, drop it in.

The pickup is the trained policy, told where the ball is, with the real arm's speed limits
and the arm turned to face the ball. The carry is ``understudy.carry.carry``, the fixed move
the real arm would make. The bowl (``understudy.sim_bowl``) is a real thing in the scene:
the ball and the gripper can hit it, and it moves if struck. An attempt is scored by where
the ball ends: inside the bowl once the gripper has opened, and still there a second later.

    pip install -r requirements-dev.txt
    python scripts/mujoco_carry.py --policy results/pickup/2026-11-24-claw/policy.onnx --attempts 200

The bowl's size, how the ball bounces in it, how far the bowl really is from where the carry
was told, and how high the ball is let go can all be changed, to find out how much each
matters. ``--bare_desk`` puts the ball down on the desk instead.

Needs no GPU and no rendering, except with --video.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from understudy import arm, pickup, runtime, sim_bowl  # noqa: E402
from understudy.camera_collect import MAX_JOINT_SPEED, move_seconds  # noqa: E402
from understudy.carry import (  # noqa: E402
    NEAREST_EDGE_M, RELEASE_ABOVE_M, RELEASE_SECONDS, SetDown, carry, grasp_point, in_the_way, set_down_poses,
)
from understudy.kinematics import Chain  # noqa: E402
from understudy.mujoco_arm import MujocoArm, build_model  # noqa: E402
from understudy.runner import (  # noqa: E402
    AIM_LEFT_M, MAX_TURN_RAD, REACH_BAND_M, SWIVEL_AXIS_X_M, PlacedBall, in_turned_frame, run_pickup, turn_to_face, turned,
)

LOW_M = 0.11  # the pickup's sideways swing is measured while the gripper's lowest point is under this
NEAR_M = 0.03  # on the bare desk, a ball this close to the place counts as there
BOWL_MOVED_M = 0.005  # a bowl that ends this far from where it stood was pushed


def polar(distance: float, bearing: float) -> tuple[float, float]:
    return (SWIVEL_AXIS_X_M + distance * math.cos(bearing), distance * math.sin(bearing))


class Film:
    """Pictures of the scene from outside, every other control step, for --video."""

    def __init__(self, sim: MujocoArm):
        self._renderer = mujoco.Renderer(sim.model, height=540, width=720)
        self._camera = mujoco.MjvCamera()
        self._camera.lookat[:] = (0.05, 0.0, 0.07)
        self._camera.distance, self._camera.azimuth, self._camera.elevation = 0.85, 150.0, -28.0
        self.frames: list[np.ndarray] = []
        self._step = 0

    def take(self, sim: MujocoArm) -> None:
        self._step += 1
        if self._step % 2 == 0:
            self._renderer.update_scene(sim.data, camera=self._camera)
            self.frames.append(self._renderer.render()[:, :, ::-1].copy())

    def save(self, path: Path) -> None:
        import cv2

        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), arm.CONTROL_HZ / 2, (720, 540))
        for frame in self.frames:
            writer.write(frame)
        writer.release()
        chosen = [self.frames[int(i)] for i in np.linspace(0, len(self.frames) - 1, 8)]
        strip = np.vstack([np.hstack(chosen[:4]), np.hstack(chosen[4:])])
        cv2.imwrite(str(path.with_name(path.stem + "-frames.png")), strip)
        self._renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", required=True)
    parser.add_argument("--attempts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bowl_opening_cm", type=float, default=15.0, help="Across the inside of the bowl's rim.")
    parser.add_argument("--bowl_rim_cm", type=float, default=6.5, help="The height of the bowl's rim above the desk.")
    parser.add_argument("--bowl_mass_g", type=float, default=400.0)
    parser.add_argument("--bounce", type=float, default=1.0,
                        help="How the ball lands in the bowl: 1 does not bounce, 0.3 comes back up about a fifth of the way "
                        "it fell, 0.15 about a half.")
    parser.add_argument("--place_error_cm", type=float, default=0.0,
                        help="The bowl really stands this far, in a direction picked at random, from where the carry is told it is.")
    parser.add_argument("--rim_error_cm", type=float, default=0.0,
                        help="The bowl's rim is really this much higher than the carry is told it is.")
    parser.add_argument("--release_above_cm", type=float, default=RELEASE_ABOVE_M * 100,
                        help="How far above the rim (or the bare desk) the bottom of the ball is when the gripper opens. "
                        "Below zero the ball is let go inside the bowl.")
    parser.add_argument("--bare_desk", action="store_true", help="No bowl: put the ball down on the desk.")
    parser.add_argument("--rolling_resistance", type=float, default=0.0,
                        help="With --bare_desk: the ball's rolling resistance, as a share of its weight. The simulated ball "
                        "has none, so that a pickup that knocks it is found out; a foam ball on carpet is nearer 0.05 to 0.1.")
    parser.add_argument("--carry_speed", type=float, default=MAX_JOINT_SPEED, help="Fastest any joint turns during the carry, rad/s.")
    parser.add_argument("--bearing_deg", type=float, nargs=2, default=[30.0, 90.0], metavar=("LEAST", "MOST"),
                        help="Places are this far round from straight ahead, to either side.")
    parser.add_argument("--side", choices=["left", "right", "either"], default="either", help="Which side of the arm the places are on.")
    parser.add_argument("--distance_cm", type=float, nargs=2, metavar=("NEAR", "FAR"),
                        help="Places are this far from the swivel axis (default: as near as a bowl may stand, to 26 cm).")
    parser.add_argument("--max_speed", type=float, default=0.3, help="The pickup's speed limit, rad/s.")
    parser.add_argument("--max_lead", type=float, default=0.05, help="The pickup's lead limit, rad.")
    parser.add_argument("--seconds", type=float, default=15.0, help="Length of the pickup.")
    parser.add_argument("--video", help="Film the first attempt in which the ball is lifted, to this .mp4 file (needs OpenGL).")
    parser.add_argument("--output", help="Write the results to this JSON file.")
    args = parser.parse_args()

    told = None if args.bare_desk else (args.bowl_opening_cm / 100, (args.bowl_rim_cm - args.rim_error_cm) / 100)
    bowl = None if args.bare_desk else sim_bowl.Bowl(
        opening_m=args.bowl_opening_cm / 100, rim_m=args.bowl_rim_cm / 100, mass_kg=args.bowl_mass_g / 1000, contact_damping=args.bounce)
    distances = args.distance_cm or [16.0 if bowl is None else (NEAREST_EDGE_M + bowl.opening_m / 2) * 100, 26.0]
    release_above = args.release_above_cm / 100

    policy = runtime.OnnxPolicy(args.policy)
    model = build_model(extras=None if bowl is None else (lambda spec: sim_bowl.add_bowl(spec, bowl)))
    sim = MujocoArm(model, command_delay_s=0.015, profile_time_s=arm.VENDOR_PROFILE_TIME_S)
    if args.rolling_resistance > 0:
        ball_geom = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        sim.model.geom_condim[ball_geom] = 6  # MuJoCo only resists rolling in contacts of this kind
        sim.model.geom_friction[ball_geom][2] = args.rolling_resistance * pickup.BALL_RADIUS_M
    ball_speed = int(sim.model.body_dofadr[sim._ball_body])
    gripper = sim._ctrl[5]
    rng = random.Random(args.seed)
    rows, refused_pairs, unusable_places = [], 0, 0
    film = Film(sim) if args.video else None
    chain = Chain()

    while len(rows) < args.attempts:
        ball = polar(rng.uniform(*REACH_BAND_M), rng.uniform(-MAX_TURN_RAD, MAX_TURN_RAD) + math.asin(AIM_LEFT_M / 0.2))
        side = {"left": 1.0, "right": -1.0}.get(args.side) or rng.choice((-1.0, 1.0))
        place = SetDown(side * math.radians(rng.uniform(*args.bearing_deg)), rng.uniform(*distances) / 100,
                        *((told[1], told[0]) if told else ()))
        try:
            set_down_poses(place, release_above_m=release_above)
        except ValueError:
            unusable_places += 1  # the carry would refuse this place before anything moved
            continue
        if in_the_way(place, ball):
            refused_pairs += 1  # the bowl is where the gripper must go for this ball: refused before anything moves
            continue
        try:
            turn = turn_to_face(ball)
        except ValueError:
            turn = math.copysign(MAX_TURN_RAD, ball[1])
        target = [rng.uniform(*pickup.HOLD_TARGET[axis]) for axis in "xyz"]
        off = rng.uniform(-math.pi, math.pi)
        spot = (place.xy[0] + args.place_error_cm / 100 * math.cos(off), place.xy[1] + args.place_error_cm / 100 * math.sin(off))

        sim.reset(ball)
        if bowl is not None:
            sim_bowl.place_bowl(sim, spot)
        track = {"phase": "pickup", "since": 0, "at_release": None, "leaving": None, "top_speed": 0.0,
                 "arm_in_pickup": False, "arm_before": False, "ball_before": False, "arm_after": False,
                 "in_when_open": None, "in_a_second_later": None, "swing_right": 0.0, "swing_left": 0.0, "out": 0.0}

        def watch(arm_sim: MujocoArm) -> None:
            if film is not None and film.frames is not None:
                film.take(arm_sim)
            arm_touches, ball_touches = sim_bowl.touching(arm_sim) if bowl is not None else (False, False)
            if track["phase"] == "pickup":
                track["arm_in_pickup"] |= arm_touches
                angles = arm_sim.joint_angles()[:5]
                tip, held = chain.tip_position(angles), grasp_point(angles, chain)
                if min(tip[2], held[2] - pickup.BALL_RADIUS_M) < LOW_M:
                    track["swing_right"] = max(track["swing_right"], turn - angles[0])
                    track["swing_left"] = max(track["swing_left"], angles[0] - turn)
                    track["out"] = max(track["out"], math.hypot(tip[0] - SWIVEL_AXIS_X_M, tip[1]), math.hypot(held[0] - SWIVEL_AXIS_X_M, held[1]))
                return
            track["top_speed"] = max(track["top_speed"], float(np.max(np.abs([arm_sim.data.qvel[a] for a in arm_sim._qvel[:5]]))))
            if track["at_release"] is None and float(arm_sim.data.ctrl[gripper]) > pickup.GRIPPER_CLOSED_RAD + 0.02:
                track["at_release"] = arm_sim.ball().copy()  # the gripper has begun to open
            if track["at_release"] is None:
                track["arm_before"] |= arm_touches
                track["ball_before"] |= ball_touches
                return
            track["arm_after"] |= arm_touches
            track["since"] += 1
            if track["since"] == int(RELEASE_SECONDS * arm.CONTROL_HZ):  # the gripper is open and has not yet moved away
                track["leaving"] = float(np.linalg.norm(arm_sim.data.qvel[ball_speed : ball_speed + 2]))
                track["in_when_open"] = bowl is not None and sim_bowl.ball_is_in(arm_sim, bowl)
            if track["since"] == int((RELEASE_SECONDS + 1.0) * arm.CONTROL_HZ):
                track["in_a_second_later"] = bowl is not None and sim_bowl.ball_is_in(arm_sim, bowl)

        def not_this_one() -> None:
            sim.after_step = None
            if film is not None and film.frames is not None:
                film.frames.clear()  # film the first attempt that gets as far as the carry

        sim.after_step = watch
        frame = turned(sim.calibration, turn)
        sim.move_slowly(frame.to_ticks(pickup.START_POSE) + [frame.finger_to_ticks(pickup.GRIPPER_OPEN_RAD)], move_seconds([0.0], [turn]))
        row = {"place_bearing_deg": round(math.degrees(place.bearing_rad), 1), "place_distance_cm": round(place.distance_m * 100, 1),
               "ball_turn_deg": round(math.degrees(turn), 1), "lifted": False, "carried": False, "said": None}
        try:
            run_pickup(sim, policy, frame, PlacedBall(*in_turned_frame((*ball, 0.0), turn)[:2]), target, args.seconds,
                       advance=sim.advance, max_speed=args.max_speed, max_lead=args.max_lead)
        except RuntimeError:
            rows.append({**row, "said": "the pickup was stopped by the runner's safety check", "arm_touched_bowl_in_pickup": track["arm_in_pickup"]})
            not_this_one()
            continue
        row["lifted"] = bool(sim.ball()[2] > pickup.LIFTED_HEIGHT_M)
        row["arm_touched_bowl_in_pickup"] = track["arm_in_pickup"]
        row["swung_right_deg"], row["swung_left_deg"] = round(math.degrees(track["swing_right"]), 1), round(math.degrees(track["swing_left"]), 1)
        row["reached_beyond_ball_cm"] = round(100 * (track["out"] - math.hypot(ball[0] - SWIVEL_AXIS_X_M, ball[1])), 1)
        if bowl is not None:
            row["bowl_pushed_in_pickup_mm"] = round(1000 * math.dist(sim_bowl.bowl_xy(sim), spot), 1)
        if not row["lifted"]:
            rows.append(row)
            not_this_one()
            continue

        track["phase"] = "carry"
        stood = sim_bowl.bowl_xy(sim) if bowl is not None else spot
        result = carry(sim, sim.calibration, place, sim.advance, max_speed=args.carry_speed,
                       release_above_m=release_above, say=lambda _: None)
        for _ in range(int(2.0 * arm.CONTROL_HZ)):  # let the ball come to rest
            sim.advance()
        sim.after_step = None
        if film is not None and film.frames is not None:
            film.save(Path(args.video))
            film.frames = None
        rest = sim.ball()
        let_go = track["at_release"]
        row.update(
            carried=result.done, said=result.said, fastest_joint_rad_s=round(track["top_speed"], 3),
            let_go_mm_from_the_middle=None if let_go is None else round(1000 * math.dist(let_go[:2], stood), 1),
            rest_height_mm=round(1000 * float(rest[2]), 1),
        )
        if bowl is not None:
            row.update(
                in_bowl_when_the_gripper_had_opened=track["in_when_open"], in_bowl_a_second_later=track["in_a_second_later"],
                in_bowl_at_the_end=sim_bowl.ball_is_in(sim, bowl),
                arm_touched_bowl_before_letting_go=track["arm_before"], ball_touched_bowl_before_letting_go=track["ball_before"],
                arm_touched_bowl_after_letting_go=track["arm_after"],
                bowl_pushed_mm=round(1000 * math.dist(sim_bowl.bowl_xy(sim), stood), 1),
            )
        else:
            row.update(
                rested_mm_from_place=round(1000 * math.dist(rest[:2], spot), 1),
                rolled_mm=None if let_go is None else round(1000 * math.dist(rest[:2], let_go[:2]), 1),
                speed_once_open_cm_s=None if track["leaving"] is None else round(100 * track["leaving"], 2),
            )
        rows.append(row)

    lifted = [row for row in rows if row["lifted"]]

    def share(some: list, of: list) -> float | None:
        return round(len(some) / len(of), 4) if of else None

    def summary(some: list) -> dict:
        finished = [row for row in some if row["carried"]]
        out = {
            "balls_lifted": len(some),
            "carried_and_let_go": share(finished, some),
            "lost_on_the_way": share([r for r in some if r["said"] and r["said"].startswith("stopped")], some),
            "median_mm_from_the_middle_when_let_go": float(np.median([r["let_go_mm_from_the_middle"] for r in finished])) if finished else None,
            "worst_mm_from_the_middle_when_let_go": max((r["let_go_mm_from_the_middle"] for r in finished), default=None),
        }
        if bowl is not None:
            out.update({
                "in_the_bowl_once_the_gripper_had_opened": share([r for r in finished if r["in_bowl_when_the_gripper_had_opened"]], some),
                "in_the_bowl_then_and_a_second_later": share(
                    [r for r in finished if r["in_bowl_when_the_gripper_had_opened"] and r["in_bowl_a_second_later"]], some),
                "in_the_bowl_at_the_end": share([r for r in finished if r["in_bowl_at_the_end"]], some),
                "arm_touched_the_bowl": share([r for r in some if r.get("arm_touched_bowl_before_letting_go") or r.get("arm_touched_bowl_after_letting_go")], some),
                "held_ball_touched_the_bowl": share([r for r in some if r.get("ball_touched_bowl_before_letting_go")], some),
                "bowl_pushed_5_mm_or_more": share([r for r in some if r.get("bowl_pushed_mm", 0.0) >= BOWL_MOVED_M * 1000], some),
                "furthest_the_bowl_was_pushed_mm": max((r.get("bowl_pushed_mm", 0.0) for r in some), default=None),
            })
        else:
            out.update({
                "at_rest_within_3_cm_of_the_place": share([r for r in finished if r["rested_mm_from_place"] <= NEAR_M * 1000], some),
                "median_mm_from_place_at_rest": float(np.median([r["rested_mm_from_place"] for r in finished])) if finished else None,
                "worst_mm_from_place_at_rest": max((r["rested_mm_from_place"] for r in finished), default=None),
                "median_mm_rolled_after_release": float(np.median([r["rolled_mm"] for r in finished])) if finished else None,
            })
        return out

    edges = [(-90, -60), (-60, -30), (30, 60), (60, 90)]
    result = {
        "simulator": f"MuJoCo {mujoco.__version__}, vendor model",
        "policy": str(args.policy),
        "attempts": len(rows),
        "seed": args.seed,
        "pickup": {"ball_position": "given", "max_speed_rad_s": args.max_speed, "max_lead_rad": args.max_lead, "seconds": args.seconds,
                   "lifted": share(lifted, rows),
                   "arm_touched_the_bowl": share([r for r in rows if r.get("arm_touched_bowl_in_pickup")], rows),
                   "furthest_the_bowl_was_pushed_mm": max((r.get("bowl_pushed_in_pickup_mm", 0.0) for r in rows), default=None),
                   "with_the_gripper_under_11_cm": {
                       "furthest_swung_right_of_facing_deg": max((r.get("swung_right_deg", 0.0) for r in rows), default=None),
                       "furthest_swung_left_of_facing_deg": max((r.get("swung_left_deg", 0.0) for r in rows), default=None),
                       "furthest_reached_beyond_the_ball_cm": max((r.get("reached_beyond_ball_cm", 0.0) for r in rows), default=None)}},
        "bowl": None if bowl is None else {
            "opening_cm": args.bowl_opening_cm, "rim_cm": args.bowl_rim_cm, "flat_middle_cm": round(bowl.flat_radius_m * 200, 1),
            "mass_g": args.bowl_mass_g, "free_to_move": True, "bounce_setting": args.bounce,
            "ball_rebounds_this_share_of_a_5_cm_drop": round(sim_bowl.rebound(bowl), 2),
            "really_stands_cm_from_where_told": args.place_error_cm, "rim_really_higher_than_told_cm": args.rim_error_cm},
        "ball_bottom_above_the_rim_when_let_go_cm": args.release_above_cm,
        "carry_speed_rad_s": args.carry_speed,
        "fastest_joint_seen_rad_s": max((row.get("fastest_joint_rad_s") or 0.0 for row in lifted), default=None),
        "ball_rolling_resistance": args.rolling_resistance,
        "places": f"{args.bearing_deg[0]:g} to {args.bearing_deg[1]:g} degrees round, {args.side} side; "
                  f"{distances[0]:g} to {distances[1]:g} cm from the swivel axis",
        "ball_and_place_pairs_refused_because_the_bowl_was_in_the_grippers_way": refused_pairs,
        "places_refused_as_unusable": unusable_places,
        "of_balls_lifted": summary(lifted),
        "by_side_and_how_far_round": {f"{low:+d} to {high:+d} deg": summary([r for r in lifted if low <= r["place_bearing_deg"] < high or r["place_bearing_deg"] == high == 90])
                                      for low, high in edges},
        "what_was_said": {said: sum(row["said"] == said for row in rows) for said in sorted({str(row["said"]) for row in rows})},
        "attempt_records": rows,
    }
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "attempt_records"}, indent=1))


if __name__ == "__main__":
    main()
