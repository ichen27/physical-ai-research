# Understudy: instructions for agents

Repository: https://github.com/ichen27/understudy (public, MIT). Canonical checkout: Mac mini SSD, `code/understudy`.

## What this is

Sim-to-real reinforcement learning for the ROBOTIS OMX follower arm. Stage 1 is reaching a target point. Stage 2 is picking up a foam golf ball. The pickup holds the ball in both the training simulator and the MuJoCo cross-check. Nothing has run on the real arm.

## Architecture

- `understudy/` is pure Python with no third-party imports. It is the single source of truth for joint order, limits, servo assumptions, unit conversions and kinematics. Both the simulator task and the real-arm runner import it. Do not duplicate these values elsewhere.
- `understudy_isaac/` holds everything that imports Isaac Lab. Its `__init__.py` must not import Isaac Lab, because task registration happens before the simulator starts.
- Task configuration files (`*_env_cfg.py`, `*_cfg.py`) are also loaded before the simulator starts. They may import Isaac Lab configuration classes but not modules that pull in the simulator, such as action or asset implementations. Name an implementation class by string in its configuration, as `servo_action_cfg.py` does. Getting this wrong crashes the simulator at startup with `free(): invalid pointer`.
- `assets/open_manipulator_description/` is vendored from ROBOTIS under Apache-2.0. Do not edit those files. Record the upstream commit in `THIRD_PARTY_NOTICES.md` when updating them.
- Collision shapes must have the real part's shape where it touches things. Physics engines collide with a mesh's convex hull; for the claw-shaped fingers that filled the hollow they grip with, and no amount of training fixed the resulting grip. `scripts/build_finger_collision.py` builds the fingers from convex pieces, used by both the Isaac Lab arm and the MuJoCo arm. Check any new gripper or object the same way before training on it.
- Observations the policy receives in simulation must be obtainable on the real arm. Do not add simulator-only state to the policy observation.

## Where things run

- Library and tests: anywhere with Python 3.11+.
- Training: a Linux NVIDIA GPU host with Isaac Lab. The author uses a rented Brev instance. It bills by the hour, so stop it when a session ends and say so in the handoff. Agree a time budget with the author before starting it.
- Copy checkpoints off the rented machine as training goes, not only at the end. On 2026-11-24 the account's credit ran out three minutes before a 100-minute run finished, the machine was stopped, and the final checkpoint was left on its disk.
- Before training a contact task, run its `scripts/check_*_scene.py` script. A scripted grasp that fails means the scene is wrong, and training would be wasted.
- After training a pickup policy, run it in MuJoCo with `scripts/mujoco_pickup.py` (no GPU). A result in the training simulator alone is not evidence that a policy will transfer; three separate problems were found only this way.
- `understudy/runtime.py` and `understudy/runner.py` are the single path from sensor readings to servo commands, used by the MuJoCo cross-check and the real arm alike. Change the policy's inputs or outputs there and in the Isaac Lab task together, re-record `tests/fixtures/pickup_rollout.json` with `scripts/record_rollout.py`, and keep `interface.json` beside every exported policy.
- Measure every trained policy with the `scripts/evaluate*.py` scripts, including the servo-stiffness stress tests, and record the numbers under `results/`. The training log's own success rate is more lenient and is not the reported figure.
- Real arm: the arm's USB controller is attached to the author's laptop, not the development machine.
- Wrist camera: pictures come only through `understudy/camera_feed.py`, which opens the arm's camera by name. Never add a fallback to another camera; a laptop has its own and may see a phone's. macOS refuses the camera to programs started over a remote login, so one-at-a-time pictures go through `scripts/camera_watch.sh`, which the author starts in Terminal.
- What happens after the pickup (`understudy/carry.py`: carry the ball to a bowl and drop it in) is a fixed move, not a learnt one, and is the same whoever picked the ball up. Its places are in the arm's own frame, never a turned one.
- Before putting anything else on the desk near the arm, measure what the pickup sweeps; do not reason from where the gripper comes down. Lifting, the trained policy swings the arm up to 42 degrees to the right of where it faces with the gripper still low. That was found only when a simulated bowl was stood there and struck. `understudy.carry.in_the_way` holds the measured figures and `scripts/mujoco_carry.py` re-measures them on every run; re-measure after training a new policy.
- A camera run turns "straight ahead" to face the ball by shifting the swivel's zero in the calibration (`understudy.runner.turned`). Everything computed from that calibration is then in the turned frame; do not mix it with positions from the unturned one.

## Hardware safety

Moving the arm is a physical action. Read servo state before commanding motion, keep moves slow and bounded, never send policy output without a calibration file and limit clamping, and end every session with torque off and verified.

While torque is on, the computer must speak to the servos at least every 500 ms, or their watchdog cuts power and a raised arm drops. This has happened once on the real arm and was caught once more before a run. Anything that takes longer than a few tens of milliseconds between reads (fitting, planning, waiting for a picture) must read the arm as it goes: see `understudy.camera_collect.Pulse`. A second thread does not do it; that was measured. `tests/sim_rig.py`'s `SimRobot` records the longest silence while powered, and tests of any powered command should assert on it.

## Validation

```sh
bash ci/run.sh
```

State plainly which of these a change was checked against: unit tests, a simulator run, the real arm. A change to `understudy_isaac/` is unverified until it has run in Isaac Lab.

## Workflow

Feature branches and pull requests into `main`. No direct pushes of implementation to `main`. Merge only when the `test` check passes on the current revision. Commit as the repository owner only, with no tool or agent attribution anywhere.

## Shipping decisions (2026-11-24)

- Launcher (Limelight Labs OS): no.
- Repository visibility: public.
- Name: brand name, Understudy.
- License: MIT.
- Announcement: neither for now. Ask again when there is a working result.
