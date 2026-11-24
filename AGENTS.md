# Understudy: instructions for agents

Repository: https://github.com/ichen27/understudy (public, MIT). Canonical checkout: Mac mini SSD, `code/understudy`.

## What this is

Sim-to-real reinforcement learning for the ROBOTIS OMX follower arm. Stage 1 is reaching a target point. Stage 2 is picking up a foam golf ball. Both work in simulation only; neither has run on the real arm.

## Architecture

- `understudy/` is pure Python with no third-party imports. It is the single source of truth for joint order, limits, servo assumptions, unit conversions and kinematics. Both the simulator task and the real-arm runner import it. Do not duplicate these values elsewhere.
- `understudy_isaac/` holds everything that imports Isaac Lab. Its `__init__.py` must not import Isaac Lab, because task registration happens before the simulator starts.
- Task configuration files (`*_env_cfg.py`, `*_cfg.py`) are also loaded before the simulator starts. They may import Isaac Lab configuration classes but not modules that pull in the simulator, such as action or asset implementations. Name an implementation class by string in its configuration, as `servo_action_cfg.py` does. Getting this wrong crashes the simulator at startup with `free(): invalid pointer`.
- `assets/open_manipulator_description/` is vendored from ROBOTIS under Apache-2.0. Do not edit those files. Record the upstream commit in `THIRD_PARTY_NOTICES.md` when updating them.
- Observations the policy receives in simulation must be obtainable on the real arm. Do not add simulator-only state to the policy observation.

## Where things run

- Library and tests: anywhere with Python 3.11+.
- Training: a Linux NVIDIA GPU host with Isaac Lab. The author uses a rented Brev instance. It bills by the hour, so stop it when a session ends and say so in the handoff. Agree a time budget with the author before starting it.
- Before training a contact task, run its `scripts/check_*_scene.py` script. A scripted grasp that fails means the scene is wrong, and training would be wasted.
- Measure every trained policy with the `scripts/evaluate*.py` scripts, including the servo-stiffness stress tests, and record the numbers under `results/`. The training log's own success rate is more lenient and is not the reported figure.
- Real arm: the arm's USB controller is attached to the author's laptop, not the development machine.

## Hardware safety

Moving the arm is a physical action. Read servo state before commanding motion, keep moves slow and bounded, never send policy output without a calibration file and limit clamping, and end every session with torque off and verified.

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
