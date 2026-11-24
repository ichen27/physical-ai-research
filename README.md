# Understudy

Train a small robot arm in simulation, then run the trained policy on the real arm.

![The simulated arm picking up a ball](results/pickup/2026-11-24-first/pickup-frames.png)

The arm is the ROBOTIS OMX follower (five joints plus a gripper, DYNAMIXEL XL430 and XL330 servos). Policies are trained with reinforcement learning in NVIDIA Isaac Lab on a rented GPU and are meant to run on the physical arm from a laptop.

## Status

This is an early project. Read this section before relying on anything.

| Part | State |
| --- | --- |
| Arm description, unit conversions, calibration format | Working, tested |
| Forward kinematics read from the vendor URDF | Working, tested. Matches the simulator to 0.00 mm |
| Isaac Lab reach task (`understudy_isaac/`) | Working in Isaac Lab 3.0.0 |
| Trained reach policy | Trained and measured **in simulation only**. See [results](results/reach/2026-11-24-baseline/README.md) |
| Ball pickup task | Working in Isaac Lab 3.0.0 |
| Trained pickup policy | Trained and measured **in simulation only**. See [results](results/pickup/2026-11-24-first/README.md) |
| Real-arm runner and calibration capture | Not done |
| Locating the ball with a camera | Not started |

In simulation the first policy reached within 2 cm on 5,000 of 5,000 scored reaches, with a median error of 1 mm. With servos six times stiffer or 2.5 times softer than it was trained on, it still reached within 2 cm every time, but the median error rose to 7–11 mm. Nothing has been run on a real arm yet.

The pickup policy lifts a foam golf ball off the desk in 5,000 of 5,000 simulated attempts and holds it within 3 cm of a target point in 89% of them. It is not ready for hardware: with arm servos six times stiffer than it was trained on, which is a plausible value for the real arm, it lifts the ball in only 6% of attempts.

Known limitations:

- The simulated servo stiffness and damping in `understudy/arm.py` are estimates from the servo data sheets. They have not been measured on a real arm. The policy is not yet trained across a range of stiffness, so it loses accuracy when the servos differ.
- An arm set up through LeRobot has its own encoder offsets. It needs a calibration file (see [Calibration](#calibration)) before any policy output is sent to it.
- The joint limits are a conservative choice for one arm on a clear desk. Check them against your own setup.

## Layout

| Path | Purpose |
| --- | --- |
| `understudy/arm.py` | Joint order, limits, servo assumptions, tick and radian conversion, calibration |
| `understudy/kinematics.py` | Gripper-tip position from joint angles, read from the URDF |
| `understudy_isaac/` | Isaac Lab task and training settings. Needs Isaac Lab |
| `understudy/pickup.py` | The ball, the gripper commands and the success criteria for the pickup task |
| `understudy/sim_urdf.py` | Generates the corrected URDF used in simulation |
| `scripts/isaaclab_run.py` | Runs Isaac Lab's train and play scripts with this project's task registered |
| `scripts/check_sim_model.py` | Checks the simulated arm's masses, limits and kinematics against the shared description |
| `scripts/check_pickup_scene.py` | Runs a fixed, unlearned grasp to confirm the simulated gripper can hold the ball |
| `scripts/evaluate.py` | Scores a trained reach policy on thousands of reaches and exports it |
| `scripts/evaluate_pickup.py` | Scores a trained pickup policy on thousands of attempts and exports it |
| `results/` | Trained policies and their measurements |
| `assets/generated/` | The generated simulation URDF |
| `assets/open_manipulator_description/` | Vendor arm model. See [third-party notices](THIRD_PARTY_NOTICES.md) |
| `tests/` | Checks that run without a simulator or an arm |
| `ci/run.sh` | The one validation command, used locally and in CI |

## Prerequisites

- Python 3.11 or newer for the library and tests. No third-party packages are needed.
- For training: a Linux machine with an NVIDIA RTX GPU and [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 3.0 installed. Isaac Sim does not run on macOS.
- For the real arm (not implemented yet): a ROBOTIS OMX follower arm and its USB controller.

## Quickstart

The parts that work today need only Python:

```sh
git clone https://github.com/ichen27/understudy.git
cd understudy
python3 -m venv .venv && . .venv/bin/activate
bash ci/run.sh
```

Compute where the gripper tip is for a set of joint angles:

```sh
python3 -c "
from understudy import arm
from understudy.kinematics import Chain
print(Chain().tip_position([j.default for j in arm.ARM_JOINTS]))
"
```

This prints the tip position in metres in the arm's base frame, about `(0.195, -0.002, 0.068)` for the default pose.

## Training in Isaac Lab

Verified with Isaac Lab 3.0.0 on one NVIDIA L40S. On a machine with Isaac Lab, from this repository:

```sh
export ISAACLAB_PATH=/path/to/IsaacLab

# Check the simulated arm against the shared description (masses, limits, kinematics).
$ISAACLAB_PATH/isaaclab.sh -p scripts/check_sim_model.py --headless

# Train. About ten minutes for 1,000 iterations with 4,096 simulated arms.
$ISAACLAB_PATH/isaaclab.sh -p scripts/isaaclab_run.py train --task Understudy-Reach-OMX-v0 --headless

# Score the result on 5,000 reaches and export the policy.
$ISAACLAB_PATH/isaaclab.sh -p scripts/evaluate.py --headless \
    --checkpoint logs/rsl_rl/understudy_reach_omx/<run>/model_999.pt --output results/reach/<name>
```

The reach task: the gripper tip must reach a random point in a 14 × 24 × 18 cm box in front of the arm. A reach counts as a success within 2 cm. The policy sees five joint angles, five joint velocities, the target and its own last action, and outputs five joint position targets at 30 Hz.

The pickup task uses the same scripts with its own task name. A foam golf ball starts at rest in a 6 × 12 cm area in front of the arm, and the arm must lift it and hold it at a target point 10–16 cm above the desk within five seconds. Expect about an hour of training.

```sh
# Confirm the simulated gripper can physically hold the ball, with no learning involved.
$ISAACLAB_PATH/isaaclab.sh -p scripts/check_pickup_scene.py --headless

$ISAACLAB_PATH/isaaclab.sh -p scripts/isaaclab_run.py train --task Understudy-Pickup-Ball-OMX-v0 --headless
$ISAACLAB_PATH/isaaclab.sh -p scripts/evaluate_pickup.py --headless \
    --checkpoint logs/rsl_rl/understudy_pickup_omx/<run>/model_1499.pt --output results/pickup/<name>
```

In the pickup task the ball's position is read from the simulator. On a real arm it would have to come from a camera, which is not built.

## Calibration

`understudy.arm.Calibration` maps a real arm's encoder readings to the model's joint angles. It is a JSON file:

```json
{"zero_ticks": [2048, 2048, 2048, 2048, 2048], "signs": [1, 1, 1, 1, 1]}
```

`zero_ticks` is each servo's reading with the arm held in the model's zero pose (upper arm vertical, forearm and gripper pointing straight forward). `signs` is `1` where increasing ticks increase the model's angle and `-1` where they decrease it. A tool to capture this from a live arm is planned.

## Validation

```sh
bash ci/run.sh
```

This runs the unit tests and compiles every Python file. It does not start a simulator or touch hardware. The Isaac Lab tasks are checked by hand with the `scripts/check_*.py` and `scripts/evaluate*.py` scripts on a GPU machine; CI does not run them.

## Results and artifacts

| Run | What it is | Contents |
| --- | --- | --- |
| [Reach baseline, 2026-11-24](results/reach/2026-11-24-baseline/README.md) | First reach policy, simulation only | Exported policy (ONNX and TorchScript), training checkpoint, run settings, training curves, five evaluation files |
| [Ball pickup, first policy, 2026-11-24](results/pickup/2026-11-24-first/README.md) | First pickup policy, simulation only | Video and still frames, exported policy, training checkpoint, run settings, training curves, seven evaluation files |

## Contributing

Issues and pull requests are welcome. Please run `bash ci/run.sh` before opening a pull request, and say whether a change was tested in simulation, on a real arm, or neither. Report problems through GitHub issues, including your Isaac Lab version for simulation problems.

## License

MIT. See [LICENSE](LICENSE). The arm model under `assets/open_manipulator_description/` is from ROBOTIS under Apache-2.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
