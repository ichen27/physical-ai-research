# Understudy

Train a small robot arm in simulation, then run the trained policy on the real arm.

The arm is the ROBOTIS OMX follower (five joints plus a gripper, DYNAMIXEL XL430 and XL330 servos). Policies are trained with reinforcement learning in NVIDIA Isaac Lab on a rented GPU and are meant to run on the physical arm from a laptop.

## Status

This is an early project. Read this section before relying on anything.

| Part | State |
| --- | --- |
| Arm description, unit conversions, calibration format | Working, tested |
| Forward kinematics read from the vendor URDF | Working, tested |
| Isaac Lab reach task (`understudy_isaac/`) | Written against Isaac Lab 3.0.0-beta2 source. **Not yet run.** |
| Trained reach policy and its measured success rate | Not done |
| Real-arm runner | Not done |
| Cube lift with a camera | Not started |

Known limitations:

- The simulated servo stiffness and damping in `understudy/arm.py` are estimates from the servo data sheets. They have not been measured on a real arm, and a policy trained on wrong values may not transfer.
- An arm set up through LeRobot has its own encoder offsets. It needs a calibration file (see [Calibration](#calibration)) before any policy output is sent to it.
- The joint limits are a conservative choice for one arm on a clear desk. Check them against your own setup.

## Layout

| Path | Purpose |
| --- | --- |
| `understudy/arm.py` | Joint order, limits, servo assumptions, tick and radian conversion, calibration |
| `understudy/kinematics.py` | Gripper-tip position from joint angles, read from the URDF |
| `understudy_isaac/` | Isaac Lab task and training settings. Needs Isaac Lab |
| `scripts/isaaclab_run.py` | Runs Isaac Lab's train and play scripts with this project's task registered |
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

Not yet verified. On a machine with Isaac Lab, from this repository:

```sh
export ISAACLAB_PATH=/path/to/IsaacLab
$ISAACLAB_PATH/isaaclab.sh -p scripts/isaaclab_run.py train --task Understudy-Reach-OMX-v0 --headless
$ISAACLAB_PATH/isaaclab.sh -p scripts/isaaclab_run.py play --task Understudy-Reach-OMX-Play-v0
```

The task: the gripper tip must reach a random point in a 14 × 24 × 18 cm box in front of the arm. A reach counts as a success within 2 cm. The policy sees five joint angles, five joint velocities, the target and its own last action, and outputs five joint position targets at 30 Hz.

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

This runs the unit tests and compiles every Python file. It does not start a simulator or touch hardware. The Isaac Lab task has no automated check yet.

## Results and artifacts

None yet. Trained policies, training curves, evaluation numbers and videos will be added under `results/` with an index here.

## Contributing

Issues and pull requests are welcome. Please run `bash ci/run.sh` before opening a pull request, and say whether a change was tested in simulation, on a real arm, or neither. Report problems through GitHub issues, including your Isaac Lab version for simulation problems.

## License

MIT. See [LICENSE](LICENSE). The arm model under `assets/open_manipulator_description/` is from ROBOTIS under Apache-2.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
