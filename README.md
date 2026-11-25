# Understudy

Train a small robot arm in simulation, then run the trained policy on the real arm.

![The simulated arm picking up a ball](results/pickup/2026-11-24-claw/mujoco-pickup-frames.png)

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
| Trained pickup policy | Holds the ball in two simulators. **Not yet tried on a real arm.** See [results](results/pickup/2026-11-24-claw/README.md) |
| Cross-check in a second simulator (MuJoCo, vendor arm model) | Working, no GPU needed |
| Real-arm runner and calibration tool | Written and tested against a simulated arm. On the real arm only the read-only `status` step has run; **no motor has been driven** |
| Locating the ball with a camera | Not started |

In simulation the first policy reached within 2 cm on 5,000 of 5,000 scored reaches, with a median error of 1 mm. With servos six times stiffer or 2.5 times softer than it was trained on, it still reached within 2 cm every time, but the median error rose to 7–11 mm. Nothing has been run on a real arm yet.

The current pickup policy is trained on the vendor's servo model with every simulated arm, ball and camera different, and with the gripper fingers given their real claw shape. Run in a second simulator, on the vendor's own model of the arm, it lifts the ball in 200 of 200 attempts and still holds it at the end in 96%. It holds the ball 3–4 cm from the requested point, not on it, and nothing has been tried on a real arm. It is the checkpoint from 1,500 of a planned 4,000 training iterations; the rest of that run is on a rented machine that stopped when its credit ran out.

Getting there took three corrections that only showed up in the second simulator: the first policy depended on one servo stiffness, the training ball was too reluctant to roll, and the simulated fingers were solid wedges where the real ones are hollow claws.

Known limitations:

- The servo values in `understudy/arm.py` come from the vendor's MuJoCo model, which labels them approximate. They have not been measured on a real arm. Pickup training varies them widely for that reason; the reach baseline was trained earlier on single data-sheet estimates.
- An arm set up through LeRobot has its own encoder offsets. It needs a calibration file (see [Calibration](#calibration)) before any policy output is sent to it.
- The joint limits are a conservative choice for one arm on a clear desk. Check them against your own setup.

## Layout

| Path | Purpose |
| --- | --- |
| `understudy/arm.py` | Joint order, limits, servo assumptions, tick and radian conversion, calibration |
| `understudy/kinematics.py` | Gripper-tip position from joint angles, read from the URDF |
| `understudy_isaac/` | Isaac Lab task and training settings. Needs Isaac Lab |
| `understudy/pickup.py` | The ball, the gripper commands and the success criteria for the pickup task |
| `understudy/variation.py` | How much each simulated arm, ball and camera differs during training |
| `understudy/runtime.py` | The policy's inputs and outputs outside the training simulator |
| `understudy/runner.py` | Runs a policy on an arm through servo units; used for the real arm and for the MuJoCo cross-check |
| `understudy/dynamixel.py` | The real arm's servos. Reads have run on the real arm; **no motor has been driven** |
| `understudy/mujoco_arm.py` | The vendor's MuJoCo model of the arm behind the same servo interface |
| `understudy/sim_urdf.py` | Generates the corrected URDF used in simulation |
| `scripts/build_finger_collision.py` | Builds the fingers' collision shapes as convex pieces that keep their claw shape |
| `scripts/isaaclab_run.py` | Runs Isaac Lab's train and play scripts with this project's task registered |
| `scripts/check_sim_model.py` | Checks the simulated arm's masses, limits and kinematics against the shared description |
| `scripts/check_pickup_scene.py` | Runs a fixed, unlearned grasp to confirm the simulated gripper can hold the ball |
| `scripts/evaluate.py` | Scores a trained reach policy on thousands of reaches and exports it |
| `scripts/evaluate_pickup.py` | Scores a trained pickup policy on thousands of attempts and exports it |
| `scripts/mujoco_pickup.py` | Runs a pickup policy in MuJoCo on the vendor's model, with no GPU |
| `scripts/record_rollout.py` | Records one attempt in Isaac Lab as a fixture for the tests |
| `scripts/real_arm.py` | Calibrates the real arm and runs a policy on it. Only `status` has run on the real arm |
| `results/` | Trained policies and their measurements |
| `assets/generated/` | The generated simulation URDF and the fingers' collision pieces |
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

The pickup task uses the same scripts with its own task name. A foam golf ball starts at rest in a 6 × 12 cm area in front of the arm, and the arm must lift it and hold it at a target point 10–16 cm above the desk within five seconds. Every simulated arm, ball and camera differs, within the ranges in `understudy/variation.py`. Expect about an hour of training.

```sh
# Confirm the simulated gripper can physically hold the ball, with no learning involved.
$ISAACLAB_PATH/isaaclab.sh -p scripts/check_pickup_scene.py --headless

$ISAACLAB_PATH/isaaclab.sh -p scripts/isaaclab_run.py train --task Understudy-Pickup-Ball-OMX-v0 --headless
$ISAACLAB_PATH/isaaclab.sh -p scripts/evaluate_pickup.py --headless \
    --checkpoint logs/rsl_rl/understudy_pickup_omx/<run>/model_1499.pt --output results/pickup/<name>
```

In the pickup task the ball's position is read from the simulator and then delayed and perturbed as a camera's report would be. On a real arm it would have to come from a camera, which is not built.

## Checking a policy without a GPU

`scripts/mujoco_pickup.py` runs an exported pickup policy in MuJoCo, a different physics engine from the one it was trained in, on the vendor's own model of the arm. It goes through `understudy/runner.py`, the same code that drives the real servos, in encoder ticks.

```sh
pip install -r requirements-dev.txt
python scripts/mujoco_pickup.py --policy results/pickup/2026-11-24-claw/policy.onnx --attempts 200
python scripts/mujoco_pickup.py --policy results/pickup/2026-11-24-claw/policy.onnx --attempts 3 --video pickup.mp4
```

This check has paid for itself three times: it showed that an early policy depended on one servo stiffness, that the training scene's ball was too reluctant to roll, and that the simulated fingers could not hold a ball the way the real ones would.

## Real arm

**Only the first step, `status`, has been run on a real arm (2026-11-25); nothing here has driven its motors.** That first read found a bug in the driver's error handling, since fixed, which is a fair sample of what to expect from the untested steps. `understudy/dynamixel.py` and `scripts/real_arm.py` are written from the servo documentation and from register addresses that earlier read-and-move scripts used successfully on the author's arm, and their logic is tested against a simulated arm, but the first real run is still to come. Treat it as untested code driving motors: keep a hand on the arm's power switch.

What the tool changes on the servos: only volatile settings (position gains set to the vendor's 1000, 0, 1000; a 50 ms time-based profile; a 500 ms bus watchdog; the gripper's goal current). It restores them and switches torque off when it exits. It never writes permanent settings and refuses to run unless they already match the vendor follower configuration (time-based drive mode, position modes on the arm joints, current-based position mode on the gripper).

Go through these in order. The first three only read.

```sh
pip install dynamixel_sdk onnxruntime numpy
python scripts/real_arm.py status          # servo models, modes, errors, temperatures
python scripts/real_arm.py clear-errors    # only if status reports a hardware error; restarts those servos, nothing moves
python scripts/real_arm.py calibrate       # you hold the limp arm in the zero pose
python scripts/real_arm.py check           # you touch the gripper tip to a ruler mark; reports the error
python scripts/real_arm.py start-pose      # first powered motion: a slow move to the start pose and back
python scripts/real_arm.py run --policy results/pickup/2026-11-24-claw/policy.onnx --ball 0.21 0.0 --dry-run
python scripts/real_arm.py run --policy results/pickup/2026-11-24-claw/policy.onnx --ball 0.21 0.0
```

Positions are metres from the centre of the arm's base plate, on the desk: x forward, y to the arm's left.

Two things are missing for a proper run:

- **A camera.** The policy is trained to be given the ball's position continuously. There is no camera pipeline yet. `--ball X Y` is a stand-in: the ball is assumed to stay where you put it until the gripper closes on something, and then to be at the gripper. It cannot see the ball roll away. In the MuJoCo cross-check the stand-in does as well as an exact camera (99.5% held), so a first attempt does not have to wait for a camera.
- **Knowledge of the real servos.** Training covers a wide range of servo stiffness because the real value is unknown. Measuring it would allow a narrower, more reliable policy.

### Calibration file

`understudy.arm.Calibration` maps a real arm's encoder readings to the model's joint angles. `scripts/real_arm.py calibrate` writes it to `calibration.local.json`, which is not committed because it belongs to one physical arm:

```json
{"zero_ticks": [2048, 2048, 2048, 2048, 2048], "signs": [1, 1, 1, 1, 1], "gripper_closed_tick": 1740, "gripper_sign": 1}
```

`zero_ticks` is each arm servo's reading in the model's zero pose (upper arm vertical, forearm and gripper pointing straight forward). `signs` is `1` where increasing ticks increase the model's angle. `gripper_closed_tick` is the gripper servo's reading when closed on nothing, and `gripper_sign` is `1` if opening increases its ticks.

## Validation

```sh
bash ci/run.sh
```

This runs the unit tests and compiles every Python file. The tests include a replay of an attempt recorded in Isaac Lab, which checks that the code used outside the simulator builds the policy's input and reads its output exactly as Isaac Lab did, and an end-to-end pickup in MuJoCo through the servo interface, once on an arm with offset and reversed servos. It does not start Isaac Lab or touch hardware. The Isaac Lab tasks are checked by hand with the `scripts/check_*.py` and `scripts/evaluate*.py` scripts on a GPU machine; CI does not run them.

## Results and artifacts

| Run | What it is | Contents |
| --- | --- | --- |
| [Reach baseline, 2026-11-24](results/reach/2026-11-24-baseline/README.md) | First reach policy, simulation only | Exported policy (ONNX and TorchScript), training checkpoint, run settings, training curves, five evaluation files |
| [Ball pickup, first policy, 2026-11-24](results/pickup/2026-11-24-first/README.md) | First pickup policy, identical simulated arms | Video and still frames, exported policy, training checkpoint, run settings, training curves, seven evaluation files |
| [Ball pickup with claw-shaped fingers, 2026-11-24](results/pickup/2026-11-24-claw/README.md) | Current pickup policy, holds the ball in two simulators | MuJoCo video and still frames, exported policy and its interface file, eleven MuJoCo evaluation files. The training checkpoint is on a stopped rented machine |
| [Ball pickup with variation, 2026-11-24](results/pickup/2026-11-24-varied/README.md) | Earlier pickup policy, which dropped the ball in the second simulator | Videos from both simulators, exported policy and its interface file, training checkpoint, run settings, training curves, eight Isaac Lab and six MuJoCo evaluation files, and the earlier absolute-command version |

## Contributing

Issues and pull requests are welcome. Please run `bash ci/run.sh` before opening a pull request, and say whether a change was tested in simulation, on a real arm, or neither. Report problems through GitHub issues, including your Isaac Lab version for simulation problems.

## License

MIT. See [LICENSE](LICENSE). The arm model under `assets/open_manipulator_description/` is from ROBOTIS under Apache-2.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
