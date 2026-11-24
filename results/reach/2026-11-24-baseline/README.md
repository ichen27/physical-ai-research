# Reach baseline, 2026-11-24

First trained reach policy for the OMX arm. **Simulation only.** It has not been run on a real arm.

## How it was produced

- Isaac Lab 3.0.0 with Isaac Sim 6.0.1, PhysX, one NVIDIA L40S.
- Task `Understudy-Reach-OMX-v0`, 4,096 simulated arms, 1,000 PPO iterations, seed 42.
- Training time: 566 seconds.
- Code: commit `3a3d907` plus the evaluation script added in the commit that introduced this directory.

```sh
$ISAACLAB_PATH/isaaclab.sh -p scripts/isaaclab_run.py train --task Understudy-Reach-OMX-v0 --headless --seed 42
$ISAACLAB_PATH/isaaclab.sh -p scripts/evaluate.py --headless \
    --checkpoint logs/rsl_rl/understudy_reach_omx/<run>/model_999.pt --output results/reach/<name>
```

## Results

Each row is 5,000 reaches by the same policy. A reach is scored on the mean gripper-tip error over the last half second of a four-second hold. "Servo stiffness" multiplies the stiffness the policy was trained with, keeping the damping ratio.

| File | Servo stiffness | Sensor noise | Within 2 cm | Within 1 cm | Median error | 95th percentile | Worst |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `evaluation.json` | ×1 (as trained) | off | 100% | 99.9% | 1.0 mm | 4.2 mm | 11.7 mm |
| `evaluation-noise.json` | ×1 | on | 100% | 99.9% | 1.5 mm | 4.3 mm | 11.8 mm |
| `evaluation-soft-x0.4.json` | ×0.4 | off | 100% | 92.6% | 7.4 mm | 10.3 mm | 14.4 mm |
| `evaluation-soft-x0.4-noise.json` | ×0.4 | on | 100% | 91.8% | 7.4 mm | 10.4 mm | 15.7 mm |
| `evaluation-stiff-x6.json` | ×6 | off | 100% | 29.7% | 10.9 mm | 12.8 mm | 18.8 mm |

## What the numbers mean

- With the servos it was trained on, the policy is accurate to about a millimetre. The simulated arm sags 1–3° under its own weight, and the policy has learned to command past the target to cancel that.
- That correction is tuned to one servo stiffness. With stiffer or softer servos the same correction is wrong, and the typical error rises to 7–11 mm. Every reach still lands inside the 2 cm success radius.
- The real servo stiffness is not known. The value used here is a data-sheet estimate, and the vendor's own simulation model uses a value about six times higher, which is the ×6 row. Expect roughly centimetre accuracy on a real arm from this policy, not millimetre accuracy, until the stiffness is measured or the policy is retrained across a range of stiffness.

## What this does not cover

- No real-arm run.
- No control delay, encoder quantisation, backlash or friction beyond the simulator defaults.
- No obstacles. Targets are free-space points in a 14 × 24 × 18 cm box in front of the arm.

## Files

| File | Purpose | How to use |
| --- | --- | --- |
| `policy.onnx`, `policy.onnx.data` | Exported policy for running outside Isaac Lab | ONNX Runtime; input `obs` of shape `[1, 22]`, output `actions` of shape `[1, 5]` |
| `policy.pt` | The same policy as TorchScript | `torch.jit.load` |
| `model_999.pt` | Training checkpoint, for evaluation or further training | `scripts/evaluate.py --checkpoint` |
| `params/env.yaml`, `params/agent.yaml` | Exact task and PPO settings of the run | Read as YAML |
| `training.tfevents` | Training curves | `tensorboard --logdir .` |
| `evaluation*.json` | The measurements in the table above | Read as JSON |

## Policy interface

Input, 22 numbers in this order:

1. Five arm joint angles minus the default pose in `understudy/arm.py`, radians.
2. Five arm joint velocities, radians per second.
3. The target: position in metres in the arm base frame, then the quaternion `0, 0, 0, 1`, which is constant (confirmed against the simulator).
4. The previous five outputs.

Output: five numbers. The joint position target is `default + 0.5 × output`, clamped to the joint limits, sent at 30 Hz.

A recording of this run exists but its camera cut off most of the arm, so it is not included. The camera settings need fixing before a demonstration video is added.
