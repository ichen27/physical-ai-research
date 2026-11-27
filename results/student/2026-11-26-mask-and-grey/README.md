# A student given a filtered picture: the ball's colour as a mask, beside the picture in grey, 2026-11-26

The same student as [stage A](../2026-11-26-stage-a/README.md), trained the same way from the same seed, with one change: it is not shown the colour picture. Each picture first goes through a fixed filter, the colour test of the hand-written ball detector (`understudy.vision.orange_mask`, made wide: red through orange to yellow, pale included), and the student is given the filter's answer, which pixels are the ball's colour, beside the picture in grey. Simulated and real pictures go through identical code, so what colour the real camera happens to give the ball is dealt with once, in the filter, and not learnt.

The simulated pictures also changed for this run: the ball is shown in the colour the real camera gives it (red at 255, green about 187, blue about 133) in four looks out of five, over a spread about three times what the real pictures showed.

**Verdict so far: its eye recognises the real ball.** On saved real pictures, and in one run on the real arm (below). Its measurements in simulation are still being made, and the student's own moves have not run on the real arm.

## On real pictures

65 real pictures show the ball whole, from three sessions on one desk; for 19 the arm's pose was recorded and the ball's place is known. `real-pictures-all.json`.

| Student | Sure the ball is there (65 pictures) | Sure (the 19) | Error in the ball's place, median (the 19) | Worst | Within 1 cm |
| --- | --- | --- | --- | --- | --- |
| Colour picture, first simulated pictures | 3 of 65 | 0 of 19 | 36.3 mm | 89.7 mm | 1 of 19 |
| Colour picture, stage A | 18 of 65 | 3 of 19 | 9.8 mm | 44.7 mm | 10 of 19 |
| **Mask beside grey, this one** | **65 of 65** | **19 of 19** | **8.1 mm** | **24.0 mm** | **13 of 19** |

On ten real pictures without the ball on the desk (the bare desk, its speckled surface, the other things on it) it is at most 10% sure, and under 4% in nine of them, where it needs half.

This is not a blind test. The filter's width and the way training masks are spoiled (patches of specks, a hole in the ball, a bite out of it) were chosen after looking at these pictures. All of them are of one ball on one desk in one room, with the camera's exposure on automatic, and the arm was still for every one.

## On the real arm, once

`run --camera look --eye`, 2026-11-26, from a working copy of this code a few minutes older than the commit: from the look pose the eye was 1.00 sure, the arm turned 2 degrees, the eye was 0.99 sure again and put the ball 20.5 cm from the swivel axis, and the policy picked it up under the speed limits: by the arm's own readings the gripper closed at 2.3 s and the fingers stayed on the ball to the end. Up to the grasp the real arm and the simulated one were within a millimetre of each other. One attempt, one place; not a success rate. The recording is local data and is not in this repository.

Two things in that recording differ from simulation and matter for what comes next. The wrist stopped bending back at 10 degrees, where the simulated arm goes on to 97 and the policy asks for it on every lift; whether the camera, its cable or a limit stored in the servo stops it is not yet known. And lifting, the arm swung 41 degrees to the right of where it faced, against 34 for the same ball in simulation. In simulation, with the wrist stopped at 10 degrees (`scripts/mujoco_carry.py --wrist_stop_deg=-10`, 150 attempts), the lift rate, the swing and the carry to a bowl were unchanged (98% lifted, 42 degrees at the worst, every lifted ball in the bowl) and only the pose the pickup ends in was higher.

## What the filter passes on real pictures

Of the ball's disc, 95 to 99% at the median (57% at the least, in one picture). Of the rest of the picture, 3 to 4% at the median and 8% at the most: mostly specks of the beige surface, in patches. That is the new gap this opens, the mask's shape in place of the ball's colour, and the training masks are spoiled to cover it.

## The eye, exported

`eye.onnx` (0.4 MB) with `eye.json` beside it is this student's eye on its own: from one picture and the arm's joint readings, where the ball is and how sure it is that the ball is in the picture. `understudy.eye.Eye` runs it with numpy, OpenCV and onnxruntime; PyTorch is not needed. `scripts/export_eye.py` made it and checked that it says what the student says (to 1e-7).

`scripts/real_arm.py run --camera look --eye results/student/2026-11-26-mask-and-grey/eye.onnx` uses it in place of the hand-written finder; see the main README.

`driver.onnx` (0.9 MB) with `driver.json` is the whole student, memory and moves and all, one control step at a time (`understudy.eye.Driver`); it agrees with the PyTorch student to 3e-6 over 60 steps in a row. `run --camera live --driver` lets it move the arm from the camera's stream. One step takes about 1 ms on an M4 Mac (3 ms at the slowest of 300), shrinking and filtering the picture included.

## Training

| Round | Student drove: lifted | First guess of the ball's place, median error |
| --- | --- | --- |
| 1 | 0 of 66 | 12.7 mm |
| 2 | 28 of 86 | 12.3 mm |
| 3 | 39 of 94 | 9.0 mm |
| 4 | 57 of 109 | 9.5 mm |
| 5 | 113 of 120 | 9.5 mm |
| 6 | 94 of 120 | 11.5 mm |
| 7 | 106 of 120 | 9.5 mm |

```sh
python scripts/train_student.py --camera camera.local.json --data /tmp/student-data --out out --rounds 8 --input mask+grey
python scripts/export_eye.py --student out/student.pt --output out/eye.onnx
python scripts/student_real_checks.py all --student out/student.pt --samples FOLDER --calibrated cal4
```

## Contents

| File | What it is |
| --- | --- |
| `student.pt` | The trained student (PyTorch) |
| `eye.onnx`, `eye.json` | Its eye, exported, and what it was trained for |
| `driver.onnx`, `driver.json` | The whole student, exported |
| `training.jsonl` | Each round's attempts, lifts and losses |
| `real-pictures-all.json` | The check on every real picture |

The real pictures are not in this repository.
