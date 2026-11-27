# Runs on the real arm, 25 to 27 November 2026

Every recorded run of a pickup on the real ROBOTIS OMX arm so far, in one table, with the step-by-step recording of each.

**Verdict: the arm has picked up a real ball three different ways, a few times each. None of these is a success rate.** The counts are small, the ball was always the same foam golf ball on the same desk, and the runs were made while the tools were still changing.

| How the ball's position reached the arm | Pickups attempted | Ball lifted and held |
| --- | --- | --- |
| Typed in, after resting the gripper tip on the spot (no camera) | 1 recorded (an earlier, unlimited attempt hit the desk and was not recorded) | 1 |
| A trained eye looks once from a raised pose; the teacher policy does the pickup | 6 | 4 |
| A trained student drives the arm itself from live video | 3 | 2 (the third was stopped by the student's own stop rule after 0.6 s) |

All runs used the speed limits (joint targets move at most 0.3 rad/s and sit at most 0.05 rad from the joint) and 15 second attempts.

## Every run

"Ball at" is the ball's distance from the arm's swivel axis by the arm's own estimate (for the first row, from the typed position). "Turn" is how far the arm turned to face the ball. "Held" is how long the fingers were stopped by something between them after closing, and "lifted to" is the highest the gripper tip got while they were; both are read from the recording, not judged by eye. The same numbers are in [`runs.json`](runs.json).

| Date | Wiring | Model | Ball at | Turn | Outcome | Gripper closed at | Held | Lifted to |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25 Nov | Position typed in | Teacher ([claw](../../pickup/2026-11-24-claw/README.md)) | 21.7 cm | 0° | Picked up; the author knocked the ball out by hand | 2.0 s | 6.5 s | 18.4 cm |
| 26 Nov | Eye, then teacher | [Mask beside grey](../../student/2026-11-26-mask-and-grey/README.md) | 20.5 cm | -2° | Picked up, held to the end | 2.3 s | 12.3 s | 21.9 cm |
| 26 Nov 21:08 | Eye, then teacher | Mask beside grey | 17.6 cm | -1° | Closed on nothing | 2.2 s | 0 | |
| 26 Nov 21:10 | Eye, then teacher | Mask beside grey | 16.9 cm | +19° | Closed on nothing | 2.1 s | 0 | |
| 26 Nov 21:10 | Eye, then teacher | Mask beside grey | | | Stopped before the pickup: no ball seen (the ball was at the edge of the picture, about 30 cm away) | | | |
| 26 Nov 21:11 | Eye, then teacher | Mask beside grey | 23.4 cm | -25° | Picked up, held to the end | 2.6 s | 12.1 s | 18.3 cm |
| 26 Nov 21:20 | Eye, then teacher | Mask beside grey | 21.0 cm | -8° | Picked up, held to the end | 2.5 s | 12.2 s | 22.0 cm |
| 26 Nov 21:32 | Eye, then teacher | Mask beside grey | 19.5 cm | +13° | Picked up; the author knocked the ball out by hand at 13 s | 2.4 s | 10.5 s | 24.0 cm |
| 27 Nov 01:11 | Student driving | [Mask alone](../../student/2026-11-26-mask-alone/README.md) | 22.0 cm | +3° | Picked up, held to the end | 2.5 s | 12.1 s | 16.8 cm |
| 27 Nov 01:13 | Student driving | Mask alone | 25.1 cm | +29° | Refused before the pickup: beyond the 15 to 24 cm reach band | | | |
| 27 Nov 01:14 | Student driving | Mask alone | 25.1 cm | +29° | Refused: beyond the reach band | | | |
| 27 Nov 01:14 | Student driving | Mask alone | 24.8 cm | +16° | Refused: beyond the reach band | | | |
| 27 Nov 01:15 | Student driving | Mask alone | 22.0 cm | +1° | Picked up, held to the end | 2.5 s | 12.1 s | 17.5 cm |
| 27 Nov 01:15 | Student driving | Mask alone | | | Stopped before the pickup: the ball was seen once and not again after the arm turned to face it | | | |
| 27 Nov 01:16 | Student driving | Mask alone | 23.7 cm | +11° | Stopped 0.6 s into the pickup: the student had not been sure of the ball for 20 steps in a row. The arm lowered itself and switched off | | | |

## What the recordings show

- **Both misses were the two nearest balls** (16.9 and 17.6 cm), and every ball from 19.5 cm out was picked up. In simulation it is the other way round: near balls are the easiest for this wiring (88% lifted at 15 to 18 cm, 73% further out). So the misses are not what the simulation predicts.
- **In both misses the arm went where it was told.** At the moment the gripper closed, its tip was within a centimetre of where the eye had said the ball was, every joint was following its target normally, and the fingers then closed completely. The ball was not between them.
- **The teacher leaves no margin on near balls.** Told the true position in simulation, it closes with the gripper about 4 mm past the ball's centre at 19 cm but level with the centre at 15 to 16 cm. A few millimetres of error from anywhere then closes on nothing. In one of the two misses the real arm also stopped 7 mm shorter than the simulated arm does.
- **What is not settled is where those millimetres come from.** A second, hand-written ball finder agrees with the trained eye to 1 to 3 mm on two of these pictures, and both use the same arm calibration and camera calibration. A direct check, finding the ball and then touching it with the gripper tip, has not been done yet.
- **The gripper presses below desk level.** By the arm's readings the tip went 1.1 to 1.9 cm "below the desk" during the teacher's pickups and 2.6 cm during the student's. The teacher asks for about 1.5 cm below the desk in simulation too, where the desk is rigid. On the real arm something gives by 2 to 3 cm: the surface, the base, or the arm itself.
- **The wrist stops early.** The wrist-bend joint stops at about 10 to 12 degrees back on every real lift, where the simulated arm goes on to 97. What stops it (the camera, its cable, or the joint) is not known. In simulation a wrist stopped there changes neither the lift rate nor the carry.
- **The live video arrives at 10 pictures a second, not 30.** The record of the student's run has a new picture on every third control step. The student was trained on 30 a second and still picked the ball up twice. How late each picture is has not been measured.
- **The far stop was the camera's view, not the student's hand, and probably the slow video too.** From the lowered start pose the camera cannot see a ball much beyond 19 cm whole. In simulation about 38% of a ball at 23.7 cm is in that picture, against 89% at 22 cm, and the mask-alone eye is sure of it in 22% of looks against 97%. On the real arm the student was 0.00 sure on all 19 recorded steps and the stop rule fired on the twentieth, as designed. In simulation the same start does not end the attempt: the arm moves and the ball comes into view within the 20 steps, and the mask-beside-grey student lifted 28 of 28 balls at 23.2 to 24.9 cm (27 of 28 with pictures 100 ms late). The mask-alone student was not run on that set. With 10 pictures a second, 20 steps is six or seven pictures, so the likeliest reading is that the view did not catch up in time. That is an inference: this run's pictures were not kept.

## Contents

| File | What it is |
| --- | --- |
| `runs.json` | The table above as data, plus the lowest height the gripper tip reported in each run |
| `records/*.json` | One file per run: the run's settings and, for each control step (30 a second), the joint angles, the joint targets, whether the gripper was commanded open, the gripper tip's position, where the ball was taken to be, and the policy's output. The student's runs also record how sure it was and how old its picture was. A run that stopped before the pickup has no steps |

The pictures the camera took during these runs are not in this repository.

Read a record with any JSON reader. The chart of one run in the main README is drawn from `records/live-run-20261127-011102.json` by `docs/figures/make_figures.py`.
