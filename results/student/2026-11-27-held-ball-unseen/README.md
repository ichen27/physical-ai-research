# Does the student need to see the ball it is holding? 2026-11-27

On the real arm the wrist camera changed its own colours once the ball filled its view, and the filter lost the ball for the rest of the run ([first vision runs](../../real-arm/2026-11-27-first-vision-runs/README.md)). This folder asks two things: what the pictures of that run show, and whether the student's lift depends on the picture after the gripper has closed.

**Verdict: in simulation the lift does not need the picture after the grasp, and the student's "sure" was never a sign that it is holding the ball.** Taking the ball's colour out of every picture after the gripper closes changed the result for none of 153 balls. The simulated student also stops being sure of the ball within two seconds of closing with the colours left alone, so the low sureness seen on the real arm while it held the ball is what the student was trained to report, whatever the camera does. The real camera's colour shift is real and is measured below from one run.

Everything in the two simulation sections is MuJoCo only. The real-arm section is read from the saved pictures of one run; no motor was powered and nothing was read from the arm for this work.

## The real run's pictures

Run `live-run-20261127-011102` ([mask-alone](../2026-11-26-mask-alone/README.md) student driving, ball picked up). Each picture at the student's size, 128 by 72, is split into the pixels that are reddish whatever the tint ("ball") and the rest ("background"). The gripper closed at 2.5 s.

| Time | Tip height by the arm's readings | Ball, red green blue | Background | Background's saturation | Filter's share of the picture | Sure |
| --- | --- | --- | --- | --- | --- | --- |
| 3.3 s | -2.5 cm | 247 145 122 | 143 141 141 | 0.02 | 0.56 | 1.00 |
| 4.3 s | -1.7 cm | 217 112 86 | 125 132 133 | 0.07 | 0.57 | 0.99 |
| 5.0 s | -0.1 cm | 188 126 102 | 73 110 111 | 0.34 | 0.55 | 0.99 |
| 5.7 s | 1.3 cm | 181 133 133 | 69 110 125 | 0.45 | 0.26 | 0.05 |
| 6.3 s | 1.9 cm | 164 122 130 | 105 121 142 | 0.26 | 0.00 | 0.00 |
| 7.7 s | 9.0 cm | 125 85 94 | 103 113 138 | 0.25 | 0.00 | 0.00 |
| 14.0 s | 15.0 cm | 163 125 118 | 106 132 134 | 0.21 | 0.23 | 0.30 |

What this supports:

- **The camera changed its colours; the light did not.** The background was neutral grey (saturation 0.02) until 4.0 s and then turned blue-green while the arm was still pressed to the desk. A shadow darkens all three colours together; here the background's red fell by 40 to 50% and its blue by about 10% or less.
- **Its exposure changed as well.** Before the shift 78% of the ball's pixels had red at the ceiling of 255; afterwards none did.
- **It is slow and does not come back.** The ball covered over half the picture from 2.3 s; the shift ran from about 4.3 to 5.7 s; the background was still tinted when the run ended at 15 s.
- **The filter loses the ball by a little.** It accepts hues from 0 to 55 as it measures them. The pink ball reads -9 to -13, and its saturation (0.26 to 0.30) is only just above the least accepted (0.25). From about 9 s the hue creeps back above 0 and a fifth of the picture is marked again.

A lower edge of -20 on the hue would keep the held ball marked through that stretch (0.23 to 0.32 of the picture from 6 to 9 s; the column `filter_share_hue_from_minus_20` in [`real-run-colours.json`](real-run-colours.json)). **That is not a tested fix.** No student has been run with the wider band, and what else it would let through on a real desk has not been measured.

This is one run with one camera under one light. Whether the camera does the same every time, and whether it can be told not to, is not known.

## With the ball's colour taken away after the grasp

The exported student driving the simulated arm from a stream of pictures, wired as `run --camera live --driver` wires it, under the real arm's speed limits (0.3 rad/s, 0.05 rad lead, 15 s). The same 200 balls as [the earlier comparison of the wirings](../2026-11-26-mask-and-grey/wirings); "attempted" are those the student was sure of twice from the look pose, and the counts are of balls 15 to 24 cm out. From the given number of control steps after the gripper closes, every picture is turned grey, so the filter marks nothing. Eighty steps (2.7 s) is about when the real filter began to fail.

| Student | Colour removed | Lifted | Stopped itself before closing | Held high and then lost | Opened the gripper again | Ball's height at the end, median (lowest) |
| --- | --- | --- | --- | --- | --- | --- |
| Mask alone | never | 144 of 153 (94.1%) | 6 | 0 | 0 | 12.7 cm (10.0) |
| Mask alone | 80 steps after closing | 144 of 153 (94.1%) | 6 | 0 | 0 | 14.6 cm (13.2) |
| Mask alone | from the step it closes | 144 of 153 (94.1%) | 6 | 0 | 0 | 14.5 cm (13.1) |
| Mask beside grey | 80 steps after closing | 148 of 155 (95.5%) | 3 | 0 | 0 | 15.4 cm (12.6) |

- **The same balls were lifted in all three rows of the mask-alone student**, ball for ball. The pictures did change what it did: the ball's final height differs in 159 and 164 of the 191 attempts, by up to 5 cm, and with no colour it holds the ball about 2 cm higher. No ball was dropped and the gripper never reopened.
- **For the mask-beside-grey student** only the changed row was run here. An earlier run of the same set with pictures left alone lifted 147 of 154 (95.5%).
- The nine balls not lifted are the same nine in every mask-alone row: six stopped by the not-sure rule before the gripper closed and three it did not lift. None was lost after the colour went.

## What "sure" does after the grasp, colours left alone

The mask-alone student again, pictures untouched, for the 22 balls it lifted out of the first 30 of the same set ([`timecourse-mask-alone.json`](timecourse-mask-alone.json)).

| After closing | Simulation: sure, median | Simulation: balls it is still sure of | Simulation: filter's share | Real run: sure | Real run: filter's share |
| --- | --- | --- | --- | --- | --- |
| 0 s | 0.47 | 11 of 22 | 0.57 | 1.00 | 0.74 |
| 0.5 s | 0.08 | 9 | 0.53 | 1.00 | 0.69 |
| 1.0 s | 0.00 | 7 | 0.41 | 0.99 | 0.56 |
| 1.5 s | 0.00 | 4 | 0.13 | 0.98 | 0.56 |
| 2.0 s | 0.00 | 0 | 0.05 | 1.00 | 0.57 |
| 2.5 s | 0.00 | 0 | 0.03 | 1.00 | 0.55 |
| 3.0 s | 0.00 | 0 | 0.04 | 0.26 | 0.26 |
| 4.0 s | 0.00 | 0 | 0.05 | 0.00 | 0.00 |
| 10.0 s | 0.00 | 0 | 0.05 | 0.60 | 0.23 |

- **In simulation the student stops being sure of the ball as it closes on it**, with the ball still orange and still marked. That follows from how it was trained: "in the picture" means at least half of the ball's disc can be seen, and closed fingers hide more than that (`IN_VIEW` in `understudy/dagger.py`). Over all 153 balls of the first row above, its sureness from 3.3 s after closing averaged 0.001, and for no ball more than 0.03.
- So a sureness near zero while holding is the trained behaviour. It cannot be used as a check that the ball is still held, and its fall on the real arm does not by itself show that anything went wrong.
- **The real run differs from simulation in the other direction.** The real student stayed sure for 2.5 s after closing where the simulated one does not, and on the real arm the held ball covers about a third of the picture from 9 s on, where in simulation the filter marks only about 5% of it from two seconds after closing. Why is not known. The real wrist stops bending back at about 10 degrees where the simulated one goes to 97, which would leave the gripper, and the camera on it, pointing differently; that has not been tested as the cause.

## What follows

- The pickup as it stands is not put at risk by the camera's colour shift, as far as simulation can say: one real run agrees (it lifted and held).
- Anything that is to use the picture after the grasp (carrying the ball to the bowl by sight, or checking it is still held) cannot rely on the simulated held-ball pictures. They differ from the real ones in colour, in how much of the ball shows, and in where it sits.
- For training in the second simulator: fix the real camera's white balance and exposure if it allows it, and in any case show the ball in the shifted colours as well as the desk colour.
- The next look after a run starts from a camera that may still be tinted. Whether the first look of the next attempt is affected has not been checked.

## Not tested

- Nothing here ran on the real arm. The real-arm figures are from the saved pictures of one run.
- Removing the colour is a stand-in for the real shift: it empties the filter's mask, which is all the mask-alone student sees, but it is not the real tinted picture. The mask-beside-grey student would also see a real change in the grey picture's brightness, which is not reproduced.
- The colour picture student was not run.
- Pictures arrived on time, 30 a second; the real stream gave 10.
- Whether the camera's adjustment can be switched off.

## Reproducing

From the repository's root, with `requirements-train.txt` installed (about 15 minutes a row on an M4 Mac mini):

```sh
python results/student/2026-11-27-held-ball-unseen/held_ball_unseen.py results/student/2026-11-26-mask-alone/driver.onnx 8 200 out.json 80
python results/student/2026-11-27-held-ball-unseen/held_timecourse.py results/student/2026-11-26-mask-alone/driver.onnx 30 timecourse.json
python results/student/2026-11-27-held-ball-unseen/colour_through_run.py RECORD.json RECORD.pictures.npz colours.json
```

The last needs the pictures the tool saves beside a record of `run --camera live --driver`. They are the real camera's and are not in the repository.

| File | What it is |
| --- | --- |
| `held_ball_unseen.py` | The measurement with the colour removed after closing; the last argument is the number of steps, -1 for never |
| `mask-alone-left-alone.json`, `mask-alone-after-80.json`, `mask-alone-after-0.json`, `mask-and-grey-after-80.json` | One row of the table each, a line per ball |
| `held_timecourse.py`, `timecourse-mask-alone.json` | Sureness, the filter's share and the ball's height at every step for 22 lifted balls |
| `colour_through_run.py`, `real-run-colours.json` | The colours through the real run, every tenth step |
