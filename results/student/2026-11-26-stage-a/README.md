# A pickup policy that sees the camera's picture, stage A, 2026-11-26

A student network that is shown the wrist camera's picture instead of being told where the ball is. It was trained for free on a Mac, in MuJoCo, by copying the pickup policy that is told ([2026-11-24-claw](../../pickup/2026-11-24-claw/README.md), "the teacher" below).

**What this stage is for.** It is not meant to produce the policy that runs on the arm. The final picture-driven policy is to be trained in Isaac Lab on a rented GPU (stage B). This stage's job is to find out, at no cost, what stage B has to get right, so that paid runs are not spent rediscovering it. [What stage B has to carry over](#what-stage-b-has-to-carry-over) is the result.

**Verdict.** In simulation the student finds the ball every time and lifts it in about 70% of attempts, against 98% for the teacher. On real pictures it puts the ball about 1 cm from where it was, and is sure the ball is there in only 17 of 65. The cause of that is found and measured: the real camera shows the ball as a pale peach that the simulated pictures almost never contained. **Nothing here has run on the real arm.** No motor was powered and nothing was read from the arm for this work; the real pictures are ones already taken while the camera was being calibrated.

## What was built

| Part | Where |
| --- | --- |
| Simulated wrist-camera pictures with a different look every attempt | `understudy/sim_look.py` |
| The student network, and the one-step-at-a-time wrapper that would run on the arm | `understudy/student.py` |
| Teaching it by letting it drive while the teacher says what it would have done | `understudy/dagger.py`, `scripts/train_student.py` |
| Measuring it in simulation | `scripts/evaluate_student.py`, `scripts/evaluate_search.py`, `scripts/evaluate_student_carry.py` |
| Scoring it on real pictures | `scripts/student_on_real_pictures.py`, `scripts/student_real_checks.py` |

The student sees a 128 x 72 colour picture (the real camera's picture shrunk ten times) and the arm's own joint readings, 30 times a second. It is never given the ball's position. It has 228,000 weights and takes half a millisecond a step on one processor thread, so it needs no GPU to run. Besides the arm's moves it says where it thinks the ball is and whether the ball is in the picture, which is what the real-picture tests score.

Each attempt has its own desk colour and pattern, lighting, glare, haze over the lens, exposure, blur, noise, picture delay, other things on the desk (red and orange ones among them), and a camera a little off where the calibration put it (3 mm, 1.5 degrees, 4% in focal length). The camera is the real one as calibrated on 2026-11-26.

## Training

Eight rounds. In each the arm makes 120 to 160 attempts with the real arm's speed limits (0.3 rad/s, 0.05 rad lead), with more and more of them driven by the student, while the teacher's answer is recorded at every step; then the student trains on everything gathered so far. 94 minutes on a Mac mini (M4, 16 GB), about 0.8 attempts a second and 2,500 to 3,800 pictures a second, peak memory about 3.5 GB.

| Round | Teacher drove: lifted | Student drove: lifted | Student's first guess of the ball's place, median error |
| --- | --- | --- | --- |
| 0 | 159 of 160 | | |
| 1 | 57 of 59 | 0 of 61 | 9.6 mm |
| 2 | 33 of 33 | 23 of 87 | 13.0 mm |
| 3 | 20 of 20 | 7 of 100 | 7.0 mm |
| 4 | 12 of 12 | 82 of 108 | 6.9 mm |
| 5 | | 90 of 120 | 7.7 mm |
| 6 | | 89 of 120 | 7.9 mm |
| 7 | | 106 of 120 | 8.1 mm |

Copying the teacher's attempts alone teaches nothing useful (round 1: none lifted); the student has to drive and be corrected. An earlier run on pictures made slightly differently (see [the ball's colour](#the-balls-colour-is-what-did-not-carry-over)) learnt faster, 116 of 120 by round 5. In this run the student's eye was as good or better at every round, and the part that lagged was when to close the gripper: that loss was about twice as high throughout. With one run of each, chance cannot be ruled out.

## In simulation

MuJoCo, vendor arm model, 200 attempts a row unless said, the real arm's speed limits, looks the student was not trained on (new draws, and two desk patterns kept out of training). "Lifted at the end" means the ball's centre is more than 5 cm above the desk over the final half second.

**Balls ahead of the arm (one look, then the pickup)**

| File | Who drives | Ball position given to the driver | Ball found | Lifted at the end | Lifted at some point | First-guess error, median |
| --- | --- | --- | --- | --- | --- | --- |
| `mujoco-teacher.json` | teacher | the true position | 100.0% | 97.5% | 97.5% | n/a |
| `mujoco-student.json` | student | none: pictures only | 100.0% | 72.5% | 73.5% | 9.9 mm |
| `mujoco-eye-and-hand.json` | teacher | the student's own estimate | 100.0% | 77.5% | 83.0% | 9.9 mm |
| `mujoco-blank-0.5s.json` | student (picture blanked for steps 40 to 55) | none: pictures only | 100.0% | 55.0% | 56.5% | 9.9 mm |
| `mujoco-student-camera-2.5x-out.json` | student (camera out by up to 7.5 mm, 3.75 deg, 10% focal) | none: pictures only | 100.0% | 61.0% | 61.0% | 13.2 mm |

**Searching, to 57 degrees either side**

| Balls | Attempts | Student: found | Student: refused | Student: lifted at the end | Student: looks, median | Student: first-guess error | Teacher: lifted at the end |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All | 200 | 100.0% | 13.0% | 69.5% | 1.0 | 10.2 mm | 98.0% (200) |
| Turn to face: -57 to -34 deg | 47 | 100.0% | 10.6% | 72.3% | 1.0 | 10.2 mm | 97.9% (47) |
| Turn to face: -34 to -11 deg | 37 | 100.0% | 16.2% | 64.9% | 1.0 | 10.0 mm | 94.6% (37) |
| Turn to face: -11 to +11 deg | 32 | 100.0% | 6.2% | 78.1% | 1.0 | 9.9 mm | 100.0% (32) |
| Turn to face: +11 to +34 deg | 42 | 100.0% | 11.9% | 78.6% | 1.0 | 10.1 mm | 97.6% (42) |
| Turn to face: +34 to +57 deg | 42 | 100.0% | 19.1% | 54.8% | 1.0 | 10.5 mm | 100.0% (42) |
| Distance: 15 to 18 cm | 66 | 100.0% | 3.0% | 83.3% | 1.0 | 9.7 mm | 100.0% (66) |
| Distance: 18 to 21 cm | 58 | 100.0% | 3.5% | 87.9% | 1.0 | 10.8 mm | 96.5% (58) |
| Distance: 21 to 24 cm | 76 | 100.0% | 28.9% | 43.4% | 1.0 | 9.7 mm | 97.4% (76) |

The student's refusals: 20 as too far, 2 as too near, 4 as too far round. It found the ball every time.

**Searching, to 85 degrees either side, the turn limit lifted**

| Balls | Attempts | Student: found | Student: refused | Student: lifted at the end | Student: looks, median | Student: first-guess error | Teacher: lifted at the end |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All | 120 | 99.2% | 10.8% | 73.3% | 1.0 | 10.4 mm | 98.3% (120) |
| Turn to face: -85 to -51 deg | 29 | 96.5% | 10.3% | 69.0% | 3.0 | 10.1 mm | 100.0% (29) |
| Turn to face: -51 to -17 deg | 20 | 100.0% | 15.0% | 80.0% | 1.0 | 9.0 mm | 100.0% (20) |
| Turn to face: -17 to +17 deg | 25 | 100.0% | 16.0% | 68.0% | 1.0 | 11.1 mm | 100.0% (25) |
| Turn to face: +17 to +51 deg | 21 | 100.0% | 14.3% | 71.4% | 1.0 | 11.5 mm | 90.5% (21) |
| Turn to face: +51 to +85 deg | 25 | 100.0% | 0.0% | 80.0% | 2.0 | 9.9 mm | 100.0% (25) |
| Distance: 15 to 18 cm | 46 | 100.0% | 2.2% | 95.7% | 1.0 | 9.5 mm | 100.0% (46) |
| Distance: 18 to 21 cm | 37 | 97.3% | 0.0% | 81.1% | 1.0 | 11.7 mm | 100.0% (37) |
| Distance: 21 to 24 cm | 37 | 100.0% | 32.4% | 37.8% | 2.0 | 9.5 mm | 94.6% (37) |

The student's refusals: 10 as too far, 1 as too near, 2 as too far round. It did not find one ball.

**No ball on the desk**

- `search-no-ball-other-things.json` (only the other things, red and orange ones among them), 150 searches of five looks: said there was a ball in 10.7%; went on to attempt a pickup in 4.0%
- `search-no-ball-bare-desk.json` (bare), 150 searches of five looks: said there was a ball in 0.7%; went on to attempt a pickup in 0.0%

**A ball out of reach**

- nearer than 15 cm (63 balls): found 79.4%; refused as too near 65.1%; pickup attempted 11.1%; of those attempted, within 1 cm of the band 57.1%
- further than 24 cm (57 balls): found 91.2%; refused as too far 89.5%; pickup attempted 1.8%; of those attempted, within 1 cm of the band 0.0%

What these say:

- **The eye limits it, not the hand.** The teacher, driving from the student's idea of where the ball is, does little better than the student's own moves (77.5% against 72.5%), and far worse than the teacher told the truth (97.5%). With these speed limits the teacher needs the ball's place to a few millimetres; the student's first guess is 1 cm out at the median and its running estimate 5 mm.
- **Far balls are the weak spot.** From 21 to 24 cm the student lifts 43% and refuses 29% as "too far", because it puts balls near the far edge about 1 cm further out than they are.
- **It needs the picture all the way in.** Blanking the picture for half a second just before the grasp costs 17 points.
- **Searching wider costs nothing more.** A look from a swivelled arm is shown to the student as a look from straight ahead in a turned frame, so the sweep needed no training. Out to 85 degrees either side the student does as well as within 57 (73% of 120), but the arm does not turn that far today: the limit of 57 degrees is in `understudy/runner.py`, and the swivel's own travel and the cable are the real limits.
- **Other orange things fool it sometimes.** With no ball and only the other things on the desk it says there is a ball in 11% of searches and goes on to attempt a pickup in 4%.

### The whole job: find, pick up, carry to a bowl

100 attempts a row. A bowl 15 cm across with its rim 6.5 cm up stands at the arm's left, 23 to 26 cm from the swivel axis, as a free-standing 400 g thing in the scene and in the camera's pictures; the ball bounces back half its fall. Balls the bowl is in the way of are left out (17 were), as the real tool refuses them. The carry is the fixed move of [the carry's results](../../carry/2026-11-26-bowl/README.md).

| Who picks up | Found the ball | Refused | Lifted | Ended in the bowl | Of those lifted, ended in the bowl | Arm touched the bowl | File |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Teacher, told where the ball is | 100% | 0% | 99% | 99% | 100% | never | `whole-job-teacher.json` |
| Student, pictures only, searching first | 100% | 8% | 74% | 74% | 100% | never | `whole-job-student.json` |

Every ball that was lifted, by either, ended in the bowl. The student lifts from where the teacher would and the carry is the same, so the whole job succeeds as often as the pickup does.

## On real pictures

65 real pictures show the ball whole, from three sessions on one desk. For 19 of them the arm's pose was recorded and the ball's place is known from the camera's calibration (which agrees with itself to 1.5 mm). The student was shown each picture with the arm's joint readings.

| Student | Sure the ball is there (65 pictures) | Sure (the 19) | Error in the ball's place, median (the 19) | Within 1 cm |
| --- | --- | --- | --- | --- |
| Trained on the first simulated pictures | 3 of 65 | 0 of 19 | 36.3 mm | 1 of 19 |
| This one | 17 of 65 | 3 of 19 | 9.8 mm | 10 of 19 |

These are no longer blind tests: the second student's pictures were changed after the first one failed on these same 19.

### The ball's colour is what did not carry over

In all 65 real pictures the ball is the same pale peach: its red is at the top of the scale, 255, in every one; its green is between 146 and 246 (median 185) and its blue between 111 and 161 (median 131). The camera sets its own exposure, overexposes the ball, and its processing then lifts the weaker colours. The ball itself is a strong orange.

The simulated balls were orange. Only 5% of them have the real ball's hue, saturation and brightness together; their blue is typically 61 where the real one is 131 (`real-pictures-colour.json`).

Changing one thing at a time in the 19 real pictures (`real-pictures-changes.json`):

| Change to the real picture | Sure the ball is there | Error, median |
| --- | --- | --- |
| None | 3 of 19 | 9.8 mm |
| The ball repainted a plain orange, keeping its shading | 17 of 19 | 6.1 mm |
| The ball's colour made 1.6 times stronger | 13 of 19 | 7.6 mm |
| The pale and blown-out parts of the ball filled with its own colour | 8 of 19 | 6.5 mm |
| The whole picture 20% darker | 7 of 19 | 8.8 mm |
| The ball 20% darker | 5 of 19 | 7.7 mm |
| The veil taken off the whole picture | 2 of 19 | 9.4 mm |

So with nothing changed but the ball's colour, the same student, looking at the real desk with its real clutter, glare and haze, is sure of the ball in 17 of 19 pictures and places it to 6 mm.

**What carried over and what did not.** The geometry carried over: at the 19 real poses the simulated camera puts the ball within 8 pixels (of 1280) of where the real picture has it, and the student's error on real pictures is close to its error on simulated ones once the colour is put right. Clutter, glare and haze carried over, by the repaint test. The ball's colour did not.

The first fix, already in this student, was the order of two steps: the haze must be added before exposure, as in a camera, not after. That took the student from 3 to 17 of 65 and from 36 mm to 10 mm. It is not enough.

## What stage B has to carry over

Each with the evidence. "Measured" means in simulation or on saved real pictures; nothing was measured on the powered arm.

1. **The camera's real lens and mount.** 975 pixels of focal length, 67 degrees across, looking almost straight down from the start pose. The first guess, a wide lens taken from the maker's listing, was wrong by a factor of nearly two, and a training run on it was thrown away. With the calibrated camera the simulated ball lands within 8 pixels of the real one at the 19 real poses.
2. **Look from a raised pose, and sweep.** From the start pose the real camera shows, whole, 43% of the places a ball can be picked up from. From the look pose it shows all of them within 35 degrees either side, and four more looks cover 57 degrees either side. The student found the ball in 200 of 200 searches. Train on looks from the look pose as well as from the approach.
3. **The ball as the real camera shows it.** Red at 255, green about 185, blue about 131, in 65 of 65 pictures. In a renderer that works in real light this should come from exposing so that the ball's red clips and then applying the camera's tone curve and its haze; whether it does is to be checked against these numbers before any training, with `student_real_checks.py colour`. Spread the colour for other rooms and lamps: across three sessions the real ball's green moved over 100 levels and its blue over 50.
4. **The camera's settings, as used.** All 65 pictures were taken with exposure and white balance on automatic. If they can be fixed on this camera, so that the ball's red does not clip, the real ball will look much more like a plain orange and will stop changing with what else is in view. Whatever is chosen must be what the training pictures imitate, and it changes what "the real camera's picture" means for every later test. To measure at the arm: with each setting, the ball's red, green and blue (`look --save`, then this folder's colour check), over the ball near and far and in the arm's shadow; and whether the setting survives from one capture to the next, since each picture starts the camera afresh.
5. **Haze before exposure, and exposure that wanders.** Measured: the other order cannot produce the real ball's colour at all, and cost 14 of 65 pictures.
6. **Geometry, clutter, glare and blur as they are here.** They carried over. They do not need a better renderer on this evidence.
7. **The speed limits the real arm needs, in training.** The teacher was trained without them. With them it lifts 62% from its trained patch and 98% only when the arm is first turned to put the ball 3 cm to its left, and it swings wide as it lifts (next item). A teacher trained with the limits would need neither workaround.
8. **The pickup swings to its right.** Lifting, the teacher swings up to 42 degrees to the right of where it faces with the gripper still under 11 cm. Anything standing there is struck. The student copies it. See [the carry](../../carry/2026-11-26-bowl/README.md).
9. **The carry and the bowl.** A fixed move after the pickup, not learnt; the bowl stands at the arm's left. The bowl is in the camera's pictures during a search, so it belongs in the training pictures: this student was never shown one, and with a pale bowl in its pictures it still found the ball in 100 of 100 searches and lifted 74. A bowl of another colour was not tried.
10. **How the system is wired.** The student's eye feeding the teacher's hand is as good as the student's own moves, or a little better. So the picture's job is to say where the ball is, precisely, and a policy told the ball's place can do the rest. That is also easier to test on real pictures, as here, than a policy whose only output is motion.
11. **Precision is the target.** A few millimetres, not a centimetre: the teacher falls from 97.5% to 77.5% when its ball position is the student's estimate, and the student lifts 12 points less when the camera is 2.5 times further from its calibration.
12. **Do not refuse on the estimate's edge.** The band the arm lifts from was applied to the student's own distance, and cost 13% of attempts in refusals of balls that were inside it. Train out to further than the band, or widen what is accepted by the estimate's error.
13. **Let the student drive in training.** Copying recorded attempts gave 0 of 61. Train its outputs as the share of the speed-limited step actually taken; the teacher's raw outputs are far past the limits and nearly all saturated.
14. **Two traps in building the pictures.** Things parked far away to hide them stretched the scene's size and pushed the renderer's near limit out to 15 cm, so the picture went black as the gripper came down. And the camera passed through things on the desk that the arm does not collide with. Both look like a learning failure and are not.

## Still unknown

- **Whether putting the ball's colour right makes the student recognise the real ball.** The repaint test says it should. A student trained on simulated pictures with the ball in the real camera's colour, and two given a filtered picture instead of the colour one, are being trained; their results will be added here.
- **Why this run learnt more slowly than the one before it.** One run of each.
- **Anything about the real arm in motion:** the picture's real delay, the rolling shutter, blur that depends on distance (the lens is fixed-focus and near), and whether the student closes the loop on real pictures at all. Every real picture here was taken with the arm still.
- **One desk, one room, one ball.** All 65 real pictures.
- **How right the arm's joint zeros are with the arm raised.** The camera's calibration inherits them.

## How the student's pictures differ from real ones

Alike, by measurement: where the ball appears and how large; haze; glare; clutter. Different, known: the ball's colour (above); no dimples on the ball; one fixed blur for the whole picture where the real lens blurs by distance; no rolling shutter; the arm, its shadow and the gripper's fingers are the model's shapes with a plain shade, not the real printed parts and their camera cable; the desk is a flat pattern, not a carpet with depth. Different, unknown: anything that shows only when the arm moves.

## Is stage B worth paying for?

On this evidence, not yet for the reason first given. Stage B was proposed for a ray-traced camera, on the expectation that picture quality would be the gap. The gap that was found is one colour, and it was found and can be closed for nothing. Geometry, clutter and glare already carry over from a plain renderer.

What stage B would buy is different: a teacher trained with the real arm's speed limits (item 7), which the Mac cannot train; and the same picture work done at a scale where precision of a few millimetres is realistic. Before paying for either, three free things should be settled: whether the colour fix makes a student recognise the real ball (running now); whether the camera's exposure can be fixed (at the arm, with the author); and a clean real check of the arm's own accuracy with the camera, by finding a ball and touching it.

## Reproducing

```sh
pip install -r requirements-dev.txt -r requirements-train.txt
# Train: about 95 minutes on an Apple-silicon Mac. --camera is a calibrated camera file; without it a rough built-in one is used.
python scripts/train_student.py --camera camera.local.json --data /tmp/student-data --out out --rounds 8
# In simulation
python scripts/evaluate_student.py --student out/student.pt --rows teacher student eye-and-hand blank-0.5s --output-dir out
python scripts/evaluate_search.py --student out/student.pt --part student --output out/search-student.json
python scripts/evaluate_student_carry.py --student out/student.pt --part student --output out/whole-job-student.json
# On real pictures (they are not in this repository)
python scripts/student_on_real_pictures.py --student out/student.pt --sightings sightings.local.json \
    --calibration calibration.local.json --shots FOLDER
python scripts/student_real_checks.py all --student out/student.pt --samples FOLDER --calibrated cal4
```

## Contents

| File | What it is |
| --- | --- |
| `student.pt` | The trained student (PyTorch, 0.9 MB). It records the camera it was trained for |
| `training.jsonl` | Each round's attempts, lifts and losses |
| `mujoco-*.json` | Balls ahead of the arm: teacher, student, the student's eye with the teacher's hand, the picture blanked, the camera further out |
| `search-*.json` | Searching the whole sector, to 85 degrees, with no ball, and with balls out of reach |
| `whole-job-*.json` | Find, pick up, carry to a bowl and drop in |
| `real-pictures-*.json` | The three checks on real pictures: every picture, the ball's colour, one change at a time |

The real pictures themselves are not in this repository.
