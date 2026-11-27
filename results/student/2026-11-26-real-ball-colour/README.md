# The colour-picture student with the simulated ball shown as the real camera shows it, 2026-11-26

The [stage A](../2026-11-26-stage-a/README.md) student retrained, same recipe and seed, with one change to its simulated pictures: in four looks out of five the ball is shown in the colour the real camera gives it (red at the top of the scale, green about 187, blue about 133), over a spread about three times what the real pictures showed, and a fifth of the other things on the desk are given that look too. The student still sees the colour picture; nothing is filtered.

The colour was set from the first real session's 21 pictures only. The other two sessions, 44 pictures including all 19 with a known ball place, were held out.

**Verdict: putting the ball's colour right is enough by itself.** On saved real pictures and in simulation only; **this student has not run on the real arm.**

## On real pictures

| Student | Sure the ball is there (65 pictures) | Sure (the 19 with a known place) | Error in the ball's place, median | Worst | Within 1 cm |
| --- | --- | --- | --- | --- | --- |
| Colour, first simulated pictures | 3 of 65 | 0 of 19 | 36.3 mm | 89.7 mm | 1 of 19 |
| Colour, stage A (haze before exposure) | 18 of 65 | 3 of 19 | 9.8 mm | 44.7 mm | 10 of 19 |
| **Colour, ball as the real camera shows it (this one)** | **62 of 65** | **18 of 19** | **7.0 mm** | **11.9 mm** | **17 of 19** |
| [Mask beside grey](../2026-11-26-mask-and-grey/README.md) | 65 of 65 | 19 of 19 | 8.1 mm | 24.0 mm | 13 of 19 |

By session: 21 of 21 in the one the colour was set from, and 36 of 39 and 5 of 5 in the two held out. `real-pictures-all.json`.

With its simulated pictures changed, 23% of simulated balls now have the real ball's hue, saturation and brightness together, where 5% did (`real-pictures-colour.json`). And the earlier test turns round: repainting the real ball a plain orange now makes this student a little less sure (16 of 19), where it made the stage A student sure (`real-pictures-changes.json`).

So the gap that stage A found closes in two ways, each for nothing: by matching the ball's colour in the simulated pictures, as here, or by filtering the colour away. Neither needed a better renderer.

## In simulation

MuJoCo, the real arm's speed limits, looks not trained on, 200 attempts a row (150 for the searches with no ball).

| | Stage A | This one | Mask beside grey |
| --- | --- | --- | --- |
| Its own moves, balls ahead: lifted | 72.5% | 87.5% | 95.5% |
| Its eye asked every step, the teacher's hand: lifted | 77.5% | 80.5% | 85.5% |
| Searching 57 degrees either side: lifted, of all | 69.5% | 83.0% | 81.0% |
| Searching: refused | 13.0% | 6.5% | 14.0% |
| No ball, other things on the desk: said there was a ball | 10.7% | 21.3% | 29.3% |
| No ball: went on to a pickup | 4.0% | 2.7% | 7.3% |

One training run of each, and runs of the same recipe have differed by this much before, so the order of the three on pickup rate is not settled. What the table does support: both changes help, and the student shown colour goes on to fewer false pickups than the one shown only a mask.

## Contents

| File | What it is |
| --- | --- |
| `student.pt`, `training.jsonl` | The trained student and each round's attempts, lifts and losses |
| `eye.onnx`, `driver.onnx` and their `.json` | Its eye, and the whole student, exported for the arm tool (`--eye`, `--driver`) |
| `real-pictures-*.json` | The three checks on real pictures |
| `mujoco-*.json`, `search-*.json` | Its measurements in simulation |

```sh
python scripts/train_student.py --camera camera.local.json --data /tmp/student-data --out out --rounds 8
python scripts/student_real_checks.py all --student out/student.pt --samples FOLDER --calibrated cal4 --sets cal2 cal4 find1
```

The real pictures are not in this repository.
