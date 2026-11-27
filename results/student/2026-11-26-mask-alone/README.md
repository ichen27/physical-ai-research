# A student given only the ball's colour as a mask, 2026-11-26

The third of three students trained the same way from the same seed on the same simulated pictures (the ball shown in the colour the real camera gives it), differing only in what they are shown of each picture:

- [the colour picture](../2026-11-26-real-ball-colour/README.md);
- [the filter's mask beside the picture in grey](../2026-11-26-mask-and-grey/README.md);
- the filter's mask alone, which is this one.

**Verdict: it recognises the real ball and lifts as well as the others in simulation, and it is the most easily fooled of the three.** It has also been run on the real arm, by the session working with the author, driving the arm itself from the live stream: one attempt, one pickup. One attempt is not a success rate.

## The three side by side

MuJoCo, the real arm's speed limits, looks not trained on, 200 attempts a row (150 for the searches with no ball); and the saved real pictures. One training run of each: two runs of one recipe have differed by 15 points before, so small differences between columns mean little.

| | Colour | Mask beside grey | Mask alone |
| --- | --- | --- | --- |
| Its own moves, balls ahead: lifted | 87.5% | 95.5% | 94.0% |
| Its eye asked every step, the teacher's hand: lifted | 80.5% | 85.5% | 77.5% |
| Searching 57 degrees either side: lifted, of all | 83.0% | 81.0% | 84.5% |
| Searching: refused | 6.5% | 14.0% | 9.5% |
| No ball, other things on the desk: said there was a ball | 21.3% | 29.3% | 44.7% |
| No ball: went on to a pickup | 2.7% | 7.3% | 12.7% |
| Real pictures: sure the ball is there (of 65) | 62 | 65 | 65 |
| Real pictures: error in the ball's place, median (the 19 with a known place) | 7.0 mm | 8.1 mm | 8.0 mm |
| Real pictures: worst | 11.9 mm | 24.0 mm | 19.9 mm |
| Real pictures: within 1 cm | 17 of 19 | 13 of 19 | 10 of 19 |

What the table supports:

- All three recognise the real ball, where the student before them was sure of it in 18 of 65. The simulated ball's colour was what they shared; the filter was not needed for that.
- The less a student is shown, the more easily something else is taken for the ball: with no ball on a cluttered desk, a pickup was attempted in 2.7%, 7.3% and 12.7% of searches.
- On pickup rate with their own moves the three cannot be told apart from one run each.

## A second round patch of the ball's colour

Shown only a mask, a second patch that is round, solid and the ball's size cannot be told from the ball, and this student's answer goes somewhere between the two with no loss of sureness.

In simulation (`../2026-11-26-mask-and-grey/wirings/second_orange_patch.py`; 300 looks from the look pose, the ball whole and near the middle of the picture, a bare desk, and three quarters of a ball-sized disc painted into the corner of the picture):

| Eye | Its answer moves, median | 90th percentile | Still sure |
| --- | --- | --- | --- |
| Colour | 3 to 4 mm | 10 to 13 mm | 100% |
| Mask beside grey | 4 to 5 mm | 8 to 49 mm | 100% |
| Mask alone | 70 mm | 120 mm | 100% |

That is the worst case and overstates what a real desk does. On six real look-pose pictures the same painted patch moved this student's answer by 3, 3, 3, 68, 7 and 60 mm: nothing in half of them, centimetres in the rest. A patch that was only a quarter of a disc moved it 2 to 27 mm, and a real orange bag lying in the corner of three of those pictures, which is neither round nor solid, moved it 0 to 6 mm when greyed out. The other two students moved by 1 to 11 mm in every case.

So: keep other round orange, red or yellow things out of the camera's view, with this student above all, and do not take its sureness as a sign that nothing else is in the picture. `--compare_finders` prints the hand-written finder's answer beside the eye's on every look, which shows a disagreement that the sureness does not.

## Contents

| File | What it is |
| --- | --- |
| `student.pt`, `training.jsonl` | The trained student and each round's attempts, lifts and losses |
| `eye.onnx`, `driver.onnx` and their `.json` | Its eye, and the whole student, exported for the arm tool |
| `real-pictures-all.json` | The check on every real picture |
| `mujoco-*.json`, `search-*.json` | Its measurements in simulation |

```sh
python scripts/train_student.py --camera camera.local.json --data /tmp/student-data --out out --rounds 8 --input mask
```

The real pictures are not in this repository.
