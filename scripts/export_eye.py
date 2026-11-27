"""Export a trained student to ONNX, so that the computer at the arm needs no PyTorch.

    python scripts/export_eye.py --student results/student/<name>/student.pt --output results/student/<name>/eye.onnx \
        --driver results/student/<name>/driver.onnx

``--output`` is the student's eye alone: where the ball is from one still picture
(``understudy.eye.Eye``). ``--driver``, if given, is the whole student, with its memory and
its moves, one control step at a time (``understudy.eye.Driver``). Beside each a ``.json``
file is written that says what input it takes and what camera it was trained for. Each
export is checked against the student itself before it is kept.

Needs torch and onnxruntime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from understudy import eye, student  # noqa: E402


class _Driver(torch.nn.Module):
    def __init__(self, net: student.StudentNet):
        super().__init__()
        self.net = net

    def forward(self, shown: torch.Tensor, readings: torch.Tensor, guess: torch.Tensor, state: torch.Tensor):
        fused, seen = self.net.look_at(shown, readings)
        fused = torch.cat([fused, guess * student.BALL_SCALE], dim=1)[:, None]
        remembered, state = self.net.memory(fused, state)
        out = self.net.act(torch.cat([remembered, fused], dim=2))[:, 0]
        return torch.tanh(out[:, :5]), out[:, 5], out[:, 6:9] / student.BALL_SCALE, seen, state


class _Eye(torch.nn.Module):
    def __init__(self, net: student.StudentNet):
        super().__init__()
        self.net = net

    def forward(self, shown: torch.Tensor, readings: torch.Tensor) -> torch.Tensor:
        return self.net.look_at(shown, readings)[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--student", required=True)
    parser.add_argument("--output", required=True, help="The eye's .onnx file to write; a .json file is written beside it.")
    parser.add_argument("--driver", help="Also write the whole student, memory and moves and all, to this .onnx file.")
    args = parser.parse_args()
    net, notes = student.load(args.student)
    out = Path(args.output)
    layers = eye.INPUTS[net.input]
    said = {"input": net.input, "picture": [eye.PICTURE_WIDTH, eye.PICTURE_HEIGHT], "exported_from": Path(args.student).name,
            "student_training": notes}
    example = (torch.zeros(1, layers, eye.PICTURE_HEIGHT, eye.PICTURE_WIDTH), torch.zeros(1, eye.PROPRIO_SIZE))
    torch.onnx.export(_Eye(net).eval(), example, str(out), input_names=["shown", "readings"], output_names=["seen"],
                      dynamic_axes={"shown": {0: "pictures"}, "readings": {0: "pictures"}, "seen": {0: "pictures"}},
                      opset_version=17, dynamo=False)
    out.with_suffix(".json").write_text(json.dumps({
        **said, "output": "x, y, z of the ball in metres, then how sure the eye is that the ball is in the picture, before the sigmoid",
    }, indent=2) + "\n")

    # The exported eye must say what the student says.
    rng = np.random.default_rng(0)
    exported = eye.Eye(out)
    brain = student.Brain(net)
    worst = 0.0
    for _ in range(20):
        picture = rng.integers(0, 256, size=(eye.PICTURE_HEIGHT, eye.PICTURE_WIDTH, 3), dtype=np.uint8)
        picture[20:50, 40:70] = (40, 120, 240)  # something the ball's colour, so that a filtered eye has something to see
        angles = [float(v) for v in rng.uniform(-0.5, 0.5, size=5)]
        ours, sure = exported.glance(picture, angles)
        theirs, their_sure = brain.glance(picture, eye.readings_for(angles))
        worst = max(worst, abs(sure - their_sure), *(abs(a - b) for a, b in zip(ours, theirs)))
    if worst > 1e-4:
        out.unlink()
        raise SystemExit(f"the exported eye differs from the student by {worst:g}; nothing was kept")
    print(f"wrote {out} and {out.with_suffix('.json').name}: input {net.input}, {out.stat().st_size / 1024:.0f} KB; "
          f"it agrees with the student to {worst:.1e} on 20 pictures")
    if not args.driver:
        return

    whole = Path(args.driver)
    example = (*example, torch.zeros(1, eye.GUESS_SIZE), torch.zeros(1, 1, net.width))
    names = ["shown", "readings", "guess", "state"]
    torch.onnx.export(_Driver(net).eval(), example, str(whole), input_names=names,
                      output_names=["moves", "open", "ball", "seen", "state_after"], opset_version=17, dynamo=False)
    whole.with_suffix(".json").write_text(json.dumps({
        **said, "memory": net.width,
        "output": "each arm joint's move as a share of the speed-limited step; the gripper (open at or above zero); the ball from "
                  "memory, metres; the ball from this picture and how sure, as the eye gives them; the memory to pass back",
        "speed_limit_rad_s": eye.SPEED_LIMIT_RAD_S, "lead_limit_rad": eye.LEAD_LIMIT_RAD,
    }, indent=2) + "\n")
    # Over a run of steps, memory and all, the exported student must do what the student does.
    exported, brain, worst = eye.Driver(whole), student.Brain(net), 0.0
    brain.reset()
    for step in range(60):
        picture = rng.integers(0, 256, size=(eye.PICTURE_HEIGHT, eye.PICTURE_WIDTH, 3), dtype=np.uint8)
        picture[20:50, 40 + step % 20 : 70 + step % 20] = (40, 120, 240)
        readings = rng.uniform(-0.3, 0.3, size=eye.PROPRIO_SIZE).astype(np.float32)
        ours, theirs = exported.step(picture, readings, (0.2, 0.03)), brain.step(picture, readings, (0.2, 0.03))
        worst = max(worst, float(np.abs(ours["moves"] - theirs["moves"]).max()), abs(ours["in_view"] - theirs["in_view"]),
                    *(abs(a - b) for a, b in zip(ours["ball"], theirs["ball"])), float(ours["open"] != theirs["open"]))
    if worst > 1e-4:
        whole.unlink()
        raise SystemExit(f"the exported student differs from the student by {worst:g} over 60 steps; it was not kept")
    print(f"wrote {whole} and {whole.with_suffix('.json').name}: {whole.stat().st_size / 1024:.0f} KB; "
          f"it agrees with the student to {worst:.1e} over 60 steps in a row")


if __name__ == "__main__":
    main()
