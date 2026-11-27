"""Export a trained student's eye to ONNX, so that the computer at the arm needs no PyTorch.

    python scripts/export_eye.py --student results/student/<name>/student.pt --output results/student/<name>/eye.onnx

Writes ``eye.onnx`` and, beside it, ``eye.json``, which says what input the eye takes and
what camera it was trained for. ``understudy.eye.Eye`` runs it. The export is checked
against the student itself on random pictures before it is kept.

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


class _Eye(torch.nn.Module):
    def __init__(self, net: student.StudentNet):
        super().__init__()
        self.net = net

    def forward(self, shown: torch.Tensor, readings: torch.Tensor) -> torch.Tensor:
        return self.net.look_at(shown, readings)[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--student", required=True)
    parser.add_argument("--output", required=True, help="The .onnx file to write; a .json file is written beside it.")
    args = parser.parse_args()
    net, notes = student.load(args.student)
    out = Path(args.output)
    layers = eye.INPUTS[net.input]
    example = (torch.zeros(1, layers, eye.PICTURE_HEIGHT, eye.PICTURE_WIDTH), torch.zeros(1, eye.PROPRIO_SIZE))
    torch.onnx.export(_Eye(net).eval(), example, str(out), input_names=["shown", "readings"], output_names=["seen"],
                      dynamic_axes={"shown": {0: "pictures"}, "readings": {0: "pictures"}, "seen": {0: "pictures"}},
                      opset_version=17, dynamo=False)
    out.with_suffix(".json").write_text(json.dumps({
        "input": net.input,
        "picture": [eye.PICTURE_WIDTH, eye.PICTURE_HEIGHT],
        "output": "x, y, z of the ball in metres, then how sure the eye is that the ball is in the picture, before the sigmoid",
        "exported_from": Path(args.student).name,
        "student_training": notes,
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


if __name__ == "__main__":
    main()
