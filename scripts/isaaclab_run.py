"""Run one of Isaac Lab's reinforcement learning scripts with Understudy's tasks registered.

    python scripts/isaaclab_run.py train --task Understudy-Reach-OMX-v0 --headless
    python scripts/isaaclab_run.py play  --task Understudy-Reach-OMX-Play-v0

Run it with Isaac Lab's Python. Set ISAACLAB_PATH to the Isaac Lab checkout.
"""

import os
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "play"):
        sys.exit("usage: isaaclab_run.py {train|play} [Isaac Lab script arguments]")
    isaaclab = os.environ.get("ISAACLAB_PATH")
    if not isaaclab:
        sys.exit("set ISAACLAB_PATH to the Isaac Lab checkout")
    script = Path(isaaclab) / "scripts" / "reinforcement_learning" / "rsl_rl" / f"{sys.argv[1]}.py"
    if not script.is_file():
        sys.exit(f"not found: {script}")

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(script.parent))
    import understudy_isaac  # noqa: F401  (registers the task names)

    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
