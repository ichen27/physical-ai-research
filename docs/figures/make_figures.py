"""Draws the charts in the README from the numbers reported under results/.

    python docs/figures/make_figures.py

Each chart is written twice, for light and dark pages (NAME.png and NAME-dark.png).
Numbers that come from a results table are typed in below with the file they are from;
the real-arm charts are read from results/real-arm/2026-11-27-first-vision-runs/.
Needs matplotlib and numpy.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REAL = REPO / "results" / "real-arm" / "2026-11-27-first-vision-runs"

# Colours by job. Series colours are a fixed order checked for colour-blind separation;
# status colours are only ever used for an outcome and always come with a label or a marker shape.
THEMES = {
    "": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", grid="#e1e0d9", axis="#c3c2b7",
             series=["#2a78d6", "#eb6834", "#1baf7a"], neutral="#8a8982", good="#0ca30c", bad="#d03b3b", band="#2a78d6"),
    "-dark": dict(surface="#1a1a19", text="#f0efec", muted="#c3c2b7", grid="#2c2c2a", axis="#4a4a46",
                  series=["#3987e5", "#d95926", "#199e70"], neutral="#8a8982", good="#0ca30c", bad="#e66767", band="#3987e5"),
}


def style(t: dict) -> None:
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["text"], "axes.labelcolor": t["muted"], "xtick.color": t["muted"], "ytick.color": t["muted"],
        "axes.edgecolor": t["axis"], "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": t["grid"], "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 12, "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "figure.dpi": 100,
    })


def bars(ax, labels, values, colours, t, fmt="{:.1f}%", ymax=100.0, width=0.62):
    x = np.arange(len(labels))
    made = ax.bar(x, values, width=width, color=colours, linewidth=0)
    for rect, value in zip(made, values):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + ymax * 0.015, fmt.format(value),
                ha="center", va="bottom", fontsize=10, color=t["text"])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, ymax * 1.12)
    return made


def save(fig, name: str, suffix: str) -> None:
    fig.savefig(HERE / f"{name}{suffix}.png", dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def teacher_versions(t, suffix):
    # results/pickup/2026-11-24-varied/README.md and 2026-11-24-claw/README.md, MuJoCo, vendor settings, 200 attempts.
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    groups = ["Second policy\n(solid wedge fingers)", "Third policy\n(claw-shaped fingers)"]
    lifted, kept = [71.5, 100.0], [3.5, 96.0]
    x = np.arange(2)
    for offset, values, colour, label in ((-0.19, lifted, t["series"][0], "Lifted at some point"),
                                          (0.19, kept, t["series"][1], "Still holding at the end")):
        made = ax.bar(x + offset, values, width=0.34, color=colour, linewidth=0, label=label)
        for rect, value in zip(made, values):
            ax.text(rect.get_x() + rect.get_width() / 2, value + 1.5, f"{value:g}%", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Share of 200 attempts")
    ax.set_title("The same pickup, checked in a second simulator (MuJoCo)")
    ax.legend(loc="upper center", ncols=2, bbox_to_anchor=(0.5, -0.2))
    save(fig, "teacher-second-simulator", suffix)


def gentle_and_facing(t, suffix):
    # results/camera/2026-11-25-wrist-camera/README.md: mujoco-fast-tracked, gentle-tracked-not-faced, gentle-tracked-faced, gentle-around-tracked.
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    labels = ["Full speed", "Speed limits on", "Limits + turn\nto face the ball", "Limits + facing,\nball anywhere around"]
    bars(ax, labels, [99.0, 61.5, 91.5, 99.0], [t["neutral"], t["series"][0], t["series"][0], t["series"][0]], t)
    ax.set_ylabel("Still holding at the end")
    ax.set_title("Slowing the arm down broke the pickup; turning to face the ball fixed it")
    save(fig, "speed-limits-and-facing", suffix)


def reach_by_distance(t, suffix):
    # results/camera/2026-11-25-wrist-camera/README.md, "How far away the ball can be", 40 attempts per centimetre.
    distance = np.arange(14, 29)
    lifted = [52.5, 100, 100, 100, 97.5, 97.5, 100, 100, 100, 92.5, 77.5, 40, 0, 0, 0]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.axvspan(15, 24, color=t["band"], alpha=0.10, linewidth=0)
    ax.text(19.5, 108, "the arm only attempts balls in this band", ha="center", va="center", fontsize=9.5, color=t["muted"])
    ax.plot(distance + 0.5, lifted, color=t["series"][0], linewidth=2, marker="o", markersize=6,
            markeredgecolor=t["surface"], markeredgewidth=1.5)
    ax.set_xticks(np.arange(14, 30, 2))
    ax.set_xlim(13.5, 29.5)
    ax.set_ylim(-4, 116)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Ball's distance from the arm's swivel axis (cm)")
    ax.set_title("How far away the ball can be (simulation, speed limits on)")
    save(fig, "reach-by-distance", suffix)


def real_pictures(t, suffix):
    # results/student/2026-11-26-mask-alone/README.md and 2026-11-26-real-ball-colour/README.md; 65 real pictures, 19 with a known place.
    labels = ["First\ncolour", "Stage A\ncolour", "Colour, ball\nrecoloured", "Mask +\ngrey", "Mask\nalone"]
    sure, error = [3, 18, 62, 65, 65], [36.3, 9.8, 7.0, 8.1, 8.0]
    colours = [t["neutral"], t["neutral"], t["series"][0], t["series"][1], t["series"][2]]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.6, 3.7))
    bars(left, labels, sure, colours, t, fmt="{:.0f}", ymax=65)
    left.set_ylabel("Pictures (of 65)")
    left.set_title("Sure the real ball is there")
    bars(right, labels, error, colours, t, fmt="{:.1f}", ymax=40)
    right.set_ylabel("Millimetres")
    right.set_title("Error in the ball's position (median)")
    for ax in (left, right):
        ax.tick_params(axis="x", labelsize=9)
    fig.subplots_adjust(wspace=0.22)
    save(fig, "real-pictures", suffix)


def three_students(t, suffix):
    # results/student/2026-11-26-mask-alone/README.md, "The three side by side". 200 attempts a row, 150 for no-ball searches.
    names = ["Colour", "Mask +\ngrey", "Mask\nalone"]
    panels = [("Own moves, ball ahead:\nlifted", [87.5, 95.5, 94.0], 100), ("Searching a wide sector:\nlifted, of all balls", [83.0, 81.0, 84.5], 100),
              ("No ball, cluttered desk:\nwent for something else", [2.7, 7.3, 12.7], 100)]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.8), sharey=True)
    for ax, (title, values, ymax) in zip(axes, panels):
        bars(ax, names, values, t["series"], t, ymax=ymax)
        ax.set_title(title, fontsize=11)
        ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    fig.subplots_adjust(wspace=0.08, top=0.80)
    save(fig, "three-students", suffix)


def lateness(t, suffix):
    # results/student/2026-11-26-mask-and-grey/README.md, "The two ways of wiring it at the arm", the 15 to 24 cm band.
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    bars(ax, ["On time", "100 ms late", "200 ms late", "300 ms late"], [95.5, 92.9, 87.7, 50.3], [t["series"][0]] * 4, t)
    line = ax.axhline(78.0, color=t["series"][1], linewidth=2, linestyle=(0, (5, 3)))
    ax.legend([line], ["For comparison: one still look, then the teacher does the pickup (78.0%)"], loc="upper center",
              bbox_to_anchor=(0.5, -0.24))
    ax.set_xlabel("How late each picture reaches the student")
    ax.set_ylabel("Lifted")
    ax.set_title("The student driving itself, by picture lateness (simulation)")
    save(fig, "picture-lateness", suffix)


def real_arm_runs(t, suffix):
    runs = json.loads((REAL / "runs.json").read_text())
    rows = ["position typed in", "eye feeding the teacher", "student driving"]
    names = ["Position typed in\n(no camera)", "Trained eye, one look,\nthen the teacher", "Student drives itself\nfrom live video"]
    looks = {"picked up": dict(marker="o", color=t["good"], label="Picked up"),
             "closed on nothing": dict(marker="X", color=t["bad"], label="Closed on nothing"),
             "other": dict(marker="s", color=t["neutral"], label="Refused or stopped safely")}
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.axvspan(15, 24, color=t["band"], alpha=0.10, linewidth=0)
    seen, stack = set(), {}
    for run in runs:
        if run["ball_distance_cm"] is None:
            continue
        kind = run["outcome"] if run["outcome"] in looks else "other"
        y = rows.index(run["wiring"])
        key = (y, round(run["ball_distance_cm"], 1))
        stack[key] = stack.get(key, 0) + 1
        ax.scatter(run["ball_distance_cm"], y + 0.16 * (stack[key] - 1), s=110, marker=looks[kind]["marker"], color=looks[kind]["color"],
                   edgecolors=t["surface"], linewidths=1.5, zorder=3, label=None if kind in seen else looks[kind]["label"])
        seen.add(kind)
    ax.set_yticks(range(3), names)
    ax.set_ylim(-0.6, 2.75)
    ax.set_xlim(14, 27)
    ax.set_xlabel("Where the ball was, by the arm's own estimate (cm from the swivel axis)")
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_title("Every real-arm run with a ball in view, 25 to 27 November 2026")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncols=3, handletextpad=0.3, columnspacing=1.6)
    ax.text(19.5, 2.6, "reach band, 15 to 24 cm", ha="center", va="center", fontsize=9.5, color=t["muted"])
    save(fig, "real-arm-runs", suffix)


def real_live_run(t, suffix):
    record = json.loads((REAL / "records" / "live-run-20261127-011102.json").read_text())
    steps, hz = record["steps"], record["control_hz"]
    seconds = np.arange(len(steps)) / hz
    closed = next(i for i, step in enumerate(steps) if not step["gripper_open"]) / hz
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.8, 4.6), sharex=True)
    top.plot(seconds, [100 * step["tip"][2] for step in steps], color=t["series"][0], linewidth=2)
    top.axhline(0, color=t["axis"], linewidth=1)
    top.set_ylabel("Gripper tip height (cm)")
    top.set_title("First real pickup driven by the trained student (27 Nov, ball 22 cm away)")
    bottom.plot(seconds, [step["angles"][5] for step in steps], color=t["series"][1], linewidth=2)
    bottom.axhline(0.07, color=t["muted"], linewidth=1, linestyle=(0, (4, 3)))
    bottom.text(14.9, 0.26, "dashed line: above it with the gripper closed means something is held", ha="right", va="bottom",
                fontsize=9, color=t["muted"])
    bottom.set_ylabel("Finger angle (rad)")
    bottom.set_xlabel("Seconds since the pickup began")
    for ax in (top, bottom):
        ax.axvline(closed, color=t["muted"], linewidth=1, linestyle=":")
    top.text(closed + 0.15, top.get_ylim()[1] * 0.92, f"gripper closes at {closed:.1f} s", fontsize=9, color=t["muted"], va="top")
    fig.subplots_adjust(hspace=0.12)
    save(fig, "real-live-run", suffix)


def pipeline(t, suffix):
    fig, ax = plt.subplots(figsize=(11.6, 3.5))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    boxes = [
        (1, "1. Teacher", "Isaac Lab, rented GPU\n4,096 simulated arms\nReinforcement learning\nTold where the ball is", t["series"][0]),
        (26, "2. Cross-check", "MuJoCo, on a Mac\nSecond physics engine\nSame code as the real arm\nFound three hidden bugs", t["neutral"]),
        (51, "3. Student", "MuJoCo, on a Mac\nSees the wrist camera\nCopies the teacher while\ndriving (DAgger)", t["series"][1]),
        (76, "4. Real arm", "OMX arm + wrist camera\nCalibrated by touch\nSpeed limits, stop rules\nPicks up the ball", t["series"][2]),
    ]
    for x, title, body, colour in boxes:
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, 4), 22, 22, boxstyle="round,pad=0.2,rounding_size=1.2",
                                                       facecolor=colour, edgecolor=colour, linewidth=2))
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x + 0.25, 4.25), 21.5, 16.6, boxstyle="round,pad=0.0,rounding_size=0.9",
                                                       facecolor=t["surface"], edgecolor="none"))
        ax.text(x + 11, 23.6, title, ha="center", va="center", fontsize=12, fontweight="bold", color="#ffffff")
        ax.text(x + 11, 12.6, body, ha="center", va="center", fontsize=9.8, color=t["text"], linespacing=1.55)
    for x, label in ((23.4, "policy\nfile"), (48.4, "teacher's\nanswers"), (73.4, "model\nfiles")):
        ax.annotate("", xy=(x + 2.5, 15), xytext=(x - 0.2, 15), arrowprops=dict(arrowstyle="-|>", color=t["muted"], linewidth=1.6))
    save(fig, "pipeline", suffix)


def main() -> None:
    for suffix, theme in THEMES.items():
        style(theme)
        for draw in (pipeline, teacher_versions, gentle_and_facing, reach_by_distance, real_pictures, three_students, lateness,
                     real_arm_runs, real_live_run):
            draw(theme, suffix)
    print("charts written to", HERE)


if __name__ == "__main__":
    main()
