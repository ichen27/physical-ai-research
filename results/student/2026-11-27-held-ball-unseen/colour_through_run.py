"""The ball's colour, the background's colour and the filter's share of the picture through one recorded run of
the student on the real arm, from the pictures the tool saves beside the record (``<log>.pictures.npz``, which
is not in the repository: the pictures are the real camera's).

    colour_through_run.py RECORD.json RECORD.pictures.npz OUTPUT.json
"""
import colorsys, dataclasses, json, sys
import numpy as np
sys.path.insert(0, ".")
from understudy import eye, vision

steps = json.load(open(sys.argv[1]))["steps"]
pictures = np.load(sys.argv[2])["pictures"]          # blue, green, red, at the student's size
wider = dataclasses.replace(eye.FILTER, hue_low=-20.0)
rows = []
for i in range(0, min(len(steps), len(pictures)), 10):
    bgr = pictures[i]; rgb = bgr[..., ::-1].astype(float); r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    ballish = (r >= g) & (g >= b - 10) & (r > b + 25) & (rgb.max(-1) > 80)       # loosely: reddish, whatever the tint
    if ballish.sum() < 20 or (~ballish).sum() < 20:
        continue
    ball, rest = rgb[ballish].mean(0), rgb[~ballish].mean(0)
    hue, saturation, _ = vision.colour_parts(bgr, eye.FILTER)
    rows.append({
        "seconds": round(i / 30, 2), "gripper_open": bool(steps[i]["gripper_open"]), "tip_height_cm": round(steps[i]["tip"][2] * 100, 1),
        "sure": steps[i]["sure"], "filter_share": round(float(vision.orange_mask(bgr, eye.FILTER).mean()), 3),
        "filter_share_hue_from_minus_20": round(float(vision.orange_mask(bgr, wider).mean()), 3),
        "ballish_share": round(float(ballish.mean()), 3), "ball_rgb": [round(v) for v in ball], "background_rgb": [round(v) for v in rest],
        "background_saturation": round(colorsys.rgb_to_hsv(*(rest / 255))[1], 3),
        "ball_red_at_ceiling": round(float(np.mean(r[ballish] >= 250)), 2),
        "ball_hue_as_the_filter_measures_it": round(float(np.median(hue[ballish])), 1),
        "ball_saturation_as_the_filter_measures_it": round(float(np.median(saturation[ballish])), 2),
    })
json.dump({"record": sys.argv[1].split("/")[-1], "filter": dataclasses.asdict(eye.FILTER), "every_tenth_step": rows}, open(sys.argv[3], "w"), indent=1)
for row in rows:
    print(f"{row['seconds']:5.1f} s  tip {row['tip_height_cm']:5.1f} cm  sure {row['sure']:.2f}  filter {row['filter_share']:.2f} (wider {row['filter_share_hue_from_minus_20']:.2f})  "
          f"ball {row['ball_rgb']} hue {row['ball_hue_as_the_filter_measures_it']:5.1f}  background {row['background_rgb']} sat {row['background_saturation']:.2f}")
