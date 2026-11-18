"""Compare our own COWC-trained model with the ready-made DOTA model.
1. Pick our model's confidence threshold on the validation tile (6_8) only.
2. Test both models on the held-out tile 2_12 (never used for training or tuning).
3. Run both on the Englos hand-check box and save panels for checking by eye.
"""
import json, sys
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
from carcount import load_rgb, detect_cars, gt_points, match

COWC_GSD = 300 / 2220
IGN_GSD = 700 / 2576
R = 1.5
own = YOLO(sys.argv[1] if len(sys.argv) > 1 else "runs/detect/runs/cowc_n/weights/best.pt")
dota = YOLO("w/yolo11m-obb.pt")
OWN_SCALE_COWC = COWC_GSD / 0.10        # our model works at 0.10 m per pixel
res = {}

# 1. threshold on validation tile
img = load_rgb("cowc/top_potsdam_6_8_RGB.png"); gt = gt_points("cowc/top_potsdam_6_8_RGB_Annotated_Cars.png")
d = detect_cars(own, img, scale=OWN_SCALE_COWC, conf=0.05, dedup_px=R / COWC_GSD)
best = None
for c in np.arange(0.1, 0.75, 0.05):
    dd = d[d[:, 2] >= c]; tp, fp, fn = match(dd[:, :2], gt, R / COWC_GSD)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    print(f"val conf {c:.2f}: tp {tp} fp {fp} fn {fn} f1 {f1:.3f}")
    if best is None or f1 > best[1]:
        best = (float(c), f1)
CONF = best[0]; res["own_conf"] = CONF; print("chosen conf", CONF)

# 2. held-out test tile
img = load_rgb("data/top_potsdam_2_12_RGB.png"); gt = gt_points("data/top_potsdam_2_12_RGB_Annotated_Cars.png")
for name, m, kw in [("own", own, dict(scale=OWN_SCALE_COWC, conf=CONF)), ("dota", dota, dict(scale=1.0, conf=0.25))]:
    d = detect_cars(m, img, dedup_px=R / COWC_GSD, **kw)
    tp, fp, fn = match(d[:, :2], gt, R / COWC_GSD)
    res[f"test_{name}"] = dict(true=len(gt), detected=len(d), tp=tp, fp=fp, fn=fn)
    print("TEST 2_12", name, res[f"test_{name}"])
    np.save(f"out/test212_{name}.npy", d)

# 3. Englos check box (same box as the earlier hand check)
ign = load_rgb("data/auchan_englos.jpg")
crop = ign[700:1000, 1300:1700]
for name, m, kw in [("own", own, dict(scale=IGN_GSD / 0.10, conf=CONF)), ("dota", dota, dict(scale=5.0, sharpen=True, conf=0.2))]:
    d = detect_cars(m, crop, dedup_px=1.5 / IGN_GSD, **kw)
    res[f"ign_box_{name}"] = len(d)
    np.save(f"out/ignbox_{name}.npy", d)
    k = 0
    for y0 in (0, 150):
        for x0 in (0, 200):
            c = Image.fromarray(crop[y0:y0 + 150, x0:x0 + 200]).resize((1000, 750), Image.LANCZOS); dr = ImageDraw.Draw(c)
            for x, y, *_ in d:
                if x0 <= x < x0 + 200 and y0 <= y < y0 + 150:
                    X, Y = (x - x0) * 5, (y - y0) * 5; dr.ellipse([X - 16, Y - 16, X + 16, Y + 16], outline=(0, 140, 255), width=3)
            c.save(f"out/check_{name}_{k}.png"); k += 1
    print("IGN box", name, len(d))
json.dump(res, open("out/own_vs_dota.json", "w"), indent=1)
