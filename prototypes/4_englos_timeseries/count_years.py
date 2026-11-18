"""Count cars at Auchan Englos in every IGN photo.
Usage: python count_years.py dota
       python count_years.py own ../3_own_model/own_cowc_yolo11n.pt 0.2
All photos cover the same 700 m x 700 m box, saved at 2576 px (0.27 m per pixel).
"""
import sys, json, os
import numpy as np
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "2_car_counter"))
from ultralytics import YOLO
from carcount import load_rgb, detect_cars

GSD = 700 / 2576
which = sys.argv[1]
if which == "own":
    model = YOLO(sys.argv[2]); kw = dict(scale=GSD / 0.10, conf=float(sys.argv[3]))
else:
    model = YOLO("yolo11m-obb.pt")  # downloads the free weights on first use
    kw = dict(scale=5.0, sharpen=True, conf=0.2)

# Auchan's own car parks, as fractions of the image (same box every year)
ZONES = [(0.50, 0.00, 0.685, 0.30), (0.66, 0.30, 1.00, 0.53)]
# Flight dates from IGN's mosaic layer (candidates for this area)
DATES = {2004: "2004 (day not published)", 2009: "2009 (day not published)",
         2015: "11 Jun (Thu), 30 Jun (Tue), 1 Jul (Wed) or 10 Jul (Fri) 2015",
         2018: "8 May (Tue, public holiday), 29 Jun (Fri) or 26 Jul (Thu) 2018",
         2021: "1 Jun (Tue), 7 Jun (Mon), 9 Jun (Wed) or 13 Jun (Sun) 2021",
         2025: "20 May 2025 (Tue)"}
RES = {2004: 50, 2009: 50, 2015: 20, 2018: 20, 2021: 20, 2025: 20}


def in_zone(p, W, H):
    x, y = p[0] / W, p[1] / H
    return any(a <= x < c and b <= y < d for a, b, c, d in ZONES)


out = {}
os.makedirs("out", exist_ok=True)
for year in sorted(RES):
    f = os.path.join(HERE, "..", "..", "data", "ign_englos", f"englos_{year}.jpg")
    if not os.path.exists(f):
        continue
    img = load_rgb(f); H, W = img.shape[:2]
    d = detect_cars(model, img, dedup_px=1.5 / GSD, **kw)
    z = int(sum(in_zone(p, W, H) for p in d))
    out[year] = dict(total=len(d), auchan=z, date=DATES[year], native_cm=RES[year])
    np.save(f"out/{which}_{year}.npy", d)
    im = Image.fromarray(img); dr = ImageDraw.Draw(im)
    for a, b, c, e in ZONES:
        dr.rectangle([a * W, b * H, c * W, e * H], outline=(235, 104, 52), width=5)
    for x, y, *_ in d:
        dr.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(0, 150, 255), width=2)
    im.resize((1200, 1200), Image.LANCZOS).save(f"out/{which}_{year}.jpg", quality=80)
    print(year, out[year], flush=True)
json.dump(out, open(f"out/series_{which}.json", "w"), indent=1)
