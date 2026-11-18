"""Step 1: measure the detector on COWC (hand-labelled cars) and pick the best image scale.
Step 2: count cars in the IGN photo of Auchan Englos with that scale.
Usage: python run_eval.py <folder with COWC pairs and auchan_englos.jpg> <out folder>
"""
import sys, glob, os, json, time
import numpy as np
from ultralytics import YOLO
from carcount import load_rgb, detect_cars, gt_points, match, draw

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
model = YOLO(os.path.join(os.path.dirname(__file__), "w/yolo11m-obb.pt"))

COWC_GSD, IGN_GSD = 0.15, 0.20          # metres per pixel
RADIUS_M = 1.5                           # a detection counts if within 1.5 m of a labelled car centre
SCALES = [0.75, 1.0, 1.5, 2.0]

pairs = []
for a in sorted(glob.glob(os.path.join(src, "*_Annotated_Cars.png"))):
    im = a.replace("_Annotated_Cars.png", ".png")
    if os.path.exists(im):
        pairs.append((im, a))
print("COWC pairs:", [os.path.basename(p[0]) for p in pairs])

res = {"cowc": [], "scales": {}}
cache = {}
for s in SCALES:
    TP = FP = FN = 0
    for im_p, an_p in pairs:
        img = cache.setdefault(im_p, load_rgb(im_p))
        gt = gt_points(an_p)
        t = time.time()
        d = detect_cars(model, img, scale=s)
        tp, fp, fn = match(d[:, :2], gt, RADIUS_M / COWC_GSD)
        TP += tp; FP += fp; FN += fn
        res["cowc"].append(dict(image=os.path.basename(im_p), scale=s, true_cars=len(gt), detected=len(d),
                                tp=tp, fp=fp, fn=fn, seconds=round(time.time() - t, 1)))
        print(res["cowc"][-1])
    P = TP / max(TP + FP, 1); R = TP / max(TP + FN, 1)
    res["scales"][s] = dict(precision=P, recall=R, f1=2 * P * R / max(P + R, 1e-9), tp=TP, fp=FP, fn=FN)
    print("scale", s, res["scales"][s])

best = max(res["scales"], key=lambda s: res["scales"][s]["f1"])
res["best_scale"] = best
# pictures for the best scale
for im_p, an_p in pairs:
    img = cache[im_p]; gt = gt_points(an_p); d = detect_cars(model, img, scale=best)
    draw(img, d[:, :2], gt, path=os.path.join(out, "cowc_" + os.path.basename(im_p).replace(".png", ".jpg")))

ign = [] if os.environ.get("SKIP_IGN") else glob.glob(os.path.join(src, "auchan*.jp*g")) + glob.glob(os.path.join(src, "auchan*.png"))
if ign:
    img = load_rgb(ign[0])
    s_ign = best * IGN_GSD / COWC_GSD       # same car size in pixels as the best COWC setting
    t = time.time()
    d = detect_cars(model, img, scale=s_ign)
    res["ign"] = dict(image=os.path.basename(ign[0]), size=img.shape[:2], scale=s_ign, cars=len(d),
                      seconds=round(time.time() - t, 1))
    np.save(os.path.join(out, "ign_detections.npy"), d)
    draw(img, d[:, :2], r=7, path=os.path.join(out, "auchan_detected.jpg"), max_side=3500)
    print(res["ign"])

json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1, default=float)
