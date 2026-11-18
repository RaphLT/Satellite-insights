"""Turn COWC tiles (photo + one dot per car) into a YOLO training set.

- Every image is resampled to TARGET_GSD (0.10 m/pixel), so a car is ~44 px long
  whatever the source. At prediction time we resample IGN photos the same way.
- Each car dot becomes a square box BOX_M metres wide (COWC gives no car outline).
- For every tile we also make a "soft" copy: shrunk to 0.25 m/pixel, JPEG-compressed
  and enlarged back. It mimics the softer IGN photos so the model learns both looks.
Usage: python train_prep.py <out_dir> <tile.png> [<tile.png> ...]   (labels: <tile>_Annotated_Cars.png)
"""
import sys, os, io
import numpy as np
from PIL import Image
from carcount import gt_points

Image.MAX_IMAGE_PIXELS = None
SRC_GSD = 300 / 2220        # COWC Potsdam tiles: 300 m across in 2220 px
TARGET_GSD = 0.10
BOX_M = 3.6
CROP = 640
STEP = 560


def soften(im, gsd_from, gsd_soft=0.25, quality=70):
    w, h = im.size
    small = im.resize((int(w * gsd_from / gsd_soft), int(h * gsd_from / gsd_soft)), Image.LANCZOS)
    buf = io.BytesIO(); small.save(buf, "JPEG", quality=quality); buf.seek(0)
    return Image.open(buf).convert("RGB").resize((w, h), Image.BICUBIC)


def write_crops(im, pts, out, split, stem):
    os.makedirs(f"{out}/images/{split}", exist_ok=True); os.makedirs(f"{out}/labels/{split}", exist_ok=True)
    a = np.array(im); H, W = a.shape[:2]; b = BOX_M / TARGET_GSD; n = 0
    for y in range(0, max(H - CROP, 0) + 1, STEP):
        for x in range(0, max(W - CROP, 0) + 1, STEP):
            sel = pts[(pts[:, 0] >= x) & (pts[:, 0] < x + CROP) & (pts[:, 1] >= y) & (pts[:, 1] < y + CROP)]
            name = f"{stem}_{x}_{y}"
            Image.fromarray(a[y:y + CROP, x:x + CROP]).save(f"{out}/images/{split}/{name}.jpg", quality=92)
            with open(f"{out}/labels/{split}/{name}.txt", "w") as f:
                for px, py in sel:
                    cx, cy = (px - x) / CROP, (py - y) / CROP
                    f.write(f"0 {cx:.5f} {cy:.5f} {b / CROP:.5f} {b / CROP:.5f}\n")
            n += 1
    return n


if __name__ == "__main__":
    out, tiles = sys.argv[1], sys.argv[2:]
    val_tile = [t for t in tiles if "6_8" in t] or tiles[-1:]
    total = 0
    for t in tiles:
        split = "val" if t in val_tile else "train"
        im = Image.open(t).convert("RGB")
        pts = gt_points(t.replace(".png", "_Annotated_Cars.png"))
        f = SRC_GSD / TARGET_GSD
        big = im.resize((int(im.width * f), int(im.height * f)), Image.LANCZOS)
        p = pts * f
        stem = os.path.basename(t).replace(".png", "")
        total += write_crops(big, p, out, split, stem)
        total += write_crops(soften(big, TARGET_GSD), p, out, split, stem + "_soft")
        print(t, split, len(pts), "cars")
    with open(f"{out}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(out)}\ntrain: images/train\nval: images/val\nnames:\n  0: car\n")
    print("crops:", total)
