"""Car counting from overhead images with an open, pretrained detector.

Detector: Ultralytics YOLO11 oriented-box model pretrained on DOTA v1
(aerial images, class "small vehicle" = cars). We do NOT train anything:
the point is to test whether an off-the-shelf open model is enough.
"""
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

Image.MAX_IMAGE_PIXELS = None


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def car_class_ids(model):
    """'small vehicle' in the DOTA model; 'car' in our own COWC-trained model."""
    return [i for i, n in model.names.items() if n.replace("-", " ").lower() in ("small vehicle", "car")]


def enhance(im):
    """Stretch contrast and sharpen: helps on hazy, compressed aerial photos."""
    from PIL import ImageOps, ImageFilter
    im = ImageOps.autocontrast(im, cutoff=1)
    return im.filter(ImageFilter.UnsharpMask(radius=3, percent=150, threshold=2))


def detect_cars(model, img, scale=1.0, tile=1024, overlap=160, conf=0.25, imgsz=1024, dedup_px=10, sharpen=False):
    """Run the detector on a large image by tiling it.

    scale: resize factor applied before detection (to match the detector's preferred car size).
    dedup_px: two detections closer than this (original pixels) are the same car.
    Returns an array (N, 4): x, y, confidence, box length (all in ORIGINAL image pixels).
    """
    h0, w0 = img.shape[:2]
    pil = Image.fromarray(img)
    if scale != 1.0:
        pil = pil.resize((int(w0 * scale), int(h0 * scale)), Image.LANCZOS)
    if sharpen:
        pil = enhance(pil)
    im = np.array(pil)
    h, w = im.shape[:2]
    step = tile - overlap
    cls = car_class_ids(model)
    out = []
    ys = list(range(0, max(h - overlap, 1), step))
    xs = list(range(0, max(w - overlap, 1), step))
    for y in ys:
        for x in xs:
            crop = im[y:y + tile, x:x + tile]
            if crop.shape[0] < 32 or crop.shape[1] < 32:
                continue
            r = model.predict(crop, imgsz=imgsz, conf=conf, classes=cls, verbose=False)[0]
            if r.obb is not None:                      # oriented boxes (DOTA model)
                if len(r.obb) == 0:
                    continue
                xywh = r.obb.xywhr.cpu().numpy()[:, :4]; cf = r.obb.conf.cpu().numpy()
            else:                                      # ordinary boxes (our own model)
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xywh = r.boxes.xywh.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
            for (cx, cy, bw, bh), c in zip(xywh, cf):
                out.append((x + cx, y + cy, c, max(bw, bh)))
    if not out:
        return np.zeros((0, 4))
    d = np.array(out, dtype=float)
    # merge duplicates from overlapping tiles: keep highest confidence within dedup_px
    dedup_px = dedup_px * scale          # given in original-image pixels
    from scipy.spatial import cKDTree
    order = np.argsort(-d[:, 2])
    tree = cKDTree(d[:, :2])
    removed = np.zeros(len(d), bool)
    keep = []
    for i in order:                       # greedy: keep the most confident, drop its neighbours
        if removed[i]:
            continue
        keep.append(i)
        for j in tree.query_ball_point(d[i, :2], dedup_px):
            removed[j] = True
    d = d[keep]
    d[:, :2] /= scale
    d[:, 3] /= scale
    return d


def gt_points(annot_path):
    """COWC ground truth: one small mark at the centre of every car."""
    a = np.array(Image.open(annot_path))
    if a.ndim == 3:
        a = a[..., :3].max(axis=2) if a.shape[2] >= 3 else a[..., 0]
    lab, n = ndimage.label(a > 0)
    if n == 0:
        return np.zeros((0, 2))
    c = ndimage.center_of_mass(np.ones_like(lab), lab, range(1, n + 1))
    return np.array([(x, y) for y, x in c])


def match(det_xy, gt_xy, radius):
    """One-to-one matching of detections to true cars within `radius` pixels."""
    if len(det_xy) == 0 or len(gt_xy) == 0:
        return 0, len(det_xy), len(gt_xy)
    D = np.hypot(det_xy[:, None, 0] - gt_xy[None, :, 0], det_xy[:, None, 1] - gt_xy[None, :, 1])
    cost = np.where(D <= radius, D, 1e6)
    r, c = linear_sum_assignment(cost)
    tp = int((cost[r, c] < 1e6).sum())
    return tp, len(det_xy) - tp, len(gt_xy) - tp


def draw(img, det_xy, gt_xy=None, matched=None, r=9, path=None, max_side=2400):
    """Save a picture with detections (blue rings) and, if given, true cars (orange dots)."""
    from PIL import ImageDraw
    im = Image.fromarray(img).copy()
    dr = ImageDraw.Draw(im)
    if gt_xy is not None:
        for x, y in gt_xy:
            dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(235, 104, 52))
    for x, y in det_xy:
        dr.ellipse([x - r, y - r, x + r, y + r], outline=(42, 120, 214), width=3)
    if max(im.size) > max_side:
        f = max_side / max(im.size)
        im = im.resize((int(im.size[0] * f), int(im.size[1] * f)), Image.LANCZOS)
    if path:
        im.save(path, quality=88)
    return im
