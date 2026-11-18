"""Reference (Python + onnxruntime) of the exact pipeline the web page runs in JavaScript.
Used to choose settings and to check the browser gives the same counts."""
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import onnxruntime as ort

MODELS = {
    "own":  dict(path="w/own_cowc_yolo11n.onnx", size=640,  obb=False, car=0,  target=0.10, conf=0.20),
    "dota": dict(path="w/yolo11n-obb.onnx",      size=1024, obb=True,  car=10, target=0.09, conf=0.25),
}
_sess = {}


def session(name):
    if name not in _sess:
        _sess[name] = ort.InferenceSession(MODELS[name]["path"], providers=["CPUExecutionProvider"])
    return _sess[name]


def count(img, gsd, name, conf=None, target=None, stretch=False, sharpen=False, overlap=96, return_len=False):
    m = MODELS[name]; conf = m["conf"] if conf is None else conf; target = target or m["target"]
    S = m["size"]; sc = gsd / target
    pil = Image.fromarray(img)
    pil = pil.resize((max(1, round(pil.width * sc)), max(1, round(pil.height * sc))), Image.BILINEAR)
    if stretch:
        pil = ImageOps.autocontrast(pil, cutoff=1)
    if sharpen:
        pil = pil.filter(ImageFilter.UnsharpMask(radius=3, percent=150, threshold=2))
    a = np.asarray(pil, dtype=np.float32) / 255.0
    H, W = a.shape[:2]; step = S - overlap
    sess = session(name); inp = sess.get_inputs()[0].name
    dets = []
    for y in range(0, max(H - overlap, 1), step):
        for x in range(0, max(W - overlap, 1), step):
            t = np.full((S, S, 3), 114 / 255, np.float32)
            crop = a[y:y + S, x:x + S]; t[:crop.shape[0], :crop.shape[1]] = crop
            o = sess.run(None, {inp: t.transpose(2, 0, 1)[None]})[0][0]   # (C, N)
            score = o[4 + m["car"]]
            k = score > conf
            for cx, cy, w, h, s in zip(o[0, k], o[1, k], o[2, k], o[3, k], score[k]):
                if cx < crop.shape[1] and cy < crop.shape[0]:
                    dets.append((x + cx, y + cy, s, max(w, h)))
    if not dets:
        return np.zeros((0, 4))
    d = np.array(dets); d[:, [0, 1, 3]] /= sc
    # merge duplicates: keep most confident within 1.5 m
    r = 1.5 / gsd
    order = np.argsort(-d[:, 2]); keep = []; removed = np.zeros(len(d), bool)
    from scipy.spatial import cKDTree
    tree = cKDTree(d[:, :2])
    for i in order:
        if removed[i]:
            continue
        keep.append(i)
        removed[tree.query_ball_point(d[i, :2], r)] = True
    return d[keep]
