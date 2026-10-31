"""FastAPI backend for the React UI.
Run:  uvicorn api:app --port 8000
Needs outputs/final.keras (from train.py).
"""
import os, json, base64, threading
from typing import List
import numpy as np, cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tensorflow import keras
from common import *
from predict import gradcam

OUT = os.environ.get("OUT_DIR", "outputs")
model = keras.models.load_model(os.path.join(OUT, "final.keras"))
lock = threading.Lock()

app = FastAPI(title="Traffic Sign API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
if os.path.isdir(OUT):
    app.mount("/outputs", StaticFiles(directory=OUT), name="outputs")


# ---------- helpers ----------
def decode(upload: UploadFile):
    data = np.frombuffer(upload.file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Image read nahi ho payi")
    return img


def to_b64(rgb, size=None):
    if size:
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def run(x):
    with lock:
        return model.predict(x, verbose=0)


def entropy(p):
    return float(-(p * np.log(p + 1e-9)).sum() / np.log(len(p)))


# ---------- perturbations ----------
def p_brightness(img, l): return cv2.convertScaleAbs(img, alpha=1, beta=l)
def p_blur(img, l):
    k = int(l) | 1
    return img if l <= 1 else cv2.GaussianBlur(img, (k, k), 0)
def p_noise(img, l):
    n = np.random.default_rng(0).normal(0, l, img.shape)
    return np.clip(img.astype(float) + n, 0, 255).astype(np.uint8)
def p_rotate(img, l):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), l, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
def p_occlude(img, l):
    h, w = img.shape[:2]; s = int(min(h, w) * l); out = img.copy()
    y, x = (h - s) // 2, (w - s) // 2
    out[y:y + s, x:x + s] = 0
    return out

PERTURB = {
    "brightness": ("Brightness (dark ↔ bright)", p_brightness, [-120, -80, -40, 0, 40, 80, 120]),
    "blur":       ("Blur (focus / motion)",      p_blur,       [1, 3, 5, 7, 9, 11, 15]),
    "noise":      ("Gaussian noise",             p_noise,      [0, 10, 20, 40, 60, 80, 100]),
    "rotation":   ("Rotation (degrees)",         p_rotate,     [-40, -25, -10, 0, 10, 25, 40]),
    "occlusion":  ("Occlusion (centre patch)",   p_occlude,    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
}


# ---------- routes ----------
@app.get("/api/health")
def health():
    return {"ok": True, "classes": CLASS_NAMES, "disclaimer": DISCLAIMER}


@app.post("/api/predict")
def predict_ep(file: UploadFile = File(...)):
    bgr = decode(file)
    rgb = preprocess(bgr)
    x = rgb[None].astype("float32")
    p = run(x)[0]
    top = int(p.argmax())
    cam_img = None
    try:
        with lock:
            cam = gradcam(model, x)
        base = cv2.resize(rgb, (192, 192), interpolation=cv2.INTER_NEAREST)
        heat = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (192, 192))), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        cam_img = to_b64(cv2.addWeighted(base, 0.6, heat, 0.4, 0))
    except Exception:
        pass
    return {"classes": CLASS_NAMES, "probs": [float(v) for v in p], "top": top,
            "name": CLASS_NAMES[top], "conf": float(p.max()), "entropy": entropy(p),
            "model_input": to_b64(rgb, 192), "gradcam": cam_img}


@app.post("/api/batch")
def batch_ep(files: List[UploadFile] = File(...)):
    rows = []
    for f in files:
        try:
            bgr = decode(f)
        except HTTPException:
            continue
        p = run(preprocess(bgr)[None].astype("float32"))[0]
        rows.append({"file": f.filename, "prediction": CLASS_NAMES[int(p.argmax())],
                     "confidence": float(p.max()), "entropy": entropy(p)})
    return {"rows": rows}


@app.get("/api/perturbations")
def perturbations():
    return [{"key": k, "label": v[0]} for k, v in PERTURB.items()]


@app.post("/api/robustness")
def robustness(file: UploadFile = File(...), kind: str = Form("brightness")):
    if kind not in PERTURB:
        raise HTTPException(400, "unknown perturbation")
    bgr = decode(file)
    _, fn, levels = PERTURB[kind]
    imgs = [fn(bgr, l) for l in levels]
    X = np.stack([preprocess(i) for i in imgs]).astype("float32")
    P = run(X)
    base = int(run(preprocess(bgr)[None].astype("float32"))[0].argmax())
    items = [{"level": l, "image": to_b64(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)),
              "pred": CLASS_NAMES[int(p.argmax())], "conf": float(p.max()), "base_conf": float(p[base])}
             for l, im, p in zip(levels, imgs, P)]
    return {"base": CLASS_NAMES[base], "items": items}


@app.get("/api/report")
def report():
    mp = os.path.join(OUT, "metrics.json")
    if not os.path.exists(mp):
        raise HTTPException(404, "metrics.json nahi mila")
    names = ["confusion_matrix.png", "class_f1.png", "history.png", "examples.png", "difficult_examples.png"]
    return {"metrics": json.load(open(mp)), "classes": CLASS_NAMES,
            "images": [n for n in names if os.path.exists(os.path.join(OUT, n))]}
