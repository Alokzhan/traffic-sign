"""FastAPI backend for the React UI.
Run:  uvicorn api:app --port 8000
Needs outputs/final.keras (train.py). Optional: outputs/advanced.json (evaluate_advanced.py).
"""
import os, json, base64, threading
from typing import List
import numpy as np, cv2
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tensorflow import keras
from common import (
    CLASS_NAMES, OTHER_CLASS_NAME, OTHER_IDX, NUM_CLASSES,
    DISCLAIMER, preprocess
)
from predict import gradcam

OUT = os.environ.get("OUT_DIR", "outputs")
model_path = os.path.join(OUT, "final.keras")
model = keras.models.load_model(model_path) if os.path.exists(model_path) else None
lock = threading.Lock()

app = FastAPI(title="Traffic Sign API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if os.path.isdir(OUT):
    app.mount("/outputs", StaticFiles(directory=OUT), name="outputs")

ADV = os.path.join(OUT, "advanced.json")
if os.path.exists(ADV):
    try:
        _d = json.load(open(ADV)).get("data_dir")
        if _d and os.path.isdir(os.path.join(_d, "Test")):
            app.mount("/testimg", StaticFiles(directory=os.path.join(_d, "Test")), name="testimg")
    except Exception:
        pass


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


_last_mtime = 0
_current_loaded_file = None

def get_model():
    global model, _last_mtime, _current_loaded_file
    candidates = []
    for fname in ["final.keras", "best.keras"]:
        p = os.path.join(OUT, fname)
        if os.path.exists(p):
            try:
                candidates.append((os.path.getmtime(p), p))
            except OSError:
                pass
    if not candidates:
        return None
    candidates.sort(reverse=True)
    mtime, chosen = candidates[0]
    if model is None or mtime > _last_mtime or _current_loaded_file != chosen:
        try:
            m = keras.models.load_model(chosen)
            model = m
            _last_mtime = mtime
            _current_loaded_file = chosen
            print(f"API loaded {chosen} (outputs={model.output_shape[-1]})")
        except Exception as e:
            print("Model load error (retrying with current):", e)
    return model

def run(x):
    m = get_model()
    if m is None:
        raise HTTPException(503, "Model not loaded yet")
    with lock:
        return m.predict(x, verbose=0)


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
FGSM_LABEL = "⚔️ Adversarial attack (FGSM, pixel ε)"
FGSM_EPS = [0, 1, 2, 4, 8, 16, 32]


def fgsm_sweep(bgr):
    """Fast Gradient Sign Method: model ke gradient ki disha mein chhota noise."""
    rgb = preprocess(bgr)
    x = tf.convert_to_tensor(rgb[None].astype("float32"))
    m = get_model()
    if m is None:
        raise HTTPException(503, "Model not loaded")
    with lock:
        label = int(np.argmax(m(x, training=False).numpy()[0]))
        with tf.GradientTape() as tape:
            tape.watch(x)
            pred = m(x, training=False)
            loss = keras.losses.sparse_categorical_crossentropy(tf.constant([label]), pred)
        sign = tf.sign(tape.gradient(loss, x)).numpy()
    advs = [np.clip(rgb[None].astype("float32") + e * sign, 0, 255).astype("uint8") for e in FGSM_EPS]
    P = run(np.concatenate(advs).astype("float32"))
    n_out = P.shape[-1]
    classes = CLASS_NAMES[:n_out]
    items = [{"level": e, "image": to_b64(adv[0]), "pred": classes[int(p.argmax())],
              "conf": float(p.max()), "base_conf": float(p[label])}
             for e, adv, p in zip(FGSM_EPS, advs, P)]
    return {"base": classes[label], "items": items}


# ---------- routes ----------
@app.get("/api/health")
def health():
    m = get_model()
    n_out = m.output_shape[-1] if m is not None else len(CLASS_NAMES)
    return {"ok": True, "classes": CLASS_NAMES[:n_out], "disclaimer": DISCLAIMER}


@app.post("/api/predict")
def predict_ep(file: UploadFile = File(...), cam: int = 1):
    bgr = decode(file)
    rgb = preprocess(bgr)
    x = rgb[None].astype("float32")
    p = run(x)[0]
    n_out = len(p)
    classes = CLASS_NAMES[:n_out]
    top = int(p.argmax())
    m = get_model()
    cam_img = None
    if cam and m is not None:
        try:
            with lock:
                heat_map = gradcam(m, x)
            base = cv2.resize(rgb, (192, 192), interpolation=cv2.INTER_NEAREST)
            heat = cv2.applyColorMap(np.uint8(255 * cv2.resize(heat_map, (192, 192))), cv2.COLORMAP_JET)
            heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
            cam_img = to_b64(cv2.addWeighted(base, 0.6, heat, 0.4, 0))
        except Exception:
            pass

    is_other = bool("Other" in classes[top])
    return {
        "classes": classes,
        "probs": [float(v) for v in p],
        "top": top,
        "name": classes[top],
        "is_other": is_other,
        "conf": float(p.max()),
        "entropy": entropy(p),
        "model_input": to_b64(rgb, 192),
        "gradcam": cam_img
    }


@app.post("/api/batch")
def batch_ep(files: List[UploadFile] = File(...)):
    rows = []
    for f in files:
        try:
            bgr = decode(f)
        except HTTPException:
            continue
        p = run(preprocess(bgr)[None].astype("float32"))[0]
        n_out = len(p)
        classes = CLASS_NAMES[:n_out]
        top = int(p.argmax())
        rows.append({
            "file": f.filename,
            "prediction": classes[top],
            "is_other": bool("Other" in classes[top]),
            "confidence": float(p.max()),
            "entropy": entropy(p)
        })
    return {"rows": rows}


@app.get("/api/perturbations")
def perturbations():
    return [{"key": k, "label": v[0]} for k, v in PERTURB.items()] + [{"key": "fgsm", "label": FGSM_LABEL}]


@app.post("/api/robustness")
def robustness(file: UploadFile = File(...), kind: str = Form("brightness")):
    if kind != "fgsm" and kind not in PERTURB:
        raise HTTPException(400, "unknown perturbation")
    bgr = decode(file)
    if kind == "fgsm":
        return fgsm_sweep(bgr)
    _, fn, levels = PERTURB[kind]
    imgs = [fn(bgr, l) for l in levels]
    X = np.stack([preprocess(i) for i in imgs]).astype("float32")
    P = run(X)
    n_out = P.shape[-1]
    classes = CLASS_NAMES[:n_out]
    base = int(run(preprocess(bgr)[None].astype("float32"))[0].argmax())
    items = [{"level": l, "image": to_b64(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)),
              "pred": classes[int(p.argmax())], "conf": float(p.max()), "base_conf": float(p[base])}
             for l, im, p in zip(levels, imgs, P)]
    return {"base": classes[base], "items": items}


@app.get("/api/report")
def report():
    mp = os.path.join(OUT, "metrics.json")
    if not os.path.exists(mp):
        raise HTTPException(404, "metrics.json nahi mila")
    names = ["confusion_matrix.png", "class_f1.png", "history.png", "examples.png", "difficult_examples.png"]
    return {
        "metrics": json.load(open(mp)),
        "classes": CLASS_NAMES,
        "images": [n for n in names if os.path.exists(os.path.join(OUT, n))]
    }


@app.get("/api/advanced")
def advanced():
    if not os.path.exists(ADV):
        raise HTTPException(404, "advanced.json nahi mila. Pehle evaluate_advanced.py chalayein.")
    return json.load(open(ADV))


@app.get("/api/before_after")
def before_after():
    ba_path = os.path.join(OUT, "before_after.json")
    if not os.path.exists(ba_path):
        raise HTTPException(404, "before_after.json nahi mila.")
    return json.load(open(ba_path))
