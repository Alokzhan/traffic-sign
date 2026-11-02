import os, glob, cv2, keras, numpy as np
from common import CLASS_NAMES, preprocess, ALL_GTSRB_CLASSES
from ml_classifier import ml_hybrid
from api import smart_square_crop

m_path = "outputs/final.keras"
m = keras.models.load_model(m_path)
print(f"Loaded {m_path} (name={m.name})")

feat_m = keras.Model(m.inputs, m.get_layer("latent_features").output)

speed_cids = [0, 1, 2, 3, 4, 5, 6, 7, 8]

print("\n--- Testing sample_signs/ speed limits on CONVERGED FINAL MODEL ---")
for cid in speed_cids:
    name = ALL_GTSRB_CLASSES[cid]
    matches = glob.glob(f"sample_signs/{cid:02d}_*.png")
    if not matches:
        continue
    f = matches[0]
    raw = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    crop = smart_square_crop(raw)
    x = preprocess(crop)[None].astype("float32")
    p = m.predict(x, verbose=0)[0]
    emb = feat_m(x, training=False).numpy()[0]
    blend = ml_hybrid.ensemble_blend(p, emb)
    bp = blend["blended_probs"]
    top = int(bp.argmax())
    pred_name = CLASS_NAMES[top]
    ok = "[OK]" if top == cid else "[FAIL]"
    ml_val = f"{blend['ml_probs'][top]:.1%}" if blend["ml_probs"] is not None else "N/A"
    print(f"{ok} Target: {name:25s} -> Predicted: {pred_name:25s} ({bp[top]:.1%}) [Deep: {p[top]:.1%}, ML: {ml_val}]")

print("\n--- Testing user uploaded image (debug_uploaded_raw.png) ---")
user_raw = cv2.imread("outputs/debug_uploaded_raw.png")
if user_raw is not None:
    crop = smart_square_crop(user_raw)
    x = preprocess(crop)[None].astype("float32")
    p = m.predict(x, verbose=0)[0]
    emb = feat_m(x, training=False).numpy()[0]
    blend = ml_hybrid.ensemble_blend(p, emb)
    bp = blend["blended_probs"]
    top = int(bp.argmax())
    pred_name = CLASS_NAMES[top]
    ml_val = f"{blend['ml_probs'][top]:.1%}" if blend["ml_probs"] is not None else "N/A"
    print(f"[PREDICTION] User image -> Predicted: {pred_name:25s} ({bp[top]:.1%}) [Deep: {p[top]:.1%}, ML: {ml_val}]")
