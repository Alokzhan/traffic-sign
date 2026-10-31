"""Advanced evaluation: calibration (ECE), unseen-sign rejection (OOD), sample explorer data.
Run (training ke baad, ek baar):
    python evaluate_advanced.py --data "<gtsrb folder>" --out outputs
Phir uvicorn restart karein.
"""
import os, json, argparse
import numpy as np, cv2, pandas as pd
from tensorflow import keras
from sklearn.metrics import roc_auc_score
# Ensure the project root is in PYTHONPATH so we can import common utilities
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common import *


def entropy(P):
    return -(P * np.log(P + 1e-9)).sum(1) / np.log(P.shape[1])


def load(df, root):
    X, keep = [], []
    for i, (_, r) in enumerate(df.iterrows()):
        img = cv2.imread(os.path.join(root, r.Path))
        if img is None:
            continue
        X.append(preprocess(img)); keep.append(i)
    return np.array(X), df.iloc[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--max-ood", type=int, default=2500)
    a = ap.parse_args()

    model = keras.models.load_model(os.path.join(a.out, "final.keras"))
    df = pd.read_csv(os.path.join(a.data, "Test.csv"))
    df_id = df[df.ClassId.isin(CLASS_IDS)]
    df_ood = df[~df.ClassId.isin(CLASS_IDS)]          # model ne ye classes kabhi dekhi hi nahi
    if len(df_ood) > a.max_ood:
        df_ood = df_ood.sample(a.max_ood, random_state=42)

    Xi, df_id = load(df_id, a.data)
    Xo, df_ood = load(df_ood, a.data)
    Pi = model.predict(Xi.astype("float32"), batch_size=128, verbose=0)
    Po = model.predict(Xo.astype("float32"), batch_size=128, verbose=0)
    yi = np.array([ID_TO_IDX[c] for c in df_id.ClassId])
    ci, co = Pi.max(1), Po.max(1)
    pi = Pi.argmax(1)
    correct = pi == yi

    # --- Calibration (ECE, 10 bins) ---
    edges = np.linspace(0, 1, 11)
    rel, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ci > lo) & (ci <= hi)
        if m.sum() == 0:
            continue
        acc, conf = float(correct[m].mean()), float(ci[m].mean())
        ece += m.mean() * abs(acc - conf)
        rel.append({"conf": conf, "acc": acc, "n": int(m.sum())})

    # --- Unseen-sign detection: confidence/entropy se ID vs OOD alag kar sakte hain? ---
    lab = np.r_[np.zeros(len(ci)), np.ones(len(co))]
    auroc_conf = float(roc_auc_score(lab, np.r_[1 - ci, 1 - co]))
    auroc_ent = float(roc_auc_score(lab, np.r_[entropy(Pi), entropy(Po)]))

    sweep = []
    for t in list(np.arange(0.30, 1.0, 0.02)) + [0.99]:
        acc_mask = ci >= t
        sweep.append({"thr": round(float(t), 2),
                      "id_accept": float(acc_mask.mean()),
                      "ood_accept": float((co >= t).mean()),
                      "selective_acc": float(correct[acc_mask].mean()) if acc_mask.any() else None})

    # --- Kaun si unseen classes model ko sabse zyada bewakoof banati hain (thr=0.7) ---
    worst = []
    for c in sorted(df_ood.ClassId.unique()):
        m = (df_ood.ClassId.values == c)
        if m.sum() >= 20:
            worst.append({"cls": int(c), "n": int(m.sum()), "accepted": float((co[m] >= 0.7).mean())})
    worst = sorted(worst, key=lambda d: -d["accepted"])[:8]

    # --- Explorer samples ---
    rng = np.random.default_rng(42)
    samples = []
    def add(row, p, true_name, ood, ok):
        samples.append({"file": os.path.basename(row.Path), "true": true_name,
                        "pred": CLASS_NAMES[int(p.argmax())], "conf": float(p.max()),
                        "entropy": float(entropy(p[None])[0]), "ood": ood, "correct": ok})
    wrong_idx = np.where(~correct)[0]
    right_idx = rng.choice(np.where(correct)[0], min(120, correct.sum()), replace=False)
    for i in list(wrong_idx) + list(right_idx):
        add(df_id.iloc[i], Pi[i], CLASS_NAMES[yi[i]], False, bool(correct[i]))
    acc_ood = np.where(co >= 0.7)[0]
    rest_ood = np.where(co < 0.7)[0]
    pick = list(rng.permutation(acc_ood)[:100]) + list(rng.permutation(rest_ood)[:140])
    for i in pick:
        add(df_ood.iloc[i], Po[i], f"Unseen class {int(df_ood.iloc[i].ClassId)}", True, False)

    out = {"data_dir": os.path.abspath(a.data), "n_id": int(len(ci)), "n_ood": int(len(co)),
           "id_accuracy": float(correct.mean()), "ece": float(ece), "reliability": rel,
           "auroc_conf": auroc_conf, "auroc_entropy": auroc_ent, "sweep": sweep,
           "worst_unseen": worst, "samples": samples}
    json.dump(out, open(os.path.join(a.out, "advanced.json"), "w"), indent=1)
    print(f"ID acc={correct.mean():.4f}  ECE={ece:.4f}  AUROC(conf)={auroc_conf:.3f}  AUROC(entropy)={auroc_ent:.3f}")
    print(f"Unseen signs accepted at thr 0.70: {(co >= 0.7).mean():.1%}")
    print("Saved:", os.path.join(a.out, "advanced.json"), "\n", DISCLAIMER)


if __name__ == "__main__":
    main()
