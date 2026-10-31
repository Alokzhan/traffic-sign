"""Advanced evaluation: calibration (ECE), unseen-sign rejection (OOD), before/after comparison.
Run:
    python evaluate_advanced.py --data "<gtsrb folder>" --out outputs
"""
import os, json, argparse
import numpy as np, cv2, pandas as pd
from tensorflow import keras
from sklearn.metrics import roc_auc_score
from common import (
    CLASS_IDS, CLASS_NAMES, ID_TO_IDX, HELD_OUT_CLASSES, HELD_OUT_NAMES,
    preprocess, DISCLAIMER
)


def entropy(P):
    return -(P * np.log(P + 1e-9)).sum(1) / np.log(P.shape[1])


def load(df, root):
    X, keep = [], []
    for i, (_, r) in enumerate(df.iterrows()):
        img = cv2.imread(os.path.join(root, r.Path))
        if img is None:
            continue
        X.append(preprocess(img))
        keep.append(i)
    return np.array(X, dtype="float32"), df.iloc[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    v1_path = os.path.join(a.out, "final_v1.keras")
    v2_path = os.path.join(a.out, "final.keras")
    assert os.path.exists(v2_path), f"New model not found at {v2_path}"

    model_v2 = keras.models.load_model(v2_path)
    model_v1 = keras.models.load_model(v1_path) if os.path.exists(v1_path) else None

    df = pd.read_csv(os.path.join(a.data, "Test.csv"))
    # In-distribution (8 selected sign classes)
    df_id = df[df.ClassId.isin(CLASS_IDS)]
    # Unseen classes: the 8 fully held-out GTSRB classes
    df_held = df[df.ClassId.isin(HELD_OUT_CLASSES)]

    print(f"Loading {len(df_id)} ID test images and {len(df_held)} unseen test images...")
    Xi, df_id = load(df_id, a.data)
    Xh, df_held = load(df_held, a.data)

    yi = np.array([ID_TO_IDX[c] for c in df_id.ClassId])

    # Inference with V2 model (9 classes)
    Pi_v2 = model_v2.predict(Xi, batch_size=128, verbose=0)
    Ph_v2 = model_v2.predict(Xh, batch_size=128, verbose=0)

    pi_v2 = Pi_v2.argmax(1)
    ci_v2 = Pi_v2.max(1)
    ch_v2 = Ph_v2.max(1)
    ph_v2 = Ph_v2.argmax(1)

    # In-distribution accuracy on the 8 sign classes
    correct_id_v2 = (pi_v2 == yi)
    id_acc_v2 = float(correct_id_v2.mean())

    # --- Calibration (ECE, 10 bins) on ID test data ---
    edges = np.linspace(0, 1, 11)
    rel, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ci_v2 > lo) & (ci_v2 <= hi)
        if m.sum() == 0:
            continue
        acc, conf = float(correct_id_v2[m].mean()), float(ci_v2[m].mean())
        ece += m.mean() * abs(acc - conf)
        rel.append({"conf": conf, "acc": acc, "n": int(m.sum())})

    # --- Unseen-sign detection: False Accept Rate (FAR) at thr=0.70 ---
    # False accept = predicted as one of the 8 signs AND confidence >= 0.70
    fa_v2_mask = (ph_v2 < 8) & (ch_v2 >= 0.70)
    overall_far_v2 = float(fa_v2_mask.mean())
    other_pred_rate_v2 = float((ph_v2 == 8).mean())

    # Class-wise FAR for each of the 8 held-out classes
    classwise_v2 = {}
    for c in HELD_OUT_CLASSES:
        m = (df_held.ClassId.values == c)
        c_fa = float((fa_v2_mask[m]).mean())
        c_other = float((ph_v2[m] == 8).mean())
        classwise_v2[str(c)] = {
            "name": HELD_OUT_NAMES[c],
            "count": int(m.sum()),
            "false_accept_rate": c_fa,
            "classified_as_other": c_other
        }

    # Evaluate baseline V1 if available
    classwise_v1 = {}
    overall_far_v1 = None
    if model_v1 is not None:
        Ph_v1 = model_v1.predict(Xh, batch_size=128, verbose=0)
        ch_v1 = Ph_v1.max(1)
        fa_v1_mask = (ch_v1 >= 0.70)
        overall_far_v1 = float(fa_v1_mask.mean())
        for c in HELD_OUT_CLASSES:
            m = (df_held.ClassId.values == c)
            c_fa = float((fa_v1_mask[m]).mean())
            classwise_v1[str(c)] = {
                "name": HELD_OUT_NAMES[c],
                "count": int(m.sum()),
                "false_accept_rate": c_fa
            }

    # --- AUROC for ID vs Unseen detection ---
    lab = np.r_[np.zeros(len(ci_v2)), np.ones(len(ch_v2))]
    # Higher score = more likely to be unseen
    # For v2, Ph_v2[:, 8] directly predicts P(Other), plus 1 - max(sign probs)
    sign_conf_id = Pi_v2[:, :8].max(1)
    sign_conf_held = Ph_v2[:, :8].max(1)
    auroc_conf = float(roc_auc_score(lab, np.r_[1 - sign_conf_id, 1 - sign_conf_held]))
    auroc_ent = float(roc_auc_score(lab, np.r_[entropy(Pi_v2), entropy(Ph_v2)]))

    # --- Sweep over thresholds ---
    sweep = []
    for t in list(np.arange(0.30, 1.0, 0.02)) + [0.99]:
        id_acc_mask = ci_v2 >= t
        fa_mask_t = (ph_v2 < 8) & (ch_v2 >= t)
        sweep.append({
            "thr": round(float(t), 2),
            "id_accept": float(id_acc_mask.mean()),
            "ood_accept": float(fa_mask_t.mean()),
            "selective_acc": float(correct_id_v2[id_acc_mask].mean()) if id_acc_mask.any() else None
        })

    # --- Explorer samples ---
    rng = np.random.default_rng(42)
    samples = []
    def add(row, p, true_name, ood, ok):
        samples.append({
            "file": os.path.basename(row.Path),
            "true": true_name,
            "pred": CLASS_NAMES[int(p.argmax())],
            "conf": float(p.max()),
            "entropy": float(entropy(p[None])[0]),
            "ood": ood,
            "correct": ok
        })

    wrong_idx = np.where(~correct_id_v2)[0]
    right_idx = rng.choice(np.where(correct_id_v2)[0], min(120, correct_id_v2.sum()), replace=False)
    for i in list(wrong_idx) + list(right_idx):
        add(df_id.iloc[i], Pi_v2[i], CLASS_NAMES[yi[i]], False, bool(correct_id_v2[i]))

    acc_ood = np.where(fa_v2_mask)[0]
    rest_ood = np.where(~fa_v2_mask)[0]
    pick = list(rng.permutation(acc_ood)[:100]) + list(rng.permutation(rest_ood)[:140])
    for i in pick:
        cid = int(df_held.iloc[i].ClassId)
        add(df_held.iloc[i], Ph_v2[i], f"Unseen: {HELD_OUT_NAMES.get(cid, cid)}", True, False)

    worst_unseen = []
    for c in HELD_OUT_CLASSES:
        m = (df_held.ClassId.values == c)
        worst_unseen.append({
            "cls": int(c),
            "name": HELD_OUT_NAMES[c],
            "n": int(m.sum()),
            "accepted": float((fa_v2_mask[m]).mean()),
            "other_classified": float((ph_v2[m] == 8).mean())
        })
    worst_unseen = sorted(worst_unseen, key=lambda d: -d["accepted"])

    # Output advanced.json
    out_adv = {
        "data_dir": os.path.abspath(a.data),
        "n_id": int(len(ci_v2)),
        "n_ood": int(len(ch_v2)),
        "id_accuracy": id_acc_v2,
        "ece": float(ece),
        "reliability": rel,
        "auroc_conf": auroc_conf,
        "auroc_entropy": auroc_ent,
        "sweep": sweep,
        "worst_unseen": worst_unseen,
        "samples": samples
    }
    json.dump(out_adv, open(os.path.join(a.out, "advanced.json"), "w"), indent=1)

    # Output before_after.json
    before_after = {
        "metric": "False Accept Rate on 8 Held-out Unseen Classes (conf >= 0.70)",
        "v1_closed_set_8class": {
            "overall_false_accept_rate": overall_far_v1,
            "classwise": classwise_v1
        },
        "v2_with_other_9class": {
            "overall_false_accept_rate": overall_far_v2,
            "overall_classified_as_other": other_pred_rate_v2,
            "id_accuracy": id_acc_v2,
            "ece": float(ece),
            "auroc_conf": auroc_conf,
            "classwise": classwise_v2
        }
    }
    json.dump(before_after, open(os.path.join(a.out, "before_after.json"), "w"), indent=2)

    # Print Clean Report
    print("=" * 80)
    print("BEFORE / AFTER COMPARISON ON 8 HELD-OUT UNSEEN CLASSES (3,660 test images)")
    print("Definition: False Accept = Predicted as one of 8 signs AND confidence >= 70%")
    print("=" * 80)
    print(f"{'Class ID & Name':<35} | {'N Images':>8} | {'v1 (8-class)':>13} | {'v2 (9-class)':>13}")
    print("-" * 80)
    for c in HELD_OUT_CLASSES:
        sc = str(c)
        cname = f"{c}: {HELD_OUT_NAMES[c]}"
        cnt = classwise_v2[sc]["count"]
        far1_str = f"{classwise_v1[sc]['false_accept_rate']*100:5.1f}%" if sc in classwise_v1 else "N/A"
        far2_str = f"{classwise_v2[sc]['false_accept_rate']*100:5.1f}%"
        print(f"{cname:<35} | {cnt:8d} | {far1_str:>13} | {far2_str:>13}")
    print("-" * 80)
    far1_tot = f"{overall_far_v1*100:5.1f}%" if overall_far_v1 is not None else "N/A"
    far2_tot = f"{overall_far_v2*100:5.1f}%"
    print(f"{'OVERALL UNSEEN FALSE ACCEPT RATE':<35} | {len(df_held):8d} | {far1_tot:>13} | {far2_tot:>13}")
    print("=" * 80)
    print(f"ID Test Accuracy (8 classes): {id_acc_v2*100:.2f}%")
    print(f"Calibration ECE:              {ece:.4f}")
    print(f"AUROC (Sign Conf vs Unseen):  {auroc_conf:.3f}")
    print(f"Saved: {os.path.join(a.out, 'advanced.json')} and {os.path.join(a.out, 'before_after.json')}")
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
