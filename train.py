"""Traffic-sign classifier: data -> train -> evaluate -> reports.
Run: python train.py --data /path/to/gtsrb --out outputs --epochs 40
Dataset layout (Kaggle GTSRB): <data>/Train/<class_id>/*.png, <data>/Test.csv, <data>/Test/*.png
"""
import os, json, argparse
import numpy as np, cv2, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from common import *

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)


# ---------- 1. Data ----------
def load_train(root):
    X, y, groups = [], [], []
    for cid in CLASS_IDS:
        folder = os.path.join(root, "Train", str(cid))
        for f in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, f))
            if img is None:
                continue
            X.append(preprocess(img)); y.append(ID_TO_IDX[cid])
            parts = os.path.splitext(f)[0].split("_")
            track = parts[-2] if len(parts) >= 3 else parts[0]  # Kaggle: class_track_frame, original: track_frame
            groups.append(f"{cid}_{track}")  # ek physical sign ke saare frames = 1 group
    return np.array(X), np.array(y), np.array(groups)


def load_test(root):
    df = pd.read_csv(os.path.join(root, "Test.csv"))
    df = df[df.ClassId.isin(CLASS_IDS)]
    X, y = [], []
    for _, r in df.iterrows():
        img = cv2.imread(os.path.join(root, r.Path))
        if img is None:
            continue
        X.append(preprocess(img)); y.append(ID_TO_IDX[r.ClassId])
    return np.array(X), np.array(y)


# ---------- 2. Model ----------
def build_model(n_classes):
    # Horizontal flip jaan-bujhkar nahi: left/right turn signs ka meaning badal jata hai.
    aug = keras.Sequential([
        layers.RandomRotation(0.06), layers.RandomZoom(0.1),
        layers.RandomTranslation(0.08, 0.08), layers.RandomContrast(0.2),
    ], name="augment")
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = aug(inp)                       # sirf training me active
    x = layers.Rescaling(1 / 255)(x)
    for i, f in enumerate((32, 64, 128)):
        for j in range(2):
            name = "last_conv" if (i == 2 and j == 1) else None
            x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=name)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        if i < 2:
            x = layers.MaxPooling2D()(x)
            x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    m = keras.Model(inp, out, name="sign_cnn")
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


# ---------- 3. Plots ----------
def plot_history(h, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for a, k in zip(ax, ("loss", "accuracy")):
        a.plot(h.history[k], label="train"); a.plot(h.history["val_" + k], label="val")
        a.set_title(k); a.set_xlabel("epoch"); a.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES))); ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (held-out test)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_class_f1(report, path):
    f1 = [report[n]["f1-score"] for n in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(CLASS_NAMES, f1, color="#3b82f6"); ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1"); ax.set_title("Class-wise F1 (test)")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def show_grid(X, y, probs, idx, title, path):
    cols = 6; rows = max(1, int(np.ceil(len(idx) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6))
    axes = np.array(axes).reshape(-1)
    for a in axes:
        a.axis("off")
    for a, i in zip(axes, idx):
        p = probs[i].argmax(); ok = p == y[i]
        a.imshow(X[i])
        a.set_title(f"T: {CLASS_NAMES[y[i]]}\nP: {CLASS_NAMES[p]} {probs[i][p]:.0%}",
                    fontsize=7, color="green" if ok else "red")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ---------- 4. Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    X, y, g = load_train(a.data)
    Xte, yte = load_test(a.data)
    # Group split: ek hi physical sign ke frames train+val dono me nahi aayenge (leakage rokta hai)
    tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED).split(X, y, g))
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    print(f"train={len(Xtr)} val={len(Xva)} test(held-out)={len(Xte)}")

    cw = compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=ytr)
    model = build_model(len(CLASS_NAMES)); model.summary()
    cbs = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5),
        keras.callbacks.ModelCheckpoint(os.path.join(a.out, "best.keras"), save_best_only=True,
                                        monitor="val_accuracy"),
    ]
    h = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=a.epochs, batch_size=a.batch,
                  class_weight=dict(enumerate(cw)), callbacks=cbs)

    # Test set sirf yahan, final evaluation ke liye
    probs = model.predict(Xte, verbose=0); pred = probs.argmax(1)
    rep = classification_report(yte, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, pred)
    print(classification_report(yte, pred, target_names=CLASS_NAMES, zero_division=0))

    json.dump({"test_accuracy": float((pred == yte).mean()), "report": rep, "disclaimer": DISCLAIMER},
              open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    plot_history(h, os.path.join(a.out, "history.png"))
    plot_confusion(cm, os.path.join(a.out, "confusion_matrix.png"))
    plot_class_f1(rep, os.path.join(a.out, "class_f1.png"))

    rng = np.random.default_rng(SEED)
    show_grid(Xte, yte, probs, rng.choice(len(Xte), 12, replace=False),
              "Example predictions (green=correct, red=wrong)", os.path.join(a.out, "examples.png"))
    conf = probs.max(1)
    wrong = np.where(pred != yte)[0]
    wrong = wrong[np.argsort(-conf[wrong])][:12]            # confidently-wrong pehle
    if len(wrong) < 12:                                      # bacha hua: sabse kam-confidence correct
        right = np.where(pred == yte)[0]
        wrong = np.concatenate([wrong, right[np.argsort(conf[right])][:12 - len(wrong)]])
    show_grid(Xte, yte, probs, wrong, "Difficult examples (misclassified / low confidence)",
              os.path.join(a.out, "difficult_examples.png"))
    model.save(os.path.join(a.out, "final.keras"))
    print("Done. Outputs ->", a.out, "\n", DISCLAIMER)


if __name__ == "__main__":
    main()
