"""Traffic-sign classifier (9-class with 'Other / not a sign'): data -> train -> evaluate -> reports.
Run: python train.py --data /path/to/gtsrb --out outputs --epochs 40
Dataset layout (Kaggle GTSRB): <data>/Train/<class_id>/*.png, <data>/Test.csv, <data>/Test/*.png
"""
import os, json, argparse, shutil, pickle
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
from common import (
    CLASS_IDS, CLASS_NAMES, ID_TO_IDX, OTHER_IDX, HELD_OUT_CLASSES,
    IMG_SIZE, DISCLAIMER, preprocess
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------- 1. Data Generation & Augmentation ----------
def apply_resolution_diversity(img_rgb, rng):
    """Randomly downscales or blurs images so the CNN cannot use resolution artifacts as a shortcut."""
    p = rng.random()
    if p < 0.35:
        low_res = int(rng.integers(20, 36))
        small = cv2.resize(img_rgb, (low_res, low_res), interpolation=cv2.INTER_AREA)
        interp = cv2.INTER_LINEAR if rng.random() > 0.5 else cv2.INTER_CUBIC
        return cv2.resize(small, (IMG_SIZE, IMG_SIZE), interpolation=interp)
    elif p < 0.60:
        k = int(rng.choice([3, 5]))
        return cv2.GaussianBlur(img_rgb, (k, k), 0)
    return img_rgb


def generate_traffic_light(rng):
    """Synthetic traffic light in dark housing against outdoor/textured background."""
    bg_type = int(rng.integers(0, 5))
    if bg_type == 0:  # sky
        img = np.full((48, 48, 3), [int(rng.integers(200, 255)), int(rng.integers(180, 220)), int(rng.integers(120, 160))], dtype=np.uint8)
    elif bg_type == 1:  # dark tarmac / road
        img = np.full((48, 48, 3), [int(x) for x in rng.integers(40, 90, size=3)], dtype=np.uint8)
    elif bg_type == 2:  # light gray / building
        img = np.full((48, 48, 3), [int(x) for x in rng.integers(200, 245, size=3)], dtype=np.uint8)
    else:  # gradient
        v = np.linspace(int(rng.integers(50, 100)), int(rng.integers(150, 220)), 48, dtype=np.uint8)
        img = np.tile(v[:, None, None], (1, 48, 3)).copy()

    box_w = int(rng.integers(14, 22))
    box_h = int(rng.integers(30, 44))
    x0 = int(rng.integers(6, 48 - box_w - 4))
    y0 = int(rng.integers(2, 48 - box_h - 2))
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (25, 25, 25), -1)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (60, 60, 60), 1)

    lamp_r = int(rng.integers(3, 5))
    cx = int(x0 + box_w // 2)
    lamp_ys = [int(y0 + box_h // 4), int(y0 + box_h // 2), int(y0 + 3 * box_h // 4)]

    lit = int(rng.integers(0, 3))
    colors = {
        0: (0, 0, int(rng.integers(210, 255))),                                  # Red (BGR)
        1: (0, int(rng.integers(190, 240)), 255),                                # Yellow/Amber (BGR)
        2: (int(rng.integers(40, 80)), int(rng.integers(210, 255)), int(rng.integers(20, 60)))  # Green (BGR)
    }

    for i, ly in enumerate(lamp_ys):
        if i == lit:
            glow_c = tuple(int(c * 0.45) for c in colors[i])
            cv2.circle(img, (cx, ly), lamp_r + 2, glow_c, -1)
            cv2.circle(img, (cx, ly), lamp_r, colors[i], -1)
        else:
            cv2.circle(img, (cx, ly), lamp_r, (40, 40, 40), -1)

    noise = rng.normal(0, float(rng.uniform(3, 10)), (48, 48, 3))
    return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)


def generate_circular_distractor(rng):
    """Synthetic circular non-sign distractor: coins, colored discs, concentric patterns."""
    bg = np.full((48, 48, 3), [int(x) for x in rng.integers(50, 220, size=3)], dtype=np.uint8)
    cx, cy = int(rng.integers(18, 30)), int(rng.integers(18, 30))
    r = int(rng.integers(10, 20))
    c_type = int(rng.integers(0, 4))
    if c_type == 0:  # purple/magenta
        col = (int(rng.integers(150, 255)), 0, int(rng.integers(150, 255)))
    elif c_type == 1:  # cyan / teal
        col = (int(rng.integers(150, 255)), int(rng.integers(150, 255)), 0)
    elif c_type == 2:  # metallic bronze / coin
        col = (int(rng.integers(40, 80)), int(rng.integers(120, 180)), int(rng.integers(180, 230)))
    else:  # dark disc
        col = (int(rng.integers(30, 80)), int(rng.integers(30, 80)), int(rng.integers(30, 80)))

    cv2.circle(bg, (cx, cy), r, col, -1)
    if rng.random() > 0.4:
        cv2.circle(bg, (cx, cy), max(2, r - 5), tuple(int(x) for x in rng.integers(0, 255, size=3)), -1)

    noise = rng.normal(0, float(rng.uniform(3, 12)), (48, 48, 3))
    return np.clip(bg.astype(float) + noise, 0, 255).astype(np.uint8)


# ---------- 2. Data Loading ----------
def load_train(root):
    X, y, groups = [], [], []
    rng = np.random.default_rng(SEED)

    # 1. Load the 8 selected sign classes
    print("Loading 8 selected GTSRB sign classes...")
    for cid in CLASS_IDS:
        folder = os.path.join(root, "Train", str(cid))
        for f in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, f))
            if img is None:
                continue
            proc = preprocess(img)
            proc = apply_resolution_diversity(proc, rng)
            X.append(proc)
            y.append(ID_TO_IDX[cid])
            parts = os.path.splitext(f)[0].split("_")
            track = parts[-2] if len(parts) >= 3 else parts[0]
            groups.append(f"{cid}_{track}")

    # 2. Load "Other / not a sign" (index 8) ~1,600 images:
    print("Loading 9th class 'Other / not a sign'...")
    # (a) Unused GTSRB classes (excluding the 8 held-out honest test classes): ~600 images
    unused_cids = [c for c in range(43) if c not in CLASS_IDS and c not in HELD_OUT_CLASSES]
    unused_tracks = []
    for cid in unused_cids:
        folder = os.path.join(root, "Train", str(cid))
        if not os.path.isdir(folder):
            continue
        tracks_in_cls = {}
        for f in os.listdir(folder):
            parts = os.path.splitext(f)[0].split("_")
            track = parts[-2] if len(parts) >= 3 else parts[0]
            tracks_in_cls.setdefault(track, []).append((cid, f))
        for trk, files in tracks_in_cls.items():
            unused_tracks.append((cid, trk, files))

    rng.shuffle(unused_tracks)
    gtsrb_other_count = 0
    for cid, trk, files in unused_tracks:
        if gtsrb_other_count >= 600:
            break
        folder = os.path.join(root, "Train", str(cid))
        for _, f in files:
            img = cv2.imread(os.path.join(folder, f))
            if img is None:
                continue
            proc = preprocess(img)
            proc = apply_resolution_diversity(proc, rng)
            X.append(proc)
            y.append(OTHER_IDX)
            groups.append(f"{cid}_{trk}")
            gtsrb_other_count += 1
    print(f"  - Loaded {gtsrb_other_count} unused GTSRB images for 'Other'")

    # (b) CIFAR-100 real non-sign training images: ~500 images
    cifar_path = "data_cifar/train"
    if os.path.exists(cifar_path):
        with open(cifar_path, "rb") as f:
            d_cifar = pickle.load(f, encoding="bytes")
        x_cifar_train = d_cifar[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        cifar_idx = rng.choice(len(x_cifar_train), 500, replace=False)
        for i, idx in enumerate(cifar_idx):
            bgr = cv2.cvtColor(x_cifar_train[idx], cv2.COLOR_RGB2BGR)
            proc = preprocess(bgr)
            proc = apply_resolution_diversity(proc, rng)
            X.append(proc)
            y.append(OTHER_IDX)
            groups.append(f"cifar_{i // 15}")
        print("  - Loaded 500 CIFAR-100 images for 'Other'")

    # (c) Synthetic hard negatives (traffic lights + circular distractors): ~500 images
    for i in range(500):
        img = generate_traffic_light(rng) if rng.random() > 0.35 else generate_circular_distractor(rng)
        proc = preprocess(img)
        proc = apply_resolution_diversity(proc, rng)
        X.append(proc)
        y.append(OTHER_IDX)
        groups.append(f"synth_{i // 15}")
    print("  - Generated 500 synthetic hard negatives (traffic lights / distractors)")

    return np.array(X), np.array(y), np.array(groups)


def load_test(root):
    print("Loading held-out test set...")
    df = pd.read_csv(os.path.join(root, "Test.csv"))
    # Selected 8 sign classes
    df_sel = df[df.ClassId.isin(CLASS_IDS)]
    X, y = [], []
    for _, r in df_sel.iterrows():
        img = cv2.imread(os.path.join(root, r.Path))
        if img is None:
            continue
        X.append(preprocess(img))
        y.append(ID_TO_IDX[r.ClassId])

    # Held-out test samples for Class 8 ("Other / not a sign"): ~480 images
    rng = np.random.default_rng(SEED + 99)
    # (a) Unused GTSRB classes (excluding HELD_OUT_CLASSES)
    unused_cids = [c for c in range(43) if c not in CLASS_IDS and c not in HELD_OUT_CLASSES]
    df_unused = df[df.ClassId.isin(unused_cids)]
    sample_gtsrb_test = df_unused.sample(min(160, len(df_unused)), random_state=SEED + 99)
    for _, r in sample_gtsrb_test.iterrows():
        img = cv2.imread(os.path.join(root, r.Path))
        if img is not None:
            X.append(preprocess(img))
            y.append(OTHER_IDX)

    # (b) CIFAR-100 test split: ~160 images
    cifar_test_path = "data_cifar/test"
    if os.path.exists(cifar_test_path):
        with open(cifar_test_path, "rb") as f:
            d_cifar = pickle.load(f, encoding="bytes")
        x_cifar_test = d_cifar[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        cifar_test_idx = rng.choice(len(x_cifar_test), 160, replace=False)
        for idx in cifar_test_idx:
            bgr = cv2.cvtColor(x_cifar_test[idx], cv2.COLOR_RGB2BGR)
            X.append(preprocess(bgr))
            y.append(OTHER_IDX)

    # (c) Synthetic hard negatives test: ~160 images
    for _ in range(160):
        img = generate_traffic_light(rng) if rng.random() > 0.4 else generate_circular_distractor(rng)
        X.append(preprocess(img))
        y.append(OTHER_IDX)

    return np.array(X), np.array(y)


# ---------- 3. Model Architecture ----------
def build_model(n_classes=9):
    # Data augmentation layer: rotation, zoom, translation, contrast (no horizontal flip)
    aug = keras.Sequential([
        layers.RandomRotation(0.06),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomContrast(0.2),
    ], name="augment")
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = aug(inp)
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
    m = keras.Model(inp, out, name="sign_cnn_v2")
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


# ---------- 4. Plots ----------
def plot_history(h, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for a, k in zip(ax, ("loss", "accuracy")):
        a.plot(h.history[k], label="train")
        a.plot(h.history["val_" + k], label="val")
        a.set_title(k)
        a.set_xlabel("epoch")
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (held-out test)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_class_f1(report, path):
    f1 = [report[n]["f1-score"] for n in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    colors = ["#3b82f6"] * 8 + ["#f59e0b"]
    ax.bar(CLASS_NAMES, f1, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("Class-wise F1 (held-out test)")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def show_grid(X, y, probs, idx, title, path):
    cols = 6
    rows = max(1, int(np.ceil(len(idx) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6))
    axes = np.array(axes).reshape(-1)
    for a in axes:
        a.axis("off")
    for a, i in zip(axes, idx):
        p = probs[i].argmax()
        ok = p == y[i]
        a.imshow(X[i])
        a.set_title(f"T: {CLASS_NAMES[y[i]]}\nP: {CLASS_NAMES[p]} {probs[i][p]:.0%}",
                    fontsize=7, color="green" if ok else "red")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------- 5. Main Execution ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # 1. Back up baseline final.keras to final_v1.keras if it does not exist yet
    v1_path = os.path.join(a.out, "final_v1.keras")
    final_path = os.path.join(a.out, "final.keras")
    if not os.path.exists(v1_path) and os.path.exists(final_path):
        shutil.copy(final_path, v1_path)
        print(f"Backed up baseline: {final_path} -> {v1_path}")
    elif os.path.exists(v1_path):
        print(f"Existing baseline preserved: {v1_path}")

    # 2. Load dataset
    X, y, g = load_train(a.data)
    Xte, yte = load_test(a.data)

    # Group split: physical sign track ids remain disjoint between train and val
    tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED).split(X, y, g))
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    print(f"train={len(Xtr)} val={len(Xva)} test(held-out)={len(Xte)}")

    cw = compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=ytr)
    model = build_model(len(CLASS_NAMES))
    model.summary()

    cbs = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5),
        keras.callbacks.ModelCheckpoint(os.path.join(a.out, "best.keras"), save_best_only=True,
                                        monitor="val_accuracy"),
    ]

    print("Starting CNN training...")
    h = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=a.epochs, batch_size=a.batch,
                  class_weight=dict(enumerate(cw)), callbacks=cbs)

    # 3. Evaluation on held-out test set
    probs = model.predict(Xte, verbose=0)
    pred = probs.argmax(1)
    rep = classification_report(yte, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, pred)
    print("\n9-Class Classification Report (Held-out test set):")
    print(classification_report(yte, pred, target_names=CLASS_NAMES, zero_division=0))

    # Evaluate 8 sign classes accuracy separately for verification
    sign_mask = yte < 8
    sign_acc = float((pred[sign_mask] == yte[sign_mask]).mean())
    overall_acc = float((pred == yte).mean())
    other_acc = float((pred[~sign_mask] == yte[~sign_mask]).mean())
    print(f"8 original sign classes accuracy: {sign_acc*100:.2f}%")
    print(f"9th 'Other / not a sign' accuracy: {other_acc*100:.2f}%")
    print(f"Overall 9-class test accuracy:     {overall_acc*100:.2f}%")

    out_metrics = {
        "test_accuracy": overall_acc,
        "sign_classes_accuracy": sign_acc,
        "other_class_accuracy": other_acc,
        "report": rep,
        "disclaimer": DISCLAIMER
    }
    json.dump(out_metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)

    plot_history(h, os.path.join(a.out, "history.png"))
    plot_confusion(cm, os.path.join(a.out, "confusion_matrix.png"))
    plot_class_f1(rep, os.path.join(a.out, "class_f1.png"))

    rng = np.random.default_rng(SEED)
    show_grid(Xte, yte, probs, rng.choice(len(Xte), 12, replace=False),
              "Example predictions (green=correct, red=wrong)", os.path.join(a.out, "examples.png"))
    conf = probs.max(1)
    wrong = np.where(pred != yte)[0]
    wrong = wrong[np.argsort(-conf[wrong])][:12]
    if len(wrong) < 12:
        right = np.where(pred == yte)[0]
        wrong = np.concatenate([wrong, right[np.argsort(conf[right])][:12 - len(wrong)]])
    show_grid(Xte, yte, probs, wrong, "Difficult examples (misclassified / low confidence)",
              os.path.join(a.out, "difficult_examples.png"))

    model.save(os.path.join(a.out, "final.keras"))
    print(f"Done! Saved new model to {os.path.join(a.out, 'final.keras')}")
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
