"""Traffic-sign classifier (All 43 GTSRB Signs + 1 Other Class = 44 Classes Total).
Deep Residual CNN with skip connections, batch normalization, and dual pooling.
Run: python train.py --data /path/to/gtsrb --out outputs --epochs 30 --batch 64
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
    CLASS_IDS, CLASS_NAMES, ID_TO_IDX, OTHER_IDX, NUM_CLASSES,
    IMG_SIZE, DISCLAIMER, preprocess
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------- 1. Data Augmentation & Synthesis ----------
def apply_resolution_diversity(img_rgb, rng):
    """Randomly downscales or blurs images so the CNN does not cheat by resolution differences."""
    p = rng.random()
    if p < 0.20:  # 35→20%: kam aggressive, warna digits blur ho jaate hain
        low_res = int(rng.integers(36, 52))  # 20→36 min: digits readable rahein
        small = cv2.resize(img_rgb, (low_res, low_res), interpolation=cv2.INTER_AREA)
        interp = cv2.INTER_LINEAR if rng.random() > 0.5 else cv2.INTER_CUBIC
        return cv2.resize(small, (IMG_SIZE, IMG_SIZE), interpolation=interp)
    elif p < 0.40:
        k = int(rng.choice([3, 5]))
        return cv2.GaussianBlur(img_rgb, (k, k), 0)
    return img_rgb


def generate_traffic_light(rng):
    """Synthetic traffic light in dark housing against outdoor/textured background."""
    bg_type = int(rng.integers(0, 5))
    if bg_type == 0:
        img = np.full((48, 48, 3), [int(rng.integers(200, 255)), int(rng.integers(180, 220)), int(rng.integers(120, 160))], dtype=np.uint8)
    elif bg_type == 1:
        img = np.full((48, 48, 3), [int(x) for x in rng.integers(40, 90, size=3)], dtype=np.uint8)
    elif bg_type == 2:
        img = np.full((48, 48, 3), [int(x) for x in rng.integers(200, 245, size=3)], dtype=np.uint8)
    else:
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
    """Synthetic circular non-sign distractor: coins, colored discs, patterns."""
    bg = np.full((48, 48, 3), [int(x) for x in rng.integers(50, 220, size=3)], dtype=np.uint8)
    cx, cy = int(rng.integers(18, 30)), int(rng.integers(18, 30))
    r = int(rng.integers(10, 20))
    c_type = int(rng.integers(0, 4))
    if c_type == 0:
        col = (int(rng.integers(150, 255)), 0, int(rng.integers(150, 255)))
    elif c_type == 1:
        col = (int(rng.integers(150, 255)), int(rng.integers(150, 255)), 0)
    elif c_type == 2:
        col = (int(rng.integers(40, 80)), int(rng.integers(120, 180)), int(rng.integers(180, 230)))
    else:
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

    print("Loading all 43 GTSRB sign classes...")
    for cid in CLASS_IDS:
        folder = os.path.join(root, "Train", str(cid))
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, f))
            if img is None:
                continue
            proc = preprocess(img)
            proc = apply_resolution_diversity(proc, rng)
            X.append(proc)
            y.append(cid)
            parts = os.path.splitext(f)[0].split("_")
            track = parts[-2] if len(parts) >= 3 else parts[0]
            groups.append(f"{cid}_{track}")

    print(f"Loaded {len(X)} traffic sign images across 43 classes.")

    # 44th Class: "Other / not a sign" (index 43) ~1,000 images
    print("Loading 44th class 'Other / not a sign'...")
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
        print("  - Added 500 CIFAR-100 real non-sign images.")

    for i in range(500):
        img = generate_traffic_light(rng) if rng.random() > 0.35 else generate_circular_distractor(rng)
        proc = preprocess(img)
        proc = apply_resolution_diversity(proc, rng)
        X.append(proc)
        y.append(OTHER_IDX)
        groups.append(f"synth_{i // 15}")
    print("  - Added 500 synthetic hard negatives (traffic lights / distractors).")

    return np.array(X), np.array(y), np.array(groups)


def load_test(root):
    print("Loading test set...")
    df = pd.read_csv(os.path.join(root, "Test.csv"))
    X, y = [], []
    for _, r in df.iterrows():
        img = cv2.imread(os.path.join(root, r.Path))
        if img is None:
            continue
        X.append(preprocess(img))
        y.append(int(r.ClassId))

    # Add test samples for 44th class ("Other / not a sign"): ~300 images
    rng = np.random.default_rng(SEED + 99)
    cifar_test_path = "data_cifar/test"
    if os.path.exists(cifar_test_path):
        with open(cifar_test_path, "rb") as f:
            d_cifar = pickle.load(f, encoding="bytes")
        x_cifar_test = d_cifar[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        cifar_test_idx = rng.choice(len(x_cifar_test), 150, replace=False)
        for idx in cifar_test_idx:
            bgr = cv2.cvtColor(x_cifar_test[idx], cv2.COLOR_RGB2BGR)
            X.append(preprocess(bgr))
            y.append(OTHER_IDX)

    for _ in range(150):
        img = generate_traffic_light(rng) if rng.random() > 0.4 else generate_circular_distractor(rng)
        X.append(preprocess(img))
        y.append(OTHER_IDX)

    return np.array(X), np.array(y)


# ---------- 3. Deep Residual Architecture ----------
def res_block(x, filters, stride=1, name=None):
    shortcut = x
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=name)(y)
    y = layers.BatchNormalization()(y)

    out = layers.add([shortcut, y])
    return layers.ReLU()(out)


def build_strong_model(n_classes=44):
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    # Efficient data augmentation on CPU
    aug = keras.Sequential([
        layers.RandomRotation(0.04),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomZoom(0.06),
    ], name="augment")
    x = aug(inp)
    x = layers.Rescaling(1 / 255.0)(x)

    # Stage 1: Edge & Low-level feature extraction (48x48 -> 24x24)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    s1 = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(s1)

    # Stage 2: Geometric shape & contour extraction (24x24 -> 12x12)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    s2 = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(s2)

    # Stage 3: Digit, symbol & pictogram extraction (12x12 -> 6x6)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False, name="last_conv")(x)
    x = layers.BatchNormalization()(x)
    s3 = layers.ReLU()(x)

    # Multi-Scale Fusion: Branch from Stage 2 (shape) + Branch from Stage 3 (digits)
    pool_s2 = layers.GlobalAveragePooling2D()(s2)
    gap_s3 = layers.GlobalAveragePooling2D()(s3)
    gmp_s3 = layers.GlobalMaxPooling2D()(s3)
    fused = layers.concatenate([pool_s2, gap_s3, gmp_s3])

    # Dense Latent Feature Representation
    feat = layers.Dense(256, activation="relu", name="latent_features")(fused)
    feat = layers.BatchNormalization()(feat)
    feat = layers.Dropout(0.4)(feat)

    out = layers.Dense(n_classes, activation="softmax")(feat)
    m = keras.Model(inp, out, name="expert_multiscale_cnn")
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
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=90, ha="center", fontsize=6)
    ax.set_yticklabels(CLASS_NAMES, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (Held-out test)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_class_f1(report, path):
    f1 = [report[n]["f1-score"] for n in CLASS_NAMES if n in report]
    names = [n for n in CLASS_NAMES if n in report]
    fig, ax = plt.subplots(figsize=(15, 4.5))
    colors = ["#3b82f6" if not n.startswith("Other") else "#f59e0b" for n in names]
    ax.bar(names, f1, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("Class-wise F1 across 44 Classes")
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def show_grid(X, y, probs, idx, title, path):
    cols = 6
    rows = max(1, int(np.ceil(len(idx) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.8))
    axes = np.array(axes).reshape(-1)
    for a in axes:
        a.axis("off")
    for a, i in zip(axes, idx):
        p = probs[i].argmax()
        ok = p == y[i]
        a.imshow(X[i])
        a.set_title(f"T: {CLASS_NAMES[y[i]]}\nP: {CLASS_NAMES[p]} {probs[i][p]:.0%}",
                    fontsize=6, color="green" if ok else "red")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------- 5. Main Execution ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--epochs", type=int, default=35)  # 25→35: digit distinction ke liye zyada training
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    v1_path = os.path.join(a.out, "final_v1.keras")
    final_path = os.path.join(a.out, "final.keras")
    if not os.path.exists(v1_path) and os.path.exists(final_path):
        shutil.copy(final_path, v1_path)
        print(f"Backed up baseline: {final_path} -> {v1_path}")

    # Load dataset
    X, y, g = load_train(a.data)
    Xte, yte = load_test(a.data)

    tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED).split(X, y, g))
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    print(f"train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

    cw = compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=ytr)

    # Check for existing checkpoint to resume training
    best_path = os.path.join(a.out, "best.keras")
    model = None
    if os.path.exists(best_path):
        try:
            prev_m = keras.models.load_model(best_path)
            if prev_m.output_shape[-1] == len(CLASS_NAMES) and prev_m.input_shape[1] == IMG_SIZE:
                print(f"Resuming from existing checkpoint: {best_path}")
                model = prev_m
                model.compile(optimizer=keras.optimizers.Adam(3e-4),
                              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        except Exception as e:
            print(f"Could not resume checkpoint ({e}), building fresh model...")

    if model is None:
        model = build_strong_model(len(CLASS_NAMES))
    model.summary()

    cbs = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5),
        keras.callbacks.ModelCheckpoint(os.path.join(a.out, "best.keras"), save_best_only=True,
                                        monitor="val_accuracy"),
    ]

    print("Starting Deep Residual CNN Training (44 Classes)...")
    h = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=a.epochs, batch_size=a.batch,
                  class_weight=dict(enumerate(cw)), callbacks=cbs)

    # Train Calibrated ML Classifier on latent deep representations
    print("\nFitting Calibrated ML Classifier on latent deep representations...")
    try:
        dense_layers = [l for l in model.layers if "dense" in l.name.lower()]
        target_layer = dense_layers[0] if len(dense_layers) > 1 else model.layers[-2]
        feat_extractor = keras.Model(inputs=model.inputs, outputs=target_layer.output)

        rng = np.random.default_rng(SEED)
        ml_sub_idx = rng.choice(len(Xtr), min(6000, len(Xtr)), replace=False)
        Xtr_emb = feat_extractor.predict(Xtr[ml_sub_idx], batch_size=128, verbose=0)
        from ml_classifier import ml_hybrid
        ml_hybrid.train_on_embeddings(Xtr_emb, ytr[ml_sub_idx])
        print("ML Hybrid Ensemble trained and saved to outputs/ml_classifier.pkl successfully!")
    except Exception as e:
        print("ML Hybrid training skipped due to:", e)

    probs = model.predict(Xte, verbose=0)
    pred = probs.argmax(1)
    rep = classification_report(yte, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, pred)
    print("\n44-Class Classification Report (Test Set):")
    print(classification_report(yte, pred, target_names=CLASS_NAMES, zero_division=0))

    overall_acc = float((pred == yte).mean())
    sign_mask = yte < 43
    sign_acc = float((pred[sign_mask] == yte[sign_mask]).mean())
    other_acc = float((pred[~sign_mask] == yte[~sign_mask]).mean())

    print(f"All 43 traffic sign classes test accuracy: {sign_acc*100:.2f}%")
    print(f"44th 'Other / not a sign' accuracy:        {other_acc*100:.2f}%")
    print(f"Overall 44-class test accuracy:            {overall_acc*100:.2f}%")

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
    if len(wrong) > 0:
        wrong = wrong[np.argsort(-conf[wrong])][:12]
    if len(wrong) < 12:
        right = np.where(pred == yte)[0]
        wrong = np.concatenate([wrong, right[np.argsort(conf[right])][:12 - len(wrong)]])
    show_grid(Xte, yte, probs, wrong, "Difficult examples (misclassified / low confidence)",
              os.path.join(a.out, "difficult_examples.png"))

    model.save(os.path.join(a.out, "final.keras"))
    print(f"Done! Saved strong 44-class model to {os.path.join(a.out, 'final.keras')}")
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
