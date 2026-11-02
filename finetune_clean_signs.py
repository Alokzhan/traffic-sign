import os, glob, cv2, numpy as np, tensorflow as tf
from tensorflow import keras
from common import CLASS_NAMES, NUM_CLASSES, IMG_SIZE, preprocess
from ml_classifier import ml_hybrid
from api import smart_square_crop

print("1. Loading converged final model...")
m = keras.models.load_model("outputs/final.keras")
# Freeze nothing or keep all trainable with low lr
m.compile(optimizer=keras.optimizers.Adam(1e-4),
          loss="sparse_categorical_crossentropy",
          metrics=["accuracy"])

print("2. Generating synthetic icon diversity from sample_signs/...")
X_synth, y_synth = [], []
rng = np.random.default_rng(42)

for f in sorted(os.listdir("sample_signs")):
    if not (f.endswith(".png") or f.endswith(".jpg")):
        continue
    cid = int(f.split("_")[0])
    img_path = os.path.join("sample_signs", f)
    raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        continue
    
    # Generate 80 variations of this canonical sign
    for _ in range(80):
        # Random background if RGBA
        if raw.ndim == 3 and raw.shape[2] == 4:
            b, g, r, a = cv2.split(raw)
            alpha = a.astype(float) / 255.0
            bg_choice = rng.choice([255, 220, 180, 128, 70])
            bg = np.full_like(b, bg_choice)
            b_m = (b * alpha + bg * (1.0 - alpha)).astype(np.uint8)
            g_m = (g * alpha + bg * (1.0 - alpha)).astype(np.uint8)
            r_m = (r * alpha + bg * (1.0 - alpha)).astype(np.uint8)
            merged = cv2.merge([b_m, g_m, r_m])
        else:
            merged = raw.copy()
            
        # Random slight brightness / contrast
        scale = float(rng.uniform(0.85, 1.15))
        shift = float(rng.uniform(-20, 20))
        adj = np.clip(merged.astype(float) * scale + shift, 0, 255).astype(np.uint8)
        
        # Smart crop + preprocess
        crop = smart_square_crop(adj)
        proc = preprocess(crop)
        X_synth.append(proc)
        y_synth.append(cid)

X_synth = np.array(X_synth, dtype=np.uint8)
y_synth = np.array(y_synth, dtype=np.int32)
print(f"Generated {len(X_synth)} augmented clean/web sign samples across {len(set(y_synth))} classes.")

print("3. Mixing with 6,000 real GTSRB road samples from cache...")
data = np.load("outputs/dataset_cache.npz")
X_real = data["X"]
y_real = data["y"]
idx = rng.choice(len(X_real), 6000, replace=False)

X_train = np.concatenate([X_synth, X_real[idx]], axis=0)
y_train = np.concatenate([y_synth, y_real[idx]], axis=0)

perm = rng.permutation(len(X_train))
X_train = X_train[perm]
y_train = y_train[perm]
print(f"Total fine-tuning dataset: {len(X_train)} samples.")

print("4. Fine-tuning multi-scale CNN for 4 epochs (learning rate 1e-4)...")
m.fit(X_train, y_train, batch_size=64, epochs=4, verbose=1)

m.save("outputs/final.keras")
m.save("outputs/best.keras")
print("Saved updated outputs/final.keras and outputs/best.keras!")

print("5. Retraining Calibrated ML Classifier on updated features...")
feat_m = keras.Model(m.inputs, m.get_layer("latent_features").output)
emb = feat_m.predict(X_train[:8000], batch_size=128, verbose=0)
ml_hybrid.train_on_embeddings(emb, y_train[:8000])
print("Updated outputs/ml_classifier.pkl successfully!")

