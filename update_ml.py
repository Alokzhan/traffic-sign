import os, numpy as np, keras
from ml_classifier import ml_hybrid

print("Loading cached dataset...")
data = np.load("outputs/dataset_cache.npz")
X = data["X"]
y = data["y"]

model_file = "outputs/final.keras" if os.path.exists("outputs/final.keras") else "outputs/best.keras"
print(f"Loading converged model {model_file}...")
m = keras.models.load_model(model_file)
latent_layer = m.get_layer("latent_features")
feat_model = keras.Model(m.inputs, latent_layer.output)

print(f"Extracting 256-dim latent embeddings for 15,000 samples from {len(X)} total...")
rng = np.random.default_rng(42)
n_samples = min(15000, len(X))
idx = rng.choice(len(X), n_samples, replace=False)

# X is stored as uint8 RGB (0..255), feat_model expects float32/uint8 directly since Rescaling is in the model
emb = feat_model.predict(X[idx], batch_size=256, verbose=1)

print("Fitting Calibrated ML Classifier on converged embeddings...")
ml_hybrid.train_on_embeddings(emb, y[idx])
print("Successfully updated outputs/ml_classifier.pkl with converged 256 features!")
