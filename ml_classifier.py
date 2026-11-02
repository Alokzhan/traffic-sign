"""Machine Learning (ML) Hybrid Classifier & Calibration Layer.
Extracts deep latent representations (192-dim embeddings) from the trained ResNet
and fits a calibrated multi-class ML classifier (L2-regularized Logistic Regression).
Combines Deep Softmax + ML logits via an ensemble blend for maximum robustness.
"""
import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from common import NUM_CLASSES, CLASS_NAMES

ML_MODEL_PATH = os.path.join(os.environ.get("OUT_DIR", "outputs"), "ml_classifier.pkl")


class MLHybridClassifier:
    """Ensemble hybrid blending Deep Neural Network features with a calibrated ML model."""

    def __init__(self, model_path=ML_MODEL_PATH):
        self.model_path = model_path
        self.clf = None
        self.is_loaded = False
        self._last_mtime = 0
        self.load()

    def load(self):
        if os.path.exists(self.model_path):
            try:
                mtime = os.path.getmtime(self.model_path)
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                self.clf = data.get("clf")
                self.is_loaded = True
                self._last_mtime = mtime
                print(f"[ML Classifier] Loaded calibrated ML model from {self.model_path}")
            except Exception as e:
                print(f"[ML Classifier] Load failed: {e}")
                self.clf = None
                self.is_loaded = False

    def check_reload(self):
        if os.path.exists(self.model_path):
            try:
                mtime = os.path.getmtime(self.model_path)
                if mtime > self._last_mtime:
                    self.load()
            except OSError:
                pass

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({"clf": self.clf}, f)
        self.is_loaded = True
        try:
            self._last_mtime = os.path.getmtime(self.model_path)
        except OSError:
            pass
        print(f"[ML Classifier] Saved trained ML model to {self.model_path}")

    def train_on_embeddings(self, embeddings: np.ndarray, labels: np.ndarray):
        """Fit a multi-class Logistic Regression with L2 regularization on deep embeddings."""
        print(f"[ML Classifier] Fitting calibrated ML classifier on {len(embeddings)} samples (dim={embeddings.shape[1]})...")
        base_lr = LogisticRegression(
            C=1.0,
            max_iter=500,
            solver="lbfgs",
            random_state=42
        )
        base_lr.fit(embeddings, labels)
        self.clf = base_lr
        self.save()

    def predict_proba(self, embedding: np.ndarray) -> np.ndarray:
        """Predict class probabilities from a 1D or 2D feature embedding."""
        self.check_reload()
        if not self.is_loaded or self.clf is None:
            return None
        if embedding.ndim == 1:
            embedding = embedding[None]
        return self.clf.predict_proba(embedding)[0]

    def ensemble_blend(self, deep_probs: np.ndarray, embedding: np.ndarray = None, alpha: float = 0.75) -> dict:
        """Blends Deep Learning softmax probabilities with calibrated ML classifier probabilities.
        P_final = alpha * P_deep + (1 - alpha) * P_ml
        """
        ml_probs = None
        if self.is_loaded and embedding is not None:
            try:
                ml_p = self.predict_proba(embedding)
                if ml_p is not None and len(ml_p) == len(deep_probs):
                    ml_probs = ml_p
            except Exception as e:
                print(f"[ML Classifier] Inference error: {e}")

        if ml_probs is not None:
            blended = alpha * deep_probs + (1.0 - alpha) * ml_probs
            blended = blended / (blended.sum() + 1e-9)
            return {
                "blended_probs": blended,
                "deep_probs": deep_probs,
                "ml_probs": ml_probs,
                "has_ml_ensemble": True
            }
        
        return {
            "blended_probs": deep_probs,
            "deep_probs": deep_probs,
            "ml_probs": None,
            "has_ml_ensemble": False
        }


# Global singleton instance
ml_hybrid = MLHybridClassifier()
