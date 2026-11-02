"""Download GTSRB dataset from Kaggle and retrain the model.
Usage: python download_and_retrain.py
"""
import os
import sys
import zipfile
import shutil
import subprocess

GTSRB_DIR = "gtsrb"


def download_gtsrb():
    """Download GTSRB dataset."""
    print("=" * 60)
    print("GTSRB Dataset Download")
    print("=" * 60)

    if os.path.isdir(os.path.join(GTSRB_DIR, "Train")) and os.path.exists(os.path.join(GTSRB_DIR, "Test.csv")):
        print(f"Dataset already exists in '{GTSRB_DIR}/'. Skipping download.")
        return True

    # Try kagglehub first
    try:
        import kagglehub
        print("Downloading GTSRB via kagglehub...")
        path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
        print(f"Downloaded to: {path}")

        # Copy to gtsrb directory
        os.makedirs(GTSRB_DIR, exist_ok=True)
        for item in os.listdir(path):
            src = os.path.join(path, item)
            dst = os.path.join(GTSRB_DIR, item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        print(f"Dataset copied to '{GTSRB_DIR}/'")
        return True
    except ImportError:
        print("kagglehub not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        import kagglehub
        print("Downloading GTSRB via kagglehub...")
        path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
        print(f"Downloaded to: {path}")

        os.makedirs(GTSRB_DIR, exist_ok=True)
        for item in os.listdir(path):
            src = os.path.join(path, item)
            dst = os.path.join(GTSRB_DIR, item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        print(f"Dataset copied to '{GTSRB_DIR}/'")
        return True
    except Exception as e:
        print(f"kagglehub failed: {e}")
        print("\nAlternative: Manually download from:")
        print("  https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
        print(f"  Extract into '{GTSRB_DIR}/' folder with Train/ and Test.csv")
        return False


def retrain():
    """Run training with improved settings."""
    print("\n" + "=" * 60)
    print("Starting Retraining with Improved Settings")
    print("  - IMG_SIZE: 48 → 64 (better digit resolution)")
    print("  - CLAHE: clipLimit 2.0 → 3.0 (better contrast)")
    print("  - Sharpening: unsharp mask added")
    print("  - Augmentation: less aggressive (preserve digits)")
    print("  - Epochs: 25 → 35")
    print("=" * 60)

    cmd = [
        sys.executable, "train.py",
        "--data", GTSRB_DIR,
        "--out", "outputs",
        "--epochs", "35",
        "--batch", "64"
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    print("\n✅ Retraining complete! New model saved to outputs/final.keras")


if __name__ == "__main__":
    if download_gtsrb():
        retrain()
    else:
        print("\n❌ Dataset download fail hua. Pehle manually download karein, phir:")
        print(f"   python train.py --data {GTSRB_DIR} --out outputs --epochs 35")
