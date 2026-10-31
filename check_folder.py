"""Check all images in a folder with the traffic-sign classifier.
Usage:
    python check_folder.py --model outputs/final.keras --folder real_test_non_signs
    python check_folder.py --model outputs/final_v1.keras --folder real_test_non_signs
"""
import os, argparse
import cv2, numpy as np
from tensorflow import keras
from common import CLASS_NAMES, preprocess, DISCLAIMER


def entropy(p):
    return float(-(p * np.log(p + 1e-9)).sum() / np.log(len(p)))


def main():
    ap = argparse.ArgumentParser(description="Evaluate traffic sign model on a folder of images.")
    ap.add_argument("--model", required=True, help="Path to .keras model file")
    ap.add_argument("--folder", required=True, help="Folder containing test images")
    ap.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold (default 0.70)")
    a = ap.parse_args()

    assert os.path.exists(a.model), f"Model not found: {a.model}"
    assert os.path.isdir(a.folder), f"Folder not found: {a.folder}"

    model = keras.models.load_model(a.model)
    n_out = model.output_shape[-1]
    classes = CLASS_NAMES[:n_out]
    has_other_class = (n_out == 9)

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    files = sorted([f for f in os.listdir(a.folder) if f.lower().endswith(valid_exts)])
    if not files:
        print(f"No image files found in {a.folder}")
        return

    print("=" * 80)
    print(f"Model: {a.model} ({n_out} classes: {'9-class with Other' if has_other_class else '8-class closed-set'})")
    print(f"Folder: {a.folder} ({len(files)} images) | Acceptance Threshold: {a.threshold:.0%}")
    print("=" * 80)
    header = f"{'File':<32} | {'Prediction':<22} | {'Conf':>6} | {'Ent':>5} | {'Verdict'}"
    print(header)
    print("-" * 80)

    n_accepted_signs = 0
    n_other = 0
    n_uncertain = 0

    for f in files:
        path = os.path.join(a.folder, f)
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"{f:<32} | [FAILED TO READ]")
            continue
        rgb = preprocess(bgr)
        x = rgb[None].astype("float32")
        probs = model.predict(x, verbose=0)[0]
        top = int(probs.argmax())
        conf = float(probs[top])
        ent = entropy(probs)
        pred_name = classes[top]

        # Determine verdict
        if has_other_class and top == 8:
            verdict = "REJECTED (Other / not a sign)"
            n_other += 1
        elif conf < a.threshold:
            verdict = "REJECTED (UNCERTAIN)"
            n_uncertain += 1
        else:
            verdict = f"FALSE ACCEPT -> {pred_name}"
            n_accepted_signs += 1

        print(f"{f:<32} | {pred_name:<22} | {conf*100:5.1f}% | {ent:5.2f} | {verdict}")

    print("-" * 80)
    print(f"Summary on {len(files)} non-sign images:")
    print(f"  - Wrongly accepted as one of 8 signs (conf >= {a.threshold:.0%}): {n_accepted_signs}/{len(files)} ({n_accepted_signs/len(files)*100:.1f}%)")
    if has_other_class:
        print(f"  - Correctly identified as 'Other / not a sign':          {n_other}/{len(files)} ({n_other/len(files)*100:.1f}%)")
    print(f"  - Rejected via low confidence (< {a.threshold:.0%}):             {n_uncertain}/{len(files)} ({n_uncertain/len(files)*100:.1f}%)")
    print("=" * 80)
    print(DISCLAIMER)


if __name__ == "__main__":
    main()

