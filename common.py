"""Shared config + preprocessing (training aur inference dono yahi use karte hain)."""
import cv2

IMG_SIZE = 48

# GTSRB ke 8 selected classes (class_id -> naam)
SELECTED = {
    1: "Speed limit 30", 2: "Speed limit 50", 13: "Yield", 14: "Stop",
    17: "No entry", 25: "Road work", 33: "Turn right ahead", 38: "Keep right",
}
CLASS_IDS = sorted(SELECTED)

OTHER_CLASS_NAME = "Other / not a sign"
# 9 classes total: index 0..7 are GTSRB signs, index 8 is "Other / not a sign"
CLASS_NAMES = [SELECTED[i] for i in CLASS_IDS] + [OTHER_CLASS_NAME]
ID_TO_IDX = {cid: i for i, cid in enumerate(CLASS_IDS)}
OTHER_IDX = 8
NUM_SIGN_CLASSES = 8
NUM_CLASSES = 9

# 8 Held-out unseen GTSRB classes (never used in train/val of any class, honest OOD test)
HELD_OUT_CLASSES = [3, 4, 9, 11, 12, 18, 26, 35]
HELD_OUT_NAMES = {
    3: "Speed limit 60",
    4: "Speed limit 70",
    9: "No passing",
    11: "Right-of-way at intersection",
    12: "Priority road",
    18: "General caution",
    26: "Traffic signals",
    35: "Ahead only",
}

DISCLAIMER = ("Educational demo only. Model is NOT suitable for real driving "
              "or any safety-critical use.")

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def preprocess(img_bgr):
    """BGR image -> resized + contrast-normalised (CLAHE) RGB uint8 48x48."""
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = _clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
