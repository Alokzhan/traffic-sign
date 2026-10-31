"""Shared config + preprocessing (training aur inference dono yahi use karte hain)."""
import cv2

IMG_SIZE = 48

# GTSRB ke 8 selected classes (class_id -> naam)
SELECTED = {
    1: "Speed limit 30", 2: "Speed limit 50", 13: "Yield", 14: "Stop",
    17: "No entry", 25: "Road work", 33: "Turn right ahead", 38: "Keep right",
}
CLASS_IDS = sorted(SELECTED)
CLASS_NAMES = [SELECTED[i] for i in CLASS_IDS]
ID_TO_IDX = {cid: i for i, cid in enumerate(CLASS_IDS)}

DISCLAIMER = ("Educational demo only. Model is NOT suitable for real driving "
              "or any safety-critical use.")

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def preprocess(img_bgr):
    """BGR image -> resized + contrast-normalised (CLAHE) RGB uint8 48x48."""
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = _clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
