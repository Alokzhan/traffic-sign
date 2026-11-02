"""Shared config + preprocessing for 43 GTSRB signs + 1 Other class (44 classes total)."""
import cv2

IMG_SIZE = 64  # 48→64: zyada resolution se digits (3 vs 7) ache se distinguish hote hain

ALL_GTSRB_CLASSES = {
    0: "Speed limit 20",
    1: "Speed limit 30",
    2: "Speed limit 50",
    3: "Speed limit 60",
    4: "Speed limit 70",
    5: "Speed limit 80",
    6: "End of speed limit 80",
    7: "Speed limit 100",
    8: "Speed limit 120",
    9: "No passing",
    10: "No passing for vehicles > 3.5t",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles > 3.5t prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing for vehicles > 3.5t",
}

CLASS_IDS = sorted(ALL_GTSRB_CLASSES.keys())
SELECTED = ALL_GTSRB_CLASSES
ID_TO_IDX = {cid: cid for cid in CLASS_IDS}

OTHER_CLASS_NAME = "Other / not a sign"
OTHER_IDX = 43
NUM_SIGN_CLASSES = 43
NUM_CLASSES = 44

CLASS_NAMES = [ALL_GTSRB_CLASSES[i] for i in range(43)] + [OTHER_CLASS_NAME]

# 8 Held-out unseen GTSRB classes from initial benchmark (for v1 baseline comparisons)
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

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Stronger contrast for digit edges


def preprocess(img_bgr):
    """BGR image -> resized + contrast-normalised (CLAHE) + sharpened RGB uint8 64x64."""
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = _clahe.apply(lab[..., 0])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    # Unsharp mask: digit edges (3 vs 7, 5 vs 6 etc.) ko sharp karta hai
    blurred = cv2.GaussianBlur(rgb, (0, 0), 2.0)
    rgb = cv2.addWeighted(rgb, 1.5, blurred, -0.5, 0)
    return rgb

