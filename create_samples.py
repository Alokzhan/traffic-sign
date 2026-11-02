import os, shutil
from common import ALL_GTSRB_CLASSES

meta = "C:/Users/OM/.cache/kagglehub/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/versions/1/Meta"
out = "sample_signs"
os.makedirs(out, exist_ok=True)

for cid, name in ALL_GTSRB_CLASSES.items():
    clean_name = name.replace(" ", "_").replace(">", "gt").replace("/", "_")
    src = f"{meta}/{cid}.png"
    dst = f"{out}/{cid:02d}_{clean_name}.png"
    if os.path.exists(src):
        shutil.copyfile(src, dst)

# Add 2 sample non-signs from real_test_non_signs
non_sign_src = "real_test_non_signs/real_traffic_light_red_1.jpg"
if os.path.exists(non_sign_src):
    shutil.copyfile(non_sign_src, f"{out}/43_Other_traffic_light_red.jpg")

print(f"Exported {len(os.listdir(out))} sample test images to {out}/")

