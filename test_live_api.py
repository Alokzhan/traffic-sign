import requests

url = "http://127.0.0.1:8000/api/predict"

test_files = [
    ("User Uploaded 50 Sign", "outputs/debug_uploaded_raw.png"),
    ("Speed limit 20", "sample_signs/00_Speed_limit_20.png"),
    ("Speed limit 30", "sample_signs/01_Speed_limit_30.png"),
    ("Speed limit 50", "sample_signs/02_Speed_limit_50.png"),
    ("Speed limit 60", "sample_signs/03_Speed_limit_60.png"),
    ("Speed limit 70", "sample_signs/04_Speed_limit_70.png"),
    ("Speed limit 80", "sample_signs/05_Speed_limit_80.png"),
    ("Speed limit 100", "sample_signs/07_Speed_limit_100.png"),
    ("Speed limit 120", "sample_signs/08_Speed_limit_120.png"),
]

print("=== LIVE API PREDICTION TEST (PORT 8000) ===")
for label, path in test_files:
    try:
        with open(path, "rb") as f:
            resp = requests.post(url, files={"file": (path, f, "image/png")})
        if resp.status_code == 200:
            d = resp.json()
            name = d.get("name")
            conf = d.get("conf", 0) * 100
            dp = d.get("deep_conf", 0) * 100
            ml = (d.get("ml_conf") or 0) * 100
            status = "[OK]" if label in name or name in label else "[MISMATCH]"
            print(f"{status} {label:<24} -> {name:<22} ({conf:.1f}%) [Deep: {dp:.1f}%, ML: {ml:.1f}%]")
        else:
            print(f"[ERR] {label}: {resp.status_code}")
    except Exception as e:
        print(f"[FAIL] {label}: {e}")
