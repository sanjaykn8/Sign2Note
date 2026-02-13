import json
import csv
from pathlib import Path

WLASL_JSON = Path("data/wlasl/WLASL_v0.3.json")
FEATURE_DIR = Path("data/features")
OUT_CSV = Path("data/index.csv")

with open(WLASL_JSON, "r") as f:
    data = json.load(f)

rows = []

for entry in data:
    gloss = entry["gloss"]
    for inst in entry["instances"]:
        vid = inst["video_id"]
        feature_file = FEATURE_DIR / f"{vid}.npy"
        if feature_file.exists():
            rows.append([vid, gloss])

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["video_id", "label"])
    writer.writerows(rows)

print("Index created:", len(rows))
