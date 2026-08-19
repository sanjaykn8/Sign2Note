import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wlasl_json", default="data/wlasl/WLASL_v0.3.json")
    ap.add_argument("--feature_dir", default="data/features")
    ap.add_argument("--out_csv", default="data/index.csv")
    ap.add_argument("--vocab_json", default="config/vocab.json")
    ap.add_argument("--max_classes", type=int, default=20,
                    help="Keep the N most represented glosses for the MVP.")
    ap.add_argument("--min_samples", type=int, default=20)
    args = ap.parse_args()

    with open(args.wlasl_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    available = []
    counts = Counter()
    for entry in data:
        gloss = entry["gloss"]
        vids = []
        for inst in entry["instances"]:
            vid = str(inst["video_id"])
            if (Path(args.feature_dir) / f"{vid}.npy").exists():
                vids.append(vid)
        if vids:
            counts[gloss] += len(vids)
            available.append((gloss, vids))

    selected = [g for g, c in counts.most_common() if c >= args.min_samples][:args.max_classes]
    if not selected:
        raise SystemExit("No classes met --min_samples. Extract more features or lower the threshold.")
    selected_set = set(selected)
    label2id = {label: i for i, label in enumerate(selected)}

    rows = []
    for gloss, vids in available:
        if gloss not in selected_set:
            continue
        rows.extend((vid, gloss) for vid in vids)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "label"])
        w.writerows(rows)

    Path(args.vocab_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.vocab_json).write_text(
        json.dumps({"label2id": label2id, "id2label": {str(i): l for l, i in label2id.items()},
                    "counts": {g: counts[g] for g in selected}}, indent=2),
        encoding="utf-8",
    )
    print(f"Index created: {len(rows)} samples across {len(selected)} classes")
    print("Classes:", ", ".join(selected))


if __name__ == "__main__":
    main()
