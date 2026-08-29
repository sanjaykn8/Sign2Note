import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def build_from_wlasl(wlasl_json, feature_dir, max_classes, min_samples):
    with open(wlasl_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    available = []
    counts = Counter()
    for entry in data:
        gloss = entry["gloss"]
        vids = []
        for inst in entry["instances"]:
            vid = str(inst["video_id"])
            if (Path(feature_dir) / f"{vid}.npy").exists():
                vids.append(vid)
        if vids:
            counts[gloss] += len(vids)
            available.append((gloss, vids))

    selected = [g for g, c in counts.most_common() if c >= min_samples]
    if max_classes and max_classes > 0:
        selected = selected[:max_classes]
    if not selected:
        raise SystemExit("No classes met --min_samples. Extract more features or lower the threshold.")
    selected_set = set(selected)
    label2id = {label: i for i, label in enumerate(selected)}

    rows = []
    for gloss, vids in available:
        if gloss not in selected_set:
            continue
        rows.extend({"video_id": vid, "label": gloss} for vid in vids)

    counts_out = {g: counts[g] for g in selected}
    return rows, label2id, counts_out


def build_from_fdmse(metadata_csv, feature_dir, max_classes, min_samples):
    """FDMSE-ISL metadata CSVs (id,video_dir,video_name,class,split): the
    video's filename stem is the video_id feature_extraction.py's
    extract_fdmse_dataset() saves .npy files under, and `class` is used
    directly as the label — it's already a human-readable gloss string
    (e.g. "Whistle"), no separate vocab lookup needed."""
    feature_dir = Path(feature_dir)
    df = pd.read_csv(metadata_csv)
    df["video_id"] = df["video_name"].apply(lambda v: Path(v).stem)
    df = df[df["video_id"].apply(lambda v: (feature_dir / f"{v}.npy").exists())].copy()

    if df.empty:
        raise SystemExit(
            f"No rows in {metadata_csv} have a matching .npy under {feature_dir}. "
            "Run feature_extraction.py first."
        )

    counts = df["class"].value_counts()
    selected = [c for c in counts.index if counts[c] >= min_samples]
    if max_classes and max_classes > 0:
        selected = selected[:max_classes]
    if not selected:
        raise SystemExit("No classes met --min_samples. Extract more features or lower the threshold.")
    selected_set = set(selected)
    label2id = {label: i for i, label in enumerate(selected)}

    kept = df[df["class"].isin(selected_set)]
    rows = [
        {"video_id": vid, "label": label, "split": split}
        for vid, label, split in zip(kept["video_id"], kept["class"], kept["split"])
    ]

    counts_out = {c: int(counts[c]) for c in selected}
    return rows, label2id, counts_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_format", choices=["fdmse", "wlasl"], default="fdmse",
                    help="Which dataset's metadata to read. Defaults to FDMSE-ISL.")
    # FDMSE-ISL
    ap.add_argument("--metadata_csv", default="data/data_meta/metadata_400.csv",
                    help="FDMSE-ISL metadata CSV (metadata.csv / metadata_400.csv / "
                         "metadata_atomic.csv / metadata_composite.csv).")
    # WLASL (legacy)
    ap.add_argument("--wlasl_json", default="data/wlasl/WLASL_v0.3.json")
    # shared
    ap.add_argument("--feature_dir", default="data/features")
    ap.add_argument("--out_csv", default="data/index.csv")
    ap.add_argument("--vocab_json", default="config/vocab.json")
    ap.add_argument("--max_classes", type=int, default=0,
                    help="Keep only the N most represented classes. 0 = no cap "
                         "(keep every class that meets --min_samples).")
    ap.add_argument("--min_samples", type=int, default=10)
    args = ap.parse_args()

    if args.dataset_format == "fdmse":
        rows, label2id, counts_out = build_from_fdmse(
            args.metadata_csv, args.feature_dir, args.max_classes, args.min_samples)
        fieldnames = ["video_id", "label", "split"]
    else:
        rows, label2id, counts_out = build_from_wlasl(
            args.wlasl_json, args.feature_dir, args.max_classes, args.min_samples)
        fieldnames = ["video_id", "label"]

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    Path(args.vocab_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.vocab_json).write_text(
        json.dumps({"label2id": label2id, "id2label": {str(i): l for l, i in label2id.items()},
                    "counts": counts_out}, indent=2),
        encoding="utf-8",
    )
    print(f"Index created: {len(rows)} samples across {len(label2id)} classes -> {args.out_csv}")
    print(f"Vocab written -> {args.vocab_json}")


if __name__ == "__main__":
    main()
