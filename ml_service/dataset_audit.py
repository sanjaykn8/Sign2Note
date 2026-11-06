"""
dataset_audit.py -- audit an FDMSE-ISL metadata CSV before training.

Checks:
  - rows / classes / samples per class, per split
  - missing video files on disk
  - (optional --check_videos) videos that cannot be opened or report 0 frames
  - invalid labels (empty / whitespace-only / non-string)
  - duplicate rows, duplicate video files, the same video listed in two splits
  - class imbalance (min/max/mean samples per class, classes missing from a split)
  - signer independence: signer ids are parsed from the path
    (data/<signer>/<view>/...) and the script reports whether any signer
    appears in more than one split
  - (optional --feature_dir) which rows have extracted features, and the
    extraction failure list written by feature_extraction.py

Writes a JSON report and a Markdown summary. Read-only with respect to the
dataset itself.

Usage:
    python dataset_audit.py --metadata_csv data/data_meta/metadata.csv \\
        --dataset_root data/FDMSE-ISL --feature_dir data/features_v2 \\
        --out_json reports/dataset_audit.json --out_md reports/dataset_audit.md
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def audit(metadata_csv, dataset_root, feature_dir=None, check_videos=False):
    df = pd.read_csv(metadata_csv)
    report = {"metadata_csv": str(metadata_csv), "num_rows": int(len(df))}
    required = {"video_dir", "video_name", "class", "split"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise SystemExit(f"metadata is missing columns {sorted(missing_cols)}")

    df["video_id"] = df["video_name"].map(lambda v: Path(str(v)).stem)
    parts = df["video_dir"].astype(str).str.split("/")
    df["signer"] = parts.map(lambda p: p[1] if len(p) > 2 else None)
    df["view"] = parts.map(lambda p: p[2] if len(p) > 3 else None)

    bad_label = df["class"].isna() | (df["class"].astype(str).str.strip() == "")
    report["invalid_labels"] = df.loc[bad_label, "video_id"].tolist()
    report["num_classes"] = int(df.loc[~bad_label, "class"].nunique())
    report["splits"] = {s: int(n) for s, n in df["split"].value_counts().items()}
    report["views"] = {str(v): int(n) for v, n in df["view"].value_counts().items()}

    report["duplicate_rows"] = int(df.duplicated(subset=["video_dir", "class", "split"]).sum())
    report["duplicate_video_files"] = int(df["video_dir"].duplicated().sum())
    multi_split = df.groupby("video_dir")["split"].nunique()
    report["videos_in_multiple_splits"] = multi_split[multi_split > 1].index.tolist()

    counts = df.groupby("class").size()
    report["samples_per_class"] = {
        "min": int(counts.min()), "max": int(counts.max()),
        "mean": round(float(counts.mean()), 3), "median": float(counts.median()),
        "classes_below_20": {c: int(n) for c, n in counts[counts < 20].items()},
    }
    per_split_classes = df.groupby("split")["class"].nunique().to_dict()
    report["classes_per_split"] = {k: int(v) for k, v in per_split_classes.items()}
    all_classes = set(df["class"])
    report["classes_missing_from_split"] = {
        s: sorted(all_classes - set(g["class"])) for s, g in df.groupby("split")
    }

    signers = {s: sorted(g["signer"].dropna().unique().tolist()) for s, g in df.groupby("split")}
    report["signers_per_split"] = signers
    seen = {}
    overlap = []
    for split, ss in signers.items():
        for s in ss:
            if s in seen and seen[s] != split:
                overlap.append(s)
            seen.setdefault(s, split)
    report["signers_in_multiple_splits"] = sorted(set(overlap))
    report["signer_independent_splits"] = not overlap

    root = Path(dataset_root)
    exists = df["video_dir"].map(lambda p: (root / p).exists())
    report["missing_video_files"] = df.loc[~exists, "video_dir"].tolist()

    if check_videos:
        import cv2
        unreadable = []
        for p in df.loc[exists, "video_dir"]:
            cap = cv2.VideoCapture(str(root / p))
            ok = cap.isOpened() and (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0
            cap.release()
            if not ok:
                unreadable.append(p)
        report["unreadable_videos"] = unreadable

    if feature_dir:
        fdir = Path(feature_dir)
        has = df["video_id"].map(lambda v: (fdir / f"{v}.npy").exists())
        report["features"] = {
            "feature_dir": str(fdir),
            "rows_with_features": int(has.sum()),
            "rows_without_features": int((~has).sum()),
            "per_split_with_features": {s: int(n) for s, n in df.loc[has, "split"].value_counts().items()},
        }
        fail = fdir / "extraction_failures.csv"
        if fail.exists():
            report["features"]["extraction_failures"] = pd.read_csv(fail).to_dict("records")
        meta = fdir / "feature_meta.json"
        if meta.exists():
            report["features"]["feature_meta"] = json.loads(meta.read_text(encoding="utf-8"))
    return report


def to_markdown(r):
    lines = [
        "# Dataset audit",
        "",
        f"- Metadata: `{r['metadata_csv']}`",
        f"- Rows: {r['num_rows']} | classes: {r['num_classes']} | views: {r['views']}",
        f"- Split sizes: {r['splits']}",
        f"- Classes per split: {r['classes_per_split']}",
        f"- Samples per class: min {r['samples_per_class']['min']}, max {r['samples_per_class']['max']}, "
        f"mean {r['samples_per_class']['mean']}; classes with < 20 samples: "
        f"{r['samples_per_class']['classes_below_20'] or 'none'}",
        f"- Classes missing from a split: "
        + ", ".join(f"{s}: {len(c)}" for s, c in r["classes_missing_from_split"].items()),
        f"- Invalid labels: {len(r['invalid_labels'])}",
        f"- Duplicate rows: {r['duplicate_rows']} | duplicate video files: {r['duplicate_video_files']} | "
        f"videos listed in more than one split: {len(r['videos_in_multiple_splits'])}",
        f"- Missing video files: {len(r['missing_video_files'])}",
    ]
    if "unreadable_videos" in r:
        lines.append(f"- Unreadable / zero-frame videos: {len(r['unreadable_videos'])}")
    lines += [
        "",
        "## Signer independence",
        "",
        f"- Signer-independent splits: **{r['signer_independent_splits']}**",
    ]
    for s, ss in r["signers_per_split"].items():
        lines.append(f"- {s}: {len(ss)} signers ({', '.join(ss)})")
    if r["signers_in_multiple_splits"]:
        lines.append(f"- Signers in more than one split: {r['signers_in_multiple_splits']}")
    if "features" in r:
        f = r["features"]
        lines += ["", "## Extracted features", "",
                  f"- Directory: `{f['feature_dir']}`",
                  f"- Rows with features: {f['rows_with_features']} / {r['num_rows']} "
                  f"(per split: {f['per_split_with_features']})",
                  f"- Extraction failures: {len(f.get('extraction_failures', []))}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata_csv", default="data/data_meta/metadata.csv")
    ap.add_argument("--dataset_root", default="data/FDMSE-ISL")
    ap.add_argument("--feature_dir", default=None)
    ap.add_argument("--check_videos", action="store_true", help="Open every video (slow).")
    ap.add_argument("--out_json", default="reports/dataset_audit.json")
    ap.add_argument("--out_md", default="reports/dataset_audit.md")
    a = ap.parse_args()
    r = audit(a.metadata_csv, a.dataset_root, a.feature_dir, a.check_videos)
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(r, indent=2), encoding="utf-8")
    Path(a.out_md).write_text(to_markdown(r), encoding="utf-8")
    print(to_markdown(r))


if __name__ == "__main__":
    main()
