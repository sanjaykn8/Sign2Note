"""
feature_extraction.py
Extracts MediaPipe Holistic (hand+body+face) keypoints from sign-language
videos, per the canonical schema in feature_schema.py / FEATURE_SCHEMA.md.

Schema v2 note: this used to use `mp.solutions.hands` (hand-only, 126
dims). That module doesn't exist at all in current `mediapipe` releases
(the legacy Solutions API was dropped; `pip install mediapipe` today ships
Tasks-API-only) -- so this rewrite fixes a latent "crashes on a fresh
install" bug as a side effect of the hand+body+face schema migration, not
just a feature addition. See FEATURE_SCHEMA.md for the full rationale.

Speed characteristics vs the v1 hand-only version:
  - multiprocessing Pool, but now with a per-WORKER (not per-video)
    HolisticLandmarker instance -- Holistic is three sub-models (hand,
    pose, face), so re-creating it for every single video like v1 did for
    the lightweight Hands-only model would be wasteful. See _init_worker().
  - same frame_skip / caching / skip-existing behavior as before.
  - a feature_meta.json is now written once per output directory,
    recording feature_schema_version, feature dim, mediapipe version, and
    the exact extraction config used -- so a stale/incompatible feature
    directory is detectable before it silently corrupts a training run
    (see check_feature_dir_compatible()).
"""

import csv
import json
import platform
import argparse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import feature_schema as fs

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_VIDEOS_DIR = "data/wlasl/videos"
# v2 features are NOT compatible with v1 (data/features) -- see
# FEATURE_SCHEMA.md "History". Default output dir is deliberately
# different so the two can never be silently mixed by accident.
DEFAULT_OUT_DIR    = "data/features_v2"
DEFAULT_WLASL_JSON = "data/wlasl/WLASL_v0.3.json"
DEFAULT_FRAME_SKIP = 8          # unchanged from v1 -- no evidence to change it
DEFAULT_MAX_VIDEOS = None       # None = process all
DEFAULT_WORKERS    = min(cpu_count(), 8)

# FDMSE-ISL layout: data/FDMSE-ISL/<video_dir from metadata.csv>, metadata in
# data/data_meta/metadata*.csv (id,video_dir,video_name,class,split).
DEFAULT_FDMSE_ROOT     = "data/FDMSE-ISL"
# NOTE: metadata_400.csv is a 400-class SUBSET for fast experiments (RULE
# 14). For the full-dataset experiment (RULE 15), pass
# --metadata_csv data/data_meta/metadata.csv explicitly.
DEFAULT_FDMSE_METADATA = "data/data_meta/metadata_400.csv"

# Where the HolisticLandmarker .task model asset is cached locally after
# first download -- mirrors the browser's hand_landmarker.task caching
# pattern (see frontend/src/lib/handLandmarker.ts / TROUBLESHOOTING.md).
DEFAULT_MODEL_ASSET_PATH = "models/mediapipe/holistic_landmarker.task"
# Google's official model CDN. Same model family the browser SDK fetches,
# so both languages run the same underlying detector (see
# FEATURE_SCHEMA.md "Detector"). NOT reachable from every sandboxed/
# offline environment -- if this download fails, point
# --model_asset_path at a manually-downloaded copy instead.
MODEL_ASSET_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)


def ensure_model_asset(path: str = DEFAULT_MODEL_ASSET_PATH) -> str:
    """Downloads the HolisticLandmarker .task model asset to `path` if it
    isn't already there. Returns `path`. Raises a clear error (not a
    confusing urllib traceback) if the download fails and no local copy
    exists -- the caller needs to know to fetch it manually."""
    p = Path(path)
    if p.exists():
        return str(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"[features] Downloading HolisticLandmarker model asset to {p} ...")
    try:
        urllib.request.urlretrieve(MODEL_ASSET_URL, str(p))
    except Exception as e:
        raise RuntimeError(
            f"Could not download the HolisticLandmarker model asset from "
            f"{MODEL_ASSET_URL} ({e}). If this environment can't reach "
            f"Google's model CDN, download it manually on a machine that "
            f"can and place it at {p}, or pass --model_asset_path to point "
            f"at an existing copy."
        ) from e
    return str(p)


def check_feature_dir_compatible(out_dir) -> None:
    """Reads out_dir/feature_meta.json (if present) and raises
    fs.FeatureSchemaError if it was extracted with a different schema
    version than the one this module currently implements. Called before
    a batch run starts writing into an existing directory, and by
    dataset.py/train.py before loading features for training -- see RULE
    12 ("do not silently mix old 126-dim features with new features")."""
    meta_path = Path(out_dir) / "feature_meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    if meta.get("feature_schema_version") != fs.FEATURE_SCHEMA_VERSION:
        raise fs.FeatureSchemaError(
            f"{out_dir} was extracted with feature_schema_version="
            f"{meta.get('feature_schema_version')!r}, but this code "
            f"implements v{fs.FEATURE_SCHEMA_VERSION}. Point --out_dir at "
            f"a fresh directory, or re-extract this one from scratch."
        )
    if meta.get("feature_dim") != fs.FEATURE_DIM:
        raise fs.FeatureSchemaError(
            f"{out_dir}/feature_meta.json says feature_dim="
            f"{meta.get('feature_dim')}, but the current schema expects "
            f"{fs.FEATURE_DIM}. Re-extract into a fresh directory."
        )


def _write_feature_meta(out_dir, frame_skip, dataset_format, extra=None):
    """Writes (or overwrites) out_dir/feature_meta.json describing exactly
    how the features in this directory were produced -- required by RULE
    12/14: a feature directory must be self-describing so a dimension or
    schema mismatch is caught at load time, not deep inside a training
    crash."""
    meta = {
        "feature_schema_version": fs.FEATURE_SCHEMA_VERSION,
        "feature_type": fs.FEATURE_TYPE,
        "feature_dim": fs.FEATURE_DIM,
        "feature_groups": [
            {"name": name, "start": start, "end": end} for name, start, end in fs.FEATURE_GROUPS
        ],
        "mediapipe_version": mp.__version__,
        "mediapipe_api": "tasks.python.vision.HolisticLandmarker",
        "python_version": platform.python_version(),
        "frame_skip": frame_skip,
        "dataset_format": dataset_format,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        meta.update(extra)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "feature_meta.json").write_text(json.dumps(meta, indent=2))


# ── per-worker HolisticLandmarker (loaded ONCE per process, not per video) ────
_worker_landmarker = None


def _init_worker(model_asset_path):
    """multiprocessing.Pool initializer: loads one HolisticLandmarker per
    worker PROCESS, reused across every video that process handles. Unlike
    v1's lightweight Hands model (cheap enough to recreate per-video),
    Holistic is three sub-models and worth loading once per worker."""
    global _worker_landmarker
    base_options = BaseOptions(model_asset_path=model_asset_path)
    options = mp_vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_hand_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
    )
    _worker_landmarker = mp_vision.HolisticLandmarker.create_from_options(options)


def _result_to_feature_vector(result) -> np.ndarray:
    """HolisticLandmarkerResult -> one canonical FEATURE_DIM vector, via
    feature_schema.build_feature_vector (the single shared implementation
    -- see FEATURE_SCHEMA.md)."""
    return fs.build_feature_vector(
        result.left_hand_landmarks,
        result.right_hand_landmarks,
        result.pose_landmarks,
        result.face_landmarks,
    )


# ── per-video worker (must be top-level for multiprocessing) ──────────────────
def _process_one(args):
    """Worker: extract keypoints from one video and save .npy. Uses the
    per-process `_worker_landmarker` set up by _init_worker(); falls back
    to a fresh one-off instance if called directly (e.g. from
    extract_single_video(), outside a Pool)."""
    vid, video_path, out_path, frame_skip, model_asset_path = args
    video_path = Path(video_path)
    out_path   = Path(out_path)

    if out_path.exists():
        try:
            existing_dim = np.load(out_path, mmap_mode="r").shape[-1]
        except Exception:
            existing_dim = None
        if existing_dim == fs.FEATURE_DIM:
            return (vid, True, "cached")
        # Wrong-dimension cached file (e.g. leftover v1 hand-only output) --
        # do NOT silently reuse it (RULE 12). Re-extract.

    landmarker = _worker_landmarker
    owns_landmarker = False
    if landmarker is None:
        _init_worker(model_asset_path)
        landmarker = _worker_landmarker
        owns_landmarker = True

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            frames.append(_result_to_feature_vector(result))
            frame_idx += 1
    finally:
        cap.release()
        if owns_landmarker:
            landmarker.close()

    if len(frames) == 0:
        return (vid, False, "no frames")

    np.save(out_path, np.stack(frames))
    return (vid, True, "ok")


# ── public helper: process a SINGLE video (used by api.py) ───────────────────
def extract_single_video(video_path, out_dir, frame_skip=DEFAULT_FRAME_SKIP,
                         model_asset_path=None):
    """
    Extract keypoints from one video file (also the path api.py's /process
    and /recognize endpoints use for an uploaded video, and what a single
    ad-hoc `python feature_extraction.py --single_video ...` CLI run calls).
    Returns path to the saved .npy file, or None on failure.
    """
    video_path = Path(video_path)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}.npy"
    model_asset_path = ensure_model_asset(model_asset_path or DEFAULT_MODEL_ASSET_PATH)

    _, ok, _ = _process_one((video_path.stem, str(video_path), str(out_path), frame_skip, model_asset_path))
    return out_path if ok else None


# ── batch extraction (WLASL dataset; legacy, kept for backward compat) ───────
def extract_dataset(videos_dir, out_dir, wlasl_json,
                    frame_skip=DEFAULT_FRAME_SKIP,
                    max_videos=DEFAULT_MAX_VIDEOS,
                    n_workers=DEFAULT_WORKERS,
                    model_asset_path=None):

    videos_dir = Path(videos_dir)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    check_feature_dir_compatible(out_dir)

    with open(wlasl_json, "r") as f:
        data = json.load(f)

    tasks = []
    for entry in data:
        for inst in entry["instances"]:
            vid        = inst["video_id"]
            video_file = videos_dir / f"{vid}.mp4"
            out_file   = out_dir / f"{vid}.npy"
            if not video_file.exists():
                continue
            tasks.append((vid, str(video_file), str(out_file)))
            if max_videos and len(tasks) >= max_videos:
                break
        if max_videos and len(tasks) >= max_videos:
            break

    _run_extraction(tasks, frame_skip, n_workers, model_asset_path)
    _write_feature_meta(out_dir, frame_skip, dataset_format="wlasl")


# ── batch extraction (FDMSE-ISL dataset) ─────────────────────────────────────
def extract_fdmse_dataset(metadata_csv, dataset_root, out_dir,
                          frame_skip=DEFAULT_FRAME_SKIP,
                          max_videos=DEFAULT_MAX_VIDEOS,
                          n_workers=DEFAULT_WORKERS,
                          splits=None,
                          model_asset_path=None):
    """Batch-extract keypoints for the FDMSE-ISL dataset.

    `metadata_csv` is one of data/data_meta/metadata*.csv, with columns
    id,video_dir,video_name,class,split. `video_dir` is the path to the
    clip *relative to `dataset_root`* (e.g. dataset_root/data/s0015/front/...).
    The video's filename stem (e.g. "s0015_f_w000842") is globally unique
    across signers/sessions in this dataset, so it's used directly as the
    video_id — this is also what build_index.py expects to match against.

    `splits`, if given (e.g. {"train", "val"}), restricts extraction to
    those rows of the CSV — handy for extracting val/test separately, or
    (per RULE 16) extracting exactly the official split you need for a
    leakage-free final evaluation.

    Pass metadata_csv=".../metadata.csv" (the full dataset, not
    metadata_400.csv) for the full-vocabulary experiment (RULE 15); the
    400-class CSV remains the default so a fast experiment is still one
    command away (RULE 14).
    """
    dataset_root = Path(dataset_root)
    out_dir      = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    check_feature_dir_compatible(out_dir)

    tasks = []
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if splits and row["split"] not in splits:
                continue
            vid        = Path(row["video_name"]).stem
            video_file = dataset_root / row["video_dir"]
            out_file   = out_dir / f"{vid}.npy"
            if not video_file.exists():
                continue
            tasks.append((vid, str(video_file), str(out_file)))
            if max_videos and len(tasks) >= max_videos:
                break

    _run_extraction(tasks, frame_skip, n_workers, model_asset_path)
    _write_feature_meta(
        out_dir, frame_skip, dataset_format="fdmse",
        extra={"metadata_csv": str(metadata_csv), "splits": sorted(splits) if splits else "all"},
    )


def _run_extraction(tasks, frame_skip, n_workers, model_asset_path=None):
    """Shared multiprocessing runner for both dataset formats. `tasks` is a
    list of (video_id, video_path, out_path) triples; frame_skip is the
    same for every task in a given run so it's injected here. Each worker
    process loads its own HolisticLandmarker once via _init_worker (see
    module docstring), rather than per video."""
    model_asset_path = ensure_model_asset(model_asset_path or DEFAULT_MODEL_ASSET_PATH)
    full_tasks = [(vid, vp, op, frame_skip, model_asset_path) for vid, vp, op in tasks]
    print(f"[feature_extraction] {len(full_tasks)} videos | workers={n_workers} | frame_skip={frame_skip} "
          f"| schema=v{fs.FEATURE_SCHEMA_VERSION} ({fs.FEATURE_DIM} dims)")

    if not full_tasks:
        print("[feature_extraction] No matching videos found on disk — "
              "double check --dataset_root / --videos_dir against the paths "
              "in your metadata file.")
        return

    done = failed = 0
    with Pool(processes=n_workers, initializer=_init_worker, initargs=(model_asset_path,)) as pool:
        for vid, ok, msg in tqdm(
                pool.imap_unordered(_process_one, full_tasks), total=len(full_tasks)):
            if ok: done   += 1
            else:  failed += 1

    print(f"[feature_extraction] Done={done}  Failed/skipped={failed}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_format", choices=["fdmse", "wlasl"], default="fdmse",
                        help="Which dataset layout to read. Defaults to FDMSE-ISL.")
    # FDMSE-ISL options
    parser.add_argument("--metadata_csv",  default=DEFAULT_FDMSE_METADATA,
                        help="FDMSE-ISL metadata CSV (metadata.csv = full dataset / "
                             "metadata_400.csv = fast 400-class subset / "
                             "metadata_atomic.csv / metadata_composite.csv).")
    parser.add_argument("--dataset_root",  default=DEFAULT_FDMSE_ROOT,
                        help="Root the metadata CSV's video_dir column is relative to.")
    parser.add_argument("--splits",        default=None,
                        help="Comma-separated subset of official splits to extract, "
                             "e.g. train,val or just test. Default: all rows in the CSV.")
    # WLASL (legacy) options
    parser.add_argument("--videos_dir",  default=DEFAULT_VIDEOS_DIR)
    parser.add_argument("--wlasl_json",  default=DEFAULT_WLASL_JSON)
    # single-video mode (ad-hoc extraction / debugging, outside api.py)
    parser.add_argument("--single_video", default=None,
                        help="Path to one video file. If set, extracts just this video "
                             "into --out_dir and ignores all dataset-wide options above.")
    # shared options
    parser.add_argument("--out_dir",     default=DEFAULT_OUT_DIR,
                        help=f"Output directory for .npy feature files + feature_meta.json. "
                             f"Default is schema-v2-specific ({DEFAULT_OUT_DIR}) so it can "
                             f"never collide with old v1 (hand-only) feature files.")
    parser.add_argument("--frame_skip",  type=int, default=DEFAULT_FRAME_SKIP)
    parser.add_argument("--max_videos",  type=int, default=DEFAULT_MAX_VIDEOS)
    parser.add_argument("--workers",     type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--model_asset_path", default=DEFAULT_MODEL_ASSET_PATH,
                        help="Local path to the HolisticLandmarker .task model asset. "
                             "Downloaded automatically on first use if missing (from "
                             f"{MODEL_ASSET_URL}); pass this to point at a manually "
                             "downloaded copy if your environment can't reach that URL.")
    a = parser.parse_args()

    if a.single_video:
        result = extract_single_video(a.single_video, a.out_dir, a.frame_skip, a.model_asset_path)
        print(f"[feature_extraction] {'wrote ' + str(result) if result else 'FAILED'}")
    elif a.dataset_format == "fdmse":
        splits = set(s.strip() for s in a.splits.split(",")) if a.splits else None
        extract_fdmse_dataset(a.metadata_csv, a.dataset_root, a.out_dir,
                              a.frame_skip, a.max_videos, a.workers, splits,
                              a.model_asset_path)
    else:
        extract_dataset(a.videos_dir, a.out_dir, a.wlasl_json,
                        a.frame_skip, a.max_videos, a.workers, a.model_asset_path)
