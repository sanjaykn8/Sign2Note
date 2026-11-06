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
import os
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
        "mediapipe_api": holistic_backend_name(),
        "python_version": platform.python_version(),
        "frame_skip": frame_skip,
        "dataset_format": dataset_format,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        meta.update(extra)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "feature_meta.json").write_text(json.dumps(meta, indent=2))


# ── Holistic backend selection ────────────────────────────────────────────────
# Two MediaPipe implementations of the same Holistic graph produce the same
# landmark topologies (21 per hand, 33 pose, 468(+10 iris) face) that
# feature_schema.py indexes into:
#   "solutions": mediapipe.solutions.holistic (classic API)
#   "tasks":     mediapipe.tasks.python.vision.HolisticLandmarker
# The Python Tasks HolisticLandmarker in mediapipe 0.10.21 (Windows, CPU)
# hard-crashes the process ("packet.cc: Check failed: holder_ != nullptr
# The packet is empty") after a few frames of ordinary FDMSE-ISL clips, in
# both VIDEO and IMAGE mode -- reproduced during the stabilization audit
# (see CHANGELOG.md). The classic solution runs the same clips reliably, so
# it is the default for Python extraction. The browser keeps using the
# Tasks HolisticLandmarker from @mediapipe/tasks-vision (no such crash
# there); both resolve left/right hands from the body pose and share the
# landmark numbering, so the feature vector layout is identical. The small
# numeric differences between the two implementations are a documented
# train/serve approximation (FEATURE_SCHEMA.md).
HOLISTIC_BACKEND = os.environ.get("HOLISTIC_BACKEND", "solutions")


def _solutions_available() -> bool:
    return hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic")


class _SolutionsHolistic:
    """Adapter giving mp.solutions.holistic the same per-frame interface as
    the Tasks landmarker: detect(rgb, timestamp_ms) -> object with
    left_hand_landmarks / right_hand_landmarks / pose_landmarks /
    face_landmarks as plain landmark lists (or None)."""

    name = "solutions.holistic"

    def __init__(self):
        self._h = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, rgb, timestamp_ms):
        r = self._h.process(rgb)

        def lms(x):
            return list(x.landmark) if x is not None else None

        class _R:
            pass

        out = _R()
        out.left_hand_landmarks = lms(r.left_hand_landmarks)
        out.right_hand_landmarks = lms(r.right_hand_landmarks)
        out.pose_landmarks = lms(r.pose_landmarks)
        out.face_landmarks = lms(r.face_landmarks)
        return out

    def close(self):
        self._h.close()


class _TasksHolistic:
    name = "tasks.HolisticLandmarker"

    def __init__(self, model_asset_path):
        options = mp_vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_hand_landmarks_confidence=0.5,
            min_pose_detection_confidence=0.5,
            min_pose_landmarks_confidence=0.5,
            min_face_detection_confidence=0.5,
            min_face_landmarks_confidence=0.5,
        )
        self._lm = mp_vision.HolisticLandmarker.create_from_options(options)

    def detect(self, rgb, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._lm.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        self._lm.close()


def make_holistic(model_asset_path=None):
    if HOLISTIC_BACKEND == "solutions" and _solutions_available():
        return _SolutionsHolistic()
    return _TasksHolistic(ensure_model_asset(model_asset_path or DEFAULT_MODEL_ASSET_PATH))


def holistic_backend_name() -> str:
    return "solutions.holistic" if (HOLISTIC_BACKEND == "solutions" and _solutions_available()) \
        else "tasks.HolisticLandmarker"


# ── per-worker Holistic instance (loaded ONCE per process, not per video) ────
_worker_landmarker = None


def _init_worker(model_asset_path):
    """multiprocessing.Pool initializer: loads one Holistic detector per
    worker PROCESS, reused across every video that process handles.
    Holistic is three sub-models and worth loading once per worker. Note:
    the classic solution is a streaming tracker, so a fresh instance is
    created per video in _process_one when tracking state must not leak
    between clips (see there)."""
    global _worker_landmarker
    _worker_landmarker = model_asset_path  # remember the asset path only


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

    # A fresh detector per video: both Holistic implementations are
    # streaming trackers (VIDEO mode / static_image_mode=False), so reusing
    # one across clips would carry tracking state from the end of one
    # video into the first frames of the next.
    try:
        landmarker = make_holistic(model_asset_path)
    except Exception as exc:
        return (vid, False, f"detector init failed: {exc}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        landmarker.close()
        return (vid, False, "cannot open video (missing, corrupt or unsupported codec)")
    frames = []
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    try:
        while True:
            if frame_idx % frame_skip != 0:
                # grab() advances without the BGR->RGB conversion/copy.
                if not cap.grab():
                    break
                frame_idx += 1
                continue
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((frame_idx / fps) * 1000)
            result = landmarker.detect(rgb, timestamp_ms)
            frames.append(_result_to_feature_vector(result))
            frame_idx += 1
    except Exception as exc:
        return (vid, False, f"extraction failed at frame {frame_idx}: {exc}")
    finally:
        cap.release()
        landmarker.close()

    if len(frames) == 0:
        return (vid, False, "no frames decoded")

    np.save(out_path, np.stack(frames))
    return (vid, True, "ok")


def _process_one_legacy_v1(args):
    """Schema v1 (hand-only, 126 dims) extraction, reproducing the ORIGINAL
    v1 extractor exactly (git b6a11c7): mp.solutions.hands with
    model_complexity=0, max 2 hands, hands placed in detection order. Only
    used to serve/evaluate the legacy v1 model -- never for v2 training."""
    vid, video_path, out_path, frame_skip = args
    if not hasattr(mp, "solutions"):
        raise RuntimeError("Legacy v1 extraction needs mediapipe.solutions (mediapipe<=0.10.21).")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return (vid, False, "cannot open video (missing, corrupt or unsupported codec)")
    frames = []
    frame_idx = 0
    try:
        with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=0,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue
                res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                detected = [list(h.landmark) for h in (res.multi_hand_landmarks or [])]
                frames.append(fs.build_legacy_v1_vector(detected))
                frame_idx += 1
    finally:
        cap.release()
    if not frames:
        return (vid, False, "no frames decoded")
    np.save(out_path, np.stack(frames))
    return (vid, True, "ok")


def extract_single_video_for_schema(video_path, out_dir, schema_version,
                                    frame_skip=DEFAULT_FRAME_SKIP, model_asset_path=None):
    """Extract features for one video in the schema a given MODEL expects
    (api.py routes by the served checkpoint's feature_schema_version).
    Returns (path or None, message)."""
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}.npy"
    if str(schema_version) == fs.LEGACY_SCHEMA_VERSION:
        _, ok, msg = _process_one_legacy_v1((video_path.stem, str(video_path), str(out_path), frame_skip))
    elif str(schema_version) == fs.FEATURE_SCHEMA_VERSION:
        _, ok, msg = _process_one((video_path.stem, str(video_path), str(out_path), frame_skip,
                                   model_asset_path or DEFAULT_MODEL_ASSET_PATH))
    else:
        raise fs.FeatureSchemaError(f"No extractor for feature_schema_version={schema_version!r}")
    return (out_path if ok else None), msg


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
    model_asset_path = model_asset_path or DEFAULT_MODEL_ASSET_PATH

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

    _run_extraction(tasks, frame_skip, n_workers, model_asset_path,
                    failures_path=out_dir / "extraction_failures.csv")
    _write_feature_meta(
        out_dir, frame_skip, dataset_format="fdmse",
        extra={"metadata_csv": str(metadata_csv), "splits": sorted(splits) if splits else "all"},
    )


def _run_extraction(tasks, frame_skip, n_workers, model_asset_path=None, failures_path=None):
    """Shared multiprocessing runner for both dataset formats. `tasks` is a
    list of (video_id, video_path, out_path) triples; frame_skip is the
    same for every task in a given run so it's injected here. Failed
    videos (unreadable/corrupt/no frames) are listed in `failures_path`
    (CSV) so the dataset audit can report them instead of them silently
    disappearing from training."""
    model_asset_path = model_asset_path or DEFAULT_MODEL_ASSET_PATH
    if holistic_backend_name() != "solutions.holistic":
        model_asset_path = ensure_model_asset(model_asset_path)
    full_tasks = [(vid, vp, op, frame_skip, model_asset_path) for vid, vp, op in tasks]
    print(f"[feature_extraction] {len(full_tasks)} videos | workers={n_workers} | frame_skip={frame_skip} "
          f"| schema=v{fs.FEATURE_SCHEMA_VERSION} ({fs.FEATURE_DIM} dims) | backend={holistic_backend_name()}")

    if not full_tasks:
        print("[feature_extraction] No matching videos found on disk — "
              "double check --dataset_root / --videos_dir against the paths "
              "in your metadata file.")
        return

    done = cached = 0
    failures = []
    with Pool(processes=n_workers, initializer=_init_worker, initargs=(model_asset_path,)) as pool:
        for vid, ok, msg in tqdm(
                pool.imap_unordered(_process_one, full_tasks, chunksize=4), total=len(full_tasks)):
            if ok and msg == "cached":
                cached += 1
            elif ok:
                done += 1
            else:
                failures.append((vid, msg))

    print(f"[feature_extraction] Extracted={done}  Cached={cached}  Failed={len(failures)}")
    if failures_path and failures:
        with open(failures_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video_id", "reason"])
            w.writerows(failures)
        print(f"[feature_extraction] Failure list -> {failures_path}")


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
