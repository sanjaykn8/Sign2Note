"""
feature_extraction.py
Extracts MediaPipe hand keypoints from sign-language videos.

Speed optimisations vs the original:
  - FRAME_SKIP 4 → 8  (process half as many frames)
  - multiprocessing Pool  (~N_CPU× parallel workers)
  - early-exit on already-processed videos
  - accepts CLI args so api.py can call it properly
"""

import os
import csv
import json
import argparse
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_VIDEOS_DIR = "data/wlasl/videos"
DEFAULT_OUT_DIR    = "data/features"
DEFAULT_WLASL_JSON = "data/wlasl/WLASL_v0.3.json"
DEFAULT_FRAME_SKIP = 8          # was 4; 2x extraction speed
DEFAULT_MAX_VIDEOS = None       # None = process all
DEFAULT_WORKERS    = min(cpu_count(), 8)

# FDMSE-ISL layout: data/FDMSE-ISL/<video_dir from metadata.csv>, metadata in
# data/data_meta/metadata*.csv (id,video_dir,video_name,class,split).
DEFAULT_FDMSE_ROOT     = "data/FDMSE-ISL"
DEFAULT_FDMSE_METADATA = "data/data_meta/metadata_400.csv"


# ── keypoint helper ───────────────────────────────────────────────────────────
def _extract_keypoints(results):
    def lm_to_arr(lm):
        if lm is None:
            return np.zeros((21, 3), dtype=np.float32)
        return np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)

    left  = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)
    if results.multi_hand_landmarks:
        for i, h in enumerate(results.multi_hand_landmarks):
            if i == 0: left  = lm_to_arr(h)
            else:      right = lm_to_arr(h)
    return np.concatenate([left.flatten(), right.flatten()])   # (126,)


# ── per-video worker (must be top-level for multiprocessing) ──────────────────
def _process_one(args):
    """Worker: extract keypoints from one video and save .npy."""
    vid, video_path, out_path, frame_skip = args
    video_path = Path(video_path)
    out_path   = Path(out_path)

    if out_path.exists():
        return (vid, True, "cached")

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_idx = 0
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            img     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img)
            frames.append(_extract_keypoints(results))
            frame_idx += 1

    cap.release()

    if len(frames) == 0:
        return (vid, False, "no frames")

    np.save(out_path, np.stack(frames))
    return (vid, True, "ok")


# ── public helper: process a SINGLE video (used by api.py) ───────────────────
def extract_single_video(video_path, out_dir, frame_skip=DEFAULT_FRAME_SKIP):
    """
    Extract keypoints from one video file.
    Returns path to the saved .npy file, or None on failure.
    """
    video_path = Path(video_path)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}.npy"

    _, ok, _ = _process_one((video_path.stem, str(video_path), str(out_path), frame_skip))
    return out_path if ok else None


# ── batch extraction (WLASL dataset) ─────────────────────────────────────────
def extract_dataset(videos_dir, out_dir, wlasl_json,
                    frame_skip=DEFAULT_FRAME_SKIP,
                    max_videos=DEFAULT_MAX_VIDEOS,
                    n_workers=DEFAULT_WORKERS):

    videos_dir = Path(videos_dir)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    _run_extraction(tasks, frame_skip, n_workers)


# ── batch extraction (FDMSE-ISL dataset) ─────────────────────────────────────
def extract_fdmse_dataset(metadata_csv, dataset_root, out_dir,
                          frame_skip=DEFAULT_FRAME_SKIP,
                          max_videos=DEFAULT_MAX_VIDEOS,
                          n_workers=DEFAULT_WORKERS,
                          splits=None):
    """Batch-extract keypoints for the FDMSE-ISL dataset.

    `metadata_csv` is one of data/data_meta/metadata*.csv, with columns
    id,video_dir,video_name,class,split. `video_dir` is the path to the
    clip *relative to `dataset_root`* (e.g. dataset_root/data/s0015/front/...).
    The video's filename stem (e.g. "s0015_f_w000842") is globally unique
    across signers/sessions in this dataset, so it's used directly as the
    video_id — this is also what build_index.py expects to match against.

    `splits`, if given (e.g. {"train", "val"}), restricts extraction to
    those rows of the CSV — handy for extracting val/test separately or
    doing a quick partial run.
    """
    dataset_root = Path(dataset_root)
    out_dir      = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    _run_extraction(tasks, frame_skip, n_workers)


def _run_extraction(tasks, frame_skip, n_workers):
    """Shared multiprocessing runner for both dataset formats. `tasks` is a
    list of (video_id, video_path, out_path) triples; frame_skip is the
    same for every task in a given run so it's injected here."""
    full_tasks = [(vid, vp, op, frame_skip) for vid, vp, op in tasks]
    print(f"[feature_extraction] {len(full_tasks)} videos | workers={n_workers} | frame_skip={frame_skip}")

    if not full_tasks:
        print("[feature_extraction] No matching videos found on disk — "
              "double check --dataset_root / --videos_dir against the paths "
              "in your metadata file.")
        return

    done = failed = 0
    with Pool(processes=n_workers) as pool:
        for vid, ok, msg in tqdm(
                pool.imap_unordered(_process_one, full_tasks), total=len(full_tasks)):
            if ok: done   += 1
            else:  failed += 1

    print(f"[feature_extraction] Done={done}  Failed/skipped={failed}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_format", choices=["fdmse", "wlasl"], default="fdmse",
                        help="Which dataset layout to read. Defaults to FDMSE-ISL.")
    # FDMSE-ISL options
    parser.add_argument("--metadata_csv",  default=DEFAULT_FDMSE_METADATA,
                        help="FDMSE-ISL metadata CSV (metadata.csv / metadata_400.csv / "
                             "metadata_atomic.csv / metadata_composite.csv).")
    parser.add_argument("--dataset_root",  default=DEFAULT_FDMSE_ROOT,
                        help="Root the metadata CSV's video_dir column is relative to.")
    parser.add_argument("--splits",        default=None,
                        help="Comma-separated subset of splits to extract, e.g. train,val. "
                             "Default: all rows in the CSV.")
    # WLASL (legacy) options
    parser.add_argument("--videos_dir",  default=DEFAULT_VIDEOS_DIR)
    parser.add_argument("--wlasl_json",  default=DEFAULT_WLASL_JSON)
    # shared options
    parser.add_argument("--out_dir",     default=DEFAULT_OUT_DIR)
    parser.add_argument("--frame_skip",  type=int, default=DEFAULT_FRAME_SKIP)
    parser.add_argument("--max_videos",  type=int, default=DEFAULT_MAX_VIDEOS)
    parser.add_argument("--workers",     type=int, default=DEFAULT_WORKERS)
    a = parser.parse_args()

    if a.dataset_format == "fdmse":
        splits = set(s.strip() for s in a.splits.split(",")) if a.splits else None
        extract_fdmse_dataset(a.metadata_csv, a.dataset_root, a.out_dir,
                              a.frame_skip, a.max_videos, a.workers, splits)
    else:
        extract_dataset(a.videos_dir, a.out_dir, a.wlasl_json,
                        a.frame_skip, a.max_videos, a.workers)
