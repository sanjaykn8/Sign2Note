"""
End-to-end check against a RUNNING gateway (backend/server.js) + ML service.

  1. GET /health and /model/meta
  2. single real dataset clips -> POST /upload -> glosses + notes
  3. a "continuous" video stitched from N clips of one held-out signer (known
     ground-truth order) -> POST /recognize/video, comparing the old
     windowing (64 real frames, stride 12, frame_skip 8) with the current
     defaults; reports sequence accuracy (edit distance) and latency
  4. glosses -> /notes in all three styles and /transcript

This is NOT a substitute for the official-split evaluation (evaluate.py);
it checks that the deployed pipeline works on real video and measures it.

Usage (from the repo root, services running):
    py -3.11 ml_service/tools/e2e_check.py \\
        --gateway http://127.0.0.1:3001 --signer s0004 --n_clips 6 --out reports/e2e_legacy.json
"""
import argparse
import json
import random
import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import httpx as requests


def edit_distance(a, b):
    d = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return d[-1][-1]


def stitch(paths, out_path, gap_seconds=0.5):
    """Concatenate clips (same size/fps), with a short still gap between
    signs (repeating the last frame), like a pause between signs."""
    caps = [cv2.VideoCapture(str(p)) for p in paths]
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(caps[0].get(3)), int(caps[0].get(4))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for cap in caps:
        last = None
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if fr.shape[1] != w or fr.shape[0] != h:
                fr = cv2.resize(fr, (w, h))
            writer.write(fr)
            last = fr
        for _ in range(int(gap_seconds * fps)):
            if last is not None:
                writer.write(last)
        cap.release()
    writer.release()
    return fps


def post_video(url, path, data, headers):
    with open(path, "rb") as f:
        t = time.time()
        r = requests.post(url, files={"file": (Path(path).name, f, "video/mp4")}, data=data, headers=headers, timeout=1800)
        return r, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://127.0.0.1:3001")
    ap.add_argument("--token", default="")
    ap.add_argument("--metadata_csv", default="data/data_meta/metadata.csv")
    ap.add_argument("--dataset_root", default="data/FDMSE-ISL")
    ap.add_argument("--signer", default="s0004", help="a TEST-split signer")
    ap.add_argument("--n_clips", type=int, default=6)
    ap.add_argument("--n_single", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/e2e_check.json")
    a = ap.parse_args()
    H = {"Authorization": f"Bearer {a.token}"} if a.token else {}
    report = {}

    report["health"] = requests.get(f"{a.gateway}/health", timeout=20).json()
    meta = requests.get(f"{a.gateway}/model/meta", headers=H, timeout=60).json()
    report["model"] = {k: meta.get(k) for k in ("model_version", "feature_schema_version", "input_dim", "num_classes",
                                                  "architecture", "best_val_accuracy", "used_official_split", "ready")}
    vocab = set(meta["label2id"])

    m = pd.read_csv(a.metadata_csv)
    m = m[(m.video_dir.str.contains(f"/{a.signer}/")) & (m["class"].isin(vocab))]
    rng = random.Random(a.seed)
    rows = m.sample(n=a.n_clips + a.n_single, random_state=a.seed).to_dict("records")
    root = Path(a.dataset_root)

    singles = []
    for r in rows[:a.n_single]:
        resp, dt = post_video(f"{a.gateway}/upload", root / r["video_dir"], {"style": "concise"}, H)
        body = resp.json()
        singles.append({"truth": r["class"], "status": resp.status_code, "seconds": round(dt, 2),
                        "glosses": body.get("gloss_list"), "low_confidence": body.get("low_confidence"),
                        "notes_generator": (body.get("generation") or {}).get("generator"),
                        "has_notes": bool(body.get("notes_md"))})
    report["single_clip_uploads"] = singles
    report["single_clip_top1_in_glosses"] = sum(s["truth"] in (s["glosses"] or []) for s in singles)

    seq_rows = rows[a.n_single:]
    truth = [r["class"] for r in seq_rows]
    with tempfile.TemporaryDirectory() as td:
        vid = Path(td) / "continuous.mp4"
        fps = stitch([root / r["video_dir"] for r in seq_rows], vid)
        dur = cv2.VideoCapture(str(vid)).get(cv2.CAP_PROP_FRAME_COUNT) / fps
        report["continuous_video"] = {"clips": len(seq_rows), "seconds": round(dur, 1), "truth": truth}
        configs = {
            "old_defaults(window=64,stride=12,frame_skip=8)": {"window_frames": "64", "stride": "12", "frame_skip": "8"},
            "new_defaults(server)": {},
        }
        for name, data in configs.items():
            resp, dt = post_video(f"{a.gateway}/recognize/video", vid, data, H)
            body = resp.json()
            pred = [e["label"] for e in body.get("events") or []] or body.get("glosses") or []
            report["continuous_video"][name] = {
                "status": resp.status_code, "seconds": round(dt, 2), "predicted": pred,
                "edit_distance": edit_distance(pred, truth), "correct_in_order_hits": sum(t in pred for t in truth),
                "events": body.get("events"), "frame_skip": body.get("frame_skip"), "window_frames": body.get("window_frames"),
            }

    glosses = truth
    notes = {}
    for style in ("concise", "detailed", "academic"):
        r = requests.post(f"{a.gateway}/notes", json={"gloss_list": glosses, "style": style}, headers=H, timeout=300)
        notes[style] = {"status": r.status_code, "generation": r.json().get("generation"),
                        "headings": [l for l in r.json().get("notes_md", "").splitlines() if l.startswith("#")]}
    report["notes_by_style"] = notes
    r = requests.post(f"{a.gateway}/transcript", json={"gloss_list": glosses}, headers=H, timeout=300)
    report["transcript"] = {"status": r.status_code, "generation": r.json().get("generation")}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "health"}, indent=1)[:6000])


if __name__ == "__main__":
    main()
