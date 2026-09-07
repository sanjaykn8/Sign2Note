# Architecture

## Overview

```text
                    ┌──────────────────┐
                    │   React Frontend │
                    │  (Upload / Webcam)│
                    └────────┬─────────┘
                             │
                 ┌───────────┴────────────┐
                 │                        │
          Video upload               Live webcam
                 │                        │
                 ▼                        ▼
        ┌────────────────┐      ┌──────────────────────┐
        │  Node gateway   │      │  In-browser pipeline  │
        │ (backend/       │      │  MediaPipe (WASM)      │
        │  server.js)     │      │  + onnxruntime-web      │
        └────────┬────────┘      └──────────┬────────────┘
                 │                           │
                 ▼                           │ (gloss words only,
        ┌────────────────┐                  │  on "Generate Notes")
        │ FastAPI ML      │◄─────────────────┘
        │ service         │
        │ (ml_service/    │
        │  api.py)        │
        └────────┬────────┘
                 │
     ┌───────────┼────────────────┐
     ▼           ▼                ▼
MediaPipe   Temporal CNN     Viterbi/HMM
(server-    (ONNX Runtime    smoothing +
 side, for   or PyTorch)     duplicate
 uploads)                    collapse
     │           │                │
     └───────────┴────────────────┘
                 │
                 ▼
        Ordered, timestamped
          gloss sequence
                 │
        ┌────────┴─────────┐
        ▼                  ▼
  Template Engine     Local LLM
  (always available)  (Ollama / llama.cpp,
                        env-configured)
        └────────┬─────────┘
                 ▼
        Structured Markdown notes
```

There are two independent recognition paths that both end up at the same
note-generation step:

1. **Video upload** → Node gateway → FastAPI → server-side MediaPipe +
   ONNX/PyTorch inference → notes.
2. **Live webcam** → entirely in-browser MediaPipe + ONNX inference →
   (only on "Generate Notes") gloss words → FastAPI `/notes` → notes.

This split exists because of the privacy requirement that webcam video
never be uploaded (see `PRIVACY.md`) — it's not an accidental duplication,
it's two different trust boundaries for two different input sources.

## Component responsibilities

| Component | File(s) | Responsibility |
|---|---|---|
| React frontend | `frontend/src/pages/Index.tsx`, `Webcam.tsx` | Upload UI, live webcam UI, results/notes display |
| Node gateway | `backend/server.js` | Holds uploads in memory, proxies to FastAPI, never touches raw video on disk |
| FastAPI ML service | `ml_service/api.py` | HTTP API: `/process` (video), `/notes` (gloss→notes), `/model/meta`, `/model/onnx`, `/health` |
| Feature extraction | `ml_service/feature_extraction.py` | Video → per-frame MediaPipe hand keypoints (126-dim vectors), batch (training) and single-video (inference) paths |
| Dataset | `ml_service/dataset.py` | Loads `data/index.csv` + `.npy` features into training samples, with augmentation |
| Model | `ml_service/model.py` | `TemporalCNN` — the lightweight recognition model |
| Training | `ml_service/train.py` | Trains the model, exports `.pt` checkpoint + ONNX |
| Inference | `ml_service/infer.py` | Sliding-window inference, LLM-backed and template note generation |
| Smoothing | `ml_service/inference_viterbi.py` | Viterbi decoding: `viterbi_decode()` (flat label list) and `viterbi_events()` (timestamped events) |
| Notes | `ml_service/notes_generator.py` | Deterministic template engine + LLM prompt construction |
| Client-side pipeline | `frontend/src/lib/webcamPipeline.ts` | Pure logic: keypoint construction, normalization, buffering, real-time smoothing (unit-tested) |
| Client-side ML | `frontend/src/lib/onnxSession.ts`, `handLandmarker.ts` | Browser wrappers around onnxruntime-web and MediaPipe Tasks Vision |

## Long-video support

**The design goal:** a video may contain several distinct signs in
sequence (`DEFINITION → EXAMPLE → QUESTION → ...`), and the system must
recognize all of them with timestamps, not just make one prediction for
the whole clip.

```text
Video
 ↓
Keypoint extraction (frame-by-frame, streamed via cv2.VideoCapture —
   never loads the whole video into memory; a several-minute clip's
   extracted keypoint sequence is a few hundred KB at most, since it's
   126 floats per kept frame, not raw pixels)
 ↓
Sliding temporal windows (ml_service/infer.py: _window_batch)
   — each window is independently normalized (same as a training sample),
     NOT the whole clip normalized once — see "A bug we found" below
 ↓
Model inference, run in bounded-size chunks (INFER_CHUNK_SIZE, default 64)
   — this is what actually bounds memory/compute for very long videos:
     however many windows a long clip produces, they're processed a fixed
     number at a time, not all at once
 ↓
Per-window confidence + label (ml_service/api.py: `segments`)
 ↓
Viterbi/HMM smoothing (ml_service/inference_viterbi.py: viterbi_events())
   — favors staying in the same state (stay_prob=0.92) to resist window-
     to-window flicker, then collapses consecutive same-label windows into
     one event spanning (start_time, end_time)
 ↓
Ordered, timestamped gloss sequence (`events` in the /process response)
```

### Timestamps

Each window's frame span (in *extracted-frame* units, i.e. post-
frame_skip) is converted to real seconds using the **original** video's
fps (probed via `cv2.VideoCapture(...).get(cv2.CAP_PROP_FPS)` in
`api.py`) and the `frame_skip` used at extraction time:

```text
seconds = (extracted_frame_index * frame_skip) / original_fps
```

### A bug we found and fixed while building this

The sliding-window function originally normalized the **entire** long
keypoint sequence once (global mean/std across the whole clip) before
slicing it into windows. `dataset.py` normalizes each **training sample**
independently (per-clip mean/std, since a training sample IS one
short clip). For a short clip these are the same thing; for a long video
where the signing scale/position drifts over time, they diverge —
normalizing globally smears each window's statistics away from what the
model was actually trained to expect, and it gets worse the longer the
video is. We caught this by building a synthetic 5-phase, 40-second test
video and finding it collapsed into one wrong label; the fix (normalize
each window independently, matching `dataset.py`'s pad-then-normalize
order exactly) restored correct multi-sign detection with timestamps. See
the code comment on `_window_batch` in `infer.py`.

## Live webcam

```text
Browser webcam (getUserMedia)
 ↓
MediaPipe HandLandmarker (WASM, in-browser)
   — sampled roughly every 280ms, approximating the training-time
     frame_skip=8 @ ~25-30fps cadence (browsers don't give the same exact
     frame-count control that offline video decoding does)
 ↓
keypointsFromLandmarks() — mirrors feature_extraction.py's
   _extract_keypoints() exactly: first detected hand → first 63 values,
   second detected hand → next 63, zero-padded if fewer than 2 hands.
   Cross-validated numerically against the Python implementation.
 ↓
KeypointBuffer (sliding window, ring buffer, zero-pads at the end when
   not yet full — mirrors infer.py's _pad_window())
 ↓
normalizeWindow() — mirrors infer.py's _normalize() exactly (verified to
   match Python's output to 5-6 decimal places on identical input)
 ↓
onnxruntime-web inference (WASM, in-browser) — same ONNX model file the
   backend serves, fetched once via GET /model/onnx
 ↓
SessionSmoother — real-time confidence gating + stability smoothing
 ↓
Committed session events (label, confidence, timestamp)
 ↓
"Generate Notes" → POST /notes with the ordered gloss list only
```

### Why the live webcam smoothing is NOT the same algorithm as long-video Viterbi

`viterbi_events()` is a proper dynamic-programming decoder, but it needs
the **entire clip's** window probabilities up front to run its
forward/backward pass — that doesn't exist yet in a live stream (you'd
have to wait until the session ends to see any predictions, defeating the
point of a live display). Instead, `SessionSmoother`
(`frontend/src/lib/webcamPipeline.ts`) uses a simpler, real-time-friendly
rule:

- A prediction below `acceptThreshold` (default **0.70**, per spec) is
  surfaced as "Low confidence — please repeat" and never touches the
  session history.
- A label must repeat for `stableCount` (default 3) consecutive accepted
  predictions before it's committed as one event — this is what collapses
  "QUESTION QUESTION QUESTION QUESTION" (one held sign, sampled
  repeatedly) into a single `QUESTION` entry, and rejects one-frame noise.
- Once a label is committed, repeating it again doesn't re-commit — only
  a **different** label (once it's also stable) starts a new event.

This is a deliberate simplification, not a claim of true HMM decoding
in-browser (see the development principle "do not over-engineer" in the
project brief). It's covered by 19 unit tests in
`frontend/src/lib/__tests__/webcamPipeline.test.ts`.

### Why /process (video upload) and the webcam session have different confidence policies

This looks like an inconsistency at first glance, so it's worth stating
explicitly: they're deliberately different, for different UX goals.

- **`/process` (upload a video, get notes)**: always returns usable notes.
  If no window crosses the confidence threshold, it falls back to a
  majority-vote best guess across all windows (`_best_guess_gloss()` in
  `api.py`) rather than showing a dead-end "couldn't process" screen. This
  was an explicit product decision for the upload flow — see the git
  history / prior session notes.
- **Live webcam session**: the opposite. A low-confidence prediction is
  surfaced as "please repeat" and is deliberately **not** added to the
  session history, because the user is present and can just re-sign it —
  there's no reason to guess when the real thing is one gesture away.
  This matches the spec's confidence-handling requirement and Acceptance
  Test 4.

## Model requirements

The recognition model (`ml_service/model.py: TemporalCNN`) is intentionally
a lightweight 1D-convolutional temporal classifier, not a Transformer —
sized to run comfortably on a 6GB laptop GPU (RTX 4050) and, for the
webcam demo, to run in a browser via WASM without a GPU at all. See
`SETUP.md` for training/hardware recommendations.

## Confidence handling reference

| Setting | Where | Default | Effect |
|---|---|---|---|
| `CONFIDENCE_THRESHOLD` (env) | `ml_service/.env` | 0.55 | Default `/process` threshold (per-request overridable) |
| `threshold` (form field) | Upload request | 0.55 | Per-upload override |
| `SessionSmoother.acceptThreshold` | `webcamPipeline.ts` | 0.70 | Webcam: below this → "please repeat", never appended |
| `SessionSmoother.stableCount` | `webcamPipeline.ts` | 3 | Webcam: consecutive agreeing predictions needed to commit one event |
