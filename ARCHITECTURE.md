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
| React frontend | `frontend/src/pages/Index.tsx`, `Webcam.tsx`, `LiveTranscription.tsx` | Upload UI, live webcam UI, live transcription UI, results/notes display |
| Node gateway | `backend/server.js` | Holds uploads in memory, proxies to FastAPI, never touches raw video on disk |
| FastAPI ML service | `ml_service/api.py` | HTTP API: `/process` (video), `/notes` (gloss→notes), `/transcript` (gloss→natural-language transcript), `/recognize` (keypoints→glosses, no notes), `/model/meta`, `/model/onnx`, `/health` |
| Feature extraction | `ml_service/feature_extraction.py`, `feature_schema.py` | Video → per-frame MediaPipe **Holistic** (hand+body+face) keypoints, canonical 285-dim vectors — see `FEATURE_SCHEMA.md`. Batch (training) and single-video (inference) paths. |
| Dataset | `ml_service/dataset.py` | Loads `data/index.csv` + `.npy` (schema v2) features into training samples, with augmentation |
| Model | `ml_service/model.py` | `TemporalCNN` — the lightweight recognition model (default architecture; see `--architecture` in `train.py` for the optional CNN+BiLSTM experiment) |
| Training | `ml_service/train.py` | Trains the model, exports `.pt` checkpoint + ONNX |
| Inference | `ml_service/infer.py` | Sliding-window inference, LLM-backed and template note generation |
| Smoothing | `ml_service/inference_viterbi.py` | Viterbi decoding: `viterbi_decode()` (flat label list) and `viterbi_events()` (timestamped events) |
| Notes | `ml_service/notes_generator.py` | Deterministic template engine (style-aware) + LLM prompt construction, in three genuinely different styles (concise/detailed/academic) |
| Client-side pipeline | `frontend/src/lib/webcamPipeline.ts`, `featureSchema.ts` | Pure logic: keypoint construction (mirrors `feature_schema.py` exactly, cross-checked by a golden-vector test), normalization, buffering, real-time smoothing (unit-tested) |
| Client-side ML | `frontend/src/lib/onnxSession.ts`, `holisticLandmarker.ts` | Browser wrappers around onnxruntime-web and MediaPipe Tasks Vision's HolisticLandmarker |

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
   285 floats per kept frame — hand+body+face, schema v2, see
   FEATURE_SCHEMA.md — not raw pixels)
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
MediaPipe HolisticLandmarker (WASM, in-browser) -- one model, one pass,
   gives hand+pose+face landmarks together (see FEATURE_SCHEMA.md
   "Detector") — sampled roughly every 280ms, approximating the
   training-time frame_skip=8 @ ~25-30fps cadence (browsers don't give the
   same exact frame-count control that offline video decoding does)
 ↓
buildFeatureVector() (featureSchema.ts) — mirrors feature_schema.py's
   build_feature_vector() exactly: canonical LEFT_HAND → RIGHT_HAND →
   POSE → FACE layout, 285 dims. Left/right hand identity comes directly
   from HolisticLandmarker (leftHandLandmarks/rightHandLandmarks), not
   detection order. Cross-validated numerically against the Python
   implementation via a golden-vector fixture (see
   featureSchema.crosscheck.test.ts).
 ↓
KeypointBuffer (sliding window, ring buffer, zero-pads at the end when
   not yet full — mirrors feature_schema.py's pad_or_trim())
 ↓
normalizeSequence() (featureSchema.ts) — mirrors feature_schema.py's
   normalize_sequence() exactly, including excluding pose-visibility dims
   from z-scoring (verified to match Python's output to 3-4 decimal
   places on identical input)
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

- Confidence is read into three zones (per spec): below `ignoreThreshold`
  (default **0.50**) is **NO_SIGN** — surfaced as "No sign detected",
  never touches session history, and resets the stability streak (this
  handles idle hands, transitions between signs, and general noise
  without needing a trained NO_SIGN class yet). Between `ignoreThreshold`
  and `acceptThreshold` (default **0.75**) is **uncertain** — surfaced as
  "Uncertain — hold steady"; it's a soft tick that neither advances nor
  resets an in-progress streak, so one shaky frame in the middle of a held
  sign doesn't force the user to start over. At or above `acceptThreshold`
  is **confident**, and can accumulate stability toward a commit.
- A label must repeat for `stableCount` (default 3) consecutive confident
  predictions before it's committed as one event — this is what collapses
  "QUESTION QUESTION QUESTION QUESTION" (one held sign, sampled
  repeatedly) into a single `QUESTION` entry, and rejects one-frame noise.
- Once a label is committed, repeating it again doesn't re-commit — only
  a **different** label (once it's also stable) starts a new event. If the
  user undoes or deletes that event from the UI, `forgetLastCommitted()`
  lets the same label commit again later.

This is a deliberate simplification, not a claim of true HMM decoding
in-browser (see the development principle "do not over-engineer" in the
project brief). It's covered by unit tests in
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
- **Live webcam session**: the opposite. A NO_SIGN or uncertain prediction
  is surfaced inline ("No sign detected" / "Uncertain — hold steady") and
  is deliberately **not** added to the session history, because the user
  is present and can just re-sign it — there's no reason to guess when the
  real thing is one gesture away. This matches the spec's confidence-
  handling requirement and Acceptance Test 4. The user can also edit,
  delete, or undo any committed event afterward, so a misread sign isn't
  permanent even after it's been added.

## Live Transcription (Mode 3)

`frontend/src/pages/Webcam.tsx` (Mode 2) and
`frontend/src/pages/LiveTranscription.tsx` (Mode 3) share the exact same
recognition pipeline -- model/camera loading, the Holistic detection
loop, `SessionSmoother`, session history editing -- via one hook,
`frontend/src/lib/useSignRecognitionSession.ts`. Live Transcription adds:

- **A screen/tab capture source**, alongside the webcam, via the
  browser's native `getDisplayMedia()` -- for transcribing sign-language
  content already playing on the device rather than signing directly at
  the camera (project brief section 22, "watching sign-language content
  on a laptop/mobile/system screen"). This is genuinely what the browser
  supports (the user picks a tab, a window, or their screen in the
  browser's own dialog) -- there's no way for a web page to force true
  OS-wide capture, and this doesn't pretend to (RULE 19).
- **A Stop & Generate flow that calls the LLM exactly once**, not per
  sign and not continuously (project brief section 21 / RULE 9):

  ```text
  ACTIVE: recognize -> stability filter -> append gloss -> update live transcript
      (SessionSmoother; the LLM is never touched here)
   ↓ user clicks "Stop & Generate"
  STOPPING: stop the camera/detection loop, freeze the gloss sequence
   ↓ (if the frozen sequence is empty, stop here with an error -- nothing to generate)
  GENERATING: POST /transcript AND POST /notes in parallel, with the SAME
      frozen gloss list
   ↓
  COMPLETED: show the natural-language transcript (Layer 2) and the
      structured notes (Layer 3) together
  ```

  A `genPhase` state (`idle` / `stopping` / `generating` / `completed` /
  `error`) guards this specifically against a rapid double-click firing
  two generation requests -- the one place in the whole UI where a
  duplicate request would actually cost something (two LLM calls
  instead of one). Covered by a real rendered-component test,
  `frontend/src/pages/__tests__/LiveTranscription.test.tsx`, including a
  simulated double-click and an LLM-failure + retry-with-preserved-
  glosses scenario (project brief section 31: don't lose the session on
  a network failure).
- **A separate `/transcript` endpoint**, not a `style` value on `/notes`:
  a transcript (flowing prose, in gloss order) and structured notes
  (headed/bulleted, reorganized by topic) are genuinely different outputs
  built from different prompts -- see `notes_generator.py`'s module
  docstring on the "three output layers" (recognition -> transcript ->
  notes). `/transcript` has no `style` field; a transcript's job doesn't
  change with register the way notes do.

## Model requirements

The recognition model (`ml_service/model.py: TemporalCNN`) is intentionally
a lightweight 1D-convolutional temporal classifier, not a Transformer --
sized to run comfortably on a 6GB laptop GPU (RTX 4050) and, for the
webcam demo, to run in a browser via WASM without a GPU at all. See
`SETUP.md` for training/hardware recommendations.

## Confidence handling reference

| Setting | Where | Default | Effect |
|---|---|---|---|
| `CONFIDENCE_THRESHOLD` (env) | `ml_service/.env` | 0.55 | Default `/process` threshold (per-request overridable) |
| `threshold` (form field) | Upload request | 0.55 | Per-upload override |
| `SessionSmoother.ignoreThreshold` | `webcamPipeline.ts` | 0.50 | Webcam: below this → NO_SIGN ("no sign detected"), resets stability, never appended |
| `SessionSmoother.acceptThreshold` | `webcamPipeline.ts` | 0.75 | Webcam: 0.50-0.74 is "uncertain" (soft tick, doesn't reset streak); at/above this a prediction can accumulate toward a commit |
| `SessionSmoother.stableCount` | `webcamPipeline.ts` | 3 | Webcam: consecutive agreeing confident predictions needed to commit one event |
| `SessionSmoother.forgetLastCommitted()` | `webcamPipeline.ts` | — | Called by the UI on undo/delete so an un-done sign can be re-committed later |
