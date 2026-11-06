# Architecture

## Components

```text
 Browser (any device on the LAN)                         Main laptop
 ┌──────────────────────────────────────┐   HTTP   ┌────────────────────────┐   HTTP (127.0.0.1)  ┌──────────────────────────┐
 │ React app (frontend/)                │ ───────► │ Gateway (backend/      │ ──────────────────► │ ML service (ml_service/  │
 │  • Video Upload                      │          │  server.js, Express)   │                     │  api.py, FastAPI)        │
 │  • Live Webcam / Live Transcription  │          │  token auth, limits,   │                     │  RECOGNITION             │
 │    MediaPipe Holistic (WASM)         │          │  timeouts, CORS        │                     │   /recognize, /recognize/│
 │    + optional local ONNX recognizer  │          └────────────────────────┘                     │   video, /model/*        │
 └──────────────────────────────────────┘                                                         │  LANGUAGE GENERATION     │
                                                                                                  │   /notes, /transcript ──►│──► local LLM (optional)
                                                                                                  │  COMBINED /process       │
                                                                                                  └──────────────────────────┘
```

- **Recognition service** — input: keypoints in the served model's schema
  (`POST /recognize`) or a video (`POST /recognize/video`); output: ordered
  glosses, timestamped `events` (label, mean confidence, start/end s),
  per-window `segments` (top-3 alternatives), model version and schema.
- **Language-generation service** — input: ordered glosses, style
  (`concise|detailed|academic`, notes only), optional per-gloss
  confidences and optional user context; output: text plus `generation`
  metadata (`generator: llm|template`, `fallback_reason`,
  `prompt_version`). Never sees video or keypoints.
- `/process` = `/recognize/video` + `/notes` in one call (used by the upload page).
- Both services run in one FastAPI process on the main laptop for
  convenience; they share no state and can be split later.

## Endpoints

| Gateway route | ML route | Purpose |
|---|---|---|
| `GET /health` (public) | `/health` | gateway/ML/model/LLM status; `model_ready` means the checkpoint was loaded and is self-consistent, not merely present |
| `GET /model/meta` | same | served model: `feature_schema_version`, `input_dim`, `max_len`, labels, `vocab_hash`, `sample_hz`, `window_frames`, `ready`, `onnx_available` |
| `GET /model/onnx` | same | ONNX export of the served model |
| `POST /recognize` | same | keypoints → glosses (422 on schema mismatch with expected/got dims) |
| `POST /recognize/video` | same | video → glosses |
| `POST /notes` | same | glosses → notes |
| `POST /transcript` | same | glosses → transcript |
| `POST /upload` | `/process` | video → glosses + notes |

Errors are always JSON `{error}`. Status codes: 401 (token), 413 (too
large), 415 (not a video), 422 (invalid input), 502 (ML unreachable), 503
(no usable model), 504 (upstream timeout).

## Model registry and schema routing

`api.py` serves `MODEL_PATH`, defaulting to the newest available of
`models/sign_recog_v2/checkpoints/best.pt` (schema v2, 285-dim
hand+body+face) and `models/sign_recog/checkpoints/best.pt` (legacy schema
v1, 126-dim hand-only). The checkpoint's `feature_schema_version` decides
everything downstream:

- uploads are extracted with that schema's extractor (v2: Holistic; v1:
  the original `mp.solutions.hands` detection-order extractor);
- `/recognize` rejects features of any other dimension;
- the browser converts its Holistic vector to that schema
  (`featureSchema.toSchema`) or refuses if it can't.

A v1 model is therefore never fed v2 vectors (or vice versa). The ONNX
file is checked against the checkpoint (input dim, class count) before use.
Caches are per file (path + modification time), so replacing a model on
disk is picked up without a restart.

## Recognition pipeline

1. **Sampling.** Models are trained on 60 fps clips sampled every 8th frame
   (7.5 samples/s). Uploads use `frame_skip = round(fps / 7.5)` unless the
   caller sets one; the browser samples every `1000 / sample_hz` ms.
2. **Features.** Per frame, the canonical vector (FEATURE_SCHEMA.md);
   missing hands/pose/face → zeros, never NaN.
3. **Windows.** Training clips are ~20–35 sampled frames zero-padded to
   `max_len` = 64 and z-scored per clip over the padded array. Continuous
   video is cut into windows of `WINDOW_FRAMES` (24) real frames, stride
   `WINDOW_STRIDE` (4), each padded to 64 and normalized the same way.
   (Before this pass, windows were 64 real frames ≈ 8.5 s, i.e. several
   signs per window — unlike anything seen in training.) Windows are built
   and run in chunks of `INFER_CHUNK_SIZE`, so memory doesn't grow with
   video length.
4. **Decoding (uploads).** Viterbi over window posteriors with a sticky
   transition model; windows below the threshold are dropped; runs of the
   same label become one `event` with mean confidence and a time span. If
   no window passes the threshold, `/process` returns the majority label
   flagged `low_confidence` (upload UX); live modes never do this.
5. **Live stability (webcam/transcription).** No future context exists, so
   `SessionSmoother` commits a label after `stableCount` (3) consecutive
   confident (≥ 0.75) predictions; 0.50–0.75 is "uncertain" (does not reset
   a streak); < 0.50 is "no sign". The same label is not committed twice
   in a row unless there were ≥ `repeatGapTicks` (4) no-sign samples in
   between — a held sign is one event, a repeated sign after a pause is
   two. No visible hands for 3 samples → "no sign" without running the
   classifier. One failed sample never stops a session; repeated failures
   are shown.

### Local vs server recognition (live modes)

```text
GET /model/meta → schema supported by this browser and dims match?
  ├─ no  → error: incompatible model (nothing is guessed)
  └─ yes → GET /model/onnx → onnxruntime-web session → test inference → classes == num_classes?
        ├─ yes → LOCAL: landmarks + recognition in the browser; only gloss words leave the device (on Generate)
        └─ no  → SERVER: landmark coordinates of the current window → POST /recognize (≤1 request in flight, every 2nd sample)
Glosses (either way) → POST /notes, /transcript on the main laptop, only when the user asks.
```

A client without landmark extraction at all (not a browser) must send
video to `/recognize/video`; it cannot send glosses it does not have.

## Language generation

`notes_generator.py` holds three separate prompt builders and three
deterministic renderers (concise: title + flat bullets + important terms;
detailed: topic/subtopic sections, explanations limited to what glosses
support; academic: formal section hierarchy, definitions only for
term-like glosses, explicit ban on filler). All prompts share the
grounding rules (no invented facts/examples/dates; hedge or omit glosses
marked `(?)`; treat consecutive repeats as one; user context is not a fact
source). Versions are in `PROMPT_VERSIONS`. The LLM call is bounded
(`LLM_TIMEOUT`, `LLM_MAX_RETRIES`); output is validated (non-empty, has
Markdown structure, `<think>` blocks and code fences stripped); any failure
→ deterministic, style-aware fallback with `fallback_reason`.

Live Transcription calls `/transcript` and `/notes` **once**, after Stop.
Neither live mode calls the language service per sign.

## Exports

Client-side only (`frontend/src/lib/notesExport.ts`). Markdown, Text and
PDF keep their original layout (Text/PDF add the Sign2Notes header/footer
they always had); DOCX contains exactly the displayed Markdown — headings
(H1–H6), paragraphs, bullet and numbered lists, bold/italic — and nothing
else. The on-screen renderer and all exporters share one Markdown line
parser, so the display and the files can't diverge. What is exported is
the text currently shown, including the user's edits.

## Data and models on disk

```text
data/FDMSE-ISL/...            raw videos (read-only)
data/data_meta/*.csv          official metadata + splits
data/features_v2/             schema-v2 .npy per clip + feature_meta.json + extraction_failures.csv
data/index.csv, config/vocab.json   legacy v1 index/vocabulary (kept for the legacy model)
models/sign_recog/            legacy v1 model (kept for rollback/comparison)
models/sign_recog_v2/         schema-v2 model (when trained)
data/tmp/                     temporary uploads/features, deleted after each request
```
