# Developer Guide

How to work on, train, test and maintain Sign2Notes. Install steps are in
SETUP.md; the design is in ARCHITECTURE.md. All Python commands use the
ML venv: `ml_service\venv\Scripts\python` (written `py` below for brevity:
`Set-Alias py ml_service\venv\Scripts\python` in PowerShell, run from the
repo root).

## Repository layout

```text
ml_service/
  api.py                 FastAPI: recognition (/recognize, /recognize/video, /model/*) and
                         language generation (/notes, /transcript); /process combines them
  infer.py               model loading (per-file caches, ONNX↔checkpoint checks), windowed
                         inference, Viterbi events, LLM calls + validation + fallback
  feature_schema.py      canonical feature layouts: v2 hand+body+face (285), legacy v1 (126)
  feature_extraction.py  video → .npy features (classic Holistic for v2; original Hands for v1)
  notes_generator.py     versioned prompts (concise/detailed/academic, transcript) + deterministic fallbacks
  inference_viterbi.py   Viterbi decoding (gloss list and timestamped events)
  dataset.py, model.py, train.py, evaluate.py, eval_metrics.py, split_utils.py, checkpoint_meta.py
  build_index.py         metadata CSV + features → index CSV + vocabulary JSON
  dataset_audit.py       dataset checks → reports/dataset_audit.{md,json}
  tools/e2e_check.py     end-to-end check against running services
  tests/                 pytest (167 tests)
backend/server.js        Express gateway (auth, limits, timeouts, JSON errors); test/ = node:test
frontend/src/
  pages/Index.tsx, Webcam.tsx, LiveTranscription.tsx    the three modes
  lib/serverConfig.ts    server URL/token, apiFetch (timeouts, bounded retries)
  lib/api.ts             typed client for every endpoint
  lib/useSignRecognitionSession.ts   live pipeline (local/server recognizer selection)
  lib/webcamPipeline.ts  KeypointBuffer, SessionSmoother (pure, unit-tested)
  lib/featureSchema.ts   TS twin of feature_schema.py (golden-vector cross-check test)
  lib/notesExport.ts, lib/markdown.ts   shared Markdown parser; MD/TXT/PDF/DOCX export
  components/            NotesPanel, ResultsPanel, GlossEditor, ConnectionStatus, RecognizerStatus, ...
reports/                 dataset audit, evaluation reports
models/, data/           git-ignored (models, dataset, features, temp files)
```

`frontend/src/App.js`, `src/index.js` and `frontend/react-app/` are
leftovers of an older Create-React-App scaffold; they are not part of the
Vite build (`index.html` → `src/main.tsx`) and were left untouched.

## Training and evaluating a model

The official FDMSE-ISL splits are used as-is (train → training, val →
checkpoint selection/early stopping, test → final evaluation only).
Nothing below reads test rows before `evaluate.py`.

```powershell
# 0. Audit the dataset (read-only)
py ml_service/dataset_audit.py --feature_dir data/features_v2

# 1. Extract schema-v2 features. Resumable: existing correct files are skipped.
#    ~4 h for all 40,033 clips on the dev laptop CPU with 12 workers, ON MAINS POWER.
py ml_service/feature_extraction.py --metadata_csv data/data_meta/metadata.csv --out_dir data/features_v2 --workers 12 --splits train,val
py ml_service/feature_extraction.py --metadata_csv data/data_meta/metadata.csv --out_dir data/features_v2 --workers 12 --splits test
#    Failures (unreadable videos) are listed in data/features_v2/extraction_failures.csv.

# 2. Index + vocabulary for v2 (separate files: the legacy data/index.csv and
#    config/vocab.json belong to the legacy model and are kept unchanged)
py ml_service/build_index.py --metadata_csv data/data_meta/metadata.csv --feature_dir data/features_v2 --out_csv data/index_v2.csv --vocab_json config/vocab_v2.json --min_samples 10
#    Smaller vocabulary for a quick experiment: add --max_classes 400 (writes the
#    400 most frequent classes; use different --out_csv/--vocab_json names).

# 3. Train into a CANDIDATE directory (the API never serves it automatically)
py ml_service/train.py --index_csv data/index_v2.csv --vocab_json config/vocab_v2.json --feature_dir data/features_v2 --out_dir models/sign_recog_v2_candidate/checkpoints --epochs 40 --patience 8 --batch_size 64 --preload

# 3b. Hand-only ablation on the SAME split/protocol (for the baseline comparison;
#     produces a deliberately non-servable checkpoint)
py ml_service/train.py --index_csv data/index_v2.csv --vocab_json config/vocab_v2.json --feature_dir data/features_v2 --feature_groups left_hand,right_hand --out_dir models/ablation_hands_only/checkpoints --epochs 40 --patience 8 --batch_size 64 --preload

# 3c. Optional CNN+BiLSTM experiment (keep TemporalCNN unless this is measurably better)
#     add: --architecture cnn_bilstm --out_dir models/exp_cnn_bilstm/checkpoints

# 4. Evaluate: validation and test reported separately
py ml_service/evaluate.py --checkpoint models/sign_recog_v2_candidate/checkpoints/best.pt --index_csv data/index_v2.csv --feature_dir data/features_v2 --vocab_json config/vocab_v2.json --splits val,test --out_dir reports/eval_v2
py ml_service/evaluate.py --checkpoint models/ablation_hands_only/checkpoints/best.pt --index_csv data/index_v2.csv --feature_dir data/features_v2 --vocab_json config/vocab_v2.json --splits val,test --out_dir reports/eval_hands_only
```

`train.py` writes `best.pt` (best val accuracy), `demo.pt` (last epoch),
`train_history.json`, `../sign_recog.onnx` (from `best.pt`, with a
PyTorch↔ONNX parity check that fails the run if they diverge) and
`../sign_recog.json` (metadata). Every checkpoint records: model version,
architecture, feature schema version, input dim, class count, `label2id`,
vocabulary hash, training config (index, vocabulary, feature extraction
backend/frame_skip, whether the official split was used), best validation
accuracy. `evaluate.py` writes per split `results.json` (Top-1/3/5,
macro-F1, weighted F1, per-class precision/recall/F1, class sample
counts), `results.txt`, `confusion_matrix.npz` (`--confusion_matrix_png`
for an image).

CPU training: the TemporalCNN is small; `--preload` loads ~1.5 GB of
features into RAM and avoids per-sample file reads. `--threads N` caps
PyTorch threads.

### Replacing the served model safely

1. Train into `models/<candidate>/` and evaluate it (above).
2. Try it without moving files: start the ML service with
   `$env:MODEL_PATH="models\sign_recog_v2_candidate\checkpoints\best.pt"`
   (the ONNX next to it is used automatically) and check `/health` and
   `tools/e2e_check.py`.
3. Promote: rename `models/sign_recog_v2_candidate` → `models/sign_recog_v2`
   (rename any existing `sign_recog_v2` to `sign_recog_v2_previous` first).
   The ML service picks up the new file on the next request; browsers pick
   up the new ONNX on reload.
4. Browsers with the old model cached reload `/model/meta` on page load;
   an incompatible schema is refused, never guessed.

### Rolling back

- Model: set `MODEL_PATH` to the previous checkpoint (e.g.
  `models\sign_recog\checkpoints\best.pt` for the legacy hand-only model,
  or `models\sign_recog_v2_previous\...`) and restart the ML service, or
  rename the directories back. Nothing is deleted by training.
- Code: `git checkout baseline-before-stabilization` (the state before this
  pass) or any earlier commit; the stabilization work is on branch
  `stabilization-and-completion`.

## End-to-end check

With the three services running:

```powershell
py ml_service/tools/e2e_check.py --gateway http://127.0.0.1:3001 --signer s0004 --n_clips 6 --n_single 5 --out reports/e2e_check.json
```

Uploads real test-signer clips, recognizes a video stitched from several
clips (known order; compares old vs new windowing, reports edit distance
and latency), and generates notes in all styles plus a transcript.

## Tests

| Suite | Command | Count |
|---|---|---|
| ML (pytest) | `cd ml_service; .\venv\Scripts\python -m pytest -q` | 167 |
| Gateway (node:test) | `cd backend; npm test` | 4 |
| Frontend (vitest) | `cd frontend; npx vitest run` | 82 |
| Type check / build | `cd frontend; npx tsc --noEmit -p tsconfig.app.json; npx vite build` | — |

What the tests cover: feature schema (dims, handedness, missing landmarks,
normalization, Python↔TS golden vector), legacy-schema conversion, split
integrity and alignment, label mapping, checkpoint metadata and
self-consistency, ONNX↔PyTorch parity, per-checkpoint caching, single and
long (streamed) window inference, Viterbi, train+evaluate end to end on
synthetic data, API validation (empty/oversized/wrong-type uploads,
wrong-dimension keypoints, missing model, all styles), LLM fallback
(unreachable server, fake server, output validation), gateway auth/
timeouts/limits, frontend smoothing, buffers, server config/timeouts/
retries, DOCX content, Markdown parsing, gloss editing/undo, upload-result
regeneration, Live Transcription stop/generate/retry.

Not covered by automated tests (needs hardware or a person): real camera
and screen capture in a browser, MediaPipe WASM + onnxruntime-web together
in a browser, two physical devices over a LAN, a real local LLM.

## Common changes

- **New notes style**: add a prompt builder and a template renderer in
  `notes_generator.py`, register both, add a `PROMPT_VERSIONS` entry, add
  the option to the style selects in the frontend.
- **Changing a prompt**: bump its `PROMPT_VERSIONS` value; keep the shared
  grounding rules.
- **New feature schema**: add constants + builder in `feature_schema.py`
  and `featureSchema.ts`, register in `SCHEMA_DIMS` / `featureDimForSchema`,
  regenerate the golden fixture (`generate_crosscheck_fixture.py`), and
  give the extractor a new output directory. Never reuse a schema version.
- **New API field**: add it to the FastAPI response and the TS interface
  in `lib/api.ts`; keep existing fields (clients may depend on them).

## Conventions

- Recognition never generates text; language generation never sees video
  or keypoints.
- Never present validation numbers as test numbers; never let test rows
  influence training or model selection.
- Pure logic stays separate from browser APIs so it can be unit-tested.
