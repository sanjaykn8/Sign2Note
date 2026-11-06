# Changelog

All meaningful changes, newest first. Every entry says what changed, why,
and how it was verified. Git tag `baseline-before-stabilization` marks the
commit (`7a77030`, "live base") this pass started from.

## Stabilization and completion pass — branch `stabilization-and-completion`

### Python environment: global Python 3.11 instead of a venv

The isolated `ml_service/venv` was removed at the owner's request; the ML
service, extraction and training run on the global Python 3.11 (which has
PyTorch with CUDA). `requirements.txt` now pins `numpy>=1.24,<2` and
`protobuf>=4.25.3,<5`, the versions mediapipe 0.10.21 requires; with them
the global interpreter imports mediapipe, TensorFlow 2.17 and torch
2.14+cu130 together. Verified: all 167 ML tests pass on the global
interpreter with these pins, and one FDMSE-ISL clip extracted there is
identical to the feature file produced earlier in the venv.

### Baseline found (before any change)

| Area | Baseline state | How verified |
|---|---|---|
| Frontend | Builds; `tsc` clean; 60/60 vitest tests pass; ESLint: 10 errors / 7 warnings (pre-existing, mostly `any`, plus stale CRA files `src/App.js`, `src/index.js`) | `npx vite build`, `npx tsc --noEmit -p tsconfig.app.json`, `npx vitest run`, `npx eslint src` |
| Backend gateway | Starts; proxies to ML service; HTML error pages on multer/body errors | `node server.js` + `curl /health`, `/notes` |
| ML service | 141/142 pytest pass; 1 failure = test isolation bug (`/model/meta` test read the real `config/vocab.json`) | `pytest` on the baseline commit in a separate worktree |
| Recognition | **Broken**: `api.py` defaulted to `models/sign_recog_v2/...` which does not exist. The only trained model (`models/sign_recog`, 126-dim hand-only, 2,001 classes) could not be served by the schema-v2 code | `/health` → `model_ready: false` path; reading `api.py` |
| v2 feature extraction | **Broken on this machine**: MediaPipe Tasks `HolisticLandmarker` (Python, mediapipe 0.10.21) hard-crashes (`packet.cc: Check failed ... The packet is empty`) after a few frames of real FDMSE-ISL clips, in VIDEO and IMAGE mode | Reproduced on `s0001` clips |
| Python environment | Global Python 3.11 cannot import MediaPipe (TensorFlow compiled for NumPy 1.x + NumPy 2.4 in site-packages) | `python -c "import mediapipe"` |
| Training | `train.py` ONNX export crashes on torch ≥ 2.9 (needs `onnxscript`); split indices misaligned whenever an index row has no feature file | New end-to-end training test |
| Dataset | 40,033 clips, 2,002 classes, 20 signers, front view; official splits train 20,016 / val 4,003 / test 16,014; **signer-independent** (10 / 2 / 8 signers); no missing files or duplicates; one class (`0/Zero`) has 13 clips | `ml_service/dataset_audit.py` → `reports/dataset_audit.md` |
| Historical 83.5 % | Validation accuracy of the hand-only model on a **random 85/15 split of all 40k clips** (git `a858f45`), i.e. test-split signers were in its training data — not comparable to an official-split result | Git history of `train.py` |

### Bugs fixed (each with a regression test)

| # | Component | Bug (reproduced) | Fix | Test |
|---|---|---|---|---|
| 1 | ML API | Default model path pointed at a non-existent v2 model; the trained model was unusable | Model resolution: v2 if present, else legacy v1; each model is served with **its own** feature schema (v1 uploads are extracted with the original v1 hand extractor) | `test_legacy_model_is_served_with_its_own_schema` |
| 2 | Extraction | Tasks HolisticLandmarker crash | Python extraction uses `mp.solutions.holistic` (same landmark topology); backend recorded in `feature_meta.json` | manual reproduction + 60-clip extraction run |
| 3 | infer.py | Model caches were one global slot — loading a 2nd checkpoint returned the 1st | Cache per file (path + mtime) | `test_torch_cache_is_per_checkpoint` |
| 4 | infer.py | ONNX file could be paired with a different checkpoint's vocabulary | Input/output dims of ONNX checked against the checkpoint | `test_onnx_checkpoint_mismatch_is_detected` |
| 5 | train/evaluate | Split taken from raw CSV row numbers while the dataset had dropped rows → crash or misalignment | Split resolved on the dataset's own rows | `test_split_indices_follow_dataset_rows_when_features_are_missing` |
| 6 | evaluate.py | Used `config/vocab.json` instead of the checkpoint's label mapping; test split only | Uses checkpoint `label2id`; `--splits val,test` written separately with Top-1/3/5, macro-F1, per-class, counts, confusion matrix | `test_train_and_evaluate_end_to_end_on_synthetic_official_split` |
| 7 | train.py | ONNX export crash on torch ≥ 2.9 | TorchScript exporter (`dynamo=False`) + ONNX/PyTorch parity check | same |
| 8 | API | Upload read fully into memory before size check; heavy work inside an `async` endpoint blocked the whole server (even `/health`) | Streamed upload with size/type limits; recognition in a worker thread | `test_upload_rejects_unsupported_type_and_oversize` |
| 9 | API | Fixed `frame_skip=8` → a 30 fps upload was sampled at half the training rate | `frame_skip` derived from fps to match 7.5 samples/s unless set explicitly; gateway no longer forces `frame_skip=8` | `test_legacy_model_is_served...` (asserts frame_skip 4 at 30 fps), gateway test |
| 10 | LLM | OpenAI client defaults (600 s timeout, 2 retries) could hang a request ~30 min; empty/chatty output accepted | `LLM_TIMEOUT`, `LLM_MAX_RETRIES`, output validation (`<think>`/fence stripping, must contain Markdown structure) | `test_clean_llm_output_*`, `test_generation_meta_*` |
| 11 | Gateway | HTML error pages for oversized uploads / bad JSON; no upstream timeouts; listened on all interfaces with no auth | JSON 413/415/400, 502/504 with timeouts, optional token auth, binds 127.0.0.1 unless a token or HOST is set | `backend/test/gateway.test.js` |
| 12 | Frontend | API base hard-coded to `http://localhost:3001` → other devices called themselves | Default = page host:3001; Connection settings (URL + token) | `stabilization.test.ts` |
| 13 | Frontend | Live mode silently failed forever when the model's input dim ≠ 285 (e.g. legacy model); no server fallback | Verified local recognizer (schema/dim check + test inference) else server recognition from keypoints | manual + type checks (see "Not verified") |
| 14 | Frontend | Live sampling every 280 ms vs 133 ms in training (60 fps / 8) | Sampling interval from `ModelMeta.sample_hz` | — |
| 15 | Frontend | A legitimately repeated sign ("HELLO … HELLO") was dropped | `repeatGapTicks` in `SessionSmoother` | `a repeated sign after a pause is committed again` |
| 16 | Frontend | DOCX prepended a second title + a date line not shown in the app; all heading levels became H2; numbered lists lost | Exact notes only, H1–H6, numbered lists | `DOCX export contains exactly the displayed notes` |
| 17 | Frontend | Blob URL revoked synchronously (can cancel downloads in Firefox/Safari) | Deferred revoke | — |
| 18 | Frontend | Edit/delete buttons only visible on mouse hover (unusable on touch/keyboard) | Visible on focus / small screens, labelled | `GlossEditor.test.tsx` |
| 19 | Frontend | A failed ONNX load was cached forever | Rejected promise cleared | — |
| 20 | Tests | `/model/meta` test depended on the real `config/vocab.json` | Fixture isolates `VOCAB_PATH` | existing test now passes |

### Features completed

- **Recognition / generation separation at the API level**: `/recognize`
  (keypoints) and new `/recognize/video` (video) return glosses only;
  `/notes` and `/transcript` take glosses only and return
  `generation` metadata (generator, fallback reason, prompt version).
  `/process` kept for the upload page.
- **Notes**: versioned, structurally different prompts for concise /
  detailed / academic; uncertain glosses (confidence < 0.6) marked `(?)`;
  repeats collapsed in the fallback; optional user context that the prompt
  forbids using as a fact source.
- **Upload mode**: validation, upload progress, processing status, cancel,
  editable timestamped glosses with undo, regenerate notes in any style.
- **Live modes**: style selection, confidences passed to generation,
  recognizer status, Live Transcription transcript exportable (MD/TXT/PDF/DOCX).
- **Network**: token auth, safe binding, CORS config, timeouts, bounded
  retries, connection status; optional HTTPS for the dev server.
- **ML tooling**: `dataset_audit.py`, `--feature_groups` ablation,
  `--preload`, `--vocab_json`, `--feature_schema_version`, training
  history JSON, ONNX parity, `tools/e2e_check.py`.

### Dependencies

No new runtime dependencies. The ML service now runs from an isolated
virtualenv (`ml_service/venv`, git-ignored) because the global Python's
TensorFlow/NumPy combination breaks MediaPipe; `requirements.txt` unchanged
in substance (`pytest`, `httpx`, `python-docx` installed in the venv for
tests/tools).

---

## Earlier history (moved from README "What Was Changed")

This MVP upgrade modified the following, building on the existing
architecture rather than rewriting it (`dataset.py`'s augmentation,
`train.py`'s training loop/ONNX export, and `model.py`'s TemporalCNN were
already solid and were left alone):

**Long-video support**
- `ml_service/inference_viterbi.py` -- added `viterbi_events()` for
  timestamped, duplicate-collapsed gloss sequences, built on a refactored
  (but behaviorally-verified-identical) `_viterbi_path()` shared with the
  original `viterbi_decode()`.
- `ml_service/infer.py` -- added chunked windowed inference
  (`INFER_CHUNK_SIZE`) so memory/compute stays bounded regardless of video
  length; added timestamp computation from video fps + frame_skip +
  stride. **Found and fixed a real bug**: `_window_batch` was normalizing
  the entire long video once globally instead of per-window like training
  does -- this actively broke multi-sign detection in longer videos (see
  `ARCHITECTURE.md` for the full story).
- `ml_service/api.py` -- `/process` now returns an `events` field (ordered,
  timestamped, collapsed) alongside the existing `gloss_list`.

**Live webcam**
- New `frontend/src/lib/webcamPipeline.ts` -- pure logic (keypoint
  construction, normalization, sliding buffer, real-time smoothing),
  numerically cross-validated against the Python training pipeline and
  covered by 19 unit tests.
- New `frontend/src/lib/onnxSession.ts`, `handLandmarker.ts` -- browser
  wrappers around `onnxruntime-web` and `@mediapipe/tasks-vision`.
- New `frontend/src/pages/Webcam.tsx` -- the live session UI.
- New backend endpoints: `GET /model/meta`, `GET /model/onnx` (so the
  browser can run inference itself), `POST /notes` (generate notes from
  an already-recognized gloss list, no video/keypoints involved).
- New `backend/server.js` proxy routes for the above.

**Note generation**
- `ml_service/infer.py` -- replaced the hard-coded llama.cpp-only client
  with env-var-driven `LLM_PROVIDER`/`LLM_MODEL`/`LLM_BASE_URL`
  configuration, unifying Ollama and llama.cpp through one
  OpenAI-compatible client (both speak `/v1/chat/completions`).
- `ml_service/notes_generator.py` -- rewrote the template engine to group
  glosses into Key Concepts/Questions/Tasks sections instead of a flat
  list.
- Fixed a real bug in `backend/server.js` and `frontend/src/lib/api.ts`:
  both were force-sending a hard-coded `llm_model` default, silently
  overriding the new env-configured default whenever the caller didn't
  explicitly choose a model.

**Other fixes**
- Fixed a pre-existing dependency conflict in `frontend/package.json`
  (`vite@8` pinned against an incompatible `@vitejs/plugin-react-swc@3`)
  that would have blocked `npm install` for any new developer.
- Added `.env.example` for the ML service.
- Added the Python (`ml_service/tests/`, pytest, 37 tests) and frontend
  (`frontend/src/lib/__tests__/`, vitest, 19 new tests) suites described
  in `DEVELOPER_GUIDE.md`.
- Wrote this documentation suite (previously only a single README existed).

**Post-MVP: live transcript editing + real export** (this session, building
on the above -- see the post-MVP workplan's P0 items 1-8)
- `frontend/src/lib/webcamPipeline.ts` -- `SessionSmoother` now reads
  confidence into three zones instead of one accept/reject threshold:
  `ignoreThreshold` (0.50, "no sign detected") / `acceptThreshold` (0.75,
  confident) / the uncertain band between them, which is a soft tick that
  doesn't reset an in-progress streak. Added `forgetLastCommitted()` so
  undoing/deleting a history event lets that label commit again later.
  Test suite grew to 23 tests covering the new zones and undo behavior.
- `frontend/src/pages/Webcam.tsx` -- Pause/Resume (freezes recognition,
  keeps elapsed-time accounting correct via a paused-duration offset),
  per-event inline edit and delete on Sign History rows, and an Undo
  button for the most recent event. "No sign detected" is now shown
  distinctly from "Uncertain -- hold steady".
- New `frontend/src/components/NotesPanel.tsx` -- the **Preview -> Edit ->
  Export** flow (workplan section 7/8): edit generated notes inline before
  committing to a download, plus Regenerate. Shared by both the upload
  flow (`ResultsPanel.tsx`) and the live webcam page.
- New `frontend/src/lib/notesExport.ts` -- client-side Markdown/Text/PDF
  export (PDF via a dynamically-imported `jspdf`, so it doesn't add to the
  initial bundle for users who never export to PDF). The plain-text/PDF
  layout matches the workplan's suggested document format (SIGN2NOTES
  header, date/duration, uppercase section headers, footer disclaimer).
- New `frontend/src/lib/markdown.ts` -- the Markdown-subset-to-HTML
  renderer, extracted out of `ResultsPanel.tsx` so `NotesPanel` can share
  it instead of duplicating it.
- Added `jspdf` to `frontend/package.json`.

**Hand+body+face schema v2 migration (this session)** -- the biggest
architecture change since the original MVP; see `FEATURE_SCHEMA.md` for
the full spec.
- New `ml_service/feature_schema.py` and `frontend/src/lib/featureSchema.ts`
  -- the canonical 285-dim hand+body+face feature layout, proven
  numerically identical across languages by a golden-vector test
  (`generate_crosscheck_fixture.py` / `featureSchema.crosscheck.test.ts`),
  not just code review.
- Migrated both Python and browser detection from separate/legacy APIs
  to MediaPipe Tasks' `HolisticLandmarker` -- one model, one pass, gives
  hand+pose+face together and resolves left/right hand identity itself
  (no more "detection order isn't handedness" caveat). This also fixed a
  latent break: `mp.solutions.hands` (schema v1's detector) doesn't exist
  in current `mediapipe` PyPI releases at all.
- `ml_service/feature_extraction.py` rewritten around Holistic, with
  per-directory `feature_meta.json` versioning and a dimension-mismatch
  guard so v1 (126-dim) and v2 (285-dim) features can never be silently
  mixed (RULE 12).
- New `frontend/src/lib/holisticLandmarker.ts` replaces
  `handLandmarker.ts`; `webcamPipeline.ts` slimmed down to session/buffer
  logic only, with feature-vector construction and normalization now
  living in `featureSchema.ts`.
- `model.py` gained an optional `CNNBiLSTM` architecture (opt-in via
  `train.py --architecture`, default stays `temporal_cnn` -- RULE 2).
- `train.py` now uses the official train/val split from `index.csv`'s
  `split` column when present (loud warning + random-split fallback when
  it isn't -- never silent), and checkpoints record full v2 metadata
  (`checkpoint_meta.py`: architecture, feature schema version, vocab
  hash, training config, best validation accuracy).
- New `split_utils.py`, `eval_metrics.py`, `evaluate.py` -- dedicated
  official TEST-set evaluation (Top-1/3/5, macro/weighted F1, per-class
  metrics, confusion matrix PNG), deliberately separate from training so
  "validation accuracy" and "test accuracy" can never be confused (RULE 3).
- `api.py` gained `/recognize` (keypoints in, glosses out, no notes --
  the distributed-fallback recognition endpoint), enriched `/model/meta`
  (feature schema version, architecture, vocab hash, compatibility
  warnings), and request validation (invalid mode/style, oversized
  uploads, gloss/frame count limits). `backend/server.js` proxies
  `/recognize`. `frontend/src/lib/api.ts` gained a typed
  `recognizeFromFeatures()` client.
- 131 tests now cover this work (94 Python -- schema, extraction, notes,
  checkpoint metadata, splits, eval metrics, all torch-free and actually
  run; 37 TypeScript). The torch-dependent Python modules (`model.py`,
  `train.py`, `dataset.py`, `infer.py`, `api.py`) are syntax- and
  pyflakes-checked but not execution-tested in this environment --
  PyTorch couldn't be installed here (disk-constrained sandbox, see
  DEVELOPER_GUIDE.md); they need a real run on a machine with PyTorch
  before the numbers in this README can be trusted end-to-end.

**Live Transcription mode (this session)** -- Mode 3.
- Extracted `frontend/src/lib/useSignRecognitionSession.ts`, a shared hook
  covering model/camera loading, the Holistic detection loop, and session
  history editing -- now used by both `Webcam.tsx` (Mode 2) and the new
  `LiveTranscription.tsx` (Mode 3), instead of duplicating that logic.
  Also adds a real (not faked) screen/tab capture input source via
  `getDisplayMedia`, alongside the webcam, behind the same pipeline
  (project brief section 22).
- New Layer-2 output -- a natural-language TRANSCRIPT, genuinely distinct
  from structured notes (Layer 3): `notes_generator.py` gained
  `build_transcript_prompt()` / `template_transcript_from_tokens()`,
  `infer.py` gained `generate_transcript()`, and `api.py`/`server.js`/
  `api.ts` gained a `/transcript` endpoint (kept separate from `/notes`
  rather than overloading one endpoint with two response shapes).
- The LLM is called exactly once per session, only on "Stop & Generate"
  -- never per-sign, never continuously. That flow's duplicate-request
  guard, empty-session guard, and error/retry-with-preserved-glosses path
  are covered by a real component test
  (`LiveTranscription.test.tsx`, 5 tests, including simulating a rapid
  double-click on Stop & Generate).
- Fixed a real bug while adding the transcript path: `generate_notes()`'s
  LLM-failure fallback wasn't passing the requested `style` through to
  the template renderer, so a failed LLM call silently produced concise-
  style notes regardless of what the user asked for.
- Added the missing `@testing-library/dom` peer dependency (required by
  `@testing-library/react` but never declared) and switched
  `src/test/setup.ts` to `@testing-library/jest-dom/vitest` so its
  matcher types (`toBeInTheDocument()` etc.) are recognized by `tsc`, not
  just at vitest runtime -- the previous setup passed tests but failed a
  standalone typecheck.

**DOCX export + full documentation pass (this session)**
- `frontend/src/lib/notesExport.ts` gained `downloadDocx()` (via the
  `docx` npm library, dynamically imported and code-split like `jspdf`)
  -- a simple document (title, date/duration, headings/bullets/paragraphs)
  with bold/italic preserved as real Word formatting rather than
  stripped, via a new `parseInlineRuns()` inline-emphasis tokenizer. Real
  DOCX generation (not just "doesn't throw") is verified by checking
  actual ZIP magic bytes on the `Packer.toBuffer()` output, the same
  rigor as the confusion-matrix PNG check. Added to the same export row
  as Markdown/Text/PDF in `NotesPanel.tsx`.
- Found and fixed a real syntax bug while writing this: a docstring
  containing the literal characters `**bold**/*italic*` accidentally
  closed its own `/** ... */` comment early (the `*/` inside the text),
  silently corrupting everything after it until `tsc` caught it.
- README gained a "Distributed architecture" section (previously
  documented only in `ARCHITECTURE.md`'s endpoint table, not explained
  for a reader here) and an updated intro covering all three modes,
  hand+body+face recognition, and DOCX export -- neither had been
  updated as those features were added incrementally.
- Full pass over `ARCHITECTURE.md`, `SETUP.md`, `DEVELOPER_GUIDE.md`,
  `PRIVACY.md`, `TROUBLESHOOTING.md` plus a repository-wide grep audit
  for the stale-claim terms called out in the project brief (WLASL, ASL,
  2000-class, 400-class, "offline", HMM/Viterbi, "continuous
  translation", "replaces interpreters", 83.5%, 126-dim) -- see each
  file's own history for what changed.
