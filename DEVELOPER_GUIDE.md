# Developer Guide

This is a codebase tour for someone picking up this project — what each
file does, where to make common changes, and how the test suite is
organized. For install steps see `SETUP.md`; for how the pieces fit
together see `ARCHITECTURE.md`.

## Repository layout

```text
Sign2Note/
├── ml_service/           # Python: feature extraction, training, inference, API
│   ├── api.py             # FastAPI app — all HTTP endpoints
│   ├── infer.py           # sliding-window inference, LLM/template notes+transcripts
│   ├── inference_viterbi.py  # Viterbi smoothing (flat list + timestamped events)
│   ├── notes_generator.py    # deterministic template engine + LLM prompts (notes: 3 styles; transcript: 1)
│   ├── feature_schema.py     # canonical hand+body+face feature schema (see FEATURE_SCHEMA.md)
│   ├── feature_extraction.py # video -> keypoints, MediaPipe HolisticLandmarker (WLASL legacy + FDMSE-ISL)
│   ├── generate_crosscheck_fixture.py # regenerates the Python<->TS golden-vector test fixture
│   ├── build_index.py        # metadata -> data/index.csv + config/vocab.json
│   ├── checkpoint_meta.py    # torch-free: vocab hashing, checkpoint metadata, compatibility checks
│   ├── split_utils.py        # torch-free: official/random train-val split, test-split resolution
│   ├── eval_metrics.py       # torch-free: Top-k accuracy, F1, confusion matrix (+ PNG)
│   ├── evaluate.py           # official TEST-split evaluation (needs torch)
│   ├── dataset.py            # PyTorch Dataset, with augmentation
│   ├── model.py               # TemporalCNN (default) + optional CNNBiLSTM
│   ├── train.py                # training loop, ONNX export
│   ├── requirements.txt
│   ├── .env.example
│   └── tests/                  # pytest suite (see "Testing" below for torch-free vs. torch-dependent)
├── backend/                # Node/Express gateway (proxies to ml_service)
│   └── server.js
├── frontend/                # React (Vite) app
│   └── src/
│       ├── pages/Index.tsx     # video upload page (Mode 1)
│       ├── pages/Webcam.tsx    # live webcam session page (Mode 2)
│       ├── pages/LiveTranscription.tsx # live transcription page (Mode 3)
│       ├── lib/api.ts            # backend HTTP client
│       ├── lib/webcamPipeline.ts # pure logic: keypoint buffering, smoothing state machine
│       ├── lib/useSignRecognitionSession.ts # shared hook: camera+model+recognition, used by Webcam.tsx and LiveTranscription.tsx
│       ├── lib/featureSchema.ts  # canonical hand+body+face feature schema (mirrors ml_service/feature_schema.py)
│       ├── lib/onnxSession.ts    # onnxruntime-web wrapper
│       ├── lib/holisticLandmarker.ts # @mediapipe/tasks-vision HolisticLandmarker wrapper
│       ├── lib/notesExport.ts    # client-side Markdown/Text/PDF/DOCX export
│       ├── lib/markdown.ts       # Markdown-subset -> HTML renderer, shared by NotesPanel/ResultsPanel
│       ├── components/NotesPanel.tsx # Preview -> Edit -> Export panel (used by all 3 modes)
│       ├── components/           # other shared UI (PrivacyBanner, ResultsPanel, ...)
│       ├── lib/__tests__/        # vitest suite (pure logic)
│       └── pages/__tests__/      # vitest suite (rendered-component tests)
├── data/                    # FDMSE-ISL dataset + metadata (see README.md)
├── config/vocab.json         # label2id/id2label, written by build_index.py
├── models/sign_recog_v2/     # checkpoints + ONNX export (gitignored, generated)
├── README.md
├── ARCHITECTURE.md
├── FEATURE_SCHEMA.md
├── SETUP.md
├── TROUBLESHOOTING.md
├── PRIVACY.md
└── DEVELOPER_GUIDE.md (this file)
```

## Common tasks

### Adding a new note-generation style

Edit `_SECTION_KEYWORDS`/`style_text` maps in
`ml_service/notes_generator.py` — both `template_notes_from_tokens()` and
`build_notes_prompt()` read from small keyword/style dictionaries at the
top of the file, so adding a style or a new section-keyword category is a
localized change.

### Adding a new LLM provider

`infer.py`'s `LLM_PROVIDER`/`LLM_MODEL`/`LLM_BASE_URL` are just three env
vars feeding one OpenAI-compatible client — any provider that speaks
`/v1/chat/completions` (which is most local LLM servers at this point)
works without code changes, just a different `LLM_BASE_URL`. If you need a
provider with a genuinely different API shape, add a branch in
`_get_llm_client()`/`generate_llm_notes()`/`generate_llm_transcript()`.

### Changing the confidence threshold behavior

See the table in `ARCHITECTURE.md`'s "Confidence handling reference" —
there are two independent knobs (upload vs. webcam) by design; changing
one doesn't affect the other.

### Adding a field to the `/process` response

Add it in `ml_service/api.py`'s return dict, then add the corresponding
field to `ProcessResult` in `frontend/src/lib/api.ts` (TypeScript won't
stop you from omitting this, but `ResultsPanel.tsx` won't see the new
field until you thread it through).

### Modifying the live recognition pipeline

Almost everything you'd want to touch is in
`frontend/src/lib/webcamPipeline.ts` and `frontend/src/lib/featureSchema.ts`,
which have **zero browser API dependencies** — plain functions/classes
operating on typed arrays, specifically so they can be unit tested
without a browser or camera. If you change anything here (buffer size,
smoothing rule, normalization), run `npm test` in `frontend/` and update
`src/lib/__tests__/webcamPipeline.test.ts` (session/buffer logic) and
`src/lib/__tests__/featureSchema.test.ts` (feature vector/normalization
logic) to match. The browser-touching code
(`onnxSession.ts`, `holisticLandmarker.ts`,
`useSignRecognitionSession.ts`, `Webcam.tsx`, `LiveTranscription.tsx`)
should stay as thin wrappers/UI around these modules, not grow additional
logic — `useSignRecognitionSession.ts` in particular is the one place
that owns camera/model lifecycle and the detection loop, shared by both
pages, so a fix there fixes both instead of needing to be duplicated.

**Important:** `featureSchema.ts`'s `buildFeatureVector()` and
`normalizeSequence()` must stay numerically identical to their Python
counterparts (`feature_schema.py: build_feature_vector()` and
`normalize_sequence()`) — if you change the Python side's preprocessing,
mirror the change here too, regenerate the cross-check fixture
(`cd ml_service && python3 generate_crosscheck_fixture.py`), and confirm
`featureSchema.crosscheck.test.ts` still passes. Silently drifting here
doesn't error — it produces wrong-but-plausible-looking predictions, which
is much harder to notice than a crash.

## Testing

### Python (`ml_service/tests/`, pytest)

```powershell
cd ml_service
pytest -v
```

Two groups, because of a real environment constraint hit while building
this: `model.py`/`train.py`/`dataset.py`/`infer.py`/`api.py` all import
`torch`, and PyTorch could not be installed in the sandbox this project
was largely built in (disk-constrained container; the CUDA dependency
wheels alone need more space than was available, and no CPU-only wheel
index was reachable). So:

- **Torch-free, verified working in that environment** --
  `test_feature_schema.py`, `test_feature_extraction.py`,
  `test_notes_generator.py`, `test_viterbi.py`, `test_checkpoint_meta.py`,
  `test_split_utils.py`, `test_eval_metrics.py`. These cover the
  canonical hand+body+face feature schema (missing-landmark behavior,
  normalization, dimension guards), the Holistic-based extraction glue,
  both note-generation output layers (structured notes in three styles,
  the natural-language transcript) with their deterministic fallbacks,
  Viterbi decoding, checkpoint metadata assembly/compatibility checking,
  train/val/test split resolution (including the official-split path and
  its random-split fallback), and evaluation metrics (Top-k accuracy,
  macro/weighted F1, per-class metrics, confusion matrix -- including a
  real rendered PNG, checked via its actual magic bytes, not just "didn't
  throw").
- **Written and reviewed, not executed in that environment** --
  `test_dataset.py`, `test_api.py`, `test_normalization.py`,
  `test_infer_notes.py` (all import something that imports `torch`).
  Syntax-checked (`python -m py_compile`) and statically checked for
  undefined names (`python -m pyflakes *.py tests/*.py`) -- the latter
  is worth running after any change here, since it catches real bugs a
  syntax check alone won't (e.g. a function call left referencing a name
  that was renamed elsewhere). Run the full suite yourself on a machine
  with PyTorch before trusting these.

### Frontend (`frontend/src/`, vitest)

```powershell
cd frontend
npm test
```

- `lib/__tests__/webcamPipeline.test.ts` -- the session-specific logic
  that's actually still in `webcamPipeline.ts` post-schema-v2 (the
  `KeypointBuffer` ring/padding behavior, `SessionSmoother`'s three-zone
  confidence gating and commit/reject/stability state machine, undo
  behavior).
- `lib/__tests__/featureSchema.test.ts` -- the canonical feature schema's
  own TypeScript-side unit tests (missing-landmark cases, dimension
  bookkeeping, normalization, the pose-visibility z-score exclusion).
- `lib/__tests__/featureSchema.crosscheck.test.ts` -- the cross-language
  golden-vector test: feeds a fixture generated BY
  `ml_service/generate_crosscheck_fixture.py` through this file's
  `buildFeatureVector()`/`normalizeSequence()` and asserts the output
  matches Python's, to 3-4 decimal places. Regenerate the fixture and
  re-run this test after touching either side's schema logic.
- `lib/__tests__/notesExport.test.ts` -- Markdown/plain-text formatting,
  the inline bold/italic tokenizer, and real DOCX generation (verified
  via the actual ZIP magic bytes on `Packer.toBuffer()`'s output, not
  just "the promise resolved").
- `pages/__tests__/LiveTranscription.test.tsx` -- a real rendered-
  component test (mocking only the camera/model layer via
  `useSignRecognitionSession`) covering the Stop & Generate state
  machine: the duplicate-request guard on a simulated rapid double-click,
  the empty-session guard, and the error/retry-with-preserved-glosses
  path.

### What's NOT covered by automated tests

Neither the webcam nor the live-transcription feature has been run in an
actual browser with an actual camera as part of this work -- both are
type-checked (`npx tsc -p tsconfig.app.json --noEmit`) and confirmed to
build successfully (`npm run build`), but the real runtime combination of
`getUserMedia`/`getDisplayMedia`, MediaPipe's WASM Holistic tracking, and
ONNX Runtime Web's WASM inference together has not been exercised -- that
needs a real browser + camera, unavailable in the environment this was
built in. See "Known Limitations" in `README.md`.

## Style/conventions

- Python: standard library + the packages in `requirements.txt`, no extra
  formatting tooling enforced.
- TypeScript: `tsconfig.app.json` has `strictNullChecks: false` and
  `noImplicitAny: false` (inherited from the original project scaffold) —
  new code should still prefer explicit types where practical, but the
  compiler won't enforce it.
- Keep browser-API-touching code separate from pure logic wherever
  feasible (see `webcamPipeline.ts` above) — it's the difference between
  code you can unit test in milliseconds and code you can only check by
  hand in a real browser.
