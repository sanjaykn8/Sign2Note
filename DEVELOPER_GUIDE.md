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
│   ├── infer.py           # sliding-window inference, LLM/template notes
│   ├── inference_viterbi.py  # Viterbi smoothing (flat list + timestamped events)
│   ├── notes_generator.py    # deterministic template engine + LLM prompt
│   ├── feature_extraction.py # video -> keypoints (WLASL legacy + FDMSE-ISL)
│   ├── build_index.py        # metadata -> data/index.csv + config/vocab.json
│   ├── dataset.py            # PyTorch Dataset, with augmentation
│   ├── model.py               # TemporalCNN
│   ├── train.py                # training loop, ONNX export
│   ├── requirements.txt
│   ├── .env.example
│   └── tests/                  # pytest suite
├── backend/                # Node/Express gateway (proxies to ml_service)
│   └── server.js
├── frontend/                # React (Vite) app
│   └── src/
│       ├── pages/Index.tsx     # video upload page
│       ├── pages/Webcam.tsx    # live webcam session page
│       ├── lib/api.ts            # backend HTTP client
│       ├── lib/webcamPipeline.ts # pure logic: keypoints, normalize, smoothing
│       ├── lib/onnxSession.ts    # onnxruntime-web wrapper
│       ├── lib/handLandmarker.ts # @mediapipe/tasks-vision wrapper
│       ├── components/           # shared UI (PrivacyBanner, ResultsPanel, ...)
│       └── lib/__tests__/        # vitest suite
├── data/                    # FDMSE-ISL dataset + metadata (see README.md)
├── config/vocab.json         # label2id/id2label, written by build_index.py
├── models/sign_recog/        # checkpoints + ONNX export (gitignored, generated)
├── README.md
├── ARCHITECTURE.md
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
`_get_llm_client()`/`generate_llm_notes()`.

### Changing the confidence threshold behavior

See the table in `ARCHITECTURE.md`'s "Confidence handling reference" —
there are two independent knobs (upload vs. webcam) by design; changing
one doesn't affect the other.

### Adding a field to the `/process` response

Add it in `ml_service/api.py`'s return dict, then add the corresponding
field to `ProcessResult` in `frontend/src/lib/api.ts` (TypeScript won't
stop you from omitting this, but `ResultsPanel.tsx` won't see the new
field until you thread it through).

### Modifying the webcam recognition pipeline

Almost everything you'd want to touch is in
`frontend/src/lib/webcamPipeline.ts`, which has **zero browser API
dependencies** — it's plain functions/classes operating on typed arrays,
specifically so it can be unit tested without a browser or camera. If
you change anything here (buffer size, smoothing rule, normalization),
run `npm test` in `frontend/` and update
`src/lib/__tests__/webcamPipeline.test.ts` to match. The browser-touching
code (`onnxSession.ts`, `handLandmarker.ts`, `Webcam.tsx`) should stay as
thin wrappers/UI around this module, not grow additional logic.

**Important:** `keypointsFromLandmarks()` and `normalizeWindow()` must
stay numerically identical to their Python counterparts
(`feature_extraction.py: _extract_keypoints()` and
`infer.py: _normalize()`) — if you change the Python side's preprocessing,
mirror the change here too, or the browser model will silently produce
garbage predictions (wrong-but-plausible-looking output, not an error).

## Testing

### Python (`ml_service/tests/`, pytest)

```powershell
cd ml_service
pytest -v
```

Covers: keypoint normalization, feature extraction output shape, dataset
loading (via synthetic index.csv + .npy fixtures), label mapping,
temporal window generation, Viterbi smoothing/duplicate collapse (both
the flat and timestamped-event decoders), and note templating (both
vocabularies — the spec's example words and real FDMSE-ISL words).

### Frontend (`frontend/src/lib/__tests__/`, vitest)

```powershell
cd frontend
npm test
```

Covers the pure webcam pipeline logic: keypoint vector construction
(including the "first-detected-hand, not real handedness" ordering quirk
that must match training), per-window normalization (cross-validated
against Python's actual output), the sliding buffer's ring/padding
behavior, softmax/argmax, and the session smoother's commit/reject/
stability logic (including the "collapse a long held sign into one event"
case and the "flicker resets stability" case).

### What's NOT covered by automated tests

The actual browser integration (`Webcam.tsx`, `onnxSession.ts`,
`handLandmarker.ts`) is type-checked (`npx tsc -p tsconfig.app.json
--noEmit`) and confirmed to build successfully (`npm run build`), but
running an actual camera, WASM hand-tracking, and WASM ONNX inference
together in a real browser has not been executed as part of this work —
that requires a real browser + camera, which wasn't available in the
environment this was built in. See "Known Limitations" in `README.md`.

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
