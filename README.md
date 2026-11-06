# Sign2Notes

Sign2Notes recognizes **isolated Indian Sign Language (ISL) signs** from
video and turns the recognized sign sequence (**glosses**) into readable
**transcripts** and structured **generated notes**. It runs on a main
laptop acting as a server; other devices on the same network can connect
through a browser.

It is a constrained-vocabulary recognizer (trained on FDMSE-ISL,
word-level clips) applied to sliding windows — **not** a complete,
continuous ISL translator. Glosses are model guesses and should be
reviewed; the app lets you edit them before generating or exporting notes.

Two separate stages, two separate services:

1. **Sign recognition** — video/camera → MediaPipe landmarks → temporal
   CNN → ordered glosses (with timestamps and confidence).
2. **Language generation** — ordered glosses → transcript or notes
   (local LLM via Ollama/llama.cpp, with a deterministic fallback that
   always works).

Three modes:

| Mode | What it does |
|---|---|
| **Video Upload** | Upload a recorded video → ordered, timestamped glosses → notes. Glosses are editable; notes can be regenerated in any style. |
| **Live Webcam** | Continuous recognition with current sign, confidence, gloss history, pause/resume, edit/delete/undo/clear; notes only when you click Generate. |
| **Live Transcription** | Live gloss transcript while signing (webcam, or a shared tab/window/screen where the browser supports it); **Stop & Generate** produces a natural-language transcript and notes once. |

Exports: Markdown, Text, PDF (Latin script only) and Word (.docx),
generated in the browser from exactly the text shown.

## Status (honest)

| Area | Status |
|---|---|
| Video Upload end to end (gateway → ML service → glosses → notes) | Implemented; 5 real FDMSE-ISL clips uploaded through the running gateway + ML service with the legacy model all returned HTTP 200 (glosses + notes). Recognition accuracy on these was not recorded (run was interrupted by low battery) |
| Recognition with the legacy hand-only model (`models/sign_recog`, schema v1) | Works (served with its own 126-dim layout) |
| Hand + body + face model (schema v2) | Pipeline implemented and tested on synthetic data; **feature extraction for the full dataset is partial (≈5.7k of 40k clips) and no v2 model has been trained yet** — see "Model training status" |
| Official-split evaluation (val and test reported separately) | Implemented and tested on synthetic data; not yet run on a real v2 model |
| Notes: concise / detailed / academic, deterministic fallback | Implemented and tested (fallback); LLM path tested with a fake server and with an unreachable server — not with a real local LLM in this pass |
| Live Webcam / Live Transcription | Implemented; unit/component tested; **not exercised with a real camera in this pass** |
| Local recognizer → server notes, or server recognition fallback | Implemented (compatibility verified by schema/dims + a test inference); unit tested; not tested across two physical devices |
| Network auth, timeouts, LAN binding | Implemented and tested (gateway tests) |
| DOCX / Markdown / Text / PDF export | Implemented; DOCX content verified by unzipping in tests |

See [CHANGELOG.md](CHANGELOG.md) for the baseline found, each bug fixed and
its regression test.

## Quick start (Windows, main laptop)

Prerequisites: Python 3.11, Node.js 20+ (tested with 22), the FDMSE-ISL
dataset if you want to train. Run from the repository root.

```powershell
# 1. ML service environment (isolated venv; see SETUP.md for why)
py -3.11 -m venv ml_service\venv
ml_service\venv\Scripts\python -m pip install -r ml_service\requirements.txt

# 2. Node dependencies
cd backend;  npm install; cd ..
cd frontend; npm install; cd ..
```

Start the three services in three terminals:

```powershell
cd ml_service; .\venv\Scripts\python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

```powershell
cd backend; npm start
```

```powershell
cd frontend; npm run dev
```

Open http://localhost:8080. With no LLM configured, notes use the
deterministic generator (labelled as such in the app).

The ML service serves `models/sign_recog_v2` if it exists, otherwise the
legacy `models/sign_recog` model (override with `MODEL_PATH`). Check
`http://localhost:3001/health`.

### Other devices on the network

```powershell
# on the main laptop, before `npm start` in backend/
$env:SIGN2NOTES_API_TOKEN = "choose-a-long-random-string"
```

With a token set, the gateway listens on the network (otherwise only on
127.0.0.1). On another device open `http://<main-laptop-ip>:8080`, open
**Connection settings** (top right) and enter the token. Camera and
screen capture need HTTPS on non-localhost pages — see
[SETUP.md](SETUP.md#connecting-other-devices).

## Documentation

- [SETUP.md](SETUP.md) — installation, configuration, LLM, LAN/HTTPS
- [ARCHITECTURE.md](ARCHITECTURE.md) — services, APIs, recognition pipeline, routing
- [FEATURE_SCHEMA.md](FEATURE_SCHEMA.md) — feature layouts (v2 hand+body+face, legacy v1)
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) — training, evaluation, export, tests, model replacement and rollback
- [PRIVACY.md](PRIVACY.md) — exactly what data leaves each device in each mode
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- [CHANGELOG.md](CHANGELOG.md)

## Model training status

- **Legacy model** (`models/sign_recog`): TemporalCNN, hand-only 126-dim
  features, 2,001 classes, 83.6 % validation accuracy on a **random 85/15
  split of all 40k clips**. Because that split mixed signers and included
  official test-split clips in training, the number is not a held-out
  result and is not comparable to official-split numbers.
- **FDMSE-ISL official splits are signer-independent** (train 10 signers,
  val 2, test 8; ~10 training clips per class). Expect official-split
  accuracy to be far below 83.6 %. See `reports/dataset_audit.md`.
- **Schema-v2 model**: not trained yet. Extraction of hand+body+face
  features for all 40,033 clips takes roughly 4 hours on this laptop's
  CPU (≈2.8 clips/s with 12 workers, measured on mains power) and was
  paused at ≈5.7k clips because the laptop was on low battery. The exact
  commands to finish extraction, train the v2 model and the hand-only
  ablation on the official split, and evaluate both are in
  [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#training-and-evaluating-a-model).
  No v2 accuracy figure exists yet; none is claimed.

## Known limitations

- Isolated-sign classifier on sliding windows: co-articulated continuous
  signing is out of scope; consecutive identical signs may merge; a sign
  that spans two windows may be reported once or twice.
- Vocabulary is fixed to the training classes; anything else is mapped to
  the nearest known gloss (low confidence is flagged, not prevented).
- Confidence is the model's softmax score, not a probability of being
  correct.
- Live sampling uses the browser's frame timing (≈7.5 samples/s); a slow
  device samples less often.
- PDF export uses a Latin-only font and refuses other scripts (use DOCX).
- Screen/tab capture depends on browser support (generally unavailable on
  phones); it never captures beyond what the user selects in the browser's
  own picker.
- The Python MediaPipe Tasks Holistic API crashes on this setup; Python
  extraction uses the classic `mp.solutions.holistic` while the browser
  uses Tasks Holistic — same landmark layout, small numeric differences.
