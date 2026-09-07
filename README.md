# Sign2Notes

Sign2Notes turns a video (uploaded, or signed live on webcam) into a
sequence of recognized sign-language glosses, then into readable Markdown
notes — running entirely on a local laptop, using a constrained-vocabulary
recognition model trained on **FDMSE-ISL**.

For details beyond this overview, see:

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — full pipeline design, long-video
  and live-webcam internals, why confidence handling differs between them
- **[SETUP.md](SETUP.md)** — detailed installation reference
- **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** — codebase tour, common
  tasks, test suite
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** — specific error messages
  and fixes
- **[PRIVACY.md](PRIVACY.md)** — exactly what does and doesn't leave your
  machine, for both the upload and webcam flows

## Quick Start

```powershell
# 1. Clone and enter the project
git clone <repo-url> Sign2Note
cd Sign2Note

# 2. Python environment
python -m venv .venv
.venv\Scripts\activate
pip install -r ml_service/requirements.txt

# 3. Node services
cd backend;  npm install; cd ..
cd frontend; npm install; cd ..

# 4. Place the FDMSE-ISL dataset (see "Dataset" below) at:
#    data/data_meta/   and   data/FDMSE-ISL/

# 5. Extract keypoints, build the vocabulary, train
python ml_service/feature_extraction.py --dataset_format fdmse --metadata_csv data/data_meta/metadata_400.csv --dataset_root data/FDMSE-ISL --out_dir data/features --frame_skip 8 --workers 4
python ml_service/build_index.py --dataset_format fdmse --metadata_csv data/data_meta/metadata_400.csv --min_samples 10
python ml_service/train.py --epochs 20 --batch_size 32 --num_workers 0

# 6. Start all three services (separate terminals)
cd ml_service; python -m uvicorn api:app --host 127.0.0.1 --port 8000
cd backend;    npm start
cd frontend;   npm run dev

# 7. Open http://localhost:8080
#    "/"        -> upload a video
#    "/webcam"  -> live webcam session
```

Deterministic notes work immediately, with no LLM required. For LLM-backed
notes, see "LLM notes" in `SETUP.md`.

## Pipeline

```text
Video (upload)                       Webcam (live)
     |                                    |
     v                                    v
MediaPipe keypoints              MediaPipe keypoints
(server-side, per-frame)         (in-browser, WASM)
     |                                    |
     v                                    v
Sliding temporal windows          Sliding keypoint buffer
     |                                    |
     v                                    v
Temporal CNN (ONNX/PyTorch)       Temporal CNN (ONNX Runtime Web)
     |                                    |
     v                                    v
Viterbi/HMM smoothing             Real-time confidence gating
+ duplicate collapse              + stability smoothing
     |                                    |
     v                                    v
Ordered, timestamped              Session gloss history
gloss sequence                    (label, confidence, timestamp)
     |                                    |
     +--------------+---------------------+
                     v
          Template Engine  or  Local LLM
          (always available)   (Ollama / llama.cpp)
                     |
                     v
          Structured Markdown notes
```

See `ARCHITECTURE.md` for the full breakdown, including a bug we found and
fixed in the sliding-window normalization while building long-video
support.

## Dataset

The project trains on **FDMSE-ISL** (not WLASL — WLASL only appears in
this codebase as a legacy/previous-development dataset format still
supported by `feature_extraction.py --dataset_format wlasl` for anyone
with existing WLASL-based work, but it is not what the shipped model uses
or what `build_index.py` defaults to).

```text
data/
  data_meta/
    classes.txt              # full 2,002-class legend
    classes_400.txt          # legend for the 400-class subset
    metadata.csv             # all 2,002 classes, 40,034 clips, 20 signers
    metadata_400.csv         # curated 400-class subset (20 samples/class) - recommended default
    metadata_atomic.csv      # single-gesture signs only (1,099 classes)
    metadata_composite.csv   # multi-word/compound signs only (352 classes)
  FDMSE-ISL/
    data/
      s0001/front/*.mp4
      s0002/front/*.mp4
      ...
```

Each metadata CSV has columns `id,video_dir,video_name,class,split`:
- `video_dir` -- path to the clip, **relative to `data/FDMSE-ISL`**
  (e.g. `data/s0015/front/s0015_f_w000842.mp4`)
- `class` -- the gloss label (e.g. `"Whistle"`) -- used directly as the
  training label, no separate lookup table needed
- `split` -- the dataset's own train/val/test assignment. **Not currently
  used by `train.py`**, which does its own random split via `--val_split`
  -- this is a known simplification, listed below.

`metadata_400.csv` is the recommended starting vocabulary -- training all
2,002 classes is not realistic for a demo-scale model or a single laptop
GPU. `metadata_atomic.csv` (single-gesture signs, 1,099 classes) is a good
middle ground if you want broader coverage and can tolerate a longer
extraction/training run.

`feature_extraction.py --dataset_format fdmse` reads a metadata CSV and
writes one `.npy` keypoint file per clip (video filename stem as the
video_id -- these are unique across all 20 signers). `build_index.py`
turns a metadata CSV plus the extracted `.npy` files into `data/index.csv`
(video_id, label, split) and `config/vocab.json` (label2id/id2label),
filtering to classes with at least `--min_samples` examples.

## Training

```powershell
# 1. Extract keypoints
python ml_service/feature_extraction.py `
  --dataset_format fdmse `
  --metadata_csv data/data_meta/metadata_400.csv `
  --dataset_root data/FDMSE-ISL `
  --out_dir data/features `
  --frame_skip 8 `
  --workers 4

# 2. Build the vocabulary + index
python ml_service/build_index.py `
  --dataset_format fdmse `
  --metadata_csv data/data_meta/metadata_400.csv `
  --min_samples 10

# 3. Train
python ml_service/train.py --epochs 20 --batch_size 32 --num_workers 0

# 4. (train.py also exports ONNX automatically at the end)
```

Outputs:
```text
models/sign_recog/checkpoints/best.pt
models/sign_recog/checkpoints/demo.pt
models/sign_recog/sign_recog.onnx
models/sign_recog/sign_recog.json    # {input_dim, max_len, num_classes, ...}
```

### Recommended settings for an RTX 4050 6GB

| Setting | Recommendation | Why |
|---|---|---|
| `--batch_size` | 32 (try 64 if VRAM allows) | The model is small (1D-conv TemporalCNN), so 6GB comfortably fits a moderate batch |
| Sequence length (`max_len`, set at `build_index.py`/dataset level, not a train.py flag) | Keep at the dataset's default window length unless you have a specific reason to change it | Longer windows cost more compute per sample for limited accuracy gain on short isolated signs |
| `--frame_skip` (at extraction time) | 8 | Halves keypoint volume vs. `frame_skip=4` with little accuracy cost for isolated-sign recognition; raise further only if extraction is a bottleneck |
| `--num_workers` | **0 on Windows** | Avoids multiprocessing spawn errors in PyTorch's `DataLoader` on Windows; raise on Linux/Mac if I/O-bound |
| AMP (mixed precision) | Not required at this model size, but safe to enable if you add it -- the model is small enough that AMP mainly helps if you significantly scale up `max_len` or batch size |
| Expected VRAM | Well under 6GB at these settings -- this model was deliberately kept lightweight (1D convolutions, not a Transformer) specifically so it fits both a 6GB laptop GPU for training AND a browser via WASM for the live webcam demo |

`INFER_CHUNK_SIZE` (env var, default 64) controls how many sliding windows
are batched per forward pass **at inference time** for long videos --
lower it if you're inference-testing on a more memory-constrained machine
than you trained on.

### Why the pipeline uses windows + Viterbi

FDMSE-ISL (like WLASL before it) is word-level: each clip is one isolated
sign, not a sentence-level aligned continuous-signing dataset. To handle
videos containing a *sequence* of signs, inference breaks the extracted
keypoint sequence into overlapping windows, predicts a gloss distribution
per window, and uses Viterbi smoothing to favor stable transitions and
collapse repeated predictions into single events -- see `ARCHITECTURE.md`
for the full design and a normalization bug we found and fixed while
building this.

## Live webcam demo

Click "Live Webcam" in the nav, then **Start Session** and sign -- a
"Current Sign" display and a running "Sign History" update live. Click
**Generate Notes** when done (or **Stop Session** first if you want to
review the history before generating). **Clear Session** resets
everything. See `ARCHITECTURE.md` for the full client-side pipeline and
`PRIVACY.md` for exactly what does/doesn't leave the browser.

## LLM notes

Two modes, always available in the UI:

- **Deterministic template** (`notes_mode="template"`) -- no LLM needed,
  groups recognized glosses into Key Concepts/Questions/Tasks sections
  where it recognizes structuring keywords, otherwise a flat list.
- **Local LLM** (`notes_mode="llm"`) -- calls a local Ollama or llama.cpp
  server, configured via `ml_service/.env` (`LLM_PROVIDER`, `LLM_MODEL`,
  `LLM_BASE_URL` -- see `.env.example`). Falls back to the deterministic
  template automatically if the server is unreachable -- you always get
  notes back.

## What Was Changed

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

## Known Limitations

Being explicit about what this is and isn't, per the project's own
development principles:

- **Constrained vocabulary, not general ISL translation.** The model
  recognizes isolated signs from whatever vocabulary it was trained on
  (400 classes by default) -- it does not perform continuous,
  general-purpose Indian Sign Language translation, and should not be
  presented as such.
- **Isolated-sign vs. continuous-language gap.** FDMSE-ISL, like WLASL, is
  a word-level dataset (one clip = one sign). The sliding-window + Viterbi
  approach lets the system report *multiple* signs from a longer video,
  but this is pipeline engineering on top of an isolated-sign classifier --
  it is not the same as a model trained on continuous, sentence-level,
  co-articulated signing, which would require a sequence-aligned dataset
  and likely a CTC/sequence model.
- **Model accuracy depends heavily on training data coverage** -- camera
  angle, lighting, signer variation, and vocabulary size all matter a lot
  at this scale. `/process`'s "always return a result" fallback
  (majority-vote best guess when nothing crosses the confidence threshold)
  is a deliberate UX choice for the upload flow, not a claim that the
  fallback result is reliable -- see `ARCHITECTURE.md`.
- **The live webcam's real-time smoothing is a simplified heuristic**
  (N consecutive agreeing predictions -> commit), not the same
  Viterbi/HMM decoding the batch upload path uses -- a live stream doesn't
  have the whole clip's probabilities available up front the way a
  finished video upload does. Documented in `ARCHITECTURE.md`.
- **The webcam feature has not been tested in a real browser with a real
  camera as part of this work.** Everything reachable without a browser
  has been verified for real: the pure logic
  (`webcamPipeline.ts`) is unit-tested and numerically cross-validated
  against the Python reference implementation, the whole frontend
  type-checks cleanly and builds successfully with Vite (proving all
  imports/dependencies resolve), and the backend endpoints it calls
  (`/model/meta`, `/model/onnx`, `/notes`) are tested end-to-end. What
  hasn't been exercised is the actual runtime combination of
  `getUserMedia`, MediaPipe's WASM hand tracking, and ONNX Runtime Web's
  WASM inference together in a live browser tab -- that requires an actual
  browser + camera, unavailable in the environment this was built in.
  Test this first before relying on it for a live demo.
- **`train.py` does its own random train/val split**, not FDMSE-ISL's
  provided `split` column (preserved in `data/index.csv` for future use).
  Fine for MVP iteration; a rigorous accuracy evaluation should use the
  dataset's official split instead.
- **Hardware target is a single 6GB-class laptop GPU** (or CPU/browser
  WASM for the live demo) -- the model is deliberately small; don't expect
  large-model-level accuracy.
- **LLM-backed notes depend on a local server being available** (Ollama
  or llama.cpp) -- the deterministic template mode is the reliable
  fallback and works with zero external dependencies.
