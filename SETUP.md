# Setup

This covers installation and configuration in detail. For the fastest path
from a fresh clone to a running demo, see the **Quick Start** in
`README.md` — this document is the reference for each step, including
options and troubleshooting-adjacent notes.

## Prerequisites

- **Python** 3.10–3.12 (3.12 recommended; matches what this project was
  developed/tested against)
- **Node.js** 18+ (for both `backend/` and `frontend/`)
- **A webcam** (for the live webcam or live transcription demos; live
  transcription can alternatively use screen/tab capture instead of a
  camera -- see "Live Transcription mode" in `README.md`). Neither is
  needed for video upload.
- **Optional:** an NVIDIA GPU for faster training (CPU-only training works,
  just slower); recommendations below assume an **RTX 4050 6GB laptop GPU**
- **Optional:** [Ollama](https://ollama.com) or
  [llama.cpp](https://github.com/ggml-org/llama.cpp) for LLM-backed notes
  (the deterministic template mode needs neither)

## 1. Python environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r ml_service/requirements.txt
```

This installs: `torch`, `torchvision`, `mediapipe`, `opencv-python`,
`numpy`, `pandas`, `tqdm`, `scikit-learn` and `matplotlib` (for
`evaluate.py`'s metrics/confusion-matrix), `onnx`, `onnxruntime`,
`fastapi`, `uvicorn[standard]`, `python-multipart`, `openai` (for LLM
notes/transcript mode), and `python-dotenv` (optional `.env` loading).

> If you have an NVIDIA GPU and want CUDA-accelerated training, install
> the CUDA build of PyTorch instead of the default CPU wheel — see
> [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for
> the exact command for your CUDA version. `onnxruntime-gpu` can replace
> `onnxruntime` the same way (commented out in `requirements.txt`).

## 2. Node services

```powershell
cd backend
npm install
cd ..\frontend
npm install
```

`frontend/package.json` includes `onnxruntime-web` and
`@mediapipe/tasks-vision` for the live webcam/transcription demos, and
`jspdf`/`docx` for client-side PDF/Word export -- no separate install
step needed.

## 3. LLM notes (optional)

Copy `ml_service/.env.example` to `ml_service/.env` and adjust:

```env
LLM_PROVIDER=llama_cpp   # or "ollama"
LLM_MODEL=gemma4         # llama.cpp: your --alias. Ollama: the model tag.
LLM_BASE_URL=http://127.0.0.1:8081/v1
CONFIDENCE_THRESHOLD=0.55
INFER_CHUNK_SIZE=64
```

### Option A — llama.cpp

```powershell
llama-server.exe `
    -m "C:\path\to\model.gguf" `
    --host 127.0.0.1 --port 8081 `
    -ngl 99 -c 2048 --alias gemma4
```

`--alias` must match `LLM_MODEL` in `.env`.

### Option B — Ollama

```powershell
ollama pull gemma4:e2b
```

Ollama listens on port 11434 by default and exposes the same
OpenAI-compatible `/v1/chat/completions` endpoint llama.cpp does, so:

```env
LLM_PROVIDER=ollama
LLM_MODEL=gemma4:e2b
LLM_BASE_URL=http://127.0.0.1:11434/v1
```

Either way, if the LLM server isn't reachable when notes are requested in
`notes_mode="llm"`, the app automatically falls back to deterministic
template notes — you don't need to have it running to use the app.

## 4. FDMSE-ISL dataset

See "Dataset" in `README.md` for the full layout and column reference.
Expected structure:

```text
data/
  data_meta/
    metadata_400.csv   # curated 400-class subset (recommended default)
    metadata.csv        # full 2,002 classes
    metadata_atomic.csv
    metadata_composite.csv
    classes.txt
    classes_400.txt
  FDMSE-ISL/
    data/
      s0001/front/*.mp4
      ...
```

## 5. Training pipeline

See "Training" and "Evaluation" in `README.md` for full copy-paste
commands, RTX 4050 recommendations (batch size, sequence length, frame
skip, workers, AMP), and how to run `evaluate.py` against the official
test split.

## 6. Running the services

Three terminals:

```powershell
# Terminal 1 — ML service
cd ml_service
python -m uvicorn api:app --host 127.0.0.1 --port 8000

# Terminal 2 — Node gateway
cd backend
npm start

# Terminal 3 — React frontend
cd frontend
npm run dev
```

Open `http://localhost:8080`. `/` is video upload, `/webcam` is the live
webcam demo, `/live-transcription` is live transcription mode.

## 7. Running a distributed client (optional)

A second laptop can run just the frontend + its own local recognition
model, pointed at the first laptop's backend for notes/transcript
generation (and as a recognition fallback if its own local model isn't
available or fails a compatibility check -- see "Distributed
architecture" in `README.md`):

```powershell
# On the CLIENT laptop:
cd frontend
npm install
```

Create `frontend/.env` pointing `VITE_API_URL` at the MAIN laptop's
backend (see "Frontend environment variables" below) -- e.g.
`VITE_API_URL=http://192.168.1.20:3001` (the main laptop's LAN IP; find
it with `ipconfig` on Windows or `ifconfig`/`ip addr` on Linux/Mac). Then:

```powershell
npm run dev -- --host
```

`--host` makes Vite's dev server listen on the LAN, not just localhost.
On the MAIN laptop, `ml_service/api.py`'s CORS origins currently only
allow `localhost`/`127.0.0.1` by default (see `api.py`'s
`CORSMiddleware` setup) -- add the client's origin
(`http://<client-ip>:8080`) there for a real multi-laptop setup, since
browsers enforce CORS based on the page's own origin, not the API's.

The client's browser will try to load its own local ONNX model from the
MAIN laptop's `/model/onnx` on first use (same-origin as `VITE_API_URL`)
and cache it -- there's currently no mechanism for a client to use a
*different* local model than what the main server serves; that's a
natural extension (see README's "Distributed architecture" section for
what's built vs. not yet).

## 8. Frontend environment variables (optional)

`frontend/.env` (create if you need to override defaults):

```env
VITE_API_URL=http://localhost:3001
```

## 9. Verifying your setup

```powershell
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:3001/health
```

Both should return `{"status": "ok", ...}` / `{"gateway": "ok", ...}`. If
`model_ready`/`onnx_ready` are `false`, you haven't trained a model yet —
the app still runs, but `/process` and the webcam demo will report "no
trained model found" until you do.

## 10. Running tests

```powershell
# Python (backend) -- requires a working PyTorch install for the full
# suite (model.py/train.py/dataset.py/infer.py/api.py tests); the
# schema/extraction/notes/checkpoint-metadata/split/eval-metrics tests
# are torch-free and will pass even without one
cd ml_service
pip install pytest
pytest

# Frontend
cd frontend
npm test               # vitest, single run (this is what `npm test` maps to)
npx vitest             # watch mode, if you want it while developing
npx tsc --noEmit -p tsconfig.app.json   # typecheck (catches issues vitest's own transform can miss)
```
