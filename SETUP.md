# Setup

This covers installation and configuration in detail. For the fastest path
from a fresh clone to a running demo, see the **Quick Start** in
`README.md` — this document is the reference for each step, including
options and troubleshooting-adjacent notes.

## Prerequisites

- **Python** 3.10–3.12 (3.12 recommended; matches what this project was
  developed/tested against)
- **Node.js** 18+ (for both `backend/` and `frontend/`)
- **A webcam** (only needed for the live webcam demo, not video upload)
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
`numpy`, `pandas`, `tqdm`, `onnx`, `onnxruntime`, `fastapi`,
`uvicorn[standard]`, `python-multipart`, `openai` (for LLM notes mode),
and `python-dotenv` (optional `.env` loading).

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
`@mediapipe/tasks-vision` for the live webcam demo — no separate install
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

See "Training" in `README.md` for full copy-paste commands and RTX 4050
recommendations (batch size, sequence length, frame skip, workers, AMP).

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

Open `http://localhost:8080`. `/` is video upload; `/webcam` is the live
webcam demo.

## 7. Frontend environment variables (optional)

`frontend/.env` (create if you need to override defaults):

```env
VITE_API_URL=http://localhost:3001
```

## 8. Verifying your setup

```powershell
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:3001/health
```

Both should return `{"status": "ok", ...}` / `{"gateway": "ok", ...}`. If
`model_ready`/`onnx_ready` are `false`, you haven't trained a model yet —
the app still runs, but `/process` and the webcam demo will report "no
trained model found" until you do.

## 9. Running tests

```powershell
# Python (backend)
cd ml_service
pip install pytest
pytest

# Frontend
cd frontend
npm test               # vitest, single run (this is what `npm test` maps to)
npx vitest             # watch mode, if you want it while developing
```
