# Setup

All commands are run from the repository root unless a `cd` is shown.
PowerShell syntax; `py -3.11` is the global Python 3.11 (on macOS/Linux use
`python3.11` and `run.sh`).

## Prerequisites

- Python **3.11** (MediaPipe 0.10.21 wheels exist for 3.9–3.12)
- Node.js 20+ (verified with 22.22)
- Optional: an NVIDIA GPU for faster training (everything also runs on CPU)
- Optional: the FDMSE-ISL dataset under `data/` (only needed to train/evaluate)
- Optional: Ollama or llama.cpp for LLM-generated notes

## 1. ML service (Python)

Install into the global Python 3.11. For GPU training install the CUDA
build of PyTorch from pytorch.org first; otherwise `requirements.txt` pulls
the CPU build.

```powershell
py -3.11 -m pip install -r ml_service\requirements.txt
```

`requirements.txt` pins **NumPy < 2** and **protobuf < 5**, which
mediapipe 0.10.21 requires. With NumPy 2.x / protobuf 6 installed,
`import mediapipe` fails, and if TensorFlow is installed too its import
fails first (`_pywrap_checkpoint_reader`, `_ARRAY_API`). TensorFlow 2.17,
PyTorch 2.x, OpenCV 4.11 and JAX 0.4 all work with NumPy 1.26. If another
project needs NumPy 2, give one of the two its own environment.

Check:

```powershell
py -3.11 -c "import torch, mediapipe, numpy; print(mediapipe.__version__, numpy.__version__, torch.cuda.is_available())"
```

Expected: `0.10.21 1.26.x True` (`False` without a usable NVIDIA GPU; training then runs on CPU).

Configuration: copy `ml_service/.env.example` to `ml_service/.env` and
adjust. Important variables:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | v2 `best.pt` if present, else legacy `models/sign_recog/checkpoints/best.pt` | Recognition checkpoint to serve |
| `ONNX_PATH` | `<model dir>/sign_recog.onnx` | ONNX export paired with that checkpoint (dims are checked) |
| `CONFIDENCE_THRESHOLD` | 0.55 | Default window threshold for uploads |
| `TRAINING_SAMPLE_HZ` | 7.5 | Sampling rate the model was trained at (60 fps / frame_skip 8) |
| `WINDOW_FRAMES`, `WINDOW_STRIDE` | 24, 4 | Sliding-window length/stride in sampled frames |
| `MAX_UPLOAD_SIZE`, `MAX_VIDEO_SECONDS` | 200 MB, 1800 s | Upload limits |
| `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL` | llama_cpp, gemma4, http://127.0.0.1:8081/v1 | Language-generation server |
| `LLM_TIMEOUT`, `LLM_MAX_RETRIES` | 90 s, 1 | Bounded LLM calls (then deterministic fallback) |
| `HOLISTIC_BACKEND` | `solutions` | Python landmark extractor (`tasks` crashes on 0.10.21/Windows) |

Start: `cd ml_service; py -3.11 -m uvicorn api:app --host 127.0.0.1 --port 8000`

Keep `--host 127.0.0.1`: only the gateway talks to the ML service.

## 2. Gateway (Express)

```powershell
cd backend; npm install; npm start
```

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | 3001 | |
| `HOST` | `127.0.0.1`, or `0.0.0.0` when a token is set | Bind address |
| `SIGN2NOTES_API_TOKEN` | (none) | Shared access token; required for network access |
| `CORS_ORIGINS` | reflect origin | Comma-separated allowed origins |
| `ML_SERVICE_URL` | http://127.0.0.1:8000/process | ML service location |
| `MAX_UPLOAD_MB` | 200 | Upload limit |
| `TIMEOUT_*_MS` | see server.js | Upstream timeouts |

## 3. Frontend (React/Vite)

```powershell
cd frontend; npm install; npm run dev     # http://localhost:8080
npm run build                             # production build into frontend/dist
```

Optional `frontend/.env` — see `frontend/.env.example` (`VITE_API_URL`,
`VITE_MAX_UPLOAD_MB`). The server address can also be changed at runtime in
**Connection settings**.

The browser downloads the MediaPipe WASM runtime and the Holistic model
from jsDelivr / Google's model storage the first time a live mode starts
(cached afterwards). See PRIVACY.md for self-hosting them.

## 4. Language generation (optional)

Any OpenAI-compatible local server works.

- **llama.cpp**: `llama-server.exe -m <model.gguf> --port 8081 --alias gemma4`
  → `LLM_PROVIDER=llama_cpp`, `LLM_BASE_URL=http://127.0.0.1:8081/v1`, `LLM_MODEL=gemma4`
- **Ollama**: `ollama pull <model>` → `LLM_PROVIDER=ollama`,
  `LLM_BASE_URL=http://127.0.0.1:11434/v1`, `LLM_MODEL=<model>`

If the server is unreachable, slow (`LLM_TIMEOUT`) or returns unusable
output, the deterministic generator is used and the app labels the result
"LLM unavailable — deterministic fallback used".

## Connecting other devices

1. On the main laptop set a token and start the gateway:
   `$env:SIGN2NOTES_API_TOKEN = "<long random string>"; cd backend; npm start`
   (it now listens on all interfaces). Allow ports 3001 and 8080 through
   Windows Firewall for the **Private** network profile only.
2. Find the laptop's LAN address: `ipconfig` → IPv4 address, e.g. 192.168.1.20.
3. On the other device open `http://192.168.1.20:8080`, click the status
   indicator (top right) → **Connection settings**, keep the address
   `http://192.168.1.20:3001`, enter the token, **Save**.
4. **Camera / screen capture on other devices needs HTTPS** (browsers only
   allow them on `https://` or `localhost`). Video Upload works over plain
   http. For the live modes, create a certificate for the laptop's address
   (for example with `mkcert 192.168.1.20`, installing mkcert's root CA on
   the client devices) and start the frontend with:
   ```powershell
   $env:SIGN2NOTES_HTTPS_KEY="C:\certs\192.168.1.20-key.pem"; $env:SIGN2NOTES_HTTPS_CERT="C:\certs\192.168.1.20.pem"; cd frontend; npm run dev
   ```
   The gateway is then reached by the page over `http://...:3001`, which
   browsers block as mixed content from an https page — put the gateway
   behind the same certificate (e.g. a reverse proxy) or use the client's
   own local install. This is a documented limitation, not automated here.

Never expose the gateway to the public internet; the token is a shared
secret for a trusted LAN.

## 5. Verifying the setup

```powershell
curl http://localhost:3001/health
```

Expect `"gateway":"ok"` and `ml.model_ready: true`. `ml.feature_schema_version`
shows which model is served (`1.0` = legacy hand-only, `2.0` = hand+body+face).

## 6. Tests

```powershell
cd ml_service; py -3.11 -m pytest -q                    # 167 tests (needs pytest, httpx)
cd backend; npm test                                     # 4 tests
cd frontend; npx vitest run                              # 82 tests
cd frontend; npx tsc --noEmit -p tsconfig.app.json; npx vite build
```

Training/evaluation, dataset preparation and the end-to-end check are in
DEVELOPER_GUIDE.md.
