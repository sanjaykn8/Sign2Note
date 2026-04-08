# Sign2Notes

Upload a sign-language video → get structured lecture notes.

## Architecture

```
React (port 3000)  →  Express backend (port 3001)  →  FastAPI ML service (port 8000)
```

## Setup

### 1. Python environment

```bash
cd ml_service
pip install -r requirements.txt
```

### 2. Node (backend + frontend)

```bash
cd backend && npm install
cd ../frontend/react-app && npm install
```

### 3. Download WLASL dataset

Place files at:
```
data/
  wlasl/
    WLASL_v0.3.json
    videos/
      <video_id>.mp4
      ...
```

## Training

### Step 1 – Extract features (parallelised, ~1–2 hr for full dataset)

```bash
cd ml_service
python feature_extraction.py \
  --videos_dir ./data/wlasl/videos \
  --out_dir    data/features \
  --wlasl_json ./data/wlasl/WLASL_v0.3.json \
  --frame_skip 8 \
  --workers    8
```

### Step 2 – Build index CSV

```bash
python build_index.py
```

### Step 3 – Train model (~30 min GPU / ~1–2 hr CPU)

```bash
python train.py \
  --epochs     30 \
  --batch_size 64 \
  --num_workers 4
```

This saves:
- `models/sign_recog/checkpoints/demo.pt`
- `models/sign_recog/sign_recog.onnx`

## Running

```bash
# Terminal 1 – ML service
cd ml_service
uvicorn api:app --host 0.0.0.0 --port 8000

# Terminal 2 – Node backend
cd backend
npm start          # listens on :3001

# Terminal 3 – React frontend
cd frontend/react-app
npm start          # opens on :3000
```

## Quick inference test

```bash
cd ml_service
python infer.py --video_path ../backend/uploads/test.mp4
```

## Notes generation modes

| Flag | Requires | Quality |
|------|----------|---------|
| (none) | nothing | template-based, always works |
| `--use_ollama` | Ollama running with llama3.2:3b | best quality |
| `--use_llama` + `--gguf_path` | llama.cpp binary + GGUF model | good quality |
