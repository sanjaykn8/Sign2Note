# Sign2Notes — Sign → Gloss → Notes MVP

Sign2Notes is a local laptop prototype that turns a short ASL video into a sequence of recognized glosses and then into readable Markdown notes.

## End-to-end pipeline

```text
Video upload
   ↓
MediaPipe Hands (keypoints only)
   ↓
Temporal CNN (PyTorch training)
   ↓
ONNX Runtime inference (preferred)
   ↓
Overlapping temporal windows
   ↓
Viterbi/HMM smoothing + confidence threshold
   ↓
Gloss sequence
   ↓
Ollama/Llama 3.2 3B OR deterministic fallback
   ↓
Markdown lecture notes
```

### Privacy behavior
The Node gateway keeps uploads in memory. FastAPI writes the raw video only to a temporary file, extracts keypoints, runs inference, and deletes both the video and extracted demo features in a `finally` block. WLASL training features are intentionally persisted because they are the model-training artifact; the original project workflow should keep raw videos out of the application runtime.

## Recommended MVP vocabulary
The project now trains on **FDMSE-ISL** by default (see below). Do not attempt
all 2,002 classes for a demo — `metadata_400.csv` (a curated 400-class subset,
20 samples/class) is a much more realistic starting vocabulary; `metadata_atomic.csv`
(single-gesture signs only, 1,099 classes) is another good option if you want
broader coverage of simple signs and can tolerate a longer extraction run.

## Setup

### Python

```powershell
python -m venv .venv
.venv\\Scripts\\activate
pip install -r ml_service/requirements.txt
```

### Node

```powershell
cd backend
npm install
cd ..\\frontend
npm install
```

## FDMSE-ISL data layout (default)

```text
data/
  data_meta/
    classes.txt              # full 2,002-class legend
    classes_400.txt          # legend for the 400-class subset
    metadata.csv             # all 2,002 classes, 40,034 clips, 20 signers
    metadata_400.csv         # curated 400-class subset (20 samples/class) — default
    metadata_atomic.csv      # single-gesture signs only (1,099 classes)
    metadata_composite.csv   # multi-word/compound signs only (352 classes)
  FDMSE-ISL/
    data/
      s0001/front/*.mp4
      s0002/front/*.mp4
      ...
```

Each metadata CSV has columns `id,video_dir,video_name,class,split` — `video_dir`
is relative to `data/FDMSE-ISL` (e.g. `data/s0015/front/s0015_f_w000842.mp4`),
`class` is the gloss label, and `split` is the dataset's own train/val/test
assignment (not currently used by `train.py`, which does its own random split
— see note below).

## 1. Extract training keypoints

```powershell
python ml_service/feature_extraction.py `
  --dataset_format fdmse `
  --metadata_csv data/data_meta/metadata_400.csv `
  --dataset_root data/FDMSE-ISL `
  --out_dir data/features `
  --frame_skip 8 `
  --workers 4
```

Add `--max_videos 500` for a quick first pass, or `--splits train,val` to skip
extracting the held-out test split until you're ready to evaluate.

<details>
<summary>Legacy WLASL layout (still supported)</summary>

```text
data/
  wlasl/
    WLASL_v0.3.json
    videos/
      12345.mp4
      ...
```

```powershell
python ml_service/feature_extraction.py `
  --dataset_format wlasl `
  --videos_dir data/wlasl/videos `
  --out_dir data/features `
  --wlasl_json data/wlasl/WLASL_v0.3.json `
  --frame_skip 8 `
  --max_videos 500 `
  --workers 4
```
</details>

## 2. Build a constrained vocabulary

```powershell
python ml_service/build_index.py `
  --dataset_format fdmse `
  --metadata_csv data/data_meta/metadata_400.csv `
  --min_samples 10
```

`--max_classes 0` (the default) keeps every class that clears `--min_samples`
— set it to a smaller number to further restrict the vocabulary. For the
legacy WLASL layout, pass `--dataset_format wlasl --wlasl_json ...` instead.

This creates:
- `data/index.csv` (now includes a `split` column when built from FDMSE-ISL)
- `config/vocab.json`

## 3. Train

Windows-safe defaults use `num_workers=0` to avoid multiprocessing spawn errors.

```powershell
python ml_service/train.py --epochs 20 --batch_size 32 --num_workers 0
```

> **Note:** `train.py` currently does its own random train/val split
> (`--val_split`, default 0.15) regardless of the `split` column FDMSE-ISL
> provides. This is fine for a quick MVP run, but for a rigorous evaluation
> matching the dataset's official split, that's a follow-up worth wiring in.

Outputs:

```text
models/sign_recog/checkpoints/best.pt
models/sign_recog/checkpoints/demo.pt
models/sign_recog/sign_recog.onnx
models/sign_recog/sign_recog.json
```

## 4. Install and run Ollama (optional)

Install Ollama, then pull the local model:

```powershell
ollama pull llama3.2:3b
```

Leave Ollama running. The web UI has a `Llama 3.2 via Ollama` option.

The deterministic notes mode works without Ollama.

## 5. Start the services

### Terminal 1 — ML service

```powershell
cd ml_service
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

### Terminal 2 — Node gateway

```powershell
cd backend
npm start
```

### Terminal 3 — React frontend

```powershell
cd frontend
npm run dev
```

Open `http://localhost:8080`.

## CLI inference

```powershell
python ml_service/infer.py --video_path path/to/video.mp4 --notes_mode template
```

For Ollama:

```powershell
python ml_service/infer.py --video_path path/to/video.mp4 --notes_mode ollama
```

## Why the pipeline uses windows + Viterbi

WLASL is word-level: a training clip normally corresponds to one gloss. It is not a sentence-level aligned dataset. To make a practical MVP capable of returning multiple glosses from a longer upload, inference breaks the extracted sequence into overlapping windows. Each window predicts a gloss distribution; Viterbi smoothing favors stable transitions and collapses repeats. This produces a gloss sequence without pretending that WLASL supplies frame-level sentence annotations.

## Current limitations

- Recognition quality depends strongly on the selected WLASL classes and camera viewpoint.
- The MVP is intended for a single signer and short classroom-style clips.
- The LLM expands recognized glosses; it should not be treated as a source of facts absent from the gloss input.
- Continuous sentence-level sign translation requires a sequence-aligned dataset and a stronger CTC/sequence model.

## Demo checklist

1. Extract a few hundred WLASL clips.
2. Select 5–10 classes with `build_index.py`.
3. Train until validation accuracy stabilizes.
4. Start FastAPI, Node, and React.
5. Upload a short test clip from one of the trained classes.
6. Confirm detected glosses, confidence, and generated notes.
7. Repeat with Ollama enabled.
8. Keep the final demo vocabulary narrow enough that errors are visible but manageable.
