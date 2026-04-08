"""
api.py  –  FastAPI service: receives a video upload → returns markdown notes.

Run with:
    cd ml_service
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# import project modules directly (no subprocess – faster and more reliable)
from feature_extraction import extract_single_video
from infer              import predict_from_features, generate_notes

app = FastAPI(title="Sign2Notes ML Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR  = Path("../backend/uploads")
FEATURE_DIR = Path("../data/features")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process")
async def process_upload(
    file:      UploadFile = File(...),
    use_llama: bool       = Form(False),
    use_ollama: bool      = Form(False),
):
    # ── 1. save uploaded file ─────────────────────────────────────────────────
    suffix   = Path(file.filename).suffix or ".mp4"
    stem     = uuid.uuid4().hex
    filename = f"{stem}{suffix}"
    save_path = UPLOAD_DIR / filename

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # ── 2. extract MediaPipe keypoints ────────────────────────────────────────
    feature_path = extract_single_video(str(save_path), str(FEATURE_DIR))
    if feature_path is None:
        return JSONResponse(
            {"error": "Feature extraction failed – no hand keypoints detected in video."},
            status_code=422,
        )

    # ── 3. sign recognition ───────────────────────────────────────────────────
    checkpoint = Path("models/sign_recog/checkpoints/demo.pt")
    if not checkpoint.exists():
        return JSONResponse(
            {"error": f"Model checkpoint not found at {checkpoint}. Run train.py first."},
            status_code=500,
        )

    gloss_list = predict_from_features(str(feature_path), str(checkpoint))
    if not gloss_list:
        return JSONResponse(
            {"error": "Model returned no predictions."},
            status_code=500,
        )

    # ── 4. generate notes ─────────────────────────────────────────────────────
    gguf_path = Path("../models/llama/llama-3.2-3b-instruct.gguf")
    notes_md  = generate_notes(
        gloss_list,
        use_ollama = use_ollama,
        use_llama  = use_llama and gguf_path.exists(),
        gguf_path  = str(gguf_path) if gguf_path.exists() else None,
    )

    return JSONResponse({
        "notes_md":   notes_md,
        "gloss_list": gloss_list,
    })
