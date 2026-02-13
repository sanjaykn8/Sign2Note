# ml_service/api.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import subprocess
import os
import uuid

app = FastAPI()
UPLOAD_DIR = Path("../backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/process")
async def process_upload(file: UploadFile = File(...), use_llama: bool = Form(False)):
    # Save uploaded file
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = UPLOAD_DIR / filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # call feature extraction on saved file
    feat_out_dir = Path("../data/features")
    feat_out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "ml_service/feature_extraction.py",
        "--videos_dir", str(save_path.parent),
        "--out_dir", str(feat_out_dir),
        "--ext", Path(save_path).suffix
    ]
    # We'll process just this file: feature_extraction expects directory; it's okay.
    try:
        print("Running feature extraction...")
        subprocess.check_call(cmd)
    except Exception as e:
        return JSONResponse({"error": f"feature extraction failed: {e}"}, status_code=500)

    # locate the saved npy
    stem = Path(save_path).stem
    feature_path = feat_out_dir / f"{stem}.npy"
    if not feature_path.exists():
        return JSONResponse({"error": "feature npy not found after extraction"}, status_code=500)

    # run inference -> tokens + notes
    out_tokens = Path("../out/tokens") / f"{stem}.json"
    out_notes = Path("../out/notes") / f"{stem}.md"
    out_tokens.parent.mkdir(parents=True, exist_ok=True)
    out_notes.parent.mkdir(parents=True, exist_ok=True)

    onnx_model = Path("../models/sign_recog/sign_recog.onnx")
    if not onnx_model.exists():
        return JSONResponse({"error": "onnx model not found at models/sign_recog/sign_recog.onnx"}, status_code=500)

    infer_cmd = [
        "python", "ml_service/infer.py",
        "--feature_path", str(feature_path),
        "--onnx_model", str(onnx_model),
        "--out_tokens", str(out_tokens),
        "--out_notes", str(out_notes)
    ]
    if use_llama:
        infer_cmd += ["--use_llama", "--gguf_path", str(Path("../models/llama/llama-3.2-3b-instruct.gguf"))]

    try:
        subprocess.check_call(infer_cmd)
    except Exception as e:
        return JSONResponse({"error": f"infer failed: {e}"}, status_code=500)

    notes_text = out_notes.read_text(encoding="utf-8")
    return JSONResponse({"notes_md": notes_text})

@app.get("/health")
def health():
    return {"status": "ok"}
