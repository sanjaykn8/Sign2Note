import os
import tempfile
from collections import Counter
from pathlib import Path
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from feature_extraction import extract_single_video
from infer import predict_from_features, generate_notes

app = FastAPI(title="Sign2Notes ML Service", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).resolve().parents[1]
CHECKPOINT = BASE / "models/sign_recog/checkpoints/demo.pt"
ONNX = BASE / "models/sign_recog/sign_recog.onnx"
TMP = BASE / "data/tmp"
TMP.mkdir(parents=True, exist_ok=True)

LLAMA_CPP_URL = "http://127.0.0.1:8080"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": CHECKPOINT.exists(),
        "onnx_ready": ONNX.exists(),
        "llama_cpp_url": LLAMA_CPP_URL,
    }


def _best_guess_gloss(segments):
    """Fallback for when no window crosses the confidence threshold:
    majority-vote the raw per-window predictions (ignoring the threshold)
    and return the single most-predicted label. Always returns a
    non-empty list as long as at least one window was processed, so the
    API never has to hand back an empty result."""
    if not segments:
        return []
    counts = Counter(s["label"] for s in segments)
    top_label, _ = counts.most_common(1)[0]
    return [top_label]


@app.post("/process")
async def process_upload(
    file: UploadFile = File(...),
    notes_mode: str = Form("template"),
    llm_model: str = Form("gemma4"),
    style: str = Form("concise"),
    frame_skip: int = Form(8),
    stride: int = Form(12),
    threshold: float = Form(0.05),
):
    if not CHECKPOINT.exists() and not ONNX.exists():
        return JSONResponse({"error": "No trained model found. Run build_index.py and train.py first."}, 500)

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    video_path = None
    feature_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TMP) as tmp:
            video_path = Path(tmp.name)
            content = await file.read()
            tmp.write(content)

        feature_dir = TMP / "features"
        feature_dir.mkdir(parents=True, exist_ok=True)
        feature_path = extract_single_video(str(video_path), str(feature_dir), frame_skip=frame_skip)
        if feature_path is None:
            return JSONResponse({"error": "No usable hand keypoints were detected."}, 422)

        result = predict_from_features(
            str(feature_path),
            checkpoint_path=str(CHECKPOINT),
            onnx_path=str(ONNX),
            stride=stride,
            threshold=threshold,
        )

        low_confidence = False
        if not result["gloss_list"]:
            # No window crossed `threshold`. Instead of returning an empty
            # result, fall back to a best-guess label via majority vote
            # across all windows' raw predictions, so the pipeline always
            # commits to a sign rather than dead-ending on the user.
            top = sorted(result["segments"], key=lambda s: -s["confidence"])[:5]
            print(
                f"[api] No window crossed threshold={threshold} "
                f"(mean top confidence={result['top_confidence']:.3f}); "
                f"falling back to best guess. Top windows: "
                f"{[(s['label'], round(s['confidence'], 3)) for s in top]}"
            )
            result["gloss_list"] = _best_guess_gloss(result["segments"])
            low_confidence = True

        notes = generate_notes(
            result["gloss_list"], mode=notes_mode,
            llm_model=llm_model, style=style,
        )

        return {
            "notes_md": notes,
            "gloss_list": result["gloss_list"],
            "segments": result["segments"],
            "confidence": result["top_confidence"],
            "backend": result["backend"],
            "providers": result["providers"],
            "low_confidence": low_confidence,
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, 500)
    finally:
        # Privacy invariant: raw upload and extracted demo features are temporary only.
        for p in [video_path, feature_path]:
            try:
                if p and Path(p).exists():
                    Path(p).unlink()
            except Exception:
                pass