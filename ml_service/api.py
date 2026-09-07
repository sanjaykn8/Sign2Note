import os
import tempfile
from collections import Counter
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from feature_extraction import extract_single_video
from infer import predict_from_features, generate_notes, get_model_meta, LLM_PROVIDER, LLM_MODEL, LLM_BASE_URL

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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

# Default confidence threshold for the /process endpoint. Overridable per
# request via the `threshold` form field; this just sets the form default.
DEFAULT_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.55))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": CHECKPOINT.exists(),
        "onnx_ready": ONNX.exists(),
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "llm_base_url": LLM_BASE_URL,
    }


@app.get("/model/meta")
def model_meta():
    """Model metadata (max_len, input_dim, label vocabulary) for clients
    that run inference themselves -- specifically the browser webcam demo,
    which loads the ONNX model client-side via onnxruntime-web and needs
    this to build correctly-shaped, correctly-labeled predictions."""
    if not CHECKPOINT.exists():
        return JSONResponse({"error": "No trained checkpoint found."}, status_code=404)
    try:
        return get_model_meta(str(CHECKPOINT))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/model/onnx")
def model_onnx():
    """Serves the exported ONNX model file directly so the browser can
    fetch it and run it locally with onnxruntime-web -- no video, keypoints,
    or predictions ever need to leave the browser for the webcam flow."""
    if not ONNX.exists():
        return JSONResponse(
            {"error": "No ONNX model found. Run train.py to export one."},
            status_code=404,
        )
    return FileResponse(str(ONNX), media_type="application/octet-stream", filename="sign_recog.onnx")


def _best_guess_gloss(segments):
    """Fallback for when no window crosses the confidence threshold:
    majority-vote the raw per-window predictions (ignoring the threshold)
    and return the single most-predicted label. Always returns a
    non-empty list as long as at least one window was processed, so
    /process never has to hand back an empty result for an uploaded video.

    NOTE: this fallback is deliberately specific to /process (the "upload a
    video, always get notes" flow). The live webcam session (see /notes and
    the frontend's session logic) does the opposite on purpose: low-
    confidence predictions are surfaced as "please repeat" and are NOT
    added to the session history. Two different UX goals for two different
    interaction modes -- see ARCHITECTURE.md."""
    if not segments:
        return []
    counts = Counter(s["label"] for s in segments)
    top_label, _ = counts.most_common(1)[0]
    return [top_label]


@app.post("/process")
async def process_upload(
    file: UploadFile = File(...),
    notes_mode: str = Form("template"),
    llm_model: str = Form(None),
    style: str = Form("concise"),
    frame_skip: int = Form(8),
    stride: int = Form(12),
    threshold: float = Form(DEFAULT_THRESHOLD),
):
    """Long-video-capable inference endpoint. Handles clips of any length:
    keypoints are extracted frame-by-frame (never loading the whole video
    into memory), sliding windows are run through the model in bounded-size
    chunks (see infer.INFER_CHUNK_SIZE), and Viterbi smoothing collapses
    the per-window predictions into an ordered, timestamped `events` list
    alongside the flat `gloss_list` for backward compatibility."""
    if not CHECKPOINT.exists() and not ONNX.exists():
        return JSONResponse({"error": "No trained model found. Run build_index.py and train.py first."}, status_code=500)

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    video_path = None
    feature_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TMP) as tmp:
            video_path = Path(tmp.name)
            content = await file.read()
            tmp.write(content)

        if len(content) == 0:
            return JSONResponse({"error": "Uploaded file is empty."}, status_code=422)

        # Probe fps before extraction so window timestamps can be computed
        # against the *original* video's timeline, not the subsampled one.
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        if not fps or fps <= 0:
            fps = 25.0

        feature_dir = TMP / "features"
        feature_dir.mkdir(parents=True, exist_ok=True)
        feature_path = extract_single_video(str(video_path), str(feature_dir), frame_skip=frame_skip)
        if feature_path is None:
            return JSONResponse({"error": "No usable hand keypoints were detected. Make sure hands are visible and well-lit."}, status_code=422)

        result = predict_from_features(
            str(feature_path),
            checkpoint_path=str(CHECKPOINT),
            onnx_path=str(ONNX),
            stride=stride,
            threshold=threshold,
            fps=fps,
            frame_skip=frame_skip,
        )

        low_confidence = False
        if not result["gloss_list"]:
            top = sorted(result["segments"], key=lambda s: -s["confidence"])[:5]
            print(
                f"[api] No window crossed threshold={threshold} "
                f"(mean top confidence={result['top_confidence']:.3f}); "
                f"falling back to best guess. Top windows: "
                f"{[(s['label'], round(s['confidence'], 3)) for s in top]}"
            )
            result["gloss_list"] = _best_guess_gloss(result["segments"])
            low_confidence = True

        # Prefer the timestamped events' labels for note generation when
        # available -- they're deduplicated/collapsed the same way as
        # gloss_list but retain the (start_time, end_time) span, matching
        # the requirement that longer videos yield an *ordered sequence* of
        # signs rather than one flat prediction.
        ordered_glosses = [e["label"] for e in result["events"]] if result["events"] else result["gloss_list"]

        notes = generate_notes(
            ordered_glosses, mode=notes_mode,
            llm_model=llm_model, style=style,
        )

        return {
            "notes_md": notes,
            "gloss_list": result["gloss_list"],
            "events": result["events"],
            "segments": result["segments"],
            "confidence": result["top_confidence"],
            "backend": result["backend"],
            "providers": result["providers"],
            "low_confidence": low_confidence,
            "video_fps": fps,
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        # Privacy invariant: raw upload and extracted demo features are temporary only.
        for p in [video_path, feature_path]:
            try:
                if p and Path(p).exists():
                    Path(p).unlink()
            except Exception:
                pass


class NotesRequest(BaseModel):
    gloss_list: list[str]
    notes_mode: str = "template"
    llm_model: str | None = None
    style: str = "concise"


@app.post("/notes")
def notes_from_glosses(req: NotesRequest):
    """Generate notes directly from an already-recognized gloss sequence,
    with no video/keypoints involved. This is what the live webcam session
    calls on "Generate Notes": all keypoint extraction and ONNX inference
    for the webcam flow happens client-side in the browser (see
    LIVE_WEBCAM in ARCHITECTURE.md) so that raw video and keypoints never
    leave the machine -- only the final recognized gloss labels (plain
    words like "Market" or "Whistle", not video/imagery) are sent here."""
    if not req.gloss_list:
        return JSONResponse({"error": "gloss_list is empty."}, status_code=422)
    notes = generate_notes(req.gloss_list, mode=req.notes_mode,
                           llm_model=req.llm_model, style=req.style)
    return {"notes_md": notes}
