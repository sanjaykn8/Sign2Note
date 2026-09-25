import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import List, Optional

import cv2
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, field_validator

import feature_schema as fs
from checkpoint_meta import check_checkpoint_compatible
from feature_extraction import extract_single_video
from infer import (
    predict_from_features, predict_from_array, generate_notes, generate_transcript, get_model_meta,
    LLM_PROVIDER, LLM_MODEL, LLM_BASE_URL,
)
from notes_generator import SUPPORTED_STYLES

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="Sign2Notes ML Service", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).resolve().parents[1]
# Configurable via environment so a deployment doesn't need to hardcode a
# machine-specific path (project brief section 46) -- e.g. a distributed
# client running its own local recognition model would point these at its
# own checkpoint, separate from the main server's.
CHECKPOINT = Path(os.environ.get("MODEL_PATH", BASE / "models/sign_recog_v2/checkpoints/demo.pt"))
ONNX = Path(os.environ.get("ONNX_PATH", BASE / "models/sign_recog_v2/sign_recog.onnx"))
VOCAB_PATH = Path(os.environ.get("VOCAB_PATH", BASE / "config/vocab.json"))
TMP = BASE / "data/tmp"
TMP.mkdir(parents=True, exist_ok=True)

# Default confidence threshold for the /process endpoint. Overridable per
# request via the `threshold` form field; this just sets the form default.
DEFAULT_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.55))

# Request-size guards (section 39: "validate maximum gloss count", "reject
# excessively large requests"). Deliberately generous defaults -- these
# exist to stop pathological/abusive input, not to constrain normal use;
# a real lecture session might reasonably produce a few hundred glosses.
MAX_UPLOAD_SIZE_BYTES = int(os.environ.get("MAX_UPLOAD_SIZE", 200 * 1024 * 1024))  # 200 MB
MAX_GLOSS_COUNT = int(os.environ.get("MAX_GLOSS_COUNT", 2000))
MAX_RECOGNIZE_FRAMES = int(os.environ.get("MAX_RECOGNIZE_FRAMES", 20000))  # ~a few minutes of keypoints
VALID_NOTES_MODES = ("template", "llm")


def _current_vocab_label2id():
    """Best-effort read of the currently configured vocabulary, for the
    checkpoint-compatibility check (RULE 26) -- returns None (skips that
    part of the check) rather than raising if the file is missing or
    malformed, since a missing vocab.json shouldn't take down /health or
    /model/meta."""
    try:
        import json
        return json.loads(VOCAB_PATH.read_text()).get("label2id")
    except Exception:
        return None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": CHECKPOINT.exists(),
        "onnx_ready": ONNX.exists(),
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "llm_base_url": LLM_BASE_URL,
        "feature_schema_version": fs.FEATURE_SCHEMA_VERSION,
    }


@app.get("/model/meta")
def model_meta():
    """Model metadata (max_len, input_dim, label vocabulary, architecture,
    feature_schema_version, vocab_hash, best_val_accuracy, ...) for
    clients that run inference themselves -- specifically the browser
    webcam demo, which loads the ONNX model client-side via
    onnxruntime-web, and any distributed client checking whether its own
    local model is still compatible with what the server currently
    expects (RULE 25/26). `compatibility_warnings` is populated (but
    non-fatal here -- this endpoint still returns 200) whenever the
    checkpoint's recorded schema/vocab doesn't match config/vocab.json;
    callers that need a hard stop on incompatibility should check that
    list themselves, or call /recognize, which DOES fail clearly on a
    real incompatibility (see below)."""
    if not CHECKPOINT.exists():
        return JSONResponse({"error": "No trained checkpoint found."}, status_code=404)
    try:
        meta = get_model_meta(str(CHECKPOINT))
        ckpt_like = {
            "feature_schema_version": meta.get("feature_schema_version"),
            "input_dim": meta.get("input_dim"),
            "vocab_hash": meta.get("vocab_hash"),
        }
        meta["compatibility_warnings"] = check_checkpoint_compatible(
            ckpt_like, current_vocab_label2id=_current_vocab_label2id()
        )
        return meta
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
    if notes_mode not in VALID_NOTES_MODES:
        return JSONResponse(
            {"error": f"Invalid notes_mode {notes_mode!r}. Expected one of {VALID_NOTES_MODES}."},
            status_code=422,
        )
    if style not in SUPPORTED_STYLES:
        return JSONResponse(
            {"error": f"Invalid style {style!r}. Expected one of {SUPPORTED_STYLES}."},
            status_code=422,
        )
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
        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            return JSONResponse(
                {"error": f"Uploaded file ({len(content)} bytes) exceeds the "
                          f"{MAX_UPLOAD_SIZE_BYTES}-byte limit (MAX_UPLOAD_SIZE)."},
                status_code=413,
            )

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
    gloss_list: List[str] = Field(..., min_length=1, max_length=MAX_GLOSS_COUNT)
    notes_mode: str = "template"
    llm_model: Optional[str] = None
    style: str = "concise"

    @field_validator("notes_mode")
    @classmethod
    def _validate_notes_mode(cls, v):
        if v not in VALID_NOTES_MODES:
            raise ValueError(f"must be one of {VALID_NOTES_MODES}")
        return v

    @field_validator("style")
    @classmethod
    def _validate_style(cls, v):
        if v not in SUPPORTED_STYLES:
            raise ValueError(f"must be one of {SUPPORTED_STYLES}")
        return v


@app.post("/notes")
def notes_from_glosses(req: NotesRequest):
    """Generate notes directly from an already-recognized gloss sequence,
    with no video/keypoints involved. This is what the live webcam session
    calls on "Generate Notes": all keypoint extraction and ONNX inference
    for the webcam flow happens client-side in the browser (see
    LIVE_WEBCAM in ARCHITECTURE.md) so that raw video and keypoints never
    leave the machine -- only the final recognized gloss labels (plain
    words like "Market" or "Whistle", not video/imagery) are sent here.
    Validation (empty list, too many glosses, invalid mode/style) is
    handled by NotesRequest's field validators above -- FastAPI returns a
    422 automatically before this function body ever runs, per RULE
    (section 24): 'Reject: empty gloss arrays, malformed requests,
    excessively large requests, invalid modes.'"""
    notes = generate_notes(req.gloss_list, mode=req.notes_mode,
                           llm_model=req.llm_model, style=req.style)
    return {"notes_md": notes, "style": req.style, "notes_mode": req.notes_mode}


class TranscriptRequest(BaseModel):
    gloss_list: List[str] = Field(..., min_length=1, max_length=MAX_GLOSS_COUNT)
    notes_mode: str = "template"
    llm_model: Optional[str] = None

    @field_validator("notes_mode")
    @classmethod
    def _validate_notes_mode(cls, v):
        if v not in VALID_NOTES_MODES:
            raise ValueError(f"must be one of {VALID_NOTES_MODES}")
        return v


@app.post("/transcript")
def transcript_from_glosses(req: TranscriptRequest):
    """Generate a natural-language TRANSCRIPT from an already-recognized
    gloss sequence -- Live Transcription mode's "Stop & Generate" step.
    This is deliberately a separate endpoint from /notes, not a `style`
    value on it: a transcript (flowing prose, Layer 2) and structured
    notes (headed/bulleted, Layer 3) are different output layers built
    from different prompts (see notes_generator.py's module docstring),
    and Live Transcription mode wants to be able to show BOTH from the
    same gloss sequence (project brief section 21: "Provide both: A.
    recognized gloss transcript, B. generated natural-language
    transcript") -- calling /transcript here and /notes separately (with
    the same gloss_list) is exactly that, without conflating the two
    response shapes into one endpoint.

    No `style` field: a transcript's register doesn't vary the way notes
    do (concise/detailed/academic) -- it's always "flowing prose in gloss
    order." Called ONCE, only when the user stops the session -- see
    RULE 9: never call the live gloss stream itself an LLM-generated
    transcript until this endpoint has actually been called."""
    transcript = generate_transcript(req.gloss_list, mode=req.notes_mode, llm_model=req.llm_model)
    return {"transcript": transcript, "notes_mode": req.notes_mode}


class RecognizeRequest(BaseModel):
    """Recognition-only request: pre-extracted keypoints, no video. This is
    the API a distributed client with local landmark extraction but no
    local recognition MODEL uses to fall back to the main server (RULE 8 /
    section 23 tier 2: 'keypoints -> network -> main server model ->
    glosses'). Deliberately does NOT accept or return LLM notes (section
    43: recognition and note generation are separate concerns -- use
    /notes afterward if notes are wanted)."""
    features: List[List[float]] = Field(
        ..., description="(num_frames, FEATURE_DIM) canonical hand+body+face keypoints, "
                          "see FEATURE_SCHEMA.md."
    )
    fps: Optional[float] = Field(None, description="Original capture fps, for real timestamps in `events`.")
    frame_skip: int = 8
    stride: int = 12
    threshold: float = Field(default=None)

    @field_validator("features")
    @classmethod
    def _validate_features(cls, v):
        if len(v) == 0:
            raise ValueError("features is empty -- nothing to recognize")
        if len(v) > MAX_RECOGNIZE_FRAMES:
            raise ValueError(f"features has {len(v)} frames, exceeds the {MAX_RECOGNIZE_FRAMES}-frame limit "
                             f"(MAX_RECOGNIZE_FRAMES)")
        row_len = len(v[0])
        if any(len(row) != row_len for row in v):
            raise ValueError("features rows have inconsistent lengths -- every frame must have the same dimension")
        return v


@app.post("/recognize")
def recognize_from_features(req: RecognizeRequest):
    """Recognition-only endpoint for clients that extract keypoints
    locally but don't have (or can't run) the recognition model itself --
    e.g. a lightweight/older device, or before the local ONNX model has
    finished downloading. Returns glosses/events/confidence, nothing else;
    call /notes separately with the returned glosses if notes are wanted
    (RULE: 'the LLM must never be responsible for sign recognition').

    Checks the request's feature dimension against the CURRENT schema
    before attempting inference, so a client running an old (schema v1,
    126-dim) local extractor gets a clear, actionable error instead of a
    confusing shape-mismatch crash deep in the model (RULE 26)."""
    if not CHECKPOINT.exists() and not ONNX.exists():
        return JSONResponse({"error": "No trained model found on the server."}, status_code=503)

    actual_dim = len(req.features[0])
    if actual_dim != fs.FEATURE_DIM:
        return JSONResponse(
            {
                "error": (
                    f"Feature dimension mismatch: got {actual_dim}, server expects "
                    f"{fs.FEATURE_DIM} (feature_schema_version={fs.FEATURE_SCHEMA_VERSION}). "
                    f"Local recognition model and feature schema are incompatible -- "
                    f"re-extract keypoints with the current schema, or fall back to /process "
                    f"(sending video instead) if that isn't possible client-side."
                ),
                "expected_feature_dim": fs.FEATURE_DIM,
                "got_feature_dim": actual_dim,
            },
            status_code=422,
        )

    try:
        threshold = req.threshold if req.threshold is not None else DEFAULT_THRESHOLD
        result = predict_from_array(
            req.features, checkpoint_path=str(CHECKPOINT), onnx_path=str(ONNX),
            stride=req.stride, threshold=threshold, fps=req.fps, frame_skip=req.frame_skip,
        )
        return {
            "glosses": result["gloss_list"],
            "events": result["events"],
            "segments": result["segments"],
            "confidence": result["top_confidence"],
            "backend": result["backend"],
            "model_version": str(CHECKPOINT),
            "feature_schema_version": fs.FEATURE_SCHEMA_VERSION,
        }
    except fs.FeatureSchemaError as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
