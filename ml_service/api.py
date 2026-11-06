"""Sign2Notes ML service (FastAPI).

Two separate services live behind this one process (they share a process
only for deployment convenience on the main laptop -- their APIs, inputs
and outputs are independent):

  RECOGNITION  (sign -> glosses)
    POST /recognize          keypoints (canonical schema of the served model) -> glosses/events
    POST /recognize/video    video file -> glosses/events (server-side extraction)
    GET  /model/meta         served model's metadata + compatibility status
    GET  /model/onnx         the served model's ONNX export (for in-browser recognition)

  LANGUAGE GENERATION  (glosses -> text)
    POST /notes              glosses -> structured notes (concise/detailed/academic)
    POST /transcript         glosses -> natural-language transcript

  COMBINED (kept for backward compatibility with the upload page)
    POST /process            video -> recognition -> notes, in one call

  GET /health                liveness + model/LLM availability (no secrets)

Model selection: MODEL_PATH / ONNX_PATH env vars, else the newest available
of models/sign_recog_v2 (schema v2, hand+body+face) and models/sign_recog
(legacy schema v1, hand-only). Whatever is served, uploads are converted
to the SAME feature schema that model was trained on (never mixed).
"""
import logging
import os
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from typing import List, Optional

import cv2
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, field_validator

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import feature_schema as fs
from checkpoint_meta import check_checkpoint_compatible, check_checkpoint_self_consistent
from feature_extraction import extract_single_video_for_schema
from infer import (
    predict_from_features, predict_from_array, generate_notes_with_meta,
    generate_transcript_with_meta, get_model_meta,
    LLM_PROVIDER, LLM_MODEL, LLM_BASE_URL,
)
from notes_generator import SUPPORTED_STYLES, MAX_CONTEXT_CHARS

log = logging.getLogger("sign2notes.ml")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Sign2Notes ML Service", version="2.1")

# The browser never talks to this service directly (it goes through the
# Express gateway), so CORS here only needs the local dev origins.
_cors = os.environ.get("ML_CORS_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080,"
                                          "http://localhost:3001,http://127.0.0.1:3001")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

BASE = Path(__file__).resolve().parents[1]


def _resolve_default_checkpoint() -> Path:
    """Newest schema first: v2 best -> v2 demo -> legacy v1 best -> legacy v1
    demo. best.pt is preferred over demo.pt because train.py exports the
    ONNX file from best.pt's weights (the label mapping is identical, but
    using the same weights for the torch fallback keeps the two in sync)."""
    for rel in ("models/sign_recog_v2/checkpoints/best.pt",
                "models/sign_recog_v2/checkpoints/demo.pt",
                "models/sign_recog/checkpoints/best.pt",
                "models/sign_recog/checkpoints/demo.pt"):
        if (BASE / rel).exists():
            return BASE / rel
    return BASE / "models/sign_recog_v2/checkpoints/best.pt"


CHECKPOINT = Path(os.environ.get("MODEL_PATH") or _resolve_default_checkpoint())
ONNX = Path(os.environ.get("ONNX_PATH") or (CHECKPOINT.parent.parent / "sign_recog.onnx"))
VOCAB_PATH = Path(os.environ.get("VOCAB_PATH", BASE / "config/vocab.json"))
TMP = BASE / "data/tmp"
TMP.mkdir(parents=True, exist_ok=True)

DEFAULT_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.55))

# Temporal sampling. The recognition models are trained on FDMSE-ISL clips
# (60 fps) sampled every 8th frame = 7.5 samples/second. A fixed frame_skip
# of 8 applied to a 30 fps upload samples at 3.75 Hz, so every sign looks
# twice as fast as anything seen in training. Unless the caller passes an
# explicit frame_skip, /process and /recognize/video derive it from the
# upload's fps so the sampling RATE matches training.
TRAINING_SAMPLE_HZ = float(os.environ.get("TRAINING_SAMPLE_HZ", 7.5))
# Sliding-window length (in sampled frames) for continuous video. Training
# samples are isolated clips of ~20-35 sampled frames zero-padded to
# max_len (64); a 64-real-frame window (8.5 s) would contain several signs.
# Windows of WINDOW_FRAMES real frames, zero-padded to max_len exactly like
# training, match the training distribution. See ARCHITECTURE.md.
DEFAULT_WINDOW_FRAMES = int(os.environ.get("WINDOW_FRAMES", 24))
DEFAULT_STRIDE = int(os.environ.get("WINDOW_STRIDE", 4))

MAX_UPLOAD_SIZE_BYTES = int(os.environ.get("MAX_UPLOAD_SIZE", 200 * 1024 * 1024))  # 200 MB
MAX_GLOSS_COUNT = int(os.environ.get("MAX_GLOSS_COUNT", 2000))
MAX_RECOGNIZE_FRAMES = int(os.environ.get("MAX_RECOGNIZE_FRAMES", 20000))
MAX_VIDEO_SECONDS = float(os.environ.get("MAX_VIDEO_SECONDS", 30 * 60))
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
VALID_NOTES_MODES = ("template", "llm")


def _current_vocab_label2id():
    """Best-effort read of config/vocab.json for the vocab-drift warning;
    returns None rather than raising (a missing vocab.json must not take
    down /health or /model/meta)."""
    try:
        import json
        return json.loads(VOCAB_PATH.read_text(encoding="utf-8")).get("label2id")
    except Exception:
        return None


def model_status() -> dict:
    """Whether the configured model can actually be served -- based on
    loading and checking its metadata, not just on a file existing.
    {ready, reason, meta, backend}."""
    if not CHECKPOINT.exists():
        return {"ready": False, "reason": f"No checkpoint at {CHECKPOINT}. Train one (see SETUP.md).",
                "meta": None}
    try:
        meta = get_model_meta(str(CHECKPOINT))
    except Exception as exc:
        log.exception("could not read checkpoint %s", CHECKPOINT)
        return {"ready": False, "reason": f"Checkpoint could not be read: {exc}", "meta": None}
    problems = check_checkpoint_self_consistent({
        "feature_schema_version": meta["feature_schema_version"], "input_dim": meta["input_dim"],
        "label2id": meta["label2id"], "max_len": meta["max_len"], "architecture": meta["architecture"],
    })
    if problems:
        return {"ready": False, "reason": "; ".join(problems), "meta": meta}
    return {"ready": True, "reason": None, "meta": meta}


def _err(status: int, message: str, **extra):
    return JSONResponse({"error": message, **extra}, status_code=status)


def _internal_error(where: str, exc: Exception):
    """Log the full traceback server-side; return an understandable message."""
    log.error("%s failed: %s\n%s", where, exc, traceback.format_exc())
    return _err(500, f"{where} failed: {exc}")


@app.get("/health")
def health():
    st = model_status()
    meta = st["meta"] or {}
    return {
        "status": "ok",
        "model_ready": st["ready"],
        "model_problem": st["reason"],
        "model_version": meta.get("model_version"),
        "feature_schema_version": meta.get("feature_schema_version"),
        "is_legacy_schema": meta.get("is_legacy_schema"),
        "onnx_ready": ONNX.exists(),
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "current_feature_schema_version": fs.FEATURE_SCHEMA_VERSION,
    }


@app.get("/model/meta")
def model_meta():
    """Metadata of the SERVED model (max_len, input_dim, label vocabulary,
    architecture, feature_schema_version, vocab_hash, metrics, sampling
    rate) so a client can decide whether its local recognizer is
    compatible before using it. `compatibility_warnings` lists drift
    between the checkpoint and config/vocab.json (informational)."""
    st = model_status()
    if st["meta"] is None:
        return _err(404 if not CHECKPOINT.exists() else 500, st["reason"])
    meta = dict(st["meta"])
    meta["compatibility_warnings"] = [
        w for w in check_checkpoint_compatible(
            {"feature_schema_version": meta["feature_schema_version"],
             "input_dim": meta["input_dim"], "vocab_hash": meta["vocab_hash"]},
            current_vocab_label2id=_current_vocab_label2id(),
        ) if "vocab_hash" in w
    ] + ([st["reason"]] if st["reason"] else [])
    meta["ready"] = st["ready"]
    meta["onnx_available"] = ONNX.exists()
    meta["sample_hz"] = TRAINING_SAMPLE_HZ
    meta["window_frames"] = DEFAULT_WINDOW_FRAMES
    meta["window_stride"] = DEFAULT_STRIDE
    return meta


@app.get("/model/onnx")
def model_onnx():
    """Serves the served model's ONNX export so the browser can run
    recognition locally with onnxruntime-web (only keypoints-derived
    predictions stay in the browser; no video leaves it)."""
    if not ONNX.exists():
        return _err(404, "No ONNX model found. Run train.py to export one.")
    return FileResponse(str(ONNX), media_type="application/octet-stream", filename="sign_recog.onnx")


def _best_guess_gloss(segments):
    """Fallback for /process when no window crosses the confidence
    threshold: majority-vote the raw per-window predictions and return the
    single most-predicted label, flagged low_confidence in the response.
    Deliberately specific to the upload flow (live sessions instead show
    "uncertain" and add nothing)."""
    if not segments:
        return []
    counts = Counter(s["label"] for s in segments)
    top_label, _ = counts.most_common(1)[0]
    return [top_label]


async def _save_upload(file: UploadFile) -> Path:
    """Stream the upload to a temp file in 1 MB chunks, enforcing the size
    limit WHILE streaming (the previous version read the whole body into
    memory first, then checked the size)."""
    suffix = Path(file.filename or "upload.mp4").suffix.lower() or ".mp4"
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported file type {suffix!r}. Allowed: {sorted(ALLOWED_VIDEO_SUFFIXES)}")
    size = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TMP) as tmp:
        path = Path(tmp.name)
        try:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE_BYTES:
                    raise OverflowError(
                        f"Uploaded file exceeds the {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB limit "
                        f"(MAX_UPLOAD_SIZE)."
                    )
                tmp.write(chunk)
        except BaseException:
            tmp.close()
            path.unlink(missing_ok=True)
            raise
    if size == 0:
        path.unlink(missing_ok=True)
        raise ValueError("Uploaded file is empty.")
    return path


def _probe_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None, None
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    finally:
        cap.release()
    if not fps or fps <= 0 or fps > 1000:
        fps = 25.0
    return fps, (frames / fps if frames > 0 else None)


def _auto_frame_skip(fps: float) -> int:
    return max(1, int(round(fps / TRAINING_SAMPLE_HZ)))


def _recognize_video_sync(video_path: Path, frame_skip: Optional[int], stride: int,
                          threshold: float, window_frames: int) -> dict:
    """Shared by /process and /recognize/video: probe -> extract features in
    the served model's schema -> sliding-window recognition. Returns the
    recognition result dict, or raises ValueError with a user-facing
    message."""
    st = model_status()
    if not st["ready"]:
        raise RuntimeError(f"Recognition model unavailable: {st['reason']}")
    schema = st["meta"]["feature_schema_version"]

    fps, duration = _probe_video(video_path)
    if fps is None:
        raise ValueError("The file could not be opened as a video (corrupt or unsupported codec).")
    if duration and duration > MAX_VIDEO_SECONDS:
        raise ValueError(f"Video is {duration / 60:.1f} min long; the limit is "
                         f"{MAX_VIDEO_SECONDS / 60:.0f} min (MAX_VIDEO_SECONDS).")
    skip = frame_skip if frame_skip and frame_skip > 0 else _auto_frame_skip(fps)

    feature_dir = TMP / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_path = None
    try:
        feature_path, msg = extract_single_video_for_schema(
            str(video_path), str(feature_dir), schema, frame_skip=skip)
        if feature_path is None:
            raise ValueError(f"No frames could be processed from this video ({msg}).")
        result = predict_from_features(
            str(feature_path), checkpoint_path=str(CHECKPOINT), onnx_path=str(ONNX),
            stride=stride, threshold=threshold, fps=fps, frame_skip=skip,
            window_frames=window_frames,
        )
    finally:
        if feature_path and Path(feature_path).exists():
            Path(feature_path).unlink(missing_ok=True)

    result["video_fps"] = fps
    result["video_duration_seconds"] = duration
    result["frame_skip"] = skip
    result["window_frames"] = window_frames
    result["stride"] = stride
    result["threshold"] = threshold
    result["model_version"] = st["meta"].get("model_version")
    result["feature_schema_version"] = schema
    result["low_confidence"] = False
    if not result["gloss_list"]:
        top = sorted(result["segments"], key=lambda s: -s["confidence"])[:5]
        log.info("no window crossed threshold=%s (mean top conf=%.3f); best guess. top=%s",
                 threshold, result["top_confidence"],
                 [(s["label"], round(s["confidence"], 3)) for s in top])
        result["gloss_list"] = _best_guess_gloss(result["segments"])
        result["low_confidence"] = True
    return result


async def _run_video_recognition(file, frame_skip, stride, threshold, window_frames):
    video_path = None
    try:
        try:
            video_path = await _save_upload(file)
        except OverflowError as exc:
            return None, _err(413, str(exc))
        except ValueError as exc:
            return None, _err(422, str(exc))
        try:
            # Heavy CPU work runs in a worker thread: doing it inline in an
            # async endpoint blocked the event loop, so /health and every
            # other request hung for the whole duration of an upload.
            result = await run_in_threadpool(
                _recognize_video_sync, video_path, frame_skip, stride, threshold, window_frames)
        except ValueError as exc:
            return None, _err(422, str(exc))
        except RuntimeError as exc:
            if "unavailable" in str(exc):
                return None, _err(503, str(exc))
            return None, _internal_error("Recognition", exc)
        except Exception as exc:
            return None, _internal_error("Recognition", exc)
        return result, None
    finally:
        # Privacy invariant: the raw upload is temporary only.
        if video_path is not None:
            Path(video_path).unlink(missing_ok=True)


def _recognition_payload(result: dict) -> dict:
    return {
        "glosses": result["gloss_list"],
        "gloss_list": result["gloss_list"],
        "events": result["events"],
        "segments": result["segments"],
        "confidence": result["top_confidence"],
        "low_confidence": result.get("low_confidence", False),
        "backend": result["backend"],
        "providers": result.get("providers", []),
        "model_version": result.get("model_version"),
        "feature_schema_version": result.get("feature_schema_version"),
        "video_fps": result.get("video_fps"),
        "video_duration_seconds": result.get("video_duration_seconds"),
        "frame_skip": result.get("frame_skip"),
        "window_frames": result.get("window_frames"),
    }


@app.post("/recognize/video")
async def recognize_video(
    file: UploadFile = File(...),
    frame_skip: Optional[int] = Form(None),
    stride: int = Form(DEFAULT_STRIDE),
    threshold: float = Form(DEFAULT_THRESHOLD),
    window_frames: int = Form(DEFAULT_WINDOW_FRAMES),
):
    """Recognition only (video in, ordered timestamped glosses out). For
    clients that cannot extract landmarks themselves. No notes."""
    if not (0.0 <= threshold <= 1.0) or stride < 1 or window_frames < 4:
        return _err(422, "threshold must be in [0,1], stride >= 1, window_frames >= 4")
    result, error = await _run_video_recognition(file, frame_skip, stride, threshold, window_frames)
    if error is not None:
        return error
    return _recognition_payload(result)


@app.post("/process")
async def process_upload(
    file: UploadFile = File(...),
    notes_mode: str = Form("template"),
    llm_model: str = Form(None),
    style: str = Form("concise"),
    frame_skip: Optional[int] = Form(None),
    stride: int = Form(DEFAULT_STRIDE),
    threshold: float = Form(DEFAULT_THRESHOLD),
    window_frames: int = Form(DEFAULT_WINDOW_FRAMES),
):
    """Upload page's one-call flow: recognition (as /recognize/video) and
    then notes (as /notes) on the ordered gloss sequence. Long videos are
    handled frame-by-frame and in bounded inference chunks; Viterbi
    smoothing turns per-window predictions into ordered, timestamped
    `events`."""
    if notes_mode not in VALID_NOTES_MODES:
        return _err(422, f"Invalid notes_mode {notes_mode!r}. Expected one of {VALID_NOTES_MODES}.")
    if style not in SUPPORTED_STYLES:
        return _err(422, f"Invalid style {style!r}. Expected one of {SUPPORTED_STYLES}.")
    if not (0.0 <= threshold <= 1.0) or stride < 1 or window_frames < 4:
        return _err(422, "threshold must be in [0,1], stride >= 1, window_frames >= 4")

    result, error = await _run_video_recognition(file, frame_skip, stride, threshold, window_frames)
    if error is not None:
        return error

    # Notes are generated from the timestamped events (ordered, duplicate-
    # collapsed) when available, with their confidences.
    if result["events"]:
        ordered = [e["label"] for e in result["events"]]
        confidences = [e["confidence"] for e in result["events"]]
    else:
        ordered = result["gloss_list"]
        confidences = None
    try:
        notes, gen_meta = await run_in_threadpool(
            generate_notes_with_meta, ordered, notes_mode, llm_model, style, confidences, None)
    except Exception as exc:
        return _internal_error("Notes generation", exc)

    payload = _recognition_payload(result)
    payload.update({"notes_md": notes, "generation": gen_meta, "style": style})
    return payload


def _validate_glosses(v):
    cleaned = [str(g).strip() for g in v]
    if any(not g for g in cleaned):
        raise ValueError("gloss_list contains an empty gloss")
    if any(len(g) > 100 for g in cleaned):
        raise ValueError("gloss_list contains a gloss longer than 100 characters")
    return cleaned


def _validate_confidences(v):
    if v is None:
        return v
    if any((c is not None) and not (0.0 <= float(c) <= 1.0) for c in v):
        raise ValueError("confidences must be within [0, 1]")
    return v


class NotesRequest(BaseModel):
    gloss_list: List[str] = Field(..., min_length=1, max_length=MAX_GLOSS_COUNT)
    notes_mode: str = "template"
    llm_model: Optional[str] = None
    style: str = "concise"
    # Optional per-gloss recognition confidences (same length as
    # gloss_list): low-confidence glosses are marked uncertain in the
    # prompt/fallback instead of being stated as fact.
    confidences: Optional[List[Optional[float]]] = None
    # Optional user-typed context (e.g. a lecture title). Never treated as
    # a source of facts -- see notes_generator._context_block.
    context: Optional[str] = Field(None, max_length=MAX_CONTEXT_CHARS)

    @field_validator("gloss_list")
    @classmethod
    def _check_glosses(cls, v):
        return _validate_glosses(v)

    @field_validator("confidences")
    @classmethod
    def _check_confidences(cls, v):
        return _validate_confidences(v)

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


def _aligned(confidences, glosses):
    return confidences if confidences is not None and len(confidences) == len(glosses) else None


@app.post("/notes")
def notes_from_glosses(req: NotesRequest):
    """Language generation only: ordered glosses -> structured notes. No
    video or keypoints involved (the live modes send only the recognized
    gloss words here). Validation errors return 422 automatically."""
    try:
        notes, meta = generate_notes_with_meta(
            req.gloss_list, mode=req.notes_mode, llm_model=req.llm_model, style=req.style,
            confidences=_aligned(req.confidences, req.gloss_list), context=req.context,
        )
    except Exception as exc:
        return _internal_error("Notes generation", exc)
    return {"notes_md": notes, "style": req.style, "notes_mode": req.notes_mode, "generation": meta}


class TranscriptRequest(BaseModel):
    gloss_list: List[str] = Field(..., min_length=1, max_length=MAX_GLOSS_COUNT)
    notes_mode: str = "template"
    llm_model: Optional[str] = None
    confidences: Optional[List[Optional[float]]] = None

    @field_validator("gloss_list")
    @classmethod
    def _check_glosses(cls, v):
        return _validate_glosses(v)

    @field_validator("confidences")
    @classmethod
    def _check_confidences(cls, v):
        return _validate_confidences(v)

    @field_validator("notes_mode")
    @classmethod
    def _validate_notes_mode(cls, v):
        if v not in VALID_NOTES_MODES:
            raise ValueError(f"must be one of {VALID_NOTES_MODES}")
        return v


@app.post("/transcript")
def transcript_from_glosses(req: TranscriptRequest):
    """Language generation only: ordered glosses -> natural-language
    TRANSCRIPT (flowing prose). Separate from /notes (structured notes);
    Live Transcription calls this once, when the user stops the session."""
    try:
        transcript, meta = generate_transcript_with_meta(
            req.gloss_list, mode=req.notes_mode, llm_model=req.llm_model,
            confidences=_aligned(req.confidences, req.gloss_list),
        )
    except Exception as exc:
        return _internal_error("Transcript generation", exc)
    return {"transcript": transcript, "notes_mode": req.notes_mode, "generation": meta}


class RecognizeRequest(BaseModel):
    """Recognition-only request: pre-extracted keypoints in the SERVED
    model's feature schema (see GET /model/meta -> feature_schema_version /
    input_dim). Used by clients that extract landmarks locally but have no
    compatible local recognition model. Returns glosses, never notes."""
    features: List[List[float]] = Field(
        ..., description="(num_frames, input_dim) keypoints in the served model's schema, "
                          "see FEATURE_SCHEMA.md."
    )
    fps: Optional[float] = Field(None, description="Original capture fps, for real timestamps in `events`.")
    frame_skip: int = 8
    stride: int = DEFAULT_STRIDE
    threshold: Optional[float] = Field(default=None)
    window_frames: Optional[int] = Field(default=None, description="Real frames per window; "
                                         "default: the whole input if it fits in max_len.")

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
    """Recognition-only endpoint for keypoints. Checks the request's feature
    dimension against the SERVED model's schema first, so an incompatible
    client gets a clear, actionable error instead of a shape crash."""
    st = model_status()
    if not st["ready"]:
        return _err(503, f"No usable recognition model on the server: {st['reason']}")
    meta = st["meta"]
    expected = int(meta["input_dim"])
    actual_dim = len(req.features[0])
    if actual_dim != expected:
        return _err(
            422,
            f"Feature dimension mismatch: got {actual_dim}, the served model expects {expected} "
            f"(feature_schema_version={meta['feature_schema_version']}). The client's feature "
            f"extractor and the server's recognition model are incompatible -- extract keypoints "
            f"with the matching schema, or send the video to /recognize/video instead.",
            expected_feature_dim=expected, got_feature_dim=actual_dim,
            expected_feature_schema_version=meta["feature_schema_version"],
        )
    if req.threshold is not None and not (0.0 <= req.threshold <= 1.0):
        return _err(422, "threshold must be in [0, 1]")
    try:
        threshold = req.threshold if req.threshold is not None else DEFAULT_THRESHOLD
        result = predict_from_array(
            req.features, checkpoint_path=str(CHECKPOINT), onnx_path=str(ONNX),
            stride=max(1, req.stride), threshold=threshold, fps=req.fps, frame_skip=req.frame_skip,
            window_frames=req.window_frames,
        )
        return {
            "glosses": result["gloss_list"],
            "events": result["events"],
            "segments": result["segments"],
            "confidence": result["top_confidence"],
            "backend": result["backend"],
            "model_version": meta.get("model_version"),
            "feature_schema_version": meta["feature_schema_version"],
        }
    except fs.FeatureSchemaError as exc:
        return _err(422, str(exc))
    except Exception as exc:
        return _internal_error("Recognition", exc)

