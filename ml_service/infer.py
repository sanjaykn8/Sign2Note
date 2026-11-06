"""Video -> overlapping temporal windows -> ONNX classifier -> Viterbi -> glosses.

Long-video support: predict_from_features() runs the model over ALL windows
of a clip, however long, in fixed-size chunks (see _INFER_CHUNK) rather than
one giant batch -- this bounds peak memory/compute regardless of video
length. The raw per-frame keypoint sequence itself is tiny even for a multi-
minute video (a few hundred KB at most -- see ARCHITECTURE.md), so the real
memory-efficiency lever for long videos is here, at the inference-batching
step, not at keypoint storage.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

import feature_schema as fs
from checkpoint_meta import compute_vocab_hash
from inference_viterbi import viterbi_decode, viterbi_events
from model import build_model
from notes_generator import (
    build_notes_prompt, template_notes_from_tokens,
    build_transcript_prompt, template_transcript_from_tokens,
    PROMPT_VERSIONS, TRANSCRIPT_PROMPT_VERSION,
)

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CHECKPOINT = Path("models/sign_recog/checkpoints/demo.pt")
DEFAULT_ONNX = Path("models/sign_recog/sign_recog.onnx")

# How many windows to run through the model per forward pass. Bounds peak
# memory/compute for very long videos regardless of how many windows the
# sliding-window step produces. Override with INFER_CHUNK_SIZE if you have
# more VRAM to spare (or less -- lower it on constrained GPUs).
INFER_CHUNK_SIZE = int(os.environ.get("INFER_CHUNK_SIZE", 64))

# ── LLM provider config (env-driven; no hard-coded model name) ──────────────
# Both "ollama" and "llama_cpp" speak the OpenAI-compatible chat-completions
# API (Ollama has supported this natively since 0.1.x), so one client
# handles both -- they only differ in default base URL / model name.
LLM_PROVIDER  = os.environ.get("LLM_PROVIDER", "llama_cpp")
LLM_MODEL     = os.environ.get("LLM_MODEL", "gemma4")
LLM_BASE_URL  = os.environ.get(
    "LLM_BASE_URL",
    "http://127.0.0.1:11434/v1" if LLM_PROVIDER == "ollama" else "http://127.0.0.1:8081/v1",
)

# Bounded LLM calls: the OpenAI client's defaults (10-minute timeout, 2
# automatic retries) meant a hung local server could block a notes request
# for ~30 minutes before the deterministic fallback kicked in.
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT", 90))
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", 1))
LLM_MAX_OUTPUT_CHARS = int(os.environ.get("LLM_MAX_OUTPUT_CHARS", 20000))

_llm_client = None


def _get_llm_client():
    """Lazily construct the OpenAI-compatible client, so `openai` only needs
    to be importable (and a server only needs to be running) when an LLM
    notes mode is actually used -- template mode never touches this."""
    global _llm_client
    if _llm_client is None:
        from openai import OpenAI
        _llm_client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed",
                             timeout=LLM_TIMEOUT_SECONDS, max_retries=LLM_MAX_RETRIES)
    return _llm_client


class LLMOutputError(RuntimeError):
    """The LLM answered, but with something unusable as notes."""


def clean_llm_output(text: str, require_markdown_heading: bool = False) -> str:
    """Validate/clean raw LLM output before it is shown to a user:
    strips reasoning blocks some local models emit (<think>...</think>) and
    a wrapping ```markdown fence, rejects empty/oversized output, and (for
    notes) requires at least one Markdown heading or bullet so plain chatter
    like "Sure! Here are your notes" isn't presented as notes."""
    import re
    t = (text or "").strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL | re.IGNORECASE).strip()
    fence = re.match(r"^```[a-zA-Z]*\s*\n(.*?)\n```\s*$", t, flags=re.DOTALL)
    if fence:
        t = fence.group(1).strip()
    if not t:
        raise LLMOutputError("LLM server returned an empty response")
    if len(t) > LLM_MAX_OUTPUT_CHARS:
        raise LLMOutputError(f"LLM output too long ({len(t)} chars)")
    if require_markdown_heading and not re.search(r"^\s*(#{1,6}\s|[-*]\s|\d+\.\s)", t, flags=re.MULTILINE):
        raise LLMOutputError("LLM output has no Markdown structure (no heading or list)")
    return t


_torch_cache = None
_onnx_cache = None


def _normalize(x):
    """Thin backward-compat wrapper -- see feature_schema.normalize_sequence
    (the single shared implementation dataset.py also uses). Kept as a
    module-level name because a few call sites and tests still import
    `_normalize` from here directly."""
    return fs.normalize_sequence(x)


def _pad_window(x, max_len):
    """Thin backward-compat wrapper -- see feature_schema.pad_or_trim."""
    return fs.pad_or_trim(x, max_len)


def _file_key(path) -> str:
    """Cache key for a model file: resolved path + mtime, so replacing a
    checkpoint on disk (retraining) is picked up without a restart."""
    p = Path(path).resolve()
    try:
        return f"{p}@{p.stat().st_mtime_ns}"
    except OSError:
        return str(p)


def _cache_get(cache, key):
    # Tests reset the caches by assigning None; treat that as empty.
    return cache.get(key) if isinstance(cache, dict) else None


def _read_checkpoint(checkpoint, map_location="cpu"):
    # weights_only=False: our checkpoints are plain dicts written by
    # train.py (trusted, local files) that include metadata alongside the
    # state_dict. Never point MODEL_PATH at a checkpoint from an untrusted
    # source -- torch.load unpickles it.
    try:
        return torch.load(checkpoint, map_location=map_location, weights_only=False)
    except TypeError:  # very old torch without weights_only
        return torch.load(checkpoint, map_location=map_location)


def _load_torch(checkpoint):
    """Load (and cache, PER CHECKPOINT PATH) a torch model. The cache used
    to be a single global slot, so loading a second checkpoint (e.g. the
    legacy v1 model for comparison) silently returned the first one."""
    global _torch_cache
    key = _file_key(checkpoint)
    cached = _cache_get(_torch_cache, key)
    if cached is not None:
        return cached
    ckpt = _read_checkpoint(checkpoint, map_location=DEVICE)
    architecture = ckpt.get("architecture", "temporal_cnn")
    model = build_model(architecture, int(ckpt["input_dim"]), len(ckpt["label2id"]))
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    id2label = {int(i): label for label, i in ckpt["label2id"].items()}
    entry = (model, id2label, int(ckpt["max_len"]))
    if not isinstance(_torch_cache, dict):
        _torch_cache = {}
    _torch_cache[key] = entry
    return entry


def _load_onnx(onnx_path, checkpoint=None):
    """Load (and cache, per ONNX+checkpoint pair) an onnxruntime session.
    The checkpoint supplies the label mapping and max_len; the ONNX graph's
    declared input dim is checked against the checkpoint's input_dim so a
    stale/mismatched export can't be paired with the wrong vocabulary."""
    global _onnx_cache
    if ort is None:
        raise RuntimeError("onnxruntime is not installed")
    if checkpoint is None:
        checkpoint = DEFAULT_CHECKPOINT
    key = f"{_file_key(onnx_path)}|{_file_key(checkpoint)}"
    cached = _cache_get(_onnx_cache, key)
    if cached is not None:
        return cached
    ckpt = _read_checkpoint(checkpoint, map_location="cpu")
    id2label = {int(i): label for label, i in ckpt["label2id"].items()}
    max_len = int(ckpt["max_len"])
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    onnx_dim = session.get_inputs()[0].shape[-1]
    if isinstance(onnx_dim, int) and onnx_dim != int(ckpt["input_dim"]):
        raise fs.FeatureSchemaError(
            f"ONNX model {onnx_path} expects {onnx_dim}-dim input but checkpoint {checkpoint} "
            f"says input_dim={ckpt['input_dim']} -- they are not the same model."
        )
    onnx_classes = session.get_outputs()[0].shape[-1]
    if isinstance(onnx_classes, int) and onnx_classes != len(id2label):
        raise fs.FeatureSchemaError(
            f"ONNX model {onnx_path} outputs {onnx_classes} classes but checkpoint {checkpoint} "
            f"has {len(id2label)} labels -- they are not the same model."
        )
    entry = (session, id2label, max_len, providers)
    if not isinstance(_onnx_cache, dict):
        _onnx_cache = {}
    _onnx_cache[key] = entry
    return entry


def get_model_meta(checkpoint_path=None):
    """Metadata the browser (and any distributed client) needs to run the
    same model: max_len (window length trained on), input_dim (feature
    vector size -- 285 for the current hand+body+face schema v2, 126 for
    an older hand-only v1 checkpoint), the label vocabulary, and enough
    versioning info (feature_schema_version, architecture) for a caller to
    detect an incompatible model BEFORE trying to run it (see RULE 26 /
    api.py's /model/meta and model-compatibility check). Sourced from the
    .pt checkpoint so it's always in sync with whatever model is actually
    being served. `feature_schema_version`/`architecture` default to the
    v1-era values for older checkpoints saved before those fields existed
    (RULE 67: backward compatible, not a breaking change)."""
    checkpoint = Path(checkpoint_path or DEFAULT_CHECKPOINT)
    ckpt = _read_checkpoint(checkpoint, map_location="cpu")
    id2label = {int(i): label for label, i in ckpt["label2id"].items()}
    schema = str(ckpt.get("feature_schema_version", fs.LEGACY_SCHEMA_VERSION))
    return {
        "max_len": int(ckpt["max_len"]),
        "input_dim": int(ckpt["input_dim"]),
        "num_classes": len(id2label),
        "label2id": {v: k for k, v in id2label.items()},
        "id2label": id2label,
        "feature_schema_version": schema,
        "feature_type": fs.SCHEMA_TYPES.get(schema, "unknown"),
        "is_legacy_schema": schema == fs.LEGACY_SCHEMA_VERSION,
        "architecture": ckpt.get("architecture", "temporal_cnn"),
        "vocab_hash": ckpt.get("vocab_hash") or compute_vocab_hash(ckpt["label2id"]),
        "best_val_accuracy": ckpt.get("best_val_accuracy"),
        "used_official_split": (ckpt.get("training_config") or {}).get("used_official_split"),
        "test_metrics": ckpt.get("test_metrics"),
        "trained_at": ckpt.get("trained_at"),
        "model_version": ckpt.get("model_version") or checkpoint.parent.parent.name,
    }


_meta_cache = {}


def model_input_dim(checkpoint_path) -> int:
    """input_dim recorded in a checkpoint (cached per path)."""
    key = _file_key(checkpoint_path)
    if key not in _meta_cache:
        ckpt = _read_checkpoint(checkpoint_path, map_location="cpu")
        _meta_cache[key] = int(ckpt["input_dim"])
    return _meta_cache[key]


def _window_spans(n_frames, window, stride):
    """(start, end) spans of overlapping windows of `window` real frames
    covering n_frames, the last one aligned to the end of the sequence."""
    if n_frames <= window:
        return [(0, n_frames)]
    spans = [(s, s + window) for s in range(0, n_frames - window + 1, stride)]
    if (n_frames - window) % stride:
        spans.append((n_frames - window, n_frames))
    return spans


def _window_batch(features, max_len, stride=12, window_frames=None):
    """Slice a (T, D) raw keypoint sequence into overlapping windows of
    `window_frames` real frames (default: max_len), each padded to max_len
    and normalized INDEPENDENTLY via feature_schema.prepare_window()
    (pad-then-normalize -- exactly what dataset.py does to each training
    clip). Training clips are ~20-35 sampled frames zero-padded to
    max_len, so windows shorter than max_len (api.py's WINDOW_FRAMES)
    reproduce the training distribution on continuous video. Returns the
    windows and each window's (start_frame, end_frame) span in
    extracted-frame units for timestamps."""
    x = np.asarray(features, dtype=np.float32)
    window = max_len if not window_frames else max(1, min(int(window_frames), max_len))
    spans = _window_spans(len(x), window, max(1, int(stride)))
    windows = [fs.prepare_window(x[a:b], max_len) for a, b in spans]
    return np.stack(windows).astype(np.float32), spans


def _run_windows_chunked(batch, session=None, model=None):
    """Run `batch` (N, max_len, D) through the model in fixed-size chunks
    instead of one N-sized forward pass. For a short clip N is tiny and this
    is a no-op; for a several-minutes video N can be in the hundreds, and
    chunking keeps peak memory bounded regardless of video length."""
    n = len(batch)
    outs = []
    for start in range(0, n, INFER_CHUNK_SIZE):
        chunk = batch[start:start + INFER_CHUNK_SIZE]
        if session is not None:
            outs.append(session.run(["logits"], {"keypoints": chunk})[0])
        else:
            with torch.no_grad():
                t = torch.from_numpy(chunk).to(DEVICE)
                outs.append(model(t).cpu().numpy())
    return np.concatenate(outs, axis=0)


def _run_windows_streaming(x, max_len, stride, window_frames, session=None, model=None):
    """Like _window_batch + _run_windows_chunked, but builds the padded
    windows one INFER_CHUNK_SIZE chunk at a time, so peak memory does not
    grow with video length (a 30-minute video at stride 4 is ~3400
    windows = ~250 MB if all materialized at once)."""
    window = max_len if not window_frames else max(1, min(int(window_frames), max_len))
    spans = _window_spans(len(x), window, max(1, int(stride)))
    outs = []
    for start in range(0, len(spans), INFER_CHUNK_SIZE):
        chunk_spans = spans[start:start + INFER_CHUNK_SIZE]
        chunk = np.stack([fs.prepare_window(x[a:b], max_len) for a, b in chunk_spans]).astype(np.float32)
        outs.append(_run_windows_chunked(chunk, session=session, model=model))
    return np.concatenate(outs, axis=0), spans


def predict_from_array(features, checkpoint_path=None, onnx_path=None,
                       stride=12, threshold=0.55, fps=None, frame_skip=8, window_frames=None):
    """Sliding-window recognition over an in-memory (T, input_dim) keypoint
    array (used directly by /recognize; predict_from_features() wraps it).
    `window_frames` = real frames per window (default: max_len, the
    original behavior); see _window_batch for why shorter windows padded
    to max_len match training better on continuous video."""
    features = np.asarray(features, dtype=np.float32)
    if len(features) == 0:
        return {"gloss_list": [], "segments": [], "events": [], "top_confidence": 0.0,
                "backend": "none", "providers": []}
    if features.ndim != 2 or not np.all(np.isfinite(features)):
        raise fs.FeatureSchemaError(
            "features must be a finite (num_frames, feature_dim) array"
        )

    checkpoint = Path(checkpoint_path or DEFAULT_CHECKPOINT)
    # Validate against the schema of the MODEL being served (126 for the
    # legacy v1 model, 285 for v2) -- never silently feed one schema's
    # vectors to a model trained on the other.
    fs.assert_feature_dim(
        features.shape[-1], context="in-memory features",
        expected_dim=model_input_dim(checkpoint),
    )
    onnx_file = Path(onnx_path or DEFAULT_ONNX)
    use_onnx = onnx_file.exists() and ort is not None
    if use_onnx:
        session, id2label, max_len, providers = _load_onnx(onnx_file, checkpoint)
        logits, spans = _run_windows_streaming(features, max_len, stride, window_frames, session=session)
    else:
        model, id2label, max_len = _load_torch(checkpoint)
        logits, spans = _run_windows_streaming(features, max_len, stride, window_frames, model=model)
        providers = [DEVICE]

    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    path = viterbi_decode(probs, threshold=threshold)
    gloss_list = [id2label[i] for i in path]

    segments = []
    window_times = []
    for i, (p, (fstart, fend)) in enumerate(zip(probs, spans)):
        top3 = np.argsort(-p)[:3]
        j = int(top3[0])
        seg = {"window": i, "label": id2label[j], "confidence": float(p[j]),
               "alternatives": [{"label": id2label[int(k)], "confidence": float(p[k])} for k in top3[1:]]}
        if fps:
            t0 = (fstart * frame_skip) / fps
            t1 = (fend * frame_skip) / fps
            seg["start_time"] = round(t0, 2)
            seg["end_time"] = round(t1, 2)
            window_times.append((t0, t1))
        segments.append(seg)

    events = []
    if fps and window_times:
        events = viterbi_events(probs, id2label, window_times, threshold=threshold)

    return {
        "gloss_list": gloss_list,
        "segments": segments,
        "events": events,
        "top_confidence": float(probs.max(axis=1).mean()),
        "backend": "onnx" if use_onnx else "torch",
        "providers": providers,
    }


def predict_from_features(feature_path, checkpoint_path=None, onnx_path=None,
                          stride=12, threshold=0.55, fps=None, frame_skip=8, window_frames=None):
    """Run the sliding-window pipeline over one clip's extracted keypoints,
    loaded from a .npy file on disk. Thin wrapper around
    predict_from_array() -- see that function for the actual pipeline.

    If `fps` is given (the *original* video's frames-per-second, before
    frame_skip subsampling), each window's frame span is also converted to
    real (start_time, end_time) seconds and a timestamped, duplicate-
    collapsed `events` list is returned alongside the existing `gloss_list`/
    `segments` fields -- this is what "long video" mode uses to report an
    ordered sequence of signs with timestamps instead of a single label.
    """
    feature_path = Path(feature_path)
    features = np.load(feature_path).astype(np.float32)
    if len(features) == 0:
        return {"gloss_list": [], "segments": [], "events": [], "top_confidence": 0.0,
                "backend": "none", "providers": []}
    fs.assert_feature_dim(
        features.shape[-1], context=str(feature_path),
        expected_dim=model_input_dim(checkpoint_path or DEFAULT_CHECKPOINT),
    )
    return predict_from_array(
        features, checkpoint_path=checkpoint_path, onnx_path=onnx_path,
        stride=stride, threshold=threshold, fps=fps, frame_skip=frame_skip,
        window_frames=window_frames,
    )


def _llm_client_for(base_url=None):
    if base_url and base_url != LLM_BASE_URL:
        from openai import OpenAI
        return OpenAI(base_url=base_url, api_key="not-needed",
                      timeout=LLM_TIMEOUT_SECONDS, max_retries=LLM_MAX_RETRIES)
    return _get_llm_client()


def _chat(prompt, preamble, model=None, base_url=None):
    client = _llm_client_for(base_url)
    # System instruction is folded into the user turn to avoid chat-template
    # parsing bugs seen with some llama.cpp builds.
    response = client.chat.completions.create(
        model=model or LLM_MODEL,
        messages=[{"role": "user", "content": preamble + "\n\n" + prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def generate_llm_notes(gloss_list, model=None, style="concise", base_url=None,
                       confidences=None, context=None):
    if not gloss_list:
        raise ValueError("gloss_list is empty")
    prompt = build_notes_prompt(gloss_list, style=style, confidences=confidences, context=context)
    raw = _chat(prompt, "You are an assistant that converts sign-language gloss sequences into "
                        "clear written notes, strictly grounded in the glosses provided.",
                model=model, base_url=base_url)
    return clean_llm_output(raw, require_markdown_heading=True)


# Kept as an alias: earlier versions of this project called this function
# generate_llama_cpp_notes specifically. Same implementation now serves
# both providers via LLM_PROVIDER/LLM_BASE_URL.
generate_llama_cpp_notes = generate_llm_notes

_LLM_MODES = ("llm", "llama_cpp", "ollama")


def generate_notes_with_meta(gloss_list, mode="template", llm_model=None, style="concise",
                             confidences=None, context=None):
    """Like generate_notes(), but also returns generation metadata so the
    UI can say honestly which generator produced the text:
      {"generator": "llm" | "template", "fallback_reason": str | None,
       "style": ..., "prompt_version": ..., "llm_model": ...}
    mode="template" never touches the network; an LLM mode falls back to the
    deterministic, style-aware template on ANY failure (server down,
    timeout, empty/unstructured output, missing package)."""
    meta = {"generator": "template", "fallback_reason": None, "style": style,
            "prompt_version": PROMPT_VERSIONS.get(style), "llm_model": None}
    if mode in _LLM_MODES:
        try:
            text = generate_llm_notes(gloss_list, model=llm_model, style=style,
                                      confidences=confidences, context=context)
            meta.update(generator="llm", llm_model=llm_model or LLM_MODEL)
            return text, meta
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            print(f"[infer] LLM ({LLM_PROVIDER} @ {LLM_BASE_URL}) failed: {reason}; "
                  f"falling back to template notes")
            meta["fallback_reason"] = reason[:300]
    text = template_notes_from_tokens(gloss_list, style=style, confidences=confidences)
    return text, meta


def generate_notes(gloss_list, mode="template", llm_model=None, style="concise", **_legacy_kwargs):
    """Turn a gloss sequence into notes (string only -- see
    generate_notes_with_meta() for the variant that also reports whether
    the LLM or the deterministic fallback produced it).

    mode="template" -> deterministic, always available, no network/LLM needed.
    mode="llm" / "llama_cpp" / "ollama" -> calls the configured local LLM
        server; falls back to template notes on ANY failure (server down,
        model not loaded, openai package missing, network error, etc.) so
        the caller always gets usable notes back.

    `_legacy_kwargs` swallows the old `ollama_model=` keyword so any caller
    still using the previous name doesn't crash.
    """
    if "ollama_model" in _legacy_kwargs and llm_model is None:
        llm_model = _legacy_kwargs["ollama_model"]
    text, _ = generate_notes_with_meta(
        gloss_list, mode=mode, llm_model=llm_model, style=style,
        confidences=_legacy_kwargs.get("confidences"), context=_legacy_kwargs.get("context"),
    )
    return text


def generate_llm_transcript(gloss_list, model=None, base_url=None, confidences=None):
    """LLM-backed natural-language transcript (Layer 2 -- see
    notes_generator.py's module docstring on the three output layers).
    Deliberately its own function, not a style variant of
    generate_llm_notes(): the prompt (build_transcript_prompt) asks for
    flowing prose, not structured/headed notes."""
    if not gloss_list:
        raise ValueError("gloss_list is empty")
    prompt = build_transcript_prompt(gloss_list, confidences=confidences)
    raw = _chat(prompt, "You are an assistant that converts sign-language gloss sequences into a "
                        "short natural-language transcript, strictly grounded in the glosses.",
                model=model, base_url=base_url)
    return clean_llm_output(raw, require_markdown_heading=False)


def generate_transcript_with_meta(gloss_list, mode="template", llm_model=None, confidences=None):
    meta = {"generator": "template", "fallback_reason": None,
            "prompt_version": TRANSCRIPT_PROMPT_VERSION, "llm_model": None}
    if mode in _LLM_MODES:
        try:
            text = generate_llm_transcript(gloss_list, model=llm_model, confidences=confidences)
            meta.update(generator="llm", llm_model=llm_model or LLM_MODEL)
            return text, meta
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            print(f"[infer] LLM ({LLM_PROVIDER} @ {LLM_BASE_URL}) failed: {reason}; "
                  f"falling back to template transcript")
            meta["fallback_reason"] = reason[:300]
    return template_transcript_from_tokens(gloss_list), meta


def generate_transcript(gloss_list, mode="template", llm_model=None, **_legacy_kwargs):
    """Turn a gloss sequence into a natural-language TRANSCRIPT (Layer 2 --
    flowing prose, distinct from generate_notes()'s structured Layer 3
    output). Same LLM-with-deterministic-fallback contract as
    generate_notes(): mode="template" never touches the network; mode="llm"
    falls back to the template on ANY failure so the caller always gets a
    usable transcript back.

    Called exactly ONCE per Live Transcription session, when the user
    stops it -- never continuously, never per-gloss."""
    if "ollama_model" in _legacy_kwargs and llm_model is None:
        llm_model = _legacy_kwargs["ollama_model"]
    text, _ = generate_transcript_with_meta(gloss_list, mode=mode, llm_model=llm_model,
                                            confidences=_legacy_kwargs.get("confidences"))
    return text


def predict_from_video(video_path, feature_output, frame_skip=8, **kwargs):
    from feature_extraction import extract_single_video
    feature_path = extract_single_video(video_path, feature_output, frame_skip=frame_skip)
    if feature_path is None:
        return None
    return predict_from_features(str(feature_path), frame_skip=frame_skip, **kwargs)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feature_path")
    p.add_argument("--video_path")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--onnx", default=str(DEFAULT_ONNX))
    p.add_argument("--feature_output", default="data/tmp_features")
    p.add_argument("--frame_skip", type=int, default=8)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--notes_mode", choices=["template", "llm", "llama_cpp", "ollama"], default="template")
    p.add_argument("--llm_model", default=None)
    args = p.parse_args()

    if args.feature_path:
        result = predict_from_features(args.feature_path, args.checkpoint, args.onnx,
                                        args.stride, args.threshold)
    elif args.video_path:
        result = predict_from_video(args.video_path, args.feature_output, args.frame_skip,
                                    checkpoint_path=args.checkpoint, onnx_path=args.onnx,
                                    stride=args.stride, threshold=args.threshold)
    else:
        raise SystemExit("Provide --feature_path or --video_path")

    print(result)
    print(generate_notes(result["gloss_list"], mode=args.notes_mode, llm_model=args.llm_model))
