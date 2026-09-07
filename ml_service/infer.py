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

from inference_viterbi import viterbi_decode, viterbi_events
from model import TemporalCNN
from notes_generator import build_notes_prompt, template_notes_from_tokens

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

_llm_client = None


def _get_llm_client():
    """Lazily construct the OpenAI-compatible client, so `openai` only needs
    to be importable (and a server only needs to be running) when an LLM
    notes mode is actually used -- template mode never touches this."""
    global _llm_client
    if _llm_client is None:
        from openai import OpenAI
        _llm_client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")
    return _llm_client


_torch_cache = None
_onnx_cache = None


def _normalize(x):
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-5
    return (x - mean) / std


def _pad_window(x, max_len):
    if len(x) >= max_len:
        return x[:max_len]
    return np.vstack([x, np.zeros((max_len - len(x), x.shape[1]), dtype=np.float32)])


def _load_torch(checkpoint):
    global _torch_cache
    if _torch_cache is not None:
        return _torch_cache
    ckpt = torch.load(checkpoint, map_location=DEVICE)
    model = TemporalCNN(int(ckpt["input_dim"]), len(ckpt["label2id"]))
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    id2label = {int(i): label for label, i in ckpt["label2id"].items()}
    _torch_cache = (model, id2label, int(ckpt["max_len"]))
    return _torch_cache


def _load_onnx(onnx_path, checkpoint=None):
    global _onnx_cache
    if _onnx_cache is not None:
        return _onnx_cache
    if ort is None:
        raise RuntimeError("onnxruntime is not installed")
    if checkpoint is None:
        checkpoint = DEFAULT_CHECKPOINT
    ckpt = torch.load(checkpoint, map_location="cpu")
    id2label = {int(i): label for label, i in ckpt["label2id"].items()}
    max_len = int(ckpt["max_len"])
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    _onnx_cache = (session, id2label, max_len, providers)
    return _onnx_cache


def get_model_meta(checkpoint_path=None):
    """Metadata the browser needs to run the same model client-side for the
    live webcam demo: max_len (window length the model was trained on),
    input_dim (keypoint vector size, 126 for two-hand MediaPipe landmarks),
    and the label vocabulary. Sourced from the .pt checkpoint so it's always
    in sync with whatever model is actually being served."""
    checkpoint = Path(checkpoint_path or DEFAULT_CHECKPOINT)
    ckpt = torch.load(checkpoint, map_location="cpu")
    id2label = {int(i): label for label, i in ckpt["label2id"].items()}
    return {
        "max_len": int(ckpt["max_len"]),
        "input_dim": int(ckpt["input_dim"]),
        "num_classes": len(id2label),
        "label2id": {v: k for k, v in id2label.items()},
        "id2label": id2label,
    }


def _window_batch(features, max_len, stride=12):
    """Slice a (T, D) keypoint sequence into overlapping (max_len, D)
    windows. Also returns each window's (start_frame, end_frame) span in
    *extracted-frame* units, so callers can convert to real seconds once
    they know the video's fps and the frame_skip used at extraction time."""
    x = _normalize(features)
def _window_batch(features, max_len, stride=12):
    """Slice a (T, D) raw keypoint sequence into overlapping (max_len, D)
    windows, normalizing EACH WINDOW INDEPENDENTLY (pad-then-normalize, in
    that order) -- this must exactly match dataset.py's SignDataset, which
    pads/trims each training sample to max_len and THEN computes that
    sample's own per-clip mean/std. Normalizing the whole long sequence
    once before slicing (the previous behavior) would compute one global
    mean/std across the entire clip, which drifts further from what the
    model was trained on the longer the video is and the more the
    signing scale/position varies over time -- exactly the failure mode
    long videos need to handle well. Also returns each window's
    (start_frame, end_frame) span in *extracted-frame* units for timestamps."""
    x = np.asarray(features, dtype=np.float32)
    if len(x) <= max_len:
        return _normalize(_pad_window(x, max_len))[None, ...], [(0, len(x))]

    windows = []
    spans = []
    for start in range(0, len(x) - max_len + 1, stride):
        windows.append(_normalize(x[start:start + max_len]))
        spans.append((start, start + max_len))
    if (len(x) - max_len) % stride:
        start = len(x) - max_len
        windows.append(_normalize(x[start:]))
        spans.append((start, len(x)))
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


def predict_from_features(feature_path, checkpoint_path=None, onnx_path=None,
                          stride=12, threshold=0.55, fps=None, frame_skip=8):
    """Run the sliding-window pipeline over one clip's extracted keypoints.

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

    checkpoint = Path(checkpoint_path or DEFAULT_CHECKPOINT)
    onnx_file = Path(onnx_path or DEFAULT_ONNX)
    use_onnx = onnx_file.exists() and ort is not None
    if use_onnx:
        session, id2label, max_len, providers = _load_onnx(onnx_file, checkpoint)
        batch, spans = _window_batch(features, max_len, stride)
        logits = _run_windows_chunked(batch, session=session)
    else:
        model, id2label, max_len = _load_torch(checkpoint)
        batch, spans = _window_batch(features, max_len, stride)
        logits = _run_windows_chunked(batch, model=model)
        providers = [DEVICE]

    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    path = viterbi_decode(probs, threshold=threshold)
    gloss_list = [id2label[i] for i in path]

    segments = []
    window_times = []
    for i, (p, (fstart, fend)) in enumerate(zip(probs, spans)):
        j = int(p.argmax())
        seg = {"window": i, "label": id2label[j], "confidence": float(p[j])}
        if fps:
            # fstart/fend are in *extracted-frame* units; each extracted
            # frame represents `frame_skip` original frames.
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


def generate_llm_notes(gloss_list, model=None, style="concise", base_url=None):
    """Call the configured local LLM server (Ollama or llama.cpp -- both
    speak the OpenAI-compatible chat-completions API) to turn a gloss
    sequence into notes. Requires the server to be running and the `openai`
    package installed (pip install openai)."""
    if not gloss_list:
        raise ValueError("gloss_list is empty")

    global _llm_client
    if base_url and base_url != LLM_BASE_URL:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key="not-needed")
    else:
        client = _get_llm_client()

    prompt = build_notes_prompt(gloss_list, style=style)
    response = client.chat.completions.create(
        model=model or LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You convert sign-language gloss sequences "
                    "into clear written notes."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("LLM server returned an empty response")
    return text


# Kept as an alias: earlier versions of this project called this function
# generate_llama_cpp_notes specifically. Same implementation now serves
# both providers via LLM_PROVIDER/LLM_BASE_URL.
generate_llama_cpp_notes = generate_llm_notes


def generate_notes(gloss_list, mode="template", llm_model=None, style="concise", **_legacy_kwargs):
    """Turn a gloss sequence into notes.

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

    if mode in ("llm", "llama_cpp", "ollama"):
        try:
            return generate_llm_notes(gloss_list, model=llm_model, style=style)
        except Exception as e:
            print(f"[infer] LLM ({LLM_PROVIDER} @ {LLM_BASE_URL}) unavailable: {e}; "
                  f"falling back to template notes")
    return template_notes_from_tokens(gloss_list)


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
