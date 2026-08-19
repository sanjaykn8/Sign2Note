"""Video -> overlapping temporal windows -> ONNX classifier -> Viterbi -> glosses."""

import argparse
from pathlib import Path

import numpy as np
import torch
from openai import OpenAI

from inference_viterbi import viterbi_decode
from model import TemporalCNN
from notes_generator import template_notes_from_tokens

try:
    import onnxruntime as ort
except ImportError:
    ort = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CHECKPOINT = Path("models/sign_recog/checkpoints/demo.pt")
DEFAULT_ONNX = Path("models/sign_recog/sign_recog.onnx")

LLAMA_CPP_URL = "http://127.0.0.1:8081/v1"
_llama_client = OpenAI(base_url=LLAMA_CPP_URL, api_key="not-needed")

_torch_cache = None
_onnx_cache = None
_meta_cache = None


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
    global _onnx_cache, _meta_cache
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


def _window_batch(features, max_len, stride=12):
    x = _normalize(features)
    if len(x) <= max_len:
        return _pad_window(x, max_len)[None, ...]
    windows = []
    for start in range(0, len(x) - max_len + 1, stride):
        windows.append(x[start:start + max_len])
    if (len(x) - max_len) % stride:
        windows.append(x[-max_len:])
    return np.stack(windows).astype(np.float32)


def predict_from_features(feature_path, checkpoint_path=None, onnx_path=None,
                          stride=12, threshold=0.55):
    feature_path = Path(feature_path)
    features = np.load(feature_path).astype(np.float32)
    if len(features) == 0:
        return {"gloss_list": [], "segments": [], "top_confidence": 0.0}

    checkpoint = Path(checkpoint_path or DEFAULT_CHECKPOINT)
    onnx_file = Path(onnx_path or DEFAULT_ONNX)
    if onnx_file.exists() and ort is not None:
        session, id2label, max_len, providers = _load_onnx(onnx_file, checkpoint)
        batch = _window_batch(features, max_len, stride)
        logits = session.run(["logits"], {"keypoints": batch})[0]
    else:
        model, id2label, max_len = _load_torch(checkpoint)
        batch = torch.from_numpy(_window_batch(features, max_len, stride))
        with torch.no_grad():
            logits = model(batch.to(DEVICE)).cpu().numpy()

    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    path = viterbi_decode(probs, threshold=threshold)
    gloss_list = [id2label[i] for i in path]

    segments = []
    for i, p in enumerate(probs):
        j = int(p.argmax())
        segments.append({"window": i, "label": id2label[j], "confidence": float(p[j])})

    return {
        "gloss_list": gloss_list,
        "segments": segments,
        "top_confidence": float(probs.max(axis=1).mean()),
        "backend": "onnx" if onnx_file.exists() and ort is not None else "torch",
        "providers": providers if onnx_file.exists() and ort is not None else [DEVICE],
    }


def generate_llama_cpp_notes(gloss_list, model="gemma4", style="concise"):
    """Call the local llama.cpp server (OpenAI-compatible endpoint) to turn
    a gloss sequence into notes."""
    prompt = f"""
You are the language reconstruction component of Sign2Notes.
Convert the recognized sign glosses into coherent notes.

Recognized glosses:
{", ".join(gloss_list)}

Style:
{style}

Rules:
- Preserve the meaning of the recognized signs.
- Do not invent facts.
- Do not add information that is not supported by the glosses.
- Correct obvious grammatical ordering.
- Produce concise, readable notes.
"""
    response = _llama_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You convert sign-language gloss sequences "
                    "into clear written notes."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def generate_notes(gloss_list, mode="template", ollama_model="gemma4", style="concise"):
    if mode in ("llama_cpp", "ollama"):
        try:
            return generate_llama_cpp_notes(gloss_list, model=ollama_model, style=style)
        except Exception as e:
            print(f"[infer] llama.cpp server unavailable: {e}; template fallback")
    return template_notes_from_tokens(gloss_list)


def predict_from_video(video_path, feature_output, frame_skip=8, **kwargs):
    from feature_extraction import extract_single_video
    feature_path = extract_single_video(video_path, feature_output, frame_skip=frame_skip)
    if feature_path is None:
        return None
    return predict_from_features(str(feature_path), **kwargs)


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
    p.add_argument("--notes_mode", choices=["template", "llama_cpp"], default="template")
    p.add_argument("--llm_model", default="gemma4")
    args = p.parse_args()

    if args.feature_path:
        result = predict_from_features(args.feature_path, args.checkpoint, args.onnx, args.stride, args.threshold)
    elif args.video_path:
        result = predict_from_video(args.video_path, args.feature_output, args.frame_skip,
                                    checkpoint_path=args.checkpoint, onnx_path=args.onnx,
                                    stride=args.stride, threshold=args.threshold)
    else:
        raise SystemExit("Provide --feature_path or --video_path")

    print(result)
    print(generate_notes(result["gloss_list"], mode=args.notes_mode, ollama_model=args.llm_model))