"""
infer.py  –  Inference: video → gloss sequence → structured notes.

Can be used as:
  (a) imported module  →  api.py calls predict_from_features() / generate_notes()
  (b) CLI tool         →  python infer.py --feature_path X.npy ...
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path

from model            import TemporalCNN
from notes_generator  import tokens_to_markdown, template_notes_from_tokens

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT   = Path("models/sign_recog/checkpoints/demo.pt")
MAX_LEN      = 50

# ── lazy model cache (avoid reloading on every API call) ─────────────────────
_model_cache   = None
_id2label_cache = None

def _load_model(checkpoint_path=None):
    global _model_cache, _id2label_cache
    if _model_cache is not None:
        return _model_cache, _id2label_cache

    ckpt_path = Path(checkpoint_path) if checkpoint_path else CHECKPOINT
    ckpt      = torch.load(str(ckpt_path), map_location=DEVICE)

    label2id  = ckpt["label2id"]
    id2label  = {i: l for l, i in label2id.items()}

    model = TemporalCNN(input_dim=126, num_classes=len(label2id))
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE)
    model.eval()

    _model_cache    = model
    _id2label_cache = id2label
    return model, id2label


# ── feature extraction helper (for single-video paths) ───────────────────────
def _extract_video_features(video_path, frame_skip=8):
    """Extract MediaPipe keypoints from a raw video file.
    Returns a (1, MAX_LEN, 126) tensor or None."""
    import cv2
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    cap      = cv2.VideoCapture(str(video_path))
    frames   = []
    idx      = 0

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                        model_complexity=0) as hands:
        while True:
            ret, frame = cap.read()
            if not ret: break
            if idx % frame_skip != 0:
                idx += 1
                continue
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r   = hands.process(img)

            def lm(h):
                if h is None: return np.zeros((21,3), np.float32)
                return np.array([[p.x,p.y,p.z] for p in h.landmark], np.float32)

            left = right = np.zeros((21,3), np.float32)
            if r.multi_hand_landmarks:
                for i, h in enumerate(r.multi_hand_landmarks):
                    if i == 0: left  = lm(h)
                    else:      right = lm(h)
            frames.append(np.concatenate([left.flatten(), right.flatten()]))
            idx += 1

    cap.release()
    if not frames: return None

    x = np.stack(frames)
    if len(x) >= MAX_LEN: x = x[:MAX_LEN]
    else:
        pad = np.zeros((MAX_LEN - len(x), x.shape[1]))
        x   = np.vstack([x, pad])

    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)   # (1, T, D)


# ── public: predict from a pre-extracted .npy feature file ───────────────────
def predict_from_features(feature_path, checkpoint_path=None, top_k=5):
    """
    Load a .npy keypoint array, run model, return list of gloss strings.
    top_k returns the k most-likely predictions (useful for longer sequences).
    """
    model, id2label = _load_model(checkpoint_path)

    x = np.load(str(feature_path))
    if len(x) >= MAX_LEN: x = x[:MAX_LEN]
    else:
        pad = np.zeros((MAX_LEN - len(x), x.shape[1]))
        x   = np.vstack([x, pad])

    t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(t)                          # (1, C)
        probs  = torch.softmax(logits, dim=-1)[0]  # (C,)

    # return top-k unique glosses as the "detected sequence"
    topk = torch.topk(probs, k=min(top_k, len(id2label)))
    glosses = [id2label[i.item()] for i in topk.indices]
    return glosses


# ── public: predict from a raw video file ─────────────────────────────────────
def predict_from_video(video_path, checkpoint_path=None):
    """Convenience wrapper: extract features then predict."""
    t = _extract_video_features(str(video_path))
    if t is None:
        return None

    model, id2label = _load_model(checkpoint_path)
    t = t.to(DEVICE)
    with torch.no_grad():
        pred = model(t).argmax(1).item()
    return [id2label[pred]]


# ── public: gloss list → markdown notes ──────────────────────────────────────
def generate_notes(gloss_list, use_llama=False, gguf_path=None, llama_bin=None,
                   use_ollama=False, ollama_model="llama3.2:3b"):
    """
    Convert a gloss list to structured markdown notes.
    Tries (in order): Ollama → llama.cpp → template fallback.
    """
    if not gloss_list:
        return "# Notes\n\nNo recognisable signs detected in the video."

    if use_ollama:
        try:
            import requests
            prompt = (
                "You are an academic note-generation assistant.\n\n"
                "Convert the following ASL gloss sequence into detailed, well-structured lecture notes.\n"
                "Requirements: use headings, bullet points, bold key terms, coherent academic sentences.\n\n"
                f"Gloss sequence:\n{' '.join(gloss_list)}"
            )
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": ollama_model, "prompt": prompt, "stream": False},
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["response"]
        except Exception as e:
            print(f"[infer] Ollama unavailable ({e}), falling back to template notes.")

    # llama.cpp path
    if use_llama and gguf_path:
        return tokens_to_markdown(gloss_list, use_llama=True,
                                  gguf_path=gguf_path,
                                  llama_bin_path=llama_bin or "./llama.cpp/main")

    # deterministic template fallback (always works, no dependencies)
    return template_notes_from_tokens(gloss_list)


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--feature_path", help="Path to a pre-extracted .npy file")
    group.add_argument("--video_path",   help="Path to a raw video file")

    parser.add_argument("--checkpoint",  default=str(CHECKPOINT))
    parser.add_argument("--out_tokens",  default=None, help="Save predicted glosses to .json")
    parser.add_argument("--out_notes",   default=None, help="Save notes to .md file")
    parser.add_argument("--use_ollama",  action="store_true")
    parser.add_argument("--use_llama",   action="store_true")
    parser.add_argument("--gguf_path",   default=None)
    parser.add_argument("--top_k",       type=int, default=5)
    args = parser.parse_args()

    if args.feature_path:
        glosses = predict_from_features(args.feature_path, args.checkpoint, args.top_k)
    else:
        glosses = predict_from_video(args.video_path, args.checkpoint)

    print("Predicted glosses:", glosses)

    notes = generate_notes(
        glosses,
        use_ollama = args.use_ollama,
        use_llama  = args.use_llama,
        gguf_path  = args.gguf_path,
    )
    print("\n=== Generated Notes ===\n")
    print(notes)

    if args.out_tokens:
        Path(args.out_tokens).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_tokens).write_text(json.dumps(glosses))

    if args.out_notes:
        Path(args.out_notes).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_notes).write_text(notes)
