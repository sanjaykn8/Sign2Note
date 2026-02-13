import torch
import numpy as np
import cv2
import requests
import mediapipe as mp
from pathlib import Path
from model import TemporalCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = Path("models/sign_recog/checkpoints/demo.pt")
MAX_LEN = 50

mp_hands = mp.solutions.hands


# -----------------------------
# Feature Extraction
# -----------------------------
def extract_keypoints_from_frame(results):
    def lm_to_array(hand_landmarks):
        if hand_landmarks is None:
            return np.zeros((21, 3), dtype=np.float32)
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            arr = lm_to_array(hand_landmarks)
            if idx == 0:
                left = arr
            elif idx == 1:
                right = arr

    return np.concatenate([left.flatten(), right.flatten()])


def extract_video_features(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_idx = 0

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 4 != 0:
                frame_idx += 1
                continue

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img)
            key = extract_keypoints_from_frame(results)
            frames.append(key)

            frame_idx += 1

    cap.release()

    if len(frames) == 0:
        return None

    x = np.stack(frames)

    if len(x) >= MAX_LEN:
        x = x[:MAX_LEN]
    else:
        pad = np.zeros((MAX_LEN - len(x), x.shape[1]))
        x = np.vstack([x, pad])

    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)


# -----------------------------
# Model Loading
# -----------------------------
def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    label2id = checkpoint["label2id"]
    id2label = {i: l for l, i in label2id.items()}

    input_dim = 126
    num_classes = len(label2id)

    model = TemporalCNN(input_dim, num_classes)
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()

    return model, id2label


# -----------------------------
# Prediction
# -----------------------------
def predict(video_path):
    model, id2label = load_model()

    x = extract_video_features(video_path)
    if x is None:
        return None

    x = x.to(DEVICE)

    with torch.no_grad():
        out = model(x)
        pred = out.argmax(1).item()

    gloss = id2label[pred]

    # For demo, return list
    return [gloss]


# -----------------------------
# Gloss → Structured Notes
# -----------------------------
def generate_structured_notes(gloss_list):

    if gloss_list is None or len(gloss_list) == 0:
        return "No recognizable signs detected."

    prompt = f"""
You are an academic note generation assistant.

Convert the following ASL gloss sequence into detailed, well-structured lecture notes.

Requirements:
- Use proper headings
- Use bullet points
- Use bold formatting for key terms
- Write coherent academic sentences
- Expand meaning intelligently

Gloss sequence:
{' '.join(gloss_list)}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]



# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    test_video = "backend/uploads/test.mp4"

    gloss_list = predict(test_video)

    print("Predicted Gloss Sequence:", gloss_list)

    notes = generate_structured_notes(gloss_list)

    print("\n==============================\n")
    print(notes)
