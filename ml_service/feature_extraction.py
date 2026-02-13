import os
import json
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

VIDEOS_DIR = Path("data/wlasl/videos")
OUT_DIR = Path("data/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)
WLASL_JSON = Path("data/wlasl/WLASL_v0.3.json")

mp_hands = mp.solutions.hands


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


if __name__ == "__main__":
    MAX_VIDEOS = 60
    FRAME_SKIP = 4
    count = 0

    with open(WLASL_JSON, 'r') as f:
        data = json.load(f)

    for entry in tqdm(data):
        for inst in entry["instances"]:

            if count >= MAX_VIDEOS:
                print("Reached demo limit.")
                exit()

            vid = inst["video_id"]
            video_file = VIDEOS_DIR / f"{vid}.mp4"
            if not video_file.exists():
                continue

            out_file = OUT_DIR / f"{vid}.npy"
            if out_file.exists():
                continue

            cap = cv2.VideoCapture(str(video_file))
            frames = []
            frame_idx = 0

            with mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % FRAME_SKIP != 0:
                        frame_idx += 1
                        continue

                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(img)
                    key = extract_keypoints_from_frame(results)
                    frames.append(key)

                    frame_idx += 1

            cap.release()

            if len(frames) == 0:
                continue

            frames = np.stack(frames)
            np.save(out_file, frames)

            count += 1
            print(f"Processed {count}/{MAX_VIDEOS}")
