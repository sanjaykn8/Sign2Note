import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SignDataset(Dataset):
    def __init__(
        self,
        index_csv: str,
        feature_dir: str,
        max_len: int = 50,
        labels_json: Optional[str] = None,
        augment: bool = False,
    ):
        self.df = pd.read_csv(index_csv)
        self.feature_dir = Path(feature_dir)
        self.max_len = max_len
        self.augment = augment

        # ---------------------------------------------------------
        # Keep only samples for which features actually exist
        # ---------------------------------------------------------

        valid = self.df[
            self.df["video_id"].apply(
                lambda v: (
                    self.feature_dir / f"{v}.npy"
                ).exists()
            )
        ].copy()

        self.df = valid.reset_index(drop=True)

        if self.df.empty:
            raise RuntimeError(
                "No indexed videos have extracted feature files."
            )

        # ---------------------------------------------------------
        # Vocabulary
        # ---------------------------------------------------------

        if labels_json and Path(labels_json).exists():
            payload = json.loads(
                Path(labels_json).read_text(
                    encoding="utf-8"
                )
            )

            self.label2id = payload["label2id"]

        else:
            labels = sorted(
                self.df["label"].unique().tolist()
            )

            self.label2id = {
                label: i
                for i, label in enumerate(labels)
            }

        # ---------------------------------------------------------
        # Remove labels outside selected vocabulary
        # ---------------------------------------------------------

        self.df = self.df[
            self.df["label"].isin(self.label2id)
        ].reset_index(drop=True)

        if self.df.empty:
            raise RuntimeError(
                "Selected vocabulary has no matching feature rows."
            )

        self.id2label = {
            i: label
            for label, i in self.label2id.items()
        }

        print(f"Dataset size: {len(self.df)}")
        print(f"Classes: {len(self.label2id)}")

    def __len__(self):
        return len(self.df)

    # =============================================================
    # Temporal length handling
    # =============================================================

    def pad_or_trim(self, x: np.ndarray) -> np.ndarray:

        x = np.asarray(
            x,
            dtype=np.float32
        )

        T = len(x)

        if T >= self.max_len:

            # During training, use a random temporal window.
            if self.augment:

                max_start = T - self.max_len

                if max_start > 0:
                    start = np.random.randint(
                        0,
                        max_start + 1
                    )

                    return x[
                        start:start + self.max_len
                    ]

            # Validation / fallback:
            # deterministic beginning crop.
            return x[:self.max_len]

        # ---------------------------------------------------------
        # Padding
        # ---------------------------------------------------------

        pad = np.zeros(
            (
                self.max_len - T,
                x.shape[1]
            ),
            dtype=np.float32
        )

        return np.vstack([x, pad])

    # =============================================================
    # Gaussian keypoint jitter
    # =============================================================

    def add_noise(self, x):

        if np.random.rand() < 0.50:

            noise = np.random.normal(
                loc=0.0,
                scale=0.005,
                size=x.shape
            ).astype(np.float32)

            x = x + noise

        return x

    # =============================================================
    # Random frame dropout
    # =============================================================

    def frame_dropout(self, x):

        if np.random.rand() >= 0.25:
            return x

        T = len(x)

        if T <= 8:
            return x

        drop_ratio = np.random.uniform(
            0.03,
            0.10
        )

        num_drop = max(
            1,
            int(T * drop_ratio)
        )

        drop_indices = np.random.choice(
            T,
            size=num_drop,
            replace=False
        )

        keep = np.ones(
            T,
            dtype=bool
        )

        keep[drop_indices] = False

        return x[keep]

    # =============================================================
    # Temporal speed augmentation
    # =============================================================

    def change_speed(self, x):

        if np.random.rand() >= 0.35:
            return x

        T, D = x.shape

        speed = np.random.uniform(
            0.80,
            1.20
        )

        new_T = max(
            2,
            int(T / speed)
        )

        old_positions = np.linspace(
            0,
            T - 1,
            T
        )

        new_positions = np.linspace(
            0,
            T - 1,
            new_T
        )

        result = np.empty(
            (new_T, D),
            dtype=np.float32
        )

        for d in range(D):

            result[:, d] = np.interp(
                new_positions,
                old_positions,
                x[:, d]
            )

        return result

    # =============================================================
    # Random temporal crop
    # =============================================================

    def temporal_crop(self, x):

        if np.random.rand() >= 0.30:
            return x

        T = len(x)

        if T <= 16:
            return x

        keep_ratio = np.random.uniform(
            0.85,
            1.0
        )

        crop_len = max(
            8,
            int(T * keep_ratio)
        )

        if crop_len >= T:
            return x

        start = np.random.randint(
            0,
            T - crop_len + 1
        )

        return x[
            start:start + crop_len
        ]

    # =============================================================
    # Apply all augmentations
    # =============================================================

    def augment_sequence(self, x):

        # Order matters:
        #
        # speed → crop → dropout → noise
        #
        # Then final padding/trimming happens afterward.

        x = self.change_speed(x)

        x = self.temporal_crop(x)

        x = self.frame_dropout(x)

        x = self.add_noise(x)

        return x

    # =============================================================
    # Main sample loader
    # =============================================================

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        feature_path = (
            self.feature_dir
            / f"{row['video_id']}.npy"
        )

        x = np.load(feature_path).astype(
            np.float32
        )

        # ---------------------------------------------------------
        # IMPORTANT:
        # augmentation happens BEFORE padding/trim
        # ---------------------------------------------------------

        if self.augment:
            x = self.augment_sequence(x)

        x = self.pad_or_trim(x)

        # ---------------------------------------------------------
        # Per-video normalization
        # ---------------------------------------------------------

        mean = x.mean(
            axis=0,
            keepdims=True
        )

        std = x.std(
            axis=0,
            keepdims=True
        ) + 1e-5

        x = (x - mean) / std

        label = self.label2id[
            row["label"]
        ]

        return (
            torch.from_numpy(x),
            torch.tensor(
                label,
                dtype=torch.long
            )
        )