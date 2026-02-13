import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path


class SignDataset(Dataset):
    def __init__(self, index_csv, feature_dir, max_len=50):
        self.df = pd.read_csv(index_csv)
        self.feature_dir = Path(feature_dir)
        self.max_len = max_len

        # Keep only rows where feature file exists
        valid_rows = []
        for _, row in self.df.iterrows():
            vid = row["video_id"]
            if (self.feature_dir / f"{vid}.npy").exists():
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        labels = sorted(self.df["label"].unique())
        self.label2id = {l: i for i, l in enumerate(labels)}
        self.id2label = {i: l for l, i in self.label2id.items()}

        print("Dataset size:", len(self.df))
        print("Classes:", len(self.label2id))

    def __len__(self):
        return len(self.df)

    def pad_or_trim(self, x):
        if len(x) >= self.max_len:
            return x[:self.max_len]
        pad = np.zeros((self.max_len - len(x), x.shape[1]))
        return np.vstack([x, pad])

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid = row["video_id"]
        label = row["label"]

        x = np.load(self.feature_dir / f"{vid}.npy")
        x = self.pad_or_trim(x)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.label2id[label])

        return x, y
