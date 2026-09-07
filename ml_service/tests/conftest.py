"""Shared pytest fixtures: everything here builds small, fast, synthetic
inputs so the test suite doesn't need a real dataset, a real trained
model, or GPU access to run."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def tmp_feature_dir(tmp_path):
    d = tmp_path / "features"
    d.mkdir()
    return d


@pytest.fixture
def synthetic_index_and_features(tmp_path):
    """Builds a tiny synthetic dataset: an index.csv with 3 classes, 4
    samples each, plus matching .npy keypoint files -- everything
    dataset.py's SignDataset needs, without any real video or model."""
    import csv

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    index_path = tmp_path / "index.csv"

    labels = ["DEFINITION", "EXAMPLE", "QUESTION"]
    rows = []
    rng = np.random.default_rng(0)
    for label_idx, label in enumerate(labels):
        for i in range(4):
            vid = f"{label.lower()}_{i}"
            arr = rng.random((20, 126)).astype(np.float32) + label_idx * 2
            np.save(feature_dir / f"{vid}.npy", arr)
            rows.append({"video_id": vid, "label": label, "split": "train"})

    with open(index_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "label", "split"])
        w.writeheader()
        w.writerows(rows)

    return index_path, feature_dir, labels
