"""SignDataset loading + label mapping -- dataset.py, using a synthetic
index.csv + .npy fixture (see conftest.py) instead of a real dataset."""
import json

import numpy as np
import pytest
import torch

from dataset import SignDataset
import feature_schema as fs


def test_dataset_loads_all_rows_with_matching_features(synthetic_index_and_features):
    index_path, feature_dir, labels = synthetic_index_and_features
    ds = SignDataset(str(index_path), str(feature_dir), max_len=20, augment=False)
    assert len(ds) == 4 * len(labels)


def test_dataset_skips_rows_with_missing_feature_file(tmp_path, synthetic_index_and_features):
    import csv
    index_path, feature_dir, labels = synthetic_index_and_features
    # append a row pointing at a .npy that doesn't exist
    with open(index_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ghost_video", "DEFINITION", "train"])
    ds = SignDataset(str(index_path), str(feature_dir), max_len=20, augment=False)
    assert len(ds) == 4 * len(labels)  # ghost row silently excluded, not crashed on


def test_dataset_getitem_shape_and_label_type(synthetic_index_and_features):
    index_path, feature_dir, labels = synthetic_index_and_features
    ds = SignDataset(str(index_path), str(feature_dir), max_len=20, augment=False)
    x, y = ds[0]
    assert x.shape == (20, fs.FEATURE_DIM)
    assert isinstance(y, torch.Tensor)
    assert y.dtype in (torch.int64, torch.long)


def test_dataset_raises_a_clear_error_on_wrong_dimension_features(tmp_path):
    """RULE 12: must not silently mix old (schema v1, 126-dim) feature
    files in with the current (schema v2, 285-dim) pipeline -- must fail
    clearly instead."""
    import csv

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    index_path = tmp_path / "index.csv"
    arr = np.random.default_rng(0).random((20, 126)).astype(np.float32)  # old v1 dim
    np.save(feature_dir / "old_video.npy", arr)
    with open(index_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "label", "split"])
        w.writeheader()
        w.writerow({"video_id": "old_video", "label": "DEFINITION", "split": "train"})

    ds = SignDataset(str(index_path), str(feature_dir), max_len=20, augment=False)
    with pytest.raises(fs.FeatureSchemaError, match="126"):
        ds[0]


def test_dataset_uses_provided_vocab_json_when_present(tmp_path, synthetic_index_and_features):
    index_path, feature_dir, labels = synthetic_index_and_features
    vocab_path = tmp_path / "vocab.json"
    fixed_mapping = {"QUESTION": 0, "EXAMPLE": 1, "DEFINITION": 2}
    vocab_path.write_text(json.dumps({"label2id": fixed_mapping}))

    ds = SignDataset(str(index_path), str(feature_dir), max_len=20,
                     labels_json=str(vocab_path), augment=False)
    # every sample's label id must come from the FIXED mapping, not one
    # freshly derived from this dataset's own (possibly differently-
    # ordered) unique label list
    for i in range(len(ds)):
        _, y = ds[i]
        row_label = ds.df.iloc[i]["label"]
        assert int(y) == fixed_mapping[row_label]
