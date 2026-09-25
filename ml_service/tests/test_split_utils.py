import pandas as pd
import pytest

from split_utils import resolve_test_indices, resolve_train_val_indices


def _write_index_csv(tmp_path, rows):
    """rows: list of dicts with at least a 'split' key (or whatever
    split_col the test uses)."""
    path = tmp_path / "index.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_uses_official_split_when_train_and_val_both_present(tmp_path):
    rows = (
        [{"video_id": f"train_{i}", "label": "A", "split": "train"} for i in range(6)]
        + [{"video_id": f"val_{i}", "label": "A", "split": "val"} for i in range(2)]
        + [{"video_id": f"test_{i}", "label": "A", "split": "test"} for i in range(2)]
    )
    csv = _write_index_csv(tmp_path, rows)
    train_idx, val_idx, used_official = resolve_train_val_indices(
        csv, split_col="split", force_random_split=False, val_split=0.15, seed=0, n=len(rows),
    )
    assert used_official is True
    assert len(train_idx) == 6
    assert len(val_idx) == 2


def test_test_rows_never_appear_in_train_or_val(tmp_path):
    rows = (
        [{"video_id": f"train_{i}", "split": "train"} for i in range(4)]
        + [{"video_id": f"val_{i}", "split": "val"} for i in range(2)]
        + [{"video_id": f"test_{i}", "split": "test"} for i in range(3)]
    )
    csv = _write_index_csv(tmp_path, rows)
    train_idx, val_idx, _ = resolve_train_val_indices(
        csv, split_col="split", force_random_split=False, val_split=0.15, seed=0, n=len(rows),
    )
    df = pd.read_csv(csv)
    test_positions = set(df.index[df["split"] == "test"].tolist())
    assert test_positions.isdisjoint(set(train_idx))
    assert test_positions.isdisjoint(set(val_idx))


def test_split_values_are_case_insensitive_and_validation_alias_accepted(tmp_path):
    rows = (
        [{"video_id": f"t{i}", "split": "TRAIN"} for i in range(4)]
        + [{"video_id": f"v{i}", "split": "Validation"} for i in range(2)]
    )
    csv = _write_index_csv(tmp_path, rows)
    train_idx, val_idx, used_official = resolve_train_val_indices(
        csv, split_col="split", force_random_split=False, val_split=0.15, seed=0, n=len(rows),
    )
    assert used_official is True
    assert len(train_idx) == 4
    assert len(val_idx) == 2


def test_falls_back_to_random_split_when_no_split_column(tmp_path):
    rows = [{"video_id": f"v{i}", "label": "A"} for i in range(20)]
    csv = _write_index_csv(tmp_path, rows)
    train_idx, val_idx, used_official = resolve_train_val_indices(
        csv, split_col="split", force_random_split=False, val_split=0.2, seed=0, n=len(rows),
    )
    assert used_official is False
    assert len(val_idx) == 4  # 20 * 0.2
    assert len(train_idx) == 16
    assert set(train_idx).isdisjoint(set(val_idx))
    assert sorted(train_idx + val_idx) == list(range(20))


def test_falls_back_to_random_split_when_split_column_has_no_usable_values(tmp_path):
    # A 'split' column exists but only contains e.g. "train" -- no val rows
    # to actually validate against, so this must not silently produce an
    # empty val set; it should fall back to a random split instead.
    rows = [{"video_id": f"v{i}", "split": "train"} for i in range(10)]
    csv = _write_index_csv(tmp_path, rows)
    train_idx, val_idx, used_official = resolve_train_val_indices(
        csv, split_col="split", force_random_split=False, val_split=0.3, seed=0, n=len(rows),
    )
    assert used_official is False
    assert len(val_idx) > 0


def test_force_random_split_ignores_an_available_official_split(tmp_path):
    rows = (
        [{"video_id": f"t{i}", "split": "train"} for i in range(8)]
        + [{"video_id": f"v{i}", "split": "val"} for i in range(2)]
    )
    csv = _write_index_csv(tmp_path, rows)
    train_idx, val_idx, used_official = resolve_train_val_indices(
        csv, split_col="split", force_random_split=True, val_split=0.2, seed=0, n=len(rows),
    )
    assert used_official is False
    assert len(val_idx) == 2  # 10 * 0.2, not the official split's 2-row val


def test_random_split_is_reproducible_given_the_same_seed(tmp_path):
    rows = [{"video_id": f"v{i}"} for i in range(30)]
    csv = _write_index_csv(tmp_path, rows)
    t1, v1, _ = resolve_train_val_indices(csv, "split", False, 0.2, seed=42, n=30)
    t2, v2, _ = resolve_train_val_indices(csv, "split", False, 0.2, seed=42, n=30)
    assert t1 == t2
    assert v1 == v2


def test_different_seeds_usually_give_different_splits(tmp_path):
    rows = [{"video_id": f"v{i}"} for i in range(30)]
    csv = _write_index_csv(tmp_path, rows)
    _, v1, _ = resolve_train_val_indices(csv, "split", False, 0.2, seed=1, n=30)
    _, v2, _ = resolve_train_val_indices(csv, "split", False, 0.2, seed=2, n=30)
    assert v1 != v2


def test_row_count_mismatch_raises_immediately_instead_of_silently_misaligning(tmp_path):
    rows = [{"video_id": f"v{i}"} for i in range(5)]
    csv = _write_index_csv(tmp_path, rows)
    with pytest.raises(AssertionError):
        resolve_train_val_indices(csv, "split", False, 0.2, seed=0, n=999)


# ---------------------------------------------------------------------------
# resolve_test_indices -- deliberately has NO fallback (RULE 11)
# ---------------------------------------------------------------------------

def test_resolve_test_indices_returns_only_test_rows(tmp_path):
    rows = (
        [{"video_id": f"t{i}", "split": "train"} for i in range(4)]
        + [{"video_id": f"v{i}", "split": "val"} for i in range(2)]
        + [{"video_id": f"x{i}", "split": "test"} for i in range(3)]
    )
    csv = _write_index_csv(tmp_path, rows)
    test_idx = resolve_test_indices(csv, "split", n=len(rows))
    assert len(test_idx) == 3
    df = pd.read_csv(csv)
    assert all(df.loc[i, "split"] == "test" for i in test_idx)


def test_resolve_test_indices_raises_if_no_split_column_at_all(tmp_path):
    rows = [{"video_id": f"v{i}"} for i in range(5)]
    csv = _write_index_csv(tmp_path, rows)
    with pytest.raises(ValueError, match="no official test split|no '.*' column"):
        resolve_test_indices(csv, "split", n=len(rows))


def test_resolve_test_indices_raises_if_split_column_has_no_test_rows(tmp_path):
    # Only train/val present -- must NOT silently substitute val as "test".
    rows = (
        [{"video_id": f"t{i}", "split": "train"} for i in range(4)]
        + [{"video_id": f"v{i}", "split": "val"} for i in range(2)]
    )
    csv = _write_index_csv(tmp_path, rows)
    with pytest.raises(ValueError, match="test"):
        resolve_test_indices(csv, "split", n=len(rows))
