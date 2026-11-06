"""Train/val split resolution -- kept torch-free (numpy only, no
torch.Generator/random_split) specifically so it's unit-testable without
PyTorch installed, and importable by both train.py and a future
evaluate.py-style script without pulling in the training loop."""

from typing import List, Tuple

import numpy as np
import pandas as pd


def resolve_train_val_indices(
    index_csv: str,
    split_col: str,
    force_random_split: bool,
    val_split: float,
    seed: int,
    n: int,
) -> Tuple[List[int], List[int], bool]:
    """Decide which row indices go to train vs. val. Prefers the official
    split column (RULE 16: use official splits). 'test' rows are simply
    never selected into either bucket here -- they're reserved entirely
    for a separate evaluate.py run, never mixed into training (project
    brief section 11).

    Falls back to a random split, LOUDLY (printed, not silent), if no
    usable official split column exists -- see README/FEATURE_SCHEMA.md
    "Limitations": we don't want an unearned claim that the official split
    was used when it wasn't.

    Returns (train_idx, val_idx, used_official_split).
    """
    df = pd.read_csv(index_csv)
    assert len(df) == n, (
        f"index_csv row count ({len(df)}) doesn't match dataset length ({n}) -- "
        f"the caller must be reading the same file this function just read."
    )
    return resolve_train_val_indices_from_df(df, split_col, force_random_split, val_split, seed)


def resolve_train_val_indices_from_df(
    df: pd.DataFrame,
    split_col: str,
    force_random_split: bool,
    val_split: float,
    seed: int,
) -> Tuple[List[int], List[int], bool]:
    """Same as resolve_train_val_indices(), but on the dataset's OWN rows
    (SignDataset.df). Positions returned are positions in `df`, which is
    what torch Subset needs. This is what train.py uses: SignDataset drops
    index rows that have no feature file or are outside the vocabulary,
    so positions in a fresh read of index_csv would not line up."""
    df = df.reset_index(drop=True)
    n = len(df)
    if not force_random_split and split_col in df.columns:
        values = df[split_col].astype(str).str.lower()
        train_idx = df.index[values == "train"].tolist()
        val_idx = df.index[values.isin(["val", "validation"])].tolist()
        test_count = int(values.isin(["test"]).sum())
        if train_idx and val_idx:
            print(f"[split] Using OFFICIAL split from '{split_col}' column: "
                  f"train={len(train_idx)} val={len(val_idx)} "
                  f"(test={test_count} rows excluded here, reserved for evaluate.py)")
            return train_idx, val_idx, True
        print(f"[split] WARNING: index_csv has a '{split_col}' column but it doesn't contain "
              f"usable 'train'/'val' rows (found values: {sorted(values.unique())[:10]}) -- "
              f"falling back to a random split.")

    print(f"[split] WARNING: NOT using an official train/val split -- "
          f"randomly splitting with val_split={val_split}. This is fine for a quick "
          f"experiment, but do not report this run's val_acc as if it came from the "
          f"official FDMSE-ISL split (RULE 16).")
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(n)
    val_size = max(1, int(n * val_split))
    val_idx = sorted(int(i) for i in permuted[:val_size])
    train_idx = sorted(int(i) for i in permuted[val_size:])
    return train_idx, val_idx, False


def resolve_test_indices(index_csv: str, split_col: str, n: int) -> List[int]:
    """Returns the row indices whose split_col == 'test'. Used exclusively
    by evaluate.py -- deliberately has NO random-split fallback, unlike
    resolve_train_val_indices(): if there's no official test split, the
    correct behavior is to fail loudly and say so, not to quietly
    re-purpose train or val rows as a stand-in "test" set (that would be
    exactly the test/train leakage RULE 11 exists to prevent).

    Raises ValueError if index_csv has no split_col, or no rows are
    labeled 'test'.
    """
    df = pd.read_csv(index_csv)
    assert len(df) == n, (
        f"index_csv row count ({len(df)}) doesn't match dataset length ({n})."
    )
    return resolve_split_indices_from_df(df, split_col, "test", source=index_csv)


def resolve_split_indices_from_df(df: pd.DataFrame, split_col: str, split_name: str,
                                  source: str = "dataset") -> List[int]:
    """Positions (in `df`) of rows whose split_col == split_name ('test' or
    'val'/'validation'). No fallback: a missing split is an error -- never
    substitute another split's rows."""
    df = df.reset_index(drop=True)
    index_csv = source
    if split_col not in df.columns:
        raise ValueError(
            f"{index_csv} has no '{split_col}' column -- there is no official test split "
            f"to evaluate against. Re-run build_index.py against a metadata CSV that "
            f"includes a 'split' column, or pass a different --split_col."
        )
    values = df[split_col].astype(str).str.lower()
    wanted = {"val", "validation"} if split_name in ("val", "validation") else {split_name}
    test_idx = df.index[values.isin(wanted)].tolist()
    if not test_idx:
        raise ValueError(
            f"{index_csv}'s '{split_col}' column has no rows labeled {split_name!r} "
            f"(found values: {sorted(values.unique())[:10]}) -- nothing to evaluate. "
            f"Do not substitute val or train rows for this; that would defeat the "
            f"point of a held-out test set (RULE 11)."
        )
    return test_idx
