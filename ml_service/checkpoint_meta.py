"""
Shared, torch-free helpers for building and validating checkpoint
metadata. Kept in a separate module from model.py/train.py specifically
so it can be imported (and unit tested) without pulling in torch -- and so
api.py can run a model-compatibility check (RULE 26) without needing to
load the actual model weights first.

Used by:
  - train.py: build_checkpoint_metadata() assembles everything that goes
    into the saved .pt checkpoint dict.
  - infer.py / api.py: check_checkpoint_compatible() decides whether a
    checkpoint can be safely loaded against the schema/vocab currently in
    use, BEFORE attempting to load it -- see RULE 26: "do not crash
    mysteriously... return a clear message... then activate server
    fallback."
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from typing import Dict, List, Optional

import feature_schema as fs

SUPPORTED_ARCHITECTURES = ("temporal_cnn", "cnn_bilstm")
DEFAULT_ARCHITECTURE = "temporal_cnn"


def compute_vocab_hash(label2id: Dict[str, int]) -> str:
    """Stable hash of a label2id mapping, independent of dict insertion
    order -- two vocab.json files with the same labels/ids in a different
    order hash identically; any actual label or id difference changes the
    hash. Used to detect vocab drift between a checkpoint and whatever
    config/vocab.json is currently loaded (RULE 26)."""
    canonical = json.dumps(sorted(label2id.items()), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_checkpoint_metadata(
    *,
    model_state,
    label2id: Dict[str, int],
    input_dim: int,
    max_len: int,
    architecture: str = DEFAULT_ARCHITECTURE,
    best_val_accuracy: Optional[float] = None,
    training_config: Optional[dict] = None,
    stopped_early: bool = False,
    test_metrics: Optional[dict] = None,
) -> dict:
    """Assembles the full checkpoint dict train.py saves via torch.save().
    `model_state` is whatever the caller wants stored under the "model"
    key (a real torch state_dict in production; any picklable value in
    tests, since this function itself never touches torch) -- kept
    generic so this assembly logic is testable without torch installed.

    Per RULE 47: records architecture, feature_schema_version, a vocab
    hash, best validation accuracy, and the full training configuration
    used (not just a couple of hyperparameters) -- and per RULE 48, test
    metrics are recorded SEPARATELY (test_metrics, only ever populated by
    evaluate.py after the fact, never computed inline during training) so
    "validation accuracy" and "test accuracy" can never be confused for
    each other in the saved artifact (RULE 3).
    """
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}; expected one of {SUPPORTED_ARCHITECTURES}"
        )
    return {
        "model": model_state,
        "label2id": label2id,
        "input_dim": input_dim,
        "max_len": max_len,
        "architecture": architecture,
        "feature_schema_version": fs.FEATURE_SCHEMA_VERSION,
        "vocab_hash": compute_vocab_hash(label2id),
        "best_val_accuracy": best_val_accuracy,
        "stopped_early": stopped_early,
        "training_config": training_config or {},
        "test_metrics": test_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
    }


def check_checkpoint_compatible(
    ckpt: dict, current_vocab_label2id: Optional[Dict[str, int]] = None
) -> List[str]:
    """Returns a list of human-readable incompatibility reasons (empty
    list = compatible). Deliberately does NOT raise -- this is a pure
    check; the caller (api.py) decides what to do with an incompatible
    checkpoint (return a clear error and fall back to another recognition
    source, per RULE 26), rather than this function halting execution
    unilaterally.

    Checks, in order:
      1. feature_schema_version match.
      2. input_dim match (only flagged when the schema version claims to
         match but the dim doesn't -- an actually-different schema version
         having a different dim is already covered by check 1, no need to
         double-report the same root cause).
      3. vocab hash match against `current_vocab_label2id`, if given.
    """
    reasons: List[str] = []

    ckpt_schema = ckpt.get("feature_schema_version", "1.0")
    if ckpt_schema != fs.FEATURE_SCHEMA_VERSION:
        reasons.append(
            f"checkpoint uses feature_schema_version={ckpt_schema!r}, "
            f"current code expects {fs.FEATURE_SCHEMA_VERSION!r}"
        )
    else:
        input_dim = ckpt.get("input_dim")
        if input_dim is not None and input_dim != fs.FEATURE_DIM:
            reasons.append(
                f"checkpoint input_dim={input_dim}, current schema expects {fs.FEATURE_DIM}"
            )

    if current_vocab_label2id is not None:
        ckpt_hash = ckpt.get("vocab_hash")
        current_hash = compute_vocab_hash(current_vocab_label2id)
        if ckpt_hash and ckpt_hash != current_hash:
            reasons.append(
                f"checkpoint vocab_hash={ckpt_hash} does not match the currently "
                f"loaded vocabulary (hash={current_hash}) -- label2id may differ"
            )

    return reasons
