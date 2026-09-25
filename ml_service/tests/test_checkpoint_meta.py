import pytest

import checkpoint_meta as cm
import feature_schema as fs


# ---------------------------------------------------------------------------
# compute_vocab_hash
# ---------------------------------------------------------------------------

def test_vocab_hash_is_order_independent():
    a = {"CAT": 0, "DOG": 1, "BIRD": 2}
    b = {"DOG": 1, "BIRD": 2, "CAT": 0}
    assert cm.compute_vocab_hash(a) == cm.compute_vocab_hash(b)


def test_vocab_hash_changes_if_a_label_or_id_changes():
    base = {"CAT": 0, "DOG": 1}
    changed_label = {"CAT": 0, "PUPPY": 1}
    changed_id = {"CAT": 0, "DOG": 2}
    h0 = cm.compute_vocab_hash(base)
    assert cm.compute_vocab_hash(changed_label) != h0
    assert cm.compute_vocab_hash(changed_id) != h0


def test_vocab_hash_is_deterministic_across_calls():
    v = {"A": 0, "B": 1, "C": 2}
    assert cm.compute_vocab_hash(v) == cm.compute_vocab_hash(v)


# ---------------------------------------------------------------------------
# build_checkpoint_metadata
# ---------------------------------------------------------------------------

def test_build_checkpoint_metadata_includes_all_required_fields():
    ckpt = cm.build_checkpoint_metadata(
        model_state={"fake": "state_dict"},
        label2id={"A": 0, "B": 1},
        input_dim=fs.FEATURE_DIM,
        max_len=64,
        architecture="temporal_cnn",
        best_val_accuracy=0.835,
        training_config={"lr": 5e-4, "epochs": 20},
    )
    for key in ("model", "label2id", "input_dim", "max_len", "architecture",
                "feature_schema_version", "vocab_hash", "best_val_accuracy",
                "training_config", "trained_at", "python_version"):
        assert key in ckpt, f"missing {key!r}"
    assert ckpt["feature_schema_version"] == fs.FEATURE_SCHEMA_VERSION
    assert ckpt["vocab_hash"] == cm.compute_vocab_hash({"A": 0, "B": 1})


def test_build_checkpoint_metadata_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="temporal_cnn"):
        cm.build_checkpoint_metadata(
            model_state={}, label2id={"A": 0}, input_dim=fs.FEATURE_DIM,
            max_len=64, architecture="transformer_xl_v99",
        )


def test_test_metrics_default_to_none_never_silently_equal_to_val_accuracy():
    # RULE 3: never call validation accuracy test accuracy. Confirms the
    # checkpoint doesn't fabricate a test_metrics value out of val accuracy.
    ckpt = cm.build_checkpoint_metadata(
        model_state={}, label2id={"A": 0}, input_dim=fs.FEATURE_DIM,
        max_len=64, best_val_accuracy=0.9,
    )
    assert ckpt["test_metrics"] is None


def test_build_checkpoint_metadata_defaults_architecture_to_temporal_cnn():
    ckpt = cm.build_checkpoint_metadata(
        model_state={}, label2id={"A": 0}, input_dim=fs.FEATURE_DIM, max_len=64,
    )
    assert ckpt["architecture"] == "temporal_cnn"


# ---------------------------------------------------------------------------
# check_checkpoint_compatible
# ---------------------------------------------------------------------------

def _valid_ckpt(**overrides):
    base = cm.build_checkpoint_metadata(
        model_state={}, label2id={"A": 0, "B": 1}, input_dim=fs.FEATURE_DIM, max_len=64,
    )
    base.update(overrides)
    return base


def test_a_freshly_built_checkpoint_is_compatible_with_itself():
    ckpt = _valid_ckpt()
    assert cm.check_checkpoint_compatible(ckpt) == []
    assert cm.check_checkpoint_compatible(ckpt, current_vocab_label2id={"A": 0, "B": 1}) == []


def test_old_schema_version_checkpoint_is_flagged():
    ckpt = _valid_ckpt(feature_schema_version="1.0", input_dim=126)
    reasons = cm.check_checkpoint_compatible(ckpt)
    assert len(reasons) == 1
    assert "1.0" in reasons[0]


def test_missing_schema_version_field_defaults_to_v1_and_is_flagged():
    # Old checkpoints saved before this field existed at all.
    ckpt = _valid_ckpt()
    del ckpt["feature_schema_version"]
    reasons = cm.check_checkpoint_compatible(ckpt)
    assert any("1.0" in r for r in reasons)


def test_matching_schema_but_wrong_input_dim_is_flagged_once_not_twice():
    ckpt = _valid_ckpt(input_dim=999)  # schema claims v2, but dim is wrong
    reasons = cm.check_checkpoint_compatible(ckpt)
    assert len(reasons) == 1
    assert "999" in reasons[0]


def test_vocab_mismatch_is_flagged_when_current_vocab_given():
    ckpt = _valid_ckpt()  # hashed against {"A": 0, "B": 1}
    reasons = cm.check_checkpoint_compatible(ckpt, current_vocab_label2id={"A": 0, "C": 2})
    assert any("vocab_hash" in r for r in reasons)


def test_vocab_check_skipped_when_current_vocab_not_provided():
    ckpt = _valid_ckpt()
    assert cm.check_checkpoint_compatible(ckpt, current_vocab_label2id=None) == []


def test_check_checkpoint_compatible_never_raises_only_returns_reasons():
    # Even a wildly malformed checkpoint dict should produce a reasons
    # list, not an exception -- api.py needs to be able to show this to
    # the user and fall back, not crash.
    reasons = cm.check_checkpoint_compatible({})
    assert isinstance(reasons, list)
    assert len(reasons) >= 1
