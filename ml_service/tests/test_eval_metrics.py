import numpy as np
import pytest

import eval_metrics as em


def _one_hot_probs(preds, num_classes):
    """Build a probability matrix that's 1.0 on the predicted class, 0
    elsewhere -- a simple deterministic fixture for top-k tests."""
    n = len(preds)
    probs = np.zeros((n, num_classes), dtype=np.float32)
    probs[np.arange(n), preds] = 1.0
    return probs


# ---------------------------------------------------------------------------
# top_k_accuracy
# ---------------------------------------------------------------------------

def test_top1_accuracy_all_correct():
    probs = _one_hot_probs([0, 1, 2], num_classes=3)
    y_true = np.array([0, 1, 2])
    assert em.top_k_accuracy(probs, y_true, k=1) == 1.0


def test_top1_accuracy_all_wrong():
    probs = _one_hot_probs([1, 2, 0], num_classes=3)
    y_true = np.array([0, 1, 2])
    assert em.top_k_accuracy(probs, y_true, k=1) == 0.0


def test_top3_accuracy_credits_true_label_anywhere_in_top3():
    # class 0 has highest prob, but true label (2) is still in the top 3
    # out of 5 classes.
    probs = np.array([[0.5, 0.3, 0.2, 0.0, 0.0]])
    y_true = np.array([2])
    assert em.top_k_accuracy(probs, y_true, k=3) == 1.0
    assert em.top_k_accuracy(probs, y_true, k=1) == 0.0


def test_top_k_accuracy_empty_input_returns_zero_not_nan_or_crash():
    probs = np.zeros((0, 5))
    y_true = np.array([])
    assert em.top_k_accuracy(probs, y_true, k=1) == 0.0


def test_top_k_larger_than_num_classes_is_clamped_not_a_crash():
    probs = _one_hot_probs([0], num_classes=3)
    y_true = np.array([0])
    assert em.top_k_accuracy(probs, y_true, k=100) == 1.0  # k clamped to 3, still finds it


# ---------------------------------------------------------------------------
# macro_f1 / weighted_f1
# ---------------------------------------------------------------------------

def test_macro_f1_is_one_for_perfect_predictions():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    assert em.macro_f1(y_true, y_pred, num_classes=3) == pytest.approx(1.0)


def test_macro_f1_penalizes_a_never_predicted_class_same_as_common_ones():
    # Class 2 is never predicted correctly (0 recall) -- macro-averaging
    # must not let the frequent classes' good scores hide this.
    y_true = [0, 0, 0, 1, 1, 1, 2, 2]
    y_pred = [0, 0, 0, 1, 1, 1, 0, 1]  # class 2 always misclassified
    macro = em.macro_f1(y_true, y_pred, num_classes=3)
    weighted = em.weighted_f1(y_true, y_pred, num_classes=3)
    assert macro < 1.0
    # weighted gives class 2 less weight (fewer samples), so it should
    # score higher than macro here -- demonstrating why macro is the
    # preferred metric for an imbalanced multi-class problem (RULE/section 51).
    assert weighted > macro


# ---------------------------------------------------------------------------
# per_class_metrics
# ---------------------------------------------------------------------------

def test_per_class_metrics_includes_every_class_even_with_zero_support():
    y_true = [0, 0, 1]
    y_pred = [0, 0, 1]
    # num_classes=3, but class 2 never appears in y_true or y_pred
    rows = em.per_class_metrics(y_true, y_pred, num_classes=3, id2label={0: "A", 1: "B", 2: "C"})
    assert len(rows) == 3
    class2 = next(r for r in rows if r["class_id"] == 2)
    assert class2["support"] == 0
    assert class2["label"] == "C"


def test_per_class_metrics_labels_default_to_stringified_id_without_id2label():
    rows = em.per_class_metrics([0, 1], [0, 1], num_classes=2)
    assert rows[0]["label"] == "0"
    assert rows[1]["label"] == "1"


def test_per_class_metrics_perfect_predictions_give_precision_recall_f1_of_one():
    rows = em.per_class_metrics([0, 1, 1], [0, 1, 1], num_classes=2)
    for r in rows:
        assert r["precision"] == pytest.approx(1.0)
        assert r["recall"] == pytest.approx(1.0)
        assert r["f1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# confusion_matrix
# ---------------------------------------------------------------------------

def test_confusion_matrix_diagonal_for_perfect_predictions():
    cm = em.confusion_matrix([0, 1, 2], [0, 1, 2], num_classes=3)
    assert cm.shape == (3, 3)
    assert np.array_equal(cm, np.eye(3, dtype=cm.dtype))


def test_confusion_matrix_off_diagonal_for_a_misclassification():
    cm = em.confusion_matrix([0, 0], [0, 1], num_classes=2)
    assert cm[0, 0] == 1  # one correct class-0
    assert cm[0, 1] == 1  # one class-0 misclassified as class-1


# ---------------------------------------------------------------------------
# class_distribution_stats
# ---------------------------------------------------------------------------

def test_class_distribution_stats_basic():
    stats = em.class_distribution_stats({"A": 20, "B": 20, "C": 20})
    assert stats == {"num_classes": 3, "min": 20, "max": 20, "mean": 20.0, "median": 20.0}


def test_class_distribution_stats_detects_imbalance():
    stats = em.class_distribution_stats({"A": 5, "B": 100})
    assert stats["min"] == 5
    assert stats["max"] == 100


def test_class_distribution_stats_empty_input_does_not_crash():
    stats = em.class_distribution_stats({})
    assert stats["num_classes"] == 0


# ---------------------------------------------------------------------------
# save_confusion_matrix_png -- actually renders a file, not just a mock
# ---------------------------------------------------------------------------

def test_save_confusion_matrix_png_writes_a_real_nonempty_file(tmp_path):
    cm = em.confusion_matrix([0, 1, 2, 0], [0, 1, 1, 0], num_classes=3)
    out_path = tmp_path / "cm.png"
    em.save_confusion_matrix_png(cm, out_path, labels=["A", "B", "C"])
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    with open(out_path, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"  # real PNG magic bytes, not a stub


def test_save_confusion_matrix_png_handles_large_vocab_without_labels(tmp_path):
    # 200 classes: labels should be suppressed (max_labels_shown default
    # 60), but the render must still succeed, not crash on huge tick lists.
    n = 200
    cm = np.eye(n, dtype=int)
    out_path = tmp_path / "cm_large.png"
    em.save_confusion_matrix_png(cm, out_path, labels=[str(i) for i in range(n)])
    assert out_path.exists()
    assert out_path.stat().st_size > 0
