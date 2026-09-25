"""
Torch-free evaluation metrics -- pure numpy/scikit-learn, so this is
unit-testable without PyTorch installed and reusable from any future
tooling (notebook, CLI, etc.) without importing the training stack.

evaluate.py (which DOES need torch, to actually run the model) calls
these functions on the raw (probs, y_true) arrays it produces. RULE 3:
every metric here is a TEST metric when called from evaluate.py against
the official test split -- never rename or report it as validation
accuracy, and this module has no idea which split its inputs came from,
so that labeling discipline is the caller's job (see evaluate.py).
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import confusion_matrix as _sk_confusion_matrix
from sklearn.metrics import f1_score, precision_recall_fscore_support


def top_k_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """Fraction of samples where the true label is among the top-k
    predicted classes by probability. k=1 is ordinary accuracy. Returns
    0.0 (not NaN, not a crash) for an empty input."""
    if len(y_true) == 0:
        return 0.0
    k = max(1, min(k, probs.shape[1]))
    top_k_preds = np.argsort(-probs, axis=1)[:, :k]
    hits = (top_k_preds == np.asarray(y_true)[:, None]).any(axis=1)
    return float(hits.mean())


def macro_f1(y_true, y_pred, num_classes: int) -> float:
    labels = list(range(num_classes))
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def weighted_f1(y_true, y_pred, num_classes: int) -> float:
    labels = list(range(num_classes))
    return float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))


def per_class_metrics(y_true, y_pred, num_classes: int, id2label: Optional[Dict[int, str]] = None) -> List[dict]:
    """Precision/recall/F1/support for every class 0..num_classes-1, in
    class-id order -- includes classes with zero support in the evaluated
    set (support=0), rather than silently omitting them, so a caller can
    see which classes had no test examples at all."""
    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    id2label = id2label or {}
    return [
        {
            "class_id": i,
            "label": id2label.get(i, str(i)),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in labels
    ]


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    return _sk_confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def class_distribution_stats(counts: Dict[str, int]) -> dict:
    """counts: {label: sample_count}. Returns num_classes/min/max/mean/
    median -- project brief section 49's class-imbalance audit. An empty
    `counts` returns all-zero stats rather than raising, since "no data
    yet" is a valid (if uninteresting) state to report on."""
    if not counts:
        return {"num_classes": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    vals = np.array(list(counts.values()), dtype=float)
    return {
        "num_classes": len(counts),
        "min": int(vals.min()),
        "max": int(vals.max()),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
    }


def save_confusion_matrix_png(cm: np.ndarray, out_path, labels=None, max_labels_shown: int = 60) -> None:
    """Renders a confusion-matrix heatmap PNG via matplotlib (headless
    'Agg' backend -- no display needed). For large vocabularies (FDMSE-ISL
    can be 400-2000+ classes), per-cell tick labels become illegible and
    slow to render, so labels are only drawn when num_classes <=
    max_labels_shown; otherwise the heatmap image is still useful (a
    clean diagonal = good) just without per-class text."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = cm.shape[0]
    fig_size = max(6, min(24, n * 0.15))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({n} classes)")
    if labels is not None and n <= max_labels_shown:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
