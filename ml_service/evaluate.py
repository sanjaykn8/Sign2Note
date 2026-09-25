"""
evaluate.py -- dedicated official TEST-set evaluation.

RULE 3: everything this script reports is a TEST metric, computed only
against index.csv rows labeled 'test' (via split_utils.resolve_test_indices
-- which has no fallback and no silent substitution of val/train rows;
see RULE 11). Never confuse this script's output with train.py's val_acc,
which is a validation metric used for early stopping / checkpoint
selection during training, not a final reportable result.

Loads a checkpoint, runs it over the official test split, and writes both
a machine-readable results.json and a human-readable results.txt (project
brief section 48), plus an optional confusion-matrix PNG. Never modifies
the checkpoint file itself and never re-selects which samples count as
"test" based on the results (project brief: "Do not modify the test set
based on results").

Usage:
    python3 evaluate.py --checkpoint models/sign_recog_v2/checkpoints/best.pt \\
        --index_csv data/index.csv --feature_dir data/features_v2 \\
        --confusion_matrix_png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import eval_metrics as em
from checkpoint_meta import check_checkpoint_compatible
from dataset import SignDataset
from model import build_model
from split_utils import resolve_test_indices


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint (e.g. best.pt).")
    p.add_argument("--index_csv", default="data/index.csv")
    p.add_argument("--feature_dir", default="data/features_v2")
    p.add_argument("--vocab_json", default="config/vocab.json",
                    help="Currently configured vocabulary, used only to check for drift "
                         "against the checkpoint's own vocab_hash (RULE 26) -- not used to "
                         "override the checkpoint's actual label2id.")
    p.add_argument("--split_col", default="split")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out_dir", default=None,
                    help="Where to write results.json / results.txt / confusion_matrix.png. "
                         "Defaults to <checkpoint's directory>/eval_results.")
    p.add_argument("--confusion_matrix_png", action="store_true",
                    help="Also render a confusion-matrix heatmap. Opt-in: can be slow and a "
                         "large image for big vocabularies (FDMSE-ISL can be 400-2000+ classes).")
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    current_vocab = None
    vocab_path = Path(args.vocab_json)
    if vocab_path.exists():
        current_vocab = json.loads(vocab_path.read_text()).get("label2id")
    incompatibilities = check_checkpoint_compatible(ckpt, current_vocab_label2id=current_vocab)
    if incompatibilities:
        print("[evaluate] WARNING -- checkpoint compatibility issues found:")
        for reason in incompatibilities:
            print(f"  - {reason}")
        print("[evaluate] Continuing anyway (informational for a standalone eval run, not a "
              "hard stop), but treat these results with caution until resolved.")

    label2id = ckpt["label2id"]
    id2label = {v: k for k, v in label2id.items()}
    num_classes = len(label2id)
    input_dim = ckpt["input_dim"]
    max_len = ckpt["max_len"]
    architecture = ckpt.get("architecture", "temporal_cnn")
    feature_schema_version = ckpt.get("feature_schema_version", "1.0")

    # augment=False: evaluation must see exactly what the model would see
    # in production, not a randomly-cropped/jittered training view.
    dataset = SignDataset(
        args.index_csv, args.feature_dir, max_len=max_len,
        labels_json=args.vocab_json, augment=False,
    )
    n = len(dataset)
    test_idx = resolve_test_indices(args.index_csv, args.split_col, n)
    test_ds = Subset(dataset, test_idx)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(architecture, input_dim, num_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_probs, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_true.append(y.numpy())

    if not all_probs:
        raise RuntimeError(
            f"resolve_test_indices found {len(test_idx)} test rows, but the DataLoader "
            f"produced zero batches -- check --feature_dir actually has .npy files for "
            f"those rows (a video with a metadata row but no extracted features would be "
            f"silently dropped by SignDataset; check its logs/warnings)."
        )

    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    y_pred = probs.argmax(axis=1)

    results = {
        "checkpoint": str(checkpoint_path),
        "architecture": architecture,
        "feature_schema_version": feature_schema_version,
        "num_classes": num_classes,
        "num_evaluated_samples": int(len(y_true)),
        "top1_accuracy": em.top_k_accuracy(probs, y_true, 1),
        "top3_accuracy": em.top_k_accuracy(probs, y_true, 3),
        "top5_accuracy": em.top_k_accuracy(probs, y_true, 5),
        "macro_f1": em.macro_f1(y_true, y_pred, num_classes),
        "weighted_f1": em.weighted_f1(y_true, y_pred, num_classes),
        "per_class": em.per_class_metrics(y_true, y_pred, num_classes, id2label),
        "checkpoint_compatibility_warnings": incompatibilities,
    }

    out_dir = Path(args.out_dir) if args.out_dir else checkpoint_path.parent / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    lines = [
        f"TEST SET EVALUATION -- {checkpoint_path}",
        f"(official '{args.split_col}'==test rows only -- this is NOT validation accuracy)",
        "",
        f"Architecture:        {architecture}",
        f"Feature schema:      v{feature_schema_version}",
        f"Classes:             {num_classes}",
        f"Evaluated samples:   {results['num_evaluated_samples']}",
        "",
        f"Top-1 accuracy:      {results['top1_accuracy']:.4f}",
        f"Top-3 accuracy:      {results['top3_accuracy']:.4f}",
        f"Top-5 accuracy:      {results['top5_accuracy']:.4f}",
        f"Macro F1:            {results['macro_f1']:.4f}",
        f"Weighted F1:         {results['weighted_f1']:.4f}",
        "",
    ]
    if incompatibilities:
        lines.append("Checkpoint compatibility warnings:")
        lines += [f"  - {r}" for r in incompatibilities]
        lines.append("")
    lines.append("Per-class metrics (class_id  label  precision  recall  f1  support):")
    for row in results["per_class"]:
        lines.append(
            f"  {row['class_id']:>5}  {row['label']:<30} "
            f"P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f} n={row['support']}"
        )
    (out_dir / "results.txt").write_text("\n".join(lines))

    if args.confusion_matrix_png:
        cm = em.confusion_matrix(y_true, y_pred, num_classes)
        labels = [id2label.get(i, str(i)) for i in range(num_classes)]
        em.save_confusion_matrix_png(cm, out_dir / "confusion_matrix.png", labels=labels)

    print(f"[evaluate] Wrote {out_dir}/results.json, results.txt"
          f"{', confusion_matrix.png' if args.confusion_matrix_png else ''}")
    print(f"[evaluate] TEST Top-1={results['top1_accuracy']:.4f}  "
          f"Top-3={results['top3_accuracy']:.4f}  Top-5={results['top5_accuracy']:.4f}  "
          f"MacroF1={results['macro_f1']:.4f}  n={results['num_evaluated_samples']}")


if __name__ == "__main__":
    main()
