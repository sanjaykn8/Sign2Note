"""
evaluate.py -- reproducible evaluation of a checkpoint on the OFFICIAL
validation and/or test split.

Validation and test results are computed and written SEPARATELY and are
always labeled with the split they came from (RULE 3: never report a
validation number as a test number). Split membership comes from the
index's `split` column on the rows the dataset actually serves (no random
fallback, no substitution of one split for another -- RULE 11).

The checkpoint's OWN label2id and feature schema are used (not whatever
config/vocab.json currently holds), so an evaluation can't silently run
a model against a different vocabulary or a different feature layout.

Per split, writes to <out_dir>/<split>/:
  results.json          Top-1/3/5, macro/weighted F1, per-class P/R/F1,
                        class sample counts, run metadata
  results.txt           human-readable summary + per-class table
  confusion_matrix.npz  full (num_classes x num_classes) matrix, compressed
  confusion_matrix.png  only with --confusion_matrix_png (large vocabularies
                        make a huge, unreadable image)

Usage:
    python evaluate.py --checkpoint models/sign_recog_v2/checkpoints/best.pt \\
        --index_csv data/index_v2.csv --feature_dir data/features_v2 --splits val,test
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import eval_metrics as em
import feature_schema as fs
from checkpoint_meta import check_checkpoint_compatible, check_checkpoint_self_consistent
from dataset import SignDataset
from model import build_model
from split_utils import resolve_split_indices_from_df


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint (e.g. best.pt).")
    p.add_argument("--index_csv", default="data/index.csv")
    p.add_argument("--feature_dir", default="data/features_v2")
    p.add_argument("--vocab_json", default="config/vocab.json",
                    help="Currently configured vocabulary, used only to REPORT drift against the "
                         "checkpoint's own vocab_hash -- the checkpoint's label2id is what is used.")
    p.add_argument("--split_col", default="split")
    p.add_argument("--splits", default="test",
                    help="Comma-separated official splits to evaluate, e.g. 'val,test'. Each is "
                         "evaluated and written separately.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--out_dir", default=None,
                    help="Defaults to <checkpoint's directory>/eval_results.")
    p.add_argument("--confusion_matrix_png", action="store_true",
                    help="Also render a confusion-matrix heatmap (opt-in; slow/huge for 2000 classes).")
    return p.parse_args()


def evaluate_split(model, dataset, positions, device, batch_size):
    loader = DataLoader(Subset(dataset, positions), batch_size=batch_size, shuffle=False)
    all_probs, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_true.append(y.numpy())
    if not all_probs:
        raise RuntimeError("no samples were evaluated -- check --feature_dir has .npy files for this split")
    return np.concatenate(all_probs), np.concatenate(all_true)


def summarize(split, probs, y_true, num_classes, id2label, header):
    y_pred = probs.argmax(axis=1)
    counts = Counter(int(c) for c in y_true)
    per_class = em.per_class_metrics(y_true, y_pred, num_classes, id2label)
    present = [r for r in per_class if r["support"] > 0]
    return {
        **header,
        "split": split,
        "metric_kind": "TEST" if split == "test" else "VALIDATION",
        "num_evaluated_samples": int(len(y_true)),
        "num_classes_in_model": num_classes,
        "num_classes_present_in_split": len(counts),
        "top1_accuracy": em.top_k_accuracy(probs, y_true, 1),
        "top3_accuracy": em.top_k_accuracy(probs, y_true, 3),
        "top5_accuracy": em.top_k_accuracy(probs, y_true, 5),
        # Macro-F1 over the classes that actually occur in this split (the
        # model's vocabulary may contain classes with zero support here).
        "macro_f1": float(np.mean([r["f1"] for r in present])) if present else 0.0,
        "macro_f1_all_model_classes": em.macro_f1(y_true, y_pred, num_classes),
        "weighted_f1": em.weighted_f1(y_true, y_pred, num_classes),
        "class_sample_counts": em.class_distribution_stats(
            {id2label.get(k, str(k)): v for k, v in counts.items()}),
        "per_class": per_class,
    }


def write_outputs(out_dir, results, cm, id2label, num_classes, png):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "confusion_matrix.npz", confusion_matrix=cm,
                        labels=np.array([id2label.get(i, str(i)) for i in range(num_classes)]))
    kind = results["metric_kind"]
    lines = [
        f"{kind} SET EVALUATION ({results['split']!r} split) -- {results['checkpoint']}",
        "(official split from the index's split column; "
        + ("this is NOT validation accuracy)" if kind == "TEST" else "this is NOT a test result)"),
        "",
        f"Architecture:        {results['architecture']}",
        f"Feature schema:      v{results['feature_schema_version']}",
        f"Model classes:       {num_classes}",
        f"Classes in split:    {results['num_classes_present_in_split']}",
        f"Evaluated samples:   {results['num_evaluated_samples']}",
        "",
        f"Top-1 accuracy:      {results['top1_accuracy']:.4f}",
        f"Top-3 accuracy:      {results['top3_accuracy']:.4f}",
        f"Top-5 accuracy:      {results['top5_accuracy']:.4f}",
        f"Macro F1:            {results['macro_f1']:.4f}  (classes present in split)",
        f"Weighted F1:         {results['weighted_f1']:.4f}",
        "",
    ]
    for r in results.get("warnings", []):
        lines.append(f"WARNING: {r}")
    lines.append("Per-class metrics (class_id  label  precision  recall  f1  support):")
    for row in results["per_class"]:
        if row["support"] == 0:
            continue
        lines.append(
            f"  {row['class_id']:>5}  {row['label']:<30} "
            f"P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f} n={row['support']}"
        )
    (out_dir / "results.txt").write_text("\n".join(lines), encoding="utf-8")
    if png:
        labels = [id2label.get(i, str(i)) for i in range(num_classes)]
        em.save_confusion_matrix_png(cm, out_dir / "confusion_matrix.png", labels=labels)


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    training_config = ckpt.get("training_config") or {}
    groups = training_config.get("feature_groups", "all")
    feature_columns = None
    if groups and groups != "all":
        # Ablation checkpoint (e.g. hands-only slice of schema-v2 features):
        # rebuild exactly the same column subset that training used.
        spans = {name: (a, b) for name, a, b in fs.FEATURE_GROUPS}
        feature_columns = [c for g in groups.split(",") for c in range(*spans[g.strip()])]
        if len(feature_columns) != int(ckpt["input_dim"]):
            raise SystemExit("[evaluate] feature_groups do not match the checkpoint's input_dim")
    else:
        problems = check_checkpoint_self_consistent(ckpt)
        if problems:
            raise SystemExit(f"[evaluate] checkpoint is not self-consistent: {problems}")

    warnings = []
    vocab_path = Path(args.vocab_json)
    if vocab_path.exists():
        current = json.loads(vocab_path.read_text(encoding="utf-8")).get("label2id")
        warnings += [w for w in check_checkpoint_compatible(ckpt, current_vocab_label2id=current)
                     if "vocab_hash" in w]

    label2id = ckpt["label2id"]
    id2label = {int(v): k for k, v in label2id.items()}
    num_classes = len(label2id)
    input_dim = int(ckpt["input_dim"])
    schema = str(ckpt.get("feature_schema_version", fs.LEGACY_SCHEMA_VERSION))
    architecture = ckpt.get("architecture", "temporal_cnn")
    if not training_config.get("used_official_split", False):
        warnings.append(
            "checkpoint was NOT trained with the official train/val split (random split or "
            "unknown) -- its training data may include signers/samples from the evaluated split, "
            "so these numbers can be optimistic and are not a clean held-out result."
        )

    # augment=False: evaluation sees exactly what the model sees in production.
    dataset = SignDataset(
        args.index_csv, args.feature_dir, max_len=int(ckpt["max_len"]),
        label2id=label2id, augment=False,
        expected_dim=fs.FEATURE_DIM if feature_columns is not None else input_dim,
        feature_columns=feature_columns,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(architecture, input_dim, num_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    header = {
        "checkpoint": str(checkpoint_path),
        "model_version": ckpt.get("model_version"),
        "architecture": architecture,
        "feature_schema_version": schema,
        "feature_groups": groups,
        "index_csv": args.index_csv,
        "feature_dir": args.feature_dir,
        "best_val_accuracy_recorded_in_checkpoint": ckpt.get("best_val_accuracy"),
        "trained_with_official_split": training_config.get("used_official_split"),
        "num_index_rows_dropped": dataset.num_dropped_rows,
        "warnings": warnings,
    }
    for w in warnings:
        print(f"[evaluate] WARNING: {w}")

    out_root = Path(args.out_dir) if args.out_dir else checkpoint_path.parent / "eval_results"
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        positions = resolve_split_indices_from_df(dataset.df, args.split_col, split, source=args.index_csv)
        probs, y_true = evaluate_split(model, dataset, positions, device, args.batch_size)
        results = summarize(split, probs, y_true, num_classes, id2label, header)
        cm = em.confusion_matrix(y_true, probs.argmax(axis=1), num_classes)
        write_outputs(out_root / split, results, cm, id2label, num_classes, args.confusion_matrix_png)
        print(f"[evaluate] {results['metric_kind']} ({split}): Top-1={results['top1_accuracy']:.4f}  "
              f"Top-3={results['top3_accuracy']:.4f}  Top-5={results['top5_accuracy']:.4f}  "
              f"MacroF1={results['macro_f1']:.4f}  n={results['num_evaluated_samples']} "
              f"-> {out_root / split}")


if __name__ == "__main__":
    main()
