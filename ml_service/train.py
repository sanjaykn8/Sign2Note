"""Train the Sign2Notes recognition model on extracted keypoints.

Default architecture is TemporalCNN (unchanged, ~83.5%-class-era baseline
-- see README.md's experiment results table). --architecture cnn_bilstm
is an OPTIONAL experiment (project brief section 9); it is not
automatically preferred just because it's more complex -- see
model_selection notes in README.md for how to decide between them.
"""
import argparse
import copy
import json
import time
from multiprocessing import freeze_support
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autocast
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from dataset import SignDataset
from model import build_model
from checkpoint_meta import SUPPORTED_ARCHITECTURES, DEFAULT_ARCHITECTURE, build_checkpoint_metadata
from split_utils import resolve_train_val_indices_from_df
import feature_schema as fs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index_csv", default="data/index.csv")
    p.add_argument("--feature_dir", default="data/features_v2")
    p.add_argument("--out_dir", default="models/sign_recog_v2/checkpoints")
    p.add_argument("--architecture", choices=SUPPORTED_ARCHITECTURES, default=DEFAULT_ARCHITECTURE,
                    help="Model architecture. Default (temporal_cnn) is the evidence-backed "
                         "baseline; cnn_bilstm is an optional experiment (RULE 2).")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--vocab_json", default="config/vocab.json",
                    help="label2id mapping to train with. Use the vocab written by build_index.py "
                         "for the same index_csv (e.g. a smaller --max_classes vocabulary for a quick "
                         "experiment). Rows whose label is not in it are dropped.")
    p.add_argument("--feature_schema_version", default=fs.FEATURE_SCHEMA_VERSION,
                    choices=sorted(fs.SCHEMA_DIMS),
                    help="Schema of the features in --feature_dir (2.0 = hand+body+face, "
                         "1.0 = legacy hand-only). Recorded in the checkpoint; dims are validated.")
    p.add_argument("--feature_groups", default=None,
                    help="EXPERIMENTS ONLY: comma-separated schema-v2 groups to keep, e.g. "
                         "'left_hand,right_hand' for the hand-only ablation. The resulting checkpoint "
                         "records the groups and is deliberately NOT servable by api.py (its input_dim "
                         "matches no schema), so it can't be mistaken for a production model.")
    p.add_argument("--model_version", default=None,
                    help="Human-readable model version recorded in the checkpoint "
                         "(default: name of the out_dir's parent directory).")
    p.add_argument("--preload", action="store_true",
                    help="Load all feature files into RAM once (recommended for CPU training).")
    p.add_argument("--threads", type=int, default=0, help="torch.set_num_threads (0 = default).")
    # --- official split (RULE 16) ---
    p.add_argument("--split_col", default="split",
                    help="Column in index_csv holding official train/val/test labels "
                         "(written by build_index.py from the FDMSE metadata CSV).")
    p.add_argument("--force_random_split", action="store_true",
                    help="Ignore an official split column even if present, and use a random "
                         "--val_split instead. For quick experiments only -- the final "
                         "evaluation should always use the official split (RULE 16).")
    p.add_argument("--val_split", type=float, default=0.15,
                    help="Fraction held out for validation when no usable official split "
                         "column exists (or --force_random_split is set).")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    # --- regularization / schedule knobs ---
    p.add_argument("--label_smoothing", type=float, default=0.1,
                    help="CrossEntropyLoss label smoothing (0 disables it).")
    p.add_argument("--warmup_epochs", type=int, default=0,
                    help="Linear LR warmup epochs before cosine decay.")
    p.add_argument("--min_lr", type=float, default=1e-5,
                    help="Floor LR for the cosine schedule.")
    p.add_argument("--patience", type=int, default=10,
                    help="Early-stopping patience, in epochs with no val_acc improvement.")
    p.add_argument("--min_delta", type=float, default=1e-4,
                    help="Minimum val_acc improvement to reset patience.")
    p.add_argument("--grad_clip", type=float, default=0.95,
                    help="Max grad norm for clipping (<=0 disables it).")
    # --- augmentation toggle ---
    p.add_argument("--no_augment", action="store_true",
                    help="Disable training-time augmentation (val is never augmented either way).")
    return p.parse_args()


def export_onnx(model, dummy, onnx_path):
    """TorchScript-based ONNX export (input "keypoints" -> output "logits",
    dynamic batch, opset 17) -- the same exporter that produced the existing
    sign_recog.onnx. torch>=2.9 defaults to the dynamo exporter, which
    needs the extra `onnxscript` package and crashed training at the very
    end; dynamo=False keeps the established export path."""
    kwargs = dict(
        input_names=["keypoints"], output_names=["logits"],
        dynamic_axes={"keypoints": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    try:
        torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **kwargs)
    except TypeError:  # torch < 2.5 has no `dynamo` argument
        torch.onnx.export(model, dummy, str(onnx_path), **kwargs)


def make_scheduler(optimizer, warmup_epochs, total_epochs, min_lr, base_lr):
    """Linear warmup -> cosine decay down to min_lr, stepped once per epoch."""
    warmup_epochs = max(0, min(warmup_epochs, total_epochs - 1))
    if warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)

    def warmup_fn(epoch):
        # epoch is 0-indexed; ramps LR factor from ~0 -> 1 over warmup_epochs
        return (epoch + 1) / warmup_epochs

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_fn)
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=min_lr
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )


def main():
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    print(f"Device: {device} | AMP: {use_amp}")

    # ------------------------------------------------------------------
    # Two SignDataset instances over the SAME index_csv/feature_dir:
    # one with augment=True (used only for the train split), one with
    # augment=False (used only for the val split). Row i in either
    # dataset maps to the same underlying sample, so we split *indices*
    # once and hand matching index subsets to each dataset via Subset.
    # This is what keeps augmentation strictly out of validation.
    # ------------------------------------------------------------------
    # ONE SignDataset decides which rows exist (feature file present, label
    # in vocabulary); the train view is a shallow copy of it with
    # augmentation switched on, sharing the same rows (and preload cache).
    # Split membership is read from THESE rows -- the previous version
    # re-read index_csv and used its raw row numbers, which misaligned
    # (or crashed on the row-count assert) as soon as any index row had no
    # feature file or a label outside the vocabulary.
    expected_dim = fs.feature_dim_for_schema(args.feature_schema_version)
    feature_columns = None
    if args.feature_groups:
        if args.feature_schema_version != fs.FEATURE_SCHEMA_VERSION:
            raise SystemExit("--feature_groups only applies to schema-v2 features")
        groups = {name: (a, b) for name, a, b in fs.FEATURE_GROUPS}
        wanted = [g.strip() for g in args.feature_groups.split(",") if g.strip()]
        unknown = [g for g in wanted if g not in groups]
        if unknown:
            raise SystemExit(f"unknown feature group(s) {unknown}; choose from {list(groups)}")
        feature_columns = [c for g in wanted for c in range(*groups[g])]
    val_dataset = SignDataset(
        args.index_csv, args.feature_dir, max_len=args.max_len,
        labels_json=args.vocab_json, augment=False, expected_dim=expected_dim,
        preload=args.preload, feature_columns=feature_columns,
    )
    train_dataset = copy.copy(val_dataset)
    train_dataset.augment = not args.no_augment

    n = len(val_dataset)
    train_idx_list, val_idx_list, used_official_split = resolve_train_val_indices_from_df(
        val_dataset.df, args.split_col, args.force_random_split, args.val_split, args.seed,
    )
    train_ds = Subset(train_dataset, train_idx_list)
    val_ds = Subset(val_dataset, val_idx_list)

    print(f"Dataset: {n} samples | train={len(train_ds)} (augment={not args.no_augment}) | "
          f"val={len(val_ds)} (augment=False) | official_split={used_official_split} | "
          f"architecture={args.architecture}")

    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    vloader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    # label2id / input_dim are dataset-level metadata, identical on both
    # instances since they load the same vocab/index_csv — val_dataset
    # is as good a source as train_dataset here.
    input_dim = val_dataset[0][0].shape[1]
    num_classes = len(val_dataset.label2id)
    model = build_model(args.architecture, input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = make_scheduler(optimizer, args.warmup_epochs, args.epochs, args.min_lr, args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A single training_config snapshot, saved into EVERY checkpoint this
    # run produces, so a checkpoint is self-describing months later
    # without needing to dig up whatever command line produced it.
    feature_meta = {}
    meta_path = Path(args.feature_dir) / "feature_meta.json"
    if meta_path.exists():
        feature_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_version = args.model_version or Path(args.out_dir).resolve().parent.name
    training_config = {
        "model_version": model_version,
        "index_csv": args.index_csv, "feature_dir": args.feature_dir,
        "vocab_json": args.vocab_json, "max_len": args.max_len,
        "num_classes": num_classes, "feature_schema_version": args.feature_schema_version,
        "feature_extraction": {k: feature_meta.get(k) for k in
                               ("mediapipe_version", "mediapipe_api", "frame_skip", "metadata_csv")},
        "device": str(device), "architecture": args.architecture,
        "feature_groups": args.feature_groups or "all",
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "label_smoothing": args.label_smoothing, "warmup_epochs": args.warmup_epochs,
        "min_lr": args.min_lr, "patience": args.patience, "min_delta": args.min_delta,
        "grad_clip": args.grad_clip, "augment": not args.no_augment, "seed": args.seed,
        "used_official_split": used_official_split, "val_split": args.val_split,
        "num_train_samples": len(train_ds), "num_val_samples": len(val_ds),
    }

    best_acc = -1.0
    epochs_no_improve = 0
    stopped_early = False
    history = []

    def save_ckpt(path, best, stopped):
        ckpt = build_checkpoint_metadata(
            model_state=model.state_dict(), label2id=val_dataset.label2id,
            input_dim=input_dim, max_len=args.max_len, architecture=args.architecture,
            best_val_accuracy=best, training_config=training_config,
            stopped_early=stopped,
        )
        # The schema actually used (build_checkpoint_metadata defaults to
        # the current v2 schema; a legacy-schema retrain must say 1.0).
        ckpt["feature_schema_version"] = args.feature_schema_version
        ckpt["model_version"] = model_version
        torch.save(ckpt, path)

    for epoch in range(args.epochs):
        t_epoch = time.time()
        model.train()
        running_loss = 0.0
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        train_acc = correct / max(total, 1)

        model.eval()
        val_correct = val_total = 0
        val_running_loss = 0.0
        with torch.no_grad():
            for x, y in vloader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = model(x)
                val_loss = criterion(out, y)
                val_running_loss += val_loss.item()
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / max(val_total, 1)
        val_loss_avg = val_running_loss / max(len(vloader), 1)

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:03d}/{args.epochs} | "
              f"loss={running_loss/max(len(loader),1):.4f} | "
              f"train={train_acc:.3f} | val={val_acc:.3f} | "
              f"val_loss={val_loss_avg:.4f} | lr={cur_lr:.2e} | {time.time()-t_epoch:.0f}s", flush=True)
        history.append({"epoch": epoch + 1, "train_loss": running_loss / max(len(loader), 1),
                        "train_acc": train_acc, "val_acc": val_acc, "val_loss": val_loss_avg,
                        "lr": cur_lr, "seconds": time.time() - t_epoch})
        (out_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        scheduler.step()

        if val_acc > best_acc + args.min_delta:
            best_acc = val_acc
            epochs_no_improve = 0
            save_ckpt(out_dir / "best.pt", best_acc, False)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(no val_acc improvement for {args.patience} epochs, best={best_acc:.3f})")
                stopped_early = True
                break

    save_ckpt(out_dir / "demo.pt", best_acc, stopped_early)

    # Export ONNX with dynamic batch dimension. Reload best weights so the
    # exported graph matches the checkpoint that scored best_acc, not
    # whatever was left in memory after early stopping / the last epoch.
    best_ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model"])

    onnx_path = out_dir.parent / "sign_recog.onnx"
    model.eval().cpu()
    dummy = torch.randn(1, args.max_len, input_dim)
    export_onnx(model, dummy, onnx_path)
    # ONNX meta.json mirrors the checkpoint's own metadata (minus the
    # weights themselves) so the browser / api.py's /model/meta can read
    # it without touching the .pt file at all.
    # ONNX/PyTorch parity check on random + real inputs: refuse to leave a
    # silently-diverging export next to the checkpoint.
    parity = None
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        probe = torch.randn(4, args.max_len, input_dim)
        with torch.no_grad():
            ref = model(probe).numpy()
        got = sess.run(["logits"], {"keypoints": probe.numpy()})[0]
        parity = float(np.abs(ref - got).max())
        print(f"ONNX parity: max |torch - onnx| = {parity:.2e}")
        if parity > 1e-3:
            raise RuntimeError(f"ONNX export diverges from PyTorch (max abs diff {parity})")
    except ImportError:
        print("onnxruntime not installed -- skipped ONNX parity check")
    onnx_meta = {k: v for k, v in best_ckpt.items() if k != "model"}
    onnx_meta["num_classes"] = num_classes
    onnx_meta["onnx_parity_max_abs_diff"] = parity
    Path(onnx_path).with_suffix(".json").write_text(json.dumps(onnx_meta, indent=2), encoding="utf-8")
    print(f"Saved best.pt, demo.pt and {onnx_path} "
          f"(architecture={args.architecture}, feature_schema_version={best_ckpt['feature_schema_version']}, "
          f"best_val_accuracy={best_acc:.4f})")


if __name__ == "__main__":
    freeze_support()
    main()