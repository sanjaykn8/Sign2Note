"""Train the Sign2Notes Temporal CNN on extracted keypoints."""
import argparse
import json
from multiprocessing import freeze_support
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch import autocast
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, Subset, random_split
from dataset import SignDataset
from model import TemporalCNN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index_csv", default="data/index.csv")
    p.add_argument("--feature_dir", default="data/features")
    p.add_argument("--out_dir", default="models/sign_recog/checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    # --- regularization / schedule knobs ---
    p.add_argument("--label_smoothing", type=float, default=0.05,
                    help="CrossEntropyLoss label smoothing (0 disables it).")
    p.add_argument("--warmup_epochs", type=int, default=0,
                    help="Linear LR warmup epochs before cosine decay.")
    p.add_argument("--min_lr", type=float, default=1e-5,
                    help="Floor LR for the cosine schedule.")
    p.add_argument("--patience", type=int, default=15,
                    help="Early-stopping patience, in epochs with no val_acc improvement.")
    p.add_argument("--min_delta", type=float, default=1e-4,
                    help="Minimum val_acc improvement to reset patience.")
    p.add_argument("--grad_clip", type=float, default=0.95,
                    help="Max grad norm for clipping (<=0 disables it).")
    # --- augmentation toggle ---
    p.add_argument("--no_augment", action="store_true",
                    help="Disable training-time augmentation (val is never augmented either way).")
    return p.parse_args()


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
    train_dataset = SignDataset(
        args.index_csv, args.feature_dir, max_len=args.max_len,
        labels_json="config/vocab.json", augment=not args.no_augment,
    )
    val_dataset = SignDataset(
        args.index_csv, args.feature_dir, max_len=args.max_len,
        labels_json="config/vocab.json", augment=False,
    )
    assert len(train_dataset) == len(val_dataset), \
        "train_dataset and val_dataset should see the same index_csv/rows"

    n = len(train_dataset)
    val_size = max(1, int(n * args.val_split))
    train_size = n - val_size
    train_idx_split, val_idx_split = random_split(
        range(n), [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_ds = Subset(train_dataset, train_idx_split.indices)
    val_ds = Subset(val_dataset, val_idx_split.indices)

    print(f"Dataset: {n} samples | train={len(train_ds)} (augment={not args.no_augment}) | "
          f"val={len(val_ds)} (augment=False)")

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
    model = TemporalCNN(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = make_scheduler(optimizer, args.warmup_epochs, args.epochs, args.min_lr, args.lr)
    scaler = GradScaler(enabled=use_amp)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    epochs_no_improve = 0
    stopped_early = False

    for epoch in range(args.epochs):
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
              f"val_loss={val_loss_avg:.4f} | lr={cur_lr:.2e}")

        scheduler.step()

        if val_acc > best_acc + args.min_delta:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "model": model.state_dict(),
                "label2id": val_dataset.label2id,
                "input_dim": input_dim,
                "max_len": args.max_len,
            }, out_dir / "best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(no val_acc improvement for {args.patience} epochs, best={best_acc:.3f})")
                stopped_early = True
                break

    torch.save({
        "model": model.state_dict(),
        "label2id": val_dataset.label2id,
        "input_dim": input_dim,
        "max_len": args.max_len,
    }, out_dir / "demo.pt")

    # Export ONNX with dynamic batch dimension. Reload best weights so the
    # exported graph matches the checkpoint that scored best_acc, not
    # whatever was left in memory after early stopping / the last epoch.
    best_ckpt = torch.load(out_dir / "best.pt", map_location="cpu")
    model.load_state_dict(best_ckpt["model"])

    onnx_path = out_dir.parent / "sign_recog.onnx"
    model.eval().cpu()
    dummy = torch.randn(1, args.max_len, input_dim)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["keypoints"], output_names=["logits"],
        dynamic_axes={"keypoints": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    meta = {"input_dim": input_dim, "max_len": args.max_len,
            "num_classes": num_classes, "best_val_accuracy": best_acc,
            "stopped_early": stopped_early, "augmented": not args.no_augment}
    Path(onnx_path).with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved best.pt, demo.pt and {onnx_path}")


if __name__ == "__main__":
    freeze_support()
    main()