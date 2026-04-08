"""
train.py – Train TemporalCNN on extracted sign-language features.
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from multiprocessing import freeze_support

from dataset import SignDataset
from model import TemporalCNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_csv", default="data/index.csv")
    parser.add_argument("--feature_dir", default="data/features")
    parser.add_argument("--out_dir", default="models/sign_recog/checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)  # safer default on Windows
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP")
    return parser.parse_args()


def main():
    args = parse_args()

    CHECKPOINT_DIR = Path(args.out_dir)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and not args.no_amp
    print(f"Device: {device}  |  AMP: {use_amp}")

    dataset = SignDataset(args.index_csv, args.feature_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    input_dim = dataset[0][0].shape[1]
    num_classes = len(dataset.label2id)
    max_len = dataset.max_len

    model = TemporalCNN(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler("cuda", enabled=use_amp)

    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        acc = correct / total if total else 0.0
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {total_loss:.3f} | Acc: {acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {"model": model.state_dict(), "label2id": dataset.label2id},
                CHECKPOINT_DIR / "best.pt",
            )

    torch.save(
        {"model": model.state_dict(), "label2id": dataset.label2id},
        CHECKPOINT_DIR / "demo.pt",
    )
    print(f"Saved demo.pt  (best_acc={best_acc:.3f})")

    model.eval()
    onnx_path = CHECKPOINT_DIR.parent / "sign_recog.onnx"
    dummy = torch.randn(1, max_len, input_dim).to(device)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["keypoints"],
        output_names=["logits"],
        dynamic_axes={"keypoints": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14,
    )
    print(f"ONNX model exported → {onnx_path}")


if __name__ == "__main__":
    freeze_support()
    main()