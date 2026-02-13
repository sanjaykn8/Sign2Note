import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import SignDataset
from model import TemporalCNN
from pathlib import Path

CHECKPOINT_DIR = Path("models/sign_recog/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = SignDataset("data/index.csv", "data/features")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

input_dim = dataset[0][0].shape[1]
num_classes = len(dataset.label2id)

model = TemporalCNN(input_dim, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):  # keep small for demo
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    print(f"Epoch {epoch+1} | Loss: {total_loss:.3f} | Acc: {correct/total:.3f}")

torch.save({
    "model": model.state_dict(),
    "label2id": dataset.label2id
}, CHECKPOINT_DIR / "demo.pt")

print("Model saved.")
