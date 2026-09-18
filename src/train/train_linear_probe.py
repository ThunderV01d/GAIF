"""
Train a small MLP head on cached CLIP embeddings (real vs. AI-generated).

This is the lightweight model — the CLIP backbone is frozen and never
touched here, so this trains fast even on a weak GPU or CPU. It reads
directly from the .npz cache produced by src/features/clip_embeddings.py.

Usage:
    python -m src.train.train_linear_probe \
        --embeddings data/embeddings/train_val.npz \
        --out results/linear_probe.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class LinearProbeHead(nn.Module):
    """Small MLP: embedding_dim -> hidden -> 1 (sigmoid logit for 'is fake')."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(
    embeddings_path: Path,
    out_path: Path,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 256,
    val_fraction: float = 0.15,
    device: str = "cpu",
    seed: int = 42,
):
    data = np.load(embeddings_path, allow_pickle=True)
    X, y = data["embeddings"], data["labels"].astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=seed, stratify=y
    )

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = LinearProbeHead(embedding_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = (torch.sigmoid(model(xb)) > 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch:2d} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "embedding_dim": X.shape[1]}, out_path)

    print(f"Best val_acc={best_val_acc:.4f}, saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train(
        embeddings_path=args.embeddings,
        out_path=args.out,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
