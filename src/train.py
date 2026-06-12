"""Training loop for the American put surrogate.

Trains on BS02 ground-truth prices, validates on a held-out set,
saves the best model by validation loss.

Run `python -m src.train` to train and save a model.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .data import generate_dataset
from .model import AmericanPutMLP, count_params


def train_model(
    n_train: int = 50_000,
    n_val: int = 10_000,
    epochs: int = 200,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Train the MLP and return results."""
    print("Generating training data...")
    train_data = generate_dataset(n_train, seed=seed)
    val_data = generate_dataset(n_val, seed=seed + 1)

    X_train = torch.tensor(train_data["inputs"], dtype=torch.float32)
    y_train = torch.tensor(train_data["targets"], dtype=torch.float32)
    X_val = torch.tensor(val_data["inputs"], dtype=torch.float32)
    y_val = torch.tensor(val_data["targets"], dtype=torch.float32)

    assert torch.isfinite(X_train).all(), "NaN in training inputs"
    assert torch.isfinite(y_train).all(), "NaN in training targets"
    assert torch.isfinite(X_val).all(), "NaN in validation inputs"
    assert torch.isfinite(y_val).all(), "NaN in validation targets"
    print(f"Data verified: {len(y_train)} train, {len(y_val)} val, no NaN")

    # Normalize inputs to zero-mean, unit-variance.
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )

    model = AmericanPutMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    criterion = nn.MSELoss()

    print(f"Model: {count_params(model):,} parameters")
    print(f"Training: {n_train} samples, {epochs} epochs, batch {batch_size}")

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)
        epoch_loss /= len(X_train)
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            val_loss = criterion(val_pred, y_val.to(device)).item()
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "mean": mean, "std": std,
                "val_loss": val_loss, "epoch": epoch,
            }, "models/best_model.pt")

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4d}: train={epoch_loss:.6f}  val={val_loss:.6f}"
                  f"  lr={optimizer.param_groups[0]['lr']:.1e}")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best val loss: {best_val_loss:.6f} (RMSE: {np.sqrt(best_val_loss):.4f})")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "elapsed": elapsed,
        "mean": mean, "std": std,
    }


def _demo() -> None:
    train_model(n_train=20_000, n_val=5_000, epochs=100)


if __name__ == "__main__":
    _demo()