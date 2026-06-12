"""Generate figures for the README."""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.model import AmericanPutMLP
from src.bjerksund_stensland import american_put_bs02
from src.data import PARAM_RANGES


def figure_prediction_vs_true() -> None:
    """Scatter plot: NN predicted price vs BS02 true price."""
    from src.data import generate_dataset

    checkpoint = torch.load("models/best_model.pt", weights_only=False)
    model = AmericanPutMLP()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    mean, std = checkpoint["mean"], checkpoint["std"]

    test = generate_dataset(5000, seed=77)
    X = torch.tensor(test["inputs"], dtype=torch.float32)
    X_norm = (X - mean) / std
    with torch.no_grad():
        y_pred = model(X_norm).numpy()
    y_true = test["targets"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.3, s=5, color="steelblue")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("BS02 true price (normalized)")
    ax.set_ylabel("NN predicted price")
    ax.set_title("Neural network vs Bjerksund-Stensland ground truth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/pred_vs_true.png", dpi=120)
    print("Saved figures/pred_vs_true.png")


def figure_error_by_moneyness() -> None:
    """Show how prediction error varies with moneyness."""
    from src.data import generate_dataset

    checkpoint = torch.load("models/best_model.pt", weights_only=False)
    model = AmericanPutMLP()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    mean, std = checkpoint["mean"], checkpoint["std"]

    test = generate_dataset(10_000, seed=88)
    X = torch.tensor(test["inputs"], dtype=torch.float32)
    X_norm = (X - mean) / std
    with torch.no_grad():
        y_pred = model(X_norm).numpy()
    y_true = test["targets"]
    moneyness = test["inputs"][:, 0]

    errors = np.abs(y_pred - y_true)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(moneyness, errors, alpha=0.2, s=3, color="darkorange")
    ax.set_xlabel("Moneyness (S/K)")
    ax.set_ylabel("Absolute error")
    ax.set_title("Prediction error by moneyness")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/error_by_moneyness.png", dpi=120)
    print("Saved figures/error_by_moneyness.png")


if __name__ == "__main__":
    figure_prediction_vs_true()
    figure_error_by_moneyness()