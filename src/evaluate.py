"""Evaluate the trained model: accuracy, speed, and monotonicity checks.

Run `python -m src.evaluate` after training.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .bjerksund_stensland import american_put_bs02
from .data import generate_dataset
from .model import AmericanPutMLP


def evaluate(n_test: int = 10_000, seed: int = 99) -> dict:
    """Load the best model and evaluate on a fresh test set."""
    checkpoint = torch.load("models/best_model.pt", weights_only=False)
    model = AmericanPutMLP()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    mean, std = checkpoint["mean"], checkpoint["std"]

    test_data = generate_dataset(n_test, seed=seed)
    X = torch.tensor(test_data["inputs"], dtype=torch.float32)
    X_norm = (X - mean) / std
    y_true = test_data["targets"]

    # NN prediction.
    t0 = time.time()
    with torch.no_grad():
        y_pred = model(X_norm).numpy()
    nn_time = time.time() - t0

    # BS02 timing on same inputs for comparison.
    t0 = time.time()
    for i in range(min(1000, n_test)):
        m, T, r, sigma, q = test_data["inputs"][i]
        american_put_bs02(m * 100, 100, T, r, sigma, q)
    bs02_time = (time.time() - t0) / min(1000, n_test) * n_test

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    results = {
        "mae": float(abs_errors.mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
        "max_error": float(abs_errors.max()),
        "median_error": float(np.median(abs_errors)),
        "nn_time_ms": nn_time * 1000,
        "bs02_time_ms": bs02_time * 1000,
        "speedup": bs02_time / nn_time if nn_time > 0 else float("inf"),
        "n_test": n_test,
    }

    print("Evaluation on held-out test set")
    print("=" * 50)
    print(f"  Test samples:       {n_test}")
    print(f"  MAE:                {results['mae']:.6f}")
    print(f"  RMSE:               {results['rmse']:.6f}")
    print(f"  Max error:          {results['max_error']:.6f}")
    print(f"  Median error:       {results['median_error']:.6f}")
    print()
    print(f"  NN inference time:  {results['nn_time_ms']:.1f} ms ({n_test} samples)")
    print(f"  BS02 time:          {results['bs02_time_ms']:.1f} ms ({n_test} samples)")
    print(f"  Speedup:            {results['speedup']:.0f}×")

    # Monotonicity check: price should increase with sigma (all else equal).
    print()
    print("Monotonicity check (dP/dσ > 0):")
    base = torch.tensor([[1.0, 1.0, 0.05, 0.3, 0.02]], dtype=torch.float32)
    base_norm = (base - mean) / std
    sigmas = np.linspace(0.10, 0.60, 20)
    nn_prices = []
    for sig in sigmas:
        x = base.clone()
        x[0, 3] = sig
        x_norm = (x - mean) / std
        with torch.no_grad():
            nn_prices.append(model(x_norm).item())
    is_monotone = all(nn_prices[i] <= nn_prices[i+1] for i in range(len(nn_prices)-1))
    print(f"  Price monotonically increasing in σ: {is_monotone}")
    if not is_monotone:
        violations = sum(1 for i in range(len(nn_prices)-1) if nn_prices[i] > nn_prices[i+1])
        print(f"  Violations: {violations} out of {len(nn_prices)-1} intervals")

    return results


if __name__ == "__main__":
    evaluate()