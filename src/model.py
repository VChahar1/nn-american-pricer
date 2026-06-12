"""Neural network surrogate for American put pricing.

A small MLP that maps (moneyness, T, r, sigma, q) -> normalized American
put price. The architecture is deliberately simple: 3 hidden layers with
128 neurons each, ReLU activations. The function being approximated is
smooth and 5-dimensional, so a small network is more than sufficient.

Run `python -m src.model` for architecture summary.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AmericanPutMLP(nn.Module):
    """Simple feedforward network for option pricing."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 128,
                 n_layers: int = 3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _demo() -> None:
    model = AmericanPutMLP()
    print(f"Architecture: {model}")
    print(f"Parameters: {count_params(model):,}")
    x = torch.randn(4, 5)
    y = model(x)
    print(f"Sample output shape: {y.shape}")


if __name__ == "__main__":
    _demo()