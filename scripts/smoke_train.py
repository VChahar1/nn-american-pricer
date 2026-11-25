"""Check the actual training path with a tiny generated dataset."""

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train import train_model


def main() -> None:
    """Train briefly and verify that a finite checkpoint is written."""
    result = train_model(n_train=32, n_val=16, epochs=1, batch_size=16, seed=123)
    assert math.isfinite(result["best_val_loss"])
    assert Path("models/best_model.pt").is_file()


if __name__ == "__main__":
    main()
