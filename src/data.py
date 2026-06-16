from __future__ import annotations

import numpy as np
from scipy.stats.qmc import Sobol
from tqdm import tqdm

from .bjerksund_stensland import american_put_bs02, bs_european_put


# Input ranges. We normalize S by K (moneyness) to reduce the input space.
PARAM_RANGES = {
    "moneyness": (0.7, 1.3),     # S/K
    "T":         (0.1, 3.0),     # time to expiry in years
    "r":         (0.01, 0.10),   # risk-free rate
    "sigma":     (0.10, 0.60),   # volatility
    "q":         (0.00, 0.05),   # dividend yield
}


def generate_dataset(n_samples: int = 65_536, seed: int = 0) -> dict:
    """Generate training data: input parameters + BS02 American put prices.
    """
    # Round up to next power of 2 for Sobol balance.
    n_sobol = 1
    while n_sobol < n_samples:
        n_sobol *= 2

    sampler = Sobol(d=5, scramble=True, seed=seed)
    unit_samples = sampler.random(n_sobol)

    ranges = list(PARAM_RANGES.values())
    params = np.empty((n_sobol, 5))
    for i, (lo, hi) in enumerate(ranges):
        params[:, i] = lo + (hi - lo) * unit_samples[:, i]

    prices = np.empty(n_sobol)
    european_prices = np.empty(n_sobol)

    for i in tqdm(range(n_sobol), desc="Generating BS02 prices"):
        m, T, r, sigma, q = params[i]
        K = 100.0
        S = m * K
        try:
            p = american_put_bs02(S, K, T, r, sigma, q)
            e = bs_european_put(S, K, T, r, sigma, q)
            prices[i] = p
            european_prices[i] = e
        except Exception:
            prices[i] = np.nan
            european_prices[i] = np.nan

    # Filter out NaN, negative, and unreasonably large prices.
    valid = (
        np.isfinite(prices)
        & np.isfinite(european_prices)
        & (prices >= 0)
        & (european_prices >= 0)
        & (prices < 100)  # price can't exceed strike (K=100 for a put)
    )

    params = params[valid]
    prices = prices[valid]
    european_prices = european_prices[valid]

    n_filtered = n_sobol - valid.sum()
    if n_filtered > 0:
        print(f"Filtered {n_filtered} samples with invalid BS02 prices "
              f"({n_filtered/n_sobol:.1%} of total)")

    # Trim to requested size if we oversampled.
    if len(prices) > n_samples:
        params = params[:n_samples]
        prices = prices[:n_samples]
        european_prices = european_prices[:n_samples]

    # Normalize prices by K for scale-invariance.
    prices /= 100.0
    european_prices /= 100.0

    return {
        "inputs": params,
        "targets": prices,
        "european": european_prices,
        "param_names": list(PARAM_RANGES.keys()),
    }


def _demo() -> None:
    data = generate_dataset(n_samples=1000, seed=0)
    print(f"Generated {len(data['targets'])} samples")
    print(f"Input shape: {data['inputs'].shape}")
    print(f"Price range: [{data['targets'].min():.4f}, {data['targets'].max():.4f}]")
    print(f"Mean early-exercise premium: {(data['targets'] - data['european']).mean():.4f}")


if __name__ == "__main__":
    _demo()
