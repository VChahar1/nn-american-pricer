from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _phi(S, T, gamma, H, I, r, b, sigma):
    """Helper function used in the Bjerksund-Stensland formula."""
    lam = (-r + gamma * b + 0.5 * gamma * (gamma - 1) * sigma**2) * T
    d = -(np.log(S / H) + (b + (gamma - 0.5) * sigma**2) * T) / (sigma * np.sqrt(T))
    kappa = 2 * b / (sigma**2) + (2 * gamma - 1)

    return (np.exp(lam) * S**gamma
            * (norm.cdf(d)
               - (I / S)**kappa * norm.cdf(d - 2 * np.log(I / S) / (sigma * np.sqrt(T)))))


def american_put_bs02(S: float, K: float, T: float, r: float,
                      sigma: float, q: float = 0.0) -> float:
    """Bjerksund-Stensland (2002) American put price.

    Uses put-call transformation: American put on S with strike K
    = American call on K with strike S, swapping r and q.
    """
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0:
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    # Safety: clamp extreme inputs that cause numerical overflow.
    sigma = min(sigma, 2.0)
    T = min(T, 10.0)

    result = _american_call_bs02(K, S, T, q, sigma, r)

    # Catch any NaN/inf that slipped through.
    if not np.isfinite(result) or result < 0:
        # Fall back to European put as a floor.
        return max(bs_european_put(S, K, T, r, sigma, q), 0.0)

    return result


def _american_call_bs02(S: float, K: float, T: float, r: float,
                        sigma: float, q: float) -> float:
    """Bjerksund-Stensland (2002) American call approximation."""
    b = r - q

    if b >= r:
        # When b >= r, early exercise is never optimal for a call;
        # fall back to Black-Scholes European call.
        return _bs_european_call(S, K, T, r, sigma, q)

    # Parameters for the two-part approximation.
    beta = (0.5 - b / sigma**2) + np.sqrt((b / sigma**2 - 0.5)**2 + 2 * r / sigma**2)

    B_inf = beta / (beta - 1) * K
    B_0 = max(K, r / (r - b) * K) if abs(r - b) > 1e-10 else K

    h_T = -(b * T + 2 * sigma * np.sqrt(T)) * B_0 / (B_inf - B_0)
    I = B_0 + (B_inf - B_0) * (1 - np.exp(h_T))

    if S >= I:
        return S - K

    # Midpoint for the two-part split.
    t1 = 0.5 * (np.sqrt(5) - 1) * T

    beta1 = (0.5 - b / sigma**2) + np.sqrt((b / sigma**2 - 0.5)**2 + 2 * r / (sigma**2 * (1 - np.exp(-r * t1))))
    B_t1_inf = beta1 / (beta1 - 1) * K
    B_t1_0 = max(K, r / (r - b) * K) if abs(r - b) > 1e-10 else K

    h_t1 = -(b * t1 + 2 * sigma * np.sqrt(t1)) * B_t1_0 / (B_t1_inf - B_t1_0)
    I_t1 = B_t1_0 + (B_t1_inf - B_t1_0) * (1 - np.exp(h_t1))

    if S >= I_t1:
        # Use the first part.
        alpha1 = (I_t1 - K) * I_t1 ** (-beta)
        return (alpha1 * S**beta
                - alpha1 * _phi(S, t1, beta, I_t1, I_t1, r, b, sigma)
                + _phi(S, t1, 1, I_t1, I_t1, r, b, sigma)
                - _phi(S, t1, 1, K, I_t1, r, b, sigma)
                - K * _phi(S, t1, 0, I_t1, I_t1, r, b, sigma)
                + K * _phi(S, t1, 0, K, I_t1, r, b, sigma))
    else:
        alpha2 = (I - K) * I ** (-beta)
        return (alpha2 * S**beta
                - alpha2 * _phi(S, T, beta, I, I_t1, r, b, sigma)
                + _phi(S, T, 1, I, I_t1, r, b, sigma)
                - _phi(S, T, 1, K, I_t1, r, b, sigma)
                - K * _phi(S, T, 0, I, I_t1, r, b, sigma)
                + K * _phi(S, T, 0, K, I_t1, r, b, sigma)
                + alpha1_fallback(S, t1, T, K, beta, I_t1, r, b, sigma))


def alpha1_fallback(S, t1, T, K, beta, I_t1, r, b, sigma):
    """Continuation term for the second region."""
    alpha1 = (I_t1 - K) * I_t1 ** (-beta)
    return (alpha1 * _phi(S, T, beta, I_t1, I_t1, r, b, sigma)
            - alpha1 * _phi(S, t1, beta, I_t1, I_t1, r, b, sigma)
            + _phi(S, t1, 1, I_t1, I_t1, r, b, sigma)
            - _phi(S, T, 1, I_t1, I_t1, r, b, sigma)
            + K * _phi(S, T, 0, I_t1, I_t1, r, b, sigma)
            - K * _phi(S, t1, 0, I_t1, I_t1, r, b, sigma))


def _bs_european_call(S, K, T, r, sigma, q):
    """Standard European call for the no-early-exercise case."""
    b = r - q
    d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * np.exp((b - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_european_put(S: float, K: float, T: float, r: float,
                    sigma: float, q: float = 0.0) -> float:
    """European put for comparison (lower bound for the American put)."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))


def _demo() -> None:
    print("Bjerksund-Stensland (2002) American put pricer")
    print("=" * 55)
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    am = american_put_bs02(S, K, T, r, sigma)
    eu = bs_european_put(S, K, T, r, sigma)
    print(f"  S={S}, K={K}, T={T}, r={r}, σ={sigma}")
    print(f"  European put:         {eu:.4f}")
    print(f"  American put (BS02):  {am:.4f}")
    print(f"  Early-exercise prem:  {am - eu:.4f}")


if __name__ == "__main__":
    _demo()