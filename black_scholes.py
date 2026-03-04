import numpy as np
from scipy.stats import norm


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """BS European call price (r=0)."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    """BS vega: dC/dσ (r=0)."""
    if T <= 0:
        return 0.0
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    sigma0: float = 0.20,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Implied vol via Newton-Raphson.  Raises ValueError on non-convergence."""
    sigma = sigma0
    for i in range(max_iter):
        price_bs = bs_call(S, K, T, sigma)
        vega = bs_vega(S, K, T, sigma)
        residual = price_bs - price
        if abs(residual) < tol:
            return sigma
        if abs(vega) < 1e-14:
            raise ValueError(f"Vega too small at iteration {i}; solver stalled.")
        sigma -= residual / vega
        if sigma <= 0:
            sigma = 1e-6  # keep positive
    raise ValueError(f"Implied vol solver did not converge after {max_iter} iterations.")
