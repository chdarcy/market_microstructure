import numpy as np
from scipy.stats import norm


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """
    Black-Scholes European call price with zero interest rates.

    Parameters
    ----------
    S     : spot price
    K     : strike
    T     : time to maturity (years)
    sigma : volatility (annualised)

    Returns
    -------
    Call price
    """
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    """
    Black-Scholes vega: dC/d(sigma), with zero interest rates.

    Parameters
    ----------
    S     : spot price
    K     : strike
    T     : time to maturity (years)
    sigma : volatility (annualised)

    Returns
    -------
    Vega (same units as the price)
    """
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
    """
    Implied volatility via Newton-Raphson.

    Parameters
    ----------
    price    : observed call price
    S        : spot price
    K        : strike
    T        : time to maturity (years)
    sigma0   : initial guess for volatility
    tol      : convergence tolerance on |f(sigma)|
    max_iter : maximum number of Newton steps

    Returns
    -------
    Implied volatility (annualised)

    Raises
    ------
    ValueError if the solver does not converge or vega becomes too small.
    """
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


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    S, K, T, sigma = 10.0, 10.0, 1.0, 0.15

    price = bs_call(S, K, T, sigma)
    vega  = bs_vega(S, K, T, sigma)
    iv    = implied_vol(price, S, K, T)

    print(f"Parameters : S={S}, K={K}, T={T}, sigma={sigma}")
    print(f"Call price : {price:.8f}")
    print(f"Vega       : {vega:.8f}")
    print(f"Implied vol: {iv:.10f}")
    print(f"IV error   : {abs(iv - sigma):.2e}  (should be < 1e-10)")
    assert abs(iv - sigma) < 1e-8, "IV solver did not recover the input sigma!"
    print("All checks passed.")
