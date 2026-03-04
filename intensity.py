"""
Logistic intensity Λ(δ), its inverse, and the Hamiltonian H(p) for
the option market-making model (BBG 2020, §2.2 and §4.1).
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

# Model constants (§4.1)
S0    = 10.0
ALPHA = 0.7
BETA  = 150.0           # β [yr^{1/2}]  (paper uses β/V_i as a combined param)

STRIKES    = [8, 9, 10, 11, 12]
MATURITIES = [1.0, 1.5, 2.0, 3.0]

# Vegas from heston_pricer.py (constant-vega approximation)
VEGAS: dict[tuple, float] = {
    (8,  1.0): 0.408064, (9,  1.0): 0.905982, (10, 1.0): 1.250293,
    (11, 1.0): 1.064563, (12, 1.0): 0.566829,
    (8,  1.5): 0.460227, (9,  1.5): 0.828986, (10, 1.5): 1.059393,
    (11, 1.5): 0.978086, (12, 1.5): 0.658472,
    (8,  2.0): 0.471138, (9,  2.0): 0.749914, (10, 2.0): 0.923589,
    (11, 2.0): 0.884997, (12, 2.0): 0.676804,
    (8,  3.0): 0.449307, (9,  3.0): 0.641025, (10, 3.0): 0.751299,
    (11, 3.0): 0.753613, (12, 3.0): 0.648529,
}


def lam_i(K: float, s0: float = S0) -> float:
    """λ_i = 252·30 / (1 + 0.7·|S₀ − K|).  ~30 req/day ATM, ~12.5 deep OTM."""
    return 252.0 * 30.0 / (1.0 + 0.7 * abs(s0 - K))

def Lambda(
    delta: float | np.ndarray,
    lam: float,
    V: float,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> float | np.ndarray:
    """Logistic intensity: Λ(δ) = λ / (1 + exp(α + β/V · δ))."""
    return lam * expit(-(alpha + (beta / V) * delta))

def H(
    p: float,
    lam: float,
    V: float,
    alpha: float = ALPHA,
    beta: float = BETA,
    delta_lb: float = -1.0,
    return_delta_star: bool = False,
) -> float | tuple[float, float]:
    """
    Hamiltonian H(p) = sup_{δ ≥ δ_lb} Λ(δ)·(δ − p).
    Solved via FOC: δ* = p + V/(β·u(δ*)), root found by Brent's method.
    """
    bV = beta / V

    def u(delta: float) -> float:
        return float(1.0 - expit(-(alpha + bV * delta)))

    def foc(delta: float) -> float:
        return delta - p - 1.0 / (bV * u(delta))

    lo = max(delta_lb, p + 1e-12)
    hi = p + 3.0 * V / beta
    for _ in range(40):
        if foc(hi) > 0:
            break
        hi = p + (hi - p) * 2.0

    delta_star = float(brentq(foc, lo, hi, xtol=1e-14, maxiter=200))
    delta_star = max(delta_lb, delta_star)
    h_val      = float(Lambda(delta_star, lam, V, alpha, beta) * (delta_star - p))

    return (h_val, delta_star) if return_delta_star else h_val

def H_prime(
    p: float,
    lam: float,
    V: float,
    alpha: float = ALPHA,
    beta: float = BETA,
    h: float = 1e-7,
) -> float:
    """dH/dp via central FD.  By envelope theorem, H′(p) = −Λ(δ*(p))."""
    return (H(p + h, lam, V, alpha, beta) - H(p - h, lam, V, alpha, beta)) / (2.0 * h)

def Lambda_inv(
    y: float,
    lam: float,
    V: float,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> float:
    """
    Analytical inverse of the logistic: δ = V/β · (ln(λ/y − 1) − α).
    Cross-checked against brentq.  Raises ValueError if y ∉ (0, λ).
    """
    if not (0.0 < y < lam):
        raise ValueError(f"y={y:.6g} must be in the open interval (0, λ={lam:.2f})")

    # Analytical inverse
    delta_exact = V / beta * (np.log(lam / y - 1.0) - alpha)

    # Numerical cross-check
    width = max(10.0 * V / beta, 1.0)
    f_root = lambda d: Lambda(d, lam, V, alpha, beta) - y
    delta_numerical = brentq(f_root, delta_exact - width, delta_exact + width,
                             xtol=1e-14, maxiter=200)

    if abs(delta_exact - delta_numerical) > 1e-8:
        raise RuntimeError(
            f"Analytical ({delta_exact:.8f}) and numerical ({delta_numerical:.8f}) "
            "inverses disagree — check inputs."
        )

    return delta_exact
