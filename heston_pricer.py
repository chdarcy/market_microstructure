"""
Heston Monte Carlo pricer (Q-measure, r=0).

Prices 20 European calls (5K × 4T) via log-Euler for S, full-truncation
Euler–Maruyama for ν, Cholesky-correlated increments.  Vegas by central
FD with common random numbers.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from black_scholes import implied_vol  # noqa: E402

# Model parameters (BBG §4.1)
S0      = 10.0
NU0     = 0.0225        # σ₀ = 0.15
KAPPA_Q = 3.0
THETA_Q = 0.0225
XI      = 0.2
RHO     = -0.5

STRIKES    = [8, 9, 10, 11, 12]
MATURITIES = [1.0, 1.5, 2.0, 3.0]

STEPS_PER_YEAR = 252
N_PATHS        = 100_000
SEED           = 42
FD_EPS         = NU0 * 5e-3     # bump for vega FD


def _simulate(nu0_val: float, seed: int) -> tuple[dict, dict]:
    """Simulate Heston paths; return S at maturity steps and step-index map."""
    T_max   = max(MATURITIES)
    n_steps = int(round(T_max * STEPS_PER_YEAR))
    dt      = T_max / n_steps
    sqrt_dt = np.sqrt(dt)
    rho_perp = np.sqrt(max(1.0 - RHO ** 2, 0.0))

    mat_to_step  = {T: int(round(T * STEPS_PER_YEAR)) for T in MATURITIES}
    target_steps = set(mat_to_step.values())

    rng = np.random.default_rng(seed)

    S  = np.full(N_PATHS, S0,      dtype=np.float64)
    nu = np.full(N_PATHS, nu0_val, dtype=np.float64)

    prices_at: dict[int, np.ndarray] = {}

    for step in range(n_steps):
        z1 = rng.standard_normal(N_PATHS)
        z2 = rng.standard_normal(N_PATHS)

        dW_nu = z1 * sqrt_dt
        dW_S  = (RHO * z1 + rho_perp * z2) * sqrt_dt

        nu_pos  = np.maximum(nu, 0.0)
        sqrt_nu = np.sqrt(nu_pos)

        S = S * np.exp(-0.5 * nu_pos * dt + sqrt_nu * dW_S)

        nu = nu + KAPPA_Q * (THETA_Q - nu) * dt + XI * sqrt_nu * dW_nu
        nu = np.maximum(nu, 0.0)

        if (step + 1) in target_steps:
            prices_at[step + 1] = S.copy()

    return prices_at, mat_to_step


def _option_prices(prices_at: dict, mat_to_step: dict) -> dict:
    """E[max(S_T - K, 0)] for each (K, T)."""
    result: dict[tuple, float] = {}
    for K in STRIKES:
        for T in MATURITIES:
            step          = mat_to_step[T]
            payoffs       = np.maximum(prices_at[step] - K, 0.0)
            result[(K, T)] = float(payoffs.mean())
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main

def get_option_data() -> list[dict]:
    """Return MC prices and vegas for all 20 (K, T) combinations."""
    print(f"Running Heston MC ({N_PATHS:,} paths) to price 20 options …")
    paths_base, mat_to_step = _simulate(NU0, SEED)
    prices                  = _option_prices(paths_base, mat_to_step)

    print("Computing vegas via central finite differences …")
    paths_up, _ = _simulate(NU0 + FD_EPS, SEED)
    paths_dn, _ = _simulate(NU0 - FD_EPS, SEED)
    prices_up   = _option_prices(paths_up, mat_to_step)
    prices_dn   = _option_prices(paths_dn, mat_to_step)

    sigma0 = np.sqrt(NU0)
    result = []
    for K in STRIKES:
        for T in MATURITIES:
            d_price_d_nu = (prices_up[(K, T)] - prices_dn[(K, T)]) / (2.0 * FD_EPS)
            vega = 2.0 * sigma0 * d_price_d_nu
            result.append({
                'strike':   float(K),
                'maturity': float(T),
                'price':    prices[(K, T)],
                'vega':     vega,
            })
    return result
