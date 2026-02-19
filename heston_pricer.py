"""
heston_pricer.py
================
Heston model Monte Carlo pricer under the risk-neutral (Q) measure.

Prices the 20 European call options (5 strikes × 4 maturities) used in
Baldacci, Bergault & Guéant (2020) and reproduces Figure 1 (implied-
volatility surface).

Discretisation
--------------
S  : log-Euler (exact given ν_t at each step, avoids negative prices)
     S_{t+dt} = S_t · exp(−½ν_t dt + √ν_t dW^S_t)
ν  : Euler–Maruyama with full-truncation scheme
     ν_{t+dt} = max(ν_t + κ(θ−ν_t)dt + ξ√(max(ν_t,0)) dW^ν_t, 0)

Correlated Brownian increments (Cholesky):
     dW^ν = z1 √dt
     dW^S = (ρ z1 + √(1−ρ²) z2) √dt,   z1,z2 iid N(0,1)

Vega
----
Finite-difference w.r.t. ν₀, then chain rule to get ∂/∂(√ν₀):
     V^i = ∂O^i/∂(√ν₀) = 2√ν₀ · [O(ν₀+ε) − O(ν₀−ε)] / (2ε)
Common random numbers (same RNG seed) cancel most of the MC noise.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from black_scholes import implied_vol  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Model parameters (paper §4.1)
# ─────────────────────────────────────────────────────────────────────────────
S0      = 10.0
NU0     = 0.0225        # initial variance  (σ₀ = √0.0225 = 0.15)
KAPPA_Q = 3.0           # mean-reversion speed under Q  [yr⁻¹]
THETA_Q = 0.0225        # long-run variance under Q      [yr⁻¹]
XI      = 0.2           # volatility of volatility       [yr⁻½]
RHO     = -0.5          # spot–vol correlation

STRIKES    = [8, 9, 10, 11, 12]
MATURITIES = [1.0, 1.5, 2.0, 3.0]   # years

# ─────────────────────────────────────────────────────────────────────────────
# Simulation parameters
# ─────────────────────────────────────────────────────────────────────────────
STEPS_PER_YEAR = 252
N_PATHS        = 100_000
SEED           = 42
FD_EPS         = NU0 * 5e-3     # finite-difference step for vega (0.5 % of ν₀)


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation
# ─────────────────────────────────────────────────────────────────────────────
def _simulate(nu0_val: float, seed: int) -> tuple[dict, dict]:
    """
    Euler–Maruyama simulation of the Heston model up to max(MATURITIES).

    Parameters
    ----------
    nu0_val : initial variance (perturbed for finite-difference vega)
    seed    : RNG seed — identical seed ⟹ identical Brownian paths (CRN)

    Returns
    -------
    prices_at   : {step_index: S array of shape (N_PATHS,)}
                  populated only at the steps matching MATURITIES
    mat_to_step : {maturity (float): step_index (int)}
    """
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
        # independent standard normals
        z1 = rng.standard_normal(N_PATHS)
        z2 = rng.standard_normal(N_PATHS)

        # correlated Brownian increments
        dW_nu = z1 * sqrt_dt
        dW_S  = (RHO * z1 + rho_perp * z2) * sqrt_dt

        # use max(ν,0) in both drift/diffusion coefficient (full truncation)
        nu_pos  = np.maximum(nu, 0.0)
        sqrt_nu = np.sqrt(nu_pos)

        # log-Euler for S (exact given ν_t, avoids negative prices)
        S = S * np.exp(-0.5 * nu_pos * dt + sqrt_nu * dW_S)

        # Euler–Maruyama for ν, then truncate
        nu = nu + KAPPA_Q * (THETA_Q - nu) * dt + XI * sqrt_nu * dW_nu
        nu = np.maximum(nu, 0.0)          # full truncation

        if (step + 1) in target_steps:
            prices_at[step + 1] = S.copy()

    return prices_at, mat_to_step


def _option_prices(prices_at: dict, mat_to_step: dict) -> dict:
    """
    Compute call prices as discounted (zero rates → no discounting) expected
    payoffs for every (K, T) combination.

    Returns
    -------
    {(K, T): price (float)}
    """
    result: dict[tuple, float] = {}
    for K in STRIKES:
        for T in MATURITIES:
            step          = mat_to_step[T]
            payoffs       = np.maximum(prices_at[step] - K, 0.0)
            result[(K, T)] = float(payoffs.mean())
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── 1. Base simulation ─────────────────────────────────────────────────
    print(f"Simulating {N_PATHS:,} Heston paths  ({STEPS_PER_YEAR} steps/yr) …")
    paths_base, mat_to_step = _simulate(NU0, SEED)
    prices                  = _option_prices(paths_base, mat_to_step)

    # ── 2. Finite-difference vega (common random numbers) ──────────────────
    print("Computing vegas via central finite differences …")
    paths_up, _ = _simulate(NU0 + FD_EPS, SEED)
    paths_dn, _ = _simulate(NU0 - FD_EPS, SEED)

    prices_up = _option_prices(paths_up, mat_to_step)
    prices_dn = _option_prices(paths_dn, mat_to_step)

    sigma0 = np.sqrt(NU0)
    vegas: dict[tuple, float] = {}
    for key in prices:
        # chain rule: ∂O/∂(√ν) = 2√ν · ∂O/∂ν
        d_price_d_nu = (prices_up[key] - prices_dn[key]) / (2.0 * FD_EPS)
        vegas[key]   = 2.0 * sigma0 * d_price_d_nu

    # ── 3. Implied volatilities ────────────────────────────────────────────
    print("Solving for implied volatilities …")
    ivs: dict[tuple, float] = {}
    for K in STRIKES:
        for T in MATURITIES:
            p = prices[(K, T)]
            try:
                iv = implied_vol(p, S0, K, T, sigma0=sigma0)
            except Exception as e:
                print(f"  IV failed  K={K}  T={T}: {e}")
                iv = float("nan")
            ivs[(K, T)] = iv

    # ── 4. Results table ───────────────────────────────────────────────────
    w = 13
    header = (
        f"{'Strike':>{w}} {'Maturity':>{w}} {'Price':>{w}}"
        f" {'Vega':>{w}} {'Impl. Vol':>{w}}"
    )
    print(f"\n{header}")
    print("─" * len(header))
    for T in MATURITIES:
        for K in STRIKES:
            print(
                f"{K:>{w}} {T:>{w}.1f} {prices[(K, T)]:>{w}.6f}"
                f" {vegas[(K, T)]:>{w}.6f} {ivs[(K, T)]:>{w}.6f}"
            )

    # ── 5. Figure 1 – 3-D implied volatility surface ──────────────────────
    K_arr  = np.array(STRIKES,    dtype=float)
    T_arr  = np.array(MATURITIES, dtype=float)
    # meshgrid: rows = maturities, cols = strikes  →  shape (4, 5)
    KK, TT = np.meshgrid(K_arr, T_arr)
    IV_arr = np.array(
        [[ivs[(K, T)] for K in STRIKES] for T in MATURITIES]
    )

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        KK, TT, IV_arr,
        cmap="gray_r", edgecolor="k", linewidth=0.4, alpha=0.90,
    )
    fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.10, label="Implied Volatility")

    ax.set_xlabel("Strike",                   labelpad=10)
    ax.set_ylabel("Time to Maturity (years)", labelpad=10)
    ax.set_zlabel("Implied Volatility",       labelpad=10)
    ax.set_title(
        "Figure 1 – Implied Volatility Surface\n"
        f"(Heston, Euler–Maruyama, {N_PATHS:,} paths)",
        pad=15,
    )
    ax.view_init(elev=25, azim=-55)

    plt.tight_layout()
    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figure1_iv_surface.png"
    )
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved → {out}")


def get_option_data() -> list[dict]:
    """
    Return Heston MC prices and vegas for all 20 (K, T) combinations
    used in Baldacci, Bergault & Guéant (2020).

    Returns
    -------
    list of dicts with keys: strike, maturity, price, vega
    """
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


if __name__ == "__main__":
    main()
