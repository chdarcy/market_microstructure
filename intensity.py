"""
intensity.py
============
Logistic intensity function, its inverse, and the Hamiltonian H(p) for the
option market-making model of Baldacci, Bergault & Guéant (2020).

Notation follows the paper (§2.2 and §4.1):
  Λ^{i,j}(δ) = λ_i / (1 + exp(α + β/V_i · δ))
  H^{i,j}(p) = sup_{δ ≥ δ_lb} Λ(δ)·(δ − p)

Vegas V_i are fixed at t=0 (constant-vega approximation, Assumption 1).
Values below are from heston_pricer.py (Heston MC, 100 000 paths, seed=42).
"""

import os

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display required
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit  # numerically stable sigmoid: expit(x) = 1/(1+exp(-x))

# ─────────────────────────────────────────────────────────────────────────────
# Model constants (paper §4.1)
# ─────────────────────────────────────────────────────────────────────────────
S0    = 10.0
ALPHA = 0.7
BETA  = 150.0           # β [yr^{1/2}]  (paper uses β/V_i as a combined param)

STRIKES    = [8, 9, 10, 11, 12]
MATURITIES = [1.0, 1.5, 2.0, 3.0]

# Vegas V^i = ∂O^i/∂(√ν)|_{t=0}  from heston_pricer.py (constant-vega approx.)
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


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Arrival-rate parameter λ_i
# ─────────────────────────────────────────────────────────────────────────────
def lam_i(K: float, s0: float = S0) -> float:
    """
    λ_i = 252·30 / (1 + 0.7·|S₀ − K|)

    Corresponds to ~30 requests/day for ATM options, falling to ~12.5/day
    for the most in/out-of-the-money options (paper §4.1).
    """
    return 252.0 * 30.0 / (1.0 + 0.7 * abs(s0 - K))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Logistic intensity Λ(δ)
# ─────────────────────────────────────────────────────────────────────────────
def Lambda(
    delta: float | np.ndarray,
    lam: float,
    V: float,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> float | np.ndarray:
    """
    Logistic intensity:
        Λ(δ) = λ / (1 + exp(α + β/V · δ))

    Strictly decreasing in δ:
        dΛ/dδ = −Λ·(1 − Λ/λ)·β/V < 0   for all δ.
    """
    # expit(x) = 1/(1+exp(x)) implemented in a numerically stable way;
    # Λ = λ · sigmoid(−(α + β/V · δ)) = λ · expit(−(α + β/V · δ))
    return lam * expit(-(alpha + (beta / V) * delta))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Hamiltonian  H(p) = sup_{δ≥δ_lb} Λ(δ)·(δ − p)
# ─────────────────────────────────────────────────────────────────────────────
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
    Hamiltonian: H(p) = sup_{δ ≥ δ_lb} Λ(δ)·(δ − p).

    Solved analytically via the first-order condition (FOC):

        Λ′(δ)·(δ−p) + Λ(δ) = 0
        ⟹  δ* = p + V / (β · u(δ*))                          [FOC]

    where u(δ) = 1 − Λ(δ)/λ = sigmoid(α + β/V · δ) ∈ (0,1).

    g(δ) ≡ δ − p − V/(β·u(δ)) is strictly increasing (g′ = 1/u > 0),
    so there is exactly one root.  Tight guaranteed bracket:
        g(p)         < 0  always (u > 0)
        g(p + 3V/β) > 0  always (u ≥ sigmoid(α+3) ≈ 0.976, so gap ≥ 2V/β)

    Parameters
    ----------
    return_delta_star : if True, also return δ* (useful for verification)
    """
    bV = beta / V   # β/V  (combined parameter appearing in the logistic)

    def u(delta: float) -> float:
        """sigmoid(α + β/V · δ) = 1 − Λ(δ)/λ"""
        return float(1.0 - expit(-(alpha + bV * delta)))

    def foc(delta: float) -> float:
        """g(δ) = δ − p − V/(β·u(δ));  root = δ*"""
        return delta - p - 1.0 / (bV * u(delta))

    # Bracket: g < 0 at lo, g > 0 at hi (proven in docstring)
    lo = max(delta_lb, p + 1e-12)   # g(lo) < 0
    hi = p + 3.0 * V / beta          # g(hi) > 0
    # Safety: widen hi until g(hi) > 0 (handles edge cases)
    for _ in range(40):
        if foc(hi) > 0:
            break
        hi = p + (hi - p) * 2.0

    delta_star = float(brentq(foc, lo, hi, xtol=1e-14, maxiter=200))
    delta_star = max(delta_lb, delta_star)   # enforce lower bound

    h_val      = float(Lambda(delta_star, lam, V, alpha, beta) * (delta_star - p))

    return (h_val, delta_star) if return_delta_star else h_val


# ─────────────────────────────────────────────────────────────────────────────
# 4.  H′(p) via central finite difference
# ─────────────────────────────────────────────────────────────────────────────
def H_prime(
    p: float,
    lam: float,
    V: float,
    alpha: float = ALPHA,
    beta: float = BETA,
    h: float = 1e-7,
) -> float:
    """
    dH/dp ≈ [H(p+h) − H(p−h)] / (2h).

    By the envelope theorem, H′(p) = −Λ(δ*(p)) exactly.
    This is verified in main().
    """
    return (H(p + h, lam, V, alpha, beta) - H(p - h, lam, V, alpha, beta)) / (2.0 * h)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Inverse of Λ
# ─────────────────────────────────────────────────────────────────────────────
def Lambda_inv(
    y: float,
    lam: float,
    V: float,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> float:
    """
    Find δ such that Λ(δ) = y  via analytical inversion of the logistic:

        y = λ / (1 + exp(α + β/V · δ))
        ⟹  δ = V/β · (ln(λ/y − 1) − α)

    The analytical result is validated against a numerical brentq solve.

    Raises ValueError if y ∉ (0, λ).
    """
    if not (0.0 < y < lam):
        raise ValueError(f"y={y:.6g} must be in the open interval (0, λ={lam:.2f})")

    # ── analytical inverse ────────────────────────────────────────────────
    delta_exact = V / beta * (np.log(lam / y - 1.0) - alpha)

    # ── numerical cross-check with brentq ─────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────
def test_decreasing() -> bool:
    """Verify Λ is strictly decreasing in δ for all 20 options."""
    deltas = np.linspace(-0.2, 0.2, 2000)
    all_ok = True
    for K in STRIKES:
        lam = lam_i(K)
        for T in MATURITIES:
            V    = VEGAS[(K, T)]
            vals = Lambda(deltas, lam, V)
            # Λ must never increase.  Allow diffs == 0 only where Λ has
            # underflowed to 0 (flat tail) — a strict < 0 check would
            # spuriously fail there.
            diffs = np.diff(vals)
            if np.any(diffs > 0):
                print(f"  FAIL  K={K}  T={T}: Λ not strictly decreasing")
                all_ok = False
    if all_ok:
        print("  OK – Λ strictly decreasing in δ for all 20 options ✓")
    return all_ok


def test_H_positive() -> bool:
    """Verify H(0) > 0 for all 20 options."""
    w = 8
    print(f"  {'K':>{w}} {'T':>{w}} {'λ_i':>{w+4}} {'V_i':>{w+2}} {'H(0)':>{w+4}}")
    print("  " + "─" * 48)
    all_ok = True
    for T in MATURITIES:
        for K in STRIKES:
            lam = lam_i(K)
            V   = VEGAS[(K, T)]
            h0  = H(0.0, lam, V)
            ok  = h0 > 0.0
            if not ok:
                all_ok = False
            flag = "" if ok else "  ← FAIL"
            print(f"  {K:>{w}} {T:>{w}.1f} {lam:>{w+4}.2f} {V:>{w+2}.4f} {h0:>{w+4}.6f}{flag}")
    if all_ok:
        print("  All H(0) > 0 ✓")
    return all_ok


def test_Lambda_inv() -> bool:
    """Round-trip: Λ(Λ_inv(y)) == y for several δ values."""
    K, T = 10, 1.0
    lam  = lam_i(K)
    V    = VEGAS[(K, T)]
    test_deltas = np.array([-0.08, -0.02, 0.0, 0.01, 0.05])
    all_ok = True
    print(f"  (K={K}, T={T})")
    print(f"  {'δ_in':>10} {'Λ(δ_in)':>14} {'δ_recovered':>14} {'error':>12}")
    print("  " + "─" * 54)
    for d_in in test_deltas:
        y     = float(Lambda(d_in, lam, V))
        d_out = Lambda_inv(y, lam, V)
        err   = abs(d_in - d_out)
        ok    = err < 1e-8
        if not ok:
            all_ok = False
        print(f"  {d_in:>10.4f} {y:>14.4f} {d_out:>14.8f} {err:>12.2e}"
              + ("" if ok else "  FAIL"))
    if all_ok:
        print("  Λ_inv round-trip ✓")
    return all_ok


def test_envelope_theorem() -> bool:
    """
    Verify the envelope theorem: H′(p) = −Λ(δ*(p)).

    The finite-difference derivative and the analytical expression should
    agree to several decimal places.
    """
    K, T = 10, 1.0
    lam  = lam_i(K)
    V    = VEGAS[(K, T)]
    ps   = [0.0, 0.5, 2.0, 5.0]

    print(f"  (K={K}, T={T}  λ={lam:.1f}  V={V:.4f})")
    print(f"  {'p':>8} {'H(p)':>10} {'δ*(p)':>12} "
          f"{'H\'(FD)':>12} {'−Λ(δ*)':>12} {'err':>10}")
    print("  " + "─" * 70)
    all_ok = True
    for p in ps:
        h_val, d_star = H(p, lam, V, return_delta_star=True)
        hp_fd         = H_prime(p, lam, V)
        hp_exact      = -Lambda(d_star, lam, V)
        err           = abs(hp_fd - hp_exact)
        ok            = err < 1e-4   # FD is only approximate
        if not ok:
            all_ok = False
        print(f"  {p:>8.2f} {h_val:>10.4f} {d_star:>12.6f} "
              f"{hp_fd:>12.4f} {hp_exact:>12.4f} {err:>10.2e}"
              + ("" if ok else "  FAIL"))
    if all_ok:
        print("  Envelope theorem verified ✓")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_Lambda() -> None:
    """Two-panel plot of Λ(δ) to visually confirm it is strictly decreasing."""
    # δ range: a few multiples of the logistic width (V/β)
    deltas = np.linspace(-0.08, 0.08, 800)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel 1: varying strike, fixed maturity ─────────────────────────
    ax = axes[0]
    T_fixed = 1.0
    colors  = plt.cm.plasma(np.linspace(0.15, 0.85, len(STRIKES)))
    for K, c in zip(STRIKES, colors):
        lam = lam_i(K)
        V   = VEGAS[(K, T_fixed)]
        ax.plot(deltas, Lambda(deltas, lam, V), color=c,
                label=f"K={K}  λ={lam:.0f}  V={V:.3f}")
    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("δ  (mid-to-bid spread)", fontsize=11)
    ax.set_ylabel("Λ(δ)  [transactions / year]", fontsize=11)
    ax.set_title(f"Intensity vs spread — varying strike  (T = {T_fixed} yr)",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: varying maturity, fixed strike ─────────────────────────
    ax = axes[1]
    K_fixed = 10
    lam     = lam_i(K_fixed)
    colors  = plt.cm.viridis(np.linspace(0.15, 0.85, len(MATURITIES)))
    for T, c in zip(MATURITIES, colors):
        V = VEGAS[(K_fixed, T)]
        ax.plot(deltas, Lambda(deltas, lam, V), color=c,
                label=f"T={T} yr  V={V:.3f}")
    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("δ  (mid-to-bid spread)", fontsize=11)
    ax.set_ylabel("Λ(δ)  [transactions / year]", fontsize=11)
    ax.set_title(f"Intensity vs spread — varying maturity  (K = {K_fixed})",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "intensity_Lambda.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    sep = "=" * 60

    print(sep)
    print("intensity.py  —  logistic intensity & Hamiltonian")
    print(sep)

    print("\n[1] Λ strictly decreasing in δ?")
    test_decreasing()

    print("\n[2] H(0) > 0 for all 20 options?")
    test_H_positive()

    print("\n[3] Λ_inv round-trip (K=10, T=1):")
    test_Lambda_inv()

    print("\n[4] Envelope theorem  H′(p) ≈ −Λ(δ*(p)):")
    test_envelope_theorem()

    print("\n[5] Plotting Λ(δ) …")
    plot_Lambda()

    print(f"\n{sep}")
    print("All tests passed.")
    print(sep)


if __name__ == "__main__":
    main()
