"""
param_sweeps.py
===============
Parameter sensitivity analysis for the vega market-making model.

Sweeps:
  A) α (alpha)  — intercept of the logistic intensity
  B) β (beta)   — steepness of the logistic intensity
  C) Intensity function family — logistic (baseline), exponential

For each configuration the HJB PDE is re-solved and optimal spreads are
recomputed.  Overlay plots compare mid-to-bid, ask-to-mid, total spread
vs portfolio vega, and spread vs strike at fixed vega levels.

All figures are saved to  figures/param_sweeps/  without overwriting the
main paper figures.
"""

import os
import copy
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit

# ── Local imports ─────────────────────────────────────────────────────────────
import hjb_solver as hjb
from hjb_solver import (
    nu_grid, vpi_grid, N_NU, N_VPI, V_BAR, N_T,
    solve_hjb, compute_penalties, compute_diffusion_drift,
)
from optimal_spreads import (
    compute_all_spreads, NU0, DELTA_INF, get_options,
)

ROOT        = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR   = os.path.join(ROOT, "figures", "param_sweeps")
os.makedirs(SWEEP_DIR, exist_ok=True)

# ── Baseline parameter values ─────────────────────────────────────────────────
ALPHA_BASE = hjb.ALPHA
BETA_BASE  = hjb.BETA

STRIKES    = [8, 9, 10, 11, 12]
MATURITIES = [1.0, 1.5, 2.0, 3.0]


# ═══════════════════════════════════════════════════════════════════════════════
# Parametric Hamiltonian / intensity helpers
# ═══════════════════════════════════════════════════════════════════════════════

# ---------- Logistic (paper baseline) ----------
def _hamiltonian_logistic(p, lam, V_i, alpha, beta, n_iter=15):
    """H(p) for logistic intensity Λ = λ / (1+exp(α + β/V·δ))."""
    bV  = beta / V_i
    VoB = V_i / beta
    delta = p + 2.0 * VoB
    for _ in range(n_iter):
        u = expit(alpha + bV * delta)
        delta = delta - u * (delta - p) + VoB
    u = expit(alpha + bV * delta)
    lam_val = lam * (1.0 - u)
    return lam_val * (delta - p)


def _lambda_inv_logistic(y, lam, V_i, alpha, beta):
    """Λ⁻¹(y) for logistic intensity."""
    bV = beta / V_i
    ratio = np.clip(lam / np.clip(y, 1e-12, None) - 1.0, 1e-12, None)
    return (np.log(ratio) - alpha) / bV


# ---------- Exponential: Λ(δ) = λ · exp(−(α + β/V·δ)) ----------
def _hamiltonian_exponential(p, lam, V_i, alpha, beta, n_iter=15):
    """
    H(p) = sup_δ  λ·exp(−α − β/V·δ)·(δ − p).
    FOC:  Λ(δ*) · [(δ*−p) · (−β/V) + 1] = 0
      ⟹  δ* = p + V/β   (closed-form!)
    H(p) = λ · exp(−α − β/V · (p + V/β)) · (V/β)
         = (λ V / β) · exp(−α − β/V·p − 1)
    """
    VoB = V_i / beta
    bV  = beta / V_i
    delta_star = p + VoB
    lam_val = lam * np.exp(np.clip(-(alpha + bV * delta_star), -500, 500))
    return lam_val * VoB


def _lambda_inv_exponential(y, lam, V_i, alpha, beta):
    """Λ⁻¹(y) for exponential: δ = (V/β)·(ln(λ/y) − α)."""
    bV = beta / V_i
    ratio = np.clip(lam / np.clip(y, 1e-12, None), 1e-12, None)
    return (np.log(ratio) - alpha) / bV


# ── Registry of intensity families ────────────────────────────────────────────
INTENSITY_FAMILIES = {
    "logistic":    (_hamiltonian_logistic,    _lambda_inv_logistic),
    "exponential": (_hamiltonian_exponential, _lambda_inv_exponential),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Core: solve + compute spreads with custom α, β, intensity
# ═══════════════════════════════════════════════════════════════════════════════

def _solve_with_params(options, alpha, beta, intensity_name="logistic"):
    """
    Solve HJB and compute spreads using specified (α, β, intensity).

    Temporarily patches hjb_solver globals; restores them on exit.
    Returns (v0, spreads_dict).
    """
    ham_fn, lam_inv_fn = INTENSITY_FAMILIES[intensity_name]

    # Save originals
    orig_alpha = hjb.ALPHA
    orig_beta  = hjb.BETA
    orig_ham   = hjb.hamiltonian

    # Patch hjb_solver globals
    hjb.ALPHA = alpha
    hjb.BETA  = beta
    hjb.hamiltonian = lambda p, lam, V_i, n_iter=15: ham_fn(
        p, lam, V_i, alpha, beta, n_iter
    )

    try:
        # Also patch optimal_spreads to use the same α, β, intensity
        import optimal_spreads as os_mod
        orig_os_alpha = os_mod.ALPHA
        orig_os_beta  = os_mod.BETA

        # Monkey-patch ALPHA/BETA used in lambda_inverse and hamiltonian_prime
        os_mod.ALPHA = alpha
        os_mod.BETA  = beta

        # Patch lambda_inverse to use our intensity family
        orig_lam_inv = os_mod.lambda_inverse
        os_mod.lambda_inverse = lambda y, lam, V_i: lam_inv_fn(
            y, lam, V_i, alpha, beta
        )

        # Patch hamiltonian_prime to use our hamiltonian
        orig_ham_prime = os_mod.hamiltonian_prime
        def _ham_prime_patched(p, lam, V_i, eps=1e-6):
            h_fn = lambda pp: ham_fn(pp, lam, V_i, alpha, beta)
            return (h_fn(p + eps) - h_fn(p - eps)) / (2 * eps)
        os_mod.hamiltonian_prime = _ham_prime_patched

        # Also patch the hamiltonian reference imported in optimal_spreads
        orig_os_ham = os_mod.hamiltonian
        os_mod.hamiltonian = hjb.hamiltonian

        # Solve
        v_all = solve_hjb(options)
        v0 = v_all[0]

        # Compute spreads
        spreads = compute_all_spreads(v0, options)

        return v0, spreads

    finally:
        # Restore everything
        hjb.ALPHA = orig_alpha
        hjb.BETA  = orig_beta
        hjb.hamiltonian = orig_ham
        os_mod.ALPHA = orig_os_alpha
        os_mod.BETA  = orig_os_beta
        os_mod.lambda_inverse = orig_lam_inv
        os_mod.hamiltonian_prime = orig_ham_prime
        os_mod.hamiltonian = orig_os_ham


# ═══════════════════════════════════════════════════════════════════════════════
# Overlay plotting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _overlay_vs_vega(results, param_name, K, Tm, kind, save_dir):
    """
    Overlay plot of one spread quantity vs portfolio vega for multiple
    parameter values.

    results: list of (label, color, spreads_dict)
    kind: 'bid' | 'ask' | 'spread'
    """
    vpi_plot = vpi_grid / 1e7
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for label, color, spreads in results:
        key = (K, Tm)
        if key not in spreads:
            continue
        s = spreads[key]
        if kind == "bid":
            data = s["delta_bid"]
            ylabel = "Mid-to-bid / price"
        elif kind == "ask":
            data = s["delta_ask"]
            ylabel = "Ask-to-mid / price"
        else:  # spread
            da, db = s["delta_ask"], s["delta_bid"]
            data = da + db
            ylabel = "Bid–ask spread / price"

        mask = data > DELTA_INF + 0.1
        if kind == "spread":
            mask &= (s["delta_ask"] > DELTA_INF + 0.1) & \
                     (s["delta_bid"] > DELTA_INF + 0.1)
        if not mask.any():
            continue
        ax.plot(vpi_plot[mask], data[mask] / s["price"],
                color=color, lw=1.4, alpha=0.85, label=label)

    ax.axvline(0, color="gray", lw=0.8, ls="--")
    if kind == "ask":
        ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Portfolio vega  (×10⁷)")
    ax.set_ylabel(ylabel)
    title_kind = {"bid": "Mid-to-bid", "ask": "Ask-to-mid", "spread": "Spread"}
    ax.set_title(f"{title_kind[kind]} / price  vs  Vπ  —  "
                 f"{param_name} sweep  (K={K}, T={Tm})")
    ax.legend(fontsize=7)
    plt.tight_layout()
    fname = f"sweep_{param_name}_{kind}_K{K}_T{Tm}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()
    return fname


def _overlay_spread_vs_strike(results, param_name, Tm, save_dir):
    """
    Overlay spread vs strike at Vπ=0 for multiple parameter values.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for label, color, spreads in results:
        K_arr, sp_arr = [], []
        for K in STRIKES:
            key = (K, Tm)
            if key not in spreads:
                continue
            s = spreads[key]
            da_raw, db_raw = s["delta_ask"], s["delta_bid"]
            valid = (da_raw > DELTA_INF + 0.1) & (db_raw > DELTA_INF + 0.1)
            if valid.sum() < 2:
                continue
            vpi_valid = vpi_grid[valid]
            if 0.0 < vpi_valid[0] or 0.0 > vpi_valid[-1]:
                continue
            da = np.interp(0.0, vpi_valid, da_raw[valid])
            db = np.interp(0.0, vpi_valid, db_raw[valid])
            total = da + db
            if total > 0:
                K_arr.append(K)
                sp_arr.append(total / s["price"])

        if K_arr:
            ax.plot(K_arr, sp_arr, marker="o", lw=1.4, color=color,
                    label=label, ms=6)

    ax.set_xlabel("Strike K")
    ax.set_ylabel("Bid–ask spread / price  (at Vπ = 0)")
    ax.set_title(f"Spread vs strike  —  {param_name} sweep  (T={Tm}, Vπ=0)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    fname = f"sweep_{param_name}_spread_vs_strike_T{Tm}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()
    return fname


def _generate_all_overlays(results, param_name, save_dir):
    """Generate the full set of overlay plots for one sweep."""
    fnames = []
    # Representative (K, T) combinations for vs-vega plots
    rep_options = [(10, 1.0), (10, 3.0), (8, 2.0), (12, 2.0)]
    for K, Tm in rep_options:
        for kind in ("bid", "ask", "spread"):
            f = _overlay_vs_vega(results, param_name, K, Tm, kind, save_dir)
            fnames.append(f)

    # Spread vs strike
    for Tm in [1.0, 2.0, 3.0]:
        f = _overlay_spread_vs_strike(results, param_name, Tm, save_dir)
        fnames.append(f)

    return fnames


# ═══════════════════════════════════════════════════════════════════════════════
# A)  Alpha sweep
# ═══════════════════════════════════════════════════════════════════════════════

ALPHA_VALUES = [0.2, 0.5, 0.7, 1.0, 1.5]
ALPHA_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def sweep_alpha(options, save_dir=None):
    """
    Sweep α over ALPHA_VALUES with β fixed at baseline.

    α controls the *intercept* of the logistic fill probability:
      Λ(0) = λ / (1 + exp(α))
    Low α  → high fill probability at mid (Λ(0) ≈ λ)  → tighter spreads
    High α → low fill probability at mid               → wider spreads

    Returns list of (label, color, spreads).
    """
    save_dir = save_dir or SWEEP_DIR
    results = []
    for alpha, color in zip(ALPHA_VALUES, ALPHA_COLORS):
        tag = f"α={alpha:.1f}"
        base = "★ " if abs(alpha - ALPHA_BASE) < 1e-6 else ""
        label = f"{base}{tag}"
        print(f"    {tag} …", end="", flush=True)
        t0 = time.time()
        _, spreads = _solve_with_params(
            options, alpha=alpha, beta=BETA_BASE, intensity_name="logistic"
        )
        print(f"  {time.time()-t0:.1f}s")
        results.append((label, color, spreads))

    fnames = _generate_all_overlays(results, "alpha", save_dir)
    return results, fnames


# ═══════════════════════════════════════════════════════════════════════════════
# B)  Beta sweep
# ═══════════════════════════════════════════════════════════════════════════════

BETA_VALUES = [50.0, 100.0, 150.0, 250.0, 400.0]
BETA_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def sweep_beta(options, save_dir=None):
    """
    Sweep β over BETA_VALUES with α fixed at baseline.

    β controls the *steepness* of the logistic intensity:
    Low β   → gentle slope  → fills are insensitive to spread → wider spreads
              (the MM has little pricing power — must post wide to compensate)
    High β  → steep drop-off → fills very sensitive to spread  → tighter spreads
              (raising the spread a little kills all flow, so the MM stays tight)

    Returns list of (label, color, spreads).
    """
    save_dir = save_dir or SWEEP_DIR
    results = []
    for beta, color in zip(BETA_VALUES, BETA_COLORS):
        tag = f"β={beta:.0f}"
        base = "★ " if abs(beta - BETA_BASE) < 1e-6 else ""
        label = f"{base}{tag}"
        print(f"    {tag} …", end="", flush=True)
        t0 = time.time()
        _, spreads = _solve_with_params(
            options, alpha=ALPHA_BASE, beta=beta, intensity_name="logistic"
        )
        print(f"  {time.time()-t0:.1f}s")
        results.append((label, color, spreads))

    fnames = _generate_all_overlays(results, "beta", save_dir)
    return results, fnames


# ═══════════════════════════════════════════════════════════════════════════════
# C)  Intensity family sweep
# ═══════════════════════════════════════════════════════════════════════════════

INTENSITY_NAMES  = ["logistic", "exponential"]
INTENSITY_COLORS = ["#2ca02c", "#d62728"]


def sweep_intensity(options, save_dir=None):
    """
    Sweep over intensity function families keeping (α, β) at baseline.

    • Logistic (baseline):  Λ = λ / (1 + exp(α + β/V·δ))
      – Saturates at both ends; S-shaped.  The paper's choice.

    • Exponential:  Λ = λ · exp(−α − β/V·δ)
      – Monotonically decreasing, no saturation at δ→−∞.
      – Thinner tail than logistic → fills drop faster for large δ.
      – Closed-form δ* = p + V/β.

    Returns list of (label, color, spreads).
    """
    save_dir = save_dir or SWEEP_DIR
    results = []
    for name, color in zip(INTENSITY_NAMES, INTENSITY_COLORS):
        base = "★ " if name == "logistic" else ""
        label = f"{base}{name}"
        print(f"    {name} …", end="", flush=True)
        t0 = time.time()
        _, spreads = _solve_with_params(
            options, alpha=ALPHA_BASE, beta=BETA_BASE, intensity_name=name,
        )
        print(f"  {time.time()-t0:.1f}s")
        results.append((label, color, spreads))

    fnames = _generate_all_overlays(results, "intensity", save_dir)
    return results, fnames


# ═══════════════════════════════════════════════════════════════════════════════
# Summary printer
# ═══════════════════════════════════════════════════════════════════════════════

def print_sweep_summary(results, param_name):
    """Print a compact table of spread at Vπ=0 for K=10, T=1.0."""
    key = (10, 1.0)
    print(f"\n  {param_name} sweep — spread/price at Vπ=0  (K=10, T=1.0)")
    print(f"  {'config':>20}  {'bid/C':>10}  {'ask/C':>10}  {'spread/C':>10}")
    print("  " + "─" * 56)
    for label, _, spreads in results:
        if key not in spreads:
            continue
        s = spreads[key]
        da_raw, db_raw = s["delta_ask"], s["delta_bid"]
        valid = (da_raw > DELTA_INF + 0.1) & (db_raw > DELTA_INF + 0.1)
        if valid.sum() < 2:
            print(f"  {label:>20}  {'—':>10}  {'—':>10}  {'—':>10}")
            continue
        vpi_valid = vpi_grid[valid]
        da = np.interp(0.0, vpi_valid, da_raw[valid])
        db = np.interp(0.0, vpi_valid, db_raw[valid])
        print(f"  {label:>20}  {db/s['price']:>10.5f}  "
              f"{da/s['price']:>10.5f}  {(da+db)/s['price']:>10.5f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_sweeps(options=None):
    """Run all three sweeps and save to figures/param_sweeps/."""
    if options is None:
        options = get_options()

    all_fnames = []

    print("\n  A) Alpha sweep")
    res_a, fnames_a = sweep_alpha(options)
    print_sweep_summary(res_a, "alpha")
    all_fnames.extend(fnames_a)

    print("\n  B) Beta sweep")
    res_b, fnames_b = sweep_beta(options)
    print_sweep_summary(res_b, "beta")
    all_fnames.extend(fnames_b)

    print("\n  C) Intensity family sweep")
    res_c, fnames_c = sweep_intensity(options)
    print_sweep_summary(res_c, "intensity")
    all_fnames.extend(fnames_c)

    print(f"\n  Total sweep figures: {len(all_fnames)}")
    for f in sorted(all_fnames):
        print(f"    → {f}")

    return all_fnames


if __name__ == "__main__":
    print("=" * 72)
    print("  Parameter sweeps — α, β, intensity function")
    print("=" * 72)
    run_all_sweeps()
