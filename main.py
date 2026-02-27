#!/usr/bin/env python3
"""
main.py — Full pipeline for "Algorithmic Market Making for Options"
===================================================================
Reproduces Figures 1-13 from Baldacci, Bergault & Guéant (2020) and
runs extended analysis (parameter sweeps, alternative intensities,
queue-reactive LOB model).

Output folders
--------------
  figures/original/       — Figures 1-13 from the paper + extension plots
  figures/param_sweeps/   — α / β parameter sensitivity analysis
  figures/intensity/      — Intensity family comparison + QR CTMC validation

Steps
-----
1.  Heston pricer   → option prices, vegas, IVs, Figure 1
2.  HJB solver      → value function v(0,ν,Vπ), Figure 2
3.  Optimal spreads  → Figures 4-8 (bid/price) and 9-13 (relative IV)
4.  Convergence      → Figure 3 (stationarity of quotes)
5.  Summary table    → comparison with paper values
6.  Parameter sweeps → α, β sensitivity
7.  Intensity        → logistic / exponential / queue-reactive + CTMC
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive — all output via savefig
import matplotlib.pyplot as plt

# ── Ensure local modules importable ───────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

FIGURES_DIR   = os.path.join(ROOT, "figures")
ORIGINAL_DIR  = os.path.join(FIGURES_DIR, "original")
SWEEP_DIR     = os.path.join(FIGURES_DIR, "param_sweeps")
INTENSITY_DIR = os.path.join(FIGURES_DIR, "intensity")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def banner(msg: str) -> None:
    """Print a section banner."""
    width = 72
    print()
    print("═" * width)
    print(f"  {msg}")
    print("═" * width)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Heston Pricer — prices, vegas, IVs, Figure 1
# ─────────────────────────────────────────────────────────────────────────────
def step1_heston_pricer():
    banner("Step 1 · Heston Monte-Carlo pricer")

    from heston_pricer import (
        get_option_data, main as heston_main,
        STRIKES, MATURITIES, S0, NU0, N_PATHS, SEED,
    )
    from black_scholes import implied_vol

    t0 = time.time()
    raw = get_option_data()
    elapsed = time.time() - t0

    # IVs
    sigma0 = np.sqrt(NU0)
    for r in raw:
        try:
            r["iv"] = implied_vol(r["price"], S0, r["strike"],
                                  r["maturity"], sigma0=sigma0)
        except Exception:
            r["iv"] = float("nan")

    # ── Table ─────────────────────────────────────────────────────────────
    print(f"\nHeston MC finished in {elapsed:.1f}s  "
          f"({N_PATHS:,} paths, seed={SEED})\n")
    hdr = (f"{'K':>6} {'T':>6} {'Price':>10} {'Vega':>10} {'IV':>10}")
    print(hdr)
    print("─" * len(hdr))
    for r in raw:
        print(f"{r['strike']:>6.0f} {r['maturity']:>6.1f} "
              f"{r['price']:>10.4f} {r['vega']:>10.4f} "
              f"{r['iv']:>10.5f}")

    # ── Figure 1 (IV surface) ─────────────────────────────────────────────
    K_arr = np.array(STRIKES, dtype=float)
    T_arr = np.array(MATURITIES, dtype=float)
    KK, TT = np.meshgrid(K_arr, T_arr)
    IV_arr = np.array([
        [next(r["iv"] for r in raw
              if r["strike"] == K and r["maturity"] == T)
         for K in STRIKES]
        for T in MATURITIES
    ])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(KK, TT, IV_arr, cmap="gray_r",
                    edgecolor="k", linewidth=0.4, alpha=0.90)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Time to Maturity (years)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title("Figure 1 – Implied Volatility Surface")
    ax.view_init(elev=25, azim=-55)
    plt.tight_layout()
    plt.savefig(os.path.join(ORIGINAL_DIR, "figure01_iv_surface.png"), dpi=150)
    plt.close()
    print("\n→ Saved figures/original/figure01_iv_surface.png")

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HJB Solver — value function, Figure 2
# ─────────────────────────────────────────────────────────────────────────────
def step2_hjb_solver(options):
    banner("Step 2 · HJB PDE solver")

    from hjb_solver import (
        solve_hjb, nu_grid, vpi_grid, plot_figure2,
        N_T, N_NU, N_VPI, T, dt, V_BAR, GAMMA, XI,
    )

    print(f"  Grid          : N_NU={N_NU}  N_VPI={N_VPI}  N_T={N_T}")
    print(f"  Horizon       : T = {T:.6f} yr  ({T*252:.2f} trading days)")
    print(f"  Time step     : dt = {dt:.3e} yr")
    print(f"  V_bar         : {V_BAR:.0e}")
    print(f"  γ={GAMMA}  ξ={XI}")
    print(f"  dVpi          : {vpi_grid[1]-vpi_grid[0]:.0f}")
    print(f"  dnu           : {nu_grid[1]-nu_grid[0]:.6f}")
    print()

    t0 = time.time()
    v_all = solve_hjb(options)
    elapsed = time.time() - t0

    v0 = v_all[0]
    nu0_idx = np.argmin(np.abs(nu_grid - 0.0225))
    v_at_centre = np.interp(0.0, vpi_grid, v0[nu0_idx, :])

    print(f"  Solved in {elapsed:.1f}s")
    print(f"  v(0) range    : [{v0.min():.1f}, {v0.max():.1f}]")
    print(f"  v(0,ν₀,0)    : {v_at_centre:.1f}")
    print(f"  v_peak        : {v0.max():.1f}  (paper ≈ 120,000)")

    # ── Figure 2 ──────────────────────────────────────────────────────────
    save_path = os.path.join(ORIGINAL_DIR, "figure02_value_function.png")
    VPI_mesh, NU_mesh = np.meshgrid(vpi_grid, nu_grid)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(VPI_mesh / 1e7, NU_mesh, v0,
                    cmap="gray_r", edgecolor="none", alpha=0.9)
    fig.colorbar(
        ax.plot_surface(VPI_mesh / 1e7, NU_mesh, v0,
                        cmap="gray_r", edgecolor="none", alpha=0.0),
        ax=ax, shrink=0.5, label="Value function",
    )
    ax.set_xlabel("Portfolio vega  (×10⁷)")
    ax.set_ylabel("Instantaneous variance ν")
    ax.set_zlabel("v(0, ν, V^π)")
    ax.set_title("Figure 2: Value function at t = 0")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n→ Saved {os.path.relpath(save_path, ROOT)}")

    return v_all


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Optimal Spreads — Figures 4-8 and 9-13
# ─────────────────────────────────────────────────────────────────────────────
def step3_optimal_spreads(v0, options):
    banner("Step 3 · Optimal quotes  (Figures 4-13)")

    from hjb_solver import nu_grid, vpi_grid, N_VPI
    from optimal_spreads import (
        compute_all_spreads, plot_figures_4_8, plot_figures_9_13,
        NU0, DELTA_INF, S0,
    )

    spreads = compute_all_spreads(v0, options)

    # ── Summary table ─────────────────────────────────────────────────────
    hdr = (f"{'K':>5} {'T':>5} {'price':>8} {'vega':>8} "
           f"{'bid@0':>10} {'bid/C':>10}")
    print(f"\n{hdr}")
    print("─" * len(hdr))
    for opt in options:
        key = (opt["strike"], opt["maturity"])
        s = spreads[key]
        db = np.interp(0.0, vpi_grid, s["delta_bid"])
        print(f"{opt['strike']:>5.0f} {opt['maturity']:>5.1f} "
              f"{opt['price']:>8.3f} {opt['vega']:>8.3f} "
              f"{db:>10.5f} {db/opt['price']:>10.5f}")

    # ── Figures 4-8 ───────────────────────────────────────────────────────
    _save_spread_figures(spreads, options, vpi_grid, DELTA_INF,
                         fig_range=range(4, 9), kind="bid")
    # ── Figures 9-13 ──────────────────────────────────────────────────────
    _save_iv_figures(spreads, options, vpi_grid, DELTA_INF, S0, NU0)

    # ── Extension plots ───────────────────────────────────────────────────
    from optimal_spreads import (
        plot_ask_to_mid, plot_bid_ask_spread,
        plot_spread_vs_strike, print_short_vega_commentary,
    )
    banner("Extension · Ask-to-mid, bid–ask spread, spread vs strike")

    print("\n  A) Ask-to-mid / price  vs  portfolio vega")
    plot_ask_to_mid(spreads, options, save_dir=ORIGINAL_DIR)

    print("\n  B) Bid–ask spread / price  vs  portfolio vega")
    plot_bid_ask_spread(spreads, options, save_dir=ORIGINAL_DIR)

    print("\n  C) Spread vs strike at fixed Vπ levels")
    plot_spread_vs_strike(spreads, options, save_dir=ORIGINAL_DIR)

    print_short_vega_commentary(spreads, options)

    return spreads


def _save_spread_figures(spreads, options, vpi_grid, DELTA_INF,
                         fig_range, kind):
    """Figures 4-8: bid spread / price."""
    from optimal_spreads import DELTA_INF as _DI
    STRIKES    = [8, 9, 10, 11, 12]
    MATURITIES = [1.0, 1.5, 2.0, 3.0]
    markers    = ["*", "^", "o", "s"]
    vpi_plot   = vpi_grid / 1e7

    for fig_num, K in zip(fig_range, STRIKES):
        fig, ax = plt.subplots(figsize=(8, 5))
        for m_idx, Tm in enumerate(MATURITIES):
            key = (K, Tm)
            if key not in spreads:
                continue
            s    = spreads[key]
            db   = s["delta_bid"]
            mask = db > DELTA_INF + 0.1
            ax.scatter(vpi_plot[mask], db[mask] / s["price"],
                       s=10, marker=markers[m_idx],
                       label=f"(K,T)=({K},{Tm})  C={s['price']:.2f}")
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel("Portfolio vega  (×10⁷)")
        ax.set_ylabel("Optimal mid-to-bid / price")
        ax.set_title(f"Figure {fig_num}: Optimal mid-to-bid / price  (K={K})")
        ax.legend(fontsize=7)
        plt.tight_layout()
        fname = f"figure{fig_num:02d}_K{K}_spread.png"
        plt.savefig(os.path.join(ORIGINAL_DIR, fname), dpi=150)
        plt.close()
        print(f"  → {fname}")


def _save_iv_figures(spreads, options, vpi_grid, DELTA_INF, S0, NU0):
    """Figures 9-13: relative bid IV."""
    from black_scholes import implied_vol
    STRIKES    = [8, 9, 10, 11, 12]
    MATURITIES = [1.0, 1.5, 2.0, 3.0]
    markers    = ["*", "^", "o", "s"]
    vpi_plot   = vpi_grid / 1e7

    for fig_num, K in enumerate(STRIKES, start=9):
        fig, ax = plt.subplots(figsize=(8, 5))
        for m_idx, Tm in enumerate(MATURITIES):
            key = (K, Tm)
            if key not in spreads:
                continue
            s    = spreads[key]
            iv0  = s["iv0"]
            db   = s["delta_bid"]
            mask = db > DELTA_INF + 0.1
            bid_prices = s["price"] - db[mask]
            bid_iv = np.array([
                implied_vol(max(bp, 1e-6), S0, K, Tm)
                for bp in bid_prices
            ])
            ax.scatter(vpi_plot[mask], bid_iv / iv0,
                       s=10, marker=markers[m_idx],
                       label=f"(K,T)=({K},{Tm})  IV₀={iv0:.4f}")
        ax.axhline(1.0, color="gray", lw=0.8, ls="--")
        ax.axvline(0,   color="gray", lw=0.8, ls="--")
        ax.set_xlabel("Portfolio vega  (×10⁷)")
        ax.set_ylabel("IV of optimal bid / initial IV")
        ax.set_title(f"Figure {fig_num}: Relative bid IV  (K={K})")
        ax.legend(fontsize=7)
        plt.tight_layout()
        fname = f"figure{fig_num:02d}_K{K}_relIV.png"
        plt.savefig(os.path.join(ORIGINAL_DIR, fname), dpi=150)
        plt.close()
        print(f"  → {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Convergence — Figure 3
# ─────────────────────────────────────────────────────────────────────────────
def step4_convergence(v_all, options):
    banner("Step 4 · Convergence check  (Figure 3)")

    from hjb_solver import nu_grid, vpi_grid, N_T, N_VPI, T
    from optimal_spreads import NU0, DELTA_INF
    from convergence import bid_spread_vs_time

    opt1 = next(o for o in options
                if o["strike"] == 8 and o["maturity"] == 1.0)
    nu_idx = np.argmin(np.abs(nu_grid - NU0))
    price  = opt1["price"]

    print(f"  Option: K={opt1['strike']}, T_opt={opt1['maturity']}, "
          f"price={price:.3f}, vega={opt1['vega']:.3f}")

    spr = bid_spread_vs_time(v_all, opt1, nu_idx)    # (N_T+1, N_VPI)
    t_grid = np.linspace(0, T, N_T + 1)

    # ── Convergence diagnostic ────────────────────────────────────────────
    mid_j   = np.argmin(np.abs(vpi_grid))
    s_t0    = spr[0,        mid_j] / price
    s_10pct = spr[max(1, N_T // 10), mid_j] / price
    s_T     = spr[N_T,      mid_j] / price
    rel_err = abs(s_t0 - s_10pct) / max(abs(s_t0), 1e-12)

    print(f"  spread/C at Vπ≈0:  t=0 → {s_t0:.6f}   "
          f"t=0.1T → {s_10pct:.6f}   t=T → {s_T:.6f}")
    print(f"  |s(0)−s(0.1T)|/s(0) = {rel_err:.2e}  "
          f"({'✓ converged' if rel_err < 0.01 else '✗ NOT converged'})")

    # ── Figure 3 ──────────────────────────────────────────────────────────
    n_lines    = 25
    vpi_indices = np.unique(
        np.linspace(0, N_VPI - 1, n_lines, dtype=int)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={"width_ratios": [3, 2]})
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=vpi_grid.min() / 1e7,
                         vmax=vpi_grid.max() / 1e7)

    for ax_idx, ax in enumerate(axes):
        for j in vpi_indices:
            ratio = spr[:, j] / price
            if np.all(ratio < (DELTA_INF + 0.1) / price):
                continue
            ax.plot(t_grid, ratio, color=cmap(norm(vpi_grid[j] / 1e7)),
                    lw=0.8, alpha=0.85)
        ax.set_xlabel("Time t (years)")
        ax.set_ylabel("Optimal mid-to-bid / price")
        ax.axvline(T, color="gray", lw=0.6, ls=":")
        if ax_idx == 1:
            ax.set_xlim(0.7 * T, T * 1.02)
            ax.set_title("Zoom: near terminal time T")
        else:
            ax.set_title(f"Figure 3: Convergence  "
                         f"(K={opt1['strike']}, T_opt={opt1['maturity']})")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.tolist(), pad=0.02, shrink=0.85,
                 label="Portfolio vega V^π (×10⁷)")
    fig.subplots_adjust(left=0.07, right=0.88, wspace=0.28)

    save_path = os.path.join(ORIGINAL_DIR, "figure03_convergence.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n→ Saved {os.path.relpath(save_path, ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Market simulation (placeholder)
# ─────────────────────────────────────────────────────────────────────────────
# 5.  Summary comparison table
# ─────────────────────────────────────────────────────────────────────────────
def step5_summary(raw_options, v0, spreads, options):
    banner("Step 5 · Summary comparison with paper")

    from hjb_solver import vpi_grid

    # ── Paper reference values (read from Figures 4-8 legends) ────────────
    paper_prices = {
        (8,1.0):2.06, (8,1.5):2.11, (8,2.0):2.16, (8,3.0):2.27,
        (9,1.0):1.21, (9,1.5):1.31, (9,2.0):1.41, (9,3.0):1.57,
        (10,1.0):0.58,(10,1.5):0.71,(10,2.0):0.83,(10,3.0):1.02,
        (11,1.0):0.21,(11,1.5):0.33,(11,2.0):0.44,(11,3.0):0.62,
        (12,1.0):0.06,(12,1.5):0.13,(12,2.0):0.21,(12,3.0):0.36,
    }
    paper_vegas = {
        (8,1.0):0.41, (8,1.5):0.46, (8,2.0):0.48, (8,3.0):0.46,
        (9,1.0):0.91, (9,1.5):0.84, (9,2.0):0.76, (9,3.0):0.65,
        (10,1.0):1.25,(10,1.5):1.05,(10,2.0):0.92,(10,3.0):0.76,
        (11,1.0):1.05,(11,1.5):0.96,(11,2.0):0.88,(11,3.0):0.75,
        (12,1.0):0.54,(12,1.5):0.65,(12,2.0):0.67,(12,3.0):0.65,
    }

    # ── A. Option prices & vegas ──────────────────────────────────────────
    print("\nA) Option prices and vegas vs paper (Figures 4-8 legends)")
    print()
    hdr = (f"{'K':>5} {'T':>5}  "
           f"{'C(ours)':>9} {'C(paper)':>9} {'ΔC%':>7}  "
           f"{'V(ours)':>9} {'V(paper)':>9} {'ΔV%':>7}")
    print(hdr)
    print("─" * len(hdr))

    max_price_err = 0.0
    max_vega_err  = 0.0
    for r in raw_options:
        K, Tm = r["strike"], r["maturity"]
        key   = (int(K), Tm)
        pp    = paper_prices.get(key, r["price"])
        pv    = paper_vegas.get(key, r["vega"])
        dp    = 100 * (r["price"] - pp) / max(abs(pp), 1e-12)
        dv    = 100 * (r["vega"]  - pv) / max(abs(pv), 1e-12)
        max_price_err = max(max_price_err, abs(dp))
        max_vega_err  = max(max_vega_err,  abs(dv))
        print(f"{K:>5.0f} {Tm:>5.1f}  "
              f"{r['price']:>9.4f} {pp:>9.4f} {dp:>+6.1f}%  "
              f"{r['vega']:>9.4f} {pv:>9.4f} {dv:>+6.1f}%")

    print(f"\n  Max |ΔC|: {max_price_err:.1f}%   Max |ΔV|: {max_vega_err:.1f}%")

    # ── B. Value function ─────────────────────────────────────────────────
    print("\nB) Value function v(0,ν,Vπ)  vs  paper Figure 2")
    print(f"  v_peak (ours)  = {v0.max():>10.0f}")
    print(f"  v_peak (paper) ≈ {'120,000':>10s}")
    print(f"  ratio          = {v0.max()/120000:>10.2f}")

    # ── C. Bid spread / price at Vπ=0 ────────────────────────────────────
    print("\nC) Bid spread / price at Vπ=0  vs  paper Figures 4-8")
    print()
    hdr2 = f"{'K':>5} {'T':>5}  {'bid/C (ours)':>13}"
    print(hdr2)
    print("─" * len(hdr2))
    for opt in options:
        key = (opt["strike"], opt["maturity"])
        s   = spreads[key]
        db  = np.interp(0.0, vpi_grid, s["delta_bid"])
        print(f"{opt['strike']:>5.0f} {opt['maturity']:>5.1f}  "
              f"{db/opt['price']:>13.5f}")

    # ── D. Figure inventory ───────────────────────────────────────────────
    print("\nD) Figures saved:")
    for label, d in [("original", ORIGINAL_DIR),
                     ("param_sweeps", SWEEP_DIR),
                     ("intensity", INTENSITY_DIR)]:
        if os.path.isdir(d):
            figs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
            print(f"  figures/{label}/  — {len(figs)} figures")
            for f in figs:
                print(f"    • {f}")


# ─────────────────────────────────────────────────────────────────────────────
# Build augmented option list (shared by steps 2-4)
# ─────────────────────────────────────────────────────────────────────────────
def _augment_options(raw):
    """Add λ_i, z_i, iv0 to raw Heston output; return list of dicts."""
    from black_scholes import implied_vol

    S0     = 10.0
    sigma0 = np.sqrt(0.0225)
    options = []
    for r in raw:
        K     = r["strike"]
        price = r["price"]
        vega  = r["vega"]
        lam   = 252 * 30 / (1 + 0.7 * abs(S0 - K))
        z     = 5e5 / price
        try:
            iv0 = implied_vol(price, S0, K, r["maturity"], sigma0=sigma0)
        except Exception:
            iv0 = 0.15
        options.append({
            "strike":   K,
            "maturity": r["maturity"],
            "price":    price,
            "vega":     vega,
            "lam":      lam,
            "z":        z,
            "iv0":      iv0,
        })
    return options


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Parameter sweeps — α, β
# ─────────────────────────────────────────────────────────────────────────────
def step7_param_sweeps(options):
    banner("Step 7 · Parameter sweeps  (α, β)")

    from param_sweeps import (
        sweep_alpha, sweep_beta, print_sweep_summary,
    )

    print(f"  Output directory: {os.path.relpath(SWEEP_DIR, ROOT)}/\n")

    print("  A) Alpha sweep")
    res_a, fnames_a = sweep_alpha(options, save_dir=SWEEP_DIR)
    print_sweep_summary(res_a, "alpha")

    print("\n  B) Beta sweep")
    res_b, fnames_b = sweep_beta(options, save_dir=SWEEP_DIR)
    print_sweep_summary(res_b, "beta")

    n_total = len(fnames_a) + len(fnames_b)
    print(f"\n  → {n_total} sweep figures saved to "
          f"{os.path.relpath(SWEEP_DIR, ROOT)}/")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Intensity families + Queue-Reactive CTMC
# ─────────────────────────────────────────────────────────────────────────────
def step8_intensity(options):
    banner("Step 8 · Intensity families + Queue-Reactive CTMC")

    from param_sweeps import (
        sweep_intensity, print_sweep_summary, run_queue_reactive,
    )

    print(f"  Output directory: {os.path.relpath(INTENSITY_DIR, ROOT)}/\n")

    print("  C) Intensity family sweep")
    res_c, fnames_c = sweep_intensity(options, save_dir=INTENSITY_DIR)
    print_sweep_summary(res_c, "intensity")

    print("\n  D) Queue-reactive L1 LOB model")
    fnames_d = run_queue_reactive(save_dir=INTENSITY_DIR)

    n_total = len(fnames_c) + len(fnames_d)
    print(f"\n  → {n_total} intensity figures saved to "
          f"{os.path.relpath(INTENSITY_DIR, ROOT)}/")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    wall_t0 = time.time()

    # Prepare output directories
    for d in [ORIGINAL_DIR, SWEEP_DIR, INTENSITY_DIR]:
        os.makedirs(d, exist_ok=True)

    # 1 — Heston pricer
    raw = step1_heston_pricer()

    # Build augmented option list used by steps 2-4
    options = _augment_options(raw)

    # 2 — HJB solver
    v_all = step2_hjb_solver(options)
    v0    = v_all[0]

    # 3 — Optimal spreads (Figures 4-13)
    spreads = step3_optimal_spreads(v0, options)

    # 4 — Convergence (Figure 3)
    step4_convergence(v_all, options)

    # 5 — Summary
    step5_summary(raw, v0, spreads, options)

    # 6 — Parameter sweeps (α, β)
    step7_param_sweeps(options)

    # 7 — Intensity families + Queue-Reactive CTMC
    step8_intensity(options)

    # Done
    elapsed = time.time() - wall_t0
    banner(f"Pipeline complete — total wall time {elapsed:.1f}s")
    for label, d in [("original", ORIGINAL_DIR),
                     ("param_sweeps", SWEEP_DIR),
                     ("intensity", INTENSITY_DIR)]:
        if os.path.isdir(d):
            n = len([f for f in os.listdir(d) if f.endswith(".png")])
            print(f"  figures/{label:15s}  {n} figures")
    print()


if __name__ == "__main__":
    main()
