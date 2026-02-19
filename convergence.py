"""
convergence.py
==============
Reproduce Figure 3 from Baldacci, Bergault & Guéant (2020):
  Optimal mid-to-bid spread for option 1 (K=8, T_opt=1) at ν=ν₀
  as a function of time t, for several portfolio-vega levels.

All curves should converge to the same value near t=T and fan out to
their stationary levels well before t=0, confirming the chosen horizon
T is long enough for the quotes to stabilise.
"""

import numpy as np
import matplotlib.pyplot as plt

from hjb_solver import (
    nu_grid, vpi_grid, N_T, N_VPI, V_BAR,
    ALPHA, BETA, T, dt, solve_hjb, hamiltonian,
)
from optimal_spreads import (
    get_options, hamiltonian_prime, lambda_inverse, DELTA_INF, NU0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Compute the bid spread at every time step for one option, one ν-row
# ─────────────────────────────────────────────────────────────────────────────
def bid_spread_vs_time(v_all, opt, nu_idx):
    """
    For a single option (bid side, psi=-1), compute delta_bid(t, Vpi)
    at every saved time step.

    Parameters
    ----------
    v_all  : ndarray (N_T+1, N_NU, N_VPI)  — full value-function history
    opt    : dict with keys 'vega', 'z', 'lam'
    nu_idx : int — row index into nu_grid

    Returns
    -------
    spreads : ndarray (N_T+1, N_VPI)
        spreads[n, j] = optimal mid-to-bid at t = n*dt, Vpi = vpi_grid[j]
    """
    V_i = opt['vega']
    z   = opt['z']
    lam = opt['lam']
    psi = -1.0               # bid side
    nu  = nu_grid[nu_idx]    # instantaneous variance at this grid point

    shift       = psi * z * V_i
    vpi_shifted = vpi_grid - shift
    in_domain   = np.abs(vpi_grid - shift) <= V_BAR

    n_steps = v_all.shape[0]
    spreads = np.full((n_steps, N_VPI), np.nan)

    for n in range(n_steps):
        v_row = v_all[n, nu_idx, :]          # (N_VPI,)

        # interpolate v at shifted Vpi
        v_shifted = np.interp(
            vpi_shifted, vpi_grid, v_row,
            left=v_row[0], right=v_row[-1],
        )

        p = (v_row - v_shifted) / z

        Hprime = hamiltonian_prime(p, lam, V_i)
        arg    = np.clip(-Hprime, 1e-12, lam - 1e-12)
        delta  = lambda_inverse(arg, lam, V_i)
        delta  = np.maximum(delta, DELTA_INF)
        delta  = np.where(in_domain, delta, DELTA_INF)

        spreads[n, :] = delta

    return spreads


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # ── Build options & solve HJB ─────────────────────────────────────────────
    options = get_options()

    # option 1 = K=8, T_opt=1  (first in the list)
    opt1 = next(o for o in options if o['strike'] == 8 and o['maturity'] == 1.0)
    print(f"Option 1: K={opt1['strike']}, T_opt={opt1['maturity']}, "
          f"price={opt1['price']:.3f}, vega={opt1['vega']:.3f}")

    import time as _time
    print("Solving HJB …")
    t0     = _time.time()
    v_all  = solve_hjb(options)        # (N_T+1, N_NU, N_VPI)
    print(f"Done in {_time.time() - t0:.1f}s")

    # ── Compute bid spread at every time step ────────────────────────────────
    nu_idx = np.argmin(np.abs(nu_grid - NU0))
    print(f"Using nu_grid[{nu_idx}] = {nu_grid[nu_idx]:.5f}  (ν₀ = {NU0})")

    spreads = bid_spread_vs_time(v_all, opt1, nu_idx)   # (N_T+1, N_VPI)

    # ── Time axis ────────────────────────────────────────────────────────────
    t_grid = np.linspace(0, T, N_T + 1)   # v_all[n] corresponds to t = n*dt

    # ── Select ~25 evenly spaced Vpi levels ──────────────────────────────────
    n_lines = 25
    vpi_indices = np.linspace(0, N_VPI - 1, n_lines, dtype=int)
    # remove duplicates (if N_VPI < n_lines)
    vpi_indices = np.unique(vpi_indices)

    # ── Normalise spread by option price ─────────────────────────────────────
    price = opt1['price']

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={'width_ratios': [3, 2]})

    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=vpi_grid.min() / 1e7, vmax=vpi_grid.max() / 1e7)

    for ax_idx, ax in enumerate(axes):
        for j in vpi_indices:
            col = cmap(norm(vpi_grid[j] / 1e7))
            spread_ratio = spreads[:, j] / price
            # skip lines stuck at the sentinel
            if np.all(spread_ratio < (DELTA_INF + 0.1) / price):
                continue
            ax.plot(t_grid, spread_ratio, color=col, linewidth=0.8, alpha=0.85)

        ax.set_xlabel('Time  t  (years)')
        ax.set_ylabel('Optimal mid-to-bid / price')
        ax.axvline(T, color='gray', lw=0.6, ls=':', alpha=0.7)

        if ax_idx == 1:
            # zoom into the last 30% of the horizon to see the fan-out clearly
            ax.set_xlim(0.7 * T, T * 1.02)
            ax.set_title('Zoom: near terminal time T')
        else:
            ax.set_title(
                f'Figure 3: Convergence of bid spread  '
                f'(K={opt1["strike"]}, T_opt={opt1["maturity"]},  ν₀={NU0})'
            )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), pad=0.02, shrink=0.85)
    cbar.set_label('Portfolio vega  V^π  (×10⁷)')

    fig.subplots_adjust(left=0.07, right=0.88, wspace=0.28)

    fname = 'figure3_convergence.png'
    plt.savefig(fname, dpi=150)
    print(f"Saved {fname}")
    plt.show()

    # ── Print convergence diagnostic ─────────────────────────────────────────
    # Compare spread at t=0 vs a few steps in
    mid_j = N_VPI // 2      # Vpi ≈ 0
    s_t0   = spreads[0,  mid_j] / price
    s_10pct = spreads[max(1, N_T // 10), mid_j] / price
    s_half  = spreads[N_T // 2, mid_j] / price
    s_T     = spreads[N_T, mid_j] / price

    print(f"\nBid spread / price at Vpi ≈ {vpi_grid[mid_j]:.0f}:")
    print(f"  t = 0          : {s_t0:.6f}")
    print(f"  t = 0.1·T      : {s_10pct:.6f}")
    print(f"  t = 0.5·T      : {s_half:.6f}")
    print(f"  t = T (terminal): {s_T:.6f}")
    print(f"  |s(0) − s(0.1T)| / s(0) = {abs(s_t0 - s_10pct) / max(abs(s_t0), 1e-12):.2e}")
    print()
    print("If the relative change between t=0 and t=0.1·T is small (< 1%),")
    print("the quotes have converged well before the start of the window,")
    print("confirming the horizon T is long enough.")
