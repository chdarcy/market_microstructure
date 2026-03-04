import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

from hjb_solver import (
    nu_grid, vpi_grid, N_VPI, V_BAR,
    ALPHA, BETA, hamiltonian
)
from black_scholes import bs_call, implied_vol

DELTA_INF = -1.0
S0        = 10.0
NU0       = 0.0225


def get_options():
    try:
        from heston_pricer import get_option_data
        raw = get_option_data()
    except ImportError:
        import itertools
        STRIKES    = [8, 9, 10, 11, 12]
        MATURITIES = [1.0, 1.5, 2.0, 3.0]
        approx_prices = {
            (8,1):2.06,(8,1.5):2.11,(8,2):2.16,(8,3):2.27,
            (9,1):1.21,(9,1.5):1.31,(9,2):1.41,(9,3):1.57,
            (10,1):0.58,(10,1.5):0.71,(10,2):0.83,(10,3):1.02,
            (11,1):0.21,(11,1.5):0.33,(11,2):0.44,(11,3):0.62,
            (12,1):0.06,(12,1.5):0.13,(12,2):0.21,(12,3):0.36,
        }
        approx_vegas = {
            (8,1):0.41,(8,1.5):0.46,(8,2):0.48,(8,3):0.46,
            (9,1):0.91,(9,1.5):0.84,(9,2):0.76,(9,3):0.65,
            (10,1):1.25,(10,1.5):1.05,(10,2):0.92,(10,3):0.76,
            (11,1):1.05,(11,1.5):0.96,(11,2):0.88,(11,3):0.75,
            (12,1):0.54,(12,1.5):0.65,(12,2):0.67,(12,3):0.65,
        }
        approx_iv = {
            (8,1):0.1622,(8,1.5):0.1583,(8,2):0.1563,(8,3):0.1544,
            (9,1):0.1538,(9,1.5):0.1518,(9,2):0.1515,(9,3):0.1508,
            (10,1):0.1460,(10,1.5):0.1459,(10,2):0.1468,(10,3):0.1476,
            (11,1):0.1392,(11,1.5):0.1409,(11,2):0.1427,(11,3):0.1448,
            (12,1):0.1344,(12,1.5):0.1367,(12,2):0.1395,(12,3):0.1423,
        }
        raw = [
            {'strike': K, 'maturity': Tm,
             'price':  approx_prices[(K, Tm)],
             'vega':   approx_vegas[(K, Tm)],
             'iv':     approx_iv[(K, Tm)]}
            for K, Tm in itertools.product(STRIKES, MATURITIES)
        ]

    options = []
    for opt in raw:
        K     = opt['strike']
        price = opt['price']
        vega  = opt['vega']
        lam   = 252 * 30 / (1 + 0.7 * abs(S0 - K))
        z     = 5e5 / price
        try:
            iv0 = opt.get('iv') or implied_vol(price, S0, K, opt['maturity'], sigma0=0.15)
        except Exception:
            iv0 = 0.15
        options.append({
            'strike':   K,
            'maturity': opt['maturity'],
            'price':    price,
            'vega':     vega,
            'lam':      lam,
            'z':        z,
            'iv0':      iv0,
        })
    return options

def hamiltonian_prime(p, lam, V_i, eps=1e-6):
    """dH/dp via central FD."""
    return (hamiltonian(p + eps, lam, V_i) - hamiltonian(p - eps, lam, V_i)) / (2 * eps)

def lambda_inverse(y, lam, V_i):
    """Closed-form Λ⁻¹(y): δ = V/β · (ln(λ/y − 1) − α)."""
    bV    = BETA / V_i
    ratio = np.clip(lam / np.clip(y, 1e-12, None) - 1.0, 1e-12, None)
    return (np.log(ratio) - ALPHA) / bV

def optimal_spread(v0, opt, psi, nu_idx=None):
    """
    δ*(Vπ) at t=0 for one option, one side (psi=+1 ask, -1 bid).
    Returns array of shape (N_VPI,).
    """
    if nu_idx is None:
        nu_idx = np.argmin(np.abs(nu_grid - NU0))

    nu  = nu_grid[nu_idx]
    V_i = opt['vega']
    z   = opt['z']
    lam = opt['lam']

    v_row = v0[nu_idx, :]

    shift      = psi * z * V_i
    vpi_shifted = vpi_grid - shift

    v_shifted = np.interp(
        vpi_shifted, vpi_grid, v_row,
        left=v_row[0], right=v_row[-1]
    )

    in_domain = np.abs(vpi_grid - shift) <= V_BAR

    p = (v_row - v_shifted) / z

    # H'(p) = −Λ(δ*(p)) by envelope theorem → δ* = Λ⁻¹(−H')
    Hprime = hamiltonian_prime(p, lam, V_i)

    arg   = -Hprime
    arg   = np.clip(arg, 1e-12, lam - 1e-12)
    delta = lambda_inverse(arg, lam, V_i)
    delta = np.maximum(delta, DELTA_INF)

    delta = np.where(in_domain, delta, DELTA_INF)

    return delta

def compute_all_spreads(v0, options):
    """Dict keyed by (K, T) → {delta_bid, delta_ask, price, vega, iv0, ...}."""
    spreads = {}
    nu_idx  = np.argmin(np.abs(nu_grid - NU0))

    for opt in options:
        key = (opt['strike'], opt['maturity'])
        spreads[key] = {
            'delta_bid': optimal_spread(v0, opt, psi=-1, nu_idx=nu_idx),
            'delta_ask': optimal_spread(v0, opt, psi=+1, nu_idx=nu_idx),
            'price':     opt['price'],
            'vega':      opt['vega'],
            'iv0':       opt['iv0'],
            'strike':    opt['strike'],
            'maturity':  opt['maturity'],
        }
    return spreads

def plot_ask_to_mid(spreads, options, save_dir=None):
    """Ask-to-mid / price vs Vπ (mirrors Figures 4-8 for the ask side)."""
    STRIKES    = [8, 9, 10, 11, 12]
    MATURITIES = [1.0, 1.5, 2.0, 3.0]
    markers    = ['*', '^', 'o', 's']
    vpi_plot   = vpi_grid / 1e7

    for fig_idx, K in enumerate(STRIKES):
        fig, ax = plt.subplots(figsize=(8, 5))
        for m_idx, Tm in enumerate(MATURITIES):
            key = (K, Tm)
            if key not in spreads:
                continue
            s    = spreads[key]
            da   = s['delta_ask']
            mask = da > DELTA_INF + 0.1
            ax.scatter(vpi_plot[mask], da[mask] / s['price'],
                       s=10, marker=markers[m_idx],
                       label=f"(K,T)=({K},{Tm})  C={s['price']:.2f}")

        ax.axvline(0, color='gray', lw=0.8, ls='--')
        ax.axhline(0, color='gray', lw=0.8, ls=':')
        ax.set_xlabel('Portfolio vega  (×10⁷)')
        ax.set_ylabel('Optimal ask-to-mid / price')
        ax.set_title(f'Extension A: Ask-to-mid / price  (K={K})')
        ax.legend(fontsize=7)
        # Annotate the negative region
        ax.annotate('δ_a < 0: MM posts below mid\n(aggressive selling to reduce long vega)',
                    xy=(0.98, 0.02), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=6.5,
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))
        plt.tight_layout()
        fname = f'ext_ask_to_mid_K{K}.png'
        dst   = os.path.join(save_dir, fname) if save_dir else fname
        plt.savefig(dst, dpi=150)
        plt.close()
        print(f"  → {fname}")

def plot_bid_ask_spread(spreads, options, save_dir=None):
    """Full bid-ask spread (δ_a + δ_b) / price vs Vπ."""
    STRIKES    = [8, 9, 10, 11, 12]
    MATURITIES = [1.0, 1.5, 2.0, 3.0]
    markers    = ['*', '^', 'o', 's']
    vpi_plot   = vpi_grid / 1e7

    for fig_idx, K in enumerate(STRIKES):
        fig, ax = plt.subplots(figsize=(8, 5))
        for m_idx, Tm in enumerate(MATURITIES):
            key = (K, Tm)
            if key not in spreads:
                continue
            s     = spreads[key]
            da    = s['delta_ask']
            db    = s['delta_bid']
            total = da + db
            mask  = (da > DELTA_INF + 0.1) & (db > DELTA_INF + 0.1)
            ax.scatter(vpi_plot[mask], total[mask] / s['price'],
                       s=10, marker=markers[m_idx],
                       label=f"(K,T)=({K},{Tm})  C={s['price']:.2f}")

        ax.axvline(0, color='gray', lw=0.8, ls='--')
        ax.set_xlabel('Portfolio vega  (×10⁷)')
        ax.set_ylabel('Bid–ask spread / price')
        ax.set_title(f'Extension B: Bid–ask spread / price  (K={K})')
        ax.legend(fontsize=7)
        plt.tight_layout()
        fname = f'ext_spread_K{K}.png'
        dst   = os.path.join(save_dir, fname) if save_dir else fname
        plt.savefig(dst, dpi=150)
        plt.close()
        print(f"  → {fname}")

def plot_spread_vs_strike(spreads, options, save_dir=None):
    """Spread vs strike at fixed Vπ levels for each maturity."""
    STRIKES    = [8, 9, 10, 11, 12]
    MATURITIES = [1.0, 1.5, 2.0, 3.0]

    # Fixed Vπ levels as fractions of V_bar
    vpi_fracs = [-0.8, -0.4, 0.0, 0.4, 0.8]
    vpi_vals  = [f * V_BAR for f in vpi_fracs]
    labels    = [f"Vπ={f:+.1f}·V̄" for f in vpi_fracs]
    markers   = ['v', 's', 'o', '^', 'D']
    colors    = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for Tm in MATURITIES:
        fig, ax = plt.subplots(figsize=(8, 5))
        for v_idx, (vpi_val, lab) in enumerate(zip(vpi_vals, labels)):
            K_arr     = []
            spread_arr = []
            for K in STRIKES:
                key = (K, Tm)
                if key not in spreads:
                    continue
                s  = spreads[key]
                da_raw = s['delta_ask']
                db_raw = s['delta_bid']
                valid = (da_raw > DELTA_INF + 0.1) & (db_raw > DELTA_INF + 0.1)
                if valid.sum() < 2:
                    continue
                vpi_valid = vpi_grid[valid]
                # Skip if requested Vπ is outside the valid range
                if vpi_val < vpi_valid[0] or vpi_val > vpi_valid[-1]:
                    continue
                da = np.interp(vpi_val, vpi_valid, da_raw[valid])
                db = np.interp(vpi_val, vpi_valid, db_raw[valid])
                total = da + db
                if total > 0:
                    K_arr.append(K)
                    spread_arr.append(total / s['price'])

            if K_arr:
                ax.plot(K_arr, spread_arr, marker=markers[v_idx],
                        color=colors[v_idx], label=lab, lw=1.2, ms=7)

        ax.set_xlabel('Strike K')
        ax.set_ylabel('Bid–ask spread / price')
        ax.set_title(f'Extension C: Spread vs strike  (T={Tm})')
        ax.legend(fontsize=8)
        plt.tight_layout()
        fname = f'ext_spread_vs_strike_T{Tm}.png'
        dst   = os.path.join(save_dir, fname) if save_dir else fname
        plt.savefig(dst, dpi=150)
        plt.close()
        print(f"  → {fname}")

def print_short_vega_commentary(spreads, options):
    """Spread behaviour summary at very negative portfolio vega."""
    print()
    print("─" * 72)
    print("  Net short-vega (Vπ ≪ 0) behaviour summary")
    print("─" * 72)
    print("""
  Vπ < 0 (short vega):
    δ_b tight (eager to buy), δ_a wide (reluctant to sell).
    Total spread wider than at Vπ=0, driven by the ask side.
    Strongest for high-vega (ATM, long maturity) options.
""")
    # Numerical example
    vpi_neg = -0.8 * V_BAR
    vpi_0   = 0.0
    print(f"  {'K':>5} {'T':>5}  {'spread(Vπ=−0.8V̄)/C':>22}  "
          f"{'spread(Vπ=0)/C':>16}  {'ratio':>7}")
    print("  " + "─" * 70)
    STRIKES    = [8, 9, 10, 11, 12]
    MATURITIES = [1.0, 2.0, 3.0]
    for K in STRIKES:
        for Tm in MATURITIES:
            key = (K, Tm)
            if key not in spreads:
                continue
            s   = spreads[key]
            da_raw = s['delta_ask']
            db_raw = s['delta_bid']
            valid  = (da_raw > DELTA_INF + 0.1) & (db_raw > DELTA_INF + 0.1)
            if valid.sum() < 2:
                print(f"  {K:>5} {Tm:>5.1f}  {'(boundary)':>22}  "
                      f"{'—':>16}  {'—':>7}")
                continue
            vpi_valid = vpi_grid[valid]
            if vpi_neg < vpi_valid[0] or vpi_0 > vpi_valid[-1]:
                print(f"  {K:>5} {Tm:>5.1f}  {'(boundary)':>22}  "
                      f"{'—':>16}  {'—':>7}")
                continue
            da_neg = np.interp(vpi_neg, vpi_valid, da_raw[valid])
            db_neg = np.interp(vpi_neg, vpi_valid, db_raw[valid])
            da_0   = np.interp(vpi_0,   vpi_valid, da_raw[valid])
            db_0   = np.interp(vpi_0,   vpi_valid, db_raw[valid])
            sp_neg = (da_neg + db_neg) / s['price']
            sp_0   = (da_0   + db_0)   / s['price']
            # Skip if either total spread is non-positive
            if sp_neg <= 0 or sp_0 <= 0:
                print(f"  {K:>5} {Tm:>5.1f}  {'(boundary)':>22}  "
                      f"{'—':>16}  {'—':>7}")
                continue
            if sp_0 > 1e-6:
                ratio = sp_neg / sp_0
            else:
                ratio = float('nan')
            print(f"  {K:>5} {Tm:>5.1f}  {sp_neg:>22.5f}  "
                  f"{sp_0:>16.5f}  {ratio:>7.2f}×")
    print()
