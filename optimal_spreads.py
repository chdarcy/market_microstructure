import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

# ── Import from sibling modules ───────────────────────────────────────────────
from hjb_solver import (
    nu_grid, vpi_grid, N_NU, N_VPI, V_BAR,
    ALPHA, BETA, GAMMA, XI, T,
    solve_hjb, hamiltonian
)
from black_scholes import bs_call, implied_vol

# ── Constants ─────────────────────────────────────────────────────────────────
DELTA_INF = -1.0
S0        = 10.0
NU0       = 0.0225


# ── Option data (from paper §4.1 or heston_pricer) ───────────────────────────
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
        z     = 5e5 / price             # paper §4.1: z^i = 5×10^5 / S^i_0
        # initial IV either from heston_pricer or hardcoded
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


# ── H'(p) via finite difference ───────────────────────────────────────────────
def hamiltonian_prime(p, lam, V_i, eps=1e-6):
    """Numerical derivative of H w.r.t. p (parametrised by option vega V_i)."""
    return (hamiltonian(p + eps, lam, V_i) - hamiltonian(p - eps, lam, V_i)) / (2 * eps)


# ── Lambda inverse: given y find delta s.t. Lambda(delta) = y ────────────────
def lambda_inverse(y, lam, V_i):
    """
    Lambda(delta) = lam / (1 + exp(alpha + beta/V_i * delta))
    Solve for delta: exp(alpha + beta/V_i * delta) = lam/y - 1
    => delta = V_i/beta * (log(lam/y - 1) - alpha)
    Direct closed-form inversion.
    Returns delta (same shape as y).
    """
    bV    = BETA / V_i
    ratio = np.clip(lam / np.clip(y, 1e-12, None) - 1.0, 1e-12, None)
    return (np.log(ratio) - ALPHA) / bV


# ── Optimal spread for one option, one side ───────────────────────────────────
def optimal_spread(v0, opt, psi, nu_idx=None):
    """
    Compute delta*(Vpi) at t=0 for a single option and side.

    psi = +1 for ask, -1 for bid.
    nu_idx: which nu row to use (default: closest to nu0=0.0225).
    Returns array of shape (N_VPI,).
    """
    if nu_idx is None:
        nu_idx = np.argmin(np.abs(nu_grid - NU0))

    nu  = nu_grid[nu_idx]        # instantaneous variance at this grid point
    V_i = opt['vega']            # dC^i/d(sqrt(nu)) — used for shift size only
    z   = opt['z']
    lam = opt['lam']

    v_row = v0[nu_idx, :]   # (N_VPI,)

    # shifted vpi after trade
    shift      = psi * z * V_i
    vpi_shifted = vpi_grid - shift

    # interpolate v at shifted points
    v_shifted = np.interp(
        vpi_shifted, vpi_grid, v_row,
        left=v_row[0], right=v_row[-1]
    )

    # indicator
    in_domain = np.abs(vpi_grid - shift) <= V_BAR

    # p = (v - v_shifted) / z
    p = (v_row - v_shifted) / z

    # H'(p) = -Lambda(delta*(p))  by the envelope theorem
    Hprime = hamiltonian_prime(p, lam, V_i)

    # optimal spread: delta* = Lambda^{-1}(-H'(p))
    arg   = -Hprime
    # clamp arg to (0, lam) so Lambda_inverse is defined
    arg   = np.clip(arg, 1e-12, lam - 1e-12)
    delta = lambda_inverse(arg, lam, V_i)
    delta = np.maximum(delta, DELTA_INF)

    # where indicator is 0, spread is effectively infinity (use DELTA_INF)
    delta = np.where(in_domain, delta, DELTA_INF)

    return delta


# ── Compute all spreads ────────────────────────────────────────────────────────
def compute_all_spreads(v0, options):
    """
    Returns dict keyed by (strike, maturity) with values:
      {'delta_bid': array(N_VPI), 'delta_ask': array(N_VPI)}
    """
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


# ── Figures 4-8: mid-to-bid / price ───────────────────────────────────────────
def plot_figures_4_8(spreads, options):
    STRIKES    = [8, 9, 10, 11, 12]
    MATURITIES = [1.0, 1.5, 2.0, 3.0]
    markers    = ['*', '^', 'o', 's']
    vpi_plot   = vpi_grid / 1e7

    for fig_num, K in enumerate(STRIKES, start=4):
        fig, ax = plt.subplots(figsize=(8, 5))
        for m_idx, Tm in enumerate(MATURITIES):
            key  = (K, Tm)
            if key not in spreads:
                continue
            s     = spreads[key]
            db    = s['delta_bid']
            mask  = db > DELTA_INF + 0.1          # exclude boundary sentinel values
            ratio = db[mask] / s['price']
            label = (f"(K,T)=({K},{Tm}) "
                     f"price={s['price']:.2f} vega={s['vega']:.3f}")
            ax.scatter(vpi_plot[mask], ratio, s=10, marker=markers[m_idx], label=label)

        ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_xlabel('Portfolio vega  (×10⁷)')
        ax.set_ylabel('Optimal mid-to-bid divided by price')
        ax.set_title(f'Figure {fig_num}: Optimal mid-to-bid / price  (K={K})')
        ax.legend(fontsize=7)
        plt.tight_layout()
        fname = f'figure{fig_num}_K{K}.png'
        plt.savefig(fname, dpi=150)
        print(f"Saved {fname}")
        plt.show()


# ── Figures 9-13: relative bid IV ─────────────────────────────────────────────
def plot_figures_9_13(spreads, options):
    STRIKES    = [8, 9, 10, 11, 12]
    MATURITIES = [1.0, 1.5, 2.0, 3.0]
    markers    = ['*', '^', 'o', 's']
    vpi_plot   = vpi_grid / 1e7

    for fig_num, K in enumerate(STRIKES, start=9):
        fig, ax = plt.subplots(figsize=(8, 5))
        for m_idx, Tm in enumerate(MATURITIES):
            key = (K, Tm)
            if key not in spreads:
                continue
            s    = spreads[key]
            iv0  = s['iv0']
            db   = s['delta_bid']
            mask = db > DELTA_INF + 0.1           # exclude boundary sentinel values
            bid_prices = s['price'] - db[mask]    # bid = mid - delta

            # invert BS for each bid price
            bid_iv = np.array([
                implied_vol(max(bp, 1e-6), S0, K, Tm)
                for bp in bid_prices
            ])
            rel_iv = bid_iv / iv0

            label = (f"(K,T)=({K},{Tm}) "
                     f"price={s['price']:.2f} vega={s['vega']:.3f} "
                     f"IV={iv0:.4f}")
            ax.scatter(vpi_plot[mask], rel_iv, s=10, marker=markers[m_idx], label=label)

        ax.axhline(1.0, color='gray', linewidth=0.8, linestyle='--')
        ax.axvline(0,   color='gray', linewidth=0.8, linestyle='--')
        ax.set_xlabel('Portfolio vega  (×10⁷)')
        ax.set_ylabel('IV of optimal bid / initial IV')
        ax.set_title(f'Figure {fig_num}: Relative bid IV  (K={K})')
        ax.legend(fontsize=7)
        plt.tight_layout()
        fname = f'figure{fig_num}_K{K}.png'
        plt.savefig(fname, dpi=150)
        print(f"Saved {fname}")
        plt.show()


# ── Extension A: ask-to-mid / price vs portfolio vega ─────────────────────────
def plot_ask_to_mid(spreads, options, save_dir=None):
    """
    Mirrors Figures 4-8 but for the *ask* side: delta_ask / price vs Vπ.

    δ_ask is the optimal ask-to-mid offset: the market maker posts
    ask = mid + δ_ask.  When the portfolio is net *long* vega (Vπ > 0)
    the MM wants to sell (reduce inventory), so δ_ask shrinks to attract
    sellers.  When Vπ < 0 the MM prefers not to sell, so δ_ask widens.
    """
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


# ── Extension B: bid–ask spread / price vs portfolio vega ─────────────────────
def plot_bid_ask_spread(spreads, options, save_dir=None):
    """
    Full bid–ask spread = δ_ask + δ_bid, divided by option price.

    Key observations:
    • At Vπ = 0 the spread is symmetric and close to its minimum — the MM
      has no inventory pressure.
    • As |Vπ| grows the spread widens: the side that would *increase* the
      imbalance sees a larger offset, dominating the total.
    • The spread is *not* perfectly symmetric about Vπ = 0 because of the
      variance risk premium term (aP − aQ), which tilts the value surface.
    """
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
            total = da + db            # full spread
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


# ── Extension C: spread vs strike at fixed portfolio-vega levels ──────────────
def plot_spread_vs_strike(spreads, options, save_dir=None):
    """
    For each maturity, plot the full spread (δ_a + δ_b) vs strike at
    five fixed portfolio-vega levels:

      Vπ / V_bar ∈ {−0.8, −0.4, 0, +0.4, +0.8}

    This shows how the strike smile of optimal spreads changes with
    inventory.  Commentary on the net-short-vega regime (Vπ ≪ 0):

    When the market maker is heavily net short vega, buying options reduces
    the magnitude of the (negative) vega exposure.  Therefore:
      • The *bid* offset δ_b shrinks — the MM is eager to buy and posts
        aggressive (tight) bids.
      • The *ask* offset δ_a widens — the MM is reluctant to sell more
        options that would deepen the short-vega position.
      • The total spread is wider than at Vπ = 0, dominated by the
        wide ask side.
      • The effect is strongest for high-vega (near ATM, long maturity)
        options because each trade shifts Vπ by z·V_i, so high vega
        means a larger inventory change per lot.
    Conversely, at very positive Vπ the roles flip: asks are tight and
    bids are wide.
    """
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
                # Mask out sentinel values before interpolation so np.interp
                # never blends valid data with DELTA_INF = -1.0
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


# ── Extension commentary: net-short-vega analysis ────────────────────────────
def print_short_vega_commentary(spreads, options):
    """
    Print a summary of spread behaviour at very negative portfolio vega.
    """
    print()
    print("─" * 72)
    print("  Extension: Net short-vega (Vπ ≪ 0) behaviour summary")
    print("─" * 72)
    print("""
  When the market maker is heavily net SHORT vega (Vπ < 0):

  1. Bid offset δ_b is TIGHT (small):
     – The MM wants to BUY options to reduce the magnitude of the
       negative vega exposure, so it posts aggressive bids.

  2. Ask offset δ_a is WIDE (large):
     – Selling more options would deepen the short-vega position,
       increasing inventory risk, so the MM discourages selling by
       posting a wide ask.

  3. Total spread is WIDER than at Vπ = 0:
     – The wide ask side dominates; total spread = δ_a + δ_b > spread(0).

  4. Asymmetry across strikes:
     – High-vega options (ATM, long maturity) show the largest
       spread widening because each fill shifts Vπ by z·V_i.
     – Deep ITM/OTM options with small vega are less affected.

  5. Variance risk premium tilt:
     – The (aP − aQ) drift term is positive near ν₀, so the value
       function is tilted upward for Vπ > 0. This slightly reduces
       the penalty at negative Vπ relative to a symmetric model.
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
            # Check both query points lie within valid range
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


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    options = get_options()

    # ── Run HJB solver ────────────────────────────────────────────────────────
    print("Running HJB solver …")
    import time
    t0    = time.time()
    v_all = solve_hjb(options)
    print(f"HJB solved in {time.time()-t0:.1f}s")

    v0 = v_all[0]   # value function at t=0, shape (N_NU, N_VPI)

    # ── Compute optimal spreads ───────────────────────────────────────────────
    print("Computing optimal spreads …")
    spreads = compute_all_spreads(v0, options)

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'K':>5} {'T':>5} {'price':>8} {'vega':>8} "
          f"{'bid@Vpi=0':>12} {'bid/price':>12}")
    print("-" * 60)
    for opt in options:
        key = (opt['strike'], opt['maturity'])
        s   = spreads[key]
        # interpolate to exact Vpi=0
        db  = np.interp(0.0, vpi_grid, s['delta_bid'])
        print(f"{opt['strike']:>5} {opt['maturity']:>5} "
              f"{opt['price']:>8.3f} {opt['vega']:>8.3f} "
              f"{db:>12.5f} {db/opt['price']:>12.5f}")

    # ── Figures 4-8 ───────────────────────────────────────────────────────────
    print("\nPlotting Figures 4-8 …")
    plot_figures_4_8(spreads, options)

    # ── Figures 9-13 ──────────────────────────────────────────────────────────
    print("Plotting Figures 9-13 …")
    plot_figures_9_13(spreads, options)