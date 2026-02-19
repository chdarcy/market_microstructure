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
def hamiltonian_prime(p, lam, nu, eps=1e-6):
    """Numerical derivative of H w.r.t. p (nu = instantaneous variance)."""
    return (hamiltonian(p + eps, lam, nu) - hamiltonian(p - eps, lam, nu)) / (2 * eps)


# ── Lambda inverse: given y find delta s.t. Lambda(delta) = y ────────────────
def lambda_inverse(y, lam, nu):
    """
    Lambda(delta) = lam / (1 + exp(alpha + beta/sqrt(nu) * delta))
    Solve for delta: exp(alpha + beta/sqrt(nu) * delta) = lam/y - 1
    => delta = sqrt(nu)/beta * (log(lam/y - 1) - alpha)
    Direct closed-form inversion.
    Returns delta (same shape as y).
    """
    bV    = BETA / np.sqrt(nu)
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
    Hprime = hamiltonian_prime(p, lam, nu)

    # optimal spread: delta* = Lambda^{-1}(-H'(p))
    arg   = -Hprime
    # clamp arg to (0, lam) so Lambda_inverse is defined
    arg   = np.clip(arg, 1e-12, lam - 1e-12)
    delta = lambda_inverse(arg, lam, nu)
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