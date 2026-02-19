import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

# ── Grid parameters ────────────────────────────────────────────────────────────
N_T    = 180
N_NU   = 30
N_VPI  = 80
T      = 0.0012             # paper §4.1: T = 0.0012 year (i.e. 0.3 day)
XI     = 0.2
GAMMA  = 1e-3
KAPPA_P, THETA_P = 2.0, 0.04
KAPPA_Q, THETA_Q = 3.0, 0.0225
V_BAR  = 1e7
ALPHA  = 0.7
BETA   = 150.0
NU0    = 0.0225            # initial / reference variance for fill probability

nu_grid  = np.linspace(0.0144, 0.0324, N_NU)   # shape (N_NU,)
vpi_grid = np.linspace(-V_BAR, V_BAR, N_VPI)   # shape (N_VPI,)
dt       = T / N_T
dnu      = nu_grid[1] - nu_grid[0]

# Pre-compute grid of (nu, vpi) for broadcast arithmetic
NU  = nu_grid[:, None]   # (N_NU, 1)
VPI = vpi_grid[None, :]  # (1,  N_VPI)


# ── Drift helpers ──────────────────────────────────────────────────────────────
def aP(nu): return KAPPA_P * (THETA_P - nu)
def aQ(nu): return KAPPA_Q * (THETA_Q - nu)


# ── Hamiltonian H(p) via vectorised Newton ─────────────────────────────────────
def hamiltonian(p, lam, V_i, n_iter=15):
    """
    H^{i,j}(p) = sup_{delta} Lambda(delta)*(delta - p)
    Lambda(delta) = lam / (1 + exp(alpha + beta/V_i * delta))
    where V_i is the vega of option i (paper §4.1).
    FOC: u*(delta-p) = V_i/beta  where u = expit(alpha + beta/V_i * delta)
    Newton:  delta <- delta - u*(delta-p) + V_i/beta
    Returns H value (same shape as p).
    """
    bV     = BETA / V_i        # beta / V_i
    VoB    = V_i / BETA        # V_i / beta
    delta  = p + 2.0 * VoB    # initial guess

    for _ in range(n_iter):
        u      = expit(ALPHA + bV * delta)
        delta  = delta - u * (delta - p) + VoB

    u   = expit(ALPHA + bV * delta)
    lam_val = lam * (1.0 - u)     # Lambda(delta*) = lam*(1-u)
    return lam_val * (delta - p)


# ── Diffusion / drift PDE terms ────────────────────────────────────────────────
def compute_diffusion_drift(v):
    """
    Returns RHS contribution from:
      a_P * d_nu v  (upwind, a_P > 0 always on grid)
      0.5 * xi^2 * nu * d2_nu v  (centered, Neumann BCs)
    Shape: (N_NU, N_VPI)
    """
    rhs = np.zeros_like(v)

    # ── Upwind first derivative (a_P > 0 → backward difference) ──────────────
    aP_vals = aP(nu_grid)[:, None]          # (N_NU, 1)

    dv_dnu           = np.zeros_like(v)
    dv_dnu[1:,  :]   = (v[1:, :] - v[:-1, :]) / dnu   # interior + upper bdry
    dv_dnu[0,   :]   = 0.0                              # Neumann: ghost = v[0]

    rhs += aP_vals * dv_dnu

    # ── Centered second derivative with Neumann ghost points ─────────────────
    v_ext          = np.empty((N_NU + 2, N_VPI))
    v_ext[1:-1, :] = v
    v_ext[0,    :] = v[0,  :]   # ghost below  (Neumann)
    v_ext[-1,   :] = v[-1, :]   # ghost above  (Neumann)

    d2v_dnu2  = (v_ext[2:, :] - 2.0 * v_ext[1:-1, :] + v_ext[:-2, :]) / dnu**2
    diff_coef = 0.5 * XI**2 * nu_grid[:, None]

    rhs += diff_coef * d2v_dnu2
    return rhs


# ── Gamma penalty ─────────────────────────────────────────────────────────────
def compute_penalties(v):
    """
    - gamma * xi^2 / 8 * V^pi^2    — inventory risk penalty (source term)

    Under Assumption 1, V^pi = sum_i q_i * V_i with V_i constant, so V^pi
    does not drift with nu; the variance risk premium produces no advection
    in the V^pi direction.  It only modifies the nu-direction drift, which
    is already handled by compute_diffusion_drift (uses aP, not aQ).
    Shape: (N_NU, N_VPI)
    """
    penalty = -(GAMMA * XI**2 / 8.0) * VPI**2
    return penalty


# ── H terms summed over all 20 options × 2 sides ──────────────────────────────
def compute_H_terms(v, options):
    """
    options: list of dicts with keys price, vega, lam, z
    For each option and each side (ask=+1, bid=-1):
      shift  = psi * z * V_i
      p      = (v - interp(v, vpi - shift)) / z
      contribution = z * H(p) * indicator(|vpi - shift| <= V_bar)
    Shape: (N_NU, N_VPI)
    """
    rhs = np.zeros((N_NU, N_VPI))

    for opt in options:
        V_i = opt['vega']
        z   = opt['z']
        lam = opt['lam']
        zV  = z * V_i

        for psi in (+1.0, -1.0):
            shift      = psi * zV          # how much vpi changes after trade
            vpi_shifted = vpi_grid - shift  # where to look up v

            # indicator: |vpi - shift| <= V_bar  ↔  shifted vpi in domain
            in_domain = np.abs(vpi_grid - shift) <= V_BAR

            # interpolate v at shifted vpi for every nu row
            v_shifted = np.zeros((N_NU, N_VPI))
            for k in range(N_NU):
                v_shifted[k, :] = np.interp(
                    vpi_shifted,
                    vpi_grid,
                    v[k, :],
                    left=v[k, 0],
                    right=v[k, -1]
                )

            p   = (v - v_shifted) / z                 # (N_NU, N_VPI)
            H   = hamiltonian(p, lam, V_i)            # intensity parametrised by option vega V_i (paper §4.1)

            rhs += z * H * in_domain[None, :]

    return rhs


# ── Main HJB solver ────────────────────────────────────────────────────────────
def solve_hjb(options):
    """
    Solves the HJB PDE backward from v(T)=0 using explicit Euler.
    Returns v_all of shape (N_T+1, N_NU, N_VPI),
    where v_all[0] = v at t=0  (most valuable)
    and   v_all[N_T] = 0       (terminal condition).
    """
    v     = np.zeros((N_NU, N_VPI))      # terminal condition v(T) = 0
    v_all = np.zeros((N_T + 1, N_NU, N_VPI))
    v_all[N_T] = v

    # March backward: store step n at index N_T - n
    for step in range(N_T):
        rhs  = compute_diffusion_drift(v)
        rhs += compute_penalties(v)
        rhs += compute_H_terms(v, options)

        v = v + dt * rhs
        v_all[N_T - 1 - step] = v

    return v_all


# ── Figure 2 reproduction ──────────────────────────────────────────────────────
def plot_figure2(v, save_path='figure2.png'):
    """3-D surface of value function at t=0."""
    VPI_mesh, NU_mesh = np.meshgrid(vpi_grid, nu_grid)

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        VPI_mesh / 1e7, NU_mesh, v,
        cmap='gray_r', edgecolor='none', alpha=0.9
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, label='Value function')
    ax.set_xlabel('Portfolio vega  (×10⁷)')
    ax.set_ylabel('Instantaneous variance ν')
    ax.set_zlabel('v(0, ν, V^π)')
    ax.set_title('Figure 2: Value function at t = 0')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Figure 2 saved to {save_path}")
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # ── Build option list from heston_pricer output ───────────────────────────
    # Expected: heston_pricer.py exposes get_option_data() returning a list of
    # dicts with keys: strike, maturity, price, vega
    try:
        from heston_pricer import get_option_data
        raw_options = get_option_data()
    except ImportError:
        # Fallback: hardcoded approximate values from paper Table (§4.1)
        print("heston_pricer not found — using hardcoded option data from paper.")
        import itertools
        STRIKES   = [8, 9, 10, 11, 12]
        MATURITIES = [1.0, 1.5, 2.0, 3.0]
        # Approximate prices and vegas from paper Figures 4-8
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
        raw_options = [
            {'strike': K, 'maturity': Tm,
             'price': approx_prices[(K, Tm)],
             'vega':  approx_vegas[(K, Tm)]}
            for K, Tm in itertools.product(STRIKES, MATURITIES)
        ]

    # ── Augment with lambda_i and z_i ─────────────────────────────────────────
    S0 = 10.0
    options = []
    for opt in raw_options:
        K     = opt['strike']
        price = opt['price']
        vega  = opt['vega']
        lam   = 252 * 30 / (1 + 0.7 * abs(S0 - K))   # arrivals per year
        z     = 5e5 / price                            # paper §4.1: z^i = 5×10^5 / S^i_0
        options.append({
            'strike':   K,
            'maturity': opt['maturity'],
            'price':    price,
            'vega':     vega,
            'lam':      lam,
            'z':        z,
        })

    # ── Print option summary ───────────────────────────────────────────────────
    print(f"\n{'K':>5} {'T':>5} {'price':>8} {'vega':>8} {'lam':>8} {'z':>12}")
    print("-" * 55)
    for opt in options:
        print(f"{opt['strike']:>5} {opt['maturity']:>5} "
              f"{opt['price']:>8.3f} {opt['vega']:>8.3f} "
              f"{opt['lam']:>8.0f} {opt['z']:>12.0f}")

    # ── Solve ─────────────────────────────────────────────────────────────────
    print(f"\nSolving HJB on {N_NU}×{N_VPI} grid over {N_T} time steps …")
    import time
    t0    = time.time()
    v_all = solve_hjb(options)
    print(f"Done in {time.time()-t0:.1f}s")

    v0 = v_all[0]
    print(f"\nValue function at t=0:")
    print(f"  min = {v0.min():.1f}")
    print(f"  max = {v0.max():.1f}")
    print(f"  at (nu0, Vpi=0)  ≈ {np.interp(0.0225, nu_grid, v0[:, N_VPI//2]):.1f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_figure2(v0)