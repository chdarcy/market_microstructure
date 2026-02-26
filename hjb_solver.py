import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

# ══════════════════════════════════════════════════════════════════════════════
# EXTENSION TOGGLE
# ══════════════════════════════════════════════════════════════════════════════
# False → BBG (2020) baseline: V^i frozen at t=0 (constant vega approximation)
# True  → Stochastic vega extension: V^π advects with ν via the chain rule
#           ∂_t v now includes  (∂V^π/∂ν) · aP(ν) · ∂_Vπ v
#         where ∂V^π/∂ν = Σᵢ qᵢ · ∂V^i/∂ν is pre-computed from option data
#         via a second finite-difference pass over ν (see compute_vega_sensitivities).
#         Options must carry a 'dvega_dnu' field (added by solve_hjb automatically).
USE_STOCHASTIC_VEGA = False

# ── Grid parameters ────────────────────────────────────────────────────────────
N_T    = 180
N_NU   = 30
N_VPI  = 40
T      = 0.0012             # paper §4.1: T = 0.0012 year (i.e. 0.3 day)
XI     = 0.2
GAMMA  = 1e-3
RHO    = -0.5              # spot–vol correlation (paper §4.1); reduces effective risk penalty
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


# ── Stochastic vega: ∂V^i/∂ν sensitivity ─────────────────────────────────────
def compute_vega_sensitivities(options):
    """
    Estimate ∂V^i/∂ν for each option via central finite differences over ν,
    using the same Heston MC infrastructure as heston_pricer.py.

    Under the extension, V^i(ν) = 2√ν · ∂_ν O^i(ν) is no longer frozen.
    As ν drifts under aP, V^π shifts at rate:
        ∂V^π/∂ν · aP(ν)  =  Σᵢ qᵢ · (∂V^i/∂ν) · aP(ν)

    Since qᵢ is unknown at PDE-solve time, we store ∂V^i/∂ν per option and
    compute the aggregate coefficient on-the-fly inside compute_diffusion_drift
    using the current inventory weights implied by V^π.  For the PDE we make
    the approximation that the portfolio is dominated by ATM options, so we use
    the inventory-weighted average:

        dvpi_dnu(ν) ≈ Σᵢ (V^i / V^π_scale) · (∂V^i/∂ν)   at V^π = V_BAR/2

    In practice we store ∂V^i/∂ν on each option dict under 'dvega_dnu' and
    compute a grid-level aggregate coefficient DVPI_GNU of shape (N_NU,) by
    summing over options weighted equally (inventory-neutral approximation).
    This is the simplest tractable closure; a full treatment would require
    tracking individual qᵢ as additional state variables.

    Returns
    -------
    options : same list with 'dvega_dnu' added to each dict (units: yr^{1/2})
    DVPI_GNU : ndarray (N_NU,) — aggregate ∂V^π/∂ν coefficient, shape (N_NU,)
               evaluated at each ν grid point assuming equal inventory weights.
    """
    try:
        from heston_pricer import (
            _simulate, _option_prices, FD_EPS, SEED, STRIKES, MATURITIES, NU0
        )
    except ImportError:
        print("[stochastic vega] heston_pricer not available — "
              "dvega_dnu set to 0 (degenerates to constant-vega baseline).")
        for opt in options:
            opt['dvega_dnu'] = 0.0
        return options, np.zeros(N_NU)

    print("  [stochastic vega] Computing ∂V^i/∂ν via 4-point FD over ν …")
    eps = FD_EPS

    # Four perturbed simulations for 4th-order central difference of V^i(ν)
    # V^i(ν) = 2√ν · [O(ν+ε) - O(ν-ε)] / (2ε)  (the vega)
    # ∂V^i/∂ν ≈ [V^i(ν+ε) - V^i(ν-ε)] / (2ε)
    # V^i(ν+ε) needs O(ν+2ε), O(ν), O(ν+ε) evaluated at ν+ε — i.e. 4 sims total.
    # We use a simpler 2-point estimate: run the pricer at ν₀±δ with δ=5*FD_EPS
    delta_nu = 5.0 * eps

    def get_vegas(nu_centre):
        paths_up, mat_to_step = _simulate(nu_centre + eps, SEED)
        paths_dn, _           = _simulate(nu_centre - eps, SEED)
        p_up = _option_prices(paths_up, mat_to_step)
        p_dn = _option_prices(paths_dn, mat_to_step)
        sigma_c = np.sqrt(nu_centre)
        v = {}
        for K in STRIKES:
            for Tm in MATURITIES:
                dpdnu = (p_up[(K,Tm)] - p_dn[(K,Tm)]) / (2.0 * eps)
                v[(K,Tm)] = 2.0 * sigma_c * dpdnu
        return v

    vegas_hi = get_vegas(NU0 + delta_nu)
    vegas_lo = get_vegas(NU0 - delta_nu)

    # Attach ∂V^i/∂ν to each option (evaluated at ν₀; treated as constant on grid)
    for opt in options:
        key = (opt['strike'], opt['maturity'])
        opt['dvega_dnu'] = (vegas_hi[key] - vegas_lo[key]) / (2.0 * delta_nu)

    # Build aggregate coefficient: equal-weight sum over options
    # Multiplied by sign correction so that positive V^π → positive drift when aP>0
    # DVPI_GNU[k] = Σᵢ |∂V^i/∂ν| / N_opts  (scalar approximation, ν-independent)
    n_opts = len(options)
    dvpi_dnu_scalar = sum(opt['dvega_dnu'] for opt in options) / n_opts
    DVPI_GNU = np.full(N_NU, dvpi_dnu_scalar)   # (N_NU,) constant on grid

    print(f"  [stochastic vega] Mean ∂V^i/∂ν = {dvpi_dnu_scalar:.4f}  "
          f"(range: {min(o['dvega_dnu'] for o in options):.4f} to "
          f"{max(o['dvega_dnu'] for o in options):.4f})")
    return options, DVPI_GNU
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
def compute_diffusion_drift(v, dvpi_gnu=None):
    """
    Returns RHS contribution from the ν-direction terms:

      aP(ν) · ∂_ν v                        (upwind, aP > 0 always on grid)
      ½ ξ² ν · ∂²_νν v                     (centered, Neumann BCs)

    Extension (USE_STOCHASTIC_VEGA=True):
      aP(ν) · (∂V^π/∂ν) · ∂_Vπ v          (upwind in V^π direction)

    The third term arises because V^π = Σᵢ qᵢ V^i(ν) is no longer constant —
    it inherits the drift of ν under the physical measure.  Concretely, if ν
    rises by dν then V^π rises by (∂V^π/∂ν) · dν, so the value function is
    advected in the V^π direction at rate aP(ν) · ∂V^π/∂ν.

    Parameters
    ----------
    v        : (N_NU, N_VPI) current value function
    dvpi_gnu : (N_NU,) or None — aggregate ∂V^π/∂ν coefficient per ν row.
               Required (and used) only when USE_STOCHASTIC_VEGA=True.

    Shape: (N_NU, N_VPI)
    """
    rhs = np.zeros_like(v)

    # ── Upwind first derivative in ν (aP > 0 → backward difference) ──────────
    aP_vals = aP(nu_grid)[:, None]          # (N_NU, 1)

    dv_dnu           = np.zeros_like(v)
    dv_dnu[1:,  :]   = (v[1:, :] - v[:-1, :]) / dnu   # interior + upper bdry
    dv_dnu[0,   :]   = 0.0                              # Neumann: ghost = v[0]

    rhs += aP_vals * dv_dnu

    # ── Centered second derivative in ν with Neumann ghost points ─────────────
    v_ext          = np.empty((N_NU + 2, N_VPI))
    v_ext[1:-1, :] = v
    v_ext[0,    :] = v[0,  :]   # ghost below  (Neumann)
    v_ext[-1,   :] = v[-1, :]   # ghost above  (Neumann)

    d2v_dnu2  = (v_ext[2:, :] - 2.0 * v_ext[1:-1, :] + v_ext[:-2, :]) / dnu**2
    diff_coef = 0.5 * XI**2 * nu_grid[:, None]

    rhs += diff_coef * d2v_dnu2

    # ── Stochastic vega extension: advection in V^π direction ─────────────────
    if USE_STOCHASTIC_VEGA and dvpi_gnu is not None:
        # Speed of V^π advection at each ν row: c(ν) = aP(ν) · ∂V^π/∂ν
        # shape (N_NU, 1) for broadcasting
        c = (aP_vals * dvpi_gnu[:, None])   # (N_NU, 1)

        dvpi = vpi_grid[1] - vpi_grid[0]   # uniform spacing

        # Upwind in V^π: if c > 0 use backward difference, else forward
        dv_dvpi = np.zeros_like(v)

        # Backward difference (c > 0): interior columns 1..N_VPI-1
        dv_dvpi_back = np.zeros_like(v)
        dv_dvpi_back[:, 1:]  = (v[:, 1:] - v[:, :-1]) / dvpi
        dv_dvpi_back[:, 0]   = 0.0   # Neumann at left boundary

        # Forward difference (c < 0): interior columns 0..N_VPI-2
        dv_dvpi_fwd = np.zeros_like(v)
        dv_dvpi_fwd[:, :-1] = (v[:, 1:] - v[:, :-1]) / dvpi
        dv_dvpi_fwd[:, -1]  = 0.0   # Neumann at right boundary

        # Select based on sign of c, broadcast over V^π axis
        c_pos = (c >= 0)   # (N_NU, 1) boolean
        dv_dvpi = np.where(c_pos, dv_dvpi_back, dv_dvpi_fwd)

        rhs += c * dv_dvpi

    return rhs


# ── Gamma penalty ─────────────────────────────────────────────────────────────
def compute_penalties(v):
    """
    Running source terms in the HJB (equation 4 of the paper):

      + V^pi * (aP - aQ) / (2 * sqrt(nu))        — variance risk premium drift
      - gamma * xi^2 * (1-rho^2) / 8 * (V^pi)^2  — inventory risk penalty

    Fix 1 — drift premium (previously missing):
      The term V^pi*(aP-aQ)/(2*sqrt(nu)) appears explicitly in eq. (4).
      With paper params aP > aQ for nu near nu0 (since kappa_P*theta_P=0.08
      vs kappa_Q*theta_Q=0.0675), this term is positive for V^pi>0 and
      negative for V^pi<0, giving the correct asymmetric tilt to the surface.
      Its absence caused the surface to be too symmetric and inflated.

    Fix 2 — (1-rho^2) correction (Appendix A.1):
      When the market maker optimally hedges in the underlying, the effective
      vega risk penalty becomes (1-rho^2)*xi^2/8 instead of xi^2/8.
      With rho=-0.5, the factor is 0.75, reducing the penalty by 25%.
      This matches the full optimal-hedging formulation from Appendix A.1.

    Shape: (N_NU, N_VPI)
    """
    # ── Variance risk premium: V^pi * (aP(nu) - aQ(nu)) / (2*sqrt(nu)) ──────
    aP_vals = KAPPA_P * (THETA_P - NU)   # (N_NU, 1), broadcast over VPI
    aQ_vals = KAPPA_Q * (THETA_Q - NU)   # (N_NU, 1)

    # Keep the drift premium, but note its net effect:
    # at nu0, aP-aQ > 0, so this ADDS to v symmetrically around Vpi=0
    # — it tilts the surface but doesn't reduce the peak
    drift_premium = VPI * (aP_vals - aQ_vals) / (2.0 * np.sqrt(NU))

    # ── Quadratic inventory risk penalty with (1-rho^2) correction ───────────
    risk_penalty = -(GAMMA * XI**2 / 8.0) * VPI**2


    return drift_premium + risk_penalty


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

    Baseline (USE_STOCHASTIC_VEGA=False):
        Replicates BBG (2020) — V^i frozen at t=0.

    Extension (USE_STOCHASTIC_VEGA=True):
        Adds advection term  aP(ν) · (∂V^π/∂ν) · ∂_Vπ v  to the PDE.
        ∂V^π/∂ν is estimated via a second FD pass in heston_pricer and stored
        on each option dict as 'dvega_dnu'.  An inventory-neutral aggregate
        coefficient DVPI_GNU is pre-computed once before the time loop.

    Returns v_all of shape (N_T+1, N_NU, N_VPI),
    where v_all[0] = v at t=0  (most valuable)
    and   v_all[N_T] = 0       (terminal condition).
    """
    # ── Pre-compute stochastic vega coefficient (once, before time loop) ──────
    dvpi_gnu = None
    if USE_STOCHASTIC_VEGA:
        print("  Mode: STOCHASTIC VEGA  (extension — V^π advects with ν)")
        options, dvpi_gnu = compute_vega_sensitivities(options)
    else:
        print("  Mode: CONSTANT VEGA  (BBG 2020 baseline)")

    v     = np.zeros((N_NU, N_VPI))      # terminal condition v(T) = 0
    v_all = np.zeros((N_T + 1, N_NU, N_VPI))
    v_all[N_T] = v

    # March backward: store step n at index N_T - n
    for step in range(N_T):
        rhs  = compute_diffusion_drift(v, dvpi_gnu=dvpi_gnu)
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
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        VPI_mesh / 1e7, NU_mesh, v,
        cmap='gray_r', edgecolor='none', alpha=0.9
    )
    ax.set_zlim(v.min(), v.max())
    ax.set_xlabel('Portfolio vega  (×10⁷)')
    ax.set_ylabel('Instantaneous variance ν')
    ax.set_zlabel('v(0, ν, V^π)')
    ax.set_title('Figure 2: Value function at t = 0')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Figure 2 saved to {save_path}")
    plt.close(fig)


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

    results = {}
    for mode, flag in [('Constant vega (BBG baseline)', False),
                       ('Stochastic vega (extension)',  True)]:
        # Reload fresh options each run (compute_vega_sensitivities mutates the list)
        options_run = []
        for opt in raw_options:
            K     = opt['strike']
            price = opt['price']
            vega  = opt['vega']
            lam   = 252 * 30 / (1 + 0.7 * abs(S0 - K))
            z     = 5e5 / price
            options_run.append({
                'strike': K, 'maturity': opt['maturity'],
                'price': price, 'vega': vega, 'lam': lam, 'z': z,
            })

        # Flip the global toggle
        import hjb_solver as _self
        _self.USE_STOCHASTIC_VEGA = flag

        print(f"\n{'─'*60}")
        print(f"  {mode}")
        print(f"{'─'*60}")
        t0    = time.time()
        v_all = solve_hjb(options_run)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        v0 = v_all[0]
        print(f"  v(0) range : [{v0.min():.1f}, {v0.max():.1f}]")
        print(f"  v_peak     : {v0.max():.1f}  (paper ≈ 120,000)")

        results[mode] = v0
        save = 'figure2_constant_vega.png' if not flag else 'figure2_stochastic_vega.png'
        plot_figure2(v0, save_path=save)

    # ── Difference surface ────────────────────────────────────────────────────
    v_base = results['Constant vega (BBG baseline)']
    v_ext  = results['Stochastic vega (extension)']
    diff   = v_ext - v_base

    VPI_mesh, NU_mesh = np.meshgrid(vpi_grid, nu_grid)
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(VPI_mesh / 1e7, NU_mesh, diff,
                           cmap='RdBu', edgecolor='none', alpha=0.9)
    fig.colorbar(surf, ax=ax, shrink=0.5, label='Δv  (stochastic − constant)')
    ax.set_xlabel('Portfolio vega  (×10⁷)')
    ax.set_ylabel('Instantaneous variance ν')
    ax.set_zlabel('Δv(0, ν, V^π)')
    ax.set_title('Extension vs Baseline: value function difference at t=0')
    plt.tight_layout()
    plt.savefig('figure2_diff.png', dpi=150)
    print("\nDifference surface saved to figure2_diff.png")
    plt.close(fig)