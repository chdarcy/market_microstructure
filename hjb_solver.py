import numpy as np
from scipy.special import expit

# Grid parameters
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
NU0    = 0.0225

nu_grid  = np.linspace(0.0144, 0.0324, N_NU)
vpi_grid = np.linspace(-V_BAR, V_BAR, N_VPI)
dt       = T / N_T
dnu      = nu_grid[1] - nu_grid[0]

NU  = nu_grid[:, None]   # (N_NU, 1)
VPI = vpi_grid[None, :]  # (1,  N_VPI)


# Drift helpers
def aP(nu): return KAPPA_P * (THETA_P - nu)
def aQ(nu): return KAPPA_Q * (THETA_Q - nu)

def hamiltonian(p, lam, V_i, n_iter=15):
    """
    H(p) = sup_δ Λ(δ)·(δ − p) for logistic Λ.
    Newton iteration on FOC: δ ← δ − u·(δ−p) + V/β.
    """
    bV     = BETA / V_i
    VoB    = V_i / BETA
    delta  = p + 2.0 * VoB

    for _ in range(n_iter):
        u      = expit(ALPHA + bV * delta)
        delta  = delta - u * (delta - p) + VoB

    u   = expit(ALPHA + bV * delta)
    lam_val = lam * (1.0 - u)
    return lam_val * (delta - p)


def compute_diffusion_drift(v):
    """
    ν-direction PDE terms: aP(ν)·∂_ν v  +  ½ξ²ν·∂²_νν v.
    Returns (N_NU, N_VPI).
    """
    rhs = np.zeros_like(v)

    # Upwind ∂_ν v (aP > 0 on grid → backward difference)
    aP_vals = aP(nu_grid)[:, None]

    dv_dnu           = np.zeros_like(v)
    dv_dnu[1:,  :]   = (v[1:, :] - v[:-1, :]) / dnu
    dv_dnu[0,   :]   = 0.0

    rhs += aP_vals * dv_dnu

    # Centered ∂²_νν v with Neumann ghosts
    v_ext          = np.empty((N_NU + 2, N_VPI))
    v_ext[1:-1, :] = v
    v_ext[0,    :] = v[0,  :]
    v_ext[-1,   :] = v[-1, :]

    d2v_dnu2  = (v_ext[2:, :] - 2.0 * v_ext[1:-1, :] + v_ext[:-2, :]) / dnu**2
    diff_coef = 0.5 * XI**2 * nu_grid[:, None]

    rhs += diff_coef * d2v_dnu2

    return rhs

def compute_penalties(v):
    """
    Running source terms (eq. 4):
      V^π·(aP − aQ)/(2√ν)   — variance risk premium drift
      −γξ²/8·(V^π)²          — inventory risk penalty
    Returns (N_NU, N_VPI).
    """
    aP_vals = KAPPA_P * (THETA_P - NU)
    aQ_vals = KAPPA_Q * (THETA_Q - NU)
    drift_premium = VPI * (aP_vals - aQ_vals) / (2.0 * np.sqrt(NU))

    risk_penalty = -(GAMMA * XI**2 / 8.0) * VPI**2


    return drift_premium + risk_penalty
def compute_H_terms(v, options):
    """Sum z·H(p) over all 20 options × 2 sides (ask/bid).  Shape (N_NU, N_VPI)."""
    rhs = np.zeros((N_NU, N_VPI))

    for opt in options:
        V_i = opt['vega']
        z   = opt['z']
        lam = opt['lam']
        zV  = z * V_i

        for psi in (+1.0, -1.0):
            shift      = psi * zV
            vpi_shifted = vpi_grid - shift

            in_domain = np.abs(vpi_grid - shift) <= V_BAR

            v_shifted = np.zeros((N_NU, N_VPI))
            for k in range(N_NU):
                v_shifted[k, :] = np.interp(
                    vpi_shifted,
                    vpi_grid,
                    v[k, :],
                    left=v[k, 0],
                    right=v[k, -1]
                )

            p   = (v - v_shifted) / z
            H   = hamiltonian(p, lam, V_i)

            rhs += z * H * in_domain[None, :]

    return rhs

def solve_hjb(options):
    """
    Backward explicit Euler for the HJB PDE.
    Returns v_all: (N_T+1, N_NU, N_VPI), v_all[0] = v(t=0).
    """
    v     = np.zeros((N_NU, N_VPI))
    v_all = np.zeros((N_T + 1, N_NU, N_VPI))
    v_all[N_T] = v

    for step in range(N_T):
        rhs  = compute_diffusion_drift(v)
        rhs += compute_penalties(v)
        rhs += compute_H_terms(v, options)

        v = v + dt * rhs
        v_all[N_T - 1 - step] = v

    return v_all
