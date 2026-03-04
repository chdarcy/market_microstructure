"""
Reproduce Figure 3 (BBG 2020): optimal mid-to-bid spread for option 1
(K=8, T=1) at ν=ν₀ vs time, for several Vπ levels.  All curves should
converge to the same value near t=T, confirming the horizon is long enough.
"""

import numpy as np

from hjb_solver import nu_grid, vpi_grid, N_VPI, V_BAR
from optimal_spreads import hamiltonian_prime, lambda_inverse, DELTA_INF

def bid_spread_vs_time(v_all, opt, nu_idx):
    """
    delta_bid(t, Vpi) at every saved time step for one option (bid side).
    Returns spreads: ndarray (N_T+1, N_VPI).
    """
    V_i = opt['vega']
    z   = opt['z']
    lam = opt['lam']
    psi = -1.0               # bid
    nu  = nu_grid[nu_idx]

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
