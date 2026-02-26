"""
param_sweeps.py
===============
Parameter sensitivity analysis for the vega market-making model.

Sweeps:
  A) α (alpha)  — intercept of the logistic intensity
  B) β (beta)   — steepness of the logistic intensity
  C) Intensity function family — logistic (baseline), exponential
  D) Queue-Reactive L1 LOB model (Huang-Lehalle-Rosenbaum 2015)
     — CTMC simulator with state-dependent intensities at the best
       bid/ask.  Self-contained extension: synthetic data generation,
       non-parametric estimation, simulation from estimated model,
       and comprehensive validation plots.

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
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

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


# ═══════════════════════════════════════════════════════════════════════════════
# Queue-Reactive empirical fill-probability bridge
# ═══════════════════════════════════════════════════════════════════════════════
#
# Run the CTMC and measure empirical Λ(δ): for a market-maker posting a
# limit order at offset δ from mid on the ask (bid) side, what is the fill
# rate per unit time?
#
# Mapping δ → queue position:
#   In the L1 single-tick model the ask queue has Q_a lots. A MM posting
#   at offset δ from mid joins the queue at position
#       k(δ) = max(1, round(1 + QR_KAPPA * δ))
#   where QR_KAPPA converts price-offset to lots-from-front.
#   k = 1  → front of queue (δ ≈ 0)
#   k > Q_a → behind the entire queue (will not fill unless queue grows
#             and then depletes through the MM's position).
#
# Fill event: a M_a (market buy) arrives and the post-event ask queue
#   Q_a − 1 < k  (i.e. the queue has just been eaten through the MM's
#   position).  Equivalently the MM fills whenever M_a fires AND Q_a ≤ k
#   *before* the event (since M_a removes 1 lot, the MM at position k
#   fills if the remaining queue after removal is < k, i.e. Q_a ≤ k).
#
# We symmetrise over bid/ask.  The empirical Λ is measured on a dense
# grid of δ values and stored as a PCHIP interpolant for smooth
# evaluation inside the HJB.
# ═══════════════════════════════════════════════════════════════════════════════

# Scale factor: δ (price units) → queue position (lots from front)
# At δ=0 the MM is at the front of queue (position 1).
# At δ=0.01 (~1 tick) the MM is ~1 + QR_KAPPA * 0.01 lots deep.
# κ=500 ⇒ 1 tick → position 6;  3 cents → position 16.
QR_KAPPA = 500.0    # lots per price-unit of offset

# ── Module-level cache for the empirical Λ(δ) curve ──────────────────────────
_QR_LAMBDA_INTERP  = None   # callable: δ → Λ(δ) (power-law fit)
_QR_LAMBDA_MAX     = None   # maximum Λ (at δ=0), i.e. fit parameter a
_QR_DELTA_GRID     = None   # raw δ grid (price units)
_QR_FILL_RATES     = None   # raw Λ(δ) values (fills/sec)
_QR_FIT_PARAMS     = None   # (a, b, c) power-law fit parameters


def estimate_fill_probability_from_ctmc(T_seconds=14400.0, n_delta=80,
                                        delta_max_price=0.06, seed=42):
    """
    Run the CTMC and measure the empirical fill rate Λ(δ) for a MM limit
    order posted at offset δ from mid on the ask (bid) side.

    For each δ we track a virtual MM order sitting at queue position
    k(δ) = max(1, round(1 + QR_KAPPA·δ)) lots from the front of the queue.
    The MM's remaining priority decreases by 1 each time a market order
    (or a cancel ahead of the MM) hits that side.  When the remaining
    priority reaches 0, the MM is filled.  After a fill or a queue reset
    (price move), the MM re-enters at position k.

    Symmetrised over bid + ask.

    Parameters
    ----------
    T_seconds : float
        Total CTMC simulation time (default 4 hours for good statistics).
    n_delta : int
        Number of δ grid points to evaluate.
    delta_max_price : float
        Maximum δ in price units (default 0.06 = 6 ticks).
    seed : int
        RNG seed.

    Returns
    -------
    delta_grid : ndarray (n_delta,) — δ values in price units
    fill_rates : ndarray (n_delta,) — empirical Λ(δ) in fills/sec
    """
    global _QR_LAMBDA_INTERP, _QR_LAMBDA_MAX, _QR_DELTA_GRID, _QR_FILL_RATES
    global _QR_FIT_PARAMS

    rng = np.random.default_rng(seed)

    delta_grid = np.linspace(0.0, delta_max_price, n_delta)
    # Queue position for each δ
    k_grid = np.maximum(1, np.round(1 + QR_KAPPA * delta_grid)).astype(int)

    # For each δ, track remaining lots ahead of MM on each side
    remaining_ask = k_grid.copy().astype(float)
    remaining_bid = k_grid.copy().astype(float)

    fill_counts = np.zeros(n_delta)
    total_time = 0.0

    Qb, Qa = 15, 15
    mid = 100.0
    t = 0.0

    while t < T_seconds:
        rates = {e: _gt_intensity(e, Qb, Qa) for e in EVENT_NAMES}
        total_rate = sum(rates.values())

        dt = rng.exponential(1.0 / total_rate)
        t += dt
        if t >= T_seconds:
            break

        probs = np.array([rates[e] for e in EVENT_NAMES])
        probs /= probs.sum()
        event = EVENT_NAMES[rng.choice(len(EVENT_NAMES), p=probs)]

        total_time += dt

        # ── Track MM fills via queue-position tracking ───────────────
        if event == "M_a":
            # Market buy hits ask — consumes one lot from front of queue
            remaining_ask -= 1
            filled = (remaining_ask <= 0)
            fill_counts += filled
            remaining_ask[filled] = k_grid[filled]

        if event == "M_b":
            # Market sell hits bid
            remaining_bid -= 1
            filled = (remaining_bid <= 0)
            fill_counts += filled
            remaining_bid[filled] = k_grid[filled]

        # Cancels ahead of MM also reduce remaining priority
        if event == "C_a" and Qa > 1:
            # Prob that the cancelled lot is ahead of MM ≈ (remaining-1)/(Qa-1)
            # remaining_ask includes MM's own lot, so lots ahead = remaining-1
            # pool of cancellable lots excludes the MM = Qa-1
            p_ahead = np.clip((remaining_ask - 1) / max(Qa - 1, 1), 0, 1)
            cancel_ahead = (rng.random(n_delta) < p_ahead)
            remaining_ask[cancel_ahead] -= 1
            filled = (remaining_ask <= 0)
            fill_counts += filled
            remaining_ask[filled] = k_grid[filled]

        if event == "C_b" and Qb > 1:
            p_ahead = np.clip((remaining_bid - 1) / max(Qb - 1, 1), 0, 1)
            cancel_ahead = (rng.random(n_delta) < p_ahead)
            remaining_bid[cancel_ahead] -= 1
            filled = (remaining_bid <= 0)
            fill_counts += filled
            remaining_bid[filled] = k_grid[filled]

        # ── Apply event to book ──────────────────────────────────────
        dQb, dQa = EVENT_EFFECTS[event]
        Qb += dQb * QR_LOT_SIZE
        Qa += dQa * QR_LOT_SIZE

        # Queue depletion → price move + reset (MM re-enters at position k)
        if Qb <= 0:
            mid -= 0.01
            Qb = max(1, int(rng.geometric(1.0 / QR_Q_RESET_MU)))
            Qa = max(1, int(rng.geometric(1.0 / QR_Q_RESET_MU)))
            remaining_bid[:] = k_grid
            remaining_ask[:] = k_grid
        if Qa <= 0:
            mid += 0.01
            Qb = max(1, int(rng.geometric(1.0 / QR_Q_RESET_MU)))
            Qa = max(1, int(rng.geometric(1.0 / QR_Q_RESET_MU)))
            remaining_ask[:] = k_grid
            remaining_bid[:] = k_grid

        Qb = min(Qb, QR_Q_MAX)
        Qa = min(Qa, QR_Q_MAX)

    # Per-side fill rate (averaged over bid + ask)
    fill_rates = fill_counts / (2.0 * total_time)

    # Ensure monotonically decreasing (physical requirement)
    for i in range(1, len(fill_rates)):
        fill_rates[i] = min(fill_rates[i], fill_rates[i - 1])

    # Ensure strictly positive floor for numerical safety
    fill_rates = np.maximum(fill_rates, 1e-6)

    # ── Fit a smooth parametric curve to the fill rates ──────────────
    # The raw data has staircase artefacts (consecutive equal values)
    # because multiple δ values map to the same integer queue position.
    # A PCHIP interpolant on the raw data preserves these flat segments,
    # which creates multiple local optima in the Hamiltonian's profit
    # landscape Λ(δ)·(δ−p) and produces staircase-like optimal spreads.
    #
    # Instead we fit Λ(δ) = a / (1 + b·δ)^c  — a power-law decay that
    # is guaranteed monotonically decreasing and produces a unimodal
    # profit function.  This matches the heavy-tail behaviour expected
    # from queue-reactive LOB models.
    _QR_DELTA_GRID = delta_grid.copy()
    _QR_FILL_RATES = fill_rates.copy()

    def _power_law(delta, a, b, c):
        return a / (1.0 + b * delta)**c

    try:
        p0 = [fill_rates[0], 300.0, 1.5]
        popt, _ = curve_fit(_power_law, delta_grid, fill_rates, p0=p0,
                            maxfev=10000,
                            bounds=([0, 0, 0.1], [np.inf, np.inf, 10.0]))
        _QR_FIT_PARAMS = popt   # (a, b, c)
    except RuntimeError:
        # Fallback: use a simpler exponential fit
        _QR_FIT_PARAMS = np.array([fill_rates[0], 200.0, 1.0])

    _QR_LAMBDA_MAX = _QR_FIT_PARAMS[0]   # Λ(0) = a

    # Store the fit for evaluation (no PCHIP interpolant needed)
    # _QR_LAMBDA_INTERP is kept as a callable for backward compatibility
    # For δ < 0 (MM crosses spread), cap at Λ(0) = a.
    a_fit, b_fit, c_fit = _QR_FIT_PARAMS

    def _power_law_eval(delta):
        delta = np.asarray(delta, dtype=float)
        # For δ < 0, Λ = Λ(0) = a (flat extension)
        delta_clamped = np.maximum(delta, 0.0)
        val = a_fit / (1.0 + b_fit * delta_clamped)**c_fit
        return np.maximum(val, 1e-8)

    _QR_LAMBDA_INTERP = _power_law_eval

    return delta_grid, fill_rates


def _qr_lambda(delta_price):
    """
    Evaluate the smooth queue-reactive fill rate at offset δ (price units).

    Uses the fitted power-law curve Λ(δ) = a / (1 + b·δ)^c, which is
    guaranteed monotonically decreasing and infinitely differentiable.
    Returns array matching input shape.
    """
    if _QR_LAMBDA_INTERP is None:
        raise RuntimeError("Call estimate_fill_probability_from_ctmc() first.")
    delta_price = np.asarray(delta_price)
    val = _QR_LAMBDA_INTERP(delta_price)
    return np.maximum(val, 1e-8)


# ---------- Queue-reactive: Hamiltonian (analytical, power-law) ---------------


def _hamiltonian_qr(p, lam, V_i, alpha, beta, n_iter=15):
    """
    Fully vectorised Hamiltonian for the queue-reactive model.

    Power-law fit  Λ(δ) = a / (1 + b·δ)^c.
    The FOC gives  δ* = (1 + c·b·p) / (b·(c − 1))  for c ≠ 1.

    During the HJB PDE solve, δ* is constrained to δ ≥ 0 (the fitted
    domain) to ensure numerical stability of the explicit scheme.
    The spread computation uses a separate, relaxed inversion that
    allows δ < 0 when the MM should improve the price.
    """
    p = np.asarray(p, dtype=float)

    if _QR_LAMBDA_MAX is not None and _QR_LAMBDA_MAX > 1e-10:
        scale = lam / _QR_LAMBDA_MAX
    else:
        scale = 1.0

    a, b, c = _QR_FIT_PARAMS

    # Closed-form optimal offset from FOC
    if abs(c - 1.0) < 1e-8:
        delta_foc = p + 1.0 / b
    else:
        delta_foc = (1.0 + c * b * p) / (b * (c - 1.0))

    # Interior FOC (δ ≥ 0 for PDE stability), no upper clamp
    delta_1 = np.maximum(delta_foc, 1e-10)
    lam_1 = a / (1.0 + b * delta_1)**c * scale
    H_1 = lam_1 * (delta_1 - p)

    # Boundary at δ = 0 (Λ = a)
    H_0 = a * scale * (0.0 - p)

    # sup is the better of interior and boundary
    H = np.maximum(H_1, H_0)

    return H


def _lambda_inv_qr(y, lam, V_i, alpha, beta):
    """
    Λ⁻¹(y): find δ such that Λ_QR(δ) = y.

    With the power-law fit  Λ(δ) = a / (1 + b·δ)^c  the inverse is:

        δ = ((a / y)^{1/c} − 1) / b

    Fully vectorised, no root-finding needed.  Allows negative δ
    (just like logistic / exponential) so that the MM can withdraw
    or improve quotes when the variance-risk premium pushes it.
    """
    y = np.asarray(y, dtype=float)
    scalar_input = (y.ndim == 0)
    y_flat = y.ravel()

    # Rescale y to the empirical curve's units
    if _QR_LAMBDA_MAX is not None and _QR_LAMBDA_MAX > 1e-10:
        scale = lam / _QR_LAMBDA_MAX
    else:
        scale = 1.0
    y_raw = y_flat / np.clip(scale, 1e-12, None)

    a, b, c = _QR_FIT_PARAMS

    # Closed-form inverse of the power-law (no lower clamp on ratio)
    ratio = a / np.clip(y_raw, 1e-30, None)       # can be < 1 when y > a
    result = (ratio ** (1.0 / c) - 1.0) / b       # negative when ratio < 1

    if scalar_input:
        return result.item()
    return result.reshape(y.shape)


# ── Registry of intensity families ────────────────────────────────────────────
INTENSITY_FAMILIES = {
    "logistic":      (_hamiltonian_logistic,    _lambda_inv_logistic),
    "exponential":   (_hamiltonian_exponential, _lambda_inv_exponential),
    "queue-reactive": (_hamiltonian_qr,         _lambda_inv_qr),
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

        # For queue-reactive, the standard H' → Λ⁻¹ pipeline saturates
        # at δ = 0 because H' = −λ exactly on the boundary.  Instead
        # we compute δ* directly from the closed-form FOC, which allows
        # δ < 0 (MM improves inside the spread when the VRP pushes it).
        orig_opt_spread = os_mod.optimal_spread
        _is_qr = (intensity_name == "queue-reactive")

        def _optimal_spread_patched(v0_local, opt, psi, nu_idx=None):
            """optimal_spread with QR-specific direct FOC for δ*."""
            if nu_idx is None:
                nu_idx = np.argmin(np.abs(os_mod.nu_grid - os_mod.NU0))
            V_i = opt['vega']
            z   = opt['z']
            lam_loc = opt['lam']
            v_row = v0_local[nu_idx, :]
            shift = psi * z * V_i
            vpi_shifted = os_mod.vpi_grid - shift
            v_shifted = np.interp(
                vpi_shifted, os_mod.vpi_grid, v_row,
                left=v_row[0], right=v_row[-1]
            )
            in_domain = np.abs(os_mod.vpi_grid - shift) <= os_mod.V_BAR
            p = (v_row - v_shifted) / z

            if _is_qr:
                # Direct FOC: δ* = (1 + c·b·p) / (b·(c−1))
                # This naturally goes negative for large negative p,
                # matching the logistic/exponential behaviour.
                a_pw, b_pw, c_pw = _QR_FIT_PARAMS
                if abs(c_pw - 1.0) < 1e-8:
                    delta = p + 1.0 / b_pw
                else:
                    delta = (1.0 + c_pw * b_pw * p) / (b_pw * (c_pw - 1.0))
            else:
                # Standard H' → Λ⁻¹ pipeline for other families
                Hprime = os_mod.hamiltonian_prime(p, lam_loc, V_i)
                arg = np.clip(-Hprime, 1e-12, lam_loc - 1e-12)
                delta = os_mod.lambda_inverse(arg, lam_loc, V_i)

            delta = np.maximum(delta, os_mod.DELTA_INF)
            delta = np.where(in_domain, delta, os_mod.DELTA_INF)
            return delta
        os_mod.optimal_spread = _optimal_spread_patched

        # Solve HJB
        v_all = solve_hjb(options)
        v0 = v_all[0]

        # For queue-reactive, the explicit Euler PDE accumulates ~8×
        # more grid-scale noise in v0 than logistic (the QR Hamiltonian
        # has broader support), and the closed-form FOC amplifies this
        # by c/(c−1) ≈ 4×, producing visibly choppy spread curves.
        # We apply a light Savitzky–Golay smooth to each ν-row of v0
        # before computing spreads.  This is pure post-processing on
        # the converged PDE solution — the HJB itself is unchanged.
        # Window=19 (47% of N_VPI=40), polyorder=3: removes noise while
        # preserving U-shape curvature (max bias < 0.5% of v0 range).
        if _is_qr:
            for i in range(v0.shape[0]):
                v0[i, :] = savgol_filter(v0[i, :], window_length=19,
                                         polyorder=3)

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
        os_mod.optimal_spread = orig_opt_spread


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

INTENSITY_NAMES  = ["logistic", "exponential", "queue-reactive"]
INTENSITY_COLORS = ["#2ca02c", "#d62728", "#1f77b4"]


def sweep_intensity(options, save_dir=None):
    """
    Sweep over intensity function families keeping (α, β) at baseline.

    • Logistic (baseline):  Λ = λ / (1 + exp(α + β/V·δ))
      – Saturates at both ends; S-shaped.  The paper's choice.

    • Exponential:  Λ = λ · exp(−α − β/V·δ)
      – Monotonically decreasing, no saturation at δ→−∞.
      – Thinner tail than logistic → fills drop faster for large δ.
      – Closed-form δ* = p + V/β.

    • Queue-reactive (CTMC):  Λ(δ) estimated empirically from a
      queue-reactive LOB simulation (Huang-Lehalle-Rosenbaum 2015).
      δ → queue position → fill rate via market-order flow.

    Returns list of (label, color, spreads).
    """
    save_dir = save_dir or SWEEP_DIR

    # ── Pre-compute empirical Λ(δ) from CTMC if not already cached ──
    if _QR_LAMBDA_INTERP is None:
        print("    [QR bridge] Running CTMC to estimate fill probability …",
              end="", flush=True)
        t0 = time.time()
        dg, fr = estimate_fill_probability_from_ctmc(
            T_seconds=14400.0, n_delta=80, delta_max_price=0.06, seed=42,
        )
        print(f"  {time.time()-t0:.1f}s  "
              f"(Λ(0)={fr[0]:.3f}/s, Λ(max)={fr[-1]:.6f}/s)")

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

    # ── Extra plot: Λ(δ) curves for all three intensity families ─────
    fname_lam = _plot_fill_probability_comparison(save_dir)
    if fname_lam:
        fnames.append(fname_lam)

    return results, fnames


def _plot_fill_probability_comparison(save_dir):
    """
    Plot the fill-probability curves Λ(δ) for logistic, exponential,
    and queue-reactive side by side.  Uses a representative option
    (λ and V_i from the K=10, T=1.0 option).
    """
    # Representative option parameters (K=10, T=1.0)
    lam_rep = 252 * 30 / (1 + 0.7 * abs(10.0 - 10.0))   # ≈ 7560
    V_rep   = 1.25   # approximate vega for ATM
    alpha, beta = ALPHA_BASE, BETA_BASE
    bV = beta / V_rep

    # δ in price units for plotting
    delta_price = np.linspace(0.0, 0.05, 300)
    # δ in HJB units (β/V · δ_price)
    delta_hjb = delta_price * bV

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left panel: Λ(δ) ─────────────────────────────────────────────
    ax = axes[0]

    # Logistic
    lam_logistic = lam_rep / (1.0 + np.exp(alpha + bV * delta_price))
    ax.plot(delta_price * 100, lam_logistic / lam_rep, lw=2,
            color="#2ca02c", label="Logistic (BBG 2020)")

    # Exponential
    lam_exp = lam_rep * np.exp(np.clip(-(alpha + bV * delta_price), -500, 500))
    ax.plot(delta_price * 100, lam_exp / lam_rep, lw=2,
            color="#d62728", label="Exponential (A-S)")

    # Queue-reactive (empirical)
    if _QR_FILL_RATES is not None and _QR_DELTA_GRID is not None:
        # Interpolate on the same grid
        lam_qr = _qr_lambda(delta_price)
        # Normalise so Λ(0) = lam_rep for fair comparison
        lam_qr_norm = lam_qr / _QR_LAMBDA_MAX
        ax.plot(delta_price * 100, lam_qr_norm, lw=2,
                color="#1f77b4", label="Queue-reactive (CTMC)")

    ax.set_xlabel("Offset δ  (cents from mid)")
    ax.set_ylabel("Λ(δ) / λ   (normalised fill probability)")
    ax.set_title("Fill Probability: Parametric vs Queue-Reactive")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)

    # ── Right panel: Λ(δ) on log scale ──────────────────────────────
    ax = axes[1]
    ax.semilogy(delta_price * 100, lam_logistic / lam_rep, lw=2,
                color="#2ca02c", label="Logistic")
    ax.semilogy(delta_price * 100,
                np.clip(lam_exp / lam_rep, 1e-15, None), lw=2,
                color="#d62728", label="Exponential")
    if _QR_FILL_RATES is not None:
        ax.semilogy(delta_price * 100,
                    np.clip(lam_qr_norm, 1e-15, None), lw=2,
                    color="#1f77b4", label="Queue-reactive")
    ax.set_xlabel("Offset δ  (cents from mid)")
    ax.set_ylabel("Λ(δ) / λ   (log scale)")
    ax.set_title("Fill Probability (Log Scale)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Intensity Functions: Λ(δ) Comparison",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = "sweep_intensity_lambda_comparison.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()
    print(f"    → {fname}")
    return fname


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


# NOTE: run_all_sweeps() is defined at the bottom of the file, after Section D.


# ═══════════════════════════════════════════════════════════════════════════════
# D)  Queue-Reactive Model — L1 LOB CTMC Simulator
# ═══════════════════════════════════════════════════════════════════════════════
#
# A completely different extension: instead of varying the *fill probability*
# Λ(δ) inside the HJB, we simulate the Level-1 order book as a continuous-time
# Markov chain (CTMC) following the queue-reactive framework of
# Huang, Lehalle & Rosenbaum (2015).
#
# State:  X_t = (Q_b, Q_a)  — best bid / ask queue sizes in lots.
#
# Six event types, each with an intensity that depends on the current state:
#   L_b : limit add at bid      Q_b → Q_b + v      intensity λ_Lb(Q_b, Q_a)
#   C_b : cancel at bid         Q_b → Q_b − v      intensity λ_Cb(Q_b, Q_a)
#   M_b : market sell (hits bid) Q_b → Q_b − v     intensity λ_Mb(Q_b, Q_a)
#   L_a : limit add at ask      Q_a → Q_a + v      intensity λ_La(Q_b, Q_a)
#   C_a : cancel at ask         Q_a → Q_a − v      intensity λ_Ca(Q_b, Q_a)
#   M_a : market buy (hits ask)  Q_a → Q_a − v     intensity λ_Ma(Q_b, Q_a)
#
# When a queue hits 0 → price moves one tick (mid shifts); queues are reset
# by sampling from an empirical distribution.
#
# The model is estimated on binned (Q_b, Q_a) state space:
#   λ_e(bin) = count_e(bin) / total_time_in_bin
#
# Deliverables:
#   1. Synthetic data generator (ground-truth Poisson-based L1 book)
#   2. Estimator: bin counts → intensity surfaces
#   3. CTMC simulator using competing exponentials
#   4. Validation plots:
#      a) Predicted vs empirical event rates by queue imbalance
#      b) Distribution of inter-event durations
#      c) Mid-price move frequency
#      d) Queue size distributions
# ═══════════════════════════════════════════════════════════════════════════════

QR_DIR = os.path.join(ROOT, "figures", "queue_reactive")
os.makedirs(QR_DIR, exist_ok=True)

# ── Event labels and their effect on (Q_b, Q_a) ──────────────────────────────
EVENT_NAMES = ["L_b", "C_b", "M_b", "L_a", "C_a", "M_a"]
EVENT_EFFECTS = {
    # event: (dQ_b, dQ_a)
    "L_b": (+1,  0),
    "C_b": (-1,  0),
    "M_b": (-1,  0),
    "L_a": ( 0, +1),
    "C_a": ( 0, -1),
    "M_a": ( 0, -1),
}

# ── Default ground-truth parameters ──────────────────────────────────────────
# Intensities are linear in (Q_b, Q_a) with a positive floor, loosely
# calibrated to realistic L1 dynamics (units: events per second).
#
#   λ_e(Q_b, Q_a) = max(base_e + coeff_Qb_e * Q_b + coeff_Qa_e * Q_a, floor)
#
# Economic intuition:
#   • Limit adds (L_b, L_a) increase when own queue is SMALL (mean reversion)
#     and when opposite queue is large (attracts liquidity provision).
#   • Cancellations (C_b, C_a) increase when own queue is LARGE (crowded).
#   • Market orders (M_b, M_a) increase when opposite queue is SMALL
#     (thin book → predatory trading) and own queue is large (imbalance).
#
GT_PARAMS = {
    #           base   coeff_Qb  coeff_Qa
    "L_b": (    3.0,    -0.10,     0.05),    # bid limits: attracted by thin bid, thick ask
    "C_b": (    0.5,     0.25,    -0.02),    # bid cancels: more when bid crowded
    "M_b": (    3.0,     0.08,    -0.10),    # market sells: more when ask thin, bid thick
    "L_a": (    3.0,     0.05,    -0.10),    # ask limits: symmetric to L_b
    "C_a": (    0.5,    -0.02,     0.25),    # ask cancels: symmetric to C_b
    "M_a": (    3.0,    -0.10,     0.08),    # market buys: symmetric to M_b
}
GT_FLOOR = 0.1   # minimum intensity (events/sec)

# ── Simulation parameters ────────────────────────────────────────────────────
QR_LOT_SIZE   = 1        # v: each event changes queue by this many lots
QR_Q_MAX      = 30       # maximum queue size (capped)
QR_Q_RESET_MU = 8.0      # mean of queue reset distribution (geometric)
QR_N_BINS     = 12       # number of bins per queue axis for estimation


def _gt_intensity(event, Qb, Qa):
    """Ground-truth intensity for event e at state (Qb, Qa)."""
    base, c_b, c_a = GT_PARAMS[event]
    return max(base + c_b * Qb + c_a * Qa, GT_FLOOR)


# ──────────────────────────────────────────────────────────────────────────────
# D.1  Synthetic data generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_l1_data(T_seconds=3600.0, seed=42):
    """
    Generate synthetic L1 LOB event data using competing-exponential CTMC
    with known ground-truth intensities.

    Parameters
    ----------
    T_seconds : float
        Total simulation time in seconds (default: 1 hour).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    events : list of dicts
        Each dict: {'time': float, 'event': str, 'Qb': int, 'Qa': int,
                    'mid': float}
        State (Qb, Qa) is the state *before* the event.
    """
    rng = np.random.default_rng(seed)

    Qb, Qa = 15, 15          # initial queue sizes
    mid = 100.0               # initial mid-price (arbitrary units)
    t = 0.0

    events = []

    while t < T_seconds:
        # Compute intensities for all 6 events
        rates = {e: _gt_intensity(e, Qb, Qa) for e in EVENT_NAMES}
        total_rate = sum(rates.values())

        # Time to next event ~ Exp(total_rate)
        dt = rng.exponential(1.0 / total_rate)
        t += dt
        if t >= T_seconds:
            break

        # Which event fires? (categorical draw proportional to rates)
        probs = np.array([rates[e] for e in EVENT_NAMES])
        probs /= probs.sum()
        event = EVENT_NAMES[rng.choice(len(EVENT_NAMES), p=probs)]

        # Record event with PRE-event state
        events.append({
            'time':  t,
            'event': event,
            'Qb':    Qb,
            'Qa':    Qa,
            'mid':   mid,
        })

        # Apply event
        dQb, dQa = EVENT_EFFECTS[event]
        Qb += dQb * QR_LOT_SIZE
        Qa += dQa * QR_LOT_SIZE

        # Queue depletion → price move + reset
        if Qb <= 0:
            mid -= 0.01        # mid shifts down one tick
            Qb = max(1, int(rng.geometric(1.0 / QR_Q_RESET_MU)))
            Qa = max(1, int(rng.geometric(1.0 / QR_Q_RESET_MU)))
        if Qa <= 0:
            mid += 0.01        # mid shifts up one tick
            Qb = max(1, int(rng.geometric(1.0 / QR_Q_RESET_MU)))
            Qa = max(1, int(rng.geometric(1.0 / QR_Q_RESET_MU)))

        # Cap queues
        Qb = min(Qb, QR_Q_MAX)
        Qa = min(Qa, QR_Q_MAX)

    return events


# ──────────────────────────────────────────────────────────────────────────────
# D.2  Estimator: bin events → intensity surfaces
# ──────────────────────────────────────────────────────────────────────────────

def estimate_intensities(events, n_bins=QR_N_BINS):
    """
    Non-parametric estimation of λ_e(Q_b, Q_a) on a binned grid.

    Method:
        λ_e(bin) = count_e(bin) / time_spent_in_bin

    Parameters
    ----------
    events : list of dicts from generate_synthetic_l1_data
    n_bins : int
        Number of bins per queue axis.

    Returns
    -------
    bin_edges_b, bin_edges_a : arrays of bin edges
    intensities : dict  event_name → 2D array (n_bins, n_bins) of estimated λ
    time_in_bin : 2D array (n_bins, n_bins) — total time spent in each bin
    count_in_bin : dict  event_name → 2D array of event counts per bin
    """
    # Determine bin edges from data range
    Qb_all = np.array([e['Qb'] for e in events])
    Qa_all = np.array([e['Qa'] for e in events])

    qb_max = max(Qb_all.max(), 1)
    qa_max = max(Qa_all.max(), 1)

    bin_edges_b = np.linspace(0, qb_max + 1, n_bins + 1)
    bin_edges_a = np.linspace(0, qa_max + 1, n_bins + 1)

    # Digitise each event's state into bins
    qb_bins = np.clip(np.digitize(Qb_all, bin_edges_b) - 1, 0, n_bins - 1)
    qa_bins = np.clip(np.digitize(Qa_all, bin_edges_a) - 1, 0, n_bins - 1)

    # Accumulate time spent in each bin
    times = np.array([e['time'] for e in events])
    # Duration in each state = time until next event
    durations = np.diff(times, prepend=0.0)

    time_in_bin = np.zeros((n_bins, n_bins))
    for i in range(len(events)):
        time_in_bin[qb_bins[i], qa_bins[i]] += durations[i]

    # Count events per bin per event type
    event_labels = np.array([e['event'] for e in events])
    count_in_bin = {}
    intensities = {}

    for ename in EVENT_NAMES:
        mask = (event_labels == ename)
        counts = np.zeros((n_bins, n_bins))
        for i in np.where(mask)[0]:
            counts[qb_bins[i], qa_bins[i]] += 1
        count_in_bin[ename] = counts

        # Intensity = count / time  (with floor to avoid division by zero)
        # Bins with very little dwell time produce unreliable outlier
        # estimates — use a minimum-time threshold and fall back to the
        # global average rate for that event type in sparse bins.
        total_count = counts.sum()
        total_time_all = time_in_bin.sum()
        global_rate = total_count / max(total_time_all, 1e-6)

        min_dwell = max(total_time_all / (n_bins * n_bins) * 0.05, 1.0)
        well_sampled = time_in_bin > min_dwell

        with np.errstate(divide='ignore', invalid='ignore'):
            lam = np.where(well_sampled, counts / time_in_bin, global_rate)
        intensities[ename] = lam

    # Bin centres for plotting
    bin_centres_b = 0.5 * (bin_edges_b[:-1] + bin_edges_b[1:])
    bin_centres_a = 0.5 * (bin_edges_a[:-1] + bin_edges_a[1:])

    return (bin_edges_b, bin_edges_a, bin_centres_b, bin_centres_a,
            intensities, time_in_bin, count_in_bin)


# ──────────────────────────────────────────────────────────────────────────────
# D.3  CTMC Simulator using estimated intensities
# ──────────────────────────────────────────────────────────────────────────────

def simulate_from_estimated(intensities, bin_edges_b, bin_edges_a,
                            T_seconds=3600.0, seed=123):
    """
    Simulate an L1 book using estimated (binned) intensities.

    At each state (Qb, Qa), look up which bin it falls in and use the
    estimated intensity for each event type.

    Returns list of event dicts (same format as generate_synthetic_l1_data).
    """
    rng = np.random.default_rng(seed)
    n_bins = len(bin_edges_b) - 1

    def _lookup(Qb, Qa, ename):
        """Look up estimated intensity for event at state."""
        ib = min(np.searchsorted(bin_edges_b, Qb, side='right') - 1, n_bins - 1)
        ia = min(np.searchsorted(bin_edges_a, Qa, side='right') - 1, n_bins - 1)
        ib = max(ib, 0)
        ia = max(ia, 0)
        return max(intensities[ename][ib, ia], GT_FLOOR)

    Qb, Qa = 15, 15
    mid = 100.0
    t = 0.0
    events = []

    while t < T_seconds:
        rates = {e: _lookup(Qb, Qa, e) for e in EVENT_NAMES}
        total_rate = sum(rates.values())
        if total_rate < 1e-12:
            break

        dt = rng.exponential(1.0 / total_rate)
        t += dt
        if t >= T_seconds:
            break

        probs = np.array([rates[e] for e in EVENT_NAMES])
        probs /= probs.sum()
        event = EVENT_NAMES[rng.choice(len(EVENT_NAMES), p=probs)]

        events.append({
            'time': t, 'event': event,
            'Qb': Qb, 'Qa': Qa, 'mid': mid,
        })

        dQb, dQa = EVENT_EFFECTS[event]
        Qb += dQb * QR_LOT_SIZE
        Qa += dQa * QR_LOT_SIZE

        if Qb <= 0:
            mid -= 0.01
            Qb = max(1, int(rng.geometric(1.0 / QR_Q_RESET_MU)))
            Qa = max(1, int(rng.geometric(1.0 / QR_Q_RESET_MU)))
        if Qa <= 0:
            mid += 0.01
            Qb = max(1, int(rng.geometric(1.0 / QR_Q_RESET_MU)))
            Qa = max(1, int(rng.geometric(1.0 / QR_Q_RESET_MU)))

        Qb = min(Qb, QR_Q_MAX)
        Qa = min(Qa, QR_Q_MAX)

    return events


# ──────────────────────────────────────────────────────────────────────────────
# D.4  Validation plots
# ──────────────────────────────────────────────────────────────────────────────

def _compute_imbalance_rates(events, n_imb_bins=12):
    """
    Compute event rates as a function of queue imbalance I = Qb/(Qb+Qa).

    Returns (imbalance_centres, rate_by_event) where rate_by_event is a
    dict  event_name → array of rates per imbalance bin.
    """
    Qb = np.array([e['Qb'] for e in events], dtype=float)
    Qa = np.array([e['Qa'] for e in events], dtype=float)
    total_q = Qb + Qa
    imbalance = np.where(total_q > 0, Qb / total_q, 0.5)

    times = np.array([e['time'] for e in events])
    durations = np.diff(times, prepend=0.0)

    imb_edges = np.linspace(0, 1, n_imb_bins + 1)
    imb_bins = np.clip(np.digitize(imbalance, imb_edges) - 1, 0, n_imb_bins - 1)

    time_in_imb = np.zeros(n_imb_bins)
    for i in range(len(events)):
        time_in_imb[imb_bins[i]] += durations[i]

    event_labels = np.array([e['event'] for e in events])
    rate_by_event = {}
    for ename in EVENT_NAMES:
        counts = np.zeros(n_imb_bins)
        mask = (event_labels == ename)
        for i in np.where(mask)[0]:
            counts[imb_bins[i]] += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            rate_by_event[ename] = np.where(time_in_imb > 0.1,
                                            counts / time_in_imb, 0.0)

    imb_centres = 0.5 * (imb_edges[:-1] + imb_edges[1:])
    return imb_centres, rate_by_event


def plot_qr_validation(events_gt, events_est, save_dir=None):
    """
    Generate all validation plots for the queue-reactive model.

    Parameters
    ----------
    events_gt  : ground-truth synthetic events
    events_est : events simulated from estimated intensities

    Returns list of saved filenames.
    """
    save_dir = save_dir or QR_DIR
    fnames = []

    # Colours for event types
    evt_colors = {
        "L_b": "#2ca02c", "C_b": "#d62728", "M_b": "#ff7f0e",
        "L_a": "#1f77b4", "C_a": "#9467bd", "M_a": "#8c564b",
    }

    # ── Plot 1: Event rates vs queue imbalance (predicted vs empirical) ──
    imb_gt, rates_gt = _compute_imbalance_rates(events_gt)
    imb_est, rates_est = _compute_imbalance_rates(events_est)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
    for idx, ename in enumerate(EVENT_NAMES):
        ax = axes[idx // 3, idx % 3]
        ax.plot(imb_gt, rates_gt[ename], 'o-', color=evt_colors[ename],
                lw=1.5, ms=5, label="Ground truth")
        ax.plot(imb_est, rates_est[ename], 's--', color=evt_colors[ename],
                lw=1.5, ms=5, alpha=0.7, label="Estimated model")
        ax.set_xlabel("Queue imbalance  Q_b / (Q_b + Q_a)")
        ax.set_ylabel("Rate  (events/sec)")
        ax.set_title(ename, fontsize=11, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Queue-Reactive Model: Event Rates vs Queue Imbalance",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = "qr_rates_vs_imbalance.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()
    fnames.append(fname)
    print(f"    → {fname}")

    # ── Plot 2: Inter-event duration distributions ───────────────────────
    times_gt = np.array([e['time'] for e in events_gt])
    times_est = np.array([e['time'] for e in events_est])
    dur_gt = np.diff(times_gt)
    dur_est = np.diff(times_est)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    bins_dur = np.linspace(0, np.percentile(dur_gt, 99), 60)
    ax.hist(dur_gt, bins=bins_dur, density=True, alpha=0.6,
            color="#2ca02c", label="Ground truth")
    ax.hist(dur_est, bins=bins_dur, density=True, alpha=0.6,
            color="#d62728", label="Estimated model")
    ax.set_xlabel("Inter-event duration (seconds)")
    ax.set_ylabel("Density")
    ax.set_title("Inter-event Duration Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    # Log-scale for tail comparison
    bins_log = np.logspace(np.log10(max(dur_gt.min(), 1e-4)),
                           np.log10(np.percentile(dur_gt, 99.5)), 50)
    ax.hist(dur_gt, bins=bins_log, density=True, alpha=0.6,
            color="#2ca02c", label="Ground truth")
    ax.hist(dur_est, bins=bins_log, density=True, alpha=0.6,
            color="#d62728", label="Estimated model")
    ax.set_xscale('log')
    ax.set_xlabel("Inter-event duration (seconds, log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Duration Distribution (Log Scale)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Queue-Reactive Model: Inter-Event Durations",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = "qr_duration_distribution.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()
    fnames.append(fname)
    print(f"    → {fname}")

    # ── Plot 3: Mid-price paths & move frequency ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mid-price sample paths
    ax = axes[0]
    t_gt = np.array([e['time'] for e in events_gt])
    mid_gt = np.array([e['mid'] for e in events_gt])
    t_est = np.array([e['time'] for e in events_est])
    mid_est = np.array([e['mid'] for e in events_est])
    ax.plot(t_gt, mid_gt, lw=0.5, alpha=0.8, color="#2ca02c",
            label="Ground truth")
    ax.plot(t_est, mid_est, lw=0.5, alpha=0.8, color="#d62728",
            label="Estimated model")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mid-price")
    ax.set_title("Mid-Price Sample Paths")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Price move frequency (moves per minute)
    ax = axes[1]

    def _moves_per_minute(events_list, window=60.0):
        if len(events_list) < 2:
            return np.array([]), np.array([])
        mids = np.array([e['mid'] for e in events_list])
        times_arr = np.array([e['time'] for e in events_list])
        moves = np.abs(np.diff(mids)) > 1e-8
        move_times = times_arr[1:][moves]
        T_max = times_arr[-1]
        edges = np.arange(0, T_max, window)
        counts, _ = np.histogram(move_times, bins=edges)
        centres = 0.5 * (edges[:-1] + edges[1:]) / 60.0  # minutes
        return centres, counts

    c_gt, m_gt = _moves_per_minute(events_gt)
    c_est, m_est = _moves_per_minute(events_est)
    if len(c_gt) > 0:
        ax.bar(c_gt, m_gt, width=0.8, alpha=0.5, color="#2ca02c",
               label=f"GT (mean={m_gt.mean():.1f}/min)")
    if len(c_est) > 0:
        ax.bar(c_est, m_est, width=0.8, alpha=0.5, color="#d62728",
               label=f"Est (mean={m_est.mean():.1f}/min)")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Mid-price moves per minute")
    ax.set_title("Mid-Price Move Frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Queue-Reactive Model: Mid-Price Dynamics",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = "qr_midprice_dynamics.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()
    fnames.append(fname)
    print(f"    → {fname}")

    # ── Plot 4: Queue size distributions ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    Qb_gt = np.array([e['Qb'] for e in events_gt])
    Qa_gt = np.array([e['Qa'] for e in events_gt])
    Qb_est = np.array([e['Qb'] for e in events_est])
    Qa_est = np.array([e['Qa'] for e in events_est])

    q_max = max(Qb_gt.max(), Qa_gt.max(), Qb_est.max(), Qa_est.max(), 1)
    q_bins = np.arange(0, q_max + 2) - 0.5

    ax = axes[0]
    ax.hist(Qb_gt, bins=q_bins, density=True, alpha=0.5, color="#2ca02c",
            label="GT  Q_b")
    ax.hist(Qb_est, bins=q_bins, density=True, alpha=0.5, color="#d62728",
            label="Est Q_b")
    ax.set_xlabel("Best Bid Queue Size")
    ax.set_ylabel("Density")
    ax.set_title("Bid Queue Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(Qa_gt, bins=q_bins, density=True, alpha=0.5, color="#1f77b4",
            label="GT  Q_a")
    ax.hist(Qa_est, bins=q_bins, density=True, alpha=0.5, color="#9467bd",
            label="Est Q_a")
    ax.set_xlabel("Best Ask Queue Size")
    ax.set_ylabel("Density")
    ax.set_title("Ask Queue Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Queue-Reactive Model: Queue Size Distributions",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = "qr_queue_distributions.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()
    fnames.append(fname)
    print(f"    → {fname}")

    # ── Plot 5: Estimated intensity heatmaps ─────────────────────────────
    # Re-estimate on ground truth to get the surfaces
    (bin_edges_b, bin_edges_a, bin_centres_b, bin_centres_a,
     est_int, time_in_bin, _) = estimate_intensities(events_gt)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for idx, ename in enumerate(EVENT_NAMES):
        ax = axes[idx // 3, idx % 3]
        lam_est = est_int[ename]

        # Ground truth surface on same grid
        Qb_mesh, Qa_mesh = np.meshgrid(bin_centres_b, bin_centres_a,
                                        indexing='ij')
        lam_true = np.vectorize(lambda qb, qa: _gt_intensity(ename, qb, qa))(
            Qb_mesh, Qa_mesh
        )

        # Plot estimated as heatmap — clip colorscale to 95th percentile
        # so that residual sparse-bin outliers don't blow out the range
        vmax = np.percentile(lam_est[lam_est > 0], 95) if np.any(lam_est > 0) else 1.0
        im = ax.imshow(lam_est.T, origin='lower', aspect='auto',
                       extent=[bin_edges_b[0], bin_edges_b[-1],
                               bin_edges_a[0], bin_edges_a[-1]],
                       cmap='YlOrRd', vmin=0, vmax=vmax)
        # Overlay ground truth as contours
        ax.contour(Qb_mesh, Qa_mesh, lam_true,
                   levels=5, colors='black', linewidths=0.8, linestyles='--')
        ax.set_xlabel("Q_b")
        ax.set_ylabel("Q_a")
        ax.set_title(f"{ename}  (heatmap=est, contour=true)", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Queue-Reactive: Estimated vs True Intensity Surfaces",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = "qr_intensity_heatmaps.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()
    fnames.append(fname)
    print(f"    → {fname}")

    return fnames


# ──────────────────────────────────────────────────────────────────────────────
# D.5  Summary statistics
# ──────────────────────────────────────────────────────────────────────────────

def print_qr_summary(events_gt, events_est):
    """Print summary comparison of ground-truth vs estimated model."""
    print("\n  Queue-Reactive Model — Summary Statistics")
    print("  " + "─" * 60)

    for label, evts in [("Ground truth", events_gt),
                        ("Estimated   ", events_est)]:
        n = len(evts)
        T = evts[-1]['time'] if n > 0 else 0
        mids = np.array([e['mid'] for e in evts])
        n_moves = np.sum(np.abs(np.diff(mids)) > 1e-8) if n > 1 else 0

        event_counts = {}
        for e in EVENT_NAMES:
            event_counts[e] = sum(1 for ev in evts if ev['event'] == e)

        Qb_arr = np.array([e['Qb'] for e in evts])
        Qa_arr = np.array([e['Qa'] for e in evts])

        print(f"\n  {label}:")
        print(f"    Events: {n:,}   Duration: {T:.0f}s   "
              f"Rate: {n/T:.1f}/s")
        print(f"    Mid-price moves: {n_moves}   "
              f"({n_moves / (T / 60):.1f}/min)")
        print(f"    Mean Q_b: {Qb_arr.mean():.1f}   "
              f"Mean Q_a: {Qa_arr.mean():.1f}")
        print(f"    Event breakdown:")
        for e in EVENT_NAMES:
            print(f"      {e}: {event_counts[e]:>6,}  "
                  f"({event_counts[e]/n*100:.1f}%)")


# ──────────────────────────────────────────────────────────────────────────────
# D.6  Entry point for queue-reactive sweep
# ──────────────────────────────────────────────────────────────────────────────

def run_queue_reactive(save_dir=None):
    """
    Full pipeline: generate data → estimate → simulate → validate.

    Returns list of figure filenames.
    """
    save_dir = save_dir or QR_DIR

    print("    Step 1: Generating synthetic L1 data (ground truth) …")
    t0 = time.time()
    events_gt = generate_synthetic_l1_data(T_seconds=7200.0, seed=42)
    print(f"            {len(events_gt):,} events in {time.time()-t0:.1f}s")

    print("    Step 2: Estimating intensities from binned data …")
    t0 = time.time()
    (bin_edges_b, bin_edges_a, bin_centres_b, bin_centres_a,
     est_int, time_in_bin, count_in_bin) = estimate_intensities(events_gt)
    print(f"            {QR_N_BINS}×{QR_N_BINS} bins, {time.time()-t0:.1f}s")

    print("    Step 3: Simulating from estimated model …")
    t0 = time.time()
    events_est = simulate_from_estimated(
        est_int, bin_edges_b, bin_edges_a,
        T_seconds=7200.0, seed=123,
    )
    print(f"            {len(events_est):,} events in {time.time()-t0:.1f}s")

    print("    Step 4: Generating validation plots …")
    fnames = plot_qr_validation(events_gt, events_est, save_dir=save_dir)

    print_qr_summary(events_gt, events_est)

    return fnames


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

    print("\n  D) Queue-reactive L1 LOB model")
    fnames_d = run_queue_reactive()
    all_fnames.extend(fnames_d)

    print(f"\n  Total sweep figures: {len(all_fnames)}")
    for f in sorted(all_fnames):
        print(f"    → {f}")

    return all_fnames


if __name__ == "__main__":
    print("=" * 72)
    print("  Parameter sweeps — α, β, intensity function, queue-reactive")
    print("=" * 72)
    run_all_sweeps()
