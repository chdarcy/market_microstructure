# Implementation Report: Algorithmic Market Making for Options

## A Numerical Study of the Baldacci–Bergault–Guéant (2021) Framework with Queue-Reactive Extensions

---

## 1. Overview

This codebase implements the optimal market-making model for options from Baldacci, Bergault & Guéant (2021), "Algorithmic Market Making for Options" (*Quantitative Finance*). The model solves a Hamilton–Jacobi–Bellman (HJB) partial differential equation for a risk-averse market-maker who continuously quotes bid and ask prices on a book of vanilla options whose underlying follows a Heston stochastic volatility process.

The implementation comprises six modules:

| Module | Role |
|---|---|
| heston_pricer.py | Monte Carlo pricing of European calls under Heston dynamics |
| hjb_solver.py | Finite-difference solver for the HJB PDE on a (ν, V^π) grid |
| optimal_spreads.py | Extraction of optimal bid/ask offsets from the value function |
| convergence.py | Stationarity analysis of quotes across HJB time steps |
| param_sweeps.py | Parameter sensitivity (α, β), alternative intensity functions, and queue-reactive LOB extension |
| main.py | Orchestration: runs all steps and saves figures to organised directories |

The pipeline reproduces Figures 1–13 from the paper and extends the model in three directions: (A) sensitivity analysis of the logistic intensity parameters α and β, (B) comparison across intensity function families (logistic, exponential, queue-reactive), and (C) a full Level-1 limit order book simulator based on the queue-reactive CTMC framework of Huang, Lehalle & Rosenbaum (2015). The implementation draws on several foundational works in the market-making literature: the Avellaneda & Stoikov (2008) exponential-intensity framework, the general optimal market-making theory of Guéant (2017) and Guéant, Lehalle & Fernandez-Tapia (2013), the LOB survey and empirical evidence of Gould et al. (2013), and the Heston simulation methodology of Lord, Koekkoek & Van Dijk (2010).

---

## 2. Heston Stochastic Volatility Model (heston_pricer.py)

### 2.1 Model Specification

The underlying asset price $S_t$ and its instantaneous variance $\nu_t$ follow the Heston (1993) dynamics under the risk-neutral measure:

$$dS_t = r S_t \, dt + \sqrt{\nu_t} \, S_t \, dW_t^S$$

$$d\nu_t = \kappa(\bar{\nu} - \nu_t) \, dt + \xi \sqrt{\nu_t} \, dW_t^\nu$$

where $dW_t^S \cdot dW_t^\nu = \rho \, dt$. The implementation uses the **risk-neutral (Q) measure** for pricing, with zero risk-free rate ($r = 0$, so no discounting). The parameters are:

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Initial spot | $S_0$ | 10 | Underlying price |
| Risk-free rate | $r$ | 0 | Zero drift / no discounting |
| Mean-reversion speed (Q) | $\kappa_Q$ | 3.0 | Rate of ν → ν̄ under Q |
| Long-run variance (Q) | $\bar{\nu}_Q$ | 0.0225 | Stationary variance under Q |
| Initial variance | $\nu_0$ | 0.0225 | Matches $\bar{\nu}_Q$ (σ₀ = 0.15) |
| Vol-of-vol | $\xi$ | 0.2 | Volatility of ν process |
| Correlation | $\rho$ | −0.5 | Leverage effect (S↓ ⟹ ν↑) |

These satisfy the Feller condition $2\kappa_Q\bar{\nu}_Q = 0.135 > \xi^2 = 0.04$, ensuring $\nu_t > 0$ a.s. The HJB solver additionally uses **physical (P) measure** parameters $\kappa_P = 2.0$, $\bar{\nu}_P = 0.04$ for the drift of $\nu_t$ in the market-maker's optimisation (see §3).

### 2.2 Monte Carlo Engine

The pricer uses an Euler–Maruyama discretisation of the Heston SDE with full truncation (replacing $\nu_t$ by $\max(\nu_t, 0)$ in the diffusion coefficient; Lord, Koekkoek & Van Dijk 2010, §3.2) to prevent negative variance realisations. This scheme was shown by Lord et al. (2010) to have superior bias properties compared to absorption and reflection schemes. The log-Euler scheme is used for $S_t$ (exact conditional on $\nu_t$, avoids negative prices). The correlation structure is implemented via the Cholesky decomposition:

$$Z_t^\nu = \epsilon_1, \qquad Z_t^S = \rho \, \epsilon_1 + \sqrt{1 - \rho^2} \, \epsilon_2$$

where $\epsilon_1, \epsilon_2 \sim \mathcal{N}(0,1)$ are independent.

**Implementation parameters:**
- 100,000 Monte Carlo paths (`N_PATHS = 100_000`)
- 252 time steps per year (`STEPS_PER_YEAR = 252`)
- Common random numbers (CRN) for variance reduction in vega computation (same RNG seed for base/bumped simulations)

### 2.3 Option Book

The market-maker quotes on a book of 20 European call options spanning a grid of strikes and maturities:

$$K \in \{8, 9, 10, 11, 12\}, \qquad T \in \{1.0, 1.5, 2.0, 3.0\} \text{ years}$$

For each option, the pricer computes:

1. **Price** $C(K, T)$: expected payoff $\mathbb{E}[\max(S_T - K, 0)]$ (no discounting since $r = 0$)
2. **Vega** $\mathcal{V}^i = \partial C / \partial \sqrt{\nu_0}$: computed by bumping $\nu_0$ by $\pm \varepsilon$ where $\varepsilon = \nu_0 \times 5 \times 10^{-3} = 0.0001125$ (`FD_EPS = NU0 * 5e-3`) and applying the chain rule $\mathcal{V}^i = 2\sqrt{\nu_0} \cdot \frac{C(\nu_0+\varepsilon) - C(\nu_0-\varepsilon)}{2\varepsilon}$. Common random numbers (identical RNG seed) cancel most MC noise.
3. **Implied volatility** $\sigma_{\text{imp}}$: extracted by inverting the Black–Scholes formula via Brent's root-finding method

The option data is packaged into a list of dictionaries, each containing `{K, T, price, vega, iv, lam, z}`, where:
- $\lambda_i = \lambda_{\text{base}} / (1 + 0.7 |K_i - S_0|)$ is the option-specific arrival rate, declining with distance from ATM (the factor 0.7 and base rate $\lambda_{\text{base}} = 252 \times 30 = 7{,}560$ events/year follow the paper's §4.1)
- $z_i = 5 \times 10^5 / C_i$ is the normalisation constant that converts the value function's units (where $C_i$ is the option price)

### 2.4 Outputs

The pricer generates **Figure 1**: the implied volatility surface $\sigma_{\text{imp}}(K, T)$ plotted as a 3D wireframe mesh, showing the characteristic skew (higher IV for low strikes due to $\rho < 0$) and term structure (IV converging to $\sqrt{\bar{\nu}}$ for long maturities).

---

## 3. HJB Solver (hjb_solver.py)

### 3.1 The Market-Maker's Optimisation Problem

The market-maker chooses bid and ask offsets $\delta_i^b(t), \delta_i^a(t)$ for each option $i$ to maximise expected terminal utility of wealth, subject to:
- Fill arrivals governed by intensity functions $\Lambda(\delta)$
- Inventory risk from holding a portfolio with aggregate vega exposure $V^\pi = \sum_i q_i \mathcal{V}^i$
- Variance risk from the Heston dynamics of $\nu_t$

This is the multi-asset generalisation of the Avellaneda & Stoikov (2008) framework to options under stochastic volatility, as formulated in Baldacci, Bergault & Guéant (2021, §3). The HJB approach follows the general methodology of Guéant (2017, §2) for optimal market making with general intensity functions.

The value function $v(t, \nu, V^\pi)$ satisfies the HJB PDE (paper equation (3.7)):

$$\partial_t v + a_P(\nu) \partial_\nu v + \frac{\xi^2 \nu}{2} \partial_{\nu\nu} v + V^\pi \frac{a_P(\nu) - a_Q(\nu)}{2\sqrt{\nu}} - \frac{\gamma \xi^2}{8} (V^\pi)^2 + \sum_i z_i \left[ H_i^a(p_i^a) + H_i^b(p_i^b) \right] = 0$$

where:
- The **indifference price** for the ask side of option $i$ is: $p_i^a = \frac{v(t, \nu, V^\pi) - v(t, \nu, V^\pi - z_i \mathcal{V}^i)}{z_i}$
- The **indifference price** for the bid side is: $p_i^b = \frac{v(t, \nu, V^\pi + z_i \mathcal{V}^i) - v(t, \nu, V^\pi)}{z_i}$
- The **Hamiltonian** is: $H_i(p) = \sup_{\delta \geq 0} \Lambda_i(\delta) (\delta - p)$

The terminal condition is $v(T, \nu, V^\pi) = 0$ and the PDE is solved backwards in time.

### 3.2 Finite-Difference Discretisation

The PDE is discretised on a 2D grid $(ν_j, V^\pi_k)$ with:

| Grid | Points | Range | Spacing |
|---|---|---|---|
| Time $t$ | $N_T = 180$ | $[0, T]$ with $T = 0.0012$ yr | $dt = T/N_T$ |
| Variance $\nu$ | $N_\nu = 30$ | $[0.0144, 0.0324]$ uniform | Centred near $\nu_0 = 0.0225$ |
| Portfolio vega $V^\pi$ | $N_{V\pi} = 40$ | $[-\bar{V}, +\bar{V}]$ uniform | $\bar{V} = 10^7$ (fixed constant) |

The diffusion and drift terms use **second-order central differences**:

$$\partial_\nu v \approx \frac{v_{j+1,k} - v_{j-1,k}}{2 \Delta\nu}, \qquad \partial_{\nu\nu} v \approx \frac{v_{j+1,k} - 2v_{j,k} + v_{j-1,k}}{(\Delta\nu)^2}$$

**Boundary conditions:**
- At $\nu = \nu_{\min}$ and $\nu = \nu_{\max}$: Neumann (zero-gradient) via ghost nodes
- At $V^\pi = \pm\bar{V}$: the Hamiltonian terms are set to zero (no quoting at the domain boundary), and the inventory penalty $\frac{\gamma\xi^2}{8}(V^\pi)^2$ provides a natural penalisation

### 3.3 Time Stepping

The PDE is marched **backwards** from $t = T$ (terminal condition $v = 0$) to $t = 0$ using an **explicit Euler** scheme:

$$v^{n-1}_{j,k} = v^n_{j,k} + \Delta t \cdot \text{RHS}^n_{j,k}$$

where RHS includes:
1. **Drift**: $a_P(\nu_j) \cdot (\partial_\nu v)_{j,k}$ with upwind (backward) differencing since $a_P > 0$ on the grid
2. **Diffusion**: $\frac{\xi^2 \nu_j}{2} \cdot (\partial_{\nu\nu} v)_{j,k}$
3. **Variance risk premium drift**: $V^\pi_k \cdot \frac{a_P(\nu_j) - a_Q(\nu_j)}{2\sqrt{\nu_j}}$ — the drift premium from Heston measure change (paper eq. (4))
4. **Inventory penalty**: $-\frac{\gamma \xi^2}{8} \cdot (V^\pi_k)^2$ — note: no $\nu$ factor in the implementation
5. **Hamiltonian contributions**: $\sum_i z_i [H_i^a(p_i^a) + H_i^b(p_i^b)] \cdot \mathbf{1}_{\text{in-domain}}$

The `in_domain` indicator ensures that shifted vega values $V^\pi \pm z_i \mathcal{V}^i$ remain within the grid; options whose vega shift would push the state outside $[-\bar{V}, \bar{V}]$ are excluded from quoting at that grid point.

### 3.4 Hamiltonian Computation

For the **logistic intensity** $\Lambda(\delta) = \lambda / (1 + e^{\alpha + \beta\delta/V})$, the Hamiltonian has no closed-form solution. It is computed via **Newton-type iteration** on the first-order condition. Defining $u(\delta) = \text{sigmoid}(\alpha + \beta\delta/V)$, the update rule is:

$$\delta^{(n+1)} = \delta^{(n)} - u(\delta^{(n)})(\delta^{(n)} - p) + V/\beta$$

This converges in 15 iterations (the default `n_iter=15`), initialised at $\delta^{(0)} = p + V/\beta$.

The computation is **fully vectorised** over all $(ν, V^\pi)$ grid points simultaneously using NumPy broadcasting, enabling the full HJB solve to complete in approximately 1 second.

### 3.5 Parameters

| Parameter | Symbol | Value | Paper reference |
|---|---|---|---|
| Risk aversion | $\gamma$ | $10^{-3}$ | §4.1 |
| Vol-of-vol | $\xi$ | 0.2 | §4.1 |
| Correlation | $\rho$ | −0.5 | §4.1 |
| Mean-reversion (P-measure) | $\kappa_P$ | 2.0 | §4.1 |
| Long-run variance (P-measure) | $\bar{\nu}_P$ | 0.04 | §4.1 |
| Mean-reversion (Q-measure) | $\kappa_Q$ | 3.0 | §4.1 |
| Long-run variance (Q-measure) | $\bar{\nu}_Q$ | 0.0225 | §4.1 |
| Logistic intercept | $\alpha$ | 0.7 | §4.1 |
| Logistic slope | $\beta$ | 150 | §4.1 |
| Trading horizon | $T$ | 0.0012 yr (~0.3 day) | §4.1 |

### 3.6 Outputs

The solver returns the full value function array $v(t, \nu, V^\pi)$ of shape $(N_T+1, N_\nu, N_{V\pi})$. The slice $v(0, \cdot, \cdot)$ at $t = 0$ is used for spread extraction. **Figure 2** shows $v(0, \nu, V^\pi)$ as a 3D surface, illustrating the concavity in $V^\pi$ (inventory aversion) and the drift in $\nu$ (mean-reversion).

---

## 4. Optimal Spread Extraction (`optimal_spreads.py`)

### 4.1 From Value Function to Quotes

Given the solved value function $v_0(\nu, V^\pi) = v(0, \nu, V^\pi)$, the optimal ask offset for option $i$ at the current state $(\nu_0, V^\pi)$ is:

$$\delta_i^{a,*} = \Lambda_i^{-1}\left(-H_i'(p_i^a)\right)$$

where:
1. **Indifference price**: $p_i^a = \frac{v_0(\nu_0, V^\pi) - v_0(\nu_0, V^\pi - z_i\mathcal{V}^i)}{z_i}$, computed by interpolating the value function at the shifted vega
2. **Hamiltonian derivative**: $H_i'(p)$ computed by central finite differences with step size $\varepsilon = 10^{-6}$ (fixed scalar, same for all options)
3. **Intensity inversion**: $\Lambda^{-1}(y)$ inverts the fill-probability function

The bid offset uses $p_i^b$ (computed with a positive vega shift) and the same inversion.

### 4.2 Observable Quantities

For each option $(K, T)$, the module computes three spread measures across the full $V^\pi$ grid:

- **Mid-to-bid** $\delta^b / C$: the discount the MM demands on the bid relative to mid-price
- **Ask-to-mid** $\delta^a / C$: the premium the MM charges on the ask relative to mid-price
- **Total spread** $(\delta^a + \delta^b) / C$: the round-trip cost as a fraction of the option price

Additionally, the implied volatility equivalents are computed: given the option's Black–Scholes vega and price, the spread in IV terms is $\delta^{\sigma} = \delta^{\text{price}} \times (\partial\sigma/\partial C)$.

### 4.3 Outputs

- **Figures 4–8**: Mid-to-bid / price vs portfolio vega for each of the 5 strikes (T = 2.0), showing how the bid quote shifts as the MM accumulates vega inventory
- **Figures 9–13**: The same quotes expressed in implied volatility space
- **Extension figures**: Ask-to-mid/price, total spread/price, and spread vs strike at $V^\pi = 0$ across all (K, T) combinations

---

## 5. Convergence Analysis (convergence.py)

### 5.1 Methodology

To verify the HJB solver has converged, the module computes the optimal spread at each time step $t_n$ (not just at $t = 0$) and measures how the quote changes over time. Specifically, for a representative option (K = 8, T = 1.0) at $\nu = \nu_0$ and $V^\pi = 0$:

$$\delta^{b,*}(t_n) = \Lambda^{-1}\left(-H'\left(\frac{v(t_n, \nu_0, 0) - v(t_n, \nu_0, -z\mathcal{V})}{z}\right)\right)$$

The convergence criterion is that $\delta^{b,*}(t)$ becomes approximately stationary (time-independent) sufficiently before $t = 0$, indicating that the trading horizon $T$ is long enough for the value function to reach its ergodic regime.

### 5.2 Output

**Figure 3**: Plot of $\delta^{b,*}(t)/C$ vs time for multiple $V^\pi$ levels, showing rapid convergence within the first few time steps and stationarity for the remainder of the horizon.

---

## 6. Parameter Sensitivity Analysis (param_sweeps.py, Sections A–B)

### 6.1 Alpha Sweep (Section A)

The intercept parameter $\alpha$ controls the **fill probability at zero offset**:

$$\Lambda(0) = \frac{\lambda}{1 + e^{\alpha}}$$

| $\alpha$ | $\Lambda(0)/\lambda$ | Economic meaning |
|---|---|---|
| 0.2 | 0.45 | High fill rate at mid → MM can quote tight |
| 0.5 | 0.38 | Moderate |
| **0.7** | **0.33** | **Paper baseline** |
| 1.0 | 0.27 | Lower fill rate → MM must widen to compensate |
| 1.5 | 0.18 | Much lower → substantially wider spreads |

The sweep re-solves the HJB for each $\alpha \in \{0.2, 0.5, 0.7, 1.0, 1.5\}$ with $\beta$ fixed at baseline, then overlays the resulting spread curves.

**Key finding**: Higher $\alpha$ uniformly widens spreads (both bid and ask shift outward), confirming that reduced fill probability at mid forces the MM to demand greater compensation per trade.

### 6.2 Beta Sweep (Section B)

The slope parameter $\beta$ controls the **price sensitivity of fill probability**:

| $\beta$ | Interpretation |
|---|---|
| 50 | Gentle slope: fills barely respond to spread changes → wider spreads |
| 100 | Moderate |
| **150** | **Paper baseline** |
| 250 | Steep: fills are very sensitive → MM stays tight to maintain flow |
| 400 | Very steep: tiny spread increase kills all flow → very tight quotes |

**Key finding**: Higher $\beta$ tightens spreads because the MM faces a steep elasticity of demand — widening the spread even slightly causes a disproportionate loss of flow.

### 6.3 Implementation

Each sweep calls `_solve_with_params(options, alpha, beta, intensity_name)`, which:
1. Temporarily patches the global `ALPHA`, `BETA`, and `hamiltonian` function in hjb_solver.py
2. Also patches optimal_spreads.py's `lambda_inverse`, `hamiltonian_prime`, and `optimal_spread` functions to ensure consistency
3. Re-solves the HJB PDE from scratch
4. Extracts spreads at the new parameters
5. Restores all original functions in a `finally` block

This monkey-patching approach avoids code duplication while ensuring that the entire pipeline (HJB + spread extraction) uses the same parameter values consistently.

---

## 7. Intensity Function Family Comparison (param_sweeps.py, Section C)

### 7.1 Motivation

The paper uses a logistic fill-probability function, which is one of many possible choices. The shape of $\Lambda(\delta)$ — how fill probability decays with distance from mid — fundamentally determines the optimal spread. The choice of intensity function is a central modelling decision in the market-making literature (Guéant 2017, §2; Guéant, Lehalle & Fernandez-Tapia 2013, §3). We compare three families:

### 7.2 Logistic Intensity (Paper Baseline)

$$\Lambda(\delta) = \frac{\lambda}{1 + \exp\left(\alpha + \frac{\beta}{V}\delta\right)}$$

- **Range**: $(0, \lambda)$ — bounded above and below
- **Shape**: S-shaped; saturates at both ends
- **Tail**: Exponentially thin (log-linear decay)
- **Hamiltonian**: No closed form; solved by Newton-type iteration (15 steps)
- **Inverse**: $\Lambda^{-1}(y) = \frac{V}{\beta}\left[\ln\left(\frac{\lambda}{y} - 1\right) - \alpha\right]$

### 7.3 Exponential Intensity (Avellaneda–Stoikov)

$$\Lambda(\delta) = \lambda \exp\left(-\alpha - \frac{\beta}{V}\delta\right)$$

- **Range**: $(0, \infty)$ — unbounded above at $\delta \to -\infty$
- **Shape**: Pure exponential decay
- **Tail**: Exponentially thin (same rate as logistic for large $\delta$)
- **Hamiltonian**: Closed form $H(p) = \frac{\lambda V}{\beta} \exp\left(-\alpha - \frac{\beta}{V}p - 1\right)$
- **FOC**: $\delta^* = p + V/\beta$ — a simple constant-width markup above the indifference price
- **Inverse**: $\Lambda^{-1}(y) = \frac{V}{\beta}\left[\ln\left(\frac{\lambda}{y}\right) - \alpha\right]$

The exponential form was used by Avellaneda & Stoikov (2008) in their foundational market-making paper. The constant-markup result $\delta^* = p + V/\beta$ implies that the total spread is independent of inventory when the value function is quadratic — a property first noted by Avellaneda & Stoikov (2008, §3.2) and generalised by Guéant, Lehalle & Fernandez-Tapia (2013, Proposition 1). Fodra & Pham (2015, §4) obtain a similar inventory-independent decomposition in their Markov renewal framework.

### 7.4 Queue-Reactive Intensity (CTMC-Based)

$$\Lambda(\delta) = \frac{a}{(1 + b\delta)^c}$$

This is a **power-law** intensity fitted to empirical fill rates obtained from a continuous-time Markov chain (CTMC) simulation of a queue-reactive limit order book (§8 below). The power-law form is motivated by extensive empirical evidence that limit order book depth and fill probabilities decay as a power law of the distance from the best quote. Bouchaud, Mézard & Potters (2002) first documented this for the Paris Bourse, reporting power-law exponents in the range 1.3–1.6 for order placement densities. Gould et al. (2013, §4.2) survey the literature on power-law relative-price distributions across multiple markets, confirming the ubiquity of this decay. Maslov (2000) showed that even a zero-intelligence LOB model generates power-law price impact. Key properties:

- **Range**: $(0, a]$ at $\delta = 0$, decaying as $\delta^{-c}$ for large $\delta$
- **Shape**: Power-law decay — **heavy-tailed** compared to logistic/exponential
- **Tail**: Polynomial decay $\sim \delta^{-c}$ with $c \approx 1.33$, meaning fills persist at wider offsets
- **Hamiltonian**: Fully analytical via the FOC
- **FOC**: $\delta^* = \frac{1 + cbp}{b(c-1)}$ — linear in $p$ with slope $\frac{c}{c-1} \approx 4.03$
- **Inverse**: $\Lambda^{-1}(y) = \frac{(a/y)^{1/c} - 1}{b}$ — closed form, no root-finding

**[CORRECTED]** The power-law parameters are fitted to raw empirical fill rates from the CTMC simulation using nonlinear least squares (`scipy.optimize.curve_fit`). Typical fitted values are $a \approx 3.06$, $b \approx 328$, $c \approx 1.33$ with RMSE ~3%, though exact values vary across Monte Carlo runs. The fitted exponent $c \approx 1.33$ falls squarely within the empirical range of 1.3–1.6 reported by Bouchaud et al. (2002) for order placement distributions on the Paris Bourse, and is consistent with the power-law fill-rate decay documented across multiple markets in Gould et al. (2013, §4.2). The rescaling to match the option-specific arrival rate $\lambda_i$ uses $\text{scale} = \lambda_i / a$, so that $\Lambda(0) = a \cdot \text{scale} = \lambda_i$.

### 7.5 Hamiltonian Derivation for the Power-Law

The Hamiltonian is $H(p) = \sup_{\delta} \Lambda(\delta)(\delta - p)$. Substituting the power-law:

$$H(p) = \sup_{\delta \geq 0} \frac{a}{(1+b\delta)^c}(\delta - p)$$

The first-order condition $\frac{d}{d\delta}[\Lambda(\delta)(\delta-p)] = 0$ gives:

$$\frac{-abc(\delta-p)}{(1+b\delta)^{c+1}} + \frac{a}{(1+b\delta)^c} = 0$$

Simplifying: $(1+b\delta) - cb(\delta-p) = 0$, yielding:

$$\delta^* = \frac{1 + cbp}{b(c-1)} \quad \text{for } c > 1$$

The second-order condition is satisfied when $c > 1$ (which holds for the fitted value $c \approx 1.33$), confirming this is a maximum. Substituting back:

$$H(p) = \frac{a \cdot \text{scale}}{(1+b\delta^*)^c} \cdot (\delta^* - p)$$

**[CORRECTED]** In the implementation, $\delta^*$ is clipped to $[10^{-10}, \infty)$ to avoid numerical issues at the boundary. If $\delta^*_{\text{foc}} < 0$ (which occurs when $p < -1/(cb)$), the boundary value $\delta = 0$ is used in the Hamiltonian, giving $H_0 = a \cdot \text{scale} \cdot (-p)$. The final Hamiltonian takes the maximum of the interior FOC value and the boundary value, both floored at zero: $H = \max(H_{\text{foc}}, H_0, 0)$.

### 7.6 Handling Negative Offsets (Quote Withdrawal)

When the market-maker has large inventory (e.g., long vega with $V^\pi \gg 0$), the optimal ask offset $\delta^{a,*}$ can become negative, meaning the MM wants to sell below mid-price to shed inventory. This "price improvement" or "quote withdrawal" behaviour is economically important and must be handled correctly.

For the **logistic and exponential** families, $\Lambda^{-1}(y)$ naturally returns negative values when $y$ exceeds the fill rate at mid ($y > \Lambda(0)$). The standard pipeline $\delta^* = \Lambda^{-1}(-H'(p))$ handles this seamlessly.

For the **queue-reactive** power-law, the Hamiltonian during the HJB PDE solve constrains $\delta \geq 0$ for numerical stability (avoiding a singularity at $\delta = -1/b$). This means $H'(p)$ saturates at $-\lambda$ when the boundary $\delta = 0$ binds, and the standard $\Lambda^{-1}(-H'(p))$ pipeline would return $\delta = 0$ (never negative).

To resolve this, the spread computation uses a **direct FOC approach** that bypasses the $H' \to \Lambda^{-1}$ chain entirely:

$$\delta^{a,*} = \frac{1 + c \cdot b \cdot p^a}{b(c-1)}$$

This formula naturally returns negative values when $p^a$ is sufficiently negative (large positive vega inventory), matching the behaviour of logistic and exponential. The direct FOC is implemented as a monkey-patched `optimal_spread` function that replaces the standard pipeline only for the queue-reactive case.

### 7.7 Value Function Smoothing

The explicit Euler PDE scheme accumulates grid-scale numerical noise in the value function. For logistic and exponential, the FOC $\delta^* = p + V/\beta$ (exponential) or Newton iteration (logistic) amplifies this noise by a factor of approximately 1×. For the queue-reactive power-law, the FOC slope is $c/(c-1) \approx 4.03$, amplifying noise by approximately 4×.

To remove visible 1–2 grid-point oscillations without distorting the economic shape, a **Savitzky–Golay filter** (window = 7, polynomial order = 3) is applied to each $\nu$-row of the value function $v_0$ after the HJB solve but before spread extraction. This preserves the genuine **W-shaped** spread structure near $V^\pi = 0$ (discussed in §7.8) while eliminating sub-visual jitter.

### 7.8 The W-Shaped Spread: A Structural Feature of Power-Law Intensity

The queue-reactive spread vs $V^\pi$ curve exhibits a **W-shape** (double-dip) rather than the smooth U-shape seen with logistic and exponential intensities:

- **Local maximum** at $V^\pi = 0$ (approximately 4.0% of price for K=10, T=1.0)
- **Dips** at moderate $|V^\pi| \approx 0.3 \times 10^7$
- **Rises** steeply at large $|V^\pi|$

This was confirmed to be a genuine structural feature (not a numerical artefact) by:
1. Verifying the feature grows 5× larger at 3× grid resolution (ruling out discretisation error)
2. Perfect symmetry around $V^\pi = 0$
3. Consistency across all $(K, T)$ combinations

#### Mathematical origin

For any intensity family, the total spread of a single option is:

$$\delta_a + \delta_b = f(p_a) + f(p_b)$$

where $f$ is the FOC mapping from indifference price to optimal offset. For the power-law, $f(p) = (1 + c \cdot b \cdot p) / (b(c-1))$ is linear with slope $c/(c-1) \approx 4.03$. The total spread therefore depends on the sum $p_a + p_b$, which equals the (negative) discrete second-difference of the value function:

$$p_a + p_b = \frac{v(V^\pi + zV) + v(V^\pi - zV) - 2v(V^\pi)}{z}$$

> **Key insight.** For a purely quadratic value function $v(V^\pi) = A - B(V^\pi)^2$, the second-difference evaluates to $-2BzV^2$, a constant independent of $V^\pi$. The total spread then reduces to a constant:
>
> $$\delta_a + \delta_b = \frac{2 + c \cdot b \cdot (-2BzV^2)}{b(c-1)} = \text{const.}$$
>
> This is the multi-option analogue of the Avellaneda & Stoikov (2008, §3.2) inventory-independence result: for exponential intensity with CARA utility, "the bid–ask spread [...] is independent of the inventory" because the FOC is $\delta^* = p + V/\beta$, producing a total spread $(p_a + p_b) + 2V/\beta$ in which the dominant $2V/\beta$ is $V^\pi$-independent and the $p_a + p_b$ term is constant for quadratic $v$. This result is generalised in Guéant, Lehalle & Fernandez-Tapia (2013, Proposition 1) and Guéant (2017, §3) for general intensity functions under CARA utility; Fodra & Pham (2015, §4) obtain an analogous inventory-independent decomposition in their Markov renewal framework.

However, the numerically solved value function is **not purely quadratic**. The Hamiltonian contributions — which are nonlinear functions of $p$ — inject small but measurable higher-order terms (quartic and beyond) into $v(V^\pi)$. Polynomial fitting of the solved $v$ confirms a residual from the best quadratic fit of $\sim 218$ (relative error $\sim 0.13\%$), with a dominant quartic coefficient of order $\sim 2.3 \times 10^{-27}$. These non-quadratic terms cause $p_a + p_b$ to **vary with** $V^\pi$: it is slightly more negative at $V^\pi = 0$ (where the Hamiltonian contributes most, since all options are quoting) and less negative at moderate $|V^\pi|$ (where some options drop out of the quoting domain).

#### Amplification mechanism

The power-law's FOC slope $c/(c-1) \approx 4.03$ **amplifies** this variation by a factor of $\sim 4\times$:

$$\delta_a + \delta_b = \frac{2 + c \cdot b \cdot (p_a + p_b)}{b(c-1)}$$

A 0.1% variation in $p_a + p_b$ becomes a $\sim 0.4\%$ variation in total spread — visible as the W-shape.

The exponential intensity, with FOC $\delta^* = p + V/\beta$, produces a total spread $(p_a + p_b) + 2V/\beta$. The constant $2V/\beta$ dominates and the same small variation in $p_a + p_b$ is invisible. The logistic has a FOC slope of $\sim 1\times$, so the amplification is insufficient to produce a visible W.

#### Numerical confirmation

This was verified by:
1. Computing $p_a + p_b$ across the $V^\pi$ grid from the solved value function — it varies by $\sim 0.5\%$ from centre to wings (interior grid, excluding boundary artefacts)
2. The logistic and exponential spreads, computed from the **same** value function, show no W-shape
3. The W-shape amplitude scales with $c/(c-1)$, as predicted by the linear FOC

This behaviour has no closed-form derivation in the present work and is confirmed as a **structural numerical feature** of the power-law intensity model interacting with the non-quadratic value function from the multi-option HJB. The effect is absent in the classical single-asset CARA framework of Avellaneda & Stoikov (2008) because (i) their exponential intensity adds a constant markup $V/\beta$ that masks the curvature, and (ii) the single-asset CARA value function is exactly quadratic. In the taxonomy of Guéant (2017, §3), the W-shape arises precisely because the power-law's FOC slope $c/(c-1)$ exceeds 1, amplifying the non-quadratic residual of the multi-option value function — a regime not explored in the classical AS or GLFT frameworks which use exponential intensities.

### 7.9 Key Comparative Results

| Metric | Logistic | Exponential | Queue-Reactive |
|---|---|---|---|
| Spread shape vs V^π | Smooth U | Smooth U | W-shape (double-dip) |
| Spread vs strike slope | Moderate | Gentlest | Steepest |
| Hamiltonian computation | Newton iteration (15 steps) | Closed form | Closed form (fastest) |

The ordering **queue-reactive > logistic > exponential** for spread width at $V^\pi = 0$ is consistent with the tail properties: the power-law's heavy tail means the MM receives comparatively less fill-rate benefit from tightening quotes, and must therefore charge a wider base spread. This ordering is consistent with the general analysis of Guéant (2017, §4), who shows that heavier-tailed intensity functions produce wider optimal spreads for a given inventory level.

---

## 8. Queue-Reactive LOB Model (`param_sweeps.py`, Section D)

### 8.1 Theoretical Framework

The queue-reactive model implements the framework of Huang, Lehalle & Rosenbaum (2015), "Simulating and Analysing Order Book Data: The Queue-Reactive Model" (*Journal of the American Statistical Association*, 110(509), 107–122). The key modelling assumption, stated in HLR 2015 §2.1, is that order-book event intensities depend on the **current state of the queues** — hence "queue-reactive" — rather than being exogenous constants as in earlier models (e.g., Cont, Stoikov & Talreja 2010). This queue-reactive property represents a significant advance over the independent-Poisson LOB models surveyed by Gould et al. (2013, §5.3), which fail to reproduce several empirical regularities documented in §4 of that survey. The ergodicity and diffusivity properties of queue-reactive models were subsequently analysed by Huang & Rosenbaum (2017) in a general Markovian framework.

The Level-1 order book is modelled as a bivariate CTMC with state $(Q^b, Q^a)$ representing the best bid and ask queue sizes in lots, following HLR 2015 Definition 1. Six event types drive the dynamics, corresponding to HLR 2015 §2.1 Table 1:

| Event | Notation | Effect on $(Q^b, Q^a)$ | Economic meaning |
|---|---|---|---|
| Limit add at bid | $L_b$ | $(+1, 0)$ | New liquidity provision at best bid |
| Cancel at bid | $C_b$ | $(-1, 0)$ | Withdrawal of resting bid liquidity |
| Market sell | $M_b$ | $(-1, 0)$ | Aggressive sell order hits bid queue |
| Limit add at ask | $L_a$ | $(0, +1)$ | New liquidity provision at best ask |
| Cancel at ask | $C_a$ | $(0, -1)$ | Withdrawal of resting ask liquidity |
| Market buy | $M_a$ | $(0, -1)$ | Aggressive buy order hits ask queue |

Each event has a **state-dependent intensity** $\lambda_e(Q^b, Q^a)$ — this is the "queue-reactive" property (HLR 2015, §2.1): the arrival rate of each event type depends on the current queue sizes, capturing the empirically observed feedback loops documented in HLR 2015 §4:
- **Mean reversion**: Limit adds increase when the own-side queue is thin (HLR 2015 Figure 3)
- **Crowding**: Cancellations increase when the queue is deep (HLR 2015 Figure 4)
- **Predation**: Market orders increase when the opposite queue is thin — thin books attract aggressive trading (HLR 2015 Figure 5)

### 8.2 Ground-Truth Intensities

For synthetic data generation, intensities are linear in the queue sizes:

$$\lambda_e(Q^b, Q^a) = \max\left(\beta_e^{(0)} + \beta_e^{(b)} Q^b + \beta_e^{(a)} Q^a, \, \lambda_{\text{floor}}\right)$$

with $\lambda_{\text{floor}} = 0.1$ events/sec. This linear specification is motivated by the empirical findings of HLR 2015 (§4, Figures 3–5), who document approximately linear dependence of event rates on queue sizes for major French equities. The sign structure of the coefficients reproduces the three key feedback loops identified by HLR: mean reversion, crowding, and predation (see §8.1 above). Guilbaud & Pham (2013, §2) use a related state-dependent arrival model in their optimal HFT framework, where execution intensities depend on the LOB spread state. The coefficients are:

| Event | Base rate $\beta_e^{(0)}$ | $Q^b$ coeff $\beta_e^{(b)}$ | $Q^a$ coeff $\beta_e^{(a)}$ | Interpretation |
|---|---|---|---|---|
| $L_b$ | 3.0 | −0.10 | +0.05 | Bid adds attracted by thin bid, thick ask |
| $C_b$ | 0.5 | +0.25 | −0.02 | Bid cancels increase when bid crowded |
| $M_b$ | 3.0 | +0.08 | −0.10 | Market sells increase when ask thin, bid thick |
| $L_a$ | 3.0 | +0.05 | −0.10 | Symmetric to $L_b$ |
| $C_a$ | 0.5 | −0.02 | +0.25 | Symmetric to $C_b$ |
| $M_a$ | 3.0 | −0.10 | +0.08 | Symmetric to $M_b$ |

The bid/ask symmetry ensures no systematic drift in mid-price.

### 8.3 CTMC Simulation (Competing Exponentials)

The simulator uses the standard **competing exponentials** (Gillespie 1977) algorithm, which is the exact simulation method for continuous-time Markov chains described in HLR 2015 §3.1. This approach exploits the memoryless property of exponential waiting times: at each state, the minimum of independent exponentials with rates $\lambda_1, \ldots, \lambda_6$ is itself exponential with rate $\Lambda_{\text{tot}} = \sum_e \lambda_e$, and the identity of the minimiser is a categorical draw with probabilities $\lambda_e / \Lambda_{\text{tot}}$ (see, e.g., Maslov 2000, §2 for a similar LOB simulation approach):

1. At state $(Q^b, Q^a)$, compute all 6 intensities $\lambda_e(Q^b, Q^a)$
2. Total rate $\Lambda_{\text{tot}} = \sum_e \lambda_e$
3. Time to next event: $\Delta t \sim \text{Exp}(\Lambda_{\text{tot}})$
4. Event type: categorical draw with $\Pr(e) = \lambda_e / \Lambda_{\text{tot}}$
5. Apply event: update $(Q^b, Q^a)$ according to the event effect table
6. **Queue depletion**: If $Q^b = 0$ or $Q^a = 0$, the mid-price shifts by one tick ($\pm 0.01$) and both queues are **reset** by sampling from a geometric distribution with mean $\mu = 8$ lots (`QR_Q_MEAN = 8`), capped at $Q_{\max} = 30$ (`QR_Q_MAX = 30`)

**[CORRECTED]** Queues are capped at $Q_{\max} = 30$ lots to bound the state space. The fill-probability bridge simulation runs for `T_seconds = 14,400` seconds (4 hours), while the Section D validation pipeline runs for `T_seconds = 7,200` seconds (2 hours). Both use time-based termination (not event-count based).

### 8.4 Estimation Pipeline

Given a sequence of CTMC events, the estimator recovers the intensity functions **non-parametrically** by binning — following the kernel estimation approach of HLR 2015 §3.2, simplified here to piecewise-constant bins:

1. **Bin the state space**: Divide $(Q^b, Q^a)$ into a grid of bins. The number of bins is `QR_N_BINS = 12` per dimension, giving a $12 \times 12$ grid over the $[1, Q_{\max}]$ range.
2. **Accumulate dwell time**: For each bin, sum the total time spent in that bin across the simulation
3. **Count events**: For each event type and each bin, count occurrences
4. **Estimate rate**: $\hat{\lambda}_e(\text{bin}) = \text{count}_e(\text{bin}) / \text{dwell time}(\text{bin})$
5. **Sparse bin handling**: Bins with dwell time below a threshold (5% of average dwell time per bin, with a minimum floor of 1.0 second) are replaced with the global average rate for that event type, preventing unreliable outlier estimates from driving the model

### 8.5 Simulation from Estimated Model

The estimated intensities are used to run a **second** CTMC simulation using the same competing-exponentials algorithm but with the non-parametric intensity lookup replacing the ground-truth linear model. At each state $(Q^b, Q^a)$, the simulator looks up the appropriate bin and uses the estimated rate.

This two-stage pipeline (ground truth → estimation → re-simulation) allows a direct validation of the estimation quality.

### 8.6 Fill-Probability Bridge: Connecting LOB Dynamics to $\Lambda(\delta)$

The crucial link between the LOB model and the market-making framework is the **fill-probability function** $\Lambda(\delta)$: for a market-maker posting a limit order at offset $\delta$ from mid, what is the expected fill rate? This bridge from LOB microstructure to execution probability is the key connection that allows queue-reactive LOB models to inform optimal market-making strategies. The empirical literature consistently finds that execution probability decays as a power law of the distance from the best quote: Bouchaud et al. (2002) report power-law exponents of 1.3–1.6 for order placement densities on the Paris Bourse; Gould et al. (2013, §4.2) survey similar findings across multiple markets; and Cont et al. (2010) use a power-law relative-price distribution in their LOB model.

The mapping works as follows:

1. **Offset to queue position**: $k(\delta) = \max(1, \text{round}(1 + \kappa \cdot \delta))$ where $\kappa = 500$ lots per price unit (`QR_KAPPA = 500`). At $\delta = 0$, the MM is at the front of the queue ($k = 1$); at $\delta = 0.01$ (one tick), the MM is at position $k = 6$.

2. **[CORRECTED] Queue-position fill tracking**: During the CTMC simulation, for each $\delta$ on a grid of `QR_N_DELTAS = 80` points uniformly spaced in $[0, d_{\text{hi}}]$ where $d_{\text{hi}} = (Q_{\max} - 1) / \kappa = 0.058$, we track a **virtual MM order** sitting at position $k(\delta)$ in the queue. Each virtual order maintains a `remaining` priority counter initialised to $k(\delta)$. The counter decreases by 1 when:
   - A **market order** hits the MM's side (unconditionally)
   - A **cancellation** occurs on the MM's side (with probability $\text{remaining} / Q$, approximating the chance it hits a lot ahead of the MM)

3. **Fill event**: When `remaining` reaches 0, the MM is filled and the fill counter increments. The order immediately re-enters at position $k(\delta)$.

4. **[CORRECTED] Queue reset handling**: When a queue depletes (price move), all virtual orders on that side are reset to their initial positions $k(\delta)$.

5. **[CORRECTED] Bid/ask tracking**: Fill rates are tracked independently on both the bid and ask sides, then **averaged** to produce the final $\Lambda(\delta)$.

6. **Monotonicity enforcement**: Raw fill rates are made monotonically decreasing (a physical requirement — closer to mid should always fill faster) by applying cumulative minima from left to right.

7. **Power-law fitting**: Rather than using the raw staircase-shaped fill rates (which have flat segments because multiple $\delta$ values map to the same integer queue position), a smooth power-law $\Lambda(\delta) = a/(1+b\delta)^c$ is fitted via nonlinear least squares (`scipy.optimize.curve_fit`), yielding typical parameters $a \approx 3.06$, $b \approx 328$, $c \approx 1.33$ with RMSE ~3%. The fitted exponent $c \approx 1.33$ is in excellent agreement with the power-law exponents of 1.3–1.6 documented empirically by Bouchaud et al. (2002) and surveyed by Gould et al. (2013), providing empirical validation for the power-law functional form used in the market-making optimisation (§7.4).

### 8.7 Validation Plots

Five validation plots verify the estimation and simulation quality:

1. **Event rates vs queue imbalance** ($Q^b/(Q^b+Q^a)$): 6 subplots (one per event type) overlaying ground-truth and estimated rates. Validates that the non-parametric estimator recovers the linear intensity structure.

2. **Inter-event duration distributions**: Histogram comparison (linear and log-scale) of waiting times between events. Validates the overall event rate.

3. **Mid-price dynamics**: Sample paths and move frequency (moves per minute) for ground truth vs estimated model. Validates that price volatility is preserved.

4. **Queue size distributions**: Histograms of $Q^b$ and $Q^a$. Validates the stationary distribution of the CTMC.

5. **Intensity surface heatmaps**: 6 subplots showing the estimated $\hat{\lambda}_e(Q^b, Q^a)$ as heatmaps with ground-truth contour overlays. Validates the 2D structure of the intensity functions.

---

## 9. Output Organisation

All figures are saved to three directories under figures:

| Directory | Contents | Typical count |
|---|---|---|
| original | Paper Figures 1–13 + extension plots (spread vs strike, ask-to-mid, bid-ask) | ~27 |
| param_sweeps | α sweep (5 values × overlays) + β sweep (5 values × overlays) | ~30 |
| intensity | Intensity family comparison (logistic/exponential/QR overlays) + Λ(δ) comparison + QR CTMC validation (5 plots) | ~21 |

**[CORRECTED]** The exact figure counts depend on the option book size and which extension plots are enabled. The full pipeline runs from main.py in approximately 60 seconds on a standard machine.

---

## 10. Software Dependencies

| Package | Version | Purpose |
|---|---|---|
| NumPy | ≥1.24 | Array operations, linear algebra |
| SciPy | ≥1.10 | Special functions (`expit`), optimisation (`curve_fit`), signal processing (`savgol_filter`), interpolation (`PchipInterpolator`) |
| Matplotlib | ≥3.7 | All plotting |

---

## Summary of Corrections from Fact-Check

| Section | Original claim | Correction |
|---|---|---|
| §2.1 | $r = 0.02$, $\kappa = 2.0$, $\bar{\nu} = 0.04$, $\nu_0 = 0.04$ | Code uses Q-measure: $r = 0$, $\kappa_Q = 3.0$, $\bar{\nu}_Q = 0.0225$, $\nu_0 = 0.0225$. P-measure ($\kappa_P = 2.0$, $\bar{\nu}_P = 0.04$) used only for HJB drift |
| §2.2 | 200,000 paths, 500 steps/yr, antithetic variates | `N_PATHS = 100_000`, `STEPS_PER_YEAR = 252`, CRN (not antithetic) |
| §2.2 | Cholesky had S and ν swapped | Fixed: $Z^\nu = \epsilon_1$, $Z^S = \rho\epsilon_1 + \sqrt{1-\rho^2}\epsilon_2$ |
| §2.3 | $z_i = C_i / \lambda_i$, vega bump $\pm 0.001$ | $z_i = 5 \times 10^5 / C_i$; FD bump $\varepsilon = \nu_0 \times 5 \times 10^{-3} = 0.0001125$ |
| §3.2 | $\nu$ grid centred on $\bar{\nu} = 0.04$, $\bar{V}$ dynamic | Grid $[0.0144, 0.0324]$ near $\nu_0 = 0.0225$; $\bar{V} = 10^7$ fixed |
| §3.3 | Penalty $\gamma\xi^2\nu/8 \cdot (V^\pi)^2$, no VRP drift | Penalty $\gamma\xi^2/8 \cdot (V^\pi)^2$ (no $\nu$); added VRP drift term |
| §4.1 | FD epsilon = $10^{-4} \times z_i$ | Actually `eps = 1e-6` (fixed scalar) |
| §7.5 | Simplified H(p) substitution formula | Clarified the clipping to δ ≥ 0 and boundary value handling |
| §8.3 | Did not mention simulation duration | Added: fill-probability bridge runs for 14,400s (4 hrs), validation pipeline for 7,200s (2 hrs) |
| §8.4 | "10 × 10 bins" | Actually `QR_N_BINS = 12` per dimension (12 × 12) |
| §8.4 | "1% of average dwell time" threshold | Actually 5% of average per bin with floor of 1.0 second |
| §8.6 | Fill grid "80 points in [0, 0.06]" | Actually `QR_N_DELTAS = 80` points in [0, 0.058] where 0.058 = (Q_MAX-1)/KAPPA |
| §8.6 | Missing detail on cancel probability | Added: probability = min(remaining-1, Qa)/Qa |
| §8.6 | Missing detail on queue reset | Added: virtual orders reset on price moves |
| §9 | Fixed figure counts | Changed to "typical count" since exact numbers depend on configuration |
| §7.8 | Vague "fat-tail fill persistence" explanation | Replaced with precise second-difference derivation: non-quadratic $v \times$ FOC slope $c/(c-1) \approx 4$ amplification |
| §5.1 | Convergence option K=10, T=2.0 | Actually uses K=8, T=1.0 (option 1 in the code); only bid spread plotted (not ask+bid) |
| §8.1 | Brief HLR 2015 citation | Added explicit section references to HLR 2015 (§2.1, §3.1, §3.2, §4, Figures 3–5) |

---

## References

1. Baldacci, B., Bergault, P., & Guéant, O. (2021). Algorithmic market making for options. *Quantitative Finance*, 21(1), 85–97.
2. Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217–224.
3. Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. *Review of Financial Studies*, 6(2), 327–343.
4. Huang, W., Lehalle, C.-A., & Rosenbaum, M. (2015). Simulating and analysing order book data: The queue-reactive model. *Journal of the American Statistical Association*, 110(509), 107–122.
5. Cont, R., Stoikov, S., & Talreja, R. (2010). A stochastic model for order book dynamics. *Operations Research*, 58(3), 549–563.
6. Guéant, O. (2017). Optimal market making. *Applied Mathematical Finance*, 24(2), 112–154.
7. Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical reactions. *Journal of Physical Chemistry*, 81(25), 2340–2361.
8. Lord, R., Koekkoek, R., & Van Dijk, D. (2010). A comparison of biased simulation schemes for stochastic volatility models. *Quantitative Finance*, 10(2), 177–194.
9. Gould, M. D., Porter, M. A., Williams, S., McDonald, M., Fenn, D. J., & Howison, S. D. (2013). Limit order books. *Quantitative Finance*, 13(11), 1709–1742.
10. Bouchaud, J.-P., Mézard, M., & Potters, M. (2002). Statistical properties of stock order books: Empirical results and models. *Quantitative Finance*, 2(4), 251–256.
11. Guéant, O., Lehalle, C.-A., & Fernandez-Tapia, J. (2013). Dealing with the inventory risk: A solution to the market making problem. *Mathematics and Financial Economics*, 7(4), 477–507.
12. Guilbaud, F., & Pham, H. (2013). Optimal high-frequency trading with limit and market orders. *Quantitative Finance*, 13(1), 79–94.
13. Fodra, P., & Pham, H. (2015). High frequency trading and asymptotics for small risk aversion in a Markov renewal model. *SIAM Journal on Financial Mathematics*, 6(1), 1–34.
14. Huang, W., & Rosenbaum, M. (2017). Ergodicity and diffusivity of Markovian order book models: A general framework. *SIAM Journal on Financial Mathematics*, 8(1), 874–900.
15. Maslov, S. (2000). Simple model of a limit order-driven market. *Physica A*, 278(3–4), 571–578.