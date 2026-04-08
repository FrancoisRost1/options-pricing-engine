# CLAUDE.md — Project 9: Options Pricing Engine

> Local project spec. This is the single source of truth for this project.
> The master `~/Documents/CODE/CLAUDE.md` carries only a summary — this file has full detail.
> Read this fully before writing or modifying any code.

---

## What this project is

**Project 9 of 11** in a "God-Tier" finance GitHub portfolio. First project in the 🟣 ELITE tier.

**One-line:** Options pricing and Greeks calculator with Black-Scholes, binomial tree, Monte Carlo pricing, empirical implied volatility surface construction, and model-vs-market validation.

**Repo:** `github.com/FrancoisRost1/options-pricing-engine`

**Cross-project reuse:** The Options Greeks module feeds directly into Project 11 (mini-bloomberg-terminal).

---

## Coding rules (inherited from master CLAUDE.md)

- One file = one responsibility. `main.py` orchestrates only.
- All weights, thresholds, assumptions → `config.yaml`. Never hardcode.
- Docstring on every class and method — explain financial rationale, not just mechanics.
- Edge cases: division by zero → `np.nan`, missing values, negative inputs handled gracefully.
- No file longer than ~150 lines. Split if needed.
- `pandas` for data manipulation. `numpy` only where needed.
- Config loaded once via `utils/config_loader.py`, passed as dict.
- Streamlit import rule: launch from project root with `streamlit run app/app.py`. All imports inside `app/` use relative imports. `app/` must contain `__init__.py`.
- Bloomberg dark mode dashboard (copy `style_inject.py` and `.streamlit/config.toml` from DESIGN.md).
- Python: `python3` (not `python`). Package manager: `pip3`. OS: macOS, zsh.

---

## Architecture overview

### Pricing models

| Model | Scope | Greeks method |
|-------|-------|---------------|
| **Black-Scholes** | European calls & puts (analytical) | Analytical closed-form (standard 5 + higher-order) |
| **CRR Binomial Tree** | European & American calls & puts | Central finite difference (standard 5 only) |
| **Monte Carlo (GBM)** | European calls & puts only | Central finite difference (standard 5 only) |

**American option scope:** American pricing handled by CRR binomial tree only. Monte Carlo supports European pricing only. Longstaff-Schwartz for American MC is out of scope for v1 — may be added as an explicit extension.

### Option types

- **European:** priced by all 3 models (BS, Binomial, MC)
- **American:** priced by Binomial tree only (early exercise via backward induction)

---

## Data source

### Options chain data

- **Primary:** yfinance — US equity options chains
- **Fallback:** Synthetic data generator (GBM-based) for testing / when yfinance unavailable
- **Market price definition:** mid price = (bid + ask) / 2
- **Fallback hierarchy:**
  1. Mid price if both bid and ask are valid and non-zero
  2. If bid/ask invalid or missing → exclude contract from model-vs-market analysis
  3. Last price used only with an explicit `price_source` flag column — never silently mixed

### Risk-free rate

- **Primary:** yfinance Treasury yields
  - Short-dated options (DTE < threshold in config) → `^IRX` (13-week T-bill rate)
  - Longer-dated options → `^TNX` (10-year Treasury yield)
  - Threshold defined in `config.yaml`
- **Fallback:** Fixed rate from `config.yaml`
- **Simplifying assumption:** Flat yield curve (stated explicitly). No full term structure interpolation in v1.
- **Critical:** yfinance returns percentage values → must divide by 100

### Dividends

- Continuous dividend yield estimated from yfinance trailing annual dividend / current price
- Simplifying assumption: constant continuous yield over option life

---

## Implied volatility extraction

### Solver

- **Method:** Brent's root-finding (`scipy.optimize.brentq`)
- **Objective:** find σ such that BS_price(σ) - market_price = 0
- **Pre-solve validation:**
  - Check market price > intrinsic value (otherwise no valid IV)
  - Check market price < upper bound (S for calls, K×e^(-rT) for puts)
  - Check market price > 0
- **Bounds:** σ ∈ [0.001, 10.0] (configurable in `config.yaml`)
- **Failure policy:** If solver fails or bounds check fails → return `np.nan`. **Never fake-fill.**
- **Tolerance:** configurable in `config.yaml` (default 1e-8)

---

## Options chain filtering (garbage filter)

All thresholds configurable in `config.yaml`:

| Filter | Config field | Default |
|--------|-------------|---------|
| Max bid-ask spread % | `max_spread_pct` | 50% |
| Min volume | `min_volume` | 10 |
| Min open interest | `min_open_interest` | 10 |
| Moneyness bounds (log) | `moneyness_bounds` | [-0.5, 0.5] |
| Min days to expiry | `min_dte` | 1 |

Contracts failing any filter are excluded from model-vs-market analysis and surface construction. Raw chain still available in Chain Explorer tab.

---

## Greeks

### Standard Greeks (all 3 models)

| Greek | Definition | Unit |
|-------|-----------|------|
| **Delta** | ∂V/∂S | per $1 move in spot |
| **Gamma** | ∂²V/∂S² | per $1 move in spot |
| **Theta** | ∂V/∂t | **per calendar day** (÷365) — locked convention |
| **Vega** | ∂V/∂σ | per +1 vol point (+1%, i.e. +0.01 in σ) |
| **Rho** | ∂V/∂r | per +1% rate move (+0.01 in r) |

### Higher-order Greeks (Black-Scholes analytical only)

| Greek | Definition | Unit |
|-------|-----------|------|
| **Vanna** | ∂²V/∂S∂σ = ∂Delta/∂σ | per +1 vol point per $1 spot |
| **Volga (Vomma)** | ∂²V/∂σ² = ∂Vega/∂σ | per +1 vol point |
| **Charm** | ∂²V/∂S∂t = ∂Delta/∂t | per calendar day per $1 spot |

**Higher-order Greeks scope:** Analytical closed-form under BS only. Do NOT force higher-order through binomial/MC — numerical noise makes them unreliable. Better to have fewer credible outputs than fake precision.

### Finite difference method (Binomial & MC)

- Central difference: `(V(x+h) - V(x-h)) / (2h)`
- Bump sizes configurable in `config.yaml`:
  - `bump_spot`: default 0.01 (1% of S)
  - `bump_vol`: default 0.01 (1 vol point)
  - `bump_rate`: default 0.01 (1% rate)
  - `bump_time`: default 1/365 (1 day)

---

## Pricing models — detailed spec

### Black-Scholes

```
d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
d2 = d1 - σ√T

Call = S × e^(-qT) × N(d1) - K × e^(-rT) × N(d2)
Put  = K × e^(-rT) × N(-d2) - S × e^(-qT) × N(-d1)

Where:
  S = spot price
  K = strike price
  r = risk-free rate (continuous)
  q = continuous dividend yield
  T = time to expiry (years)
  σ = volatility (annualized)
  N() = standard normal CDF
```

### CRR Binomial Tree

```
u = e^(σ√dt)           # up factor
d = 1/u                 # down factor
p = (e^((r-q)dt) - d) / (u - d)   # risk-neutral probability

dt = T / N              # time step (N = number of steps, default 200)

Backward induction:
  European: V(i,j) = e^(-r×dt) × [p × V(i+1,j+1) + (1-p) × V(i+1,j)]
  American: V(i,j) = max(exercise_value, continuation_value)
```

- Steps configurable in `config.yaml` (default 200)
- Early exercise check at every node for American options

### Monte Carlo

```
S(t+dt) = S(t) × exp((r - q - σ²/2)dt + σ√dt × Z)

Where Z ~ N(0,1)
dt = T / time_steps (explicit discretization)
```

**Variance reduction techniques:**
1. **Antithetic variates:** For each Z, also simulate -Z. Average payoffs.
2. **Control variate:** Use BS analytical price as control.
   - Adjusted payoff = MC_payoff - β × (MC_control - BS_price)
   - Two modes (configurable in `config.yaml`):
     - `fixed`: β = 1.0 (standard assumption)
     - `estimate`: β = Cov(MC_payoff, CV_payoff) / Var(CV_payoff), estimated from all paths
   - Dashboard and output must clearly state which β mode was used for each run

**Configuration:**
- `mc_paths`: default 100,000
- `mc_time_steps`: default 252 (daily)
- `mc_seed`: default 42 (reproducibility)

**Scope limitation:** MC prices European options only in v1. Clearly documented: MC is not needed for vanilla European (BS is exact) — MC is implemented as foundation for path-dependent extensions and to demonstrate variance reduction.

---

## Volatility surface

### Construction

- **Type:** Empirical implied volatility surface via interpolation/smoothing
- **NOT** a fully arbitrage-free calibrated surface (no SVI, no arbitrage constraints in v1)
- **Axes:** Log-moneyness ln(K/S) × Time to expiry (years)
- **Interpolation:** `scipy.interpolate.griddata` (cubic) or `RBFInterpolator`
- **Input:** Filtered implied vols from options chain (post garbage filter)

### Visualizations

1. **Smile per expiry:** IV vs log-moneyness for each expiry (2D line plot, overlay expiries)
2. **Term structure:** ATM implied vol vs time to expiry
3. **3D surface:** Strike × Expiry × IV (Plotly surface plot)
4. **2D slices:** User-selectable cuts through the surface

### Skew metrics

- **25Δ Risk Reversal:** σ(25Δ call) − σ(25Δ put)
  - 25Δ strikes obtained by interpolation on the smile: find K where BS_delta(K, σ(K)) = 0.25 (call) or -0.25 (put)
  - Equity convention caveat: these metrics are borrowed from FX vol conventions. For equities, they serve as skew summary statistics. The method of obtaining 25Δ strikes (interpolation on the smile) is documented explicitly.
- **25Δ Butterfly:** 0.5 × [σ(25Δ call) + σ(25Δ put)] − σ(ATM)
  - Measures curvature/convexity of the smile

---

## Model vs Market comparison

### Price comparison

- Scatter plot: model price (y) vs market mid price (x)
- Error = model price − market mid price
- Error statistics: RMSE, MAE, mean error, max error
- **Breakdown by:** moneyness bucket × maturity bucket (not just raw scatter)

### Put-call parity validation

- **European options only** — do NOT validate on American chains
- Same strike, same expiry, same dividend/rate assumptions
- Parity: `C - P = S × e^(-qT) - K × e^(-rT)`
- Report parity deviation in $ and as % of mid price
- Flag violations above configurable threshold

---

## P&L scenario analysis

### Heatmap

- Axes: spot price change (%) × implied vol change (%)
- Cell value: option P&L (or new option value)
- Time decay axis as optional third dimension (slider for days forward)

### Scenario presets

| Preset | Spot change | Vol change | Rate change | Description |
|--------|------------|------------|-------------|-------------|
| Earnings | ±5%, ±10% | +10 vol pts pre, −15 vol pts post | 0 | Pre/post earnings vol dynamics |
| Vol crush | 0% | −10, −20 vol pts | 0 | Post-event vol compression |
| Rate hike | 0% | 0 | +25 bps, +50 bps | Small impact but included for completeness |

Presets are mechanical — no fake macro storytelling. Values configurable in `config.yaml`.

---

## Delta hedging simulation

### Design

- **Framing:** Self-financing replication portfolio illustration. NOT a profitable strategy.
- **Portfolio components:**
  1. Option position (long/short configurable)
  2. Stock hedge (delta shares)
  3. Cash account accruing at risk-free rate r
  4. Transaction costs + slippage

### Implementation

- **Hedging frequency:** Configurable (default daily)
- **Paths:**
  - Realized historical path (from yfinance) when available
  - Simulated GBM paths as fallback / comparison
- **Costs:**
  - Transaction cost: `hedge_tc_bps` (default 5 bps) per rebalance on notional traded
  - Slippage: `hedge_slippage_bps` (default 2 bps) additional

### P&L decomposition (critical output)

```
Total hedging P&L = Gamma/Theta effect + Hedging error + Transaction costs

Where:
  Gamma/Theta = ½ × Gamma × (ΔS)² − Theta × Δt  (per period)
  Hedging error = residual (discrete vs continuous hedging gap)
  TC = turnover × (tc_bps + slippage_bps) / 10000
```

### Output

- Cumulative P&L over time
- Decomposition chart: gamma/theta P&L vs hedging error vs TC drag
- Summary statistics: total P&L, hedging error std dev, cost drag

---

## Model agreement / convergence (interview gold feature)

### Cross-model comparison

- BS vs Binomial vs MC price for same contract
- Table + chart showing agreement/disagreement
- Expected: near-perfect agreement for European vanilla options

### Convergence analysis

- **Binomial:** Price vs number of steps (show convergence to BS)
  - Plot price at N = 10, 25, 50, 100, 200, 500, 1000
  - Optional: show odd/even oscillation characteristic of CRR
- **Monte Carlo:** Price vs number of paths
  - Plot price + 95% CI at paths = 1k, 5k, 10k, 50k, 100k
  - Show variance reduction effect: standard MC vs antithetic vs control variate

---

## Dashboard — 7 tabs

### Tab 1: Chain Explorer (entry point)

- Ticker input + load all expiries
- Full options chain display (calls + puts)
- Filters: expiry, moneyness range, liquidity (volume, OI, spread)
- Smart defaults: near ATM, liquid options, nearest 3-4 expiries
- Columns: strike, bid, ask, mid, last, volume, OI, IV, delta

### Tab 2: Pricing

- Single option pricer: input S, K, T, r, q, σ → price from all 3 models
- Cross-model comparison table (BS vs Binomial vs MC)
- Convergence charts (binomial steps, MC paths)
- Variance reduction comparison (standard vs antithetic vs control variate)

### Tab 3: Greeks

- Greeks table for selected option or full chain
- Greeks vs spot (P&L profile curves)
- Greeks vs moneyness
- Higher-order Greeks (BS only) clearly labeled
- Interpretation callout on every chart (financial meaning, not just math)

### Tab 4: Volatility Surface

- 3D surface plot (log-moneyness × expiry × IV)
- Smile per expiry (2D overlay)
- Term structure (ATM vol vs expiry)
- Skew metrics: 25Δ RR and butterfly per expiry
- All axes labeled with units

### Tab 5: Model vs Market

- Price comparison scatter with 45° line
- Error breakdown by moneyness × maturity
- Put-call parity violations (European only)
- Error statistics panel

### Tab 6: Scenario Analysis

- P&L heatmap (spot × vol)
- Time decay slider
- Preset buttons (Earnings, Vol Crush, Rate Hike)
- Position builder: select option(s) to analyze

### Tab 7: Delta Hedge

- Hedging simulation controls (frequency, costs, path source)
- Cumulative P&L chart
- P&L decomposition: gamma/theta vs hedging error vs TC
- Summary statistics
- Interpretation callout: "This illustrates discrete hedging replication — not a trading strategy"

---

## File structure

```
options-pricing-engine/
├── CLAUDE.md                    ← this file
├── README.md
├── config.yaml
├── requirements.txt
├── main.py                      ← orchestration only
├── analysis.md                  ← investment thesis / secret sauce
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           ← yfinance options chain + spot + rates
│   ├── synthetic_data.py        ← GBM-based synthetic chain generator
│   ├── black_scholes.py         ← BS pricing + analytical Greeks (all)
│   ├── binomial_tree.py         ← CRR pricer (European + American)
│   ├── monte_carlo.py           ← MC pricer (European only, variance reduction)
│   ├── greeks.py                ← finite difference Greeks wrapper
│   ├── implied_vol.py           ← IV extraction (Brent solver)
│   ├── vol_surface.py           ← surface construction + interpolation
│   ├── skew_metrics.py          ← 25Δ RR, butterfly, smile analytics
│   ├── model_comparison.py      ← cross-model price comparison + convergence
│   ├── parity_check.py          ← put-call parity validation (European only)
│   ├── scenario_analysis.py     ← P&L heatmaps + presets
│   ├── delta_hedge.py           ← hedging simulation + P&L decomposition
│   └── chain_filter.py          ← garbage filter (configurable thresholds)
│
├── utils/
│   ├── __init__.py
│   └── config_loader.py         ← load config.yaml once, pass as dict
│
├── tests/
│   ├── __init__.py
│   ├── test_black_scholes.py
│   ├── test_binomial_tree.py
│   ├── test_monte_carlo.py
│   ├── test_greeks.py
│   ├── test_implied_vol.py
│   ├── test_vol_surface.py
│   ├── test_skew_metrics.py
│   ├── test_model_comparison.py
│   ├── test_parity_check.py
│   ├── test_scenario_analysis.py
│   ├── test_delta_hedge.py
│   ├── test_chain_filter.py
│   └── test_integration.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
│
├── app/
│   ├── __init__.py
│   ├── app.py                   ← main Streamlit app (7 tabs)
│   ├── style_inject.py          ← Bloomberg dark mode (from DESIGN.md)
│   ├── tab_chain_explorer.py
│   ├── tab_pricing.py
│   ├── tab_greeks.py
│   ├── tab_vol_surface.py
│   ├── tab_model_vs_market.py
│   ├── tab_scenarios.py
│   └── tab_delta_hedge.py
│
├── .streamlit/
│   └── config.toml              ← dark theme config
│
├── docs/
│   └── analysis.md
│
└── outputs/
```

---

## Dependencies

```
pandas
numpy
scipy
yfinance
pyyaml
streamlit
plotly
pytest
numpy-financial
```

---

## Simplifying assumptions (documented explicitly)

1. **Flat yield curve** — single risk-free rate mapped by DTE bucket, not full term structure
2. **Continuous dividend yield** — constant over option life, estimated from trailing data
3. **Log-normal returns** — GBM assumption for BS and MC (no jumps, no stochastic vol)
4. **No transaction costs in pricing** — costs only in delta hedging simulation
5. **European exercise for MC** — no Longstaff-Schwartz in v1
6. **Empirical vol surface** — interpolation/smoothing only, not arbitrage-free calibrated
7. **25Δ skew metrics** — equity adaptation of FX convention, strikes found by interpolation on smile

---

## Key lessons from prior projects (prevent recurring bugs)

1. Plotly chart titles: always explicit string title, never let theme overwrite with dict
2. Streamlit imports: relative imports + `__init__.py` in `app/`
3. Sharpe: arithmetic mean excess return, not CAGR
4. Matrix inversion: use `np.linalg.solve()` not `np.linalg.inv()`
5. Solver success: always check `result.success` from scipy optimizers, fallback on failure
6. Interpretation callouts: every chart gets a one-liner explaining financial meaning
7. Framing: this is understanding the instrument, not beating the market

---

*Project 9 CLAUDE.md — Options Pricing Engine*
*Spec locked: 2026-04-08*
