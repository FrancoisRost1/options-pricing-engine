# Options Pricing Engine

Options pricing and Greeks calculator with Black-Scholes, CRR binomial tree, and Monte Carlo (GBM) pricing. Empirical implied volatility surface construction, model-vs-market validation, scenario analysis, and delta hedging simulation.

**Project 9 of 11** in a finance engineering portfolio. First project in the Elite tier.

---

## Features

### Three Pricing Models
- **Black-Scholes** — analytical closed-form for European calls and puts, 8 analytical Greeks (Delta, Gamma, Theta, Vega, Rho + Vanna, Volga, Charm)
- **CRR Binomial Tree** — European and American options, early exercise via backward induction, convergence analysis vs BS
- **Monte Carlo (GBM)** — European options with antithetic variates and control variate variance reduction, configurable beta estimation

### Implied Volatility
- Brent's root-finding with pre-solve arbitrage bound validation
- Batch extraction across full options chains
- Failure policy: `np.nan` — never fake-filled

### Volatility Surface
- Empirical IV surface via cubic interpolation (log-moneyness x expiry)
- Smile per expiry, ATM term structure
- 25-delta risk reversal and butterfly skew metrics (FX convention adapted for equities)

### Model vs Market
- BS model price vs market mid scatter with error breakdown by moneyness bucket
- Put-call parity validation (European only, with American option guardrail)
- RMSE, MAE, mean error, max error statistics

### Scenario Analysis
- P&L heatmap: spot change (%) x vol change (vol points)
- Time decay slider (days forward)
- Mechanical presets: Earnings, Vol Crush, Rate Hike

### Delta Hedging Simulation
- Self-financing replication portfolio with financing carry
- P&L decomposition: Gamma/Theta + Hedging error + Transaction costs + Financing
- Configurable frequency (daily/weekly), TC (bps), slippage
- GBM simulated paths or historical data

### Dashboard
7-tab Streamlit app with Bloomberg dark mode theme:
1. **Chain Explorer** — browse raw/filtered options chain
2. **Pricing** — single-option pricer, cross-model comparison, convergence charts
3. **Greeks** — all 8 Greeks with profiles vs spot
4. **Vol Surface** — 3D surface, smile overlay, term structure, skew metrics
5. **Model vs Market** — price scatter, error breakdown, parity check
6. **Scenarios** — P&L heatmap with presets
7. **Delta Hedge** — hedging simulation with P&L decomposition

---

## Screenshot

> *Screenshot placeholder — add after first deployment*

---

## File Structure

```
options-pricing-engine/
├── CLAUDE.md                    # Project spec (source of truth)
├── README.md                    # This file
├── config.yaml                  # All parameters, thresholds, assumptions
├── requirements.txt             # Python dependencies
├── main.py                      # CLI entry point
│
├── src/
│   ├── black_scholes.py         # BS pricing + 8 analytical Greeks
│   ├── binomial_tree.py         # CRR pricer (European + American)
│   ├── monte_carlo.py           # MC pricer with variance reduction
│   ├── greeks.py                # Finite difference Greeks (binomial/MC)
│   ├── implied_vol.py           # IV extraction (Brent solver)
│   ├── data_loader.py           # yfinance options chain + spot + rates
│   ├── synthetic_data.py        # GBM-based synthetic chain fallback
│   ├── chain_filter.py          # Configurable garbage filter
│   ├── vol_surface.py           # Surface construction + interpolation
│   ├── skew_metrics.py          # 25-delta RR, butterfly
│   ├── model_comparison.py      # Cross-model comparison + convergence
│   ├── parity_check.py          # Put-call parity (European only)
│   ├── scenario_analysis.py     # P&L heatmaps + presets
│   └── delta_hedge.py           # Hedging simulation + P&L decomposition
│
├── utils/
│   └── config_loader.py         # Load config.yaml once, pass as dict
│
├── tests/                       # 230 tests (pytest)
│   ├── test_black_scholes.py    # Textbook values, edge cases, all Greeks
│   ├── test_binomial_tree.py    # Convergence, American premium, edges
│   ├── test_monte_carlo.py      # BS tolerance, variance reduction, seeds
│   ├── test_greeks.py           # FD vs analytical, American guard
│   ├── test_implied_vol.py      # Round-trip, 9 failure modes
│   ├── test_chain_filter.py     # Each filter criterion independently
│   ├── test_vol_surface.py      # Surface, smile, term structure
│   ├── test_skew_metrics.py     # Skew direction, butterfly
│   ├── test_model_comparison.py # 3-model agreement, convergence
│   ├── test_parity_check.py     # Parity on BS prices, violation detection
│   ├── test_scenario_analysis.py# Grid, presets, time decay
│   ├── test_delta_hedge.py      # Simulation, TC, positions, GBM paths
│   ├── test_synthetic_data.py   # Chain structure, skew, reproducibility
│   └── test_integration.py      # Full pipeline end-to-end
│
├── app/
│   ├── app.py                   # Main Streamlit app (7 tabs)
│   ├── style_inject.py          # Bloomberg dark mode design system
│   ├── tab_chain_explorer.py
│   ├── tab_pricing.py
│   ├── tab_greeks.py
│   ├── tab_vol_surface.py
│   ├── tab_model_vs_market.py
│   ├── tab_scenarios.py
│   └── tab_delta_hedge.py
│
├── .streamlit/config.toml       # Dark theme config
├── docs/analysis.md             # Investment thesis + assumptions
└── data/                        # Raw, processed, cache directories
```

---

## Installation

```bash
git clone https://github.com/FrancoisRost1/options-pricing-engine.git
cd options-pricing-engine
pip3 install -r requirements.txt
```

## Usage

### CLI
```bash
python3 main.py                    # Default ticker (AAPL)
python3 main.py --ticker MSFT      # Specific ticker
python3 main.py --synthetic        # Synthetic data (no network)
```

### Dashboard
```bash
streamlit run app/app.py
```

### Tests
```bash
python3 -m pytest tests/ -v
```

---

## Key Results

- **3 models agree** within $0.002 on ATM European options (BS, Binomial 200 steps, MC 100k paths)
- **230 tests passing** — textbook values (Hull), edge cases, integration pipeline
- **Codex-audited** (GPT-5.4, HIGH reasoning) — 7 bugs fixed including financing carry in delta hedge, charm sign convention, IV upper bound with dividends
- **650+ valid IVs** extracted from live AAPL chain (809 filtered contracts)
- **7-tab Bloomberg dark mode dashboard** with interpretation callouts on every chart

## Simplifying Assumptions

1. Flat yield curve (short/long rate by DTE bucket, not full term structure)
2. Continuous dividend yield (constant over option life)
3. Log-normal returns (GBM — no jumps, no stochastic vol)
4. No transaction costs in pricing (costs only in delta hedging simulation)
5. European exercise for MC (no Longstaff-Schwartz in v1)
6. Empirical vol surface (interpolation only, not arbitrage-free calibrated)

---

## Dependencies

```
pandas, numpy, scipy, yfinance, pyyaml, streamlit, plotly, pytest, numpy-financial
```

---

*Built as part of a finance engineering portfolio signaling real financial understanding, strong data engineering, and decision-making tools investors actually use.*
