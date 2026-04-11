# Analysis: Options Pricing Engine

## Investment Thesis

This project is not about generating alpha from options trading. It is about **understanding the instrument**, how options are priced, how Greeks drive risk exposure, and how hedging works in practice versus theory.

The thesis is that a practitioner who can build an options pricing engine from scratch, implementing Black-Scholes, binomial trees, Monte Carlo simulation, implied volatility extraction, and delta hedging with P&L decomposition, demonstrates a depth of understanding that is qualitatively different from someone who can only use Bloomberg's OVME screen or plug numbers into a formula sheet.

### Why This Matters for PE/HF

1. **Structured products due diligence**, PE firms increasingly encounter embedded options in deal structures (earn-outs, management equity with hurdles, convertible instruments). Pricing these correctly requires understanding the machinery, not just the output.

2. **Risk management**, Hedge funds with equity exposure need to understand how their options overlays (protective puts, covered calls, collar strategies) behave under different scenarios. The scenario analysis and delta hedging modules directly address this.

3. **Volatility as an asset class**, Vol trading desks think in terms of implied vs realized vol, skew dynamics, and term structure. The vol surface module and skew metrics (25-delta RR, butterfly) speak directly to this language.

4. **Interview signal**, Being able to explain put-call parity, why American calls on non-dividend stocks are never exercised early, how gamma and theta are two sides of the same coin, and why delta hedging is not a free lunch, these are the questions that separate candidates.

---

## Key Risks

### Model Risk
- **Log-normal assumption**: GBM does not capture fat tails, jumps, or stochastic volatility. Real-world returns exhibit all three. The engine prices vanilla options where this assumption is standard but would be inadequate for exotic path-dependent payoffs.
- **Flat yield curve**: Using a single short/long rate bucket instead of a full term structure introduces pricing error for long-dated options or steep yield curve environments.
- **Continuous dividend yield**: Discrete dividends (especially large special dividends) can materially affect option prices near ex-dates. The continuous yield approximation smooths this out.

### Data Risk
- **yfinance reliability**: Bid/ask quotes from yfinance can be stale, especially for illiquid options. The garbage filter mitigates this but cannot fix fundamentally bad data.
- **American vs European confusion**: US equity options are American-style, but the engine's parity check and MC pricer assume European exercise. The American guard on parity check prevents false violation flags, but users must understand the distinction.

### Numerical Risk
- **IV solver failure**: Brent's method can fail for extreme strikes or near-expiry options where the price surface is nearly flat. The engine returns `np.nan` rather than a misleading value.
- **FD gamma on binomial trees**: Second-order finite differences amplify discretization noise from the CRR tree. The engine documents this limitation and only provides analytical higher-order Greeks via Black-Scholes.

---

## Model Assumptions

| Assumption | Justification | Limitation |
|-----------|---------------|------------|
| Log-normal returns (GBM) | Standard for vanilla options pricing; analytically tractable | No jumps, no stochastic vol, no fat tails |
| Continuous dividend yield | Simplifies BS formula; reasonable for diversified indices | Inaccurate for single stocks near ex-dates |
| Flat yield curve | Two-bucket (short/long) approximation matches most use cases | Misprices in steep curve environments |
| No transaction costs in pricing | Industry standard, costs are a hedging concern, not a pricing concern | Delta hedge module adds costs back explicitly |
| European MC only | Longstaff-Schwartz adds complexity without changing vanilla pricing | Cannot price American options via MC in v1 |
| Empirical vol surface | Shows actual market skew without imposing parametric form | Not arbitrage-free; extrapolation unreliable |

---

## Return Scenarios (Hedging/Risk Management Use Cases)

These are not trading recommendations. They illustrate how the engine's analytics map to real portfolio decisions.

### Scenario 1: Protective Put on Concentrated Equity Position

**Context**: An investor holds a concentrated stock position and wants to hedge downside over the next 3 months.

**Engine output used**: Greeks tab (delta/gamma of the put), Scenario Analysis (P&L under spot x vol moves), Vol Surface (is near-term put skew expensive?).

| Outcome | Spot Move | Vol Move | Put P&L | Net Position |
|---------|-----------|----------|---------|-------------|
| Crash protection works | -20% | +15pts | Large positive | Hedge offsets equity loss |
| Slow grind lower | -10% | +5pts | Moderate positive | Partial offset, theta drag |
| Flat / slow up | 0% to +5% | -5pts | Negative (theta + vol crush) | Put expires worthless, cost of insurance |
| Sharp rally | +15% | -10pts | Total loss of premium | Equity gains dominate |

**Key insight**: The put is insurance, not a trade. Theta is the premium. Gamma is the payoff in a crash. The engine's scenario heatmap shows exactly where the breakeven is.

### Scenario 2: Covered Call on Dividend Stock

**Context**: Income-oriented investor sells calls against a dividend-paying stock to enhance yield.

**Engine output used**: Pricing tab (BS vs binomial for American early exercise risk), Delta Hedge tab (what happens if stock rallies through strike), Skew metrics (is call skew rich enough to sell?).

| Outcome | Spot at Expiry | Call Outcome | Total Return |
|---------|---------------|-------------|-------------|
| Below strike | $95 (stock at $100) | Expires worthless | Dividend + premium - stock loss |
| At strike | $100 | Expires worthless | Dividend + full premium |
| Above strike (capped) | $110 | Assigned, stock called away | Premium + (strike - entry) + dividend, miss upside |
| Far above (regret) | $130 | Deep ITM assignment | Capped at strike, significant opportunity cost |

**Key insight**: The covered call trades upside optionality for current income. The engine's American pricing via binomial tree correctly captures early exercise risk near ex-dividend dates.

### Scenario 3: Earnings Straddle (Vol Event)

**Context**: Trader evaluates buying a straddle before earnings to capture a large move.

**Engine output used**: Vol Surface (is pre-earnings IV elevated?), Scenario presets (earnings preset shows pre/post vol dynamics), Greeks (vega exposure sizing).

| Outcome | Realized Move | Post-Earnings IV | Straddle P&L |
|---------|-------------|-----------------|-------------|
| Large move, vol holds | +/-8% | -10pts | Positive (gamma > vol crush) |
| Large move, vol crushes | +/-8% | -20pts | Breakeven to small positive |
| Small move, vol crushes | +/-2% | -20pts | Negative (vol crush > gamma) |
| No move, vol crushes | 0% | -25pts | Maximum loss (double theta + vol crush) |

**Key insight**: Buying straddles before earnings is buying gamma against selling vega. The engine's earnings preset shows the exact breakeven: how much spot must move to overcome the vol crush. This is where most retail traders get burned, they buy expensive vol without quantifying the hurdle.

---

## Framing

This engine is a **learning and analysis tool**, not a trading system. It does not generate signals, recommend positions, or claim any edge in options markets. The value is in understanding the instrument, and that understanding is what makes a practitioner dangerous in the best sense of the word.

Every simplifying assumption is documented. Every edge case returns `np.nan` rather than a misleading number. Every chart has an interpretation callout explaining the financial meaning, not just the math. This is how a quant communicates: precisely, with stated assumptions, and with explicit uncertainty bounds.
