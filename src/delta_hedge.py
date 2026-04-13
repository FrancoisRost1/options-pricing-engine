"""
Delta hedging simulation with P&L decomposition.

Simulates a self-financing replication portfolio to illustrate
discrete hedging dynamics. This is NOT a profitable strategy —
it demonstrates how delta hedging works in practice, including
the gap between continuous (theoretical) and discrete (real) hedging.

P&L decomposition:
  Total = Gamma/Theta effect + Hedging error + Transaction costs
"""

import numpy as np
import pandas as pd
from src import black_scholes as bs


def _parse_frequency(freq, n_days):
    """
    Convert frequency string/int to rebalance indices.

    'daily' → every day, 'weekly' → every 5th day,
    integer N → every N days.
    """
    if freq == "daily":
        step = 1
    elif freq == "weekly":
        step = 5
    elif isinstance(freq, int):
        step = max(freq, 1)
    else:
        step = 1

    return list(range(0, n_days, step))


def simulate_hedge(S_path, K, T, r, sigma, q=0.0, option_type="call",
                   position="long_call", config=None):
    """
    Run a delta hedging simulation along a price path.

    The hedger holds an option position and dynamically rebalances
    a stock hedge to maintain delta neutrality. The cash account
    finances stock purchases/sales and accrues interest at r.

    P&L decomposition isolates three components:
      1. Gamma/Theta: the theoretical P&L from holding gamma exposure
         offset by theta decay, the core risk/reward of options
      2. Hedging error: the residual from discrete vs continuous
         rebalancing, smaller with more frequent hedging
      3. Transaction costs: the drag from bid-ask spread and slippage
         on each rebalance

    Args:
        S_path: Array of daily spot prices (length = trading days + 1).
        K: Strike price.
        T: Time to expiry in years at start.
        r: Risk-free rate.
        sigma: Volatility used for hedge ratios (typically realized or implied).
        q: Dividend yield.
        option_type: 'call' or 'put'.
        position: 'long_call', 'long_put', 'short_call', 'short_put'.
        config: Configuration dict.

    Returns:
        Dict with 'daily' (DataFrame of daily P&L components),
        'summary' (dict of aggregate stats), 'config_used' (dict).
    """
    if config is None:
        config = {}

    dh_cfg = config.get("delta_hedge", {})
    freq = dh_cfg.get("hedge_frequency", "daily")
    tc_bps = dh_cfg.get("tc_bps", 5.0)
    slip_bps = dh_cfg.get("slippage_bps", 2.0)
    multiplier = dh_cfg.get("contract_multiplier", 100)
    n_contracts = dh_cfg.get("num_contracts", 1)

    total_cost_bps = (tc_bps + slip_bps) / 10000.0
    n_days = len(S_path) - 1
    dt = T / n_days if n_days > 0 else 1 / 252

    # Position sign: +1 for long, -1 for short
    is_long = "long" in position
    pos_sign = 1.0 if is_long else -1.0
    opt_type = "call" if "call" in position else "put"

    rebalance_idx = set(_parse_frequency(freq, n_days))

    # Initialize tracking arrays
    delta_held = 0.0
    cash = 0.0
    daily_rows = []

    for i in range(n_days):
        S_now = S_path[i]
        S_next = S_path[i + 1]
        T_now = max(T - i * dt, 1e-6)

        # Current Greeks
        d = bs.delta(S_now, K, T_now, r, sigma, q, opt_type)
        g = bs.gamma(S_now, K, T_now, r, sigma, q, opt_type)
        th = bs.theta(S_now, K, T_now, r, sigma, q, opt_type)

        if np.isnan(d):
            d = 0.0
        if np.isnan(g):
            g = 0.0
        if np.isnan(th):
            th = 0.0

        # Target delta hedge (opposite sign to option position)
        target_shares = pos_sign * d * multiplier * n_contracts

        # Rebalance if this is a rebalance day
        tc_cost = 0.0
        if i in rebalance_idx:
            shares_traded = abs(target_shares - delta_held)
            tc_cost = shares_traded * S_now * total_cost_bps
            cash -= (target_shares - delta_held) * S_now  # Buy/sell stock
            cash -= tc_cost
            delta_held = target_shares

        # Daily interest on cash (financing carry)
        cash_before = cash
        cash *= np.exp(r * dt)
        financing_pnl = cash - cash_before

        # Price changes
        dS = S_next - S_now

        # Option P&L (mark-to-model)
        opt_val_now = bs.price(S_now, K, T_now, r, sigma, q, opt_type)
        T_next = max(T - (i + 1) * dt, 1e-6)
        opt_val_next = bs.price(S_next, K, T_next, r, sigma, q, opt_type)

        if np.isnan(opt_val_now):
            opt_val_now = 0.0
        if np.isnan(opt_val_next):
            opt_val_next = 0.0

        opt_pnl = pos_sign * (opt_val_next - opt_val_now) * multiplier * n_contracts

        # Stock hedge P&L
        stock_pnl = delta_held * dS

        # Gamma/theta decomposition (per period, scaled by position)
        gamma_pnl = pos_sign * 0.5 * g * dS ** 2 * multiplier * n_contracts
        theta_pnl = pos_sign * th * 365.0 * dt * multiplier * n_contracts

        # Total hedged P&L includes option + stock + financing - costs
        total_pnl = opt_pnl + stock_pnl + financing_pnl - tc_cost
        hedge_error = total_pnl - gamma_pnl - theta_pnl

        daily_rows.append({
            "day": i,
            "spot": S_now,
            "T_remaining": T_now,
            "delta": d,
            "gamma": g,
            "theta": th,
            "shares_held": delta_held,
            "option_pnl": opt_pnl,
            "stock_pnl": stock_pnl,
            "financing_pnl": financing_pnl,
            "gamma_pnl": gamma_pnl,
            "theta_pnl": theta_pnl,
            "hedge_error": hedge_error,
            "tc_cost": tc_cost,
            "total_pnl": total_pnl,
        })

    daily_df = pd.DataFrame(daily_rows)

    # Cumulative P&L
    daily_df["cum_total_pnl"] = daily_df["total_pnl"].cumsum()
    daily_df["cum_gamma_pnl"] = daily_df["gamma_pnl"].cumsum()
    daily_df["cum_theta_pnl"] = daily_df["theta_pnl"].cumsum()
    daily_df["cum_hedge_error"] = daily_df["hedge_error"].cumsum()
    daily_df["cum_tc_cost"] = daily_df["tc_cost"].cumsum()

    summary = {
        "total_pnl": float(daily_df["total_pnl"].sum()),
        "gamma_theta_pnl": float((daily_df["gamma_pnl"] + daily_df["theta_pnl"]).sum()),
        "financing_pnl": float(daily_df["financing_pnl"].sum()),
        "hedge_error_total": float(daily_df["hedge_error"].sum()),
        "hedge_error_std": float(daily_df["hedge_error"].std()),
        "tc_drag": float(daily_df["tc_cost"].sum()),
        "n_rebalances": len(rebalance_idx),
        "avg_turnover_per_rebalance": float(
            daily_df.loc[daily_df["day"].isin(rebalance_idx), "tc_cost"].mean()
        ) if len(rebalance_idx) > 0 else 0.0,
    }

    config_used = {
        "frequency": freq,
        "tc_bps": tc_bps,
        "slippage_bps": slip_bps,
        "position": position,
        "n_contracts": n_contracts,
    }

    return {
        "daily": daily_df,
        "summary": summary,
        "config_used": config_used,
    }


def generate_gbm_paths(S0, T, r, sigma, q=0.0, n_paths=100,
                        n_steps=None, seed=42):
    """
    Generate GBM price paths for simulated hedging comparison.

    When historical data is unavailable, these paths provide a
    controlled environment to study hedging dynamics under known
    volatility, useful for isolating hedging error from vol
    misspecification.
    """
    if n_steps is None:
        n_steps = int(T * 252)
    n_steps = max(n_steps, 1)

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    Z = rng.standard_normal((n_paths, n_steps))
    log_returns = drift + diffusion * Z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1
    )
    return S0 * np.exp(log_paths)
