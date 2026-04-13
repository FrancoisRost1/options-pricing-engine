"""
Implied volatility extraction via Brent's root-finding method.

Finds the volatility sigma such that BS_price(sigma) = market_price.
Uses scipy.optimize.brentq with configurable bounds and tolerance.

Pre-solve validation ensures the market price is within arbitrage
bounds before attempting to solve, this prevents wasting compute
on contracts where no valid IV exists (e.g., price below intrinsic).

Failure policy: return np.nan. Never fake-fill.
"""

import numpy as np
from scipy.optimize import brentq
from src import black_scholes as bs


def _intrinsic_value(S, K, r, q, T, option_type):
    """
    Compute the lower bound (intrinsic value) for an option price.

    For calls: max(S*e^(-qT) - K*e^(-rT), 0)
    For puts:  max(K*e^(-rT) - S*e^(-qT), 0)
    """
    fwd = S * np.exp(-q * T)
    pv_k = K * np.exp(-r * T)
    if option_type == "call":
        return max(fwd - pv_k, 0.0)
    else:
        return max(pv_k - fwd, 0.0)


def _upper_bound(S, K, r, q, T, option_type):
    """
    Compute the upper bound for an option price.

    Calls cannot exceed S*e^(-qT) (PV of receiving stock at expiry).
    Puts cannot exceed K*e^(-rT) (PV of strike).
    With dividends, the forward is S*e^(-qT), not S.
    """
    if option_type == "call":
        return S * np.exp(-q * T)
    else:
        return K * np.exp(-r * T)


def extract(market_price, S, K, T, r, q=0.0, option_type="call",
            config=None):
    """
    Extract implied volatility from a market price.

    Uses Brent's method to find sigma where BS(sigma) = market_price.
    Brent's method is chosen because it is guaranteed to converge
    when the function changes sign on the bracket, and option price
    is monotonically increasing in sigma, so this is always satisfied
    when the market price is within arbitrage bounds.

    Args:
        market_price: Observed market price (mid).
        S: Spot price.
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free rate (continuous).
        q: Continuous dividend yield.
        option_type: 'call' or 'put'.
        config: Configuration dict with solver parameters.

    Returns:
        Implied volatility (float), or np.nan if solver fails.
    """
    if config is None:
        config = {}

    iv_cfg = config.get("implied_vol", {})
    vol_lo = iv_cfg.get("vol_lower_bound", 0.001)
    vol_hi = iv_cfg.get("vol_upper_bound", 10.0)
    tol = iv_cfg.get("tolerance", 1e-8)
    max_iter = iv_cfg.get("max_iterations", 100)

    # Pre-solve validation
    if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan

    intrinsic = _intrinsic_value(S, K, r, q, T, option_type)
    upper = _upper_bound(S, K, r, q, T, option_type)

    if market_price < intrinsic:
        return np.nan
    if market_price > upper:
        return np.nan

    # Objective: BS_price(sigma) - market_price = 0
    def objective(sigma):
        return bs.price(S, K, T, r, sigma, q, option_type) - market_price

    try:
        iv = brentq(objective, vol_lo, vol_hi, xtol=tol, maxiter=max_iter)
        return float(iv)
    except (ValueError, RuntimeError):
        return np.nan


def extract_chain(chain_df, S, r, q=0.0, config=None):
    """
    Extract implied volatility for an entire options chain.

    Adds an 'iv' column to the DataFrame. Contracts where IV
    extraction fails get np.nan, never fake-filled.

    Args:
        chain_df: DataFrame with columns 'strike', 'mid', 'T',
                  'option_type'. 'T' is time to expiry in years.
        S: Spot price.
        r: Risk-free rate.
        q: Continuous dividend yield.
        config: Configuration dict.

    Returns:
        DataFrame with added 'iv' column.
    """
    df = chain_df.copy()
    ivs = []
    for _, row in df.iterrows():
        iv = extract(
            market_price=row["mid"],
            S=S,
            K=row["strike"],
            T=row["T"],
            r=r,
            q=q,
            option_type=row["option_type"],
            config=config,
        )
        ivs.append(iv)

    df["iv"] = ivs
    return df
