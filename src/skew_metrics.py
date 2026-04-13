"""
Volatility skew metrics, 25-delta risk reversal and butterfly.

These metrics summarize the shape of the volatility smile:
  - Risk Reversal: measures skew direction (equity skew = negative RR)
  - Butterfly: measures smile curvature/convexity

The 25-delta convention is borrowed from FX vol markets. For equities,
the 25-delta strikes are obtained by interpolation on the smile:
find K where BS_delta(K, sigma(K)) = target_delta.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from src import black_scholes as bs
from src import vol_surface


def _find_delta_strike(target_delta, S, T, r, q, chain_df,
                       option_type="call"):
    """
    Find the strike where BS delta equals target_delta.

    Iteratively searches for K such that:
      BS_delta(S, K, T, r, sigma(K), q) = target_delta

    where sigma(K) is interpolated from the smile. This couples
    the strike search with the vol smile, a key subtlety.

    Args:
        target_delta: Desired delta (e.g., 0.25 for 25-delta call).
        S: Spot price.
        T: Time to expiry.
        r: Risk-free rate.
        q: Dividend yield.
        chain_df: Chain with IV data for interpolation.
        option_type: 'call' or 'put'.

    Returns:
        (strike, iv_at_strike) tuple, or (np.nan, np.nan) on failure.
    """
    # Get smile data for this expiry
    df = chain_df.dropna(subset=["iv"]).copy()
    df = df[df["iv"] > 0]

    if len(df) < 3:
        return np.nan, np.nan

    K_min = df["strike"].min()
    K_max = df["strike"].max()

    def _get_iv_at_strike(K):
        """Interpolate IV at strike K, using only the relevant option type.

        Mixing call and put IVs distorts the smile because they can
        differ at the same strike. We filter to the current option_type
        first, falling back to the full smile only if too few points.
        """
        log_m = np.log(K / S)
        if "option_type" in df.columns:
            typed = df[df["option_type"] == option_type].sort_values("log_moneyness")
            if len(typed) >= 3:
                return np.interp(log_m, typed["log_moneyness"], typed["iv"])
        # Fallback: use all available IVs (deduplicated by strike)
        deduped = df.drop_duplicates(subset=["log_moneyness"]).sort_values("log_moneyness")
        return np.interp(log_m, deduped["log_moneyness"], deduped["iv"])

    def objective(K):
        """Find K where BS delta = target_delta."""
        iv = _get_iv_at_strike(K)
        if iv <= 0 or np.isnan(iv):
            return 1.0  # Push solver away
        d = bs.delta(S, K, T, r, iv, q, option_type)
        if np.isnan(d):
            return 1.0
        return d - target_delta

    try:
        K_star = brentq(objective, K_min, K_max, xtol=0.01)
        iv_star = _get_iv_at_strike(K_star)
        return float(K_star), float(iv_star)
    except (ValueError, RuntimeError):
        return np.nan, np.nan


def compute_skew_metrics(chain_df: pd.DataFrame, S: float, r: float,
                         q: float, config: dict) -> pd.DataFrame:
    """
    Compute 25-delta risk reversal and butterfly per expiry.

    Risk Reversal = sigma(25D call) - sigma(25D put)
      Negative RR = steeper put skew (typical for equities —
      market prices crash protection more expensively).

    Butterfly = 0.5 * [sigma(25D call) + sigma(25D put)] - sigma(ATM)
      Positive BF = smile has curvature (wings trade richer than ATM).

    Args:
        chain_df: Filtered chain with 'iv' column.
        S: Spot price.
        r: Risk-free rate.
        q: Dividend yield.
        config: Configuration dict.

    Returns:
        DataFrame with columns: expiry, dte, T, atm_iv, call_25d_iv,
        put_25d_iv, call_25d_strike, put_25d_strike, risk_reversal,
        butterfly.
    """
    skew_cfg = config.get("skew_metrics", {})
    delta_level = skew_cfg.get("delta_level", 0.25)

    df = chain_df.dropna(subset=["iv"]).copy()
    df = df[df["iv"] > 0]

    if df.empty:
        return pd.DataFrame()

    rows = []
    for exp, group in df.groupby("expiry"):
        T = group["T"].iloc[0]
        dte = group["dte"].iloc[0] if "dte" in group.columns else np.nan

        # ATM IV: closest strike to spot
        group_sorted = group.copy()
        group_sorted["dist"] = np.abs(group_sorted["strike"] - S)
        atm_iv = group_sorted.loc[group_sorted["dist"].idxmin(), "iv"]

        # 25-delta call strike and IV
        K_call, iv_call = _find_delta_strike(
            delta_level, S, T, r, q, group, "call"
        )

        # 25-delta put strike and IV (delta is negative for puts)
        K_put, iv_put = _find_delta_strike(
            -delta_level, S, T, r, q, group, "put"
        )

        rr = iv_call - iv_put if not (np.isnan(iv_call) or np.isnan(iv_put)) else np.nan
        bf = (0.5 * (iv_call + iv_put) - atm_iv
              if not (np.isnan(iv_call) or np.isnan(iv_put) or np.isnan(atm_iv))
              else np.nan)

        rows.append({
            "expiry": exp,
            "dte": dte,
            "T": T,
            "atm_iv": atm_iv,
            "call_25d_iv": iv_call,
            "put_25d_iv": iv_put,
            "call_25d_strike": K_call,
            "put_25d_strike": K_put,
            "risk_reversal": rr,
            "butterfly": bf,
        })

    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)
