"""
Options chain garbage filter.

Applies configurable quality filters to remove illiquid, mispriced,
or extreme contracts from analysis. Filtered chains are used for
model-vs-market comparison and vol surface construction; the raw
chain remains available in the Chain Explorer tab.

All thresholds are driven by config.yaml, never hardcoded.
"""

import numpy as np
import pandas as pd


def apply_filters(chain_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Filter an options chain by liquidity, moneyness, and spread.

    Removes contracts that would distort IV extraction or model
    comparison: wide spreads (market-maker uncertainty), low volume
    (stale quotes), extreme strikes (unreliable extrapolation).

    Args:
        chain_df: Raw options chain DataFrame.
        config: Configuration dict with 'chain_filter' section.

    Returns:
        Filtered DataFrame (copy, original is not modified).
    """
    if chain_df.empty:
        return chain_df.copy()

    cfg = config.get("chain_filter", {})
    max_spread = cfg.get("max_spread_pct", 0.50)
    min_vol = cfg.get("min_volume", 10)
    min_oi = cfg.get("min_open_interest", 10)
    m_bounds = cfg.get("moneyness_bounds", {"lower": -0.50, "upper": 0.50})
    min_dte = cfg.get("min_dte", 1)

    df = chain_df.copy()

    # Filter: valid mid price (not excluded)
    if "price_source" in df.columns:
        df = df[df["price_source"] != "excluded"]

    # Filter: mid > 0
    df = df[df["mid"] > 0]

    # Filter: bid-ask spread percentage
    if "bid" in df.columns and "ask" in df.columns:
        spread_pct = (df["ask"] - df["bid"]) / df["mid"]
        df = df[spread_pct <= max_spread]

    # Filter: minimum volume
    if "volume" in df.columns:
        df = df[df["volume"] >= min_vol]

    # Filter: minimum open interest
    if "openInterest" in df.columns:
        df = df[df["openInterest"] >= min_oi]

    # Filter: moneyness bounds (log-moneyness)
    if "log_moneyness" in df.columns:
        lower = m_bounds.get("lower", -0.50) if isinstance(m_bounds, dict) else -0.50
        upper = m_bounds.get("upper", 0.50) if isinstance(m_bounds, dict) else 0.50
        df = df[(df["log_moneyness"] >= lower) & (df["log_moneyness"] <= upper)]

    # Filter: minimum DTE
    if "dte" in df.columns:
        df = df[df["dte"] >= min_dte]

    return df.reset_index(drop=True)


def apply_dashboard_defaults(chain_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply stricter default filters for initial dashboard view.

    Narrows to near-ATM, more liquid options and limits expiries
    shown. Users can relax these in the Chain Explorer tab.

    Args:
        chain_df: Already-filtered options chain.
        config: Configuration dict with 'dashboard_defaults' section.

    Returns:
        Further-filtered DataFrame for default display.
    """
    if chain_df.empty:
        return chain_df.copy()

    defaults = config.get("dashboard_defaults", {})
    max_exp = defaults.get("max_expiries_shown", 4)
    m_bounds = defaults.get("moneyness_default_bounds", {"lower": -0.15, "upper": 0.15})
    min_vol = defaults.get("min_volume_default", 50)

    df = chain_df.copy()

    # Limit to nearest N expiries
    if "expiry" in df.columns:
        unique_exp = sorted(df["expiry"].unique())
        if len(unique_exp) > max_exp:
            df = df[df["expiry"].isin(unique_exp[:max_exp])]

    # Tighter moneyness bounds
    if "log_moneyness" in df.columns:
        lower = m_bounds.get("lower", -0.15) if isinstance(m_bounds, dict) else -0.15
        upper = m_bounds.get("upper", 0.15) if isinstance(m_bounds, dict) else 0.15
        df = df[(df["log_moneyness"] >= lower) & (df["log_moneyness"] <= upper)]

    # Stricter volume filter
    if "volume" in df.columns:
        df = df[df["volume"] >= min_vol]

    return df.reset_index(drop=True)


def filter_summary(raw_df: pd.DataFrame, filtered_df: pd.DataFrame) -> dict:
    """
    Summary statistics comparing raw vs filtered chain.

    Useful for displaying how many contracts were removed
    and why, helps users understand data quality.
    """
    return {
        "raw_contracts": len(raw_df),
        "filtered_contracts": len(filtered_df),
        "removed": len(raw_df) - len(filtered_df),
        "removal_pct": (1.0 - len(filtered_df) / max(len(raw_df), 1)) * 100,
        "unique_expiries_raw": raw_df["expiry"].nunique() if "expiry" in raw_df.columns else 0,
        "unique_expiries_filtered": filtered_df["expiry"].nunique() if "expiry" in filtered_df.columns else 0,
    }
