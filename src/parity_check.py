"""
Put-call parity validation for European options.

Verifies the fundamental no-arbitrage relationship:
  C - P = S * e^(-qT) - K * e^(-rT)

This must hold for European options. Violations signal either
data quality issues (stale quotes, wrong mid prices) or
genuine mispricings. American options are excluded, early
exercise premium breaks parity.
"""

import numpy as np
import pandas as pd


def check_parity(chain_df: pd.DataFrame, S: float, r: float,
                 q: float, config: dict,
                 exercise: str = "european") -> pd.DataFrame:
    """
    Check put-call parity for all matched call-put pairs.

    EUROPEAN OPTIONS ONLY. Put-call parity does not hold for
    American options due to early exercise premium. If exercise
    is 'american', returns an empty DataFrame with a warning flag
    rather than producing misleading results.

    Args:
        chain_df: Filtered chain with mid prices.
        S: Spot price.
        r: Risk-free rate.
        q: Dividend yield.
        config: Configuration dict with 'parity_check' section.

    Returns:
        DataFrame with columns: strike, expiry, T, call_mid, put_mid,
        parity_theoretical, parity_actual, deviation_dollar,
        deviation_pct, violation (bool).
    """
    if exercise == "american":
        import warnings
        warnings.warn(
            "Put-call parity validation skipped: not valid for American "
            "options (early exercise premium breaks parity).",
            stacklevel=2,
        )
        return pd.DataFrame()

    pc_cfg = config.get("parity_check", {})
    threshold = pc_cfg.get("max_parity_deviation_pct", 0.02)

    df = chain_df.dropna(subset=["mid"]).copy()

    calls = df[df["option_type"] == "call"][["strike", "expiry", "T", "mid"]].copy()
    calls = calls.rename(columns={"mid": "call_mid"})

    puts = df[df["option_type"] == "put"][["strike", "expiry", "mid"]].copy()
    puts = puts.rename(columns={"mid": "put_mid"})

    # Match by strike and expiry
    merged = calls.merge(puts, on=["strike", "expiry"], how="inner")

    if merged.empty:
        return pd.DataFrame()

    # Theoretical parity: C - P = S*e^(-qT) - K*e^(-rT)
    merged["parity_theoretical"] = (
        S * np.exp(-q * merged["T"]) - merged["strike"] * np.exp(-r * merged["T"])
    )
    merged["parity_actual"] = merged["call_mid"] - merged["put_mid"]
    merged["deviation_dollar"] = merged["parity_actual"] - merged["parity_theoretical"]

    # Deviation as % of average mid price
    avg_mid = (merged["call_mid"] + merged["put_mid"]) / 2.0
    merged["deviation_pct"] = np.where(
        avg_mid > 0,
        np.abs(merged["deviation_dollar"]) / avg_mid,
        np.nan,
    )

    merged["violation"] = merged["deviation_pct"] > threshold

    return merged.sort_values(["expiry", "strike"]).reset_index(drop=True)


def parity_summary(parity_df: pd.DataFrame) -> dict:
    """
    Summary statistics for put-call parity analysis.

    Returns:
        Dict with total pairs, violations count/pct, mean/max deviation.
    """
    if parity_df.empty:
        return {
            "total_pairs": 0, "violations": 0, "violation_pct": 0.0,
            "mean_deviation_dollar": np.nan, "max_deviation_dollar": np.nan,
            "mean_deviation_pct": np.nan, "max_deviation_pct": np.nan,
        }

    return {
        "total_pairs": len(parity_df),
        "violations": int(parity_df["violation"].sum()),
        "violation_pct": float(parity_df["violation"].mean() * 100),
        "mean_deviation_dollar": float(parity_df["deviation_dollar"].abs().mean()),
        "max_deviation_dollar": float(parity_df["deviation_dollar"].abs().max()),
        "mean_deviation_pct": float(parity_df["deviation_pct"].mean() * 100),
        "max_deviation_pct": float(parity_df["deviation_pct"].max() * 100),
    }
