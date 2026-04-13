"""
Empirical implied volatility surface construction.

Builds an IV surface from filtered options chain data using
interpolation/smoothing. This is NOT an arbitrage-free calibrated
surface (no SVI, no arbitrage constraints in v1), it is an
empirical visualization tool.

Axes: log-moneyness ln(K/S) x time to expiry (years).
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata, RBFInterpolator


def build_surface(chain_df: pd.DataFrame, config: dict) -> dict:
    """
    Build an implied volatility surface from chain data.

    Requires the chain to have 'log_moneyness', 'T', and 'iv'
    columns (IV must be pre-extracted via implied_vol module).

    The surface is an empirical interpolation, it will show the
    actual market skew and term structure but does not enforce
    no-arbitrage conditions (calendar spread or butterfly).

    Args:
        chain_df: Filtered chain with 'iv' column (NaN rows dropped).
        config: Configuration dict with 'vol_surface' section.

    Returns:
        Dict with 'grid_moneyness', 'grid_T', 'grid_iv' (2D arrays),
        'points' (raw data), 'method', 'n_points'.
        Returns None if insufficient data.
    """
    vs_cfg = config.get("vol_surface", {})
    method = vs_cfg.get("interpolation_method", "cubic")
    min_pts = vs_cfg.get("min_points_for_surface", 20)

    df = chain_df.dropna(subset=["iv", "log_moneyness", "T"]).copy()
    # Post-IV bounds: drop solver artifacts outside [1%, 300%]
    df = df[(df["iv"] >= 0.01) & (df["iv"] <= 3.0)]

    if len(df) < min_pts:
        return None

    x = df["log_moneyness"].values
    y = df["T"].values
    z = df["iv"].values

    # Build regular grid for interpolation
    m_range = np.linspace(x.min(), x.max(), 50)
    t_range = np.linspace(y.min(), y.max(), 30)
    grid_m, grid_t = np.meshgrid(m_range, t_range)

    try:
        grid_iv = griddata(
            points=np.column_stack([x, y]),
            values=z,
            xi=(grid_m, grid_t),
            method=method,
        )
    except Exception:
        # Fall back to linear if cubic fails
        grid_iv = griddata(
            points=np.column_stack([x, y]),
            values=z,
            xi=(grid_m, grid_t),
            method="linear",
        )
        method = "linear"

    return {
        "grid_moneyness": grid_m,
        "grid_T": grid_t,
        "grid_iv": grid_iv,
        "points": {"moneyness": x, "T": y, "iv": z},
        "method": method,
        "n_points": len(df),
    }


def smile_per_expiry(chain_df: pd.DataFrame) -> dict:
    """
    Extract volatility smile for each expiry.

    Returns a dict keyed by expiry date, each containing arrays
    of log_moneyness and implied vol. Used for 2D smile overlay
    plots, the classic way to visualize skew.
    """
    df = chain_df.dropna(subset=["iv", "log_moneyness"]).copy()
    df = df[(df["iv"] >= 0.01) & (df["iv"] <= 3.0)]

    smiles = {}
    for exp, group in df.groupby("expiry"):
        group = group.sort_values("log_moneyness")
        smiles[exp] = {
            "log_moneyness": group["log_moneyness"].values,
            "iv": group["iv"].values,
            "strikes": group["strike"].values,
            "dte": int(group["dte"].iloc[0]) if "dte" in group.columns else None,
        }
    return smiles


def term_structure(chain_df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """
    ATM implied vol term structure.

    For each expiry, finds the strike closest to ATM and returns
    its IV. This shows how the market prices vol across time —
    typically upward-sloping in calm markets (more uncertainty
    over longer horizons) and inverted around events.
    """
    df = chain_df.dropna(subset=["iv"]).copy()
    df = df[(df["iv"] >= 0.01) & (df["iv"] <= 3.0)]

    if df.empty:
        return pd.DataFrame(columns=["expiry", "T", "dte", "atm_iv"])

    rows = []
    for exp, group in df.groupby("expiry"):
        # Find strike closest to spot (ATM)
        group = group.copy()
        group["dist"] = np.abs(group["strike"] - spot)
        atm = group.loc[group["dist"].idxmin()]
        rows.append({
            "expiry": exp,
            "T": atm["T"],
            "dte": atm.get("dte", np.nan),
            "atm_iv": atm["iv"],
        })

    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def interpolate_iv(chain_df: pd.DataFrame, moneyness: float,
                   T: float, config: dict) -> float:
    """
    Interpolate IV at a specific (moneyness, T) point on the surface.

    Uses RBF interpolation for point queries. Returns np.nan if
    the point is outside the data range.
    """
    df = chain_df.dropna(subset=["iv", "log_moneyness", "T"]).copy()
    df = df[df["iv"] > 0]

    if len(df) < 5:
        return np.nan

    x = df[["log_moneyness", "T"]].values
    z = df["iv"].values

    # Reject queries outside the convex hull of observed data
    # to prevent silent extrapolation into unreliable territory
    m_min, m_max = x[:, 0].min(), x[:, 0].max()
    t_min, t_max = x[:, 1].min(), x[:, 1].max()
    if moneyness < m_min or moneyness > m_max or T < t_min or T > t_max:
        return np.nan

    try:
        rbf = RBFInterpolator(x, z, kernel="thin_plate_spline", smoothing=0.01)
        result = rbf(np.array([[moneyness, T]]))[0]
        return float(result) if result > 0 else np.nan
    except Exception:
        return np.nan
