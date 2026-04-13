"""
P&L scenario analysis, heatmaps and presets.

Computes option value under hypothetical spot, vol, rate, and
time changes. The heatmap axes are spot change (%) x vol change
(vol points), with an optional time decay slider.

Presets are mechanical, no fake macro storytelling. They apply
specific bump combinations that correspond to common market events
(earnings, vol crush, rate moves).
"""

import numpy as np
import pandas as pd
from src import black_scholes as bs


def compute_pnl_grid(S, K, T, r, sigma, q=0.0, option_type="call",
                     entry_price=None, config=None):
    """
    Compute P&L heatmap over spot x vol grid.

    Each cell shows the option's new value (or P&L if entry_price
    is provided) under a specific combination of spot and vol changes.

    Args:
        S: Current spot.
        K: Strike.
        T: Time to expiry in years.
        r: Risk-free rate.
        sigma: Current implied vol.
        q: Dividend yield.
        option_type: 'call' or 'put'.
        entry_price: If provided, cells show P&L = new_price - entry.
        config: Configuration dict with 'scenarios' section.

    Returns:
        Dict with 'spot_changes', 'vol_changes', 'values' (2D array),
        'is_pnl' (bool).
    """
    if config is None:
        config = {}

    sc_cfg = config.get("scenarios", {})
    spot_changes = sc_cfg.get("spot_range_pct",
                              [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20])
    vol_changes = sc_cfg.get("vol_range_pts",
                             [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15])

    values = np.full((len(vol_changes), len(spot_changes)), np.nan)

    for i, dv in enumerate(vol_changes):
        for j, ds in enumerate(spot_changes):
            new_S = S * (1.0 + ds)
            new_sigma = max(sigma + dv, 0.001)

            val = bs.price(new_S, K, T, r, new_sigma, q, option_type)

            if entry_price is not None and not np.isnan(val):
                val = val - entry_price

            values[i, j] = val

    return {
        "spot_changes": spot_changes,
        "vol_changes": vol_changes,
        "values": values,
        "is_pnl": entry_price is not None,
    }


def compute_time_decay(S, K, T, r, sigma, q=0.0, option_type="call",
                       entry_price=None, config=None):
    """
    Compute option value at different time horizons.

    Shows how the P&L surface shifts as time passes, critical
    for understanding theta decay and its interaction with spot
    and vol moves.

    Returns:
        Dict keyed by days_forward, each containing a pnl_grid result.
    """
    if config is None:
        config = {}

    sc_cfg = config.get("scenarios", {})
    days_fwd = sc_cfg.get("time_forward_days", [0, 7, 14, 30])

    results = {}
    for d in days_fwd:
        T_new = max(T - d / 365.0, 1e-6)
        grid = compute_pnl_grid(S, K, T_new, r, sigma, q, option_type,
                                entry_price, config)
        results[d] = grid

    return results


def apply_preset(S, K, T, r, sigma, q=0.0, option_type="call",
                 entry_price=None, preset_name="earnings",
                 config=None):
    """
    Apply a scenario preset and compute resulting values.

    Presets define specific spot/vol/rate bumps that correspond
    to common market events. Values are mechanical, the names
    are labels, not predictions.

    Args:
        preset_name: One of 'earnings', 'vol_crush', 'rate_hike'.

    Returns:
        List of dicts with 'scenario', 'spot_change', 'vol_change',
        'rate_change', 'new_price', 'pnl'.
    """
    if config is None:
        config = {}

    presets = config.get("scenarios", {}).get("presets", {})
    preset = presets.get(preset_name, {})

    if not preset:
        return []

    if entry_price is None:
        entry_price = bs.price(S, K, T, r, sigma, q, option_type)

    results = []

    if preset_name == "earnings":
        spots = preset.get("spot_changes", [0.0])
        vol_pre = preset.get("vol_change_pre", 0.10)
        vol_post = preset.get("vol_change_post", -0.15)
        dr = preset.get("rate_change", 0.0)

        for ds in spots:
            # Pre-earnings: vol up
            new_S = S * (1.0 + ds)
            new_sigma = max(sigma + vol_pre, 0.001)
            new_r = r + dr
            p = bs.price(new_S, K, T, new_r, new_sigma, q, option_type)
            results.append({
                "scenario": f"Pre-earnings: spot {ds:+.0%}",
                "spot_change": ds, "vol_change": vol_pre,
                "rate_change": dr, "new_price": p,
                "pnl": p - entry_price if not np.isnan(p) else np.nan,
            })
            # Post-earnings: vol crush
            new_sigma_post = max(sigma + vol_post, 0.001)
            p2 = bs.price(new_S, K, T, new_r, new_sigma_post, q, option_type)
            results.append({
                "scenario": f"Post-earnings: spot {ds:+.0%}",
                "spot_change": ds, "vol_change": vol_post,
                "rate_change": dr, "new_price": p2,
                "pnl": p2 - entry_price if not np.isnan(p2) else np.nan,
            })

    elif preset_name == "vol_crush":
        spots = preset.get("spot_changes", [0.0])
        vol_chgs = preset.get("vol_changes", [-0.10, -0.20])
        dr = preset.get("rate_change", 0.0)

        for dv in vol_chgs:
            for ds in spots:
                new_S = S * (1.0 + ds)
                new_sigma = max(sigma + dv, 0.001)
                p = bs.price(new_S, K, T, r + dr, new_sigma, q, option_type)
                results.append({
                    "scenario": f"Vol crush {dv:+.0%}",
                    "spot_change": ds, "vol_change": dv,
                    "rate_change": dr, "new_price": p,
                    "pnl": p - entry_price if not np.isnan(p) else np.nan,
                })

    elif preset_name == "rate_hike":
        spots = preset.get("spot_changes", [0.0])
        dv = preset.get("vol_change", 0.0)
        rate_chgs = preset.get("rate_changes", [0.0025, 0.005])

        for dr in rate_chgs:
            for ds in spots:
                new_S = S * (1.0 + ds)
                new_sigma = max(sigma + dv, 0.001)
                p = bs.price(new_S, K, T, r + dr, new_sigma, q, option_type)
                results.append({
                    "scenario": f"Rate +{dr*10000:.0f}bps",
                    "spot_change": ds, "vol_change": dv,
                    "rate_change": dr, "new_price": p,
                    "pnl": p - entry_price if not np.isnan(p) else np.nan,
                })

    return results
