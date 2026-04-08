"""
Tests for volatility surface construction and interpolation.

Validates surface building, smile extraction, term structure,
and handling of insufficient data.
"""

import numpy as np
import pandas as pd
import pytest
from src import vol_surface


def _make_surface_chain(n_per_expiry=15, n_expiries=4):
    """Helper: synthetic chain with valid IV data for surface tests."""
    rows = []
    spot = 100.0
    for exp_idx in range(n_expiries):
        dte = 30 * (exp_idx + 1)
        T = dte / 365.0
        expiry = pd.Timestamp(f"2026-{exp_idx + 2:02d}-01")
        for i in range(n_per_expiry):
            K = spot * (0.85 + 0.02 * i)
            log_m = np.log(K / spot)
            # Simulated smile: higher IV at wings
            iv_val = 0.25 + 0.05 * log_m ** 2
            rows.append({
                "strike": K, "log_moneyness": log_m, "T": T,
                "dte": dte, "iv": iv_val, "expiry": expiry,
                "option_type": "call",
            })
    return pd.DataFrame(rows)


class TestBuildSurface:
    """Surface construction tests."""

    def test_returns_dict_with_grids(self):
        df = _make_surface_chain()
        result = vol_surface.build_surface(df, {"vol_surface": {"min_points_for_surface": 5}})
        assert result is not None
        assert "grid_moneyness" in result
        assert "grid_T" in result
        assert "grid_iv" in result
        assert result["n_points"] > 0

    def test_insufficient_points_returns_none(self):
        df = _make_surface_chain(n_per_expiry=2, n_expiries=1)
        result = vol_surface.build_surface(df, {"vol_surface": {"min_points_for_surface": 50}})
        assert result is None

    def test_drops_nan_iv(self):
        df = _make_surface_chain()
        df.loc[0:5, "iv"] = np.nan
        result = vol_surface.build_surface(df, {"vol_surface": {"min_points_for_surface": 5}})
        assert result is not None
        assert result["n_points"] < len(df)

    def test_drops_zero_iv(self):
        df = _make_surface_chain()
        df.loc[0:5, "iv"] = 0.0
        result = vol_surface.build_surface(df, {"vol_surface": {"min_points_for_surface": 5}})
        assert result["n_points"] < len(df)

    def test_grid_shape(self):
        df = _make_surface_chain()
        result = vol_surface.build_surface(df, {"vol_surface": {"min_points_for_surface": 5}})
        assert result["grid_moneyness"].shape == (30, 50)
        assert result["grid_T"].shape == (30, 50)


class TestSmilePerExpiry:
    """Volatility smile extraction."""

    def test_returns_dict_keyed_by_expiry(self):
        df = _make_surface_chain(n_expiries=3)
        smiles = vol_surface.smile_per_expiry(df)
        assert len(smiles) == 3

    def test_each_smile_has_arrays(self):
        df = _make_surface_chain(n_expiries=2)
        smiles = vol_surface.smile_per_expiry(df)
        for exp, data in smiles.items():
            assert "log_moneyness" in data
            assert "iv" in data
            assert len(data["log_moneyness"]) == len(data["iv"])

    def test_smile_sorted_by_moneyness(self):
        df = _make_surface_chain()
        smiles = vol_surface.smile_per_expiry(df)
        for exp, data in smiles.items():
            assert np.all(np.diff(data["log_moneyness"]) >= 0)

    def test_empty_chain(self):
        df = pd.DataFrame(columns=["iv", "log_moneyness", "strike", "dte", "expiry"])
        smiles = vol_surface.smile_per_expiry(df)
        assert len(smiles) == 0


class TestTermStructure:
    """ATM vol term structure."""

    def test_returns_dataframe(self):
        df = _make_surface_chain()
        ts = vol_surface.term_structure(df, spot=100.0)
        assert isinstance(ts, pd.DataFrame)
        assert "atm_iv" in ts.columns
        assert len(ts) > 0

    def test_sorted_by_T(self):
        df = _make_surface_chain()
        ts = vol_surface.term_structure(df, spot=100.0)
        assert ts["T"].is_monotonic_increasing

    def test_empty_chain(self):
        df = pd.DataFrame(columns=["iv", "strike", "T", "dte", "expiry"])
        ts = vol_surface.term_structure(df, spot=100.0)
        assert ts.empty


class TestInterpolateIV:
    """Point interpolation on the surface."""

    def test_interpolate_at_data_point(self):
        df = _make_surface_chain(n_per_expiry=15, n_expiries=4)
        # Query near center of data
        result = vol_surface.interpolate_iv(df, moneyness=0.0, T=0.25,
                                            config={})
        assert not np.isnan(result)
        assert result > 0

    def test_too_few_points_returns_nan(self):
        df = pd.DataFrame({
            "iv": [0.25, 0.30],
            "log_moneyness": [0.0, 0.1],
            "T": [0.5, 0.5],
        })
        result = vol_surface.interpolate_iv(df, 0.05, 0.5, {})
        assert np.isnan(result)
