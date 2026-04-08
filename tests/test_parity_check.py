"""
Tests for put-call parity validation.

Validates parity computation, violation detection, and edge cases.
"""

import numpy as np
import pandas as pd
import pytest
from src import parity_check
from src import black_scholes as bs


def _make_parity_chain(S=100, r=0.05, q=0.0, T=0.5):
    """Helper: build matched call-put pairs with BS-consistent prices."""
    strikes = [90, 95, 100, 105, 110]
    rows = []
    expiry = pd.Timestamp("2026-06-01")
    for K in strikes:
        c = bs.price(S, K, T, r, 0.25, q)
        p = bs.price(S, K, T, r, 0.25, q, option_type="put")
        rows.append({"strike": K, "mid": c, "option_type": "call",
                      "expiry": expiry, "T": T})
        rows.append({"strike": K, "mid": p, "option_type": "put",
                      "expiry": expiry, "T": T})
    return pd.DataFrame(rows)


class TestCheckParity:
    """Put-call parity validation."""

    def test_bs_consistent_prices_no_violations(self):
        """Prices from BS should satisfy parity exactly."""
        df = _make_parity_chain()
        config = {"parity_check": {"max_parity_deviation_pct": 0.02}}
        result = parity_check.check_parity(df, S=100, r=0.05, q=0.0,
                                            config=config)
        assert len(result) == 5  # 5 matched pairs
        assert result["violation"].sum() == 0

    def test_deviation_near_zero_for_bs_prices(self):
        df = _make_parity_chain()
        config = {"parity_check": {"max_parity_deviation_pct": 0.02}}
        result = parity_check.check_parity(df, S=100, r=0.05, q=0.0,
                                            config=config)
        assert result["deviation_dollar"].abs().max() < 0.01

    def test_detects_violation(self):
        """Artificially break parity → should flag."""
        df = _make_parity_chain()
        # Add $5 to all call prices → breaks parity
        df.loc[df["option_type"] == "call", "mid"] += 5.0
        config = {"parity_check": {"max_parity_deviation_pct": 0.02}}
        result = parity_check.check_parity(df, S=100, r=0.05, q=0.0,
                                            config=config)
        assert result["violation"].sum() > 0

    def test_with_dividend(self):
        df = _make_parity_chain(q=0.03)
        config = {"parity_check": {"max_parity_deviation_pct": 0.02}}
        result = parity_check.check_parity(df, S=100, r=0.05, q=0.03,
                                            config=config)
        assert result["deviation_dollar"].abs().max() < 0.01

    def test_unmatched_strikes_ignored(self):
        """Only calls → no pairs → empty result."""
        df = pd.DataFrame({
            "strike": [100], "mid": [10.0], "option_type": ["call"],
            "expiry": pd.Timestamp("2026-06-01"), "T": [0.5],
        })
        config = {"parity_check": {}}
        result = parity_check.check_parity(df, S=100, r=0.05, q=0.0,
                                            config=config)
        assert result.empty

    def test_nan_mid_dropped(self):
        df = _make_parity_chain()
        df.loc[0, "mid"] = np.nan
        config = {"parity_check": {"max_parity_deviation_pct": 0.02}}
        result = parity_check.check_parity(df, S=100, r=0.05, q=0.0,
                                            config=config)
        # One call lost → one fewer pair
        assert len(result) == 4


class TestParitySummary:
    """Summary statistics."""

    def test_no_violations_summary(self):
        df = _make_parity_chain()
        config = {"parity_check": {"max_parity_deviation_pct": 0.02}}
        parity_df = parity_check.check_parity(df, S=100, r=0.05, q=0.0,
                                               config=config)
        s = parity_check.parity_summary(parity_df)
        assert s["total_pairs"] == 5
        assert s["violations"] == 0
        assert s["violation_pct"] == 0.0

    def test_empty_df_summary(self):
        s = parity_check.parity_summary(pd.DataFrame())
        assert s["total_pairs"] == 0
        assert np.isnan(s["mean_deviation_dollar"])
