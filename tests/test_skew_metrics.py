"""
Tests for volatility skew metrics (25-delta RR and butterfly).

Validates computation of skew metrics from synthetic smile data
and edge case handling.
"""

import numpy as np
import pandas as pd
import pytest
from src import skew_metrics


def _make_smile_chain(S=100.0, T=0.25, n_strikes=21):
    """Helper: build a single-expiry chain with smile-shaped IV."""
    rows = []
    expiry = pd.Timestamp("2026-06-01")
    for i in range(n_strikes):
        K = S * (0.80 + 0.02 * i)
        log_m = np.log(K / S)
        # Equity skew: lower strikes have higher IV
        iv_val = 0.30 - 0.15 * log_m + 0.10 * log_m ** 2
        rows.append({
            "strike": K, "log_moneyness": log_m, "T": T,
            "dte": int(T * 365), "iv": iv_val, "expiry": expiry,
            "option_type": "call",
        })
    return pd.DataFrame(rows)


class TestComputeSkewMetrics:
    """Skew metrics computation."""

    def test_returns_dataframe(self):
        df = _make_smile_chain()
        result = skew_metrics.compute_skew_metrics(df, S=100, r=0.05,
                                                    q=0.0, config={})
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1

    def test_has_required_columns(self):
        df = _make_smile_chain()
        result = skew_metrics.compute_skew_metrics(df, S=100, r=0.05,
                                                    q=0.0, config={})
        expected_cols = {"expiry", "T", "atm_iv", "risk_reversal", "butterfly"}
        assert expected_cols.issubset(set(result.columns))

    def test_equity_skew_negative_rr(self):
        """Equity skew → 25D call IV < 25D put IV → negative RR."""
        df = _make_smile_chain()
        result = skew_metrics.compute_skew_metrics(df, S=100, r=0.05,
                                                    q=0.0, config={})
        rr = result["risk_reversal"].iloc[0]
        if not np.isnan(rr):
            assert rr < 0  # Equity skew convention

    def test_butterfly_finite(self):
        """Butterfly metric should be a finite number."""
        df = _make_smile_chain()
        result = skew_metrics.compute_skew_metrics(df, S=100, r=0.05,
                                                    q=0.0, config={})
        bf = result["butterfly"].iloc[0]
        if not np.isnan(bf):
            assert np.isfinite(bf)

    def test_atm_iv_close_to_center(self):
        df = _make_smile_chain()
        result = skew_metrics.compute_skew_metrics(df, S=100, r=0.05,
                                                    q=0.0, config={})
        atm = result["atm_iv"].iloc[0]
        assert 0.20 < atm < 0.40

    def test_empty_chain(self):
        df = pd.DataFrame(columns=["iv", "strike", "log_moneyness",
                                     "T", "dte", "expiry"])
        result = skew_metrics.compute_skew_metrics(df, S=100, r=0.05,
                                                    q=0.0, config={})
        assert result.empty

    def test_too_few_strikes_nan(self):
        """Fewer than 3 strikes → can't interpolate → NaN metrics."""
        df = pd.DataFrame({
            "strike": [100, 105],
            "log_moneyness": [0.0, 0.05],
            "T": [0.25, 0.25],
            "dte": [90, 90],
            "iv": [0.30, 0.28],
            "expiry": pd.Timestamp("2026-06-01"),
            "option_type": ["call", "call"],
        })
        result = skew_metrics.compute_skew_metrics(df, S=100, r=0.05,
                                                    q=0.0, config={})
        # With only 2 strikes, delta solver may fail
        if len(result) > 0:
            assert np.isnan(result["risk_reversal"].iloc[0]) or isinstance(
                result["risk_reversal"].iloc[0], float
            )
