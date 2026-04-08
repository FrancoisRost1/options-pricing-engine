"""
Tests for implied volatility extraction.

Validates round-trip accuracy (BS price → IV → BS price), edge cases
that should return np.nan, and batch extraction on a chain.
"""

import numpy as np
import pandas as pd
import pytest
from src import implied_vol as iv
from src import black_scholes as bs


class TestIVExtraction:
    """Core IV solver accuracy."""

    def test_round_trip_call(self):
        """Price with known vol → extract IV → should recover vol."""
        S, K, T, r, q = 100, 100, 1.0, 0.05, 0.0
        true_vol = 0.30
        mkt_price = bs.price(S, K, T, r, true_vol, q)
        extracted = iv.extract(mkt_price, S, K, T, r, q)
        assert pytest.approx(extracted, abs=1e-6) == true_vol

    def test_round_trip_put(self):
        S, K, T, r, q = 100, 110, 0.5, 0.05, 0.02
        true_vol = 0.40
        mkt_price = bs.price(S, K, T, r, true_vol, q, option_type="put")
        extracted = iv.extract(mkt_price, S, K, T, r, q, option_type="put")
        assert pytest.approx(extracted, abs=1e-6) == true_vol

    def test_round_trip_itm_call(self):
        S, K, T, r = 120, 100, 0.25, 0.05
        true_vol = 0.20
        mkt_price = bs.price(S, K, T, r, true_vol)
        extracted = iv.extract(mkt_price, S, K, T, r)
        assert pytest.approx(extracted, abs=1e-5) == true_vol

    def test_round_trip_otm_put(self):
        S, K, T, r = 120, 100, 0.5, 0.05
        true_vol = 0.35
        mkt_price = bs.price(S, K, T, r, true_vol, option_type="put")
        extracted = iv.extract(mkt_price, S, K, T, r, option_type="put")
        assert pytest.approx(extracted, abs=1e-5) == true_vol

    def test_high_vol_round_trip(self):
        """Should handle high vol (meme stocks)."""
        S, K, T, r = 100, 100, 0.25, 0.05
        true_vol = 2.0
        mkt_price = bs.price(S, K, T, r, true_vol)
        extracted = iv.extract(mkt_price, S, K, T, r)
        assert pytest.approx(extracted, abs=1e-4) == true_vol

    def test_low_vol_round_trip(self):
        S, K, T, r = 100, 100, 1.0, 0.05
        true_vol = 0.05
        mkt_price = bs.price(S, K, T, r, true_vol)
        extracted = iv.extract(mkt_price, S, K, T, r)
        assert pytest.approx(extracted, abs=1e-5) == true_vol


class TestIVFailureCases:
    """Cases that must return np.nan — never fake-fill."""

    def test_zero_market_price(self):
        assert np.isnan(iv.extract(0, 100, 100, 1.0, 0.05))

    def test_negative_market_price(self):
        assert np.isnan(iv.extract(-5, 100, 100, 1.0, 0.05))

    def test_price_below_intrinsic_call(self):
        """Call price below intrinsic → no valid IV."""
        S, K, T, r = 120, 100, 1.0, 0.05
        intrinsic = S - K * np.exp(-r * T)
        assert np.isnan(iv.extract(intrinsic * 0.5, S, K, T, r))

    def test_price_above_upper_bound_call(self):
        """Call price > S → no valid IV."""
        assert np.isnan(iv.extract(110, 100, 100, 1.0, 0.05))

    def test_price_above_upper_bound_put(self):
        """Put price > K*e^(-rT) → no valid IV."""
        K, r, T = 100, 0.05, 1.0
        upper = K * np.exp(-r * T)
        assert np.isnan(iv.extract(upper + 1, 100, K, T, r, option_type="put"))

    def test_zero_spot(self):
        assert np.isnan(iv.extract(5, 0, 100, 1.0, 0.05))

    def test_zero_strike(self):
        assert np.isnan(iv.extract(5, 100, 0, 1.0, 0.05))

    def test_zero_time_expired(self):
        assert np.isnan(iv.extract(5, 100, 100, 0, 0.05))

    def test_negative_time(self):
        assert np.isnan(iv.extract(5, 100, 100, -0.5, 0.05))


class TestIVConfig:
    """Config-driven solver parameters."""

    def test_custom_bounds(self):
        config = {"implied_vol": {"vol_lower_bound": 0.01,
                                   "vol_upper_bound": 5.0,
                                   "tolerance": 1e-6}}
        S, K, T, r = 100, 100, 1.0, 0.05
        mkt = bs.price(S, K, T, r, 0.30)
        extracted = iv.extract(mkt, S, K, T, r, config=config)
        assert pytest.approx(extracted, abs=1e-4) == 0.30

    def test_none_config_uses_defaults(self):
        mkt = bs.price(100, 100, 1.0, 0.05, 0.25)
        extracted = iv.extract(mkt, 100, 100, 1.0, 0.05, config=None)
        assert pytest.approx(extracted, abs=1e-5) == 0.25


class TestIVChain:
    """Batch IV extraction on a chain DataFrame."""

    def test_extract_chain_adds_iv_column(self):
        df = pd.DataFrame({
            "strike": [90, 100, 110],
            "mid": [bs.price(100, 90, 0.5, 0.05, 0.25),
                    bs.price(100, 100, 0.5, 0.05, 0.25),
                    bs.price(100, 110, 0.5, 0.05, 0.25)],
            "T": [0.5, 0.5, 0.5],
            "option_type": ["call", "call", "call"],
        })
        result = iv.extract_chain(df, S=100, r=0.05)
        assert "iv" in result.columns
        assert result["iv"].notna().all()
        assert all(pytest.approx(v, abs=1e-4) == 0.25 for v in result["iv"])

    def test_extract_chain_nan_for_bad_mid(self):
        df = pd.DataFrame({
            "strike": [100],
            "mid": [0.0],  # Invalid price
            "T": [0.5],
            "option_type": ["call"],
        })
        result = iv.extract_chain(df, S=100, r=0.05)
        assert np.isnan(result["iv"].iloc[0])

    def test_extract_chain_preserves_original(self):
        df = pd.DataFrame({
            "strike": [100], "mid": [10.0], "T": [0.5],
            "option_type": ["call"],
        })
        result = iv.extract_chain(df, S=100, r=0.05)
        assert "iv" not in df.columns  # Original untouched
