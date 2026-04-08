"""
Tests for synthetic options chain generator.
"""

import numpy as np
import pandas as pd
import pytest
from src import synthetic_data


def _default_config():
    return {
        "synthetic": {
            "spot_price": 100.0,
            "base_vol": 0.25,
            "skew_slope": -0.10,
            "expiry_days": [7, 30, 90],
            "strikes_per_expiry": 11,
            "strike_spacing_pct": 0.025,
            "risk_free_rate": 0.045,
            "dividend_yield": 0.015,
            "bid_ask_noise_pct": 0.02,
        }
    }


class TestGenerateChain:
    """Synthetic chain generation."""

    def test_returns_dict_with_required_keys(self):
        result = synthetic_data.generate_chain(_default_config())
        assert set(result.keys()) >= {"ticker", "spot", "chain",
                                       "risk_free_rate", "dividend_yield",
                                       "data_source"}

    def test_data_source_is_synthetic(self):
        result = synthetic_data.generate_chain(_default_config())
        assert result["data_source"] == "synthetic"

    def test_ticker_is_synthetic(self):
        result = synthetic_data.generate_chain(_default_config())
        assert result["ticker"] == "SYNTHETIC"

    def test_spot_matches_config(self):
        result = synthetic_data.generate_chain(_default_config())
        assert result["spot"] == 100.0

    def test_chain_is_dataframe(self):
        result = synthetic_data.generate_chain(_default_config())
        assert isinstance(result["chain"], pd.DataFrame)

    def test_chain_has_required_columns(self):
        result = synthetic_data.generate_chain(_default_config())
        required = {"strike", "bid", "ask", "mid", "volume", "openInterest",
                    "option_type", "expiry", "T", "dte", "log_moneyness"}
        assert required.issubset(set(result["chain"].columns))

    def test_chain_has_both_types(self):
        result = synthetic_data.generate_chain(_default_config())
        types = result["chain"]["option_type"].unique()
        assert "call" in types
        assert "put" in types

    def test_chain_has_expected_expiries(self):
        config = _default_config()
        result = synthetic_data.generate_chain(config)
        assert result["chain"]["dte"].nunique() == 3

    def test_bid_less_than_ask(self):
        result = synthetic_data.generate_chain(_default_config())
        chain = result["chain"]
        assert (chain["bid"] < chain["ask"]).all()

    def test_mid_between_bid_ask(self):
        result = synthetic_data.generate_chain(_default_config())
        chain = result["chain"]
        assert (chain["mid"] >= chain["bid"]).all()
        assert (chain["mid"] <= chain["ask"]).all()

    def test_positive_prices(self):
        result = synthetic_data.generate_chain(_default_config())
        chain = result["chain"]
        assert (chain["bid"] > 0).all()
        assert (chain["mid"] > 0).all()

    def test_equity_skew_present(self):
        """Lower strikes should have higher IV (negative skew slope)."""
        result = synthetic_data.generate_chain(_default_config())
        chain = result["chain"]
        calls = chain[chain["option_type"] == "call"]
        # Group by DTE, check skew direction
        for _, group in calls.groupby("dte"):
            low_strike = group[group["log_moneyness"] < -0.05]
            high_strike = group[group["log_moneyness"] > 0.05]
            if len(low_strike) > 0 and len(high_strike) > 0:
                assert low_strike["impliedVolatility"].mean() > high_strike["impliedVolatility"].mean()

    def test_reproducible(self):
        """Same config → same chain (fixed seed)."""
        r1 = synthetic_data.generate_chain(_default_config())
        r2 = synthetic_data.generate_chain(_default_config())
        pd.testing.assert_frame_equal(r1["chain"], r2["chain"])
