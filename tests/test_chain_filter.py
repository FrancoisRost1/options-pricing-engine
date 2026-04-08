"""
Tests for options chain garbage filter.

Validates that each filter criterion works independently and
that edge cases (empty df, missing columns) are handled.
"""

import numpy as np
import pandas as pd
import pytest
from src import chain_filter


def _make_chain(n=10, **overrides):
    """Helper: build a minimal valid chain DataFrame."""
    data = {
        "strike": [100 + i for i in range(n)],
        "bid": [5.0] * n,
        "ask": [5.5] * n,
        "mid": [5.25] * n,
        "volume": [100] * n,
        "openInterest": [500] * n,
        "option_type": ["call"] * n,
        "expiry": pd.Timestamp("2026-06-01"),
        "T": [0.25] * n,
        "dte": [60] * n,
        "price_source": ["mid"] * n,
        "log_moneyness": [0.0] * n,
    }
    data.update(overrides)
    return pd.DataFrame(data)


class TestApplyFilters:
    """Individual filter criterion tests."""

    def test_filters_excluded_price_source(self):
        df = _make_chain(price_source=["excluded"] * 10)
        config = {"chain_filter": {}}
        result = chain_filter.apply_filters(df, config)
        assert len(result) == 0

    def test_filters_zero_mid(self):
        df = _make_chain(mid=[0.0] * 10)
        config = {"chain_filter": {}}
        result = chain_filter.apply_filters(df, config)
        assert len(result) == 0

    def test_filters_wide_spread(self):
        df = _make_chain(bid=[1.0] * 10, ask=[10.0] * 10, mid=[5.5] * 10)
        config = {"chain_filter": {"max_spread_pct": 0.50}}
        result = chain_filter.apply_filters(df, config)
        # Spread = 9/5.5 = 1.64 > 0.50 → all removed
        assert len(result) == 0

    def test_keeps_tight_spread(self):
        df = _make_chain(bid=[5.0] * 10, ask=[5.1] * 10, mid=[5.05] * 10)
        config = {"chain_filter": {"max_spread_pct": 0.50,
                                    "min_volume": 0, "min_open_interest": 0}}
        result = chain_filter.apply_filters(df, config)
        assert len(result) == 10

    def test_filters_low_volume(self):
        df = _make_chain(volume=[5] * 10)
        config = {"chain_filter": {"min_volume": 10}}
        result = chain_filter.apply_filters(df, config)
        assert len(result) == 0

    def test_filters_low_oi(self):
        df = _make_chain(openInterest=[3] * 10)
        config = {"chain_filter": {"min_open_interest": 10}}
        result = chain_filter.apply_filters(df, config)
        assert len(result) == 0

    def test_filters_moneyness_out_of_bounds(self):
        df = _make_chain(log_moneyness=[0.6] * 10)
        config = {"chain_filter": {"moneyness_bounds": {"lower": -0.5, "upper": 0.5}}}
        result = chain_filter.apply_filters(df, config)
        assert len(result) == 0

    def test_filters_expired_options(self):
        df = _make_chain(dte=[0] * 10)
        config = {"chain_filter": {"min_dte": 1}}
        result = chain_filter.apply_filters(df, config)
        assert len(result) == 0

    def test_does_not_modify_original(self):
        df = _make_chain()
        original_len = len(df)
        config = {"chain_filter": {"min_volume": 1000}}
        chain_filter.apply_filters(df, config)
        assert len(df) == original_len

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame()
        result = chain_filter.apply_filters(df, {})
        assert result.empty


class TestDashboardDefaults:
    """Stricter default filters for dashboard view."""

    def test_limits_expiries(self):
        expiries = [pd.Timestamp(f"2026-0{i}-01") for i in range(1, 7)]
        dfs = []
        for exp in expiries:
            d = _make_chain(n=2, volume=[100] * 2)
            d["expiry"] = exp
            dfs.append(d)
        df = pd.concat(dfs, ignore_index=True)
        config = {"dashboard_defaults": {"max_expiries_shown": 3,
                                          "min_volume_default": 0}}
        result = chain_filter.apply_dashboard_defaults(df, config)
        assert result["expiry"].nunique() <= 3

    def test_empty_df_returns_empty(self):
        result = chain_filter.apply_dashboard_defaults(pd.DataFrame(), {})
        assert result.empty


class TestFilterSummary:
    """Summary statistics."""

    def test_summary_counts(self):
        raw = _make_chain(n=20)
        filtered = _make_chain(n=15)
        s = chain_filter.filter_summary(raw, filtered)
        assert s["raw_contracts"] == 20
        assert s["filtered_contracts"] == 15
        assert s["removed"] == 5
        assert pytest.approx(s["removal_pct"], abs=0.1) == 25.0

    def test_summary_zero_raw(self):
        s = chain_filter.filter_summary(pd.DataFrame(columns=["expiry"]),
                                         pd.DataFrame(columns=["expiry"]))
        assert s["raw_contracts"] == 0
        # 0 raw, 0 filtered: (1 - 0/max(0,1)) * 100 = 100.0
        assert s["removal_pct"] == 100.0
