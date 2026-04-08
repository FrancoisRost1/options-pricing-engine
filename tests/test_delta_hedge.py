"""
Tests for delta hedging simulation and P&L decomposition.
"""

import numpy as np
import pandas as pd
import pytest
from src import delta_hedge


def _flat_path(S=100, n=21):
    """Helper: constant price path (no moves)."""
    return np.full(n, S, dtype=float)


def _linear_path(S_start=100, S_end=110, n=21):
    """Helper: linear price path."""
    return np.linspace(S_start, S_end, n)


class TestSimulateHedge:
    """Delta hedging simulation."""

    def test_returns_dict_with_required_keys(self):
        path = _flat_path()
        result = delta_hedge.simulate_hedge(path, K=100, T=0.25, r=0.05,
                                             sigma=0.25)
        assert "daily" in result
        assert "summary" in result
        assert "config_used" in result

    def test_daily_df_has_rows(self):
        path = _flat_path(n=21)
        result = delta_hedge.simulate_hedge(path, K=100, T=0.25, r=0.05,
                                             sigma=0.25)
        assert len(result["daily"]) == 20  # n-1 days

    def test_daily_has_cumulative_columns(self):
        path = _flat_path()
        result = delta_hedge.simulate_hedge(path, K=100, T=0.25, r=0.05,
                                             sigma=0.25)
        df = result["daily"]
        assert "cum_total_pnl" in df.columns
        assert "cum_tc_cost" in df.columns

    def test_tc_positive(self):
        """Transaction costs are always non-negative."""
        path = _linear_path()
        result = delta_hedge.simulate_hedge(path, K=100, T=0.25, r=0.05,
                                             sigma=0.25)
        assert all(result["daily"]["tc_cost"] >= 0)

    def test_tc_drag_in_summary(self):
        path = _linear_path()
        result = delta_hedge.simulate_hedge(path, K=100, T=0.25, r=0.05,
                                             sigma=0.25)
        assert result["summary"]["tc_drag"] >= 0

    def test_flat_path_pnl_finite(self):
        """Flat price → hedging error is finite (theta dominates)."""
        path = _flat_path(n=6)
        result = delta_hedge.simulate_hedge(path, K=100, T=0.1, r=0.05,
                                             sigma=0.25)
        assert np.isfinite(result["summary"]["hedge_error_total"])

    def test_long_put_position(self):
        path = _linear_path(S_start=100, S_end=90, n=11)
        result = delta_hedge.simulate_hedge(path, K=100, T=0.25, r=0.05,
                                             sigma=0.25, position="long_put")
        assert result["config_used"]["position"] == "long_put"

    def test_short_call_position(self):
        path = _flat_path(n=11)
        result = delta_hedge.simulate_hedge(path, K=100, T=0.25, r=0.05,
                                             sigma=0.25, position="short_call")
        assert result["config_used"]["position"] == "short_call"

    def test_weekly_frequency(self):
        path = _flat_path(n=22)
        config = {"delta_hedge": {"hedge_frequency": "weekly",
                                   "tc_bps": 5.0, "slippage_bps": 2.0,
                                   "contract_multiplier": 100,
                                   "num_contracts": 1}}
        result = delta_hedge.simulate_hedge(path, K=100, T=0.25, r=0.05,
                                             sigma=0.25, config=config)
        # Weekly rebalances → fewer TC events
        tc_days = result["daily"]["tc_cost"] > 0
        assert tc_days.sum() < len(result["daily"])

    def test_n_rebalances_in_summary(self):
        path = _flat_path(n=11)
        result = delta_hedge.simulate_hedge(path, K=100, T=0.25, r=0.05,
                                             sigma=0.25)
        assert result["summary"]["n_rebalances"] > 0


class TestGBMPaths:
    """GBM path generation for simulated hedging."""

    def test_shape(self):
        paths = delta_hedge.generate_gbm_paths(100, 0.25, 0.05, 0.25,
                                                n_paths=50, n_steps=63)
        assert paths.shape == (50, 64)  # n_steps + 1

    def test_starts_at_S0(self):
        paths = delta_hedge.generate_gbm_paths(100, 0.25, 0.05, 0.25,
                                                n_paths=10, seed=42)
        assert np.all(paths[:, 0] == 100.0)

    def test_positive_prices(self):
        """GBM paths should always be positive."""
        paths = delta_hedge.generate_gbm_paths(100, 1.0, 0.05, 0.50,
                                                n_paths=100, seed=42)
        assert np.all(paths > 0)

    def test_reproducibility(self):
        p1 = delta_hedge.generate_gbm_paths(100, 0.25, 0.05, 0.25, seed=123)
        p2 = delta_hedge.generate_gbm_paths(100, 0.25, 0.05, 0.25, seed=123)
        np.testing.assert_array_equal(p1, p2)

    def test_default_steps_from_T(self):
        """n_steps defaults to int(T*252)."""
        paths = delta_hedge.generate_gbm_paths(100, 0.5, 0.05, 0.25,
                                                n_paths=5)
        expected_steps = int(0.5 * 252)
        assert paths.shape[1] == expected_steps + 1


class TestParseFrequency:
    """Rebalance frequency parsing."""

    def test_daily(self):
        idx = delta_hedge._parse_frequency("daily", 10)
        assert idx == list(range(10))

    def test_weekly(self):
        idx = delta_hedge._parse_frequency("weekly", 20)
        assert idx == [0, 5, 10, 15]

    def test_integer_frequency(self):
        idx = delta_hedge._parse_frequency(3, 10)
        assert idx == [0, 3, 6, 9]

    def test_unknown_defaults_to_daily(self):
        idx = delta_hedge._parse_frequency("monthly", 5)
        assert idx == list(range(5))
