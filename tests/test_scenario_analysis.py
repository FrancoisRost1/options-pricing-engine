"""
Tests for P&L scenario analysis — heatmaps and presets.
"""

import numpy as np
import pytest
from src import scenario_analysis as sa
from src import black_scholes as bs


S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.25


class TestPnlGrid:
    """Spot x Vol P&L heatmap."""

    def test_returns_dict_with_values(self):
        result = sa.compute_pnl_grid(S, K, T, r, sigma)
        assert "values" in result
        assert "spot_changes" in result
        assert "vol_changes" in result

    def test_grid_shape_matches_axes(self):
        config = {"scenarios": {
            "spot_range_pct": [-0.10, 0.0, 0.10],
            "vol_range_pts": [-0.05, 0.0, 0.05],
        }}
        result = sa.compute_pnl_grid(S, K, T, r, sigma, config=config)
        assert result["values"].shape == (3, 3)

    def test_center_cell_is_current_price(self):
        """At (0% spot, 0 vol) → should equal current BS price."""
        config = {"scenarios": {
            "spot_range_pct": [0.0],
            "vol_range_pts": [0.0],
        }}
        result = sa.compute_pnl_grid(S, K, T, r, sigma, config=config)
        expected = bs.price(S, K, T, r, sigma)
        assert pytest.approx(result["values"][0, 0], abs=1e-6) == expected

    def test_pnl_mode_with_entry_price(self):
        entry = bs.price(S, K, T, r, sigma)
        config = {"scenarios": {
            "spot_range_pct": [0.0],
            "vol_range_pts": [0.0],
        }}
        result = sa.compute_pnl_grid(S, K, T, r, sigma,
                                      entry_price=entry, config=config)
        assert result["is_pnl"] is True
        assert pytest.approx(result["values"][0, 0], abs=1e-6) == 0.0

    def test_no_nan_in_normal_grid(self):
        result = sa.compute_pnl_grid(S, K, T, r, sigma)
        assert not np.any(np.isnan(result["values"]))

    def test_higher_spot_higher_call_value(self):
        config = {"scenarios": {
            "spot_range_pct": [-0.10, 0.10],
            "vol_range_pts": [0.0],
        }}
        result = sa.compute_pnl_grid(S, K, T, r, sigma, config=config)
        assert result["values"][0, 1] > result["values"][0, 0]

    def test_put_lower_at_higher_spot(self):
        config = {"scenarios": {
            "spot_range_pct": [-0.10, 0.10],
            "vol_range_pts": [0.0],
        }}
        result = sa.compute_pnl_grid(S, K, T, r, sigma,
                                      option_type="put", config=config)
        assert result["values"][0, 0] > result["values"][0, 1]


class TestTimeDecay:
    """Time decay scenarios."""

    def test_returns_dict_keyed_by_days(self):
        config = {"scenarios": {"time_forward_days": [0, 7, 14]}}
        result = sa.compute_time_decay(S, K, T, r, sigma, config=config)
        assert set(result.keys()) == {0, 7, 14}

    def test_value_decreases_with_time(self):
        """ATM call loses value as time passes (theta)."""
        config = {"scenarios": {
            "time_forward_days": [0, 30],
            "spot_range_pct": [0.0],
            "vol_range_pts": [0.0],
        }}
        result = sa.compute_time_decay(S, K, T, r, sigma, config=config)
        val_now = result[0]["values"][0, 0]
        val_later = result[30]["values"][0, 0]
        assert val_later < val_now


class TestPresets:
    """Scenario presets."""

    def test_earnings_preset_returns_results(self):
        config = {"scenarios": {"presets": {
            "earnings": {
                "spot_changes": [-0.05, 0.0, 0.05],
                "vol_change_pre": 0.10,
                "vol_change_post": -0.15,
                "rate_change": 0.0,
            }
        }}}
        results = sa.apply_preset(S, K, T, r, sigma, config=config,
                                   preset_name="earnings")
        assert len(results) == 6  # 3 spots × 2 (pre + post)

    def test_vol_crush_preset(self):
        config = {"scenarios": {"presets": {
            "vol_crush": {
                "spot_changes": [0.0],
                "vol_changes": [-0.10, -0.20],
                "rate_change": 0.0,
            }
        }}}
        results = sa.apply_preset(S, K, T, r, sigma, config=config,
                                   preset_name="vol_crush")
        assert len(results) == 2
        # Vol crush → lower price
        for r_dict in results:
            assert r_dict["pnl"] < 0

    def test_rate_hike_preset(self):
        config = {"scenarios": {"presets": {
            "rate_hike": {
                "spot_changes": [0.0],
                "vol_change": 0.0,
                "rate_changes": [0.0025, 0.005],
            }
        }}}
        results = sa.apply_preset(S, K, T, r, sigma, config=config,
                                   preset_name="rate_hike")
        assert len(results) == 2

    def test_unknown_preset_returns_empty(self):
        results = sa.apply_preset(S, K, T, r, sigma, preset_name="nonexistent")
        assert results == []

    def test_each_result_has_pnl(self):
        config = {"scenarios": {"presets": {
            "vol_crush": {
                "spot_changes": [0.0],
                "vol_changes": [-0.10],
                "rate_change": 0.0,
            }
        }}}
        results = sa.apply_preset(S, K, T, r, sigma, config=config,
                                   preset_name="vol_crush")
        for r_dict in results:
            assert "pnl" in r_dict
            assert not np.isnan(r_dict["pnl"])
