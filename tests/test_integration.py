"""
Integration tests — end-to-end pipeline using synthetic data.

Tests the full flow: generate chain → filter → extract IV →
build surface → price comparison → scenario analysis.
No network calls — all synthetic.
"""

import numpy as np
import pandas as pd
import pytest
from utils.config_loader import load_config
from src import synthetic_data
from src import chain_filter
from src import implied_vol
from src import vol_surface
from src import model_comparison
from src import parity_check
from src import scenario_analysis as sa
from src import delta_hedge
from src import black_scholes as bs


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def market(config):
    return synthetic_data.generate_chain(config)


@pytest.fixture(scope="module")
def filtered(market, config):
    return chain_filter.apply_filters(market["chain"], config)


@pytest.fixture(scope="module")
def chain_with_iv(filtered, market, config):
    return implied_vol.extract_chain(
        filtered, S=market["spot"], r=market["risk_free_rate"],
        q=market["dividend_yield"], config=config,
    )


class TestEndToEndPipeline:
    """Full pipeline: synthetic → filter → IV → analysis."""

    def test_synthetic_chain_non_empty(self, market):
        assert not market["chain"].empty
        assert market["spot"] > 0

    def test_filtering_reduces_chain(self, market, filtered):
        assert len(filtered) <= len(market["chain"])
        assert len(filtered) > 0

    def test_iv_extraction_mostly_valid(self, chain_with_iv):
        valid = chain_with_iv["iv"].notna().sum()
        total = len(chain_with_iv)
        assert valid / total > 0.80  # >80% should solve

    def test_iv_values_reasonable(self, chain_with_iv):
        valid_iv = chain_with_iv["iv"].dropna()
        assert valid_iv.min() > 0.01
        assert valid_iv.max() < 5.0

    def test_surface_builds(self, chain_with_iv, config):
        surface = vol_surface.build_surface(chain_with_iv, config)
        assert surface is not None
        assert surface["n_points"] > 10

    def test_smile_extraction(self, chain_with_iv):
        smiles = vol_surface.smile_per_expiry(chain_with_iv)
        assert len(smiles) > 0

    def test_term_structure(self, chain_with_iv, market):
        ts = vol_surface.term_structure(chain_with_iv, market["spot"])
        assert len(ts) > 0
        assert "atm_iv" in ts.columns

    def test_model_comparison_agreement(self, market, config):
        """All 3 models agree on an ATM option."""
        S = market["spot"]
        K = S  # ATM
        T = 0.25
        r = market["risk_free_rate"]
        sigma = 0.25
        result = model_comparison.compare_models(S, K, T, r, sigma,
                                                  config=config)
        assert result["max_deviation"] < 0.50

    def test_parity_check_on_synthetic(self, chain_with_iv, market, config):
        result = parity_check.check_parity(
            chain_with_iv, S=market["spot"],
            r=market["risk_free_rate"],
            q=market["dividend_yield"], config=config,
        )
        if not result.empty:
            # Synthetic BS-based prices should mostly satisfy parity
            violation_pct = result["violation"].mean()
            assert violation_pct < 0.50

    def test_scenario_grid(self, market):
        S = market["spot"]
        K = S
        result = sa.compute_pnl_grid(S, K, T=0.25, r=market["risk_free_rate"],
                                      sigma=0.25)
        assert not np.any(np.isnan(result["values"]))

    def test_delta_hedge_simulation(self, market):
        S = market["spot"]
        paths = delta_hedge.generate_gbm_paths(
            S, T=0.25, r=market["risk_free_rate"], sigma=0.25,
            n_paths=1, n_steps=63, seed=42,
        )
        result = delta_hedge.simulate_hedge(
            paths[0], K=S, T=0.25,
            r=market["risk_free_rate"], sigma=0.25,
        )
        assert len(result["daily"]) == 63
        assert "total_pnl" in result["summary"]


class TestConfigLoader:
    """Config loading integration."""

    def test_config_has_required_sections(self, config):
        required = ["data", "synthetic", "risk_free_rate", "chain_filter",
                     "black_scholes", "binomial_tree", "monte_carlo",
                     "implied_vol", "vol_surface", "greeks", "scenarios",
                     "delta_hedge"]
        for section in required:
            assert section in config, f"Missing config section: {section}"

    def test_synthetic_config_has_spot(self, config):
        assert config["synthetic"]["spot_price"] > 0

    def test_binomial_default_steps(self, config):
        assert config["binomial_tree"]["default_steps"] >= 10

    def test_mc_paths_positive(self, config):
        assert config["monte_carlo"]["paths"] > 0
