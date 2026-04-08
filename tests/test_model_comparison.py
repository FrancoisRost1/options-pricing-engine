"""
Tests for cross-model comparison and convergence analysis.

Validates that BS, Binomial, and MC agree for European options
and that convergence data structures are correct.
"""

import numpy as np
import pandas as pd
import pytest
from src import model_comparison


S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.25


class TestCompareModels:
    """Cross-model price comparison."""

    def test_all_three_agree_european_call(self):
        """BS, Binomial, MC should agree within $0.10 for vanilla European."""
        result = model_comparison.compare_models(S, K, T, r, sigma)
        assert result["max_deviation"] < 0.10

    def test_all_three_agree_european_put(self):
        result = model_comparison.compare_models(S, K, T, r, sigma,
                                                  option_type="put")
        assert result["max_deviation"] < 0.10

    def test_returns_required_keys(self):
        result = model_comparison.compare_models(S, K, T, r, sigma)
        expected = {"bs_price", "binomial_price", "mc_price",
                    "mc_std_error", "max_deviation"}
        assert expected.issubset(set(result.keys()))

    def test_mc_ci_brackets_bs(self):
        """BS price should fall within MC 95% CI."""
        result = model_comparison.compare_models(S, K, T, r, sigma)
        assert result["mc_ci_lower"] <= result["bs_price"] <= result["mc_ci_upper"]

    def test_with_dividend(self):
        result = model_comparison.compare_models(S, K, T, r, sigma, q=0.03)
        assert result["max_deviation"] < 0.15

    def test_config_overrides(self):
        config = {"binomial_tree": {"default_steps": 50},
                  "monte_carlo": {"paths": 5000, "time_steps": 50, "seed": 42,
                                  "use_antithetic": True, "use_control_variate": True,
                                  "control_variate_beta_mode": "estimate"}}
        result = model_comparison.compare_models(S, K, T, r, sigma, config=config)
        assert not np.isnan(result["bs_price"])


class TestBinomialConvergence:
    """Binomial convergence to BS."""

    def test_returns_dataframe(self):
        df = model_comparison.binomial_convergence(S, K, T, r, sigma)
        assert isinstance(df, pd.DataFrame)
        assert "steps" in df.columns
        assert "error" in df.columns

    def test_error_decreases(self):
        df = model_comparison.binomial_convergence(
            S, K, T, r, sigma,
            config={"binomial_tree": {"convergence_steps": [10, 200]}}
        )
        assert abs(df["error"].iloc[-1]) < abs(df["error"].iloc[0])

    def test_bs_price_column_constant(self):
        df = model_comparison.binomial_convergence(S, K, T, r, sigma)
        assert df["bs_price"].nunique() == 1


class TestMCConvergence:
    """MC convergence analysis."""

    def test_returns_dataframe(self):
        config = {"monte_carlo": {"convergence_paths": [1000, 5000],
                                   "time_steps": 50, "seed": 42,
                                   "use_antithetic": True,
                                   "use_control_variate": True,
                                   "control_variate_beta_mode": "estimate"}}
        df = model_comparison.mc_convergence(S, K, T, r, sigma, config=config)
        assert isinstance(df, pd.DataFrame)
        assert "n_paths" in df.columns

    def test_ci_tightens(self):
        """Without CV, CI should tighten with more paths."""
        config = {"monte_carlo": {"convergence_paths": [1000, 50000],
                                   "time_steps": 50, "seed": 42,
                                   "use_antithetic": False,
                                   "use_control_variate": False,
                                   "control_variate_beta_mode": "estimate"}}
        df = model_comparison.mc_convergence(S, K, T, r, sigma, config=config)
        ci_first = df["ci_upper"].iloc[0] - df["ci_lower"].iloc[0]
        ci_last = df["ci_upper"].iloc[-1] - df["ci_lower"].iloc[-1]
        assert ci_last < ci_first


class TestCompareChain:
    """Chain-level comparison."""

    def test_adds_model_columns(self):
        chain = pd.DataFrame({
            "strike": [95, 100, 105],
            "T": [0.5, 0.5, 0.5],
            "option_type": ["call", "call", "call"],
            "iv": [0.28, 0.25, 0.23],
        })
        config = {"binomial_tree": {"default_steps": 50},
                  "monte_carlo": {"paths": 5000, "time_steps": 50, "seed": 42,
                                  "use_antithetic": True, "use_control_variate": True,
                                  "control_variate_beta_mode": "estimate"}}
        result = model_comparison.compare_chain(chain, S=100, r=0.05,
                                                 q=0.0, config=config)
        assert "bs_price" in result.columns
        assert "binomial_price" in result.columns
        assert len(result) == 3
