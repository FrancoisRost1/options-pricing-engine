"""
Tests for Monte Carlo pricer with variance reduction.

Validates pricing accuracy against BS, reproducibility via seed,
variance reduction effectiveness, and edge cases.
"""

import numpy as np
import pytest
from src import monte_carlo as mc
from src import black_scholes as bs


class TestMCPricing:
    """Core MC pricing accuracy."""

    def test_call_within_bs_tolerance(self):
        """MC call should be within 2 SE of BS price."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.25
        bs_p = bs.price(S, K, T, r, sigma)
        result = mc.price(S, K, T, r, sigma, n_paths=100000, seed=42)
        assert abs(result["price"] - bs_p) < 2 * result["std_error"] + 0.05

    def test_put_within_bs_tolerance(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.25
        bs_p = bs.price(S, K, T, r, sigma, option_type="put")
        result = mc.price(S, K, T, r, sigma, option_type="put",
                          n_paths=100000, seed=42)
        assert abs(result["price"] - bs_p) < 2 * result["std_error"] + 0.05

    def test_returns_dict_with_required_keys(self):
        result = mc.price(100, 100, 1.0, 0.05, 0.25, n_paths=1000)
        assert set(result.keys()) >= {"price", "std_error", "ci_lower",
                                       "ci_upper", "cv_beta_mode"}

    def test_ci_contains_price(self):
        result = mc.price(100, 100, 1.0, 0.05, 0.25, n_paths=10000, seed=42)
        assert result["ci_lower"] <= result["price"] <= result["ci_upper"]

    def test_reproducibility_with_seed(self):
        r1 = mc.price(100, 100, 1.0, 0.05, 0.25, seed=123, n_paths=10000)
        r2 = mc.price(100, 100, 1.0, 0.05, 0.25, seed=123, n_paths=10000)
        assert r1["price"] == r2["price"]

    def test_different_seeds_differ(self):
        r1 = mc.price(100, 100, 1.0, 0.05, 0.25, seed=1, n_paths=10000,
                      use_control_variate=False)
        r2 = mc.price(100, 100, 1.0, 0.05, 0.25, seed=2, n_paths=10000,
                      use_control_variate=False)
        assert r1["price"] != r2["price"]

    def test_with_dividend(self):
        result = mc.price(100, 100, 1.0, 0.05, 0.25, q=0.03,
                          n_paths=10000, seed=42)
        assert not np.isnan(result["price"])
        assert result["price"] > 0


class TestVarianceReduction:
    """Antithetic variates and control variate effectiveness."""

    def test_antithetic_reduces_std_error(self):
        """Antithetic should have lower SE than standard for same paths."""
        standard = mc.price(100, 100, 1.0, 0.05, 0.25, n_paths=50000,
                            seed=42, use_antithetic=False, use_control_variate=False)
        anti = mc.price(100, 100, 1.0, 0.05, 0.25, n_paths=50000,
                        seed=42, use_antithetic=True, use_control_variate=False)
        assert anti["std_error"] <= standard["std_error"] * 1.1  # allow small margin

    def test_control_variate_reduces_std_error(self):
        """Control variate should reduce SE dramatically for vanilla options."""
        standard = mc.price(100, 100, 1.0, 0.05, 0.25, n_paths=50000,
                            seed=42, use_antithetic=False, use_control_variate=False)
        cv = mc.price(100, 100, 1.0, 0.05, 0.25, n_paths=50000,
                      seed=42, use_antithetic=False, use_control_variate=True)
        assert cv["std_error"] < standard["std_error"]

    def test_cv_beta_fixed_mode(self):
        result = mc.price(100, 100, 1.0, 0.05, 0.25, n_paths=10000,
                          seed=42, cv_beta_mode="fixed", cv_beta_fixed=1.0)
        assert result["cv_beta_mode"] == "fixed"
        assert result["cv_beta_used"] == 1.0

    def test_cv_beta_estimate_mode(self):
        result = mc.price(100, 100, 1.0, 0.05, 0.25, n_paths=10000,
                          seed=42, cv_beta_mode="estimate")
        assert result["cv_beta_mode"] == "estimate"
        assert result["cv_beta_used"] is not None

    def test_variance_comparison_returns_four_methods(self):
        comp = mc.price_variance_comparison(100, 100, 1.0, 0.05, 0.25,
                                            n_paths=5000, seed=42)
        assert set(comp.keys()) == {"standard", "antithetic",
                                     "control_variate", "both"}


class TestMCEdgeCases:
    """Edge cases returning NaN dict."""

    def test_zero_spot(self):
        r = mc.price(0, 100, 1.0, 0.05, 0.25)
        assert np.isnan(r["price"])

    def test_negative_spot(self):
        r = mc.price(-10, 100, 1.0, 0.05, 0.25)
        assert np.isnan(r["price"])

    def test_zero_time(self):
        r = mc.price(100, 100, 0, 0.05, 0.25)
        assert np.isnan(r["price"])

    def test_zero_vol(self):
        r = mc.price(100, 100, 1.0, 0.05, 0)
        assert np.isnan(r["price"])

    def test_negative_vol(self):
        r = mc.price(100, 100, 1.0, 0.05, -0.2)
        assert np.isnan(r["price"])


class TestMCConvergence:
    """Convergence analysis."""

    def test_convergence_returns_list(self):
        results = mc.price_convergence(100, 100, 1.0, 0.05, 0.25,
                                       paths_list=[1000, 5000], seed=42)
        assert len(results) == 2
        assert all("n_paths" in r for r in results)

    def test_more_paths_tighter_ci(self):
        """Without CV, more paths → tighter CI."""
        results = mc.price_convergence(100, 100, 1.0, 0.05, 0.25,
                                       paths_list=[1000, 50000], seed=42,
                                       use_antithetic=False,
                                       use_control_variate=False)
        ci_1k = results[0]["ci_upper"] - results[0]["ci_lower"]
        ci_50k = results[1]["ci_upper"] - results[1]["ci_lower"]
        assert ci_50k < ci_1k
