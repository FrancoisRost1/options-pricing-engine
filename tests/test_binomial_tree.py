"""
Tests for CRR binomial tree pricer.

Validates convergence to BS, American vs European premium,
and edge cases.
"""

import numpy as np
import pytest
from src import binomial_tree
from src import black_scholes as bs


class TestBinomialPricing:
    """Core pricing accuracy."""

    def test_european_call_converges_to_bs(self):
        """CRR with 500 steps should match BS within $0.01."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.25
        bs_price = bs.price(S, K, T, r, sigma)
        bt_price = binomial_tree.price(S, K, T, r, sigma, steps=500)
        assert pytest.approx(bt_price, abs=0.01) == bs_price

    def test_european_put_converges_to_bs(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.25
        bs_p = bs.price(S, K, T, r, sigma, option_type="put")
        bt_p = binomial_tree.price(S, K, T, r, sigma, option_type="put", steps=500)
        assert pytest.approx(bt_p, abs=0.01) == bs_p

    def test_american_put_geq_european_put(self):
        """American put >= European put (early exercise premium)."""
        S, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.25
        eu = binomial_tree.price(S, K, T, r, sigma, option_type="put",
                                 exercise="european", steps=200)
        am = binomial_tree.price(S, K, T, r, sigma, option_type="put",
                                 exercise="american", steps=200)
        assert am >= eu - 1e-10

    def test_american_call_no_dividend_equals_european(self):
        """American call = European call when q=0 (never optimal to exercise early)."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.25
        eu = binomial_tree.price(S, K, T, r, sigma, exercise="european", steps=200)
        am = binomial_tree.price(S, K, T, r, sigma, exercise="american", steps=200)
        assert pytest.approx(am, abs=0.01) == eu

    def test_american_call_with_dividend_premium(self):
        """American call > European call when q > 0 (dividend capture)."""
        S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.25, 0.08
        eu = binomial_tree.price(S, K, T, r, sigma, q=q, exercise="european", steps=200)
        am = binomial_tree.price(S, K, T, r, sigma, q=q, exercise="american", steps=200)
        assert am >= eu - 1e-10

    def test_deep_itm_put_american_exercise(self):
        """Deep ITM American put should be worth at least intrinsic."""
        S, K = 50, 100
        am = binomial_tree.price(S, K, 1.0, 0.05, 0.25, option_type="put",
                                 exercise="american", steps=200)
        assert am >= K - S

    def test_price_positive(self):
        p = binomial_tree.price(100, 100, 1.0, 0.05, 0.25)
        assert p > 0

    def test_with_dividend(self):
        p = binomial_tree.price(100, 100, 1.0, 0.05, 0.25, q=0.03)
        assert p > 0 and not np.isnan(p)

    def test_few_steps(self):
        """Even 1 step should produce a valid price."""
        p = binomial_tree.price(100, 100, 1.0, 0.05, 0.25, steps=1)
        assert p > 0 and not np.isnan(p)


class TestBinomialEdgeCases:
    """Edge cases for degenerate inputs."""

    def test_zero_spot(self):
        assert np.isnan(binomial_tree.price(0, 100, 1.0, 0.05, 0.25))

    def test_negative_spot(self):
        assert np.isnan(binomial_tree.price(-10, 100, 1.0, 0.05, 0.25))

    def test_zero_strike(self):
        assert np.isnan(binomial_tree.price(100, 0, 1.0, 0.05, 0.25))

    def test_zero_time(self):
        assert np.isnan(binomial_tree.price(100, 100, 0, 0.05, 0.25))

    def test_negative_time(self):
        assert np.isnan(binomial_tree.price(100, 100, -1, 0.05, 0.25))

    def test_zero_vol(self):
        assert np.isnan(binomial_tree.price(100, 100, 1.0, 0.05, 0))

    def test_zero_steps(self):
        assert np.isnan(binomial_tree.price(100, 100, 1.0, 0.05, 0.25, steps=0))

    def test_negative_steps(self):
        assert np.isnan(binomial_tree.price(100, 100, 1.0, 0.05, 0.25, steps=-5))


class TestBinomialConvergence:
    """Convergence analysis."""

    def test_convergence_returns_list(self):
        results = binomial_tree.price_convergence(100, 100, 1.0, 0.05, 0.25)
        assert len(results) == 7  # default steps_list
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_convergence_decreasing_error(self):
        """Error should generally decrease with more steps."""
        bs_p = bs.price(100, 100, 1.0, 0.05, 0.25)
        results = binomial_tree.price_convergence(100, 100, 1.0, 0.05, 0.25,
                                                  steps_list=[50, 500])
        err_50 = abs(results[0][1] - bs_p)
        err_500 = abs(results[1][1] - bs_p)
        assert err_500 < err_50

    def test_custom_steps_list(self):
        results = binomial_tree.price_convergence(100, 100, 1.0, 0.05, 0.25,
                                                  steps_list=[5, 10])
        assert len(results) == 2
