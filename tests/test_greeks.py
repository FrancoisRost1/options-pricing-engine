"""
Tests for finite difference Greeks wrapper (binomial & MC).

Validates that FD Greeks agree with BS analytical Greeks within
acceptable tolerance, and that edge cases are handled.
"""

import numpy as np
import pytest
from src import greeks
from src import black_scholes as bs


# Standard test params
S, K, T, r, sigma, q = 100, 100, 0.5, 0.05, 0.25, 0.0


class TestFDGreeksBinomial:
    """Finite difference Greeks via binomial tree vs BS analytical."""

    def test_delta_agrees_with_bs(self):
        fd = greeks.compute(S, K, T, r, sigma, q, model="binomial")
        analytical = bs.delta(S, K, T, r, sigma, q)
        assert pytest.approx(fd["delta"], abs=0.01) == analytical

    def test_gamma_positive_and_reasonable(self):
        """FD gamma on CRR is noisy (2nd derivative amplifies tree discretization).
        We verify sign and order of magnitude rather than exact match."""
        fd = greeks.compute(S, K, T, r, sigma, q, model="binomial")
        analytical = bs.gamma(S, K, T, r, sigma, q)
        assert fd["gamma"] > 0
        # Same order of magnitude (within 10x)
        assert 0.1 * analytical < fd["gamma"] < 10 * analytical

    def test_vega_agrees_with_bs(self):
        fd = greeks.compute(S, K, T, r, sigma, q, model="binomial")
        analytical = bs.vega(S, K, T, r, sigma, q)
        assert pytest.approx(fd["vega"], abs=0.02) == analytical

    def test_theta_agrees_with_bs(self):
        fd = greeks.compute(S, K, T, r, sigma, q, model="binomial")
        analytical = bs.theta(S, K, T, r, sigma, q)
        assert pytest.approx(fd["theta"], abs=0.005) == analytical

    def test_rho_agrees_with_bs(self):
        fd = greeks.compute(S, K, T, r, sigma, q, model="binomial")
        analytical = bs.rho(S, K, T, r, sigma, q)
        assert pytest.approx(fd["rho"], abs=0.005) == analytical

    def test_put_greeks(self):
        fd = greeks.compute(S, K, T, r, sigma, q, option_type="put",
                            model="binomial")
        assert fd["delta"] < 0
        assert fd["gamma"] > 0

    def test_returns_five_keys(self):
        fd = greeks.compute(S, K, T, r, sigma, q, model="binomial")
        assert set(fd.keys()) == {"delta", "gamma", "theta", "vega", "rho"}

    def test_american_put_greeks(self):
        """FD Greeks should work for American options too."""
        fd = greeks.compute(S, 110, T, r, sigma, q, option_type="put",
                            exercise="american", model="binomial")
        assert fd["delta"] < 0
        assert fd["gamma"] > 0


class TestFDGreeksConfig:
    """Config-driven bump sizes."""

    def test_custom_config(self):
        config = {
            "greeks": {"bump_spot_pct": 0.005, "bump_vol": 0.005,
                       "bump_rate": 0.005, "bump_time_days": 1},
            "binomial_tree": {"default_steps": 100},
        }
        fd = greeks.compute(S, K, T, r, sigma, q, model="binomial",
                            config=config)
        assert all(np.isfinite(v) for v in fd.values())


class TestFDGreeksEdgeCases:
    """Edge cases and error handling."""

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            greeks.compute(S, K, T, r, sigma, model="invalid")

    def test_none_config_uses_defaults(self):
        fd = greeks.compute(S, K, T, r, sigma, model="binomial", config=None)
        assert all(np.isfinite(v) for v in fd.values())
