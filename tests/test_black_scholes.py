"""
Tests for Black-Scholes analytical pricing and Greeks.

Includes textbook reference values (Hull, Options Futures and Other
Derivatives) and edge-case coverage for degenerate inputs.
"""

import numpy as np
import pytest
from src import black_scholes as bs


# ── Textbook reference: Hull Example 15.6 ──────────────────────
# S=42, K=40, T=0.5, r=0.10, sigma=0.20, q=0
# Expected call ≈ 4.76, put ≈ 0.81
HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA = 42.0, 40.0, 0.5, 0.10, 0.20


class TestBSPricing:
    """Black-Scholes pricing against textbook values."""

    def test_call_hull_reference(self):
        p = bs.price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA)
        assert pytest.approx(p, abs=0.02) == 4.76

    def test_put_hull_reference(self):
        p = bs.price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, option_type="put")
        assert pytest.approx(p, abs=0.02) == 0.81

    def test_put_call_parity(self):
        """C - P = S*e^(-qT) - K*e^(-rT) must hold."""
        c = bs.price(100, 100, 1.0, 0.05, 0.25, q=0.02)
        p = bs.price(100, 100, 1.0, 0.05, 0.25, q=0.02, option_type="put")
        theoretical = 100 * np.exp(-0.02) - 100 * np.exp(-0.05)
        assert pytest.approx(c - p, abs=1e-10) == theoretical

    def test_deep_itm_call_near_intrinsic(self):
        """Deep ITM call ≈ S*e^(-qT) - K*e^(-rT) for low vol."""
        p = bs.price(200, 100, 1.0, 0.05, 0.01, q=0.0)
        intrinsic = 200 - 100 * np.exp(-0.05)
        assert pytest.approx(p, rel=0.01) == intrinsic

    def test_deep_otm_call_near_zero(self):
        """Deep OTM call → ~0."""
        p = bs.price(50, 200, 0.1, 0.05, 0.20)
        assert p < 0.001

    def test_atm_call_put_symmetry(self):
        """ATM forward: call ≈ put when S*e^(-qT) = K*e^(-rT)."""
        S = 100
        r, q, T = 0.05, 0.02, 1.0
        K_fwd = S * np.exp((r - q) * T)
        c = bs.price(S, K_fwd, T, r, 0.25, q)
        p = bs.price(S, K_fwd, T, r, 0.25, q, option_type="put")
        assert pytest.approx(c, abs=1e-10) == p

    def test_call_price_positive(self):
        p = bs.price(100, 100, 1.0, 0.05, 0.20)
        assert p > 0

    def test_put_price_positive(self):
        p = bs.price(100, 100, 1.0, 0.05, 0.20, option_type="put")
        assert p > 0

    def test_price_increases_with_vol(self):
        p_low = bs.price(100, 100, 1.0, 0.05, 0.10)
        p_high = bs.price(100, 100, 1.0, 0.05, 0.50)
        assert p_high > p_low

    def test_call_increases_with_spot(self):
        p_low = bs.price(90, 100, 1.0, 0.05, 0.20)
        p_high = bs.price(110, 100, 1.0, 0.05, 0.20)
        assert p_high > p_low

    def test_with_dividend_yield(self):
        """Dividend reduces call price, increases put price."""
        c_no_div = bs.price(100, 100, 1.0, 0.05, 0.25, q=0.0)
        c_div = bs.price(100, 100, 1.0, 0.05, 0.25, q=0.05)
        assert c_div < c_no_div


class TestBSEdgeCases:
    """Edge cases: zero/negative inputs → np.nan, no crashes."""

    def test_zero_spot(self):
        assert np.isnan(bs.price(0, 100, 1.0, 0.05, 0.20))

    def test_negative_spot(self):
        assert np.isnan(bs.price(-10, 100, 1.0, 0.05, 0.20))

    def test_zero_strike(self):
        assert np.isnan(bs.price(100, 0, 1.0, 0.05, 0.20))

    def test_negative_strike(self):
        assert np.isnan(bs.price(100, -50, 1.0, 0.05, 0.20))

    def test_zero_time_expired(self):
        assert np.isnan(bs.price(100, 100, 0, 0.05, 0.20))

    def test_negative_time(self):
        assert np.isnan(bs.price(100, 100, -0.5, 0.05, 0.20))

    def test_zero_vol(self):
        assert np.isnan(bs.price(100, 100, 1.0, 0.05, 0))

    def test_negative_vol(self):
        assert np.isnan(bs.price(100, 100, 1.0, 0.05, -0.20))

    def test_negative_rate_still_works(self):
        """Negative rates are economically valid (EUR/JPY)."""
        p = bs.price(100, 100, 1.0, -0.01, 0.20)
        assert not np.isnan(p)
        assert p > 0

    def test_very_small_time(self):
        """Near-expiry should not crash."""
        p = bs.price(100, 100, 1e-6, 0.05, 0.20)
        assert not np.isnan(p)

    def test_very_high_vol(self):
        """Extreme vol should not crash."""
        p = bs.price(100, 100, 1.0, 0.05, 5.0)
        assert not np.isnan(p)


class TestBSGreeks:
    """Greeks analytical values and boundary conditions."""

    def test_call_delta_range(self):
        d = bs.delta(100, 100, 1.0, 0.05, 0.25)
        assert 0 < d < 1

    def test_put_delta_range(self):
        d = bs.delta(100, 100, 1.0, 0.05, 0.25, option_type="put")
        assert -1 < d < 0

    def test_call_put_delta_relation(self):
        """Call delta - Put delta = e^(-qT)."""
        q = 0.02
        dc = bs.delta(100, 100, 1.0, 0.05, 0.25, q=q)
        dp = bs.delta(100, 100, 1.0, 0.05, 0.25, q=q, option_type="put")
        assert pytest.approx(dc - dp, abs=1e-10) == np.exp(-q * 1.0)

    def test_gamma_same_for_call_put(self):
        gc = bs.gamma(100, 100, 1.0, 0.05, 0.25)
        gp = bs.gamma(100, 100, 1.0, 0.05, 0.25, option_type="put")
        assert pytest.approx(gc, abs=1e-12) == gp

    def test_gamma_positive(self):
        g = bs.gamma(100, 100, 1.0, 0.05, 0.25)
        assert g > 0

    def test_theta_negative_for_long_call(self):
        """Long ATM call loses value over time."""
        th = bs.theta(100, 100, 1.0, 0.05, 0.25)
        assert th < 0

    def test_theta_per_calendar_day(self):
        """Theta must be per calendar day (divided by 365)."""
        th = bs.theta(100, 100, 1.0, 0.05, 0.25)
        # Typical ATM 1yr call theta: roughly -0.01 to -0.05 per day
        assert -0.10 < th < 0

    def test_vega_same_for_call_put(self):
        vc = bs.vega(100, 100, 1.0, 0.05, 0.25)
        vp = bs.vega(100, 100, 1.0, 0.05, 0.25, option_type="put")
        assert pytest.approx(vc, abs=1e-12) == vp

    def test_vega_positive(self):
        v = bs.vega(100, 100, 1.0, 0.05, 0.25)
        assert v > 0

    def test_vega_per_vol_point(self):
        """Vega scaled per 0.01 sigma change."""
        v = bs.vega(100, 100, 1.0, 0.05, 0.25)
        # Numerical check: bump sigma by 0.01, compare price diff
        p1 = bs.price(100, 100, 1.0, 0.05, 0.25)
        p2 = bs.price(100, 100, 1.0, 0.05, 0.26)
        assert pytest.approx(v, rel=0.01) == p2 - p1

    def test_call_rho_positive(self):
        r = bs.rho(100, 100, 1.0, 0.05, 0.25)
        assert r > 0

    def test_put_rho_negative(self):
        r = bs.rho(100, 100, 1.0, 0.05, 0.25, option_type="put")
        assert r < 0

    def test_greeks_nan_on_zero_spot(self):
        assert np.isnan(bs.delta(0, 100, 1.0, 0.05, 0.25))
        assert np.isnan(bs.gamma(0, 100, 1.0, 0.05, 0.25))
        assert np.isnan(bs.theta(0, 100, 1.0, 0.05, 0.25))
        assert np.isnan(bs.vega(0, 100, 1.0, 0.05, 0.25))
        assert np.isnan(bs.rho(0, 100, 1.0, 0.05, 0.25))

    def test_greeks_nan_on_expired(self):
        assert np.isnan(bs.delta(100, 100, 0, 0.05, 0.25))

    def test_all_greeks_returns_8_keys(self):
        g = bs.all_greeks(100, 100, 1.0, 0.05, 0.25)
        expected = {"delta", "gamma", "theta", "vega", "rho",
                    "vanna", "volga", "charm"}
        assert set(g.keys()) == expected


class TestHigherOrderGreeks:
    """Higher-order Greeks: Vanna, Volga, Charm."""

    def test_vanna_nan_on_invalid(self):
        assert np.isnan(bs.vanna(0, 100, 1.0, 0.05, 0.25))

    def test_volga_nan_on_invalid(self):
        assert np.isnan(bs.volga(0, 100, 1.0, 0.05, 0.25))

    def test_charm_nan_on_invalid(self):
        assert np.isnan(bs.charm(0, 100, 1.0, 0.05, 0.25))

    def test_vanna_finite(self):
        v = bs.vanna(100, 100, 1.0, 0.05, 0.25)
        assert np.isfinite(v)

    def test_volga_finite(self):
        v = bs.volga(100, 100, 1.0, 0.05, 0.25)
        assert np.isfinite(v)

    def test_charm_finite(self):
        c = bs.charm(100, 100, 1.0, 0.05, 0.25)
        assert np.isfinite(c)

    def test_charm_per_calendar_day(self):
        """Charm is delta change per calendar day — small magnitude."""
        c = bs.charm(100, 100, 1.0, 0.05, 0.25)
        assert abs(c) < 0.01  # Should be very small per day
