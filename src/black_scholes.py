"""
Black-Scholes analytical pricing and Greeks.

Implements the closed-form Black-Scholes-Merton model for European
calls and puts with continuous dividend yield. All standard Greeks
(Delta, Gamma, Theta, Vega, Rho) plus higher-order Greeks (Vanna,
Volga/Vomma, Charm) are computed analytically.

Assumes log-normal returns (GBM), no jumps, no stochastic vol.
"""

import numpy as np
from scipy.stats import norm


def _validate_inputs(S, K, T, r, sigma, q=0.0):
    """
    Validate pricing inputs. Returns np.nan for degenerate cases.

    Handles: zero/negative spot, strike, time, or vol gracefully
    so callers never see division-by-zero errors.
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return False
    return True


def _d1(S, K, T, r, sigma, q=0.0):
    """Compute d1 = [ln(S/K) + (r - q + sigma^2/2)*T] / (sigma*sqrt(T))."""
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, r, sigma, q=0.0):
    """Compute d2 = d1 - sigma*sqrt(T)."""
    return _d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def price(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Black-Scholes price for a European option.

    The model assumes continuous dividend yield q and constant
    volatility sigma over the life of the option.

    Args:
        S: Spot price of the underlying.
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free rate (continuous, annualized).
        sigma: Volatility (annualized).
        q: Continuous dividend yield (annualized).
        option_type: 'call' or 'put'.

    Returns:
        Option price (float). Returns np.nan for invalid inputs.
    """
    if not _validate_inputs(S, K, T, r, sigma, q):
        return np.nan

    d1 = _d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def delta(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Delta: sensitivity of option price to $1 move in spot.

    Call delta is in [0, 1], put delta in [-1, 0]. Near-ATM
    options have delta ~0.5 (call) or ~-0.5 (put), reflecting
    roughly equal probability of finishing ITM or OTM.
    """
    if not _validate_inputs(S, K, T, r, sigma, q):
        return np.nan

    d1 = _d1(S, K, T, r, sigma, q)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return np.exp(-q * T) * (norm.cdf(d1) - 1.0)


def gamma(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Gamma: rate of change of delta per $1 move in spot.

    Gamma is identical for calls and puts (put-call parity).
    Peaks near ATM and near expiry, this is why short-dated
    ATM options are hardest to delta-hedge.
    """
    if not _validate_inputs(S, K, T, r, sigma, q):
        return np.nan

    d1 = _d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def theta(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Theta: time decay per CALENDAR day (annualized derivative / 365).

    Theta is almost always negative for long options, the option
    loses value as time passes (all else equal). This is the
    'price' the option buyer pays for convexity (gamma).
    """
    if not _validate_inputs(S, K, T, r, sigma, q):
        return np.nan

    d1 = _d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(T)

    # First term: vol decay (always negative for longs)
    term1 = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(T))

    if option_type == "call":
        term2 = q * S * np.exp(-q * T) * norm.cdf(d1)
        term3 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        term2 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
        term3 = r * K * np.exp(-r * T) * norm.cdf(-d2)

    # Convert from per-year to per calendar day
    return (term1 + term2 + term3) / 365.0


def vega(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Vega: sensitivity to +1 vol point (+0.01 in sigma).

    Vega is identical for calls and puts. Peaks near ATM and
    increases with time to expiry, longer-dated options are
    more sensitive to vol changes.
    """
    if not _validate_inputs(S, K, T, r, sigma, q):
        return np.nan

    d1 = _d1(S, K, T, r, sigma, q)
    # Raw vega is per unit sigma; multiply by 0.01 for per vol point
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) * 0.01


def rho(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Rho: sensitivity to +1% rate move (+0.01 in r).

    Calls have positive rho (benefit from rate increases since
    the PV of the strike decreases). Puts have negative rho.
    """
    if not _validate_inputs(S, K, T, r, sigma, q):
        return np.nan

    d2 = _d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01


def vanna(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Vanna: cross-Greek dDelta/dSigma = dVega/dS.

    Measures how delta changes when vol moves. Important for
    understanding how hedging costs change in a vol event.
    Analytical BS only, unreliable via finite difference.
    """
    if not _validate_inputs(S, K, T, r, sigma, q):
        return np.nan

    d1 = _d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(T)
    # Per +1 vol point (0.01 sigma)
    return -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma * 0.01


def volga(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Volga (Vomma): dVega/dSigma = d²V/dSigma².

    Measures convexity of option price with respect to vol.
    Positive volga means the option benefits from vol-of-vol.
    Analytical BS only, unreliable via finite difference.
    """
    if not _validate_inputs(S, K, T, r, sigma, q):
        return np.nan

    d1 = _d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(T)
    raw_vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    # Per (vol point)^2: raw_vega * d1 * d2 / sigma, scaled by 0.01^2
    return raw_vega * d1 * d2 / sigma * 0.0001


def charm(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Charm: dDelta/dT, how delta changes as time passes.

    Critical for understanding overnight delta drift. A hedger
    who rebalances daily needs to know how much their delta
    shifts just from the passage of one day.
    Analytical BS only, unreliable via finite difference.
    """
    if not _validate_inputs(S, K, T, r, sigma, q):
        return np.nan

    d1 = _d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(T)

    pdf_d1 = norm.pdf(d1)
    term = 2.0 * (r - q) * T - d2 * sigma * np.sqrt(T)

    if option_type == "call":
        result = -q * np.exp(-q * T) * norm.cdf(d1) - np.exp(-q * T) * pdf_d1 * term / (2.0 * T * sigma * np.sqrt(T))
    else:
        result = q * np.exp(-q * T) * norm.cdf(-d1) - np.exp(-q * T) * pdf_d1 * term / (2.0 * T * sigma * np.sqrt(T))

    # Convert from per-year to per calendar day
    return result / 365.0


def all_greeks(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Compute all analytical Greeks in a single call.

    Returns dict with standard Greeks (Delta, Gamma, Theta, Vega, Rho)
    plus higher-order Greeks (Vanna, Volga, Charm). This is more
    efficient than calling each function separately since d1/d2
    are computed once, but we keep individual functions for clarity.
    """
    return {
        "delta": delta(S, K, T, r, sigma, q, option_type),
        "gamma": gamma(S, K, T, r, sigma, q, option_type),
        "theta": theta(S, K, T, r, sigma, q, option_type),
        "vega": vega(S, K, T, r, sigma, q, option_type),
        "rho": rho(S, K, T, r, sigma, q, option_type),
        "vanna": vanna(S, K, T, r, sigma, q, option_type),
        "volga": volga(S, K, T, r, sigma, q, option_type),
        "charm": charm(S, K, T, r, sigma, q, option_type),
    }
