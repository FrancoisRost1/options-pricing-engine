"""
Finite difference Greeks wrapper for Binomial and Monte Carlo models.

Computes standard Greeks (Delta, Gamma, Theta, Vega, Rho) via central
finite difference. Higher-order Greeks (Vanna, Volga, Charm) are NOT
computed here, numerical noise makes them unreliable through
finite difference. Use Black-Scholes analytical for those.

Central difference formula: (V(x+h) - V(x-h)) / (2h)
Bump sizes are configurable in config.yaml.
"""

import numpy as np
from src import binomial_tree
from src import monte_carlo


def _get_pricer(model):
    """
    Return the pricing function for the specified model.

    Only binomial and MC use finite difference Greeks.
    Black-Scholes Greeks are analytical, use black_scholes module directly.
    """
    if model == "binomial":
        return _price_binomial
    elif model == "monte_carlo":
        return _price_mc
    else:
        raise ValueError(f"Unknown model '{model}'. Use 'binomial' or 'monte_carlo'.")


def _price_binomial(S, K, T, r, sigma, q, option_type, exercise, config):
    """Wrapper for binomial pricing with config-driven steps."""
    steps = config.get("binomial_tree", {}).get("default_steps", 200)
    return binomial_tree.price(S, K, T, r, sigma, q, option_type, exercise, steps)


def _price_mc(S, K, T, r, sigma, q, option_type, exercise, config):
    """Wrapper for MC pricing with config-driven parameters.

    MC supports European options only. American exercise raises
    ValueError, use binomial model for American Greeks.
    """
    if exercise == "american":
        raise ValueError(
            "MC Greeks do not support American exercise. "
            "Use model='binomial' for American options."
        )
    mc_cfg = config.get("monte_carlo", {})
    result = monte_carlo.price(
        S, K, T, r, sigma, q, option_type,
        n_paths=mc_cfg.get("paths", 100000),
        n_steps=mc_cfg.get("time_steps", 252),
        seed=mc_cfg.get("seed", 42),
        use_antithetic=mc_cfg.get("use_antithetic", True),
        use_control_variate=mc_cfg.get("use_control_variate", True),
        cv_beta_mode=mc_cfg.get("control_variate_beta_mode", "estimate"),
    )
    return result["price"]


def compute(S, K, T, r, sigma, q=0.0, option_type="call",
            exercise="european", model="binomial", config=None):
    """
    Compute standard Greeks via central finite difference.

    Returns Delta, Gamma, Theta, Vega, Rho using bump-and-reprice.
    Higher-order Greeks are intentionally excluded, they require
    second-order finite differences that amplify numerical noise
    in tree and MC models.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free rate (continuous).
        sigma: Volatility (annualized).
        q: Continuous dividend yield.
        option_type: 'call' or 'put'.
        exercise: 'european' or 'american'.
        model: 'binomial' or 'monte_carlo'.
        config: Configuration dict.

    Returns:
        Dict with delta, gamma, theta, vega, rho.
    """
    if config is None:
        config = {}

    greeks_cfg = config.get("greeks", {})
    bump_spot_pct = greeks_cfg.get("bump_spot_pct", 0.01)
    bump_vol = greeks_cfg.get("bump_vol", 0.01)
    bump_rate = greeks_cfg.get("bump_rate", 0.01)
    bump_time_days = greeks_cfg.get("bump_time_days", 1)

    pricer = _get_pricer(model)
    base_price = pricer(S, K, T, r, sigma, q, option_type, exercise, config)

    h_S = S * bump_spot_pct
    h_T = bump_time_days / 365.0

    # Delta: dV/dS
    p_up = pricer(S + h_S, K, T, r, sigma, q, option_type, exercise, config)
    p_dn = pricer(S - h_S, K, T, r, sigma, q, option_type, exercise, config)
    delta_val = (p_up - p_dn) / (2.0 * h_S)

    # Gamma: d²V/dS²
    gamma_val = (p_up - 2.0 * base_price + p_dn) / (h_S ** 2)

    # Vega: dV/dSigma (per +1 vol point = +0.01)
    p_vol_up = pricer(S, K, T, r, sigma + bump_vol, q, option_type, exercise, config)
    p_vol_dn = pricer(S, K, T, r, max(sigma - bump_vol, 0.001), q,
                      option_type, exercise, config)
    actual_vol_bump = (sigma + bump_vol) - max(sigma - bump_vol, 0.001)
    vega_val = (p_vol_up - p_vol_dn) / actual_vol_bump * 0.01

    # Theta: dV/dT (per calendar day)
    # Reduce T (time passes = T decreases)
    T_fwd = max(T - h_T, 1e-6)
    T_bwd = T + h_T
    p_t_fwd = pricer(S, K, T_fwd, r, sigma, q, option_type, exercise, config)
    p_t_bwd = pricer(S, K, T_bwd, r, sigma, q, option_type, exercise, config)
    # Theta is negative when time passing reduces value
    theta_val = (p_t_fwd - p_t_bwd) / (2.0 * h_T) / 365.0

    # Rho: dV/dr (per +1% = +0.01)
    p_r_up = pricer(S, K, T, r + bump_rate, sigma, q, option_type, exercise, config)
    p_r_dn = pricer(S, K, T, r - bump_rate, sigma, q, option_type, exercise, config)
    rho_val = (p_r_up - p_r_dn) / (2.0 * bump_rate) * 0.01

    return {
        "delta": float(delta_val),
        "gamma": float(gamma_val),
        "theta": float(theta_val),
        "vega": float(vega_val),
        "rho": float(rho_val),
    }
