"""
Cox-Ross-Rubinstein (CRR) binomial tree pricer.

Prices European and American calls and puts via backward induction
on a recombining binomial lattice. American options check for early
exercise at every node, this is the only model in the engine that
handles American exercise.

CRR parameterization ensures the tree recombines (u*d = 1) and
converges to Black-Scholes as the number of steps increases.
"""

import numpy as np


def price(S, K, T, r, sigma, q=0.0, option_type="call",
          exercise="european", steps=200):
    """
    Price an option using the CRR binomial tree.

    The tree discretizes time into N steps. At each node the
    underlying can move up by factor u or down by d = 1/u.
    Terminal payoffs are discounted back through the tree using
    risk-neutral probabilities.

    For American options, at each node we compare the continuation
    value (discounted expected value) with the immediate exercise
    value and take the maximum, this captures the early exercise
    premium that makes American options worth more than European.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free rate (continuous).
        sigma: Volatility (annualized).
        q: Continuous dividend yield.
        option_type: 'call' or 'put'.
        exercise: 'european' or 'american'.
        steps: Number of tree time steps.

    Returns:
        Option price (float). Returns np.nan for invalid inputs.
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0 or steps < 1:
        return np.nan

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)

    # Risk-neutral probability must be in (0, 1)
    if p <= 0 or p >= 1:
        return np.nan

    discount = np.exp(-r * dt)

    # Build terminal asset prices using vectorized approach
    # At step N, asset price at node j is S * u^j * d^(N-j)
    j = np.arange(steps + 1)
    asset_prices = S * (u ** (steps - j)) * (d ** j)

    # Terminal payoffs
    if option_type == "call":
        option_values = np.maximum(asset_prices - K, 0.0)
    else:
        option_values = np.maximum(K - asset_prices, 0.0)

    # Backward induction
    for i in range(steps - 1, -1, -1):
        # Asset prices at step i
        asset_at_i = S * (u ** (i - j[:i + 1])) * (d ** j[:i + 1])

        # Continuation value
        option_values = discount * (p * option_values[:-1] + (1.0 - p) * option_values[1:])

        # Early exercise check for American options
        if exercise == "american":
            if option_type == "call":
                exercise_value = np.maximum(asset_at_i - K, 0.0)
            else:
                exercise_value = np.maximum(K - asset_at_i, 0.0)
            option_values = np.maximum(option_values, exercise_value)

    return float(option_values[0])


def price_convergence(S, K, T, r, sigma, q=0.0, option_type="call",
                      exercise="european", steps_list=None):
    """
    Compute option price for varying number of tree steps.

    Used to demonstrate convergence of the binomial price to the
    Black-Scholes analytical solution. The CRR tree exhibits
    characteristic odd/even oscillation that dampens as N grows.

    Args:
        steps_list: List of step counts to evaluate.
            Defaults to [10, 25, 50, 100, 200, 500, 1000].

    Returns:
        List of (steps, price) tuples.
    """
    if steps_list is None:
        steps_list = [10, 25, 50, 100, 200, 500, 1000]

    results = []
    for n in steps_list:
        p = price(S, K, T, r, sigma, q, option_type, exercise, steps=n)
        results.append((n, p))
    return results
