"""
Monte Carlo option pricer with variance reduction.

Simulates GBM paths to price European options. Includes two
variance reduction techniques:
  1. Antithetic variates, for each random draw Z, also simulate -Z
  2. Control variate, use BS analytical price as control to reduce
     estimation error

MC is not needed for vanilla European options (BS is exact). It is
implemented as a foundation for path-dependent extensions and to
demonstrate variance reduction techniques.

European options only in v1, no Longstaff-Schwartz for American.
"""

import numpy as np
from src import black_scholes as bs


def _simulate_paths(S, T, r, sigma, q, n_paths, n_steps, seed,
                    use_antithetic=False):
    """
    Simulate GBM price paths.

    Uses exact log-normal discretization:
      S(t+dt) = S(t) * exp((r - q - sigma^2/2)*dt + sigma*sqrt(dt)*Z)

    With antithetic variates, each Z is paired with -Z, doubling
    the effective number of paths while maintaining the same
    number of random draws. This reduces variance because the
    positive and negative shocks partially cancel estimation error.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    if use_antithetic:
        half = n_paths // 2
        Z = rng.standard_normal((half, n_steps))
        Z_full = np.concatenate([Z, -Z], axis=0)
    else:
        Z_full = rng.standard_normal((n_paths, n_steps))

    # Build paths using cumulative sum of log returns
    log_returns = drift + diffusion * Z_full
    log_paths = np.cumsum(log_returns, axis=1)
    # Prepend zero for initial time
    log_paths = np.concatenate(
        [np.zeros((log_paths.shape[0], 1)), log_paths], axis=1
    )
    paths = S * np.exp(log_paths)
    return paths


def price(S, K, T, r, sigma, q=0.0, option_type="call",
          n_paths=100000, n_steps=252, seed=42,
          use_antithetic=True, use_control_variate=True,
          cv_beta_mode="estimate", cv_beta_fixed=1.0):
    """
    Monte Carlo price for a European option.

    The price is the discounted expected payoff under the
    risk-neutral measure. Variance reduction narrows the
    confidence interval for the same number of paths.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free rate (continuous).
        sigma: Volatility (annualized).
        q: Continuous dividend yield.
        option_type: 'call' or 'put'.
        n_paths: Number of simulation paths.
        n_steps: Time steps per path (daily = 252).
        seed: Random seed for reproducibility.
        use_antithetic: Enable antithetic variates.
        use_control_variate: Enable control variate (BS as control).
        cv_beta_mode: 'fixed' or 'estimate' for control variate beta.
        cv_beta_fixed: Beta value when mode is 'fixed'.

    Returns:
        Dict with 'price', 'std_error', 'ci_lower', 'ci_upper',
        'cv_beta_mode', 'cv_beta_used'.
        Returns dict with np.nan price for invalid inputs.
    """
    nan_result = {
        "price": np.nan, "std_error": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan,
        "cv_beta_mode": cv_beta_mode, "cv_beta_used": np.nan,
    }

    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return nan_result

    paths = _simulate_paths(S, T, r, sigma, q, n_paths, n_steps,
                            seed, use_antithetic)
    S_T = paths[:, -1]
    actual_paths = S_T.shape[0]

    # Terminal payoffs
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    discount = np.exp(-r * T)
    discounted_payoffs = discount * payoffs

    beta_used = np.nan

    # Control variate adjustment
    if use_control_variate:
        bs_price = bs.price(S, K, T, r, sigma, q, option_type)
        # Control variate payoffs (using same terminal prices)
        if option_type == "call":
            cv_payoffs = discount * np.maximum(S_T - K, 0.0)
        else:
            cv_payoffs = discount * np.maximum(K - S_T, 0.0)

        if cv_beta_mode == "fixed":
            beta_used = cv_beta_fixed
        else:
            # Estimate beta from the same paths
            cov = np.cov(discounted_payoffs, cv_payoffs)[0, 1]
            var_cv = np.var(cv_payoffs, ddof=1)
            beta_used = cov / var_cv if var_cv > 1e-15 else 1.0

        discounted_payoffs = discounted_payoffs - beta_used * (cv_payoffs - bs_price)

    mc_price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(actual_paths)

    return {
        "price": float(mc_price),
        "std_error": float(std_error),
        "ci_lower": float(mc_price - 1.96 * std_error),
        "ci_upper": float(mc_price + 1.96 * std_error),
        "cv_beta_mode": cv_beta_mode,
        "cv_beta_used": float(beta_used) if not np.isnan(beta_used) else None,
    }


def price_convergence(S, K, T, r, sigma, q=0.0, option_type="call",
                      paths_list=None, n_steps=252, seed=42,
                      use_antithetic=True, use_control_variate=True,
                      cv_beta_mode="estimate"):
    """
    Compute MC price and CI for varying path counts.

    Shows how the estimate tightens with more paths, and how
    variance reduction techniques narrow the CI faster.

    Args:
        paths_list: List of path counts. Defaults to config values.

    Returns:
        List of dicts with 'n_paths', 'price', 'ci_lower', 'ci_upper'.
    """
    if paths_list is None:
        paths_list = [1000, 5000, 10000, 50000, 100000]

    results = []
    for n in paths_list:
        result = price(S, K, T, r, sigma, q, option_type,
                       n_paths=n, n_steps=n_steps, seed=seed,
                       use_antithetic=use_antithetic,
                       use_control_variate=use_control_variate,
                       cv_beta_mode=cv_beta_mode)
        result["n_paths"] = n
        results.append(result)
    return results


def price_variance_comparison(S, K, T, r, sigma, q=0.0,
                              option_type="call", n_paths=100000,
                              n_steps=252, seed=42):
    """
    Compare standard MC vs antithetic vs control variate.

    Demonstrates the variance reduction effect, same number
    of paths but progressively tighter confidence intervals.

    Returns:
        Dict with keys 'standard', 'antithetic', 'control_variate',
        'both', each containing the price result dict.
    """
    standard = price(S, K, T, r, sigma, q, option_type,
                     n_paths, n_steps, seed,
                     use_antithetic=False, use_control_variate=False)
    antithetic = price(S, K, T, r, sigma, q, option_type,
                       n_paths, n_steps, seed,
                       use_antithetic=True, use_control_variate=False)
    control = price(S, K, T, r, sigma, q, option_type,
                    n_paths, n_steps, seed,
                    use_antithetic=False, use_control_variate=True)
    both = price(S, K, T, r, sigma, q, option_type,
                 n_paths, n_steps, seed,
                 use_antithetic=True, use_control_variate=True)

    return {
        "standard": standard,
        "antithetic": antithetic,
        "control_variate": control,
        "both": both,
    }
