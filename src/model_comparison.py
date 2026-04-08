"""
Cross-model price comparison and convergence analysis.

Compares BS, Binomial, and MC prices for the same contract to
verify model agreement. For European vanilla options, all three
should converge to the same value — disagreements signal bugs
or insufficient steps/paths.

Also provides convergence analysis: binomial price vs steps,
MC price vs paths — demonstrating theoretical convergence
properties that are interview gold.
"""

import numpy as np
import pandas as pd
from src import black_scholes as bs
from src import binomial_tree
from src import monte_carlo


def compare_models(S, K, T, r, sigma, q=0.0, option_type="call",
                   config=None):
    """
    Price a single option with all three models and compare.

    Expected: near-perfect agreement for European options.
    Any significant deviation points to parameterization issues
    or insufficient simulation parameters.

    Args:
        S, K, T, r, sigma, q, option_type: Standard option params.
        config: Configuration dict for model parameters.

    Returns:
        Dict with 'bs_price', 'binomial_price', 'mc_price',
        'mc_std_error', 'mc_ci', 'max_deviation'.
    """
    if config is None:
        config = {}

    bs_price = bs.price(S, K, T, r, sigma, q, option_type)

    bt_cfg = config.get("binomial_tree", {})
    steps = bt_cfg.get("default_steps", 200)
    bt_price = binomial_tree.price(S, K, T, r, sigma, q, option_type,
                                   exercise="european", steps=steps)

    mc_cfg = config.get("monte_carlo", {})
    mc_result = monte_carlo.price(
        S, K, T, r, sigma, q, option_type,
        n_paths=mc_cfg.get("paths", 100000),
        n_steps=mc_cfg.get("time_steps", 252),
        seed=mc_cfg.get("seed", 42),
        use_antithetic=mc_cfg.get("use_antithetic", True),
        use_control_variate=mc_cfg.get("use_control_variate", True),
        cv_beta_mode=mc_cfg.get("control_variate_beta_mode", "estimate"),
    )

    prices = [p for p in [bs_price, bt_price, mc_result["price"]]
              if not np.isnan(p)]
    max_dev = (max(prices) - min(prices)) if len(prices) > 1 else 0.0

    return {
        "bs_price": bs_price,
        "binomial_price": bt_price,
        "mc_price": mc_result["price"],
        "mc_std_error": mc_result["std_error"],
        "mc_ci_lower": mc_result["ci_lower"],
        "mc_ci_upper": mc_result["ci_upper"],
        "mc_cv_beta": mc_result.get("cv_beta_used"),
        "max_deviation": max_dev,
    }


def compare_chain(chain_df, S, r, q, sigma_col="iv", config=None):
    """
    Compare model prices across an entire chain.

    Uses each contract's implied vol as the input sigma. This
    shows model agreement across the strike/expiry grid.

    Returns:
        DataFrame with model prices and deviations added.
    """
    if config is None:
        config = {}

    df = chain_df.dropna(subset=[sigma_col]).copy()
    results = []

    for _, row in df.iterrows():
        comp = compare_models(
            S=S, K=row["strike"], T=row["T"], r=r,
            sigma=row[sigma_col], q=q,
            option_type=row["option_type"], config=config,
        )
        results.append(comp)

    if not results:
        return df

    result_df = pd.DataFrame(results)
    for col in result_df.columns:
        df[col] = result_df[col].values

    return df


def binomial_convergence(S, K, T, r, sigma, q=0.0, option_type="call",
                         config=None):
    """
    Binomial price convergence as function of tree steps.

    Shows the characteristic CRR odd/even oscillation dampening
    as steps increase, converging to the BS analytical price.

    Returns:
        DataFrame with 'steps', 'price', 'bs_price', 'error'.
    """
    if config is None:
        config = {}

    steps_list = config.get("binomial_tree", {}).get(
        "convergence_steps", [10, 25, 50, 100, 200, 500, 1000]
    )

    bs_price = bs.price(S, K, T, r, sigma, q, option_type)
    results = binomial_tree.price_convergence(
        S, K, T, r, sigma, q, option_type, "european", steps_list
    )

    rows = [{"steps": n, "price": p, "bs_price": bs_price,
             "error": p - bs_price} for n, p in results]
    return pd.DataFrame(rows)


def mc_convergence(S, K, T, r, sigma, q=0.0, option_type="call",
                   config=None):
    """
    Monte Carlo price convergence as function of path count.

    Shows how the estimate and confidence interval tighten
    with more paths. Useful for demonstrating variance reduction.

    Returns:
        DataFrame with 'n_paths', 'price', 'ci_lower', 'ci_upper',
        'bs_price', 'error'.
    """
    if config is None:
        config = {}

    mc_cfg = config.get("monte_carlo", {})
    paths_list = mc_cfg.get("convergence_paths", [1000, 5000, 10000, 50000, 100000])

    bs_price = bs.price(S, K, T, r, sigma, q, option_type)
    results = monte_carlo.price_convergence(
        S, K, T, r, sigma, q, option_type,
        paths_list=paths_list,
        n_steps=mc_cfg.get("time_steps", 252),
        seed=mc_cfg.get("seed", 42),
        use_antithetic=mc_cfg.get("use_antithetic", True),
        use_control_variate=mc_cfg.get("use_control_variate", True),
        cv_beta_mode=mc_cfg.get("control_variate_beta_mode", "estimate"),
    )

    rows = []
    for r_dict in results:
        rows.append({
            "n_paths": r_dict["n_paths"],
            "price": r_dict["price"],
            "ci_lower": r_dict["ci_lower"],
            "ci_upper": r_dict["ci_upper"],
            "bs_price": bs_price,
            "error": r_dict["price"] - bs_price,
        })
    return pd.DataFrame(rows)
