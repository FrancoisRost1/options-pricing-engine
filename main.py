"""
Options Pricing Engine — main orchestration module.

Entry point for CLI usage. Loads config, fetches market data,
runs pricing models, and produces summary output. The Streamlit
dashboard (app/app.py) is the primary interface; this module
enables headless execution and scripting.

Usage:
    python3 main.py                  # default ticker from config
    python3 main.py --ticker AAPL    # specific ticker
    python3 main.py --synthetic      # use synthetic data
"""

import argparse
import sys
import numpy as np

from utils.config_loader import load_config
from src import data_loader
from src import synthetic_data
from src import chain_filter
from src import black_scholes as bs
from src import binomial_tree
from src import monte_carlo
from src import implied_vol
from src import model_comparison


def run(ticker: str = None, use_synthetic: bool = False):
    """
    Run the full pricing pipeline for a ticker.

    Steps:
      1. Load config
      2. Fetch market data (or generate synthetic)
      3. Filter chain
      4. Extract implied volatilities
      5. Price a representative ATM option with all 3 models
      6. Print summary
    """
    config = load_config()

    if ticker is None:
        ticker = config.get("data", {}).get("default_ticker", "AAPL")

    # Load data
    if use_synthetic:
        market = synthetic_data.generate_chain(config)
        print(f"Using synthetic data (S={market['spot']:.2f})")
    else:
        print(f"Fetching market data for {ticker}...")
        market = data_loader.load_market_data(ticker, config)
        if market["chain"].empty:
            fallback = config.get("data", {}).get("use_synthetic_fallback", True)
            if fallback:
                print("yfinance returned empty chain — falling back to synthetic data")
                market = synthetic_data.generate_chain(config)
            else:
                print("No data available and synthetic fallback disabled.")
                return

    S = market["spot"]
    r = market["risk_free_rate"]
    q = market["dividend_yield"]
    chain = market["chain"]

    print(f"\nTicker: {market['ticker']}")
    print(f"Spot: ${S:.2f}")
    print(f"Risk-free rate: {r:.4f}")
    print(f"Dividend yield: {q:.4f}")
    print(f"Raw chain: {len(chain)} contracts")

    # Filter chain
    filtered = chain_filter.apply_filters(chain, config)
    summary = chain_filter.filter_summary(chain, filtered)
    print(f"Filtered chain: {summary['filtered_contracts']} contracts "
          f"({summary['removed']} removed)")

    # Extract implied volatilities
    if not filtered.empty:
        filtered = implied_vol.extract_chain(filtered, S, r, q, config)
        valid_iv = filtered["iv"].dropna()
        print(f"Valid IVs extracted: {len(valid_iv)}")

        if not valid_iv.empty:
            print(f"IV range: {valid_iv.min():.2%} — {valid_iv.max():.2%}")

    # Price a representative ATM option
    if not filtered.empty:
        # Find nearest ATM call
        calls = filtered[filtered["option_type"] == "call"].copy()
        if not calls.empty:
            calls["dist"] = np.abs(calls["strike"] - S)
            atm = calls.loc[calls["dist"].idxmin()]
            K = atm["strike"]
            T = atm["T"]
            sigma = atm.get("iv", 0.25)
            if np.isnan(sigma):
                sigma = 0.25

            print(f"\n--- ATM Call (K={K:.2f}, T={T:.3f}y) ---")

            comp = model_comparison.compare_models(S, K, T, r, sigma, q,
                                                   "call", config)
            print(f"Black-Scholes:  ${comp['bs_price']:.4f}")
            print(f"Binomial (CRR): ${comp['binomial_price']:.4f}")
            print(f"Monte Carlo:    ${comp['mc_price']:.4f} "
                  f"(SE: {comp['mc_std_error']:.4f})")
            print(f"Max deviation:  ${comp['max_deviation']:.4f}")

            # Greeks
            greeks = bs.all_greeks(S, K, T, r, sigma, q, "call")
            print(f"\nGreeks (BS analytical):")
            print(f"  Delta: {greeks['delta']:.4f}")
            print(f"  Gamma: {greeks['gamma']:.4f}")
            print(f"  Theta: {greeks['theta']:.4f} (per day)")
            print(f"  Vega:  {greeks['vega']:.4f} (per vol pt)")
            print(f"  Rho:   {greeks['rho']:.4f} (per 1% rate)")

    print("\nDone. Launch dashboard with: streamlit run app/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options Pricing Engine")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Ticker symbol (default: from config)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of yfinance")
    args = parser.parse_args()

    run(ticker=args.ticker, use_synthetic=args.synthetic)
