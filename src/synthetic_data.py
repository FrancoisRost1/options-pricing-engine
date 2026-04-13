"""
Synthetic options chain generator (GBM-based).

Produces realistic-looking options chains when yfinance is
unavailable or for testing. Prices are generated using
Black-Scholes with configurable skew, then bid/ask spreads
are added with random noise.

This is a FALLBACK data source, clearly labeled as synthetic
in all downstream outputs.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src import black_scholes as bs


def generate_chain(config: dict) -> dict:
    """
    Generate a complete synthetic options chain.

    Creates a grid of strikes × expiries, prices each contract
    with Black-Scholes (adding vol skew), then simulates bid/ask
    spreads with configurable noise.

    Args:
        config: Configuration dict with 'synthetic' section.

    Returns:
        Dict with keys: 'ticker', 'spot', 'chain' (DataFrame),
        'risk_free_rate', 'dividend_yield', 'data_source'.
    """
    syn = config.get("synthetic", {})
    S = syn.get("spot_price", 100.0)
    base_vol = syn.get("base_vol", 0.25)
    skew_slope = syn.get("skew_slope", -0.10)
    expiry_days = syn.get("expiry_days", [7, 14, 30, 60, 90, 180])
    strikes_per = syn.get("strikes_per_expiry", 21)
    spacing = syn.get("strike_spacing_pct", 0.025)
    r = syn.get("risk_free_rate", 0.045)
    q = syn.get("dividend_yield", 0.015)
    noise_pct = syn.get("bid_ask_noise_pct", 0.02)

    rng = np.random.default_rng(42)
    today = datetime.now()
    rows = []

    for dte in expiry_days:
        T = dte / 365.0
        exp_date = today + timedelta(days=dte)

        # Generate strikes centered on ATM
        half = strikes_per // 2
        strikes = [S * (1.0 + spacing * (i - half)) for i in range(strikes_per)]

        for K in strikes:
            log_m = np.log(K / S)
            # Vol with equity skew: lower strikes → higher vol
            sigma = max(base_vol + skew_slope * log_m, 0.05)

            for opt_type in ["call", "put"]:
                theo = bs.price(S, K, T, r, sigma, q, opt_type)
                if np.isnan(theo) or theo <= 0:
                    continue

                # Simulate bid/ask around theoretical price
                spread = max(theo * noise_pct, 0.01)
                noise = rng.uniform(-0.5, 0.5) * spread * 0.5
                bid = max(theo - spread / 2.0 + noise, 0.01)
                ask = bid + spread
                mid = (bid + ask) / 2.0

                # Synthetic volume/OI, higher for ATM, lower for wings
                moneyness_penalty = np.exp(-5.0 * log_m ** 2)
                volume = max(int(rng.poisson(500 * moneyness_penalty)), 0)
                oi = max(int(rng.poisson(2000 * moneyness_penalty)), 0)

                rows.append({
                    "strike": K,
                    "bid": round(bid, 2),
                    "ask": round(ask, 2),
                    "mid": round(mid, 2),
                    "lastPrice": round(theo, 2),
                    "volume": volume,
                    "openInterest": oi,
                    "option_type": opt_type,
                    "expiry": pd.Timestamp(exp_date.date()),
                    "T": T,
                    "dte": dte,
                    "price_source": "mid",
                    "log_moneyness": log_m,
                    "impliedVolatility": sigma,
                })

    chain = pd.DataFrame(rows)

    return {
        "ticker": "SYNTHETIC",
        "spot": S,
        "chain": chain,
        "risk_free_rate": r,
        "dividend_yield": q,
        "data_source": "synthetic",
    }
