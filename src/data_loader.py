"""
Data loader — fetches options chains, spot prices, risk-free rates,
and dividend yields from yfinance.

Handles the full data pipeline: fetch → validate → compute mid price →
tag price source. Falls back to synthetic data when yfinance is
unavailable (if configured).

Key conventions:
  - Mid price = (bid + ask) / 2 when both are valid and non-zero
  - Risk-free rate: yfinance returns PERCENTAGE values → divide by 100
  - Dividend yield: trailing annual dividend / current price
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime


def fetch_spot(ticker: str) -> float:
    """
    Fetch current spot price for a ticker.

    Uses yfinance fast_info for speed. Falls back to history
    if fast_info is unavailable.
    """
    tk = yf.Ticker(ticker)
    try:
        price = tk.fast_info.get("lastPrice")
        if price and price > 0:
            return float(price)
    except Exception:
        pass

    hist = tk.history(period="1d")
    if hist.empty:
        return np.nan
    return float(hist["Close"].iloc[-1])


def fetch_risk_free_rate(dte: int, config: dict) -> float:
    """
    Fetch risk-free rate based on days to expiry.

    Short-dated options (DTE < threshold) use 13-week T-bill (^IRX).
    Longer-dated options use 10-year Treasury (^TNX).

    CRITICAL: yfinance returns rates as percentages (e.g., 4.5 for 4.5%).
    Must divide by 100 before use as a decimal rate.

    Simplifying assumption: flat yield curve — we pick one tenor
    based on DTE bucket, not a full interpolated term structure.
    """
    rf_cfg = config.get("risk_free_rate", {})
    source = rf_cfg.get("source", "yfinance")
    fallback = rf_cfg.get("fallback_rate", 0.045)

    if source != "yfinance":
        return fallback

    threshold = rf_cfg.get("short_long_dte_threshold", 90)
    ticker = rf_cfg.get("short_rate_ticker", "^IRX") if dte < threshold \
        else rf_cfg.get("long_rate_ticker", "^TNX")

    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="5d")
        if hist.empty:
            return fallback
        # yfinance returns percentage → divide by 100
        rate = float(hist["Close"].iloc[-1]) / 100.0
        if rate <= 0:
            return fallback
        return rate
    except Exception:
        return fallback


def fetch_dividend_yield(ticker: str, spot: float, config: dict) -> float:
    """
    Estimate continuous dividend yield from trailing annual dividends.

    Simplifying assumption: constant continuous yield over option life,
    estimated as trailing_annual_dividend / current_spot_price.
    """
    div_cfg = config.get("dividends", {})
    fallback = div_cfg.get("fallback_yield", 0.015)

    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        # Try multiple fields yfinance might return
        div_rate = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
        if div_rate and div_rate > 0:
            return float(div_rate)
        # Fall back to computing from trailing dividend
        annual_div = info.get("trailingAnnualDividendRate", 0)
        if annual_div and annual_div > 0 and spot > 0:
            return float(annual_div / spot)
    except Exception:
        pass

    return fallback


def fetch_options_chain(ticker: str, config: dict) -> pd.DataFrame:
    """
    Fetch full options chain for all available expiries.

    Returns a DataFrame with columns:
      strike, bid, ask, mid, lastPrice, volume, openInterest,
      option_type ('call'/'put'), expiry (datetime), T (years),
      dte (days), price_source ('mid'/'excluded'), log_moneyness

    Mid price rules:
      1. Mid = (bid + ask) / 2 if both > 0 and ask > bid
      2. Otherwise → price_source = 'excluded', mid = NaN
      3. Last price never silently mixed into mid
    """
    tk = yf.Ticker(ticker)
    spot = fetch_spot(ticker)
    if np.isnan(spot):
        return pd.DataFrame()

    expiries = tk.options
    if not expiries:
        return pd.DataFrame()

    today = datetime.now()
    all_rows = []

    for exp_str in expiries:
        try:
            chain = tk.option_chain(exp_str)
        except Exception:
            continue

        exp_date = pd.to_datetime(exp_str)
        dte = (exp_date - pd.Timestamp(today.date())).days
        if dte <= 0:
            continue
        T = dte / 365.0

        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            for _, row in df.iterrows():
                bid = row.get("bid", 0) or 0
                ask = row.get("ask", 0) or 0
                strike = row.get("strike", np.nan)

                # Mid price computation
                if bid > 0 and ask > 0 and ask > bid:
                    mid = (bid + ask) / 2.0
                    price_source = "mid"
                else:
                    mid = np.nan
                    price_source = "excluded"

                log_m = np.log(strike / spot) if strike > 0 else np.nan

                all_rows.append({
                    "strike": strike,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "lastPrice": row.get("lastPrice", np.nan),
                    "volume": row.get("volume", 0) or 0,
                    "openInterest": row.get("openInterest", 0) or 0,
                    "option_type": opt_type,
                    "expiry": exp_date,
                    "T": T,
                    "dte": dte,
                    "price_source": price_source,
                    "log_moneyness": log_m,
                    "impliedVolatility": row.get("impliedVolatility", np.nan),
                })

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return df


def load_market_data(ticker: str, config: dict) -> dict:
    """
    Load all market data for a ticker: spot, chain, rate, dividend yield.

    This is the main entry point for the data layer. Returns a dict
    with all data needed by downstream modules.

    Returns:
        Dict with keys: 'ticker', 'spot', 'chain', 'risk_free_rate',
        'dividend_yield', 'data_source'.
    """
    spot = fetch_spot(ticker)
    chain = fetch_options_chain(ticker, config)

    # Fetch short and long rates once, then assign per-row by DTE
    rf_cfg = config.get("risk_free_rate", {})
    threshold = rf_cfg.get("short_long_dte_threshold", 90)
    short_rate = fetch_risk_free_rate(threshold - 1, config)
    long_rate = fetch_risk_free_rate(threshold, config)

    if not chain.empty and "dte" in chain.columns:
        chain["risk_free_rate"] = np.where(
            chain["dte"] < threshold, short_rate, long_rate
        )
    # Scalar fallback for downstream code that expects a single rate
    median_dte = int(chain["dte"].median()) if not chain.empty and "dte" in chain.columns else 30
    scalar_rate = short_rate if median_dte < threshold else long_rate

    div_yield = fetch_dividend_yield(ticker, spot, config)

    return {
        "ticker": ticker,
        "spot": spot,
        "chain": chain,
        "risk_free_rate": scalar_rate,
        "dividend_yield": div_yield,
        "data_source": "yfinance",
    }
