"""
Options Pricing Engine — Streamlit Dashboard (7 tabs).

Launch from project root: streamlit run app/app.py

Loads market data (yfinance or synthetic fallback), filters the
chain, extracts implied vols, then presents analytics across
seven tabs covering pricing, Greeks, vol surface, model comparison,
scenarios, and delta hedging.
"""

import sys
import os
import streamlit as st
import numpy as np

# Ensure project root is on path for src/ and utils/ imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.style_inject import inject_styles, styled_header, styled_card
from utils.config_loader import load_config
from src import data_loader, synthetic_data, chain_filter, implied_vol

from app.tab_chain_explorer import render as render_chain_explorer
from app.tab_pricing import render as render_pricing
from app.tab_greeks import render as render_greeks
from app.tab_vol_surface import render as render_vol_surface
from app.tab_model_vs_market import render as render_model_vs_market
from app.tab_scenarios import render as render_scenarios
from app.tab_delta_hedge import render as render_delta_hedge

st.set_page_config(
    page_title="Options Pricing Engine",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()


@st.cache_data(ttl=900)
def _load_live_data(ticker, _config):
    """Cache yfinance data for 15 minutes."""
    return data_loader.load_market_data(ticker, _config)


@st.cache_data(ttl=3600)
def _load_synthetic(_config):
    """Cache synthetic data for 1 hour."""
    return synthetic_data.generate_chain(_config)


def _load_data(ticker, use_synthetic, config):
    """Load market data with fallback logic."""
    if use_synthetic:
        return _load_synthetic(config)

    market = _load_live_data(ticker, config)
    if market["chain"].empty:
        if config.get("data", {}).get("use_synthetic_fallback", True):
            st.warning("yfinance returned empty chain: using synthetic data.")
            return _load_synthetic(config)
        st.error("No data available.")
        return None
    return market


def main():
    """Main dashboard entry point."""
    config = load_config()

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Options Pricing Engine")
        st.markdown("---")

        ticker = st.text_input(
            "Ticker", value=config.get("data", {}).get("default_ticker", "AAPL")
        ).upper().strip()
        use_synthetic = st.checkbox("Use synthetic data", value=False)
        load_btn = st.button("Load Data", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown("### Model Limitations")
        st.markdown(
            "<div style='background:#1A2236; border:1px solid rgba(148,163,184,0.12); "
            "border-radius:10px; padding:12px; font-size:0.8rem; color:#94A3B8; "
            "line-height:1.6;'>"
            "<b style='color:#F59E0B;'>Know your assumptions:</b><br>"
            "- BS assumes constant vol (no smile)<br>"
            "- No jumps, no stochastic vol<br>"
            "- Continuous dividend yield (not discrete)<br>"
            "- Flat yield curve (short/long bucket only)<br>"
            "- Vol surface is empirical, not arbitrage-free<br>"
            "- US equity options are American-style;<br>"
            "&nbsp;&nbsp;MC and parity check assume European<br>"
            "- GBM has no fat tails"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown(
            "<small style='color:#475569;'>Data: yfinance (live) or synthetic.</small>",
            unsafe_allow_html=True,
        )

    # ── Load / cache data ────────────────────────────────────
    if load_btn or "market" not in st.session_state:
        with st.spinner("Loading data..."):
            market = _load_data(ticker, use_synthetic, config)
        if market is None:
            return
        st.session_state["market"] = market
        st.session_state["config"] = config

        # Filter chain
        raw = market["chain"]
        filtered = chain_filter.apply_filters(raw, config)
        st.session_state["raw_chain"] = raw
        st.session_state["filtered_chain"] = filtered

        # Extract IVs
        if not filtered.empty:
            with_iv = implied_vol.extract_chain(
                filtered, S=market["spot"],
                r=market["risk_free_rate"],
                q=market["dividend_yield"],
                config=config,
            )
            st.session_state["chain_iv"] = with_iv
        else:
            st.session_state["chain_iv"] = filtered

    if "market" not in st.session_state:
        styled_header("Options Pricing Engine", "Enter a ticker and click Load Data")
        return

    market = st.session_state["market"]

    # ── Header metrics ───────────────────────────────────────
    styled_header(
        "Options Pricing Engine",
        f"{market['ticker']} | Source: {market['data_source']}",
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spot", f"${market['spot']:.2f}")
    c2.metric("Risk-Free Rate", f"{market['risk_free_rate']:.2%}")
    c3.metric("Div Yield", f"{market['dividend_yield']:.2%}")
    iv_chain = st.session_state.get("chain_iv")
    valid_iv = iv_chain["iv"].dropna() if iv_chain is not None and "iv" in iv_chain.columns else []
    c4.metric("Valid IVs", f"{len(valid_iv)} / {len(st.session_state['filtered_chain'])}")

    # ── Tabs ─────────────────────────────────────────────────
    tabs = st.tabs([
        "Chain Explorer", "Pricing", "Greeks", "Vol Surface",
        "Model vs Market", "Scenarios", "Delta Hedge",
    ])

    with tabs[0]:
        render_chain_explorer(st.session_state, config)
    with tabs[1]:
        render_pricing(st.session_state, config)
    with tabs[2]:
        render_greeks(st.session_state, config)
    with tabs[3]:
        render_vol_surface(st.session_state, config)
    with tabs[4]:
        render_model_vs_market(st.session_state, config)
    with tabs[5]:
        render_scenarios(st.session_state, config)
    with tabs[6]:
        render_delta_hedge(st.session_state, config)


if __name__ == "__main__":
    main()
