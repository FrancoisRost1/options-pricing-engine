"""
Options Pricing Engine, Streamlit Dashboard (7 tabs).

Launch from project root: streamlit run app/app.py

Loads market data (yfinance or synthetic fallback), filters the
chain, extracts implied vols, then presents analytics across
seven tabs covering pricing, Greeks, vol surface, model comparison,
scenarios, and delta hedging.
"""

import sys
import os
from pathlib import Path
import streamlit as st
import numpy as np

# CWD bootstrap, ensure project root is CWD and on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from app.style_inject import inject_styles, styled_header, styled_card, styled_kpi, styled_divider, TOKENS
from utils.config_loader import load_config
from utils.env_detect import is_streamlit_cloud
from src import synthetic_data, chain_filter, implied_vol

IS_CLOUD = is_streamlit_cloud()

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
    """Cache yfinance data for 15 minutes. Lazy-imports data_loader."""
    from src import data_loader
    return data_loader.load_market_data(ticker, _config)


@st.cache_data(ttl=3600)
def _load_synthetic(_config):
    """Cache synthetic data for 1 hour."""
    return synthetic_data.generate_chain(_config)


def _load_data(ticker, use_synthetic, config):
    """Load market data with fallback logic.

    On Streamlit Cloud, always routes to synthetic data because
    yfinance is blocked on Cloud provider IPs.
    """
    if use_synthetic or IS_CLOUD:
        if IS_CLOUD and not use_synthetic:
            st.info(
                "Running on Streamlit Cloud: live market data is unavailable. "
                "Showing synthetic example data."
            )
        return _load_synthetic(config)

    try:
        market = _load_live_data(ticker, config)
    except Exception as exc:
        st.warning(
            f"yfinance error ({type(exc).__name__}): falling back to synthetic data."
        )
        return _load_synthetic(config)

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
        styled_divider()

        ticker = st.text_input(
            "Ticker", value=config.get("data", {}).get("default_ticker", "AAPL"),
            disabled=IS_CLOUD,
        ).upper().strip()
        use_synthetic = st.checkbox(
            "Use synthetic data",
            value=IS_CLOUD,
            disabled=IS_CLOUD,
        )
        load_btn = st.button("Load Data", type="primary", use_container_width=True)

        styled_divider()
        st.markdown("### Model Limitations")
        st.markdown(
            f"<div style='background:{TOKENS['bg_elevated']}; "
            f"border:1px solid {TOKENS['border_default']}; "
            f"border-radius:{TOKENS['radius_md']}; padding:12px; "
            f"font-size:0.8rem; color:{TOKENS['text_secondary']}; "
            f"line-height:1.6;'>"
            f"<b style='color:{TOKENS['accent_warning']};'>Know your assumptions:</b><br>"
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
        source_note = (
            "Data: synthetic only (Cloud mode)."
            if IS_CLOUD else "Data: yfinance (live) or synthetic."
        )
        muted = TOKENS["text_muted"]
        st.markdown(
            f"<small style='color:{muted};'>{source_note}</small>",
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
    iv_chain = st.session_state.get("chain_iv")
    valid_iv = iv_chain["iv"].dropna() if iv_chain is not None and "iv" in iv_chain.columns else []
    with c1:
        styled_kpi("SPOT", f"${market['spot']:.2f}")
    with c2:
        styled_kpi("RISK-FREE RATE", f"{market['risk_free_rate']:.2%}")
    with c3:
        styled_kpi("DIV YIELD", f"{market['dividend_yield']:.2%}")
    with c4:
        styled_kpi("VALID IVS", f"{len(valid_iv)} / {len(st.session_state['filtered_chain'])}")

    # ── Tabs ─────────────────────────────────────────────────
    tabs = st.tabs([
        "CHAIN", "PRICING", "GREEKS", "VOL SURFACE",
        "MODEL VS MARKET", "SCENARIOS", "DELTA HEDGE",
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
