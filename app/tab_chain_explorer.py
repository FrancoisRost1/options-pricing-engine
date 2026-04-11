"""
Tab 1: Chain Explorer — browse the raw and filtered options chain.

Entry point for the dashboard. Shows the full options chain with
filters for expiry, moneyness, and liquidity. Smart defaults
narrow to near-ATM liquid options on first load.
"""

import streamlit as st
import numpy as np
from src import chain_filter
from app.style_inject import styled_card


def render(state, config):
    """Render the Chain Explorer tab."""
    market = state["market"]
    raw = state.get("raw_chain")
    filtered = state.get("filtered_chain")

    if raw is None or raw.empty:
        st.warning("No chain data loaded.")
        return

    spot = market["spot"]
    summary = chain_filter.filter_summary(raw, filtered)

    # ── Filter summary ───────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Contracts", summary["raw_contracts"])
    c2.metric("After Filter", summary["filtered_contracts"])
    c3.metric("Removed", f"{summary['removed']} ({summary['removal_pct']:.0f}%)")
    c4.metric("Expiries", summary["unique_expiries_filtered"])

    styled_card(
        "Filters remove illiquid, wide-spread, and extreme-moneyness contracts. "
        "This ensures IV extraction and model comparison use reliable market data."
    )

    # ── User controls ────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        show_raw = st.checkbox("Show raw (unfiltered) chain", value=False)

    df = raw.copy() if show_raw else filtered.copy()
    if df.empty:
        st.info("No contracts match the current filters.")
        return

    with col2:
        expiries = sorted(df["expiry"].unique())
        selected_exp = st.multiselect(
            "Expiries", expiries,
            default=expiries[:min(4, len(expiries))],
            format_func=lambda x: f"{x.strftime('%Y-%m-%d')} ({int((x - np.datetime64('today')).days)}d)",
        )

    with col3:
        opt_type = st.selectbox("Type", ["All", "Calls", "Puts"])

    if selected_exp:
        df = df[df["expiry"].isin(selected_exp)]

    if opt_type == "Calls":
        df = df[df["option_type"] == "call"]
    elif opt_type == "Puts":
        df = df[df["option_type"] == "put"]

    # ── Display columns ──────────────────────────────────────
    display_cols = [
        "strike", "option_type", "bid", "ask", "mid", "lastPrice",
        "volume", "openInterest", "dte", "log_moneyness",
    ]
    if "iv" in df.columns:
        display_cols.append("iv")
    if "impliedVolatility" in df.columns and "iv" not in df.columns:
        display_cols.append("impliedVolatility")

    available = [c for c in display_cols if c in df.columns]
    display = df[available].copy()

    # Format numeric columns
    for col in ["bid", "ask", "mid", "lastPrice"]:
        if col in display.columns:
            display[col] = display[col].apply(
                lambda x: f"${x:.2f}" if not np.isnan(x) else ""
            )
    if "iv" in display.columns:
        display["iv"] = display["iv"].apply(
            lambda x: f"{x:.1%}" if not np.isnan(x) else ""
        )
    if "log_moneyness" in display.columns:
        display["log_moneyness"] = display["log_moneyness"].apply(
            lambda x: f"{x:.3f}" if not np.isnan(x) else ""
        )

    st.dataframe(display, use_container_width=True, height=500)
    st.caption(f"Showing {len(display)} contracts | Spot: ${spot:.2f}")
