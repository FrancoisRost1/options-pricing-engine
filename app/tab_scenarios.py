"""
Tab 6: Scenario Analysis — P&L heatmap (spot x vol), time decay
slider, and mechanical presets (earnings, vol crush, rate hike).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src import black_scholes as bs, scenario_analysis as sa
from app.style_inject import styled_card, apply_plotly_theme


def render(state, config):
    """Render the Scenarios tab."""
    market = state["market"]
    S = market["spot"]
    r = market["risk_free_rate"]
    q = market["dividend_yield"]

    # ── Inputs ───────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    K = c1.number_input("Strike", value=float(round(S)), step=1.0,
                         min_value=0.01, key="sc_K")
    T = c2.number_input("Time (years)", value=0.25, step=0.05,
                         min_value=0.001, key="sc_T")
    sigma = c3.number_input("Vol", value=0.25, step=0.01,
                             min_value=0.01, key="sc_sigma")
    opt_type = c4.selectbox("Type", ["call", "put"], key="sc_type")

    entry_price = bs.price(S, K, T, r, sigma, q, opt_type)
    st.markdown(f"**Entry price:** ${entry_price:.4f}")

    # ── P&L Heatmap ──────────────────────────────────────────
    st.markdown("### P&L Heatmap: Spot Change x Vol Change")

    days_fwd = st.select_slider(
        "Days forward (time decay)",
        options=config.get("scenarios", {}).get("time_forward_days", [0, 7, 14, 30]),
        value=0, key="sc_days",
    )

    T_adj = max(T - days_fwd / 365.0, 1e-6)
    grid = sa.compute_pnl_grid(S, K, T_adj, r, sigma, q, opt_type,
                                entry_price=entry_price, config=config)

    spot_labels = [f"{x:+.0%}" for x in grid["spot_changes"]]
    vol_labels = [f"{x:+.0f}pts" for x in [v * 100 for v in grid["vol_changes"]]]

    fig = go.Figure(data=go.Heatmap(
        z=grid["values"],
        x=spot_labels,
        y=vol_labels,
        colorscale=[[0, "#EF4444"], [0.5, "#1E293B"], [1, "#10B981"]],
        zmid=0,
        text=np.round(grid["values"], 2),
        texttemplate="$%{text:.2f}",
        textfont=dict(size=10),
        colorbar=dict(title="P&L ($)"),
    ))
    fig.update_layout(
        title=f"P&L Heatmap ({opt_type.title()} K={K:.0f}, +{days_fwd}d)",
        xaxis_title="Spot Change",
        yaxis_title="Vol Change",
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    styled_card(
        "Green = profit, red = loss. Rows are vol bumps, columns are spot moves. "
        "Use the time slider to see how theta decay shifts the surface."
    )

    # ── Presets ───────────────────────────────────────────────
    st.markdown("### Scenario Presets")
    styled_card(
        "Presets are mechanical bump combinations — the labels describe typical "
        "events but are not predictions. Values from config.yaml."
    )

    preset_tabs = st.tabs(["Earnings", "Vol Crush", "Rate Hike"])

    for i, (name, tab) in enumerate(zip(
        ["earnings", "vol_crush", "rate_hike"], preset_tabs
    )):
        with tab:
            results = sa.apply_preset(S, K, T, r, sigma, q, opt_type,
                                       entry_price=entry_price,
                                       preset_name=name, config=config)
            if not results:
                st.info(f"No '{name}' preset configured.")
                continue

            cols = ["scenario", "spot_change", "vol_change",
                    "rate_change", "new_price", "pnl"]
            df = st.dataframe(
                [{
                    "Scenario": row["scenario"],
                    "Spot": f"{row['spot_change']:+.0%}",
                    "Vol": f"{row['vol_change']:+.0%}",
                    "Rate": f"{row['rate_change']:+.2%}" if row["rate_change"] else "—",
                    "New Price": f"${row['new_price']:.4f}" if not np.isnan(row["new_price"]) else "—",
                    "P&L": f"${row['pnl']:+.4f}" if not np.isnan(row["pnl"]) else "—",
                } for row in results],
                use_container_width=True,
            )
