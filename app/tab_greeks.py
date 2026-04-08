"""
Tab 3: Greeks — Greeks table, profiles vs spot, and higher-order
Greeks (BS analytical only). Every chart gets an interpretation callout.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src import black_scholes as bs
from app.style_inject import styled_card, apply_plotly_theme


def render(state, config):
    """Render the Greeks tab."""
    market = state["market"]
    S = market["spot"]
    r = market["risk_free_rate"]
    q = market["dividend_yield"]

    # ── Inputs ───────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    K = c1.number_input("Strike", value=float(round(S)), step=1.0,
                         min_value=0.01, key="greeks_K")
    T = c2.number_input("Time (years)", value=0.25, step=0.05,
                         min_value=0.001, key="greeks_T")
    sigma = c3.number_input("Vol (sigma)", value=0.25, step=0.01,
                             min_value=0.01, key="greeks_sigma")
    opt_type = c4.selectbox("Type", ["call", "put"], key="greeks_type")

    # ── Greeks table ─────────────────────────────────────────
    greeks = bs.all_greeks(S, K, T, r, sigma, q, opt_type)

    st.markdown("### BS Analytical Greeks")
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Delta", f"{greeks['delta']:.4f}")
    g2.metric("Gamma", f"{greeks['gamma']:.4f}")
    g3.metric("Theta", f"{greeks['theta']:.4f} /day")
    g4.metric("Vega", f"{greeks['vega']:.4f} /pt")
    g5.metric("Rho", f"{greeks['rho']:.4f} /1%")

    h1, h2, h3 = st.columns(3)
    h1.metric("Vanna", f"{greeks['vanna']:.6f}")
    h2.metric("Volga", f"{greeks['volga']:.6f}")
    h3.metric("Charm", f"{greeks['charm']:.6f} /day")

    styled_card(
        "Standard Greeks (top) from all 3 models. Higher-order Greeks "
        "(bottom) are BS analytical only — FD on trees amplifies noise."
    )
    styled_card(
        "Higher-order Greeks (Vanna, Volga, Charm) are model-dependent and "
        "numerically sensitive. Values shown are BS analytical — actual market "
        "Greeks differ under stochastic vol."
    )

    # ── Greeks vs Spot ───────────────────────────────────────
    st.markdown("### Greeks vs Spot Price")
    spots = np.linspace(S * 0.7, S * 1.3, 80)

    delta_vals = [bs.delta(s, K, T, r, sigma, q, opt_type) for s in spots]
    gamma_vals = [bs.gamma(s, K, T, r, sigma, q, opt_type) for s in spots]
    theta_vals = [bs.theta(s, K, T, r, sigma, q, opt_type) for s in spots]
    vega_vals = [bs.vega(s, K, T, r, sigma, q, opt_type) for s in spots]

    left, right = st.columns(2)

    with left:
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=spots, y=delta_vals, name="Delta"))
        fig_d.add_vline(x=S, line_dash="dot", line_color="#475569")
        fig_d.add_vline(x=K, line_dash="dash", line_color="#EF4444",
                        annotation_text="Strike")
        fig_d.update_layout(title="Delta vs Spot",
                            xaxis_title="Spot Price ($)", yaxis_title="Delta (per $1 spot)")
        apply_plotly_theme(fig_d)
        st.plotly_chart(fig_d, use_container_width=True)
        styled_card(
            "Delta measures directional exposure. It transitions from "
            "0 (OTM) to 1 (ITM) around the strike — steeper near ATM."
        )

    with right:
        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(x=spots, y=gamma_vals, name="Gamma"))
        fig_g.add_vline(x=K, line_dash="dash", line_color="#EF4444",
                        annotation_text="Strike")
        fig_g.update_layout(title="Gamma vs Spot",
                            xaxis_title="Spot Price ($)", yaxis_title="Gamma (per $1 spot)")
        apply_plotly_theme(fig_g)
        st.plotly_chart(fig_g, use_container_width=True)
        styled_card(
            "Gamma peaks near ATM — short-dated ATM options are hardest "
            "to delta-hedge because their delta changes fastest."
        )

    left2, right2 = st.columns(2)

    with left2:
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=spots, y=theta_vals, name="Theta"))
        fig_t.add_vline(x=K, line_dash="dash", line_color="#EF4444",
                        annotation_text="Strike")
        fig_t.update_layout(title="Theta vs Spot (per calendar day)",
                            xaxis_title="Spot Price ($)", yaxis_title="Theta ($ per calendar day)")
        apply_plotly_theme(fig_t)
        st.plotly_chart(fig_t, use_container_width=True)
        styled_card(
            "Theta is the daily cost of holding an option. Most negative "
            "near ATM — the price you pay for gamma exposure."
        )

    with right2:
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=spots, y=vega_vals, name="Vega"))
        fig_v.add_vline(x=K, line_dash="dash", line_color="#EF4444",
                        annotation_text="Strike")
        fig_v.update_layout(title="Vega vs Spot (per +1pp annualized vol)",
                            xaxis_title="Spot Price ($)", yaxis_title="Vega ($ per +0.01 annualized sigma)")
        apply_plotly_theme(fig_v)
        st.plotly_chart(fig_v, use_container_width=True)
        styled_card(
            "Vega is sensitivity to implied vol changes. Peaks near ATM "
            "and increases with time to expiry."
        )
