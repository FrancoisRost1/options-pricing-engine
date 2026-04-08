"""
Tab 2: Pricing — single-option pricer, cross-model comparison,
and convergence analysis (binomial steps + MC paths).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src import black_scholes as bs, model_comparison
from app.style_inject import styled_card, apply_plotly_theme


def render(state, config):
    """Render the Pricing tab."""
    market = state["market"]
    S = market["spot"]
    r = market["risk_free_rate"]
    q = market["dividend_yield"]

    st.markdown("### Single Option Pricer")

    # ── Inputs ───────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    K = c1.number_input("Strike (K)", value=float(round(S)), step=1.0, min_value=0.01)
    T = c2.number_input("Time to expiry (years)", value=0.25, step=0.05, min_value=0.001)
    sigma = c3.number_input("Volatility (sigma)", value=0.25, step=0.01, min_value=0.01)
    r_input = c4.number_input("Risk-free rate", value=r, step=0.005, format="%.4f")
    opt_type = c5.selectbox("Type", ["call", "put"])

    # ── Cross-model comparison ───────────────────────────────
    st.markdown("### Cross-Model Comparison")
    comp = model_comparison.compare_models(S, K, T, r_input, sigma, q, opt_type, config)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Black-Scholes", f"${comp['bs_price']:.4f}")
    mc2.metric("Binomial (CRR)", f"${comp['binomial_price']:.4f}")
    mc3.metric("Monte Carlo", f"${comp['mc_price']:.4f}")
    mc4.metric("Max Deviation", f"${comp['max_deviation']:.4f}")

    styled_card(
        "All three models should agree for European vanilla options. "
        "Deviations > $0.05 suggest insufficient tree steps or MC paths."
    )

    st.markdown(f"MC 95% CI: [${comp['mc_ci_lower']:.4f}, ${comp['mc_ci_upper']:.4f}] "
                f"| SE: ${comp['mc_std_error']:.4f} "
                f"| CV beta: {comp.get('mc_cv_beta', '—')}")

    # ── Convergence charts ───────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("### Binomial Convergence")
        bt_df = model_comparison.binomial_convergence(S, K, T, r_input, sigma, q, opt_type, config)

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=bt_df["steps"], y=bt_df["price"], mode="lines+markers",
            name="Binomial price",
        ))
        fig_bt.add_hline(y=comp["bs_price"], line_dash="dash",
                         annotation_text="BS price", line_color="#EF4444")
        fig_bt.update_layout(
            title="Binomial Price vs Tree Steps",
            xaxis_title="Number of steps",
            yaxis_title="Option price ($)",
        )
        apply_plotly_theme(fig_bt)
        st.plotly_chart(fig_bt, use_container_width=True)

        styled_card(
            "CRR trees show odd/even oscillation that dampens with more steps, "
            "converging to the Black-Scholes analytical price."
        )

    with right:
        st.markdown("### Monte Carlo Convergence")
        mc_df = model_comparison.mc_convergence(S, K, T, r_input, sigma, q, opt_type, config)

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(
            x=mc_df["n_paths"], y=mc_df["price"], mode="lines+markers",
            name="MC price",
        ))
        fig_mc.add_trace(go.Scatter(
            x=mc_df["n_paths"], y=mc_df["ci_upper"],
            mode="lines", line=dict(width=0), showlegend=False,
        ))
        fig_mc.add_trace(go.Scatter(
            x=mc_df["n_paths"], y=mc_df["ci_lower"],
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(99,102,241,0.15)", name="95% CI",
        ))
        fig_mc.add_hline(y=comp["bs_price"], line_dash="dash",
                         annotation_text="BS price", line_color="#EF4444")
        fig_mc.update_layout(
            title="MC Price vs Number of Paths",
            xaxis_title="Paths", yaxis_title="Option price ($)",
        )
        apply_plotly_theme(fig_mc)
        st.plotly_chart(fig_mc, use_container_width=True)

        styled_card(
            "More paths narrow the confidence interval. Control variates "
            "and antithetic variates dramatically reduce the CI width."
        )
