"""
Tab 7: Delta Hedge, hedging simulation with P&L decomposition.

Illustrates discrete hedging replication, NOT a profitable strategy.
Shows how gamma/theta, hedging error, and TC interact.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src import delta_hedge
from app.style_inject import styled_card, apply_plotly_theme, TOKENS


def render(state, config):
    """Render the Delta Hedge tab."""
    market = state["market"]
    S = market["spot"]
    r = market["risk_free_rate"]
    q = market["dividend_yield"]

    styled_card(
        "This illustrates discrete hedging replication: not a trading strategy. "
        "It shows the gap between theoretical continuous hedging and real-world "
        "discrete rebalancing, including transaction cost drag."
    )

    # ── Inputs ───────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    K = c1.number_input("Strike", value=float(round(S)), step=1.0,
                         min_value=0.01, key="dh_K")
    T = c2.number_input("Time (years)", value=0.25, step=0.05,
                         min_value=0.01, key="dh_T")
    sigma = c3.number_input("Vol", value=0.25, step=0.01,
                             min_value=0.01, key="dh_sigma")
    position = c4.selectbox("Position", ["long_call", "long_put",
                                          "short_call", "short_put"],
                             key="dh_pos")
    freq = c5.selectbox("Rebalance", ["daily", "weekly"], key="dh_freq")

    tc_bps = st.slider("Transaction cost (bps)", 0.0, 20.0, 5.0, key="dh_tc")
    slip_bps = st.slider("Slippage (bps)", 0.0, 10.0, 2.0, key="dh_slip")

    dh_config = {
        "delta_hedge": {
            "hedge_frequency": freq,
            "tc_bps": tc_bps,
            "slippage_bps": slip_bps,
            "contract_multiplier": 100,
            "num_contracts": 1,
        }
    }

    # ── Generate path and run simulation ─────────────────────
    n_steps = max(int(T * 252), 10)
    paths = delta_hedge.generate_gbm_paths(
        S, T, r, sigma, q, n_paths=1, n_steps=n_steps, seed=42,
    )
    path = paths[0]

    result = delta_hedge.simulate_hedge(
        path, K, T, r, sigma, q,
        position=position, config=dh_config,
    )

    daily = result["daily"]
    summary = result["summary"]

    # ── Summary metrics ──────────────────────────────────────
    st.markdown("### Simulation Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total P&L", f"${summary['total_pnl']:.2f}")
    m2.metric("Gamma/Theta P&L", f"${summary['gamma_theta_pnl']:.2f}")
    m3.metric("Hedge Error", f"${summary['hedge_error_total']:.2f}")
    m4.metric("TC Drag", f"${summary['tc_drag']:.2f}")

    # Realized vol from path vs implied vol input
    log_returns = np.diff(np.log(path))
    realized_vol = float(np.std(log_returns) * np.sqrt(252))
    st.markdown(
        f"**Realized vol:** {realized_vol:.1%} | "
        f"**Implied vol (input):** {sigma:.1%} | "
        f"**Spread:** {(realized_vol - sigma):+.1%}"
    )
    styled_card(
        "P&L is negative when realized vol < implied vol: theta decay "
        "exceeds gamma gains. This is the fundamental driver of hedging P&L."
    )

    # ── Cumulative P&L chart ─────────────────────────────────
    st.markdown("### Cumulative P&L Over Time")
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=daily["day"], y=daily["cum_total_pnl"],
        mode="lines", name="Total P&L",
    ))
    fig_pnl.add_trace(go.Scatter(
        x=daily["day"], y=daily["cum_gamma_pnl"],
        mode="lines", name="Gamma P&L",
    ))
    fig_pnl.add_trace(go.Scatter(
        x=daily["day"], y=daily["cum_theta_pnl"],
        mode="lines", name="Theta P&L",
    ))
    fig_pnl.add_trace(go.Scatter(
        x=daily["day"], y=-daily["cum_tc_cost"],
        mode="lines", name="TC Drag (neg)",
    ))
    fig_pnl.update_layout(
        title="Cumulative P&L Decomposition",
        xaxis_title="Trading Day",
        yaxis_title="Cumulative P&L ($)",
    )
    apply_plotly_theme(fig_pnl)
    st.plotly_chart(fig_pnl, use_container_width=True)
    styled_card(
        "Gamma P&L comes from realized moves (good for long gamma). "
        "Theta P&L is the daily cost of carrying the position. "
        "TC drag accumulates with each rebalance."
    )

    # ── Spot path and delta ──────────────────────────────────
    left, right = st.columns(2)

    with left:
        fig_spot = go.Figure()
        fig_spot.add_trace(go.Scatter(
            x=daily["day"], y=daily["spot"], mode="lines", name="Spot",
        ))
        fig_spot.add_hline(y=K, line_dash="dash", line_color=TOKENS["accent_danger"],
                           annotation_text="Strike")
        fig_spot.update_layout(
            title="Simulated Spot Path (GBM)",
            xaxis_title="Trading Day (t)", yaxis_title="Spot Price ($)",
        )
        apply_plotly_theme(fig_spot)
        st.plotly_chart(fig_spot, use_container_width=True)
        styled_card(
            "GBM-simulated path used for this hedging simulation. "
            "In practice, use real historical prices when available."
        )

    with right:
        fig_delta = go.Figure()
        fig_delta.add_trace(go.Scatter(
            x=daily["day"], y=daily["delta"], mode="lines", name="Delta",
        ))
        fig_delta.update_layout(
            title="Option Delta Over Time",
            xaxis_title="Trading Day (t)", yaxis_title="Delta (per $1 spot)",
        )
        apply_plotly_theme(fig_delta)
        st.plotly_chart(fig_delta, use_container_width=True)
        styled_card(
            "Delta drifts as spot moves and time passes (charm). "
            "The hedger adjusts stock holdings at each rebalance to track this."
        )

    # ── Hedging error distribution ───────────────────────────
    st.markdown("### Hedging Error")
    st.metric("Hedge Error Std Dev", f"${summary['hedge_error_std']:.4f}")
    st.metric("Rebalances", summary["n_rebalances"])
    styled_card(
        "Hedging error is the residual from discrete vs continuous rebalancing. "
        "More frequent rebalancing reduces error but increases TC drag: "
        "this is the fundamental tradeoff in delta hedging."
    )
