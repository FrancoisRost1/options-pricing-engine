"""
Tab 5: Model vs Market — price comparison scatter, error breakdown
by moneyness x maturity, and put-call parity validation.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src import black_scholes as bs, parity_check
from app.style_inject import styled_card, apply_plotly_theme


def render(state, config):
    """Render the Model vs Market tab."""
    market = state["market"]
    chain_iv = state.get("chain_iv")

    if chain_iv is None or chain_iv.empty or "iv" not in chain_iv.columns:
        st.warning("No IV data. Load a chain first.")
        return

    S = market["spot"]
    r = market["risk_free_rate"]
    q = market["dividend_yield"]

    valid = chain_iv.dropna(subset=["iv"]).copy()
    if valid.empty:
        st.warning("No valid IVs to compare.")
        return

    # ── Compute BS model prices ──────────────────────────────
    valid["bs_price"] = valid.apply(
        lambda row: bs.price(S, row["strike"], row["T"], r, row["iv"], q,
                             row["option_type"]),
        axis=1,
    )
    valid["error"] = valid["bs_price"] - valid["mid"]
    valid = valid.dropna(subset=["bs_price", "mid"])

    if valid.empty:
        st.warning("No valid price comparisons.")
        return

    # ── Error stats ──────────────────────────────────────────
    st.markdown("### Price Comparison: BS Model vs Market Mid")
    rmse = np.sqrt((valid["error"] ** 2).mean())
    mae = valid["error"].abs().mean()
    mean_err = valid["error"].mean()
    max_err = valid["error"].abs().max()

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("RMSE", f"${rmse:.4f}")
    e2.metric("MAE", f"${mae:.4f}")
    e3.metric("Mean Error", f"${mean_err:.4f}")
    e4.metric("Max |Error|", f"${max_err:.4f}")

    # ── Scatter plot ─────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        fig = go.Figure()
        for ot, color in [("call", "#6366F1"), ("put", "#F59E0B")]:
            sub = valid[valid["option_type"] == ot]
            fig.add_trace(go.Scatter(
                x=sub["mid"], y=sub["bs_price"], mode="markers",
                name=ot.title(), marker=dict(color=color, size=5, opacity=0.7),
            ))
        max_val = max(valid["mid"].max(), valid["bs_price"].max()) * 1.05
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val], mode="lines",
            line=dict(dash="dash", color="#475569"), name="45-degree",
        ))
        fig.update_layout(
            title="BS Model Price vs Market Mid Price",
            xaxis_title="Market Mid ($)", yaxis_title="BS Price ($)",
        )
        apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        styled_card(
            "Points on the 45-degree line = perfect agreement. "
            "Deviations reflect model assumptions (flat vol, no skew in pricing)."
        )

    # ── Error by moneyness bucket ────────────────────────────
    with right:
        bins = config.get("model_comparison", {}).get(
            "moneyness_buckets", [-0.3, -0.15, -0.05, 0.05, 0.15, 0.3]
        )
        valid["m_bucket"] = pd.cut(valid["log_moneyness"], bins=[-np.inf] + bins + [np.inf])
        bucket_err = valid.groupby("m_bucket", observed=True)["error"].agg(["mean", "std", "count"])
        bucket_err = bucket_err.reset_index()
        bucket_err["m_bucket"] = bucket_err["m_bucket"].astype(str)

        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(
            x=bucket_err["m_bucket"], y=bucket_err["mean"],
            error_y=dict(type="data", array=bucket_err["std"].fillna(0)),
            name="Mean Error",
        ))
        fig_err.update_layout(
            title="Mean Error by Moneyness Bucket",
            xaxis_title="Log-Moneyness Bucket",
            yaxis_title="Error ($)",
        )
        apply_plotly_theme(fig_err)
        st.plotly_chart(fig_err, use_container_width=True)
        styled_card(
            "Error by moneyness shows where the model deviates most. "
            "Wings often show larger errors due to skew not captured by flat-vol pricing."
        )

    # ── Put-call parity ──────────────────────────────────────
    st.markdown("### Put-Call Parity Validation (European Only)")
    parity_df = parity_check.check_parity(chain_iv, S, r, q, config)

    if parity_df.empty:
        st.info("No matched call-put pairs for parity check.")
        return

    summary = parity_check.parity_summary(parity_df)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Pairs Tested", summary["total_pairs"])
    p2.metric("Violations", summary["violations"])
    p3.metric("Violation %", f"{summary['violation_pct']:.1f}%")
    p4.metric("Max Dev %", f"{summary['max_deviation_pct']:.2f}%")

    styled_card(
        "Put-call parity must hold for European options: C - P = Se^(-qT) - Ke^(-rT). "
        "Violations above threshold signal stale quotes or data quality issues."
    )

    violations = parity_df[parity_df["violation"]]
    if not violations.empty:
        st.markdown(f"**{len(violations)} violation(s) detected:**")
        disp = violations[["strike", "expiry", "call_mid", "put_mid",
                           "deviation_dollar", "deviation_pct"]].copy()
        disp["deviation_dollar"] = disp["deviation_dollar"].apply(lambda x: f"${x:.4f}")
        disp["deviation_pct"] = disp["deviation_pct"].apply(lambda x: f"{x:.2%}")
        st.dataframe(disp, use_container_width=True)
