"""
Tab 5: Model vs Market, price comparison scatter, error breakdown
by moneyness x maturity, and put-call parity validation.

Two modes:
  A) Calibration check, reprice with each contract's own IV (circular, should be ~0)
  B) Flat vol test, price all contracts with ATM vol per expiry (meaningful test)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src import black_scholes as bs, parity_check
from app.style_inject import styled_card, apply_plotly_theme, TOKENS


def _compute_atm_vol_per_expiry(df, S):
    """Get ATM IV for each expiry to use as flat vol."""
    atm_vols = {}
    for exp, group in df.groupby("expiry"):
        group = group.copy()
        group["dist"] = np.abs(group["strike"] - S)
        atm_row = group.loc[group["dist"].idxmin()]
        atm_vols[exp] = atm_row["iv"]
    return atm_vols


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

    # ── Mode selector ────────────────────────────────────────
    mode = st.radio(
        "Pricing mode",
        ["Flat Vol (meaningful test)", "Own IV (calibration check)"],
        horizontal=True,
        help="Flat Vol prices every contract with ATM vol for its expiry. "
             "Own IV reprices with each contract's extracted IV (circular: error should be ~0).",
    )

    use_flat = mode.startswith("Flat")

    if use_flat:
        atm_vols = _compute_atm_vol_per_expiry(valid, S)
        valid["pricing_vol"] = valid["expiry"].map(atm_vols)
        valid = valid.dropna(subset=["pricing_vol"])
    else:
        valid["pricing_vol"] = valid["iv"]

    # ── Compute BS model prices ──────────────────────────────
    valid["bs_price"] = valid.apply(
        lambda row: bs.price(S, row["strike"], row["T"], r,
                             row["pricing_vol"], q, row["option_type"]),
        axis=1,
    )
    valid["error"] = valid["bs_price"] - valid["mid"]
    valid = valid.dropna(subset=["bs_price", "mid"])

    if valid.empty:
        st.warning("No valid price comparisons.")
        return

    # ── Error stats ──────────────────────────────────────────
    label = "Flat ATM Vol" if use_flat else "Own IV (calibration)"
    st.markdown(f"### Price Comparison: BS ({label}) vs Market Mid")
    rmse = np.sqrt((valid["error"] ** 2).mean())
    mae = valid["error"].abs().mean()
    mean_err = valid["error"].mean()
    max_err = valid["error"].abs().max()

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("RMSE", f"${rmse:.4f}")
    e2.metric("MAE", f"${mae:.4f}")
    e3.metric("Mean Error", f"${mean_err:.4f}")
    e4.metric("Max |Error|", f"${max_err:.4f}")

    if use_flat:
        styled_card(
            "Flat vol test: prices every contract with the ATM implied vol for "
            "its expiry. Errors show how much the smile/skew matters: wings "
            "deviate because flat vol ignores moneyness-dependent pricing."
        )
    else:
        styled_card(
            "Calibration check: reprices each contract with its own extracted IV. "
            "Errors should be near zero: any deviation is numerical noise from "
            "the IV solver or mid-price rounding."
        )

    # ── Scatter plot ─────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        fig = go.Figure()
        for ot, color in [("call", TOKENS["accent_primary"]), ("put", TOKENS["accent_warning"])]:
            sub = valid[valid["option_type"] == ot]
            fig.add_trace(go.Scatter(
                x=sub["mid"], y=sub["bs_price"], mode="markers",
                name=ot.title(), marker=dict(color=color, size=5, opacity=0.7),
            ))
        max_val = max(valid["mid"].max(), valid["bs_price"].max()) * 1.05
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val], mode="lines",
            line=dict(dash="dash", color=TOKENS["text_muted"]), name="45-degree",
        ))
        fig.update_layout(
            title=f"BS Price ({label}) vs Market Mid",
            xaxis_title="Market Mid ($)", yaxis_title="BS Price ($)",
        )
        apply_plotly_theme(fig)
        st.plotly_chart(fig, width="stretch")

    # ── Error by moneyness bucket ────────────────────────────
    with right:
        bins = config.get("model_comparison", {}).get(
            "moneyness_buckets", [-0.3, -0.15, -0.05, 0.05, 0.15, 0.3]
        )
        valid["m_bucket"] = pd.cut(valid["log_moneyness"],
                                    bins=[-np.inf] + bins + [np.inf])
        bucket_err = valid.groupby("m_bucket", observed=True)["error"].agg(
            ["mean", "std", "count"]
        ).reset_index()
        bucket_err["m_bucket"] = bucket_err["m_bucket"].astype(str)

        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(
            x=bucket_err["m_bucket"], y=bucket_err["mean"],
            error_y=dict(type="data", array=bucket_err["std"].fillna(0)),
            name="Mean Error",
        ))
        fig_err.update_layout(
            title="Mean Error by Log-Moneyness Bucket",
            xaxis_title="Log-Moneyness ln(K/S)",
            yaxis_title="Error ($)",
        )
        apply_plotly_theme(fig_err)
        st.plotly_chart(fig_err, width="stretch")
        styled_card(
            "Wings show larger errors in flat-vol mode because the smile "
            "prices OTM puts richer (crash protection) than flat vol predicts."
        )

    # ── Put-call parity ──────────────────────────────────────
    st.markdown("### Put-Call Parity Validation (European Only)")
    styled_card(
        "US equity options are American-style. Parity deviations include "
        "early exercise premium: not true arbitrage. Treat as indicative."
    )
    styled_card(
        "High violation rates are expected. Sources of deviation: "
        "(1) early exercise premium on American options, "
        "(2) bid-ask spread noise in mid-price estimation, "
        "(3) continuous dividend yield approximation vs discrete dividends, "
        "(4) illiquid far-OTM strikes. These deviations do not indicate arbitrage."
    )
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

    violations = parity_df[parity_df["violation"]]
    if not violations.empty:
        st.markdown(f"**{len(violations)} violation(s) detected:**")
        disp = violations[["strike", "expiry", "call_mid", "put_mid",
                           "deviation_dollar", "deviation_pct"]].copy()
        disp["deviation_dollar"] = disp["deviation_dollar"].apply(
            lambda x: f"${x:.4f}")
        disp["deviation_pct"] = disp["deviation_pct"].apply(
            lambda x: f"{x:.2%}")
        st.dataframe(disp, width="stretch")
