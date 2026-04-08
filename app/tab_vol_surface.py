"""
Tab 4: Volatility Surface — 3D surface, smile per expiry,
ATM term structure, and skew metrics (25-delta RR & butterfly).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src import vol_surface, skew_metrics
from app.style_inject import styled_card, apply_plotly_theme


def render(state, config):
    """Render the Volatility Surface tab."""
    market = state["market"]
    chain_iv = state.get("chain_iv")

    if chain_iv is None or chain_iv.empty or "iv" not in chain_iv.columns:
        st.warning("No IV data. Load a chain first.")
        return

    S = market["spot"]
    r = market["risk_free_rate"]
    q = market["dividend_yield"]
    valid = chain_iv.dropna(subset=["iv"])

    if valid.empty:
        st.warning("No valid IVs extracted.")
        return

    # ── 3D Surface ───────────────────────────────────────────
    st.markdown("### Implied Volatility Surface")
    surface = vol_surface.build_surface(chain_iv, config)

    if surface is not None:
        fig_3d = go.Figure(data=[go.Surface(
            x=surface["grid_moneyness"],
            y=surface["grid_T"],
            z=surface["grid_iv"],
            colorscale="Viridis",
            colorbar=dict(title="IV"),
        )])
        fig_3d.update_layout(
            title="IV Surface: Log-Moneyness x Expiry x Implied Vol",
            scene=dict(
                xaxis_title="Log-Moneyness ln(K/S)",
                yaxis_title="Time to Expiry T (years)",
                zaxis_title="Implied Vol sigma (annualized)",
            ),
            height=550,
        )
        apply_plotly_theme(fig_3d)
        st.plotly_chart(fig_3d, use_container_width=True)
        styled_card(
            "Empirical IV surface via cubic interpolation. NOT arbitrage-free "
            "— shows market-observed skew and term structure."
        )
    else:
        st.info(f"Insufficient data for surface (need "
                f"{config.get('vol_surface', {}).get('min_points_for_surface', 20)} "
                f"valid IVs, have {len(valid)}).")

    # ── Smile per expiry & term structure ────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("### Volatility Smile by Expiry")
        smiles = vol_surface.smile_per_expiry(chain_iv)

        if smiles:
            fig_smile = go.Figure()
            for exp, data in sorted(smiles.items()):
                label = f"{data['dte']}d" if data["dte"] else str(exp)[:10]
                fig_smile.add_trace(go.Scatter(
                    x=data["log_moneyness"], y=data["iv"],
                    mode="lines+markers", name=label,
                ))
            fig_smile.update_layout(
                title="IV Smile per Expiry",
                xaxis_title="Log-Moneyness ln(K/S)",
                yaxis_title="Implied Vol sigma (annualized)",
            )
            apply_plotly_theme(fig_smile)
            st.plotly_chart(fig_smile, use_container_width=True)
            styled_card(
                "Equity smile slopes down-left (put skew) — deep OTM puts "
                "trade at higher IV, reflecting crash protection demand."
            )

    with right:
        st.markdown("### ATM Vol Term Structure")
        ts = vol_surface.term_structure(chain_iv, S)

        if not ts.empty:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=ts["T"], y=ts["atm_iv"], mode="lines+markers",
                name="ATM IV",
            ))
            fig_ts.update_layout(
                title="ATM Implied Vol vs Time to Expiry",
                xaxis_title="Time to Expiry T (years)",
                yaxis_title="ATM Implied Vol sigma (annualized)",
            )
            apply_plotly_theme(fig_ts)
            st.plotly_chart(fig_ts, use_container_width=True)
            styled_card(
                "Upward-sloping in calm markets (more uncertainty over time). "
                "Inverted when near-term event risk spikes IV at short maturities."
            )

    # ── Skew metrics ─────────────────────────────────────────
    st.markdown("### 25-Delta Skew Metrics")
    skew_df = skew_metrics.compute_skew_metrics(chain_iv, S, r, q, config)

    if not skew_df.empty:
        display = skew_df[["dte", "T", "atm_iv", "call_25d_iv", "put_25d_iv",
                           "risk_reversal", "butterfly"]].copy()
        for col in ["atm_iv", "call_25d_iv", "put_25d_iv", "risk_reversal", "butterfly"]:
            if col in display.columns:
                display[col] = display[col].apply(
                    lambda x: f"{x:.2%}" if not np.isnan(x) else "—"
                )
        st.dataframe(display, use_container_width=True)
        styled_card(
            "25-delta RR measures skew direction (negative = steeper put skew). "
            "Butterfly measures smile curvature (positive = wings trade rich vs ATM). "
            "Adapted from FX conventions."
        )
    else:
        st.info("Insufficient data for skew metrics.")
