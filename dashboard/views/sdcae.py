"""Showcase: the SD-CAE learned diagnostic map + the ablation that justifies it."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from lib import charts, data, theme


def render():
    st.markdown("## SD-CAE — learned, label-free fault map")
    st.markdown(
        "<span class='muted'>A plain autoencoder on gas concentrations encodes <b>how much</b> gas "
        "(severity). <b>SD-CAE</b> first splits each sample into a magnitude and a composition "
        "(gas proportions, via the centred log-ratio), and encodes only the composition — so the "
        "latent captures the <b>fault type</b>. Clustering it recovers Duval structure without any "
        "labels (agreement ARI 0.14 → 0.47).</span>", unsafe_allow_html=True)

    sd = data.get_sdcae()
    c1, c2 = st.columns([1.5, 1])
    with c1:
        color_by = st.radio("Colour points by", ["duval", "cluster", "magnitude"],
                            horizontal=True, key="sdcae_color", format_func=lambda v: {
                                "duval": "Duval class", "cluster": "discovered cluster",
                                "magnitude": "severity (m)"}[v])
        st.plotly_chart(charts.sdcae_scatter(sd, color_by), use_container_width=True,
                        config={"displayModeBar": False})
    with c2:
        st.markdown("<div class='section-title'>Why it works</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='card'>The encoder only ever sees the composition, so the same fault type "
            "at any gassing level maps to the same point — <b>severity invariance by "
            "construction</b>. The carbon oxides (cellulose markers) are excluded from the "
            "composition and kept for the severity term elsewhere.</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='card'>Toggle to <b>severity (m)</b>: colour no longer separates the "
            "clusters — confirming the map encodes <i>type</i>, not magnitude.</div>",
            unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Ablation: it is the log-ratio geometry, not normalisation</div>",
                unsafe_allow_html=True)
    abl = data.get_table("sdcae_ablation")
    if abl is not None and "latent_dim" in abl.columns:
        d2 = abl[abl["latent_dim"] == 2]
        names = [str(i).split(" [")[0] for i in d2.index]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="ARI vs Duval (higher = better)", x=names,
                             y=d2["ARI@k7_mean"], marker_color=theme.ACCENT,
                             text=[f"{v:.2f}" for v in d2["ARI@k7_mean"]], textposition="outside"))
        if "R2_m_mean" in d2.columns:
            fig.add_trace(go.Bar(name="Severity leakage R²(m|z) (lower = better)", x=names,
                                 y=d2["R2_m_mean"], marker_color=theme.BAD,
                                 text=[f"{v:.2f}" for v in d2["R2_m_mean"]], textposition="outside"))
        fig.update_layout(barmode="group", yaxis_title=None, xaxis_title=None)
        st.plotly_chart(theme.style_fig(fig, height=380), use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("Raw proportions barely help (0.15); the CLR composition jumps to ~0.47. The "
                   "adversarial-independence variant (λ>0) over-removes signal, so we deploy λ=0.")
    else:
        st.info("Run `python scripts/run_sdcae_ablation.py` to generate the ablation table.")
