"""Showcase: the label-free fault-type map (CLR-PCA) + the ablation behind it."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import charts, data, theme


def render():
    st.markdown("## Fault-type map — compositional (CLR-PCA)")
    st.markdown(
        "<span class='muted'>Representations built on gas <b>concentrations</b> encode <b>how "
        "much</b> gas there is (severity), not <b>which</b> fault. Splitting each sample into a "
        "magnitude and a <b>composition</b> (gas proportions through the centred log-ratio) "
        "removes severity by construction — and a simple two-dimensional <b>PCA</b> of that "
        "composition recovers the Duval fault-type structure without any labels "
        "(ARI 0.16 → 0.55). A neural autoencoder (SD-CAE) was tested and adds nothing: it is "
        "reported as a negative ablation.</span>", unsafe_allow_html=True)

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
            "<div class='card'>The projection only ever sees the composition, so the same fault "
            "type at any gassing level maps to the same point — <b>severity invariance by "
            "construction</b>. Being linear, the map is <b>deterministic</b> and its axes are "
            "readable combinations of log-ratios an engineer can interpret.</div>",
            unsafe_allow_html=True)
        st.markdown(
            "<div class='card'>Toggle to <b>severity (m)</b>: colour no longer separates the "
            "clusters — confirming the map encodes <i>type</i>, not magnitude.</div>",
            unsafe_allow_html=True)
        st.markdown(
            "<div class='card'><b>Honest negatives.</b> The SD-CAE autoencoder (ARI 0.47 ± 0.06) "
            "does not beat this linear projection (0.545 ± 0.002), and an adversarial "
            "severity-independence term makes things worse (0.30). Both are reported as negative "
            "ablations in the paper.</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Ablation: the gain is the log-ratio geometry, "
                "not the model</div>", unsafe_allow_html=True)
    tab = data.get_table("representation_baselines")
    if tab is not None:
        show = [("raw-log 5D + KMeans", "raw log"),
                ("proportions 5D + KMeans", "proportions"),
                ("CLR 5D (std) + KMeans", "CLR + KMeans"),
                ("PCA-2 of CLR + KMeans", "CLR + PCA-2 (deployed)"),
                ("[ref] AE-2D on CLR (paper)", "CLR + AE (SD-CAE, ablation)")]
        rows = [r for r, _ in show if r in tab.index]
        labels = [lbl for r, lbl in show if r in tab.index]
        vals = tab.loc[rows, "ARI_mean"]
        errs = tab.loc[rows, "ARI_std"]
        colors = ["#9aa4b2", "#9aa4b2", "#90caf9", theme.ACCENT, theme.BAD][: len(rows)]
        fig = go.Figure(go.Bar(
            x=labels, y=vals, error_y=dict(type="data", array=errs, visible=True),
            marker_color=colors, text=[f"{v:.2f}" for v in vals], textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}: ARI %{y:.3f}<extra></extra>"))
        fig.update_layout(yaxis_title="ARI vs Duval (k=7)", xaxis_title=None)
        st.plotly_chart(theme.style_fig(fig, height=360, legend=False),
                        use_container_width=True, config={"displayModeBar": False})
        st.caption("Raw proportions barely help; the CLR log-ratio geometry does the work, and a "
                   "linear PCA projection beats the neural encoder — with 25x less seed variance.")
    else:
        st.info("Run `python scripts/run_representation_baselines.py` to generate the ablation table.")
