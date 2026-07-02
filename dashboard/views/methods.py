"""Conventional rule-based baselines: Duval / IEC / Rogers on the same fleet."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import charts, data, theme


def render():
    st.markdown("## Conventional methods (baselines)")
    st.markdown("<span class='muted'>The rule-based diagnostics the unsupervised framework is "
                "compared against, computed on every sample by our reference implementation.</span>",
                unsafe_allow_html=True)

    diag = data.get_clean_diag()

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("<div class='section-title'>Duval Triangle 1 — full fleet</div>",
                    unsafe_allow_html=True)
        sample = diag.dropna(subset=["CH4", "C2H4", "C2H2"])
        if len(sample) > 3000:
            sample = sample.sample(3000, random_state=42)
        st.plotly_chart(charts.duval_triangle(sample), use_container_width=True,
                        config={"displayModeBar": False})
    with c2:
        st.markdown("<div class='section-title'>Class distribution by method</div>",
                    unsafe_allow_html=True)
        order = ["PD", "D1", "D2", "T1", "T2", "T3", "DT", "ND"]
        fig = go.Figure()
        for method, col in [("duval", theme.ACCENT), ("iec", "#06b6d4"), ("rogers", "#f59e0b")]:
            vc = diag[method].astype(str).value_counts()
            fig.add_trace(go.Bar(name=method.upper(), x=order,
                                 y=[int(vc.get(k, 0)) for k in order], marker_color=col))
        fig.update_layout(barmode="group", yaxis_title="samples", xaxis_title=None)
        st.plotly_chart(theme.style_fig(fig, height=420), use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown("<div class='section-title'>Risk-ranking power vs. the label-free index</div>",
                unsafe_allow_html=True)
    comp = data.get_table("conventional_comparison")
    if comp is not None:
        s = comp.iloc[:, 0]
        st.plotly_chart(charts.hbar(s, "AUC vs field-event label", "AUC",
                                    highlight="ours", xrange=(0.45, 0.8)),
                        use_container_width=True, config={"displayModeBar": False})
        st.caption("Conventional severity proxies rank the fleet near chance (0.52–0.56); the "
                   "label-free index reaches ~0.74 — but mind the surveillance-bias floor.")
    else:
        st.info("Run `python scripts/run_health_comparison.py` to generate the comparison table.")
