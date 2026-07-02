"""Landing page: fleet KPIs, risk distribution, validation, top units."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from lib import charts, data, theme


def render():
    theme.hero(
        "Transformer Fleet Health — DGA Risk Intelligence",
        "Label-free ranking of a 628-transformer fleet from dissolved-gas analysis, "
        "with SD-CAE compositional fault typing and an honest surveillance-bias control.",
    )

    w = st.session_state.weights
    s = data.fleet_summary()
    ev = data.get_eval(**w)

    c = st.columns(5)
    theme.kpi(c[0], "Transformers", f"{s['n_units']:,}", f"{s['n_mfg']} manufacturers")
    theme.kpi(c[1], "DGA samples", f"{s['n_samples']:,}", "2019–2024")
    theme.kpi(c[2], "Acetylene seen", f"{s['pct_acetylene']*100:.0f}%", "arcing marker, fleet")
    theme.kpi(c[3], "Ranking AUC", f"{ev['AUC']:.2f}", "vs field-event label")
    theme.kpi(c[4], "High-risk units", f"{s['n_high']}", "top risk decile")

    st.write("")
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("<div class='section-title'>Fleet risk distribution</div>", unsafe_allow_html=True)
        rk = data.get_risk(**w)
        fig = go.Figure(go.Histogram(
            x=rk["risk_score"], nbinsx=40, marker=dict(color=theme.ACCENT, opacity=0.85),
            hovertemplate="risk score %{x:.2f}<br>%{y} units<extra></extra>"))
        fig.update_layout(xaxis_title="label-free risk score", yaxis_title="units")
        st.plotly_chart(theme.style_fig(fig, height=320, legend=False),
                        use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown("<div class='section-title'>Does the ranking track real events?</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(charts.decile_bars(ev["decile_note_rate"], ev["base_rate"]),
                        use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='section-title'>Riskiest units right now</div>", unsafe_allow_html=True)
    meta = data.get_unit_meta().drop(columns=["n_samples"], errors="ignore")
    top = rk.sort_values("risk_score", ascending=False).head(10)
    show = top.join(meta, how="left")
    cols = {"risk_rank": "Rank", "risk_pct": "Risk %ile", "latest_H2": "H2",
            "latest_C2H2": "C2H2", "latest_TCG": "TCG", "rate_H2": "H2 rate/yr",
            "fault_note": "Event"}
    disp = show.reset_index()[["CODETX"] + [c for c in cols]].rename(columns={**cols, "CODETX": "Unit"})
    for mc in ("NAME", "MFG"):
        if mc in show.columns:
            disp.insert(1, mc.title(), show[mc].to_numpy())
    st.dataframe(
        disp, hide_index=True, use_container_width=True,
        column_config={
            "Risk %ile": st.column_config.ProgressColumn(
                "Risk %ile", min_value=0.0, max_value=1.0, format="%.2f"),
            "Event": st.column_config.CheckboxColumn("Event"),
        },
    )

    st.warning(
        "**Read this honestly.** The ranking is validated against operator-logged field events, "
        "but that label is **surveillance-confounded**: a trivial sample count predicts it as well "
        "as the gas chemistry. Always read the AUC above against that confound floor — see the "
        "*Surveillance bias* page.",
    )
