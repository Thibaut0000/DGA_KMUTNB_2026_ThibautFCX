"""Operational view: the full fleet ranked by the label-free risk index."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import data, theme


def render():
    st.markdown("## Fleet risk ranking")
    st.markdown("<span class='muted'>Every transformer scored by the label-free Health/Risk index "
                "(physical gas features only). Adjust the component weights in the sidebar — the "
                "ranking recomputes live.</span>", unsafe_allow_html=True)

    w = st.session_state.weights
    rk = data.get_risk(**w)
    meta = data.get_unit_meta()

    base = pd.DataFrame(index=rk.index)
    base["Rank"] = rk["risk_rank"]
    for mc, label in [("NAME", "Name"), ("MFG", "MFG"), ("KV", "kV")]:
        if mc in meta.columns:
            base[label] = meta[mc]
    base["Risk %ile"] = rk["risk_pct"]
    base["Score"] = rk["risk_score"].round(2)
    base["Sev"] = rk["comp_severity"].round(2)
    base["Acet"] = rk["comp_acetylene"].round(2)
    base["Temp"] = rk["comp_temporal"].round(2)
    base["Anom"] = rk["comp_anomaly"].round(2)
    base["H2"] = rk["latest_H2"].round(0)
    base["C2H2"] = rk["latest_C2H2"].round(0)
    base["TCG"] = rk["latest_TCG"].round(0)
    base["H2/yr"] = rk["rate_H2"].round(1)
    base["n"] = rk["n_samples"]
    base["Event"] = rk["fault_note"].astype(bool)
    base.insert(1, "Unit", base.index.astype(str))

    # ---- filters ----------------------------------------------------------
    f1, f2, f3, f4 = st.columns([1.4, 1, 1, 1.2])
    mfgs = sorted([m for m in meta.get("MFG", pd.Series(dtype=str)).dropna().unique()]) \
        if "MFG" in meta.columns else []
    sel_mfg = f1.multiselect("Manufacturer", mfgs, default=[])
    min_pct = f2.slider("Min risk %ile", 0.0, 1.0, 0.0, 0.05)
    only_event = f3.checkbox("Field-event units only", value=False)
    query = f4.text_input("Search unit / name", "")

    view = base.copy()
    if sel_mfg and "MFG" in view.columns:
        view = view[view["MFG"].isin(sel_mfg)]
    view = view[view["Risk %ile"] >= min_pct]
    if only_event:
        view = view[view["Event"]]
    if query:
        q = query.lower()
        hay = view["Unit"].str.lower()
        if "Name" in view.columns:
            hay = hay + " " + view["Name"].astype(str).str.lower()
        view = view[hay.str.contains(q, na=False)]
    view = view.sort_values("Score", ascending=False)

    st.markdown(f"<span class='muted'>{len(view)} of {len(base)} units · "
                f"{int(view['Event'].sum())} with a logged field event</span>",
                unsafe_allow_html=True)

    st.dataframe(
        view, hide_index=True, use_container_width=True, height=460,
        column_config={
            "Risk %ile": st.column_config.ProgressColumn(
                "Risk %ile", min_value=0.0, max_value=1.0, format="%.2f"),
            "Score": st.column_config.NumberColumn("Score", format="%.2f"),
            "C2H2": st.column_config.NumberColumn("C2H2", help="latest acetylene (ppm) — arcing marker"),
            "Event": st.column_config.CheckboxColumn("Event", help="logged field-event (weak label)"),
        },
    )

    d1, d2 = st.columns([1, 1.3])
    d1.download_button("Download ranking (CSV)", view.to_csv(index=False).encode("utf-8"),
                       "fleet_risk_ranking.csv", "text/csv", use_container_width=True)

    # ---- risk vs acetylene scatter ---------------------------------------
    st.markdown("<div class='section-title'>Risk vs. acetylene (arcing marker)</div>",
                unsafe_allow_html=True)
    fig = go.Figure()
    for flag, name, col in [(False, "no event", "#9aa4b2"), (True, "field event", theme.BAD)]:
        d = rk[rk["fault_note"].astype(bool) == flag]
        fig.add_trace(go.Scatter(
            x=d["risk_score"], y=np.maximum(d["latest_C2H2"], 0.1), mode="markers", name=name,
            marker=dict(size=7, color=col, opacity=0.65, line=dict(width=0)),
            text=d.index.astype(str),
            hovertemplate="unit %{text}<br>risk %{x:.2f}<br>C2H2 %{y:.0f} ppm<extra></extra>"))
    fig.update_yaxes(type="log", title="latest C2H2 (ppm, log)")
    fig.update_xaxes(title="label-free risk score")
    st.plotly_chart(theme.style_fig(fig, height=380), use_container_width=True,
                    config={"displayModeBar": False})
