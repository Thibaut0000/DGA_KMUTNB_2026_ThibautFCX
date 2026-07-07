"""Per-transformer drill-down: history, risk breakdown, Duval, diagnoses, notes."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import charts, data, theme
from lib.controls import scoring_profile

_GASES = ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2", "TCG"]


def render():
    st.markdown("## Transformer inspector")
    scoring_profile(compact=True)
    w = st.session_state.weights
    rk = data.get_risk(**w).sort_values("risk_score", ascending=False)
    meta = data.get_unit_meta()

    def _label(cid):
        name = ""
        if "NAME" in meta.columns and pd.notna(meta.loc[cid, "NAME"]):
            name = f" — {meta.loc[cid, 'NAME']}"
        return f"{cid}{name}  ·  risk {rk.loc[cid, 'risk_pct']*100:.0f}%"

    units = rk.index.tolist()
    pick = st.selectbox("Transformer", units, index=0, format_func=_label)
    row = rk.loc[pick]

    diag = data.get_clean_diag()
    hist = diag[diag["CODETX"] == pick].copy()
    hist["date"] = pd.to_datetime(hist["Sample Day"], errors="coerce")
    for g in _GASES:
        if g in hist.columns:
            hist[g] = pd.to_numeric(hist[g], errors="coerce")
    hist = hist.sort_values("date")
    latest = hist.iloc[-1] if len(hist) else None

    # ---- header: identity + gauge ----------------------------------------
    h1, h2 = st.columns([1.6, 1])
    with h1:
        bits = []
        for mc, lab in [("MFG", "Manufacturer"), ("KV", "Voltage"), ("MVA", "Rating"),
                        ("YEAR_Energized", "Energised")]:
            if mc in meta.columns and pd.notna(meta.loc[pick, mc]):
                bits.append(f"<b>{lab}:</b> {meta.loc[pick, mc]}")
        event = bool(row["fault_note"])
        evt = theme.badge("FIELD EVENT LOGGED", theme.BAD) if event else theme.badge("no event", theme.GOOD)
        rank_badge = theme.badge("rank #{} of {}".format(int(row["risk_rank"]), len(rk)), theme.ACCENT)
        n_badge = theme.badge("{} samples".format(int(row["n_samples"])), "#64748b")
        meta_line = " · ".join(bits) or "no nameplate metadata"
        st.markdown(
            "<div class='card'><div style='font-size:1.3rem;font-weight:800'>" + str(pick) + "</div>"
            "<div class='muted' style='margin:4px 0'>" + meta_line + "</div>"
            "<div style='margin-top:6px'>" + evt + " &nbsp; " + rank_badge + " &nbsp; " + n_badge
            + "</div></div>",
            unsafe_allow_html=True)
        m = st.columns(3)
        theme.kpi(m[0], "Latest H2", f"{row['latest_H2']:.0f}", "ppm · early-fault gas")
        theme.kpi(m[1], "Latest C2H2", f"{row['latest_C2H2']:.0f}", "ppm · arcing marker")
        theme.kpi(m[2], "H2 rate", f"{row['rate_H2']:.1f}", "ppm/yr (C57.104)")
    with h2:
        st.plotly_chart(charts.risk_gauge(row["risk_score"], row["risk_pct"]),
                        use_container_width=True, config={"displayModeBar": False})

    # ---- risk breakdown + gas history ------------------------------------
    b1, b2 = st.columns([1, 1.5])
    with b1:
        st.markdown("<div class='section-title'>Why this risk score</div>", unsafe_allow_html=True)
        st.plotly_chart(charts.component_radar(row.to_dict()), use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("Standardised components (σ vs fleet). Acetylene is up-weighted by domain choice.")
    with b2:
        st.markdown("<div class='section-title'>Gas trajectory</div>", unsafe_allow_html=True)
        gases = st.multiselect("Gases", _GASES, default=["H2", "CH4", "C2H4", "C2H2", "TCG"],
                               key="unit_gases")
        logy = st.toggle("log scale", value=True, key="unit_logy")
        if len(hist) and gases:
            st.plotly_chart(charts.gas_history(hist, gases, logy), use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.info("No dated samples for this unit.")

    # ---- Duval triangle + conventional diagnoses -------------------------
    d1, d2 = st.columns([1.1, 1])
    with d1:
        st.markdown("<div class='section-title'>Duval Triangle 1</div>", unsafe_allow_html=True)
        hl = None
        if latest is not None:
            hl = {g: latest.get(g) for g in ("CH4", "C2H4", "C2H2")}
        st.plotly_chart(charts.duval_triangle(hist, hl), use_container_width=True,
                        config={"displayModeBar": False})
    with d2:
        st.markdown("<div class='section-title'>Conventional diagnoses (latest)</div>",
                    unsafe_allow_html=True)
        if latest is not None:
            from dga import conventional as _conv
            for method, val in [("Duval Triangle 1", latest.get("duval")),
                                ("IEC 60599", latest.get("iec")),
                                ("Rogers ratios", latest.get("rogers"))]:
                code = str(val) if pd.notna(val) else "ND"
                desc = _conv.FAULT_LABELS.get(code, "—")
                color = theme.DUVAL_COLORS.get(code, "#64748b")
                st.markdown(
                    f"<div class='card' style='padding:10px 14px;margin-bottom:6px'>"
                    f"<span class='muted'>{method}</span><br>{theme.badge(code, color)} "
                    f"<span style='font-weight:600'>{desc}</span></div>", unsafe_allow_html=True)
        st.caption("Rules agree only partially — they are baselines, not ground truth.")

    # ---- field notes ------------------------------------------------------
    notes = hist[hist["NB"].notna()] if "NB" in hist.columns else hist.iloc[0:0]
    st.markdown("<div class='section-title'>Field notes (operator log)</div>", unsafe_allow_html=True)
    if len(notes):
        nd = notes[["date", "NB", "fault_note"]].rename(
            columns={"date": "Date", "NB": "Note", "fault_note": "Fault event"})
        st.dataframe(nd, hide_index=True, use_container_width=True,
                     column_config={"Fault event": st.column_config.CheckboxColumn("Fault event")})
    else:
        st.caption("No operator notes recorded for this unit.")
