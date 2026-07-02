"""Showcase: the honest finding — the field-event label is surveillance-confounded."""
from __future__ import annotations

import streamlit as st

from lib import charts, data, theme


def render():
    st.markdown("## Surveillance bias — why we don't oversell the risk index")
    st.markdown(
        "<span class='muted'>The label everyone validates DGA risk models against — operator-logged "
        "field events — is <b>confounded</b>. Operators re-sample the units they worry about, so a "
        "trivial <b>sample count</b> predicts a logged event as well as the gas chemistry. This page "
        "makes that visible, and offers a confound-free alternative.</span>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-title'>The confound floor</div>", unsafe_allow_html=True)
        floor = data.get_table("label_confound_floor")
        if floor is not None:
            s = floor.iloc[:, 0]
            st.plotly_chart(charts.hbar(s, "AUC vs field-event label", "AUC",
                                        highlight="confound", xrange=(0.45, 0.82)),
                            use_container_width=True, config={"displayModeBar": False})
            st.caption("A trivial sample count (red) out-predicts the physics index — so any AUC "
                       "against this label must be read against ~0.76.")
        else:
            st.info("Run `python scripts/run_label_confound.py` first.")
    with c2:
        st.markdown("<div class='section-title'>Live ranking validation</div>", unsafe_allow_html=True)
        ev = data.get_eval(**st.session_state.weights)
        st.plotly_chart(charts.decile_bars(ev["decile_note_rate"], ev["base_rate"]),
                        use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Current index AUC {ev['AUC']:.2f} — real signal, but capped by the 0.76 floor.")

    st.markdown("<div class='section-title'>The confound-free alternative: chemistry, not attention</div>",
                unsafe_allow_html=True)
    cc1, cc2 = st.columns([1, 1.1])
    with cc1:
        chem = data.get_table("chemistry_target_arcing_onset")
        if chem is not None:
            s = chem.iloc[:, 0]
            st.plotly_chart(charts.hbar(s, "AUC predicting arcing onset", "AUC",
                                        highlight="confound", xrange=(0.45, 0.72)),
                            use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Run `python scripts/run_label_confound.py` first.")
    with cc2:
        st.markdown(
            "<div class='card'>We define a target that does <b>not</b> depend on operator attention: "
            "the <b>onset of arcing</b> — acetylene appearing within two years in a unit that has "
            "none. Across 3,794 prediction points with only 45 onset events, ethylene and hydrogen "
            "forecast it at AUC <b>0.64 / 0.63</b> (permutation p &lt; 0.01), while the sample count "
            "is at chance (0.49). Modest, but genuine and confound-free — the right way to claim "
            "early-warning value.</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='card'><b>Take-away for the field:</b> any DGA risk model validated on "
            "operator-logged events should report a sample-count confound floor alongside its AUC.</div>",
            unsafe_allow_html=True)
