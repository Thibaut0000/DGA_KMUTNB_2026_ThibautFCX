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
            st.caption("A trivial sample count (red, 0.76) matches the physics index (0.74) — the "
                       "difference is not significant (paired bootstrap p = 0.58). Any AUC against "
                       "this label must be read against that floor.")
        else:
            st.info("Run `python scripts/run_label_confound.py` first.")
    with c2:
        st.markdown("<div class='section-title'>Live ranking validation</div>", unsafe_allow_html=True)
        ev = data.get_eval(**st.session_state.weights)
        st.plotly_chart(charts.decile_bars(ev["decile_note_rate"], ev["base_rate"]),
                        use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Current index AUC {ev['AUC']:.2f}. In a joint model the sample count adds "
                   "signal far beyond the physics (likelihood-ratio p = 9e-13) while the physics "
                   "retains a small real increment (p = 0.003): dominated, but not pure confound.")

    st.markdown(
        "<div class='card'><b>Label repair check.</b> 69% of the field notes contain Thai text "
        "that the English keyword label misses. Translating them adds six own-unit electrical "
        "events — and on this repaired label the count's advantage disappears entirely "
        "(AUC 0.734 vs 0.735): part of the confound's edge was label incompleteness.</div>",
        unsafe_allow_html=True)

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
            "none. Positives are rare and clustered (45 points from 17 onset units), so we use "
            "unit-blocked intervals: ethylene and hydrogen forecast onset at AUC <b>0.64</b> "
            "(95% CI 0.52–0.77) and <b>0.63</b> (0.53–0.73) — both excluding chance — while the "
            "sample count is uninformative (0.49). Robust across nine threshold-and-horizon "
            "variants (0.64–0.73).</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='card'><b>Take-away for the field:</b> any DGA risk model validated on "
            "operator-logged events should report a sample-count confound floor alongside its AUC.</div>",
            unsafe_allow_html=True)
