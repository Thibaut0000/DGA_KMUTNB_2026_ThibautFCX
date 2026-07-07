"""Shared scoring-profile control: named presets + a custom-weights popover.

The four risk-component weights live in st.session_state.weights. This widget
gives them an intuitive surface -- named profiles for everyone, sliders behind
an "Adjust weights" popover for experts -- and sits at the top of every page
that displays risk scores, instead of hiding in the navigation sidebar.
"""
from __future__ import annotations

import streamlit as st

from lib import theme
from lib.data import DEFAULT_WEIGHTS

PRESETS: dict[str, dict] = {
    "Standard (paper)": dict(DEFAULT_WEIGHTS),
    "Arcing focus": {"severity": 0.5, "acetylene": 3.0, "temporal": 0.5, "anomaly": 0.5},
    "Trend watch": {"severity": 0.5, "acetylene": 1.0, "temporal": 3.0, "anomaly": 1.0},
    "Anomaly hunt": {"severity": 0.5, "acetylene": 1.0, "temporal": 0.5, "anomaly": 3.0},
}

PRESET_HELP = ("Standard = the paper's domain weights (acetylene doubled). "
               "Arcing focus = hunt discharge faults. Trend watch = prioritise fast-rising H2. "
               "Anomaly hunt = prioritise unusual gas states.")

_SLIDERS = [
    ("w_sev", "severity", "Severity",
     "Current gassing level: hydrogen + CO2 + total combustible gas. How much gas now."),
    ("w_acet", "acetylene", "Acetylene (arcing)",
     "C2H2, the arcing marker - the most dangerous fault. The paper doubles it by domain choice."),
    ("w_temp", "temporal", "Temporal (H2 rate)",
     "How fast hydrogen is rising over time (ppm/year, IEEE C57.104 style)."),
    ("w_anom", "anomaly", "Anomaly",
     "Self-supervised novelty of the unit's gas state (autoencoder + Isolation Forest)."),
]


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("weights", dict(DEFAULT_WEIGHTS))
    for skey, wkey, _, _ in _SLIDERS:
        ss.setdefault(skey, float(ss["weights"][wkey]))
    ss.setdefault("profile_ctrl", _match(ss["weights"]))


def _match(w: dict) -> str | None:
    for name, p in PRESETS.items():
        if all(abs(float(w[k]) - float(p[k])) < 1e-9 for k in p):
            return name
    return None


def _apply_preset() -> None:
    name = st.session_state.get("profile_ctrl")
    if not name:
        return
    p = PRESETS[name]
    st.session_state.weights = dict(p)
    for skey, wkey, _, _ in _SLIDERS:
        st.session_state[skey] = float(p[wkey])


def _apply_sliders() -> None:
    w = {wkey: float(st.session_state[skey]) for skey, wkey, _, _ in _SLIDERS}
    st.session_state.weights = w
    st.session_state.profile_ctrl = _match(w)


def _popover() -> None:
    with st.popover("Adjust weights", use_container_width=True):
        st.caption("Each transformer's risk score = weighted sum of four standardised gas "
                   "signals (sigma vs fleet). Set a weight to 0 to ignore that signal; the "
                   "whole app re-ranks instantly.")
        for skey, _, label, help_ in _SLIDERS:
            st.slider(label, 0.0, 3.0, step=0.5, key=skey, on_change=_apply_sliders, help=help_)
        if st.button("Reset to paper defaults", key="w_reset", use_container_width=True):
            st.session_state.profile_ctrl = "Standard (paper)"
            _apply_preset()
            st.rerun()


def scoring_profile(compact: bool = False) -> None:
    """Render the scoring-profile control. `compact` = badge + popover only."""
    init_state()
    active = _match(st.session_state.weights) or "Custom"

    if compact:
        c1, c2 = st.columns([5, 1.6])
        c1.markdown(
            "<div style='padding-top:6px'><span class='muted'>Scoring profile:</span> "
            + theme.badge(active, theme.ACCENT) + "</div>",
            unsafe_allow_html=True)
        with c2:
            _popover()
        return

    c1, c2 = st.columns([4.4, 1.4])
    with c1:
        st.segmented_control("Scoring profile", list(PRESETS), key="profile_ctrl",
                             on_change=_apply_preset, help=PRESET_HELP,
                             label_visibility="collapsed")
    with c2:
        _popover()
    if active == "Custom":
        st.caption("Custom weights active — open *Adjust weights* to view or change them.")
