"""DGA Fleet-Health dashboard — entry point.

Run from the project root:

    streamlit run dashboard/app.py

Wired live to the `dga` package: cleaning, conventional diagnoses, the SD-CAE
compositional representation, and the label-free Health/Risk index (re-ranked
in real time as you change the component weights in the sidebar).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `lib` and `views` importable however the app is launched (streamlit run / AppTest).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(page_title="DGA Fleet Health", layout="wide",
                   initial_sidebar_state="expanded")

from lib import data, theme                                   # noqa: E402
from views import overview, fleet, unit, sdcae, surveillance, methods  # noqa: E402

theme.inject_css()

if "weights" not in st.session_state:
    st.session_state.weights = dict(data.DEFAULT_WEIGHTS)

PAGES = {
    "Overview": overview.render,
    "Fleet ranking": fleet.render,
    "Transformer inspector": unit.render,
    "SD-CAE fault map": sdcae.render,
    "Surveillance bias": surveillance.render,
    "Conventional methods": methods.render,
}

with st.sidebar:
    st.markdown("<div style='font-size:1.15rem;font-weight:800;margin-bottom:2px'>DGA Fleet Health</div>"
                "<div style='color:#9aa3bd;font-size:0.82rem;margin-bottom:10px'>label-free risk · "
                "SD-CAE · surveillance-bias control</div>", unsafe_allow_html=True)
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed", key="nav")

    st.divider()
    st.markdown("**Risk weights** &nbsp; <span style='color:#9aa3bd;font-size:0.78rem'>(live re-rank)</span>",
                unsafe_allow_html=True)
    st.caption("Each unit's risk = weighted sum of four standardised gas signals. "
               "Raise a slider to make that signal count more in the ranking — the whole "
               "fleet re-sorts instantly. (0 = ignore that signal.)")
    w = st.session_state.weights
    w["severity"] = st.slider(
        "Severity", 0.0, 3.0, float(w["severity"]), 0.5,
        help="Current gassing level: hydrogen + CO2 + total combustible gas. How much gas now.")
    w["acetylene"] = st.slider(
        "Acetylene (arcing)", 0.0, 3.0, float(w["acetylene"]), 0.5,
        help="Acetylene (C2H2): the arcing marker — the most dangerous fault. Up-weighted to 2 by default.")
    w["temporal"] = st.slider(
        "Temporal (H2 rate)", 0.0, 3.0, float(w["temporal"]), 0.5,
        help="How fast hydrogen is rising over time (ppm/year, IEEE C57.104 style).")
    w["anomaly"] = st.slider(
        "Anomaly", 0.0, 3.0, float(w["anomaly"]), 0.5,
        help="Self-supervised novelty of the unit's gas state (autoencoder reconstruction + Isolation Forest).")
    if st.button("Reset to defaults", use_container_width=True):
        st.session_state.weights = dict(data.DEFAULT_WEIGHTS)
        st.rerun()

    st.divider()
    st.caption("CESI × KMUTNB · research-initiation project. Numbers are computed live from the "
               "DGA pipeline, not hard-coded.")

PAGES[page]()
