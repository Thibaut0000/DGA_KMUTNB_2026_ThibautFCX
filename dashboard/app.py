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

from lib import controls, data, theme                          # noqa: E402
from views import overview, fleet, unit, sdcae, surveillance, methods  # noqa: E402

theme.inject_css()
controls.init_state()

PAGES = {
    "Overview": overview.render,
    "Fleet ranking": fleet.render,
    "Transformer inspector": unit.render,
    "Fault-type map": sdcae.render,
    "Surveillance bias": surveillance.render,
    "Conventional methods": methods.render,
}

with st.sidebar:
    st.markdown("<div style='font-size:1.15rem;font-weight:800;margin-bottom:2px'>DGA Fleet Health</div>"
                "<div style='color:#9aa3bd;font-size:0.82rem;margin-bottom:10px'>label-free risk · "
                "compositional fault typing · surveillance-bias control</div>",
                unsafe_allow_html=True)
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed", key="nav")

    st.divider()
    if data.using_public_data():
        st.markdown(theme.badge("anonymised demo data", theme.WARN), unsafe_allow_html=True)
        st.caption("Unit ids are hashed; free-text notes and nameplate metadata are excluded.")
    else:
        st.markdown(theme.badge("full dataset (local)", theme.GOOD), unsafe_allow_html=True)
    st.caption("CESI × KMUTNB · research-initiation project. Every number is computed live "
               "from the DGA pipeline — nothing is hard-coded.")

PAGES[page]()
