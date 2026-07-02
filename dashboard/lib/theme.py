"""Visual identity for the dashboard: colour palette, CSS, Plotly styling.

Kept in one place so every view looks consistent. No project logic here.
"""
from __future__ import annotations

import streamlit as st

# ---- palette -------------------------------------------------------------
ACCENT = "#4f46e5"          # indigo
ACCENT_SOFT = "#eef0fd"
INK = "#1c2128"
MUTED = "#5b6470"
GOOD = "#16a34a"
WARN = "#f59e0b"
BAD = "#dc2626"
CARD_BORDER = "#e7e9f0"

# Duval fault-class colours (shared with the SD-CAE map and the triangle).
DUVAL_COLORS = {
    "PD": "#7e57c2", "D1": "#ef5350", "D2": "#b71c1c",
    "T1": "#ffb300", "T2": "#fb8c00", "T3": "#e64a19",
    "DT": "#6d4c41", "ND": "#b0bec5",
}

# Continuous risk colour scale (green -> amber -> red) for Plotly.
RISK_SCALE = [(0.0, "#16a34a"), (0.5, "#f59e0b"), (1.0, "#dc2626")]

GAS_COLORS = {
    "H2": "#1e88e5", "CH4": "#43a047", "C2H2": "#e53935", "C2H4": "#fb8c00",
    "C2H6": "#8e24aa", "CO": "#00897b", "CO2": "#6d4c41", "TCG": "#37474f",
}


def risk_hex(p: float) -> str:
    """Map a 0..1 risk percentile to a green->amber->red hex colour."""
    p = max(0.0, min(1.0, float(p)))
    if p < 0.5:
        t = p / 0.5
        c0, c1 = (0x16, 0xa3, 0x4a), (0xf5, 0x9e, 0x0b)
    else:
        t = (p - 0.5) / 0.5
        c0, c1 = (0xf5, 0x9e, 0x0b), (0xdc, 0x26, 0x26)
    r, g, b = (int(c0[i] + (c1[i] - c0[i]) * t) for i in range(3))
    return f"#{r:02x}{g:02x}{b:02x}"


def style_fig(fig, height: int | None = None, legend: bool = True):
    """Apply the shared Plotly look to a figure."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", size=13, color=INK),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend=dict(bgcolor="rgba(255,255,255,0.6)", bordercolor=CARD_BORDER, borderwidth=1),
        colorway=[ACCENT, "#06b6d4", "#f59e0b", "#ef4444", "#10b981", "#8b5cf6"],
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor="#eef0f4", zerolinecolor="#e0e3ea")
    fig.update_yaxes(gridcolor="#eef0f4", zerolinecolor="#e0e3ea")
    return fig


def badge(text: str, color: str) -> str:
    """An inline coloured pill (HTML)."""
    return (f"<span style='background:{color}22;color:{color};border:1px solid {color}55;"
            f"padding:2px 9px;border-radius:999px;font-size:0.78rem;font-weight:600;"
            f"white-space:nowrap'>{text}</span>")


def inject_css():
    """Global CSS: typography, cards, hero, metric styling."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"]  { font-family: 'Inter', -apple-system, Segoe UI, sans-serif; }
        .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1280px; }

        /* hero header */
        .hero {
            background: linear-gradient(120deg, #4f46e5 0%, #6366f1 45%, #06b6d4 120%);
            color: white; border-radius: 18px; padding: 22px 26px; margin-bottom: 18px;
            box-shadow: 0 10px 30px -12px rgba(79,70,229,0.55);
        }
        .hero h1 { color: white; font-size: 1.55rem; font-weight: 800; margin: 0 0 4px 0; letter-spacing:-0.02em;}
        .hero p  { color: #e8eaff; font-size: 0.95rem; margin: 0; }

        /* metric cards */
        .kpi {
            background: #fff; border: 1px solid #e7e9f0; border-radius: 14px;
            padding: 14px 16px; height: 100%;
            box-shadow: 0 1px 2px rgba(16,24,40,0.04);
        }
        .kpi .label { color:#5b6470; font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:0.04em;}
        .kpi .value { color:#111827; font-size:1.7rem; font-weight:800; line-height:1.1; margin-top:2px;}
        .kpi .sub   { color:#8a93a2; font-size:0.78rem; margin-top:2px;}

        /* generic card */
        .card {
            background:#fff; border:1px solid #e7e9f0; border-radius:14px; padding:16px 18px;
            box-shadow:0 1px 2px rgba(16,24,40,0.04); margin-bottom:8px;
        }
        .section-title { font-weight:700; font-size:1.05rem; color:#1c2128; margin:6px 0 2px 0; }
        .muted { color:#5b6470; font-size:0.9rem; }

        /* sidebar */
        section[data-testid="stSidebar"] { background:#0f1222; }
        section[data-testid="stSidebar"] * { color:#e6e8f2; }
        section[data-testid="stSidebar"] .stRadio label { color:#cdd2e6 !important; }
        section[data-testid="stSidebar"] .stButton button {
            background:#1b1f3a; color:#e6e8f2 !important; border:1px solid #3a4170; border-radius:8px;
            font-weight:600;
        }
        section[data-testid="stSidebar"] .stButton button:hover {
            background:#262c52; color:#fff !important; border-color:#4f46e5;
        }

        /* dataframe rounding */
        [data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; border:1px solid #e7e9f0; }

        /* tighten metric default */
        [data-testid="stMetricValue"] { font-size:1.6rem; font-weight:800; }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str):
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p></div>",
                unsafe_allow_html=True)


def kpi(col, label: str, value: str, sub: str = ""):
    col.markdown(
        f"<div class='kpi'><div class='label'>{label}</div>"
        f"<div class='value'>{value}</div><div class='sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )
