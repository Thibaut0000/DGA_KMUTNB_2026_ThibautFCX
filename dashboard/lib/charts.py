"""Reusable Plotly figures shared across views."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def duval_triangle(samples: pd.DataFrame, highlight: dict | None = None) -> go.Figure:
    """Duval Triangle 1 as a ternary scatter (%CH4, %C2H4, %C2H2), coloured by class.

    `samples` needs columns CH4, C2H4, C2H2, duval. `highlight` (optional) is a
    dict with CH4/C2H4/C2H2 for the focus unit, drawn as a large star.
    """
    s = samples.copy()
    for c in ("CH4", "C2H4", "C2H2"):
        s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0).clip(lower=0)
    tot = (s["CH4"] + s["C2H4"] + s["C2H2"]).replace(0, np.nan)
    s = s.assign(pCH4=100 * s["CH4"] / tot, pC2H4=100 * s["C2H4"] / tot,
                 pC2H2=100 * s["C2H2"] / tot).dropna(subset=["pCH4"])

    fig = go.Figure()
    for cls in ["PD", "T1", "T2", "T3", "D1", "D2", "DT", "ND"]:
        m = s["duval"].astype(str) == cls
        if not m.any():
            continue
        d = s[m]
        fig.add_trace(go.Scatterternary(
            a=d["pC2H2"], b=d["pCH4"], c=d["pC2H4"], mode="markers", name=cls,
            marker=dict(size=6, color=theme.DUVAL_COLORS.get(cls, "#888"), opacity=0.6,
                        line=dict(width=0)),
            hovertemplate=f"<b>{cls}</b><br>%C2H2 %{{a:.0f}} · %CH4 %{{b:.0f}} · %C2H4 %{{c:.0f}}<extra></extra>",
        ))
    if highlight:
        ch4, c2h4, c2h2 = (max(highlight.get(g, 0) or 0, 0) for g in ("CH4", "C2H4", "C2H2"))
        t = ch4 + c2h4 + c2h2
        if t > 0:
            fig.add_trace(go.Scatterternary(
                a=[100 * c2h2 / t], b=[100 * ch4 / t], c=[100 * c2h4 / t], mode="markers",
                name="selected unit",
                marker=dict(symbol="star", size=20, color="#111827",
                            line=dict(width=1.5, color="white")),
                hovertemplate="selected unit<extra></extra>",
            ))
    fig.update_layout(
        ternary=dict(
            sum=100,
            aaxis=dict(title="% C2H2", min=0, linewidth=1, ticks="outside"),
            baxis=dict(title="% CH4", min=0, linewidth=1, ticks="outside"),
            caxis=dict(title="% C2H4", min=0, linewidth=1, ticks="outside"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return theme.style_fig(fig, height=460)


def risk_gauge(score: float, pct: float) -> go.Figure:
    """Gauge of a unit's risk percentile within the fleet."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(pct * 100, 1),
        number=dict(suffix=" %", font=dict(size=30)),
        title=dict(text="fleet risk percentile", font=dict(size=13)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1),
            bar=dict(color=theme.risk_hex(pct), thickness=0.3),
            bgcolor="white", borderwidth=0,
            steps=[dict(range=[0, 50], color="#e8f5e9"),
                   dict(range=[50, 80], color="#fff8e1"),
                   dict(range=[80, 100], color="#ffebee")],
            threshold=dict(line=dict(color="#111827", width=3), value=pct * 100),
        ),
    ))
    return theme.style_fig(fig, height=240, legend=False)


def component_radar(values: dict) -> go.Figure:
    """Radar of the four standardised risk components for one unit."""
    labels = ["Severity", "Acetylene", "Temporal", "Anomaly"]
    keys = ["comp_severity", "comp_acetylene", "comp_temporal", "comp_anomaly"]
    vals = [float(values.get(k, 0.0)) for k in keys]
    vals_c = vals + [vals[0]]
    labels_c = labels + [labels[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals_c, theta=labels_c, fill="toself",
        fillcolor="rgba(79,70,229,0.18)", line=dict(color=theme.ACCENT, width=2),
        hovertemplate="%{theta}: %{r:.2f} σ<extra></extra>",
    ))
    fig.update_layout(polar=dict(
        radialaxis=dict(showticklabels=True, tickfont=dict(size=10), gridcolor="#eef0f4"),
        angularaxis=dict(gridcolor="#eef0f4")))
    return theme.style_fig(fig, height=300, legend=False)


def gas_history(hist: pd.DataFrame, gases: list[str], logy: bool = True) -> go.Figure:
    """Per-unit gas concentration trajectory over time."""
    fig = go.Figure()
    for g in gases:
        if g not in hist.columns:
            continue
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist[g], mode="lines+markers", name=g,
            line=dict(color=theme.GAS_COLORS.get(g, "#888"), width=2),
            marker=dict(size=6),
            hovertemplate=f"<b>{g}</b><br>%{{x|%Y-%m-%d}} · %{{y:.0f}} ppm<extra></extra>",
        ))
    fig.update_layout(yaxis_title="ppm" + (" (log)" if logy else ""), xaxis_title=None,
                      hovermode="x unified")
    if logy:
        fig.update_yaxes(type="log")
    return theme.style_fig(fig, height=340)


def hbar(series: pd.Series, title: str, xlabel: str, highlight: str | None = None,
         xrange: tuple | None = None) -> go.Figure:
    """Horizontal bar chart (e.g. AUC comparisons), optional highlighted row."""
    s = series.sort_values()
    colors = [theme.BAD if (highlight and highlight.lower() in str(i).lower())
              else theme.ACCENT for i in s.index]
    fig = go.Figure(go.Bar(
        x=s.values, y=[str(i) for i in s.index], orientation="h",
        marker=dict(color=colors), text=[f"{v:.2f}" for v in s.values],
        textposition="outside", cliponaxis=False,
    ))
    fig.update_layout(title=title, xaxis_title=xlabel)
    if xrange:
        fig.update_xaxes(range=list(xrange))
    return theme.style_fig(fig, height=max(220, 40 * len(s) + 80), legend=False)


def decile_bars(rates: list[float], base: float) -> go.Figure:
    """Fault-note rate by risk decile (D1 = riskiest)."""
    rates = np.array(rates) * 100
    fig = go.Figure(go.Bar(
        x=[f"D{i+1}" for i in range(len(rates))], y=rates,
        marker=dict(color=[theme.risk_hex(1 - i / (len(rates) - 1)) for i in range(len(rates))]),
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=base * 100, line_dash="dash", line_color="#111827",
                  annotation_text=f"fleet base {base*100:.1f}%", annotation_position="top right")
    fig.update_layout(title="Fault-event rate by risk decile (D1 = riskiest)",
                      yaxis_title="fault-note rate (%)", xaxis_title=None)
    return theme.style_fig(fig, height=320, legend=False)


def sdcae_scatter(df: pd.DataFrame, color_by: str) -> go.Figure:
    """2-D SD-CAE latent scatter coloured by Duval class / cluster / severity."""
    fig = go.Figure()
    gas_cols = [c for c in df.columns if c.startswith("pct_")]
    cd = "<br>".join(f"{c[4:]} %{{customdata[{i}]}}%" for i, c in enumerate(gas_cols))
    custom = df[gas_cols].to_numpy()
    if color_by == "magnitude":
        fig.add_trace(go.Scatter(
            x=df["z1"], y=df["z2"], mode="markers", customdata=custom,
            marker=dict(size=6, color=df["magnitude"], colorscale="Turbo", opacity=0.7,
                        colorbar=dict(title="severity m"), line=dict(width=0)),
            hovertemplate="z=(%{x:.2f}, %{y:.2f})<br>" + cd + "<extra></extra>",
        ))
    else:
        cmap = theme.DUVAL_COLORS if color_by == "duval" else None
        for key in sorted(df[color_by].unique(), key=lambda v: (len(str(v)), str(v))):
            d = df[df[color_by] == key]
            cdd = d[gas_cols].to_numpy()
            fig.add_trace(go.Scatter(
                x=d["z1"], y=d["z2"], mode="markers", name=str(key), customdata=cdd,
                marker=dict(size=6, opacity=0.7, line=dict(width=0),
                            color=(cmap.get(str(key)) if cmap else None)),
                hovertemplate=f"<b>{color_by} {key}</b><br>z=(%{{x:.2f}}, %{{y:.2f}})<br>" + cd + "<extra></extra>",
            ))
    fig.update_layout(title="SD-CAE learned diagnostic map", xaxis_title="z₁", yaxis_title="z₂")
    return theme.style_fig(fig, height=520)
