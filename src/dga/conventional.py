"""Conventional rule-based DGA diagnostics — the baselines the unsupervised
framework is compared against in the paper.

Implemented:
    * Duval Triangle 1   (IEC 60599 / Duval 2002)         -> PD, D1, D2, T1, T2, T3, DT
    * IEC 60599 ratios   (C2H2/C2H4, CH4/H2, C2H4/C2H6)   -> PD, D1, D2, T1, T2, T3, ND
    * Rogers ratios      (3-ratio variant)                -> coarse fault family

NOTE ON BOUNDARIES: published Duval/IEC boundary values vary by a few percent
between sources and revisions. The thresholds below follow the commonly cited
IEC 60599 / Duval coordinates and are adequate as a reference baseline; cite the
exact edition you adopt in the paper and keep this module as the single source
of truth so results stay reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FAULT_LABELS = {
    "PD": "Partial discharge",
    "D1": "Low-energy discharge",
    "D2": "High-energy discharge",
    "T1": "Thermal fault < 300 C",
    "T2": "Thermal fault 300-700 C",
    "T3": "Thermal fault > 700 C",
    "DT": "Mixed thermal/electrical",
    "ND": "Not determined",
}


def duval_triangle_1(ch4: float, c2h4: float, c2h2: float) -> str | None:
    """Classify a sample with Duval Triangle 1 using the three key gases."""
    total = (ch4 or 0) + (c2h4 or 0) + (c2h2 or 0)
    if total <= 0:
        return None
    m = 100.0 * (ch4 or 0) / total   # %CH4
    e = 100.0 * (c2h4 or 0) / total  # %C2H4
    a = 100.0 * (c2h2 or 0) / total  # %C2H2

    if m >= 98.0:
        return "PD"
    if a >= 13.0 and e <= 23.0:
        return "D1"
    if a >= 13.0 and e <= 40.0:
        return "D2"
    if a > 29.0 and e > 40.0:
        return "D2"
    if a <= 4.0 and e <= 20.0:
        return "T1"
    if a <= 4.0 and 20.0 < e <= 50.0:
        return "T2"
    if a <= 15.0 and e > 50.0:
        return "T3"
    return "DT"


def _ratio(num: float, den: float) -> float:
    num, den = num or 0.0, den or 0.0
    if den == 0:
        return np.inf if num > 0 else 0.0
    return num / den


def iec_60599(h2, ch4, c2h2, c2h4, c2h6) -> str:
    """IEC 60599 three-ratio method. Returns 'ND' when no rule matches."""
    r1 = _ratio(c2h2, c2h4)  # C2H2/C2H4
    r2 = _ratio(ch4, h2)     # CH4/H2
    r5 = _ratio(c2h4, c2h6)  # C2H4/C2H6

    if r2 < 0.1 and r5 < 0.2:
        return "PD"
    if r1 > 1.0 and 0.1 <= r2 <= 0.5 and r5 > 1.0:
        return "D1"
    if 0.6 <= r1 <= 2.5 and 0.1 <= r2 <= 1.0 and r5 > 2.0:
        return "D2"
    if r1 < 0.1 and r2 > 1.0 and r5 < 1.0:
        return "T1"
    if r1 < 0.1 and r2 > 1.0 and 1.0 <= r5 <= 4.0:
        return "T2"
    if r1 < 0.2 and r2 > 1.0 and r5 > 4.0:
        return "T3"
    return "ND"


def rogers_ratios(h2, ch4, c2h2, c2h4, c2h6) -> str:
    """Coarse Rogers 3-ratio classification (family level)."""
    r1 = _ratio(c2h2, c2h4)
    r2 = _ratio(ch4, h2)
    r5 = _ratio(c2h4, c2h6)
    if r2 < 0.1:
        return "PD"
    if r1 > 0.1:
        return "D2" if r5 > 1 else "D1"
    if r5 < 1.0:
        return "T1"
    if r5 < 3.0:
        return "T2"
    return "T3"


def diagnose(df: pd.DataFrame) -> pd.DataFrame:
    """Run all three rule-based methods on a cleaned DGA frame.

    Returns a DataFrame (indexed like `df`) with columns duval/iec/rogers.
    """
    def _row(r):
        return pd.Series({
            "duval": duval_triangle_1(r.get("CH4"), r.get("C2H4"), r.get("C2H2")),
            "iec": iec_60599(r.get("H2"), r.get("CH4"), r.get("C2H2"),
                             r.get("C2H4"), r.get("C2H6")),
            "rogers": rogers_ratios(r.get("H2"), r.get("CH4"), r.get("C2H2"),
                                    r.get("C2H4"), r.get("C2H6")),
        })

    return df.apply(_row, axis=1)
