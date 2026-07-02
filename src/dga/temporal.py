"""Per-transformer temporal DGA features for fleet health ranking.

The dataset is longitudinal (628 units, mean 7.3 samples/unit over ~4.5 years at a
~6-month interval), so each unit has a gas *trajectory*, not a single snapshot.
For every unit we summarise that trajectory into:

  * latest_<gas>  : most recent concentration (current state)
  * rate_<gas>    : generation rate in ppm/year, by multi-point linear regression
                    over the last <=6 samples within <=2 years (IEEE C57.104 style)
  * rise_<gas>    : log1p(latest) - log1p(median of earlier samples) -- a zero-safe
                    log rise vs the unit's own baseline

plus TCG (total combustible gas = H2+CH4+C2H2+C2H4+C2H6+CO) and acetylene aggregates
(C2H2 = arcing, the most dangerous fault). Returns one row per transformer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config

# Total combustible gas (CO2 is not combustible -> excluded), per IEEE C57.104.
TCG_GASES = ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO"]


def generation_rate(days: np.ndarray, values: np.ndarray, max_points: int = 6,
                    max_years: float = 2.0, min_points: int = 3) -> float:
    """ppm/year slope by linear regression over the recent window (C57.104-style).

    `days` = day offsets (any origin), `values` = concentrations, both ascending.
    Uses the last <=max_points samples within max_years of the latest; falls back to
    all available points if fewer than min_points lie in the window. C57.104 specifies
    3-6 points, so rates from <min_points samples (noisy single-interval slopes) are
    returned as NaN; NaN also if the time span is zero.
    """
    days = np.asarray(days, float)
    values = np.asarray(values, float)
    if len(days) < min_points:
        return np.nan
    win = days >= days[-1] - 365.25 * max_years
    d, v = (days[win], values[win]) if win.sum() >= min_points else (days, values)
    d, v = d[-max_points:], v[-max_points:]
    t = (d - d[0]) / 365.25
    if t[-1] <= 0:
        return np.nan
    return float(np.polyfit(t, v, 1)[0])


def _unit_features(g: pd.DataFrame, gases: list[str]) -> dict:
    g = g.sort_values("_day")
    days = g["_day"].to_numpy()
    out = {"n_samples": len(g),
           "span_years": float((days[-1] - days[0]) / 365.25) if len(g) > 1 else 0.0,
           "latest_date": g["_date"].iloc[-1]}
    for col in [*gases, "TCG"]:
        v = g[col].to_numpy(float)
        latest = v[-1]
        baseline = np.median(v[:-1]) if len(v) > 1 else latest
        out[f"latest_{col}"] = latest
        out[f"rate_{col}"] = generation_rate(days, v)
        out[f"rise_{col}"] = float(np.log1p(latest) - np.log1p(baseline))
    out["c2h2_max"] = float(g["C2H2"].max())          # acetylene ever seen (arcing)
    return out


def build_temporal(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """One row per transformer: latest state + generation rates + rises (see module doc)."""
    cfg = cfg or load_config()
    gases = list(cfg.data.feature_gases)
    work = df.copy()
    work["_date"] = pd.to_datetime(work["Sample Day"], errors="coerce")
    # units with no parsable dates can't be temporal; keep their single latest row order
    work["_date"] = work["_date"].fillna(pd.Timestamp("1900-01-01"))
    work["_day"] = (work["_date"] - pd.Timestamp("1900-01-01")).dt.days
    for col in set(gases) | set(TCG_GASES):
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    work["TCG"] = work[TCG_GASES].sum(axis=1)

    rows = {cid: _unit_features(g, gases) for cid, g in work.groupby("CODETX")}
    feats = pd.DataFrame.from_dict(rows, orient="index")
    feats.index.name = "CODETX"
    # carry the unit-level weak label if present
    if "fault_note" in df.columns:
        feats["fault_note"] = df.groupby("CODETX")["fault_note"].any()
    return feats


if __name__ == "__main__":  # python -m dga.temporal
    from dga import data as dga_data
    t = build_temporal(dga_data.load_clean())
    print("per-unit temporal features:", t.shape)
    print("rate defined for:", int(t["rate_TCG"].notna().sum()), "/", len(t), "units")
    print(t[["n_samples", "span_years", "latest_TCG", "rate_TCG", "rise_C2H2", "c2h2_max"]].describe().round(2))
