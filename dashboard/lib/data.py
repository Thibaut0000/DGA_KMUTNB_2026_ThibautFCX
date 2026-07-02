"""Cached data layer: wires the dashboard to the real `dga` package.

Everything the views show is computed here from the actual pipeline (no
hard-coded numbers): cleaning, conventional diagnoses, the per-unit temporal
features, the label-free risk index (recomputed live when weights change), and
the SD-CAE compositional latent. Heavy steps are cached.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st

# --- make the project's `dga` package importable (mirrors scripts/_bootstrap) ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dga.config import load_config, set_seed                      # noqa: E402
from dga import data as dga_data, conventional, health, clustering  # noqa: E402
from dga.compositional import build_composition                   # noqa: E402
from dga.models.sdcae import train_sdcae                          # noqa: E402

FULL = ["severity", "acetylene", "temporal", "anomaly"]
DEFAULT_WEIGHTS = {"severity": 1.0, "acetylene": 2.0, "temporal": 1.0, "anomaly": 1.0}
FEATURE_GASES = ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2"]


@st.cache_resource(show_spinner=False)
def cfg():
    return load_config()


@st.cache_data(show_spinner="Loading and cleaning DGA records…")
def get_clean() -> pd.DataFrame:
    return dga_data.load_clean()


@st.cache_data(show_spinner="Running conventional diagnostics (Duval / IEC / Rogers)…")
def get_clean_diag() -> pd.DataFrame:
    df = get_clean().copy()
    diag = conventional.diagnose(df)
    for c in ("duval", "iec", "rogers"):
        df[c] = diag[c]
    return df


@st.cache_resource(show_spinner="Training anomaly models + assembling per-unit features…")
def get_features() -> pd.DataFrame:
    """Per-unit temporal features + self-supervised anomaly scores (trains a small AE)."""
    return health.assemble_features(get_clean(), cfg())


@st.cache_data(show_spinner=False)
def get_risk(severity: float = 1.0, acetylene: float = 2.0,
             temporal: float = 1.0, anomaly: float = 1.0) -> pd.DataFrame:
    """Per-unit risk frame for the given weights (cheap; recomputed live)."""
    weights = {"severity": severity, "acetylene": acetylene,
               "temporal": temporal, "anomaly": anomaly}
    rm = health.risk_index(get_features(), weights, FULL)
    out = rm.feats.copy()
    out["risk_pct"] = out["risk_score"].rank(pct=True)
    return out


@st.cache_data(show_spinner=False)
def get_eval(severity: float = 1.0, acetylene: float = 2.0,
             temporal: float = 1.0, anomaly: float = 1.0) -> dict:
    rk = get_risk(severity, acetylene, temporal, anomaly)
    y = rk["fault_note"].astype(int).to_numpy()
    return health.evaluate_ranking(rk["risk_score"].to_numpy(), y)


@st.cache_data(show_spinner=False)
def get_unit_meta() -> pd.DataFrame:
    """One row per transformer: nameplate metadata (best-effort from the raw frame)."""
    df = get_clean()
    cols = [c for c in ["NAME", "MFG", "SER", "KV", "MVA", "YEAR_Energized", "LOC"]
            if c in df.columns]
    g = df.groupby("CODETX")
    meta = g[cols].first() if cols else pd.DataFrame(index=g.size().index)
    meta["n_samples"] = g.size()
    return meta


@st.cache_data(show_spinner=False)
def get_unit_history(codetx) -> pd.DataFrame:
    """All samples of one unit, chronologically (gases coerced to float)."""
    df = get_clean()
    sub = df[df["CODETX"] == codetx].copy()
    sub["date"] = pd.to_datetime(sub["Sample Day"], errors="coerce")
    for c in FEATURE_GASES + ["TCG"]:
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
    return sub.sort_values("date")


@st.cache_resource(show_spinner="Training SD-CAE (compositional fault-type representation)…")
def get_sdcae() -> pd.DataFrame:
    """Per-sample 2-D SD-CAE latent (lambda=0) + Duval class + KMeans cluster + gas mix."""
    df = get_clean_diag()
    c = cfg()
    comp = build_composition(df, c)
    set_seed(42)
    scfg = SimpleNamespace(**{**dict(c.sdcae), "latent_dim": 2, "lambda_adv": 0.0, "seed": 42})
    res = train_sdcae(comp.C, comp.m, scfg, verbose=False)
    Z = res.model.encode(comp.C)
    lab, _ = clustering.fit_kmeans(Z, 7, seed=42)

    out = pd.DataFrame({"z1": Z[:, 0], "z2": Z[:, 1]}, index=comp.index)
    out["duval"] = df.loc[comp.index, "duval"].fillna("ND").astype(str).to_numpy()
    out["cluster"] = pd.Series(lab, index=comp.index).astype(int).astype(str)
    out["magnitude"] = comp.m
    out["CODETX"] = df.loc[comp.index, "CODETX"].to_numpy()
    for j, gname in enumerate(comp.gases):
        out[f"pct_{gname}"] = (comp.P[:, j] * 100).round(1)
    return out


@st.cache_data(show_spinner=False)
def get_table(name: str):
    """Load a precomputed results table if present (else None)."""
    p = PROJECT_ROOT / "results" / "tables" / f"{name}.csv"
    return pd.read_csv(p, index_col=0) if p.exists() else None


def fleet_summary() -> dict:
    """Headline fleet KPIs."""
    df = get_clean()
    rk = get_risk(**DEFAULT_WEIGHTS)
    y = rk["fault_note"].astype(int)
    c2h2_ever = (rk["c2h2_max"] > 1).mean() if "c2h2_max" in rk else np.nan
    return {
        "n_units": int(df["CODETX"].nunique()),
        "n_samples": int(len(df)),
        "n_mfg": int(df["MFG"].nunique()) if "MFG" in df else 0,
        "base_rate": float(y.mean()),
        "pct_acetylene": float(c2h2_ever),
        "n_high": int((rk["risk_pct"] >= 0.9).sum()),
    }
