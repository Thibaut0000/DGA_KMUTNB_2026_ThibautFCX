"""Label-free Health/Risk Index for fleet ranking.

Combines, per transformer and using only *physical gas* features (never the
`n_samples`/`span_years` confounds):

  * severity   -- current gassing led by H2 (the universal early-fault gas) + CO2 + TCG
  * acetylene  -- C2H2 (arcing, the most dangerous fault) -> the acetylene-aware term
  * temporal   -- H2 generation rate (ppm/yr, C57.104 style)
  * anomaly    -- self-supervised novelty of the unit's gas state (autoencoder
                  reconstruction error and/or Isolation Forest)

The components are standardised and combined with transparent weights (default
boosts acetylene). The score ranks the fleet; validate it against `fault_note`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .config import load_config, set_seed
from . import temporal as _temporal
from . import anomaly as _anomaly

# Per-unit gas/trend columns fed to the anomaly detectors (physical only).
_ANOMALY_COLS = ["latest_H2", "latest_CH4", "latest_C2H2", "latest_C2H4",
                 "latest_C2H6", "latest_CO", "latest_CO2",
                 "rate_H2", "rate_TCG", "rate_C2H2"]

DEFAULT_WEIGHTS = {"severity": 1.0, "acetylene": 1.5, "temporal": 1.0, "anomaly": 1.0}


def _z(x):
    x = pd.Series(np.asarray(x, float)).fillna(0.0)
    return ((x - x.mean()) / (x.std() + 1e-9)).to_numpy()


def _zlog(x):
    return _z(np.log1p(np.clip(np.asarray(x, float), 0, None)))


@dataclass
class RiskModel:
    feats: pd.DataFrame          # per-unit features + anomaly scores + components + score
    weights: dict
    components: list[str]


def assemble_features(df: pd.DataFrame, cfg=None, *, seed: int = 42) -> pd.DataFrame:
    """Per-unit temporal features + self-supervised anomaly scores (AE + Isolation Forest)."""
    cfg = cfg or load_config()
    set_seed(seed)
    feats = _temporal.build_temporal(df, cfg)

    X = StandardScaler().fit_transform(
        np.log1p(feats[_ANOMALY_COLS].apply(pd.to_numeric, errors="coerce")
                 .fillna(0.0).clip(lower=0.0))).astype("float32")

    # Isolation Forest novelty (higher = more anomalous)
    feats["iforest"], _ = _anomaly.isolation_forest_scores(X, seed=seed)

    # Self-supervised autoencoder reconstruction error
    from .models.autoencoder import train_autoencoder
    res = train_autoencoder(X, cfg.autoencoder, verbose=False)
    feats["ae_recon"] = res.model.reconstruction_error(X)
    return feats


def compute_components(feats: pd.DataFrame) -> pd.DataFrame:
    """Standardised risk components (label-free)."""
    comp = pd.DataFrame(index=feats.index)
    comp["severity"] = np.mean([_zlog(feats["latest_H2"]), _zlog(feats["latest_TCG"]),
                                _zlog(feats["latest_CO2"])], axis=0)
    comp["acetylene"] = np.mean([_zlog(feats["c2h2_max"]), _zlog(feats["latest_C2H2"])], axis=0)
    comp["temporal"] = _zlog(feats["rate_H2"])
    comp["anomaly"] = np.mean([_z(feats["iforest"]), _z(feats["ae_recon"])], axis=0)
    return comp


def risk_index(feats: pd.DataFrame, weights: dict | None = None,
               components: list[str] | None = None) -> RiskModel:
    """Weighted label-free Health/Risk Index; higher = riskier."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    comp = compute_components(feats)
    components = components or list(comp.columns)
    score = sum(weights.get(c, 0.0) * comp[c] for c in components)
    out = feats.copy()
    for c in comp.columns:
        out[f"comp_{c}"] = comp[c]
    out["risk_score"] = score.to_numpy()
    out["risk_rank"] = out["risk_score"].rank(ascending=False, method="first").astype(int)
    return RiskModel(feats=out, weights=weights, components=components)


def evaluate_ranking(score, y, ks=(10, 30, 63)) -> dict:
    """AUC + precision@k + decile note-rates vs a weak label `y` (e.g. fault_note)."""
    s = pd.Series(np.asarray(score, float))
    y = np.asarray(y).astype(int)
    base = y.mean()
    order = s.sort_values(ascending=False).index.to_numpy()
    out = {"AUC": float(roc_auc_score(y, s)), "base_rate": float(base)}
    for k in ks:
        out[f"prec@{k}"] = float(y[order[:k]].mean())
    dec = pd.qcut(s.rank(ascending=False, method="first"), 10, labels=False)
    out["decile_note_rate"] = [float(y[(dec == d).to_numpy()].mean()) for d in range(10)]
    return out
