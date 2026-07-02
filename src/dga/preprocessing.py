"""Turn the cleaned DGA DataFrame into a model-ready feature matrix.

Pipeline (all options live in config/default.yaml):
    1. select diagnostic gases (atmospheric O2/N2 and the TCG sum excluded by default)
    2. clip absurd data-entry outliers per column (e.g. O2 = 7e8 ppm)
    3. impute missing readings as 0  (~ below detection limit)
    4. optionally drop all-zero rows (no diagnostic signal)
    5. log1p transform  (DGA ppm are heavily right-skewed)
    6. scale (standard / robust / minmax)

`build_feature_matrix` returns everything downstream code needs, including the
fitted scaler (so the exact same transform can be re-applied to new samples).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .config import load_config

_SCALERS = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}


@dataclass
class FeatureMatrix:
    X: np.ndarray            # scaled features, shape (n_samples, n_features), float32
    features: list[str]      # gas names in column order
    scaler: object           # fitted sklearn scaler (re-apply to new data)
    index: pd.Index          # sample_ids retained (aligns back to the source df)
    transformed: pd.DataFrame  # post-transform, pre-scaling frame (for inspection)


def select_features(cfg=None) -> list[str]:
    cfg = cfg or load_config()
    feats = list(cfg.data.feature_gases)
    if cfg.data.include_atmospheric:
        feats = ["O2", "N2"] + feats
    if cfg.data.drop_tcg and "TCG" in feats:
        feats.remove("TCG")
    # de-duplicate, preserve order
    seen, ordered = set(), []
    for f in feats:
        if f not in seen:
            ordered.append(f)
            seen.add(f)
    return ordered


def build_feature_matrix(df: pd.DataFrame, cfg=None, *,
                         fit_scaler: bool = True, scaler=None) -> FeatureMatrix:
    cfg = cfg or load_config()
    feats = select_features(cfg)
    sub = df[feats].copy()

    # 1. Clip per-column data-entry outliers (keeps log space sane).
    q = cfg.preprocessing.outlier_clip_quantile
    if q and q < 1.0:
        sub = sub.clip(upper=sub.quantile(q), axis=1)

    # 2. Missing gas readings -> 0 (below detection limit).
    sub = sub.fillna(0.0)

    # 3. Drop rows with no signal at all.
    if cfg.preprocessing.drop_all_zero_rows:
        sub = sub[sub.sum(axis=1) > 0]

    # 4. Variance-stabilising transform.
    transformed = np.log1p(sub) if cfg.preprocessing.transform == "log1p" else sub

    # 5. Scale.
    scaler_cls = _SCALERS[cfg.preprocessing.scaler]
    if fit_scaler or scaler is None:
        scaler = scaler_cls().fit(transformed.values)
    X = scaler.transform(transformed.values).astype("float32")

    return FeatureMatrix(X=X, features=feats, scaler=scaler,
                         index=sub.index, transformed=transformed)
