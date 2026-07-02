"""Clustering on the learned latent space + automatic model selection.

`select_k` sweeps a range of k and returns the silhouette-optimal KMeans/GMM
fit; `fit` is a thin dispatcher over kmeans / gmm / hdbscan.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


@dataclass
class ClusterResult:
    labels: np.ndarray
    model: object
    algorithm: str
    k: int
    silhouette: float
    extra: dict


def fit_kmeans(Z, k, seed=42):
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Z)
    return km.labels_, km


def fit_gmm(Z, k, seed=42):
    gm = GaussianMixture(n_components=k, random_state=seed, covariance_type="full").fit(Z)
    return gm.predict(Z), gm


def fit_hdbscan(Z, min_cluster_size=30):
    try:
        import hdbscan
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install hdbscan to use the HDBSCAN backend") from exc
    cl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    return cl.fit_predict(Z), cl


def select_k(Z, k_range, algorithm="kmeans", seed=42) -> ClusterResult:
    """Pick k maximising silhouette over k_range (kmeans or gmm)."""
    fitter = {"kmeans": fit_kmeans, "gmm": fit_gmm}[algorithm]
    scores = {}
    best = None
    for k in k_range:
        labels, model = fitter(Z, k, seed)
        if len(set(labels)) < 2:
            continue
        s = silhouette_score(Z, labels)
        scores[k] = s
        if best is None or s > best.silhouette:
            best = ClusterResult(labels, model, algorithm, k, s, {"silhouette_by_k": scores})
    best.extra["silhouette_by_k"] = scores
    return best


def fit(Z, cfg) -> ClusterResult:
    """Dispatch using config.clustering."""
    algo = cfg.algorithm
    if algo in ("kmeans", "gmm"):
        return select_k(Z, list(cfg.k_range), algo)
    if algo == "hdbscan":
        labels, model = fit_hdbscan(Z, cfg.hdbscan_min_cluster_size)
        mask = labels >= 0  # exclude noise from silhouette
        sil = silhouette_score(Z[mask], labels[mask]) if mask.sum() > len(set(labels)) else float("nan")
        return ClusterResult(labels, model, "hdbscan", len(set(labels[mask])), sil,
                             {"n_noise": int((labels < 0).sum())})
    raise ValueError(f"unknown clustering algorithm: {algo}")
