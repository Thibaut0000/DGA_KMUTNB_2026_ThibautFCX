"""Clustering evaluation: internal quality + agreement with rule-based labels.

Because the data is unlabelled, internal metrics (silhouette, Davies-Bouldin,
Calinski-Harabasz) measure cluster quality, while external metrics (ARI, NMI,
homogeneity/completeness) measure agreement with the conventional Duval/IEC
diagnoses — this is how the paper argues the discovered structure is
physically meaningful rather than arbitrary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             completeness_score, davies_bouldin_score,
                             homogeneity_score, normalized_mutual_info_score,
                             silhouette_score)


def internal_metrics(Z: np.ndarray, labels: np.ndarray) -> dict:
    mask = np.asarray(labels) >= 0  # drop HDBSCAN noise if present
    Z, labels = np.asarray(Z)[mask], np.asarray(labels)[mask]
    if len(set(labels)) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    return {
        "silhouette": float(silhouette_score(Z, labels)),
        "davies_bouldin": float(davies_bouldin_score(Z, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(Z, labels)),
    }


def external_metrics(labels: np.ndarray, reference: np.ndarray) -> dict:
    """Agreement between cluster labels and a categorical reference (e.g. Duval)."""
    labels = np.asarray(labels)
    reference = pd.Series(reference).astype("category").cat.codes.to_numpy()
    mask = (labels >= 0) & (reference >= 0)
    labels, reference = labels[mask], reference[mask]
    return {
        "ARI": float(adjusted_rand_score(reference, labels)),
        "NMI": float(normalized_mutual_info_score(reference, labels)),
        "homogeneity": float(homogeneity_score(reference, labels)),
        "completeness": float(completeness_score(reference, labels)),
        "n_compared": int(mask.sum()),
    }


def cluster_fault_profile(labels, diagnoses: pd.Series) -> pd.DataFrame:
    """Cross-tab clusters x conventional fault label (interpretability table)."""
    tab = pd.crosstab(pd.Series(labels, name="cluster"),
                      pd.Series(np.asarray(diagnoses), name="fault"))
    return tab


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """Stack metric dicts from several methods into one paper-ready table."""
    return pd.DataFrame(results).T
