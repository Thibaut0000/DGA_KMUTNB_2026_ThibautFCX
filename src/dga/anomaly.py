"""Anomaly / incipient-fault detection.

Three interchangeable scorers, all returning a higher-is-more-anomalous score
plus a boolean flag at a chosen quantile threshold:
    * reconstruction error from the trained AE/VAE
    * Isolation Forest
    * One-Class SVM
Validate flagged anomalies against the `has_note` field events.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


def threshold_flags(scores: np.ndarray, quantile: float = 0.95) -> np.ndarray:
    return scores >= np.quantile(scores, quantile)


def reconstruction_scores(model, X: np.ndarray) -> np.ndarray:
    """Per-sample reconstruction error from an AE or VAE (must expose the method)."""
    return model.reconstruction_error(X)


def isolation_forest_scores(X: np.ndarray, seed: int = 42, contamination="auto"):
    iso = IsolationForest(random_state=seed, contamination=contamination).fit(X)
    # Negate so that higher = more anomalous (sklearn score is higher = normal).
    return -iso.score_samples(X), iso


def ocsvm_scores(X: np.ndarray, nu: float = 0.05, gamma="scale"):
    oc = OneClassSVM(nu=nu, gamma=gamma).fit(X)
    return -oc.score_samples(X), oc


def evaluate_against_notes(flags: np.ndarray, has_note: np.ndarray) -> dict:
    """How well do flagged anomalies line up with recorded field events?

    `has_note` is a weak positive signal (not exhaustive), so treat precision/
    recall as indicative rather than absolute.
    """
    flags = np.asarray(flags, bool)
    notes = np.asarray(has_note, bool)
    tp = int(np.sum(flags & notes))
    fp = int(np.sum(flags & ~notes))
    fn = int(np.sum(~flags & notes))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "n_flagged": int(flags.sum()), "n_notes": int(notes.sum())}
