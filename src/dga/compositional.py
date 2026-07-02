"""Compositional decomposition of a DGA sample for SD-CAE.

Splits each sample into:
  * magnitude  m = log1p(total diagnostic gas)          -> severity / anomaly axis
  * composition p = gas proportions on the simplex       -> fault-type signature
and maps the composition to Euclidean space with the centred log-ratio (CLR)
transform of compositional data analysis (Aitchison geometry).

CLR is undefined at zero and DGA has many true zeros (e.g. C2H2 is 0 in ~97% of
samples), so zeros are handled by a simple multiplicative replacement before the
log-ratio. All knobs live under `compositional:` in config/default.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import load_config


@dataclass
class Composition:
    C: np.ndarray            # CLR coordinates, shape (n, D), each row sums to ~0  (float32)
    P: np.ndarray            # proportions on the simplex (post zero-replacement)  (float32)
    m: np.ndarray            # magnitude = log1p(total gas), shape (n,)            (float32)
    gases: list[str]         # composition gas names (column order)
    index: pd.Index          # sample_ids retained (rows with composition signal)


def multiplicative_replacement(P: np.ndarray, delta: float) -> np.ndarray:
    """Replace zeros in each compositional row by `delta` and rescale the rest so
    the row still sums to 1 (Martin-Fernandez multiplicative replacement)."""
    P = np.asarray(P, dtype=np.float64)
    out = P.copy()
    for i in range(P.shape[0]):
        zero = P[i] == 0
        nz = int(zero.sum())
        if nz == 0:
            continue
        # keep delta feasible if a row is almost all zeros
        d = min(delta, 0.9 / max(nz, 1))
        out[i, zero] = d
        out[i, ~zero] = P[i, ~zero] * (1.0 - nz * d)
    # guard against any residual non-positive value
    out = np.clip(out, 1e-12, None)
    return out / out.sum(axis=1, keepdims=True)


def clr(P: np.ndarray) -> np.ndarray:
    """Centred log-ratio: clr(p)_i = log p_i - mean_j log p_j. Rows must be > 0."""
    L = np.log(np.asarray(P, dtype=np.float64))
    return L - L.mean(axis=1, keepdims=True)


def build_composition(df: pd.DataFrame, cfg=None) -> Composition:
    """Build the CLR composition + magnitude for the SD-CAE input."""
    cfg = cfg or load_config()
    gases = list(cfg.compositional.gases)
    delta = float(cfg.compositional.zero_delta)

    sub = df[gases].fillna(0.0).clip(lower=0.0)
    total = sub.sum(axis=1)
    keep = total > 0                      # rows with at least one hydrocarbon -> a composition
    sub, total = sub[keep], total[keep]

    P = (sub.to_numpy() / total.to_numpy()[:, None])
    P = multiplicative_replacement(P, delta)
    C = clr(P)

    # Magnitude = (log) total of the SAME gases the composition is built from. This is exactly
    # what CLR factors out, so it is the right target for the disentanglement probe/adversary.
    mag = np.log1p(total.to_numpy())

    return Composition(
        C=C.astype("float32"),
        P=P.astype("float32"),
        m=mag.astype("float32"),
        gases=gases,
        index=sub.index,
    )


if __name__ == "__main__":  # python -m dga.compositional
    from dga import data as dga_data
    comp = build_composition(dga_data.load_clean())
    print("composition:", comp.C.shape, "gases:", comp.gases)
    print("CLR row-sum (should be ~0):", float(np.abs(comp.C.sum(axis=1)).max()))
    print("magnitude range:", round(float(comp.m.min()), 2), "-", round(float(comp.m.max()), 2))
