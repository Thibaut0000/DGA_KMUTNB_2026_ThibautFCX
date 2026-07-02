"""Paper-quality plotting helpers (matplotlib). Save with save_fig()."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PROJECT_ROOT

FIG_DIR = PROJECT_ROOT / "results" / "figures"


def set_paper_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300,
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.autolayout": True,
    })


def save_fig(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight")
    return FIG_DIR / f"{name}.png"


def plot_gas_distributions(df: pd.DataFrame, gases, log=True):
    n = len(gases)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.5 * nrow))
    for ax, g in zip(axes.ravel(), gases):
        vals = df[g].dropna()
        vals = np.log1p(vals) if log else vals
        ax.hist(vals, bins=40, color="#3b7dd8")
        ax.set_title(("log1p " if log else "") + g)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    return fig


def plot_latent_2d(Z: np.ndarray, color, title="Latent space", method="pca"):
    """Project latent codes to 2D (PCA, or UMAP if installed) coloured by `color`."""
    Z = np.asarray(Z)
    if Z.shape[1] > 2:
        if method == "umap":
            try:
                import umap
                emb = umap.UMAP(random_state=42).fit_transform(Z)
            except ImportError:
                method = "pca"
        if method == "pca":
            from sklearn.decomposition import PCA
            emb = PCA(n_components=2, random_state=42).fit_transform(Z)
    else:
        emb = Z
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    codes = pd.Series(color).astype("category")
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=codes.cat.codes, cmap="tab10", s=8, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    handles = [plt.Line2D([], [], marker="o", ls="", color=plt.cm.tab10(i / 10), label=str(c))
               for i, c in enumerate(codes.cat.categories)]
    ax.legend(handles=handles, fontsize=7, markerscale=1.2, loc="best")
    return fig


# Duval Triangle 1 — fixed fault-zone colours (consistent across all figures).
DUVAL_COLORS = {
    "PD": "#fdd835",  # yellow
    "D1": "#90caf9",  # light blue
    "D2": "#1e88e5",  # blue
    "T1": "#a5d6a7",  # light green
    "T2": "#66bb6a",  # green
    "T3": "#ef5350",  # red
    "DT": "#bdbdbd",  # grey (mixed)
    "ND": "#eeeeee",  # not-determined / off-triangle
}

_DUVAL_ORDER = ["PD", "D1", "D2", "T1", "T2", "T3", "DT"]


def _duval_xy(a, e):
    """Ternary (%C2H2=a, %C2H4=e, %CH4=100-a-e) -> 2D; CH4 apex on top."""
    a, e = np.asarray(a, float), np.asarray(e, float)
    m = 100.0 - a - e
    x = e / 100.0 + (m / 100.0) * 0.5
    y = (m / 100.0) * (np.sqrt(3) / 2.0)
    return x, y


def _duval_class_grid(a, e):
    """Vectorised replica of `conventional.duval_triangle_1` (same rule order)."""
    m = 100.0 - a - e
    cls = np.full(a.shape, "DT", dtype="<U2")
    # assign in REVERSE priority so earlier rules win (first-match semantics)
    cls[(a <= 15) & (e > 50)] = "T3"
    cls[(a <= 4) & (e > 20) & (e <= 50)] = "T2"
    cls[(a <= 4) & (e <= 20)] = "T1"
    cls[(a > 29) & (e > 40)] = "D2"
    cls[(a >= 13) & (e <= 40)] = "D2"
    cls[(a >= 13) & (e <= 23)] = "D1"
    cls[m >= 98] = "PD"
    return cls


def plot_duval_triangle(df: pd.DataFrame, classes=None, color=None, title="Duval Triangle 1",
                        show_points=True, res=720):
    """Duval Triangle 1 with the 7 fault zones rasterised straight from the
    classifier (zones are therefore *exactly* consistent with the point labels),
    plus the samples on top.

    `classes` (or legacy `color`) is an optional per-row Duval label Series aligned
    to `df`; if omitted it is computed from CH4/C2H4/C2H2 via `dga.conventional`.
    """
    from matplotlib.colors import ListedColormap
    from . import conventional

    g = df[["CH4", "C2H4", "C2H2"]].fillna(0.0)
    tot = g.sum(axis=1).replace(0, np.nan)
    a = (100 * g["C2H2"] / tot)   # %C2H2
    e = (100 * g["C2H4"] / tot)   # %C2H4

    if classes is None:
        classes = color
    if classes is None:
        classes = g.apply(lambda r: conventional.duval_triangle_1(r["CH4"], r["C2H4"], r["C2H2"]),
                          axis=1)
    classes = pd.Series(classes, index=df.index).fillna("ND").astype(str)

    fig, ax = plt.subplots(figsize=(6.2, 5.6))

    # --- rasterise the zones: invert the xy->ternary map over a pixel grid ------
    xs = np.linspace(0.0, 1.0, res)
    ys = np.linspace(0.0, np.sqrt(3) / 2.0, res)
    X, Y = np.meshgrid(xs, ys)
    fm = Y / (np.sqrt(3) / 2.0)          # %CH4 fraction
    fe = X - fm * 0.5                     # %C2H4 fraction
    fa = 1.0 - fm - fe                    # %C2H2 fraction
    inside = (fm >= 0) & (fe >= 0) & (fa >= 0)
    cls = _duval_class_grid(np.where(inside, fa * 100, np.nan),
                            np.where(inside, fe * 100, np.nan))
    idx = {c: i for i, c in enumerate(_DUVAL_ORDER)}
    grid = np.full(X.shape, -1, dtype=int)
    for c in _DUVAL_ORDER:
        grid[inside & (cls == c)] = idx[c]
    grid_ma = np.ma.masked_where(grid < 0, grid)
    cmap = ListedColormap([DUVAL_COLORS[c] for c in _DUVAL_ORDER])
    ax.imshow(grid_ma, origin="lower", extent=[0, 1, 0, np.sqrt(3) / 2.0],
              cmap=cmap, vmin=0, vmax=len(_DUVAL_ORDER) - 1, alpha=0.80,
              interpolation="nearest", aspect="equal", zorder=0)

    # zone labels at the centroid of each zone's pixels
    for c in _DUVAL_ORDER:
        mask = inside & (cls == c)
        if mask.sum() > 50:
            ax.text(X[mask].mean(), Y[mask].mean(), c, ha="center", va="center",
                    fontsize=8, fontweight="bold", zorder=3)

    # outer triangle outline
    corners = np.array([_duval_xy(0, 0), _duval_xy(100, 0), _duval_xy(0, 100)])  # CH4, C2H2, C2H4
    ax.plot(*np.column_stack([corners.T, corners.T[:, :1]]), color="k", lw=1.2, zorder=2)

    # samples coloured by their own Duval class (must land on matching-colour zone)
    if show_points:
        px, py = _duval_xy(a.to_numpy(), e.to_numpy())
        pc = classes.map(DUVAL_COLORS).fillna("#333333")
        ax.scatter(px, py, c=pc, s=7, alpha=0.55, edgecolors="k", linewidths=0.2, zorder=4)

    # corner labels
    for (xx, yy), lab, va, ha in [(_duval_xy(0, 0), "%CH4", "bottom", "center"),
                                  (_duval_xy(100, 0), "%C2H2", "top", "right"),
                                  (_duval_xy(0, 100), "%C2H4", "top", "left")]:
        ax.text(xx, yy + (0.035 if va == "bottom" else -0.035), lab, ha=ha, va=va, fontsize=9)

    ax.set_xlim(-0.08, 1.08); ax.set_ylim(-0.08, 0.95)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title)
    return fig


def plot_reconstruction_error(scores: np.ndarray, threshold=None):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(scores, bins=60, color="#3b7dd8")
    if threshold is not None:
        ax.axvline(threshold, color="crimson", ls="--", label="threshold")
        ax.legend()
    ax.set_xlabel("reconstruction error")
    ax.set_ylabel("count")
    ax.set_title("Anomaly score distribution")
    return fig
