"""Multi-panel paper figures (page-budget consolidation).

    python scripts/make_paper_panels.py

Builds two double-column panels from existing results tables (no retraining):
  * representation_panel: (a) fault-type agreement across representations,
    (b) the deployed CLR-PCA diagnostic map coloured by Duval class.
  * confound_panel: (a) the surveillance-confound floor, (b) the temporal
    hold-out, (c) the confound-free arcing-onset target.
Replaces five single-column figures in the paper with two figure* floats.
"""
import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dga.config import PROJECT_ROOT, load_config
from dga import data as dga_data, conventional, viz
from dga.compositional import build_composition

TDIR = PROJECT_ROOT / "results" / "tables"
cfg = load_config()
viz.set_paper_style()


def representation_panel():
    tab = pd.read_csv(TDIR / "representation_baselines.csv", index_col=0)
    show = ["raw-log 5D + KMeans", "proportions 5D + KMeans", "CLR 5D (std) + KMeans",
            "PCA-2 of CLR + KMeans", "[ref] AE-2D on CLR (paper)"]
    labels = ["raw log", "propor-\ntions", "CLR\n5-D", "CLR\nPCA-2", "CLR AE\n(SD-CAE)"]
    vals = tab.loc[show, "ARI_mean"].to_numpy()
    errs = tab.loc[show, "ARI_std"].to_numpy()
    colors = ["#bdbdbd", "#bdbdbd", "#90caf9", "#1e88e5", "#ef9a9a"]

    fig = plt.figure(figsize=(11.0, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.25)

    ax = fig.add_subplot(gs[0])
    ax.bar(range(len(show)), vals, yerr=errs, capsize=3, color=colors)
    ax.set_xticks(range(len(show)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ARI vs Duval (k=7)")
    ax.set_title("(a) fault-type agreement: the gain is the log-ratio geometry", fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)

    df = dga_data.load_clean()
    df = df.join(conventional.diagnose(df))
    comp = build_composition(df, cfg)
    Z = PCA(2, random_state=0).fit_transform(
        StandardScaler().fit_transform(comp.C.astype("float64")))
    cls = df.loc[comp.index, "duval"].fillna("ND").astype(str)
    ax = fig.add_subplot(gs[1])
    for c in ["PD", "T1", "T2", "T3", "D1", "D2", "DT", "ND"]:
        msk = (cls == c).to_numpy()
        if msk.any():
            ax.scatter(Z[msk, 0], Z[msk, 1], s=5, alpha=0.55,
                       color=viz.DUVAL_COLORS.get(c, "#333333"), label=c)
    ax.set_title("(b) deployed diagnostic map: PCA-2 of standardised CLR", fontsize=9)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(fontsize=7, markerscale=1.6, ncol=2)
    viz.save_fig(fig, "representation_panel")


def confound_panel():
    floor = pd.read_csv(TDIR / "label_confound_floor.csv", index_col=0).iloc[:, 0]
    hold = pd.read_csv(TDIR / "label_confound_holdout.csv", index_col=0).iloc[:, 0]
    chem = pd.read_csv(TDIR / "chemistry_target_arcing_onset.csv", index_col=0).iloc[:, 0]

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.0))

    ax = axes[0]
    s = floor.sort_values()
    colors = ["#ef5350" if "CONFOUND" in k else ("#1e88e5" if "ours" in k else "#bdbdbd")
              for k in s.index]
    ax.barh(range(len(s)), s.values, color=colors)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([k.replace(" (CONFOUND)", "\n(confound)") for k in s.index], fontsize=7)
    ax.axvline(s.max(), color="#ef5350", ls="--", lw=1)
    ax.set_xlim(0.45, 0.82)
    ax.set_xlabel("AUC vs fault-note label")
    ax.set_title("(a) the sample count matches the physics", fontsize=9)
    for i, v in enumerate(s.values):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=7)

    ax = axes[1]
    order = list(hold.index)
    cols = ["#1e88e5", "#ef5350", "#90caf9"]
    ax.bar(range(len(order)), hold[order].values, color=cols)
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([k.replace(" | ", "\n| ") for k in order], fontsize=7)
    ax.set_ylim(0.45, 0.72)
    ax.set_ylabel("forward AUC (2nd-half events)")
    ax.set_title("(b) forward validity vanishes when controlled", fontsize=9)
    for i, v in enumerate(hold[order].values):
        ax.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=7)

    ax = axes[2]
    s = chem.sort_values(ascending=False)
    colors = ["#ef5350" if "confound" in k else "#1e88e5" for k in s.index]
    ax.bar(range(len(s)), s.values, color=colors)
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(s.index, rotation=20, ha="right", fontsize=7)
    ax.set_ylim(0.45, 0.70)
    ax.set_ylabel("AUC")
    ax.set_title("(c) confound-free target: chemistry predicts", fontsize=9)
    for i, v in enumerate(s.values):
        ax.text(i, v + 0.004, f"{v:.2f}", ha="center", fontsize=7)

    fig.tight_layout(w_pad=2.0)
    viz.save_fig(fig, "confound_panel")


if __name__ == "__main__":
    representation_panel()
    confound_panel()
    print("wrote figures representation_panel, confound_panel")
