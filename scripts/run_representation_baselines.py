"""Adversarial ablation: does SD-CAE's neural network actually add anything over
plain clustering / linear reduction on the same CLR features?

    python scripts/run_representation_baselines.py

If KMeans directly on the 5-D CLR (or on a 2-D PCA of it) matches the SD-CAE
latent's ARI ~0.47 vs Duval, the contribution is the *compositional geometry*,
not the autoencoder - and the paper must say so. All variants use the same rows,
the same 5 combustible gases, and the same evaluation as run_sdcae_ablation.py
(KMeans k=7, ARI/NMI vs Duval, 5 seeds). Outputs results/tables/representation_baselines.csv.
"""
import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from dga.config import PROJECT_ROOT, load_config, set_seed
from dga import data as dga_data, conventional, clustering, evaluation, viz
from dga.compositional import build_composition

cfg = load_config()
SEEDS = [0, 1, 2, 3, 4]


def kmeans_scores(X: np.ndarray, duval: pd.Series) -> dict:
    """ARI/NMI vs Duval for KMeans k=7 over 5 seeds (mean +/- std)."""
    runs = []
    for s in SEEDS:
        lab, _ = clustering.fit_kmeans(X, 7, seed=s)
        runs.append(evaluation.external_metrics(lab, duval))
    return {
        "ARI_mean": float(np.mean([r["ARI"] for r in runs])),
        "ARI_std": float(np.std([r["ARI"] for r in runs])),
        "NMI_mean": float(np.mean([r["NMI"] for r in runs])),
    }


def gmm_scores(X: np.ndarray, duval: pd.Series) -> dict:
    runs = []
    for s in SEEDS:
        set_seed(s)
        lab = GaussianMixture(n_components=7, covariance_type="full",
                              random_state=s, n_init=3).fit_predict(X)
        runs.append(evaluation.external_metrics(lab, duval))
    return {
        "ARI_mean": float(np.mean([r["ARI"] for r in runs])),
        "ARI_std": float(np.std([r["ARI"] for r in runs])),
        "NMI_mean": float(np.mean([r["NMI"] for r in runs])),
    }


def main():
    df = dga_data.load_clean()
    df = df.join(conventional.diagnose(df))
    comp = build_composition(df, cfg)
    duval = df.loc[comp.index, "duval"]
    sub = df.loc[comp.index, comp.gases].fillna(0.0).clip(lower=0.0)
    raw_log = np.log1p(sub.to_numpy())
    std = lambda A: StandardScaler().fit_transform(A).astype("float64")

    clr_std = std(comp.C)
    variants: dict[str, tuple[np.ndarray, str]] = {
        "raw-log 5D + KMeans":        (std(raw_log), "km"),
        "proportions 5D + KMeans":    (std(comp.P), "km"),
        "CLR 5D (raw) + KMeans":      (comp.C.astype("float64"), "km"),
        "CLR 5D (std) + KMeans":      (clr_std, "km"),
        "PCA-2 of CLR + KMeans":      (PCA(2, random_state=0).fit_transform(clr_std), "km"),
        "PCA-3 of CLR + KMeans":      (PCA(3, random_state=0).fit_transform(clr_std), "km"),
        # PCA on unstandardised CLR = classical Aitchison log-ratio PCA (Aitchison 1983).
        "Aitchison PCA-2 + KMeans":   (PCA(2, random_state=0).fit_transform(comp.C.astype("float64")), "km"),
        "CLR 5D (std) + GMM":         (clr_std, "gmm"),
    }

    rows = {}
    for name, (X, algo) in variants.items():
        rows[name] = kmeans_scores(X, duval) if algo == "km" else gmm_scores(X, duval)
        r = rows[name]
        print(f"{name:28s} ARI {r['ARI_mean']:.3f} +/- {r['ARI_std']:.3f}   NMI {r['NMI_mean']:.3f}")

    # Reference: the SD-CAE (AE on CLR, 2-D latent) numbers from the existing ablation.
    ref = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "sdcae_ablation.csv", index_col=0)
    ae = ref.loc["CLR (lambda=0) [dim=2]"]
    rows["[ref] AE-2D on CLR (paper)"] = {"ARI_mean": float(ae["ARI@k7_mean"]),
                                          "ARI_std": float(ae["ARI@k7_std"]),
                                          "NMI_mean": float(ae["NMI@k7_mean"])}
    print(f"{'[ref] AE-2D on CLR (paper)':28s} ARI {ae['ARI@k7_mean']:.3f} +/- {ae['ARI@k7_std']:.3f}")

    out = pd.DataFrame(rows).T.round(3)
    tdir = PROJECT_ROOT / "results" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(tdir / "representation_baselines.csv")
    print("\nwrote", tdir / "representation_baselines.csv")

    # ---- paper figures ------------------------------------------------------
    viz.set_paper_style()

    # 1. representation ablation bars (the paper's Fig. "fault-type agreement")
    show = ["raw-log 5D + KMeans", "proportions 5D + KMeans", "CLR 5D (std) + KMeans",
            "PCA-2 of CLR + KMeans", "[ref] AE-2D on CLR (paper)"]
    labels = ["raw log\n+ KMeans", "proportions\n+ KMeans", "CLR\n+ KMeans",
              "CLR + PCA-2\n+ KMeans (deployed)", "CLR + AE-2D\n(SD-CAE, ablation)"]
    vals = [rows[k]["ARI_mean"] for k in show]
    errs = [rows[k]["ARI_std"] for k in show]
    colors = ["#bdbdbd", "#bdbdbd", "#90caf9", "#1e88e5", "#ef9a9a"]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.bar(range(len(show)), vals, yerr=errs, capsize=3, color=colors)
    ax.set_xticks(range(len(show)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ARI vs Duval (k=7)")
    ax.set_title("Fault-type agreement: the gain is the log-ratio geometry,\n"
                 "and a linear projection beats the autoencoder")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)
    viz.save_fig(fig, "representation_ablation")

    # 2. deployed diagnostic map: PCA-2 of standardised CLR, coloured by Duval
    comp = build_composition(df, cfg)
    duval = df.loc[comp.index, "duval"]
    Z = PCA(2, random_state=0).fit_transform(
        StandardScaler().fit_transform(comp.C.astype("float64")))
    cls = pd.Series(duval.to_numpy(), index=comp.index).fillna("ND").astype(str)
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in ["PD", "T1", "T2", "T3", "D1", "D2", "DT", "ND"]:
        msk = (cls == c).to_numpy()
        if msk.any():
            ax.scatter(Z[msk, 0], Z[msk, 1], s=7, alpha=0.6,
                       color=viz.DUVAL_COLORS.get(c, "#333333"), label=c)
    ax.set_title("Learned diagnostic map: PCA-2 of the standardised CLR,\ncoloured by Duval class")
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(fontsize=7, markerscale=1.5)
    viz.save_fig(fig, "clr_pca_map")
    print("wrote figures representation_ablation, clr_pca_map")


if __name__ == "__main__":
    main()
