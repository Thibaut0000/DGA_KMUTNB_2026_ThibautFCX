"""Step 3 — cluster the latent space and evaluate against Duval labels.

    python scripts/run_clustering.py
Outputs:
    results/tables/cluster_metrics.csv
    results/tables/cluster_fault_profile.csv
    results/figures/latent_clusters.png
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from dga.config import PROJECT_ROOT, load_config
from dga import data as dga_data, conventional, clustering, evaluation, viz

cfg = load_config()
MDIR = PROJECT_ROOT / "results" / "models"
TDIR = PROJECT_ROOT / "results" / "tables"


def main():
    Z = np.load(MDIR / "latent.npy")
    index = np.load(MDIR / "feature_index.npy", allow_pickle=True)

    res = clustering.fit(Z, cfg.clustering)
    print(f"{res.algorithm}: k={res.k}  silhouette={res.silhouette:.3f}")

    internal = evaluation.internal_metrics(Z, res.labels)

    # External agreement with Duval labels on the same rows.
    df = dga_data.load_clean().loc[index]
    duval = conventional.diagnose(df)["duval"].to_numpy()
    external = evaluation.external_metrics(res.labels, duval)
    print("internal:", internal)
    print("external vs Duval:", external)

    TDIR.mkdir(parents=True, exist_ok=True)
    evaluation.comparison_table({"latent_clustering": {**internal, **external}}).to_csv(
        TDIR / "cluster_metrics.csv")
    evaluation.cluster_fault_profile(res.labels, duval).to_csv(
        TDIR / "cluster_fault_profile.csv")

    viz.set_paper_style()
    fig = viz.plot_latent_2d(Z, res.labels, title=f"{res.algorithm} (k={res.k})")
    viz.save_fig(fig, "latent_clusters")
    print("wrote figures + tables to results/")


if __name__ == "__main__":
    main()
