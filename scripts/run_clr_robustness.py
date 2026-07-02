"""Robustness checks for the compositional (CLR) fault-type result.

    python scripts/run_clr_robustness.py

Three attacks on the PCA-2-of-CLR + KMeans headline (ARI ~0.545 vs Duval):
  1. Zero-replacement sensitivity: C2H2 is zero in 97.4% of samples, so the CLR
     depends on the delta constant. Vary delta over {1e-4, 1e-3, 1e-2}.
  2. Temporal generalisation: fit scaler+PCA+KMeans on pre-2022 samples only,
     assign 2022+ samples to the frozen centroids, ARI on the held-out period.
  3. Unit-level evaluation: heavily-sampled units dominate per-sample ARI;
     re-evaluate on one (latest) sample per unit.
Outputs results/tables/clr_robustness.csv.
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dga.config import PROJECT_ROOT, load_config
from dga import data as dga_data, conventional, clustering, evaluation
from dga.compositional import build_composition

SEEDS = [0, 1, 2, 3, 4]


def pca_km_ari(C: np.ndarray, duval: pd.Series) -> tuple[float, float]:
    Z = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(C))
    runs = [evaluation.external_metrics(clustering.fit_kmeans(Z, 7, seed=s)[0], duval)["ARI"]
            for s in SEEDS]
    return float(np.mean(runs)), float(np.std(runs))


def main():
    df = dga_data.load_clean()
    df = df.join(conventional.diagnose(df))
    rows = {}

    # ---- 1. zero-replacement delta sensitivity ------------------------------
    for delta in (1e-4, 1e-3, 1e-2):
        cfg = load_config()
        cfg["compositional"]["zero_delta"] = delta
        comp = build_composition(df, cfg)
        duval = df.loc[comp.index, "duval"]
        m, s = pca_km_ari(comp.C, duval)
        rows[f"delta={delta:g}"] = {"ARI_mean": round(m, 3), "ARI_std": round(s, 3),
                                    "n": len(comp.index)}
        print(f"delta={delta:g}  ARI {m:.3f} +/- {s:.3f}")

    # ---- 2. temporal split (fit pre-2022, evaluate 2022+) -------------------
    cfg = load_config()
    comp = build_composition(df, cfg)
    duval_all = df.loc[comp.index, "duval"]
    dates = pd.to_datetime(df.loc[comp.index, "Sample Day"], errors="coerce")
    train = (dates < "2022-01-01").to_numpy()
    test = ~train
    scaler = StandardScaler().fit(comp.C[train])
    pca = PCA(2, random_state=0).fit(scaler.transform(comp.C[train]))
    Ztr = pca.transform(scaler.transform(comp.C[train]))
    Zte = pca.transform(scaler.transform(comp.C[test]))
    aris = []
    for s in SEEDS:
        lab_tr, model = clustering.fit_kmeans(Ztr, 7, seed=s)
        lab_te = model.predict(Zte)
        aris.append(evaluation.external_metrics(lab_te, duval_all[test])["ARI"])
    rows["temporal: fit<2022, eval>=2022"] = {
        "ARI_mean": round(float(np.mean(aris)), 3),
        "ARI_std": round(float(np.std(aris)), 3),
        "n": int(test.sum())}
    print(f"temporal hold-out  ARI {np.mean(aris):.3f} +/- {np.std(aris):.3f} "
          f"({train.sum()} train / {test.sum()} test samples)")

    # ---- 3. one (latest) sample per unit ------------------------------------
    sub = df.loc[comp.index].copy()
    sub["_d"] = dates
    latest_ids = (sub.sort_values("_d").groupby("CODETX").tail(1)).index
    mask = comp.index.isin(latest_ids)
    m, s = pca_km_ari(comp.C[mask], duval_all[mask])
    rows["one latest sample per unit"] = {"ARI_mean": round(m, 3), "ARI_std": round(s, 3),
                                          "n": int(mask.sum())}
    print(f"latest-per-unit    ARI {m:.3f} +/- {s:.3f} ({int(mask.sum())} units)")

    tab = pd.DataFrame(rows).T
    tdir = PROJECT_ROOT / "results" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(tdir / "clr_robustness.csv")
    print("\nwrote", tdir / "clr_robustness.csv")


if __name__ == "__main__":
    main()
