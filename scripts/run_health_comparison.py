"""Step 4 - evaluate the Health/Risk Index: anomaly methods + conventional severity.

    python scripts/run_health_comparison.py

(1) Anomaly methods: AE reconstruction vs Isolation Forest vs Deep SVDD (the brief's list)
    -- univariate AUC vs fault_note + marginal contribution to the index.
(2) Conventional severity baselines vs the label-free index -- shows the index beats
    rule-based severity, and lists high-risk units the conventional rules call normal.
Outputs tables + figures to results/.
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from dga.config import PROJECT_ROOT, load_config
from dga import data as dga_data, conventional, temporal, health, viz

cfg = load_config()
viz.set_paper_style()
W = {"severity": 1.0, "acetylene": 2.0, "temporal": 1.0, "anomaly": 1.0}
FULL = ["severity", "acetylene", "temporal", "anomaly"]
GASES = list(cfg.data.feature_gases)


def _auc(y, s):
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def main():
    df = dga_data.load_clean()
    feats = health.assemble_features(df, cfg, seed=42)
    y = feats["fault_note"].astype(int).to_numpy()
    rm = health.risk_index(feats, W, FULL)
    idx_auc = roc_auc_score(y, rm.feats["risk_score"])
    print(f"Health index AUC = {idx_auc:.3f}  (fault base {y.mean()*100:.1f}%)")

    # ---- (1) anomaly methods ------------------------------------------------
    from sklearn.preprocessing import StandardScaler
    from dga import anomaly
    from dga.models.autoencoder import train_autoencoder
    from dga.models.deep_svdd import deep_svdd_scores
    X = StandardScaler().fit_transform(np.log1p(
        feats[health._ANOMALY_COLS].apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0))).astype("float32")
    aerec = train_autoencoder(X, cfg.autoencoder, verbose=False).model.reconstruction_error(X)
    iforest, _ = anomaly.isolation_forest_scores(X, seed=42)
    svdd = deep_svdd_scores(X, seed=42)
    anom = pd.DataFrame({"AE reconstruction": [_auc(y, aerec)],
                         "Isolation Forest": [_auc(y, iforest)],
                         "Deep SVDD": [_auc(y, svdd)]}, index=["AUC vs fault_note"]).T
    print("\n(1) Anomaly detectors:\n", anom.round(3).to_string())
    anom.to_csv(PROJECT_ROOT / "results" / "tables" / "anomaly_methods.csv")

    # ---- (2) conventional severity baselines --------------------------------
    diag = conventional.diagnose(df)
    df2 = df.join(diag)
    arcing = df2.groupby("CODETX")["duval"].apply(lambda s: s.isin(["D1", "D2", "DT"]).any())
    # fleet 90th-percentile "condition count": how many latest gases exceed the fleet p90
    latest = feats[[f"latest_{g}" for g in GASES]].copy()
    p90 = latest.quantile(0.90)
    cond_count = (latest > p90).sum(axis=1)
    conv = pd.DataFrame(index=feats.index)
    conv["TCG (severity)"] = np.log1p(feats["latest_TCG"])
    conv["max gas p90 ratio"] = (latest / p90.clip(lower=1.0)).max(axis=1)  # floor avoids /0
    conv["C57.104-style cond. count"] = cond_count
    conv["Duval arcing (D1/D2/DT)"] = arcing.reindex(feats.index).fillna(False).astype(int)
    comp = {c: _auc(y, conv[c]) for c in conv.columns}
    comp["Health index (ours)"] = idx_auc
    comp_s = pd.Series(comp).sort_values()
    print("\n(2) Conventional severity vs health index (AUC vs fault_note):")
    print(comp_s.round(3).to_string())
    comp_s.to_csv(PROJECT_ROOT / "results" / "tables" / "conventional_comparison.csv")

    # ---- figures ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    colors = ["#bdbdbd"] * (len(comp_s) - 1) + ["#1e88e5"]
    ax.barh(range(len(comp_s)), comp_s.values, color=[c if n != "Health index (ours)" else "#1e88e5"
            for n, c in zip(comp_s.index, colors)])
    ax.set_yticks(range(len(comp_s))); ax.set_yticklabels(comp_s.index)
    ax.axvline(0.5, color="k", lw=0.8, ls=":"); ax.set_xlim(0.45, 0.8)
    ax.set_xlabel("AUC vs fault_note"); ax.set_title("Risk ranking: label-free index vs conventional severity")
    for i, v in enumerate(comp_s.values):
        ax.text(v + 0.004, i, f"{v:.2f}", va="center", fontsize=8)
    viz.save_fig(fig, "health_vs_conventional")

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar(range(len(anom)), anom["AUC vs fault_note"], color="#1e88e5")
    ax.set_xticks(range(len(anom))); ax.set_xticklabels(anom.index, rotation=15, ha="right")
    ax.axhline(0.5, color="k", lw=0.8, ls=":"); ax.set_ylim(0.45, 0.65)
    ax.set_ylabel("AUC vs fault_note"); ax.set_title("Anomaly detectors compared")
    viz.save_fig(fig, "anomaly_methods")

    # ---- units the conventional rules miss ----------------------------------
    out = rm.feats.copy()
    out["duval_arcing"] = arcing.reindex(out.index).fillna(False)
    out["cond_count"] = cond_count
    missed = out[(out["risk_rank"] <= 40) & (~out["duval_arcing"]) & (out["cond_count"] == 0)]
    cols = ["risk_rank", "risk_score", "latest_H2", "c2h2_max", "latest_TCG", "rate_H2", "fault_note"]
    print(f"\nHigh-risk units (top 40) that conventional rules call normal "
          f"(no arcing, no gas > p90): {len(missed)}")
    print(missed.sort_values("risk_rank")[cols].head(10).round(1).to_string())
    print("\nwrote anomaly_methods.csv, conventional_comparison.csv; figures health_vs_conventional, anomaly_methods")


if __name__ == "__main__":
    main()
