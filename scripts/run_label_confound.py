"""Contribution C2 - the field-note label is surveillance-confounded - + the
confound-free chemistry-target alternative. All numbers/figures are reproduced
here from repo code (not throwaway analysis), per the project convention.

    python scripts/run_label_confound.py

Three reproducible parts:
  1. Confound floor (contemporaneous): n_samples alone out-predicts the physics
     index and every conventional severity baseline vs fault_note.
  2. Temporal hold-out: a label-free score has little forward validity once
     n_samples is controlled (it drops to ~chance), while n_samples forward-predicts.
  3. Confound-free chemistry target (arcing onset): a target NOT tied to operator
     attention, weakly predictable from the gas trajectory; n_samples is at chance.
Outputs tables + figures to results/.
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression

from dga.config import PROJECT_ROOT, load_config
from dga import data as dga_data, conventional, temporal, health, viz

cfg = load_config()
viz.set_paper_style()
TDIR = PROJECT_ROOT / "results" / "tables"; TDIR.mkdir(parents=True, exist_ok=True)
GAS = ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO"]


def _z(x):
    x = pd.Series(np.asarray(x, float)).fillna(0.0)
    return ((x - x.mean()) / (x.std() + 1e-9)).to_numpy()


def part1_confound_floor(df, feats, y):
    """Contemporaneous AUC vs fault_note: index + conventional severity + the n_samples confound."""
    rm = health.risk_index(feats, {"severity": 1, "acetylene": 2, "temporal": 1, "anomaly": 1})
    diag = conventional.diagnose(df)
    arcing = df.join(diag).groupby("CODETX")["duval"].apply(lambda s: s.isin(["D1", "D2", "DT"]).any())
    latest = feats[[f"latest_{g}" for g in cfg.data.feature_gases]]
    p90 = latest.quantile(0.90)
    scores = {
        "TCG severity": np.log1p(feats["latest_TCG"]).to_numpy(),
        "Duval arcing": arcing.reindex(feats.index).fillna(False).astype(int).to_numpy(),
        "C57.104 cond. count": (latest > p90).sum(axis=1).to_numpy(),
        "Health index (ours)": rm.feats["risk_score"].to_numpy(),
        "n_samples (CONFOUND)": feats["n_samples"].to_numpy(),
    }
    aucs = {k: roc_auc_score(y, v) for k, v in scores.items()}
    res = pd.Series(aucs).sort_values()
    res.to_csv(TDIR / "label_confound_floor.csv")
    print("Part 1 - contemporaneous AUC vs fault_note:\n", res.round(3).to_string())

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    colors = ["#ef5350" if "CONFOUND" in k else ("#1e88e5" if "ours" in k else "#bdbdbd")
              for k in res.index]
    ax.barh(range(len(res)), res.values, color=colors)
    ax.set_yticks(range(len(res))); ax.set_yticklabels(res.index)
    ax.axvline(res["n_samples (CONFOUND)"], color="#ef5350", ls="--", lw=1)
    ax.set_xlim(0.45, 0.82); ax.set_xlabel("AUC vs fault_note")
    ax.set_title("The fault-note label is surveillance-confounded:\na trivial sample count out-predicts the physics")
    for i, v in enumerate(res.values):
        ax.text(v + 0.004, i, f"{v:.2f}", va="center", fontsize=8)
    viz.save_fig(fig, "label_confound_floor")
    return aucs


def part2_temporal_holdout(df):
    """First-half features -> predict a fault in the second half; control for n_samples."""
    df = df.copy(); df["d"] = pd.to_datetime(df["Sample Day"], errors="coerce")
    for c in GAS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0)
    df["TCG"] = df[GAS].sum(axis=1)
    df = df.dropna(subset=["d"]).sort_values(["CODETX", "d"])
    rows = []
    for cid, g in df.groupby("CODETX"):
        g = g.reset_index(drop=True)
        if len(g) < 4:
            continue
        mid = g["d"].iloc[0] + (g["d"].iloc[-1] - g["d"].iloc[0]) / 2
        first, second = g[g["d"] <= mid], g[g["d"] > mid]
        if len(first) < 2 or len(second) < 1:
            continue
        days = (first["d"] - first["d"].min()).dt.days.to_numpy()
        rH2 = temporal.generation_rate(days, np.log1p(first["H2"].to_numpy()))
        rows.append(dict(
            y=int(second["fault_note"].any()), n_first=len(first),
            sev=np.log1p(first["TCG"].iloc[-1]), h2=np.log1p(first["H2"].iloc[-1]),
            acet=np.log1p(first["C2H2"].max()), rate=0.0 if np.isnan(rH2) else max(rH2, 0.0)))
    D = pd.DataFrame(rows); y = D["y"].to_numpy()
    score = _z(D["h2"]) + 2 * _z(D["acet"]) + _z(D["rate"]) + _z(D["sev"])
    nf = D["n_first"].to_numpy()
    resid = score - LinearRegression().fit(nf.reshape(-1, 1), score).predict(nf.reshape(-1, 1))
    aucs = {"physics score": roc_auc_score(y, score),
            "n_samples (1st half)": roc_auc_score(y, nf),
            "physics | n_samples\ncontrolled": roc_auc_score(y, resid)}
    res = pd.Series(aucs)
    res.to_csv(TDIR / "label_confound_holdout.csv")
    print(f"\nPart 2 - temporal hold-out ({len(D)} units, {int(y.sum())} future-positive, "
          f"base {y.mean()*100:.1f}%):\n", res.round(3).to_string())

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    cols = ["#1e88e5", "#ef5350", "#90caf9"]
    ax.bar(range(3), [aucs["physics score"], aucs["n_samples (1st half)"], aucs["physics | n_samples\ncontrolled"]],
           color=cols)
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xticks(range(3)); ax.set_xticklabels(list(aucs.keys()), fontsize=8)
    ax.set_ylim(0.45, 0.72); ax.set_ylabel("forward AUC (2nd-half events)")
    ax.set_title("No forward validity beyond the confound:\nphysics drops to chance once n_samples is controlled")
    for i, v in enumerate(aucs.values()):
        ax.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    viz.save_fig(fig, "label_confound_holdout")
    return aucs


def part3_chemistry_target(df):
    """Confound-free target: onset of arcing (C2H2 appears) in the next 2 years."""
    df = df.copy(); df["d"] = pd.to_datetime(df["Sample Day"], errors="coerce")
    for c in GAS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0)
    df["TCG"] = df[GAS].sum(axis=1)
    df = df.dropna(subset=["d"]).sort_values(["CODETX", "d"]).reset_index(drop=True)
    H = 730; rows = []
    for cid, g in df.groupby("CODETX"):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            t = g["d"].iloc[i]; fut = g[(g["d"] > t) & (g["d"] <= t + pd.Timedelta(days=H))]
            if len(fut) == 0 or g["C2H2"].iloc[i] > 1:
                continue
            days = (g["d"].iloc[:i + 1] - g["d"].iloc[0]).dt.days.to_numpy()
            rH2 = temporal.generation_rate(days, np.log1p(g["H2"].iloc[:i + 1].to_numpy()))
            rows.append(dict(n=i + 1, y=int(fut["C2H2"].max() > 2),
                             C2H4=np.log1p(g["C2H4"].iloc[i]), H2=np.log1p(g["H2"].iloc[i]),
                             TCG=np.log1p(g["TCG"].iloc[i]), rateH2=0.0 if np.isnan(rH2) else max(rH2, 0.0)))
    D = pd.DataFrame(rows); y = D["y"].to_numpy()
    cor = np.corrcoef(D["n"], y)[0, 1]
    preds = {"C2H4 level": D["C2H4"], "H2 level": D["H2"], "TCG level": D["TCG"],
             "H2 rate": D["rateH2"].clip(lower=0), "n_samples (confound)": D["n"]}
    aucs = {k: roc_auc_score(y, v) for k, v in preds.items()}
    res = pd.Series(aucs).sort_values(ascending=False)
    res.to_csv(TDIR / "chemistry_target_arcing_onset.csv")
    print(f"\nPart 3 - confound-free chemistry target (arcing onset): {len(D)} points, "
          f"{int(y.sum())} positive ({y.mean()*100:.1f}%), corr(target, n_samples)={cor:+.3f}")
    print(res.round(3).to_string())

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    colors = ["#ef5350" if "confound" in k else "#1e88e5" for k in res.index]
    ax.bar(range(len(res)), res.values, color=colors)
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xticks(range(len(res))); ax.set_xticklabels(res.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0.45, 0.7); ax.set_ylabel("AUC")
    ax.set_title("Confound-free target (arcing onset): gas chemistry predicts,\nthe sampling confound does not")
    for i, v in enumerate(res.values):
        ax.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    viz.save_fig(fig, "chemistry_target_arcing_onset")
    return aucs, cor


def main():
    df = dga_data.load_clean()
    feats = health.assemble_features(df, cfg, seed=42)
    y = feats["fault_note"].astype(int).to_numpy()
    part1_confound_floor(df, feats, y)
    part2_temporal_holdout(df)
    part3_chemistry_target(df)
    print("\nwrote tables (label_confound_floor/holdout, chemistry_target_arcing_onset) and figures "
          "(label_confound_floor, label_confound_holdout, chemistry_target_arcing_onset)")


if __name__ == "__main__":
    main()
