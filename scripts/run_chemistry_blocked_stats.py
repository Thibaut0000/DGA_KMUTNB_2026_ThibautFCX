"""Unit-blocked statistics for the arcing-onset chemistry target.

    python scripts/run_chemistry_blocked_stats.py

The 3,794 prediction points cluster within ~600 units (several points per unit,
autocorrelated; one onset can yield multiple positive points), so a naive
per-point permutation test is anti-conservative. Here:
  * cluster bootstrap BY UNIT (resample units with replacement, keep all their
    points) -> 95% CI for the AUC of C2H4 / H2 / n_samples;
  * unit-blocked permutation: shuffle outcome trajectories BETWEEN units
    (a unit's y-vector moves as one block onto another unit's points, truncated
    /padded to length) is ill-defined, so we use the bootstrap CI vs 0.5 as the
    significance criterion, plus a unit-level AUC (one point per unit) as the
    fully-independent check.
Replaces the naive per-point permutation p reported previously. Paper spec:
onset > 2 ppm within 2 years, eligibility C2H2 <= 1 ppm.
Outputs results/tables/chemistry_blocked_stats.csv.
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dga.config import PROJECT_ROOT
from dga import data as dga_data

GAS = ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO"]
H, THR = 730, 2


def build_points(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["d"] = pd.to_datetime(df["Sample Day"], errors="coerce")
    for c in GAS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0)
    df = df.dropna(subset=["d"]).sort_values(["CODETX", "d"]).reset_index(drop=True)
    rows = []
    for cid, g in df.groupby("CODETX"):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            t = g["d"].iloc[i]
            fut = g[(g["d"] > t) & (g["d"] <= t + pd.Timedelta(days=H))]
            if len(fut) == 0 or g["C2H2"].iloc[i] > 1:
                continue
            rows.append(dict(unit=cid, y=int(fut["C2H2"].max() > THR),
                             C2H4=np.log1p(g["C2H4"].iloc[i]),
                             H2=np.log1p(g["H2"].iloc[i]), n=i + 1))
    return pd.DataFrame(rows)


def cluster_bootstrap_ci(D: pd.DataFrame, col: str, B: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    units = D["unit"].unique()
    groups = {u: g for u, g in D.groupby("unit")}
    aucs = []
    for _ in range(B):
        pick = rng.choice(units, len(units), replace=True)
        S = pd.concat([groups[u] for u in pick], ignore_index=True)
        if S["y"].nunique() < 2:
            continue
        aucs.append(roc_auc_score(S["y"], S[col]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    D = build_points(dga_data.load_clean())
    y = D["y"].to_numpy()
    n_onset_units = D.loc[D["y"] == 1, "unit"].nunique()
    print(f"points={len(D)}  positive points={int(y.sum())}  "
          f"onset UNITS={n_onset_units}  units total={D['unit'].nunique()}")

    rows = {}
    for col in ("C2H4", "H2", "n"):
        auc = float(roc_auc_score(y, D[col]))
        lo, hi = cluster_bootstrap_ci(D, col)
        sig = "yes" if lo > 0.5 else "no"
        rows[col] = {"AUC_point": round(auc, 3), "CI_lo": round(lo, 3),
                     "CI_hi": round(hi, 3), "excl_0.5": sig}
        print(f"{col:5s} point-AUC {auc:.3f}   unit-blocked 95% CI [{lo:.3f}, {hi:.3f}]"
              f"   excludes 0.5: {sig}")

    # fully-independent check: one point per unit (the LAST eligible point,
    # the operationally relevant 'current state' question)
    last = D.groupby("unit").tail(1)
    yl = last["y"].to_numpy()
    print(f"\nunit-level (last eligible point per unit): {len(last)} units, "
          f"{int(yl.sum())} positives")
    for col in ("C2H4", "H2", "n"):
        if yl.sum() >= 5:
            auc = float(roc_auc_score(yl, last[col]))
            rows[col][f"AUC_unit_level"] = round(auc, 3)
            print(f"  {col:5s} unit-level AUC {auc:.3f}")

    tab = pd.DataFrame(rows).T
    tdir = PROJECT_ROOT / "results" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(tdir / "chemistry_blocked_stats.csv")
    print("\nwrote", tdir / "chemistry_blocked_stats.csv")


if __name__ == "__main__":
    main()
