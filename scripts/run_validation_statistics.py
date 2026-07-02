"""Statistical backing for the validation claims (reviewer-proofing).

    python scripts/run_validation_statistics.py

Four analyses, all vs the weak `fault_note` label unless stated:
  1. Paired bootstrap: is AUC(index)=0.74 vs AUC(n_samples)=0.76 a real difference?
  2. Incremental likelihood-ratio tests: does the physics index add signal beyond
     n_samples (and vice versa)?
  3. Stratified AUC: within n_samples quartiles, does physics still rank events?
  4. Chemistry-target sensitivity: arcing-onset AUCs across onset-threshold and
     horizon choices (robustness of the 0.64 result).
Outputs results/tables/{validation_statistics,chemistry_target_sensitivity}.csv.
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from dga.config import PROJECT_ROOT, load_config
from dga import data as dga_data, health, temporal

cfg = load_config()
TDIR = PROJECT_ROOT / "results" / "tables"
TDIR.mkdir(parents=True, exist_ok=True)
W = {"severity": 1.0, "acetylene": 2.0, "temporal": 1.0, "anomaly": 1.0}
GAS = ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO"]


def _z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)


def _loglik(model, X, y):
    p = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def part1_paired_bootstrap(risk, n, y, B=5000, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    diffs, a_r, a_n = [], [], []
    for _ in range(B):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        ar, an = roc_auc_score(y[s], risk[s]), roc_auc_score(y[s], n[s])
        a_r.append(ar); a_n.append(an); diffs.append(ar - an)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    out = {
        "AUC_index": roc_auc_score(y, risk), "AUC_index_CI": (np.percentile(a_r, 2.5), np.percentile(a_r, 97.5)),
        "AUC_nsamples": roc_auc_score(y, n), "AUC_n_CI": (np.percentile(a_n, 2.5), np.percentile(a_n, 97.5)),
        "diff_CI": (lo, hi), "diff_p": p,
    }
    print("Part 1 - paired bootstrap (index vs n_samples):")
    print(f"  AUC index     {out['AUC_index']:.3f}  95% CI [{out['AUC_index_CI'][0]:.3f}, {out['AUC_index_CI'][1]:.3f}]")
    print(f"  AUC n_samples {out['AUC_nsamples']:.3f}  95% CI [{out['AUC_n_CI'][0]:.3f}, {out['AUC_n_CI'][1]:.3f}]")
    print(f"  diff (index - n) 95% CI [{lo:+.3f}, {hi:+.3f}]   two-sided p = {p:.3f}")
    return out


def part2_lr_tests(risk, n, y):
    zlogn = _z(np.log1p(n)).reshape(-1, 1)
    zrisk = _z(risk).reshape(-1, 1)
    both = np.hstack([zlogn, zrisk])
    lr = lambda X: LogisticRegression(penalty=None, max_iter=2000).fit(X, y)
    ll_n, ll_r, ll_b = (_loglik(lr(X), X, y) for X in (zlogn, zrisk, both))
    stat_add_r = 2 * (ll_b - ll_n)     # physics beyond n?
    stat_add_n = 2 * (ll_b - ll_r)     # n beyond physics?
    p_add_r = float(stats.chi2.sf(stat_add_r, 1))
    p_add_n = float(stats.chi2.sf(stat_add_n, 1))
    print("\nPart 2 - incremental likelihood-ratio tests (logistic, unit level):")
    print(f"  physics ADDS to n_samples:  LR={stat_add_r:6.2f}  p={p_add_r:.2e}")
    print(f"  n_samples ADDS to physics:  LR={stat_add_n:6.2f}  p={p_add_n:.2e}")
    return {"LR_physics_beyond_n": stat_add_r, "p_physics_beyond_n": p_add_r,
            "LR_n_beyond_physics": stat_add_n, "p_n_beyond_physics": p_add_n}


def part3_stratified(risk, n, y, k=4):
    q = pd.qcut(pd.Series(n).rank(method="first"), k, labels=False)
    rows = []
    for s in range(k):
        m = (q == s).to_numpy()
        ys, rs = y[m], risk[m]
        auc = roc_auc_score(ys, rs) if len(np.unique(ys)) == 2 else np.nan
        rows.append({"stratum": f"Q{s+1}", "n_units": int(m.sum()),
                     "n_range": f"{int(n[m].min())}-{int(n[m].max())}",
                     "events": int(ys.sum()), "AUC_physics_within": round(float(auc), 3)})
    tab = pd.DataFrame(rows).set_index("stratum")
    print("\nPart 3 - AUC of the physics index WITHIN n_samples strata:")
    print(tab.to_string())
    return tab


def part4_chemistry_sensitivity(df):
    df = df.copy()
    df["d"] = pd.to_datetime(df["Sample Day"], errors="coerce")
    for c in GAS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0)
    df = df.dropna(subset=["d"]).sort_values(["CODETX", "d"]).reset_index(drop=True)
    rows = []
    for thr in (2, 5, 10):
        for years in (1, 2, 3):
            H = int(365.25 * years)
            recs = []
            for _, g in df.groupby("CODETX"):
                g = g.reset_index(drop=True)
                for i in range(len(g) - 1):
                    t = g["d"].iloc[i]
                    fut = g[(g["d"] > t) & (g["d"] <= t + pd.Timedelta(days=H))]
                    if len(fut) == 0 or g["C2H2"].iloc[i] > 1:
                        continue
                    recs.append((i + 1, int(fut["C2H2"].max() > thr),
                                 np.log1p(g["C2H4"].iloc[i]), np.log1p(g["H2"].iloc[i])))
            D = pd.DataFrame(recs, columns=["n", "y", "C2H4", "H2"])
            y = D["y"].to_numpy()
            if y.sum() < 5:
                continue
            rows.append({
                "onset_thr_ppm": thr, "horizon_yr": years,
                "points": len(D), "positives": int(y.sum()),
                "AUC_C2H4": round(float(roc_auc_score(y, D["C2H4"])), 3),
                "AUC_H2": round(float(roc_auc_score(y, D["H2"])), 3),
                "AUC_n": round(float(roc_auc_score(y, D["n"])), 3),
            })
    tab = pd.DataFrame(rows)
    tab.to_csv(TDIR / "chemistry_target_sensitivity.csv", index=False)
    print("\nPart 4 - chemistry-target sensitivity (arcing onset):")
    print(tab.to_string(index=False))
    return tab


def main():
    df = dga_data.load_clean()
    feats = health.assemble_features(df, cfg, seed=42)
    y = feats["fault_note"].astype(int).to_numpy()
    risk = health.risk_index(feats, W).feats["risk_score"].to_numpy()
    n = feats["n_samples"].to_numpy(float)

    p1 = part1_paired_bootstrap(risk, n, y)
    p2 = part2_lr_tests(risk, n, y)
    p3 = part3_stratified(risk, n, y)
    part4_chemistry_sensitivity(df)

    flat = {
        "AUC_index": p1["AUC_index"], "AUC_index_lo": p1["AUC_index_CI"][0], "AUC_index_hi": p1["AUC_index_CI"][1],
        "AUC_n": p1["AUC_nsamples"], "AUC_n_lo": p1["AUC_n_CI"][0], "AUC_n_hi": p1["AUC_n_CI"][1],
        "diff_lo": p1["diff_CI"][0], "diff_hi": p1["diff_CI"][1], "diff_p": p1["diff_p"], **p2,
    }
    pd.Series(flat).round(4).to_csv(TDIR / "validation_statistics.csv")
    p3.to_csv(TDIR / "validation_stratified_auc.csv")
    print("\nwrote validation_statistics.csv, validation_stratified_auc.csv, chemistry_target_sensitivity.csv")


if __name__ == "__main__":
    main()
