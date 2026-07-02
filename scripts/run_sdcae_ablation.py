"""SD-CAE ablation: does a severity-invariant representation recover fault TYPE?

    python scripts/run_sdcae_ablation.py

Same five hydrocarbon gases throughout; only the representation changes:
    raw-log -> proportions -> CLR -> SD-CAE (lambda=1) -> SD-CAE (lambda=10)
For each we cluster the latent and measure internal quality, external agreement
with Duval (at the silhouette-optimal k and at k=7), and the severity leakage
R^2(m | z). Outputs a table + figures under results/.
"""
import _bootstrap  # noqa: F401
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from dga.config import PROJECT_ROOT, load_config, set_seed
from dga import data as dga_data, conventional, clustering, evaluation, viz
from dga.compositional import build_composition
from dga.models.sdcae import train_sdcae

cfg = load_config()
viz.set_paper_style()


def representation_inputs(df, comp):
    """Build the three input matrices (same 5 gases, same rows), column-standardised."""
    sub = df.loc[comp.index, comp.gases].fillna(0.0).clip(lower=0.0)
    raw_log = np.log1p(sub.to_numpy())
    std = lambda A: StandardScaler().fit_transform(A).astype("float32")
    return {
        "raw-log": std(raw_log),     # encodes magnitude + composition (severity baseline)
        "proportions": std(comp.P),  # naive per-sample normalisation
        "CLR": std(comp.C),          # compositional (Aitchison)
    }


SEEDS = [0, 1, 2, 3, 4]


def fit_repr(X, m, latent_dim, lambd, seed):
    """Train SD-CAE on input X (lambda=0 => plain AE + passive m-probe). Returns latent Z."""
    set_seed(seed)
    scfg = SimpleNamespace(**{**dict(cfg.sdcae), "latent_dim": latent_dim,
                              "lambda_adv": lambd, "seed": seed})
    res = train_sdcae(X, m, scfg, verbose=False)
    return res.model.encode(X)


def evaluate(Z, m, duval):
    """Internal + external (silhouette-k and k=7) metrics + severity leakage."""
    sel = clustering.fit(Z, cfg.clustering)                       # silhouette-optimal k
    internal = evaluation.internal_metrics(Z, sel.labels)
    ext_sel = evaluation.external_metrics(sel.labels, duval)
    lab7, _ = clustering.fit_kmeans(Z, 7, seed=42)               # fixed k=7 (Duval granularity)
    ext_7 = evaluation.external_metrics(lab7, duval)
    r2 = r2_score(m, LinearRegression().fit(Z, m).predict(Z))     # severity leakage (linear)
    return {
        "k_sel": sel.k, "silhouette": internal["silhouette"],
        "ARI@ksel": ext_sel["ARI"], "ARI@k7": ext_7["ARI"], "NMI@k7": ext_7["NMI"],
        "R2_m": float(r2),
    }, lab7


def main():
    df = dga_data.load_clean()
    df = df.join(conventional.diagnose(df))
    comp = build_composition(df, cfg)
    duval = df.loc[comp.index, "duval"]
    inputs = representation_inputs(df, comp)

    variants = [
        ("raw-log",     "raw-log",     0.0),
        ("proportions", "proportions", 0.0),
        ("CLR (lambda=0)", "CLR",       0.0),
        ("SD-CAE (lambda=1)",  "CLR",   1.0),
        ("SD-CAE (lambda=10)", "CLR",  10.0),
    ]
    keys = ["ARI@k7", "NMI@k7", "ARI@ksel", "silhouette", "R2_m"]

    rows, maps = {}, {}
    for latent_dim in (2, 3):
        for name, inp, lambd in variants:
            runs = []
            for seed in SEEDS:
                Z = fit_repr(inputs[inp], comp.m, latent_dim, lambd, seed)
                metrics, lab7 = evaluate(Z, comp.m, duval)
                runs.append(metrics)
                if latent_dim == 2 and seed == SEEDS[0]:
                    maps[name] = (Z, lab7)
            agg = {f"{k}_mean": round(float(np.mean([r[k] for r in runs])), 3) for k in keys}
            agg.update({f"{k}_std": round(float(np.std([r[k] for r in runs])), 3) for k in ("ARI@k7", "R2_m")})
            agg["k_sel"] = int(np.median([r["k_sel"] for r in runs]))
            rows[f"{name} [dim={latent_dim}]"] = {"latent_dim": latent_dim, **agg}
            print(f"[dim={latent_dim}] {name:20s} "
                  f"ARI@k7={agg['ARI@k7_mean']:.3f}+/-{agg['ARI@k7_std']:.3f}  "
                  f"NMI@k7={agg['NMI@k7_mean']:.3f}  R2_m={agg['R2_m_mean']:.3f}+/-{agg['R2_m_std']:.3f}  "
                  f"k_sel~{agg['k_sel']}")

    table = pd.DataFrame(rows).T
    tdir = PROJECT_ROOT / "results" / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(tdir / "sdcae_ablation.csv")
    print("\nwrote", tdir / "sdcae_ablation.csv")

    # --- figure 1: ARI@k7 and R2(m|z) across variants, mean+/-std (dim=2) -----
    d2 = table[table["latent_dim"] == 2]
    labels = [i.split(" [")[0] for i in d2.index]
    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(x, d2["ARI@k7_mean"], yerr=d2["ARI@k7_std"], capsize=3, color="#1e88e5")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylabel("ARI vs Duval (k=7)"); ax1.set_title("Fault-type agreement (higher = better)")
    ax2.bar(x, d2["R2_m_mean"], yerr=d2["R2_m_std"], capsize=3, color="#ef5350")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("R2(m | z)  severity leakage")
    ax2.set_title("Severity encoded in latent (lower = more disentangled)")
    viz.save_fig(fig, "sdcae_ablation")

    # --- figure 2: learned diagnostic map (best variant by mean ARI@k7, dim=2)
    best = d2["ARI@k7_mean"].astype(float).idxmax().split(" [")[0]
    Z, _ = maps[best]
    cls = pd.Series(duval.to_numpy(), index=comp.index).fillna("ND").astype(str)
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in ["PD", "T1", "T2", "T3", "D1", "D2", "DT", "ND"]:
        msk = (cls == c).to_numpy()
        if msk.any():
            ax.scatter(Z[msk, 0], Z[msk, 1], s=7, alpha=0.6,
                       color=viz.DUVAL_COLORS.get(c, "#333333"), label=c)
    ax.set_title(f"Learned diagnostic map: {best} latent, coloured by Duval")
    ax.set_xlabel("z1"); ax.set_ylabel("z2"); ax.legend(fontsize=7, markerscale=1.5)
    viz.save_fig(fig, "sdcae_diagnostic_map")

    print(f"best variant by ARI@k7 (dim=2): {best}")
    print("wrote results/figures/sdcae_ablation.{png,pdf} and sdcae_diagnostic_map.{png,pdf}")


if __name__ == "__main__":
    main()
