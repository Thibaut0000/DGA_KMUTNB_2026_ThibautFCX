"""Week-1 demo figures — EDA + the AE baseline "severity not fault-type" story.

    python scripts/make_week1_figures.py

Regenerates a coherent set of paper-style figures under results/figures/ from the
real data + the trained AE (run prepare_data.py and train_ae.py first). Nothing here
is hard-coded: numbers come from the data, the config and the saved model.
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from dga.config import PROJECT_ROOT, load_config, set_seed
from dga import data as dga_data, conventional, viz, clustering, anomaly
from dga.preprocessing import build_feature_matrix
from dga.models.autoencoder import Autoencoder

cfg = load_config()
set_seed(cfg.seed)
viz.set_paper_style()

# ----- data, conventional labels, feature matrix, latent, clusters -----------
df = dga_data.load_clean()
diag = conventional.diagnose(df)
df = df.join(diag)
fm = build_feature_matrix(df, cfg)
sub = df.loc[fm.index]                      # rows kept by preprocessing
duval = sub["duval"].fillna("ND").astype(str)   # ND = not-determined
gases = fm.features

# rebuild the trained AE and get latent + reconstruction error on the same rows
ae = Autoencoder(len(gases), tuple(cfg.autoencoder.hidden_dims), cfg.autoencoder.latent_dim,
                 cfg.autoencoder.activation, cfg.autoencoder.dropout)
ae.load_state_dict(torch.load(PROJECT_ROOT / "results" / "models" / "autoencoder.pt"))
Z = ae.encode(fm.X)
recon = ae.reconstruction_error(fm.X)

cl = clustering.fit(Z, cfg.clustering)
print(f"clusters: k={cl.k}  silhouette={cl.silhouette:.3f}")

# per-sample "severity" = total diagnostic gassing (the magnitude the AE sees)
severity = np.log1p(sub[gases].sum(axis=1).to_numpy())

# ============================ FIGURES ========================================
saved = []

# 1. gas distributions (log1p)
saved.append(viz.save_fig(viz.plot_gas_distributions(df, gases, log=True), "eda_gas_distributions"))

# 2. sparsity: % of zero readings per gas (why C2H2 is special)
zero_frac = (sub[gases] == 0).mean().sort_values() * 100
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.bar(zero_frac.index, zero_frac.values, color="#3b7dd8")
ax.set_ylabel("% zero readings"); ax.set_title("Per-gas sparsity (zeros, not NaN)")
for x, v in enumerate(zero_frac.values):
    ax.text(x, v + 1, f"{v:.0f}", ha="center", fontsize=7)
saved.append(viz.save_fig(fig, "eda_sparsity"))

# 3. conventional fault mix (Duval) — the imbalanced external labels
counts = duval.value_counts()
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.bar(counts.index, counts.values, color="#d8743b")
ax.set_ylabel("samples"); ax.set_title("Duval fault-class distribution (weak labels)")
for x, v in enumerate(counts.values):
    ax.text(x, v + 10, str(v), ha="center", fontsize=7)
saved.append(viz.save_fig(fig, "eda_duval_mix"))

# 4. correlation heatmap of diagnostic gases (log space)
gl = np.log1p(sub[gases])
corr = gl.corr()
fig, ax = plt.subplots(figsize=(4.6, 4.0))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(gases))); ax.set_xticklabels(gases, rotation=90)
ax.set_yticks(range(len(gases))); ax.set_yticklabels(gases)
for i in range(len(gases)):
    for j in range(len(gases)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
fig.colorbar(im, fraction=0.046); ax.set_title("Diagnostic-gas correlation (log1p)")
saved.append(viz.save_fig(fig, "eda_gas_correlation"))

# 5. samples inside the Duval Triangle 1, with the 7 coloured fault zones
saved.append(viz.save_fig(viz.plot_duval_triangle(sub, classes=duval), "duval_triangle"))

# 6. latent space coloured by Duval class  — KEY: classes do NOT separate
saved.append(viz.save_fig(
    viz.plot_latent_2d(Z, duval, title="AE latent (PCA) coloured by Duval class"),
    "latent_by_duval"))

# 7. latent space coloured by KMeans cluster
saved.append(viz.save_fig(
    viz.plot_latent_2d(Z, pd.Series(cl.labels, index=sub.index).astype(str),
                       title=f"AE latent (PCA) coloured by cluster (k={cl.k})"),
    "latent_by_cluster"))

# 8. latent coloured by total gassing (severity) — KEY: a clean severity gradient
pca = PCA(n_components=2, random_state=cfg.seed).fit_transform(Z)
fig, ax = plt.subplots(figsize=(5.6, 4.5))
sc = ax.scatter(pca[:, 0], pca[:, 1], c=severity, cmap="viridis", s=8, alpha=0.8)
fig.colorbar(sc, label="log1p(total diagnostic gas, ppm)")
ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
ax.set_title("AE latent (PCA) coloured by gassing severity")
saved.append(viz.save_fig(fig, "latent_by_severity"))

# 9. quantify it: best linear read-out of severity from the 3-D latent (R^2)
#    (severity is spread across latent dims, so regress on all 3 — not just PC1)
reg = LinearRegression().fit(Z, severity)
pred = reg.predict(Z)
r2 = r2_score(severity, pred)
fig, ax = plt.subplots(figsize=(5.0, 4.0))
ax.scatter(pred, severity, s=6, alpha=0.4, color="#3b7dd8")
lo, hi = severity.min(), severity.max()
ax.plot([lo, hi], [lo, hi], "k--", lw=1)
ax.set_xlabel("severity predicted from latent (3-D)"); ax.set_ylabel("log1p(total gas)")
ax.set_title(f"Latent encodes gassing severity  (R² = {r2:.2f})")
saved.append(viz.save_fig(fig, "latent_vs_severity"))

# 11. cluster x Duval cross-tab (row-normalised %) — why ARI is ~0: types not split
ct = pd.crosstab(pd.Series(cl.labels, name="cluster"), duval.rename("duval"))
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(6.2, 1.2 + 0.5 * len(ct)))
im = ax.imshow(ct_pct.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels(ct.columns)
ax.set_yticks(range(ct.shape[0])); ax.set_yticklabels([f"cluster {i}" for i in ct.index])
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        ax.text(j, i, ct.iloc[i, j], ha="center", va="center", fontsize=7,
                color="white" if ct_pct.iloc[i, j] > 55 else "black")
fig.colorbar(im, label="% of cluster", fraction=0.046)
ax.set_title("Cluster vs Duval class (counts; colour = row %)")
saved.append(viz.save_fig(fig, "cluster_vs_duval"))

# 10. anomaly score (reconstruction error): field-note rows vs the rest
thr = np.quantile(recon, cfg.anomaly.threshold_quantile)
has_note = sub["has_note"].to_numpy()
fig, ax = plt.subplots(figsize=(5.6, 3.5))
bins = np.linspace(recon.min(), np.quantile(recon, 0.999), 60)
ax.hist(recon[~has_note], bins=bins, color="#9bbce8", label="no note", density=True)
ax.hist(recon[has_note], bins=bins, color="crimson", alpha=0.55, label="field note", density=True)
ax.axvline(thr, color="k", ls="--", lw=1, label=f"top-{(1-cfg.anomaly.threshold_quantile)*100:.0f}% threshold")
ax.set_xlabel("reconstruction error"); ax.set_ylabel("density")
ax.set_title("Anomaly score: rows with a field note skew higher"); ax.legend(fontsize=8)
saved.append(viz.save_fig(fig, "anomaly_recon_error"))

# small console summary for the demo
flags = anomaly.threshold_flags(recon, cfg.anomaly.threshold_quantile)
ev = anomaly.evaluate_against_notes(flags, has_note)
print(f"severity read-out from latent: R^2 = {r2:.3f}")
print(f"cluster sizes: {np.bincount(cl.labels).tolist()}")
print(f"anomaly vs notes (top {int((1-cfg.anomaly.threshold_quantile)*100)}%): "
      f"precision={ev['precision']:.2f} recall={ev['recall']:.2f} "
      f"(flagged {ev['n_flagged']}, notes {ev['n_notes']})")
print(f"wrote {len(saved)} figures to results/figures/:")
for p in saved:
    print("  ", p.name)
