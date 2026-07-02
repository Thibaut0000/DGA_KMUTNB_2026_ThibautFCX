"""Step 3 - label-free Health/Risk Index + fleet ranking + validation figures.

    python scripts/run_health_ranking.py

Builds the per-unit risk score (severity + acetylene-aware + temporal + anomaly,
physical features only), ranks the fleet, and validates it against `fault_note`
(AUC, precision@k, note rate by risk decile). Outputs a table + figures to results/.
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dga.config import PROJECT_ROOT, load_config
from dga import data as dga_data, health, viz

cfg = load_config()
viz.set_paper_style()

# Domain-motivated weights: acetylene (arcing) is the priority -> boosted. NOT tuned on the label;
# the acetylene sweep below is reported only as a sensitivity analysis.
FINAL_W = {"severity": 1.0, "acetylene": 2.0, "temporal": 1.0, "anomaly": 1.0}
FULL = ["severity", "acetylene", "temporal", "anomaly"]


def main():
    df = dga_data.load_clean()
    feats = health.assemble_features(df, cfg)
    y = feats["fault_note"].astype(int).to_numpy()
    base = float(y.mean())

    # --- component ablation + acetylene sensitivity (transparency) -------------
    rows = {}
    for name, comps in [("severity", ["severity"]),
                        ("severity+acetylene", ["severity", "acetylene"]),
                        ("+temporal", ["severity", "acetylene", "temporal"]),
                        ("+anomaly (full)", FULL),
                        ("anomaly only", ["anomaly"])]:
        e = health.evaluate_ranking(health.risk_index(feats, FINAL_W, comps).feats["risk_score"], y)
        rows[name] = {"AUC": round(e["AUC"], 3), "prec@30": round(e["prec@30"], 3),
                      "prec@63": round(e["prec@63"], 3)}
    # TCG-only conventional severity baseline
    tcg_auc = health.evaluate_ranking(np.log1p(feats["latest_TCG"]), y)["AUC"]
    rows["[baseline] TCG only"] = {"AUC": round(tcg_auc, 3), "prec@30": np.nan, "prec@63": np.nan}
    table = pd.DataFrame(rows).T
    tdir = PROJECT_ROOT / "results" / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(tdir / "health_ranking_ablation.csv")
    print(table.to_string())

    acet_sweep = {wa: health.evaluate_ranking(
        health.risk_index(feats, {**FINAL_W, "acetylene": wa}, FULL).feats["risk_score"], y)["AUC"]
        for wa in [0.5, 1.0, 1.5, 2.0, 3.0]}
    print("\nacetylene-weight sensitivity (AUC):", {k: round(v, 3) for k, v in acet_sweep.items()})

    # --- final index ----------------------------------------------------------
    rm = health.risk_index(feats, FINAL_W, FULL)
    e = health.evaluate_ranking(rm.feats["risk_score"], y)
    print(f"\nFINAL index: AUC={e['AUC']:.3f}  base={base*100:.1f}%  "
          f"prec@10={e['prec@10']*100:.0f}% prec@30={e['prec@30']*100:.0f}% prec@63={e['prec@63']*100:.0f}%")

    # ===================== FIGURES ===========================================
    saved = []
    # 1. note rate by risk decile (the headline validation)
    dr = np.array(e["decile_note_rate"]) * 100
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(range(10), dr, color="#1e88e5")
    ax.axhline(base * 100, color="k", ls="--", lw=1, label=f"fleet base rate {base*100:.1f}%")
    ax.set_xticks(range(10)); ax.set_xticklabels([f"D{i+1}" for i in range(10)])
    ax.set_xlabel("risk decile (D1 = riskiest)"); ax.set_ylabel("fault-note rate (%)")
    ax.set_title("Fault-note rate by Health/Risk decile"); ax.legend(fontsize=8)
    saved.append(viz.save_fig(fig, "health_decile_note_rate"))

    # 2. lift curve: precision@top-k vs base
    s = pd.Series(rm.feats["risk_score"].to_numpy())
    order = s.sort_values(ascending=False).index.to_numpy()
    ks = np.arange(10, len(s) + 1, 5)
    prec = [y[order[:k]].mean() for k in ks]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(ks, np.array(prec) * 100, color="#1e88e5", lw=2, label="risk index")
    ax.axhline(base * 100, color="k", ls="--", lw=1, label=f"base rate {base*100:.1f}%")
    ax.set_xlabel("top-k riskiest units"); ax.set_ylabel("precision@k: fault-note rate (%)")
    ax.set_title("Precision@k of the risk ranking"); ax.legend(fontsize=8)
    saved.append(viz.save_fig(fig, "health_precision_at_k"))

    # 3. component-ablation AUC
    abl = table.drop(index=["anomaly only"], errors="ignore")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.barh(range(len(abl)), abl["AUC"], color="#1e88e5")
    ax.set_yticks(range(len(abl))); ax.set_yticklabels(abl.index); ax.invert_yaxis()
    ax.set_xlabel("AUC vs fault_note"); ax.set_xlim(0.5, 0.78)
    ax.set_title("Risk-index component ablation")
    for i, v in enumerate(abl["AUC"]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
    saved.append(viz.save_fig(fig, "health_component_ablation"))

    # --- the actual ranking deliverable: top 12 riskiest units ----------------
    cols = ["risk_rank", "risk_score", "latest_H2", "latest_C2H2", "c2h2_max",
            "latest_TCG", "rate_H2", "n_samples", "fault_note"]
    top = rm.feats.sort_values("risk_score", ascending=False).head(12)[cols].round(2)
    top.to_csv(tdir / "health_top_ranked.csv")
    print("\nTop 12 riskiest units:")
    print(top.to_string())
    print("\nwrote", tdir / "health_ranking_ablation.csv", "+ health_top_ranked.csv")
    print("figures:", ", ".join(p.name for p in saved))


if __name__ == "__main__":
    main()
