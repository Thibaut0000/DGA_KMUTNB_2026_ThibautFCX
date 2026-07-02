# CLAUDE.md — project context for Claude Code

> Claude Code reads this file automatically at the start of every session. Keep it
> short, current, and factual. Update it when structure or conventions change.

## What this project is

A research-initiation project (CESI × KMUTNB, 5 weeks, deliverable = **IEEE paper**)
building a **label-free, self-supervised framework for transformer health/risk ranking
from Dissolved Gas Analysis (DGA)** (aligned to the supervisor's brief, 2026-06-18).
Goal: rank a fleet by incipient-fault risk **without labels**, by learning a
**self-supervised, temporal** representation of each unit's DGA history and scoring
severity + deviation (**acetylene-aware**). The data is longitudinal (628 units, 4,563
samples, mean 7.3/unit over ~4.5 yr). The **SD-CAE** compositional model is a *component*
that separates fault *type* from severity (the interpretable "why" behind a rank), not the
headline. Benchmarked against conventional methods (IEEE C57.104 / IEC 60599 / Rogers / Duval).

Full plan + contributions: `docs/problem_statement.md`. Older plan: `docs/PROJECT_PLAN.md`.
How to use Claude here: `docs/GUIDE_CLAUDE_CODE.md`.

## Repository map

```
config/default.yaml      all knobs (features, preprocessing, model, clustering)
data/raw/                dga_main_tank.xlsx (the dataset; do not edit by hand)
data/processed/          generated clean frame
src/dga/                 the package — import as `import dga`
  data.py                load + clean the xlsx (gases are text, dates, NB notes, fault_note)
  preprocessing.py       feature matrix: clip → impute → log1p → scale
  conventional.py        Duval Triangle 1 + IEC 60599 + Rogers (baselines/labels)
  compositional.py       CLR of gas proportions + zero replacement (SD-CAE input)
  temporal.py            per-unit C57.104-style generation rates + rises (longitudinal)
  health.py              label-free Health/Risk Index (severity+acetylene+temporal+anomaly)
  models/autoencoder.py  MLP AE + train loop (early stopping)
  models/sdcae.py        severity-disentangled compositional AE (fault-type component)
  models/vae.py          beta-VAE + ELBO train loop
  clustering.py          KMeans/GMM/HDBSCAN + silhouette model selection
  anomaly.py             reconstruction / IsolationForest / OCSVM + note validation
  evaluation.py          internal (silhouette, DBI, CH) + external (ARI, NMI) metrics
  viz.py                 paper-quality matplotlib helpers
scripts/                 prepare_data → train_ae → run_clustering; run_sdcae_ablation;
                         make_week1_figures; run_health_ranking
notebooks/               01_eda … 07_evaluation (runnable starting points)
dashboard/               Streamlit app (app.py + lib/ + views/) wired live to `dga`:
                         fleet risk ranking, unit drill-down, SD-CAE map, surveillance-bias
paper/                   IEEEtran LaTeX (main.tex + sections/ + references.bib)
results/                 figures/, tables/, models/ (all generated outputs)
docs/                    problem_statement (current plan), PROJECT_PLAN (older),
                         literature, lab_notebook, GUIDE_CLAUDE_CODE, methodology
  refs/notes/            exhaustive per-paper reading notes (one .md each; index README)
```

## Commands

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/prepare_data.py      # clean + conventional diagnoses → data/processed
python scripts/train_ae.py          # train AE → results/models/{autoencoder.pt,latent.npy}
python scripts/run_clustering.py    # cluster + evaluate → results/{tables,figures}

jupyter lab                          # notebooks 01–07
streamlit run dashboard/app.py       # interactive fleet-health dashboard (localhost:8501)
cd paper && latexmk -pdf main.tex    # build the paper (no local TeX: use texlive Docker)
```
Scripts/notebooks use `scripts/_bootstrap.py` (or a `sys.path` line) to import `dga`
without installing. No GPU needed — the data is small; CPU PyTorch is fine.

## Domain cheat-sheet (so suggestions are physically sensible)

- **Diagnostic gases:** H2 (partial discharge), CH4 & C2H6 (low-temp thermal),
  C2H4 (high-temp thermal), C2H2 (arcing / high-energy discharge), CO & CO2
  (cellulose/paper degradation). O2/N2 are atmospheric (excluded by default);
  TCG is a sum of others (excluded — leakage).
- **Fault classes** (Duval/IEC): PD, D1 (low-energy discharge), D2 (high-energy
  discharge), T1 (<300 °C), T2 (300–700 °C), T3 (>700 °C), DT (mixed).
- **Data quirks:** gas values stored as *text* with `-`/`''` for missing; heavy
  right-skew (→ log1p); isolated data-entry outliers (→ clip).
- **Field notes are NOT a clean label.** `has_note` (391 rows) is a heterogeneous
  operational log — mostly admin/maintenance (*research*, *repeat*, *de-energize*,
  *overhaul*, Thai text), only a minority real faults. Use a **fault-filtered** label
  (trip|buchholz|arc|discharge|bushing… minus research|repeat|energize…) ≈ 54 units (8.6%)
  for risk validation. Raw `has_note` gives AUC 0.46 (chance); fault-filtered gives AUC ~0.70.
- **Longitudinal:** 628 units, mean 7.3 samples/unit (median 6), ~4.5-yr span, ~6-mo
  interval — enough for temporal features (`src/dga/temporal.py`: C57.104-style ppm/yr rates).
- **Confound — do NOT rank on `n_samples`/`span_years`.** They predict `fault_note` (AUC
  0.76/0.65) only because faulted units get re-sampled more (a consequence, not a cause).
  Risk index must use *physical gas* features only.
- **For risk, H2 > TCG.** H2 (the universal early-fault gas) + acetylene + CO2 + the H2
  generation rate gives a label-free risk index AUC ~0.72 vs `fault_note` (x2.7 top-decile
  lift), matching the supervised bound. TCG alone is weak (AUC ~0.52).

## Conventions

- **Reproducibility:** always `set_seed(cfg.seed)`; never hard-code paths — read
  from `config/default.yaml` via `dga.config.load_config()`.
- **Outputs:** every figure/table/model goes under `results/`. Nothing that lands
  in the paper may come from a model's head — it must be produced by code in `results/`.
- **Style:** small, typed, documented functions; prefer editing `config` over
  changing code for experiments. Keep notebooks thin — logic lives in `src/dga`.
- **Citations:** never invent references or fill in bibliographic details from
  memory. Mark unknowns with `\todo{}` and let Thibaut verify the source.
- **Professional tone — no emojis.** No emojis or decorative icons anywhere in the
  project (code, Markdown, notebooks, comments, the paper, commit messages). Use plain
  status words instead: `done` / `to do` / `partial` / `Note:` / `[x]`. Keep standard
  technical typography (arrows `→`, `≈`, `²`, units) — that is not an emoji.
- **Ambition with credibility:** this is a 5-week project by one student, but the goal is a
  genuine methodological contribution (a proposed method/algorithm), not just an empirical
  study. Aim high; keep credibility through honest evaluation, ablations, and hedged novelty
  ("to the best of our knowledge"). Never claim a *result* we have not measured, and don't
  claim to beat supervised methods (different setting). Build novelty on established ideas
  (compositional data analysis, disentanglement) applied in a new, label-free way to DGA.

## Current status — Week 2, 24 Jun (reframed after a research campaign)

Built and validated: SD-CAE fault-type representation, a label-free Health/Risk index, conventional
+ anomaly comparisons. Then a 5-probe multi-agent research campaign (Workflow) reshaped the emphasis.

**THREE contributions (see `docs/problem_statement.md`, `docs/research_review.md`):**
- **SD-CAE (primary, confound-free):** CLR-composition autoencoder separating fault *type* from
  *severity*; **ARI 0.47 +/- 0.06 vs Duval** (vs 0.14 raw, 5 seeds); adversary = negative ablation
  (deploy lambda=0). Target = Duval (a rule) -> confound-free. Code: `compositional.py`,
  `models/sdcae.py`, `scripts/run_sdcae_ablation.py`.
- **Surveillance-bias finding (primary, methodological):** the `fault_note` label is confounded -
  `n_samples` alone **AUC 0.76 > the 0.74 index**; temporal hold-out forward AUC 0.578 -> **0.514**
  once n_samples is controlled. The label measures operator attention, not gas chemistry. A
  confound-free chemistry target (arcing onset from C2H4+H2) is weakly predictable (AUC ~0.63).
- **Health/Risk index (secondary, applied):** severity(H2-led) + acetylene-aware + temporal(H2 rate)
  + anomaly(AE+IForest). AUC **0.74** vs fault_note, **beats conventional severity** (0.52-0.56), x4.4
  top-10, robust (+/-0.001). MUST be reported WITH the n_samples=0.76 confound floor. Code: `health.py`,
  `temporal.py`, `scripts/run_health_ranking.py`, `run_health_comparison.py`.

**Tested and rejected (rigorous, confound-free negatives):** prognostics/RUL (Cox C-index 0.565, not
significant; GRU gain was the n_samples confound), Deep SVDD (collapses), PyOD battery, VAE/contrastive,
SD-CAE-type-into-index, richer temporal features, Duval-Pentagon zones (unsourced).

**Next:** write the IEEE paper around the reframed (confound-free-first) contributions; flag to the
supervisor that a better label (Thai-note translation, +16 units; or KMUTNB failure records) is the
highest-leverage data improvement for any risk/prognostic claim.
