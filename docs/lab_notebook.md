# Lab notebook

Daily, ~5 minutes. This is *your* record (keep ownership). Format per entry:
**what I did · what I found (numbers) · what it implies · next**.

---

## 2026-06-15 — Day 1
- **Did:** set up env; ran the scaffold pipeline end-to-end on the real data.
- **Found:** AE val MSE ~0.10; AE+KMeans silhouette ~0.90 but ARI vs Duval ~0.015.
- **Implies:** latent likely captures gassing *severity*, not fault *type* → investigate normalisation.
- **Next:** start literature review; lock preprocessing in notebook 01.

## 2026-06-16 — Day 2 (Week 1)
- **Did:** rebuilt env (Python 3.13, CPU torch 2.12); added `pyarrow` to requirements so
  `data/processed/dga_clean.parquet` writes as the config declares (was silently falling
  back to CSV with a stale parquet lingering). Re-ran the full pipeline end-to-end OK.
  Deep EDA: enriched + executed `notebooks/01_eda.ipynb` (sparsity/skew table, O2-outlier
  callout, Duval fault-mix bar, locked-facts cell). Prepared P1 reading: `docs/refs/` drop
  folder + reading-note stubs in `literature.md` (incl. the 2026 hybrid AI-Score paper).
- **Found (from results/):** 4,563 samples · 628 transformers · 42 manufacturers · 2019–2024.
  Gas missingness 0 % — sparsity is *zeros* (C2H2 97.4 % zero, median 0; C3H6 median 0).
  O2 max = 719,534,824 ppm (data-entry error; median 7,968) → justifies the 99.9-q clip.
  C2H2/C2H4 stay heavy-skewed even after log1p. Duval mix imbalanced: PD+T1 ≈ 65 %,
  D1/D2/DT = 70 rows (~1.5 %), 7.7 % not-determined. `KV` is free-text multi-winding.
  Baseline (cluster_metrics.csv): AE+KMeans k=2 → silhouette 0.901, but ARI vs Duval
  0.0152, NMI 0.0588 (completeness 0.76 / homogeneity 0.03).
- **Implies:** high silhouette + ~0 ARI + high completeness/low homogeneity = a 2-way
  *severity* split, not fault *type*. The Week-2 per-sample normalisation (gas proportions)
  is the lever. The class imbalance means ARI alone will undersell type structure if found.
- **Next:** read P1 PDFs once dropped in `docs/refs/` (start hybrid_aiscore2026 + Duval 2002);
  finalize feature/preprocessing decisions; sharpen the research question + 2–3 contributions.
- **Reading (P1, same day):** read all 5 dropped PDFs (Duval 2002, Wang 2016, Liu/DBSCAN 2020,
  Saleh 2026, C57.104-2019 skim); notes in `literature.md`. **Key cross-paper finding:** every
  method that recovers fault *type* works on per-sample **proportions/ratios** (Duval %, Wang IEC
  ratios, Liu % content) → published support for the Week-2 normalisation experiment. Wang & Liu
  both claim "unsupervised" but use labels (BP head / amplification tuning) → my label-free angle is
  genuinely novel. Saleh 2026 is supervised & *fuses* the conventional rules (opposite of mine) →
  cite as supervised ceiling, not a head-to-head with ARI. Filled verified authors/DOIs in
  `references.bib` (Liu et al.; Saleh et al., EPSR 256:112877).
- **Figures + corrected baseline read (same day):** added `scripts/make_week1_figures.py` → 11
  paper-style PNGs in `results/figures/`. Sharper finding than "severity not type": the latent
  *does* encode magnitude (**R²=0.61** of log-total-gas linearly readable from the 3 latent dims),
  but the silhouette-optimal **k=2 is an outlier split, not a severity axis** — cluster 1 = only
  **41** extreme-gassing units (‖X‖ 10.4 vs 2.26) enriched in arcing (**23/30 D2, 14/24 D1**);
  cluster 0 = the other 4521 with **all PD/T1/T2/T3 collapsed together** → that's why ARI≈0.015.
  Anomaly (recon error, top-5%) vs field notes is weak/indicative: precision 0.21, recall 0.12.
- **Duval triangle viz + classifier audit:** rewrote `viz.plot_duval_triangle` to rasterise the 7
  fault zones straight from `conventional.duval_triangle_1` (zones now *exactly* == point labels;
  verified vectorised==scalar on 4210/4210). Classifier was **not** buggy — only the plot lacked
  zone shading. Two real findings: (a) points hug the %C2H2≈0 edge (acetylene 97% zero); (b) **PD is
  inflated — 1511 samples**, yet PD is a thin %CH4≥98% sliver; those rows are CH4-dominant with
  **median total gas 2790 ppm** (only 7/1511 below 50 ppm) → likely stray-gassing/low-temp that
  Duval Triangle 1 lacks a zone for. A concrete limitation-of-conventional-baseline argument for the
  paper. \todo{decide whether to gate Duval on a min-concentration before using it as a label.}

## 2026-06-17 — Day 3 (Week 1) · mid-week checkpoint
- **Where we are vs the Week-1 plan (M1 due Fri):**
  - done - Env + smoke test · done - deep EDA (notebook 01) · done - conventional baselines run +
    distributions · done - Duval/cluster/anomaly figures · done - P1 literature read (5) + notes +
    `references.bib` reconciled. **Ahead** on the technical side (some W2/W3 steps already peeked).
  - to do - **Owed for M1:** (1) sharpened **research question + 2–3 contributions**; (2) **draft notes
    for paper §1–3** (`paper/sections/*.tex` are still scaffold stubs); (3) literature count 13→15+
    (P2/P3 already identified, just need 2–4 more skim entries).
- **Net:** strong on data/figures/positioning, behind on **writing**. Critical path to Friday = lock
  the problem statement + contributions, then seed §1–3 from `literature.md` + the EDA facts.
- **3 open decisions carried into W2:** Duval min-concentration gating · per-sample proportion
  normalisation (the core experiment) · acetylene weighting. None block the writing.
- **Housekeeping:** removed stale Python-3.10 `__pycache__`; updated `CLAUDE.md` status.
- **Professionalism + writing pass (afternoon):** stripped all emojis from every project file and
  added a no-emoji / scope-realism rule to `CLAUDE.md` Conventions. Toned down the contributions in
  `docs/problem_statement.md` to be credible for a 5-week project (empirical study, not SOTA claims).
  Wrote first-draft prose for paper sections 1--3 (Introduction, Related Work, Data) with verified
  numbers and correct `\cite` keys; replaced `\ce{}` with plain gas names (mhchem not loaded).
  Literature now at 15 in `references.bib` (added Duval & dePablo 2001, Mirowski & LeCun 2012, both
  from reference lists of papers we hold); links for 2 optional extras in `docs/refs/README.md`.
- **Direction set: propose a method, not just a study.** Reframed the project around **SD-CAE**
  (Severity-Disentangled Compositional Autoencoder): split each sample into magnitude
  (log total gas) + composition (CLR of gas proportions), encode only the composition into a
  severity-invariant fault-type latent (optional adversarial disentanglement), cluster it, flag
  anomalies on two axes (composition novelty + magnitude), and read a learned label-free diagnostic
  map. Rationale: directly fixes the H1 severity-vs-type failure mode; novelty is the label-free
  application of compositional analysis + disentanglement to DGA. Updated `docs/problem_statement.md`
  (RQ, proposed method, 3 contributions, risks) and drafted `paper/sections/04_method.tex` with the
  equations. `CLAUDE.md` Conventions: replaced the "no SOTA" rule with "ambition with credibility".
- **SD-CAE implemented + first ablation (evening).** New `src/dga/compositional.py` (CLR + multiplicative
  zero-replacement), `src/dga/models/sdcae.py` (composition AE + gradient-reversal adversary),
  `scripts/run_sdcae_ablation.py` (5 hydrocarbons; only the representation changes; 5 seeds, latent dim
  2 and 3). **Robust results (mean +/- std over 5 seeds, dim 2), ARI vs Duval @k=7:** raw-log
  0.135+/-0.020, proportions 0.150+/-0.025, **CLR 0.474+/-0.055**, SD-CAE(lambda=1) 0.310, SD-CAE(lambda=10)
  0.296. Severity leakage R2(m|z): raw-log **0.63** (0.81 at dim 3), CLR 0.27, SD-CAE(lambda=10) 0.12.
  Findings: (1) **H1/H2 confirmed** — the CLR compositional representation recovers fault *type*
  (~3.4x ARI over raw, outside error bars); it is specifically CLR, not naive proportions. (2) **The
  adversary is a negative ablation** — it lowers R2(m|z) but hurts ARI, i.e. forcing z _|_ m erases
  legitimate fault signal; severity-invariance is better obtained by the CLR construction alone.
  (3) Learned 2-D diagnostic map shows real Duval-type separation (D1/D2, PD, T3, T1/T2).
  Figures: `results/figures/sdcae_ablation.png`, `sdcae_diagnostic_map.png`. Caveat: ARI ~0.47 is
  *moderate agreement with a flawed rule*, not accuracy; the claim is the large relative gain.
  \todo{decide framing: keep adversary only as a reported negative ablation; the core is compositional.}

## 2026-06-18 — Day 4 (Week 1) · PIVOT to the supervisor's brief
- **Direction change.** Supervisor's instruction sheet = *fleet health/risk ranking* via
  **self-supervised + temporal** DGA + an **acetylene-aware risk model**, comparison to
  C57.104/IEC/Rogers/Duval. We had drifted to fault-*type* clustering (SD-CAE). Realigned:
  health-ranking is the headline; **SD-CAE becomes the fault-type explanation component**, and its
  severity/magnitude axis feeds the health index. Updated `problem_statement.md`, `CLAUDE.md` (project
  description + status + roadmap), paper title + abstract.
- **Temporal feasibility (confirmed viable):** 628 units, mean 7.3 samples/unit (median 6); 98% have
  >=2, 88% >=5; median span ~4.5 yr; ~6-month interval. Temporal features are well supported.
- **Critical finding — the weak label was contaminated.** `has_note` (391 rows) is a heterogeneous
  operational log: most frequent English notes are *research* (29), *repeat* (9), *De-energize*,
  *ENERGIZED*, *Cold standby*, *OVERHAUL*, *oil purify* — admin/maintenance, not faults; many others
  are Thai. Genuine faults are the minority (keyword hits: trip/buchholz 54, arc/discharge 33,
  bushing 17). Using raw `has_note` as a risk target gave **AUC 0.46** (below chance) and a U-shaped
  decile curve. **Fix:** a fault-filtered note label (trip|buchholz|arc|discharge|bushing|... minus
  research/repeat/energize/...) = 68 rows, **54 units (8.6%)**.
- **Pivot validated.** With the clean fault-note label, a simple label-free score ranks event-units:
  TCG severity AUC **0.641**, acetylene (max C2H2) 0.595, **severity+acetylene AUC 0.695** with
  **top-10% fault rate 24.2% vs 8.6% base (2.8x)**. So severity + acetylene-awareness is a real ranking
  signal; crude TCG-growth temporal did not help yet (needs better temporal features).
- **Steps 1-2 done (same day).** (1) `fault_note` formalised in `data.py` (keyword heuristic, 95
  rows / 71 units / 11.3 %; ~24 % of notes are Thai-only, not captured). (2) `src/dga/temporal.py`:
  per-unit features — `latest_<g>`, C57.104-style generation `rate_<g>` (ppm/yr, multi-point
  regression <=6 pts / <=2 yr), `rise_<g>` (log latest vs baseline), TCG, acetylene aggregates.
  Tested + iterated against `fault_note`:
  - **Confound found and excluded:** `n_samples` (AUC 0.76!) and `span_years` (0.65) just reflect that
    faulted units get re-sampled more — a consequence, not a predictor. **Never put them in the risk
    index** (and the label is mildly biased toward heavily-sampled units).
  - **The key severity gas is H2, not TCG.** H2+acetylene AUC 0.679 vs TCG+acetylene 0.608. H2 is the
    universal early-fault gas. CO2 (cellulose) adds (-> 0.714).
  - **Temporal helps, but it is the H2 generation rate, not the TCG rate** (TCG rate 0.557; rate_H2
    0.618). Naive equal-weight temporal on TCG *hurt* the top decile — wrong aggregate.
  - **Best label-free index so far:** H2 + acetylene + CO2 + rate_H2 -> AUC **0.720**, top-10 %
    fault rate **30 %** vs 11.3 % base (x2.7). It matches the supervised CV upper bound (0.695-0.699),
    which supports the label-free claim. Note the label is weak/noisy, so ~0.7 is likely near its ceiling.
- **Next (step 3):** formal Health/Risk Index module + fleet ranking + figures (note rate by decile,
  precision@k), then Deep SVDD and conventional comparison. \todo{realign paper §1-4 to the headline.}
- **Step 3 done (same day) - label-free Health/Risk Index.** New `src/dga/health.py` (severity
  H2-led + acetylene-aware [x2] + temporal H2-rate + self-supervised anomaly [AE recon + Isolation
  Forest]) + `scripts/run_health_ranking.py` (ranking, ablation, figures). Robustness iteration:
  required **>=3 points for a generation rate** (C57.104 multi-point; drops noisy 2-sample slopes),
  which improved results. **FINAL index AUC 0.740 vs fault_note** (TCG-only baseline 0.521);
  **prec@10 = 50 % (x4.4), prec@30 = 37 %, prec@63 = 30 %**; monotonic decile note-rate (D1 27 % ->
  D10 2 %). Ablation: severity 0.643 -> +acetylene 0.714 -> +temporal 0.733 -> +anomaly 0.740;
  acetylene-weight sensitivity 0.713->0.743 (validates the acetylene-aware model; weight set to 2.0 by
  domain, not tuned on the label). Honest caveats: anomaly adds only +0.007; the 2 riskiest units have
  arcing (C2H2 115/125) but no fault_note - the index surfaces events the (weak/Thai) notes miss; ~0.74
  is likely near the label's ceiling. Figures: `health_decile_note_rate`, `health_precision_at_k`,
  `health_component_ablation`; ranking in `results/tables/health_top_ranked.csv`.
- **Next (step 4):** Deep SVDD anomaly; compare ranking to conventional severity (C57.104 status);
  multi-seed robustness of the index; then realign paper §1-4 + seed §5-6 with these results.

## 2026-06-24 — Week 2 · step 4 (index evaluation: robustness, anomaly methods, conventional)
- **Multi-seed robustness (8 seeds):** health index AUC **0.741 +/- 0.001** (range 0.739-0.742);
  without anomaly 0.733 +/- 0.000 (anomaly contributes a stable +0.008). The headline number is citable.
- **Anomaly methods compared** (`src/dga/models/deep_svdd.py` added; `scripts/run_health_comparison.py`):
  Isolation Forest **0.603** > AE reconstruction 0.583 > **Deep SVDD 0.520**. Deep SVDD *collapses* on
  this small low-dimensional per-unit matrix (628x10; score std ~0.002-0.006 across configs) - the known
  failure mode without AE-pretraining. Deployed anomaly component stays **AE + Isolation Forest**;
  Deep SVDD reported as a fair, evaluated negative. (Answers the brief's anomaly-method comparison.)
- **Beats conventional severity** (AUC vs fault_note): TCG 0.521, C57.104-style condition count 0.538,
  Duval arcing (D1/D2/DT) 0.560, max-gas-p90-ratio 0.560 -- vs **Health index 0.740**. The label-free
  index clearly dominates rule-based severity (figure `health_vs_conventional`).
- **Units the rules miss:** 3 of the top-40 riskiest are "normal" by conventional rules (no arcing, no
  gas > fleet p90); one (KB2KT1A, C2H2=60) actually carries a real fault note -> concrete added value.
- Figures: `health_vs_conventional`, `anomaly_methods`; tables `anomaly_methods.csv`,
  `conventional_comparison.csv`. Doc tour: Cowork added Ramarao 2026 review + `docs/refs/notes/` +
  `week1_synthesis_EN.md` (all consistent, verified, no emojis).
- **Next:** realign paper §1-4 to the health-ranking headline + seed §5-6 (results) with the figures.

## 2026-06-24 — Week 2 · deep research pass + positioning (see docs/research_review.md)
- **State of the art (Ramarao 2026 review + web 2024-26):** field is overwhelmingly *supervised
  classification*; label-free/anomaly is thin; the named frontier is prognostics/RUL, online sensors,
  federated learning, digital twins. The hot "Health Index" ML work *predicts an expert HI* (needs a
  label). -> Our **label-free, temporal, fleet risk-ranking validated on field events** is in the
  under-served frontier: credible to-the-best-of-our-knowledge novelty.
- **Tested 3 new ideas:**
  1. **Duval Pentagon (2014, 5-gas):** implemented the verified coordinate math (axes 90/162/234/306/18,
     polygon centroid; pure gases land on axes). Zone polygons NOT hard-coded (could not source exact
     coordinates -> would violate no-invent rule). Better-at-stray-gassing per literature.
  2. **Stray-gassing / "PD inflation" (W1 claim) -- TEMPERED.** Per *unit* (latest state) only **97**
     units are PD and they are **not benign** (fault rate 15.5 % vs fleet 11.3 %). The PD inflation is a
     per-*sample* artifact, minor for ranking. Soften the W1 wording.
  3. **Temporal-representation depth.** Richer trajectory features (volatility/trend/peak-gap ~0.59) do
     NOT beat the C57.104 **H2 generation rate** (0.62). `nrises` looks strong (0.76) but is the
     **sampling confound** (corr 0.87 with `n_samples`) -> excluded. Weak label caps AUC ~0.74, so a
     learned GRU/LSTM would hit the same ceiling: **not worth the complexity**. Keeps the method simple.
- **Net:** no new code adopted into the pipeline (the tested ideas did not beat the current design for
  this weak label) -- a valuable, honest calibration. Positioning + future-work (RUL, contrastive
  temporal, digital twin, federated) captured in `docs/research_review.md`.

## 2026-06-24 (cont.) — Frontier probe: PROGNOSTIC / early-warning (predict FUTURE events)
- **Reframed the temporal question:** instead of ranking *current* risk, predict a *future* fault event
  from the past DGA trajectory (uses the dated notes as time-to-event). This is the review's named
  frontier (prognostics/RUL) and finally gives the temporal dimension a real job.
- **Feasibility:** 86 % of fault events have >=1 prior DGA sample (predictable); median ~93 d from last
  routine sample to the event. **Gases rise before events:** TCG +0.18 log (72 % of cases rising), H2
  +0.09 (57 %), C2H4 +0.07 -- but **C2H2 does NOT ramp** (20 %): arcing is sudden (physically sensible).
- **Prediction task (no leakage, features <= t):** predict an event in (t, t+horizon].
  - 1-year horizon (base 3.0 %): **H2 level** is the best single predictor (AUC 0.60); level+rate 0.60.
  - 2-year horizon: the **rate** overtakes the level (TCG rate 0.59 > level 0.50) -> clean finding:
    **rate = long-lead warning, level = imminent.**
- **Operational value (1-yr early warning):** top 2 % riskiest = **12.2 % event rate (x4.1 lift)**, top
  5 % = 8.1 % (x2.7); **median lead time 163 days (~5 months).** Modest AUC (0.61) but usable as a
  predictive-maintenance screen. Bounded by rare events + sudden arcing + weak labels.
- **Strategic read:** the project now spans NOW (risk ranking, AUC 0.74) and NEXT (prognostic early
  warning, x4 lift / 5-mo lead), plus type interpretability (SD-CAE). The prognostic element is the
  frontier novelty. \todo{decide with Thibaut whether to elevate prognostics to a headline component.}
- **Prognostics dug deeper (survival + learned temporal) -- VERDICT: weak, not this dataset.**
  (1) Time-varying **Cox survival** (lifelines): hazard ratios ~1.0, all **p > 0.4**, **C-index 0.565**
  -- DGA does not significantly predict time-to-event. (2) **GRU** (group-CV) looked better (0.638 vs
  0.574 simple) but it is the **sampling-length confound**: length alone 0.623, simple+length 0.651 ~
  GRU. Surveillance bias (watched units are event-prone), not real temporal signal. -> **Prognostics is
  NOT the headline.** Keep current-risk ranking (0.74) as the core; prognostics = honest tested
  limitation + future work. (Nearly mis-reported the GRU as a win; the confound check caught it.)
  Captured in `docs/research_review.md` section 5.

## 2026-06-24 (cont.) — Multi-agent research campaign (5 probes) + the label reckoning
- Ran a parallel Workflow: 5 research probes + adversarial confound verification. **All 5 NEGATIVE,
  all confound-free** (rigour): PyOD battery (no detector beats the index), self-supervised
  forecast-residual anomaly (0.53, clean but no signal), VAE+contrastive (~chance, ARI~0.09),
  SD-CAE-type-into-index (degrades -0.023 every seed).
- **PIVOTAL (probe 5): the `fault_note` label is surveillance-confounded.** `n_samples` alone AUC
  **0.76 > index 0.74**; strict temporal hold-out forward AUC 0.578 -> **0.514 (chance) after
  controlling n_samples**; the label encodes operator ATTENTION (sampling frequency), not gas
  chemistry. +~23% Thai-only notes missed (16 units). -> the fault_note-based risk *headline* is
  fragile and must be reported with the n_samples confound floor.
- **One clean confound-free signal:** chemistry-defined forward target "arcing onset (C2H2 appears in
  next 2 yr)" is weakly predictable from C2H4+H2 (AUC ~0.62-0.64, corr with n_samples ~0). "TCG
  doubling" target is flawed (regression to mean).
- **Reshapes the emphasis:** SD-CAE type (ARI 0.47 vs Duval, a rule = confound-free) is the more
  defensible contribution; the surveillance-bias demonstration is itself a methodological result;
  risk-ranking needs caveats; a better label (Thai translation / KMUTNB failure records) would be the
  highest-leverage data improvement. Full writeup in `docs/research_review.md` section 6.

## 2026-06-24 (cont.) — Formalised C2 + the chemistry signal into reproducible artifacts
- New `scripts/run_label_confound.py` reproduces, from repo code (not the workflow's throwaway scripts):
  - **Part 1 (confound floor):** AUC vs fault_note - TCG 0.521, C57.104 cond-count 0.538, Duval-arcing
    0.560, **Health index 0.740, n_samples (confound) 0.761**. Figure `label_confound_floor`.
  - **Part 2 (temporal hold-out, 560 units, 5.5% future-pos):** physics 0.541, n_samples-1st-half
    0.679, **physics | n_samples-controlled 0.501 (chance)**. Figure `label_confound_holdout`.
  - **Part 3 (confound-free chemistry target = arcing onset, 45 pos, corr with n_samples -0.01):**
    C2H4 0.643, H2 0.625 predict; n_samples 0.492 (chance). Figure `chemistry_target_arcing_onset`.
  Tables: label_confound_floor/holdout.csv, chemistry_target_arcing_onset.csv.
- **All 3 contributions now have reproducible code + figures:** C1 SD-CAE (run_sdcae_ablation), C2
  surveillance bias (run_label_confound), C3 health index (run_health_ranking/comparison). The science
  artifacts are done; remaining work is paper writing (S2, S4-rewrite, S5-7) + supervisor alignment.
- **Paper drafted (all 7 sections).** Rewrote S2 Related Work (Ramarao backbone + the label-bias gap),
  S4 Methods (SD-CAE + index + the surveillance-bias/chemistry evaluation protocol), and wrote S5
  Experiments, S6 Results (3 subsections, 6 figures wired), S7 Conclusion (limitations + future work).
  Consistency verified: 0 missing cite keys, 0 missing figures, 0 broken \ref; 4 rendered \todo left
  (abstract one-liner, hyperparameter table, 2 figure adds). No local LaTeX -> build on Overleaf/KMUTNB.
  Next: optional Reviewer-2 pass (/review-paper), fill the 4 todos, align with supervisor.

## 2026-06-25/26 — Week 2 · Reviewer-2 pass + paper hardening + first PDF builds
- **Did:** critical Reviewer-2 pass on all sections; fixed the fatal S1/abstract contribution
  mismatch; 4 successive multi-agent coherence audits (confirmed issues 4 -> 6 -> 1 -> 2, all fixed);
  new title (approved later): "Label-Free Compositional DGA Representations and a Surveillance-Bias
  Caution for Transformer Risk Screening". First reproducible PDF builds via the texlive Docker image.
- **Found:** references.bib had % comments INSIDE @entries + @standard type -> BibTeX silently dropped
  9 entries (all citations undefined). Fixed (comments outside braces, @misc). 6 pages, 0 unresolved.
- **Implies:** the paper is compile-clean and internally consistent; hedging aligned everywhere
  (fleet-scoped claims, "apparent" forward validity).

## 2026-06-30 — Week 3 · deep adversarial audit: the linear model wins
- **Did:** 5-lens multi-agent audit + 5 new adversarial experiments against our own headlines
  (scripts: run_representation_baselines, run_clr_robustness, run_validation_statistics,
  run_chemistry_blocked_stats, run_thai_label_extension). Full findings: docs/deep_audit_2026-06-30.md.
- **Found (the big one):** **PCA-2 of standardised CLR + KMeans = ARI 0.545 ± 0.002 vs Duval — beats
  the SD-CAE autoencoder (0.474 ± 0.055).** The gain is the log-ratio geometry, not the network.
  Robustness: temporal split (fit <2022, score 2022+) ARI 0.454; delta-sensitive (0.455/0.545/0.229
  for 1e-4/1e-3/1e-2); latest-per-unit 0.346. Statistics: 0.74 vs 0.76 NOT significant (paired
  bootstrap p=0.58); physics adds beyond n_samples (LR p=0.003) but n dominates (p=9e-13); chemistry
  target = 45 positive points from only **17 onset units** -> unit-blocked CIs [0.52,0.77]/[0.53,0.73]
  (naive permutation p was anti-conservative, replaced). **Thai notes: 69% of notes contain Thai**
  (not 23%); translated all 211 distinct notes, +6 tier-A units; on the repaired label the count's
  advantage disappears (0.734 vs 0.735). Literature verified: Dukarm 2020 (simplex, no log-ratios),
  Goldstein 2016 (conditioning on encounter count = our control), Aitchison 1982.
- **Implies:** C1 reframed around the compositional geometry (SD-CAE = negative ablation); C2
  sharpened (dominated-but-not-pure confound); every AUC now carries honest intervals.
- **Also:** git init + first commit (raw data excluded); run_all.py; pytest smoke suite (10 tests);
  Streamlit dashboard built (6 pages, wired live to the dga package), smoke-tested headless.

## 2026-07-03 — Week 3 · paper reframe applied + GitHub
- **Did:** applied the reframe to all paper sections (abstract -> conclusion), new figures
  (representation_ablation, clr_pca_map), verified citations added, supervisor briefing written
  (docs/supervisor_briefing_2026-06-30.md). Pushed to github.com/Thibaut0000/DGA_KMUTNB_2026_ThibautFCX
  (private); anonymised public extract (data/public/dga_public.parquet) + dashboard fallback so the
  app runs without the confidential raw file.
- **Supervisor sign-off obtained on: C1 reframe, title, Thai table, provenance option B.**

## 2026-07-07/08 — Week 4 · consolidation
- **Did:** dashboard UX redesign (scoring profiles replace sidebar sliders; fault-map page now uses
  the deployed CLR-PCA, not the AE); data-provenance statement inserted (S3, option B: anonymised,
  multi-source, extract released); figures consolidated into two figure* panels -> back to 6 pages;
  defense decks built (HTML self-contained + native PPTX, 17 slides each).
- **Found:** adding content repeatedly pushed the paper to 7 pages; reclaimed via panel figures,
  one EDA figure dropped, conclusion trimmed. 6 pages, 0 unresolved refs.

## 2026-07-16 — Week 5 · cross-deliverable coherence audit (pre-defense)
- **Did:** full coherence check across paper, final report, decks, lab notebook, README, defense-prep
  notes, and the GitHub repository.
- **Found:** final report + defense-prep notes still told the pre-reframe story (ARI 0.47 headline,
  no surveillance-bias contribution, no 0.76 floor; prep notes even coached "AE beats PCA" — the
  opposite of our finding); deck timeline had the discovery in week 3 (actually week 2); lab notebook
  stopped at 24 Jun. All fixed today; decks rebuilt; repository synchronised.
- **Implies:** all deliverables now tell the same (reframed) story with the same numbers:
  ARI 0.16 -> 0.55 (AE 0.47 = negative); AUC 0.74 [0.68,0.80] vs count 0.76 (p=0.58); 17 onset
  units, CI [0.52,0.77]; Thai 69%, +6 units, 0.734 vs 0.735.
