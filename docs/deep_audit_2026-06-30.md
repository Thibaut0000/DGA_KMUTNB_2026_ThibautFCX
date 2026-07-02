# Deep audit and experimental campaign — 2026-06-30

A full-capacity analysis pass: 5-lens multi-agent audit (methodology, statistics,
literature with web verification, engineering, adversarial reviewer) plus five new
adversarial experiments run against our own headline results. Everything below is
reproducible: each number cites its script and output table.

## Headline discoveries (ranked by impact)

### 1. The SD-CAE neural network is unnecessary — the geometry is the contribution
`scripts/run_representation_baselines.py` -> `results/tables/representation_baselines.csv`

| variant | ARI vs Duval (k=7) |
|---|---|
| raw-log 5D + KMeans | 0.160 ± 0.002 |
| proportions 5D + KMeans | 0.119 ± 0.001 |
| CLR 5D (std) + KMeans | 0.491 ± 0.001 |
| **PCA-2 of std-CLR + KMeans** | **0.545 ± 0.002** |
| CLR + GMM | 0.504 ± 0.033 |
| AE-2D on CLR (paper's SD-CAE, ref) | 0.474 ± 0.055 |

A deterministic linear pipeline (CLR -> standardise -> PCA-2 -> KMeans) beats the
autoencoder by 0.07 ARI with 25x less variance. **Consequence:** C1 must be
reframed as "the compositional (log-ratio) geometry recovers fault type; even a
linear projection suffices — the neural encoder adds nothing" (one more honest
negative, consistent with the paper's style). The headline number improves
(0.14 -> 0.545) and becomes deterministic. Bonus: PCA loadings on CLR coordinates
give an interpretable log-ratio biplot (which gas ratios drive each axis).

### 2. Robustness of the CLR result: 2 pass, 1 caveat
`scripts/run_clr_robustness.py` -> `results/tables/clr_robustness.csv`

- Temporal generalisation: fit pre-2022, evaluate 2022+ -> ARI 0.454 (holds).
- Zero-replacement delta: ARI 0.455 (1e-4) / 0.545 (1e-3) / 0.229 (1e-2).
  The qualitative claim survives reasonable deltas but the exact number is
  delta-sensitive (97.4% of C2H2 values are zeros) — must be disclosed; a
  detection-limit-motivated delta is the principled choice to discuss.
- Unit-level (latest sample per unit): ARI 0.346 — weaker than per-sample 0.545
  (heavily-sampled units dominate the per-sample metric); needs the matching
  raw-log per-unit baseline before interpreting (todo).

### 3. The 0.76-vs-0.74 gap is not statistically real — and physics does add signal
`scripts/run_validation_statistics.py` -> `results/tables/validation_statistics.csv`

- Paired bootstrap: AUC index 0.740 [0.678, 0.799]; n_samples 0.761 [0.700, 0.822];
  difference CI [-0.097, +0.055], p = 0.58. The honest claim is "a trivial count
  does *as well as* the physics", not "better".
- Incremental likelihood-ratio tests: physics adds signal beyond n_samples
  (LR = 9.1, p = 0.0025) but n_samples adds far more beyond physics (LR = 51,
  p = 9e-13). Refines C2: contemporaneously the label is *dominated* by
  surveillance but physics is not pure confound; forward (hold-out) the physics
  increment disappears — both are true and the paper can now say so precisely.
- Stratified AUC within n_samples quartiles: 0.82 / 0.72 / 0.57 / 0.76 (small
  event counts per stratum; directionally consistent with the LR test).

### 4. The chemistry target survives the honest (unit-blocked) test — barely
`scripts/run_chemistry_blocked_stats.py` -> `results/tables/chemistry_blocked_stats.csv`

The 45 positive points cluster in only **17 onset units** — the naive permutation
p < 0.01 previously put in the paper was anti-conservative and must be replaced.
Unit-blocked cluster bootstrap: C2H4 AUC 0.643, 95% CI [0.520, 0.770]; H2 0.625
[0.532, 0.729] — both still exclude 0.5; n_samples [0.384, 0.584] (chance).
Sensitivity grid (`chemistry_target_sensitivity.csv`): AUC(C2H4) 0.64–0.73 across
all 9 threshold x horizon variants, n_samples at/below chance everywhere. The
result is robust in direction, modest in power: report "17 onset units" and the
blocked CI, drop the naive p.

### 5. Thai note translation changes the story
`scripts/run_thai_label_extension.py` -> `results/tables/thai_note_classification.csv`
(curated translation table for supervisor validation), `thai_label_extension_effect.csv`

271/391 notes (69%) contain Thai (127 Thai-only) — not the "23%" previously
documented. 15 distinct notes are genuine missed events (~18 units): breaker /
CVT explosions, abnormal noise, high insulation power factor, impedance and
winding-resistance anomalies, OLTC damage, transport impact. Chemistry-triggered
follow-ups ("follow-up because C2H2 found") stay excluded — circular with gas
features. Effect at unit level:

| label | positives | AUC index | AUC n_samples | gap |
|---|---|---|---|---|
| original (EN regex) | 71 | 0.740 | 0.761 | +0.021 |
| +Thai tier A (own-unit electrical) | 77 | 0.735 | 0.734 | 0.000 |
| +Thai A+B (nearby-equipment) | 80 | 0.724 | 0.734 | +0.010 |
| +Thai A+B+C (mechanical) | 83 | 0.714 | 0.742 | +0.028 |

On the repaired (tier-A) label the confound's advantage vanishes: part of the
sample-count edge came from label incompleteness correlated with surveillance.
Tiers B/C (external events, impacts) dilute a gas-based index, as physics predicts.

## Literature verdict (web-verified by the audit)

- **Surveillance bias:** no DGA/power-equipment prior art found — the DGA novelty
  claim holds. But the general phenomenon is established: Sackett 1979
  (diagnostic-suspicion bias), Haut & Pronovost JAMA 2011 (surveillance bias),
  Goldstein et al. Am J Epidemiol 2016 (informed-presence bias — controls for the
  *number of encounters*, the exact analogue of n_samples), Lin/Scharfstein/
  Rosenheck JRSS-B 2004 (outcome-dependent observation), Kapoor & Narayanan
  Patterns 2023 (leakage taxonomy). Reframe C2 as importing/quantifying a known
  confound in a new domain and cite these.
- **Compositional DGA:** no prior CLR/ILR/Aitchison on dissolved gases found —
  C1's log-ratio novelty holds ("to our knowledge"). Landmine to cite: Dukarm,
  Draper & Piotrowski, "Diagnostic Simplexes for Dissolved-Gas Analysis",
  Energies 13(23):6459, 2020 — recognises the simplex but keeps Euclidean
  coordinates; citing it carves our contribution precisely. Add the foundations:
  Aitchison 1982 (JRSS-B 44(2):139–177), Pawlowsky-Glahn et al. 2015 (Wiley).
  (Verify all of these by hand before adding — audit-supplied, not yet in the bib.)

## Repo-integrity findings (must fix)

1. **No git repository.** OneDrive is the only history. `git init` + first commit.
2. **Unscripted paper numbers** (violates our own convention): the 8-seed
   0.74±0.001, the old chemistry CIs/permutation p (scratchpad-computed), the
   "23% Thai" figure. Now largely superseded by the new scripts; port the 8-seed
   run and delete the stale claims.
3. Inconsistencies: `health.py` DEFAULT_WEIGHTS acetylene=1.5 vs 2.0 deployed
   everywhere; docs cite hold-out 0.578->0.514 vs reproduced 0.541->0.501;
   `run_health_comparison.py` reports max(AUC, 1-AUC) (flattering); the
   acetylene-weight sweep is printed but never saved.
4. Method/protocol mismatch: paper says k by silhouette, headline is fixed k=7.
5. Lab notebook records index design iterated against fault_note — the "not tuned
   on the label" claim needs honest qualification (design-time label contact).

## Recommended paper changes (pending owner + supervisor sign-off)

1. Reframe C1: "severity-disentangled compositional representation" achieved by
   CLR geometry; PCA-2 suffices (ARI 0.545); AE reported as a negative ("no
   better than linear"). Update abstract/intro/method/results accordingly.
2. Replace the naive chemistry p-value with unit-blocked CIs and "17 onset units".
3. Soften "out-predicts" to "matches (difference n.s., p=0.58)"; add the LR-test
   nuance (physics adds signal contemporaneously, none forward).
4. Add the Thai-note analysis as a data contribution + the tier-A label result.
5. Add citations: epidemiology/EHR bias set, Dukarm 2020, Aitchison 1982,
   Pawlowsky-Glahn 2015. Adjust novelty sentences per above.
6. Fix protocol wording (k=7), weights consistency, delta-sensitivity disclosure.

## Proposed next actions (ranked, ~3 weeks left)

1. Supervisor meeting: pivot + title + reframe of C1 (now unavoidable), Thai
   table validation. [blocking]
2. Apply the paper changes above (~half a day once approved).
3. `git init`, pin requirements, add a `run_all.py`, 6-10 smoke tests. [half day]
4. Complete the per-unit baseline comparison (raw-log latest-per-unit ARI).
5. Optional new science if time remains: trajectory features in the CLR-PCA map
   (path length / velocity per unit — an honest, confound-checked "temporal
   representation" that reconnects with the supervisor's original brief).
