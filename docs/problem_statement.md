# Problem statement, research question & contributions

> **Realigned 2026-06-18 to the supervisor's brief** (fleet health ranking, self-supervised,
> temporal). Thibaut owns the scientific claims. Established facts carry numbers from `results/`;
> items still to test are flagged. The earlier SD-CAE work is kept as a component, not the headline.

## Context

Power transformers are critical, expensive assets whose in-service failure is costly. Dissolved Gas
Analysis (DGA) detects incipient faults early, but conventional interpretation (IEEE C57.104, IEC
60599 ratios, Rogers, Duval) is rule-based, needs an expert, and leaves cases "not determined".
Supervised ML needs labels utilities rarely have. In practice a utility must **prioritise maintenance
across a whole fleet** — rank units by health/risk — without labels. Our dataset supports this: it is
**longitudinal** — 628 transformers, 4,563 samples, mean 7.3 per unit (median 6; 98 % have >= 2,
88 % >= 5), spanning a median ~4.5 years at a ~6-month sampling interval — enough to learn temporal
trends, not just snapshots.

## Research question

> **Can we rank transformers across a fleet by incipient-fault risk, label-free, by learning a
> self-supervised, temporal representation of each unit's DGA history and scoring severity and
> deviation (acetylene-aware) — and how does this ranking compare with conventional DGA
> interpretation (IEEE C57.104 / IEC 60599 / Rogers / Duval)?**

## Novelty (the four pillars of the brief)

- **Self-supervised learning** (no fault labels).
- **Temporal representation learning** (per-unit DGA trajectories, not isolated samples).
- **Unlabeled DGA**.
- **Fleet-level risk ranking** (a decision-support output, not a per-sample class).

## Method overview (pipeline)

1. **Feature engineering.** Per-sample gas levels + **TCG** (total combustible gas, a standard health
   indicator — now included) + diagnostic ratios; and **temporal features per transformer**: gas
   growth rates, trend slopes, latest-vs-baseline deltas, time since last rise.
2. **Self-supervised representation.** (i) per-sample autoencoder; (ii) **SD-CAE** compositional latent
   for the fault-*type* axis (CLR of gas proportions, severity-invariant by construction); (iii) a
   **temporal representation** of each unit's trajectory (engineered growth-rate features first; a
   sequence model as a stretch goal).
3. **Risk / health scoring (acetylene-aware).** Combine severity (TCG, levels vs C57.104 90/95th
   percentiles), deviation/novelty (reconstruction error, Isolation Forest, Deep SVDD), and an
   **acetylene weighting** (C2H2 = arcing = the most dangerous fault) into a Health/Risk Index, then
   **rank the fleet**.
4. **Fault-type explanation.** The SD-CAE compositional map says *which* fault type drives a unit's
   risk — the interpretable "why" behind its rank.
5. **Validation.** Do the top-ranked (riskiest) units carry more field-event notes than the fleet?
   Compare the ranking and flags with the conventional methods, and show units the rules miss.

## How the earlier SD-CAE work is reused (nothing wasted)

- The W1 finding "a plain AE encodes gassing *severity* (R^2 ~ 0.61)" is exactly the **health-relevant
  axis** — the magnitude/TCG signal is what a risk index should rank on.
- **SD-CAE separates type from severity**: the *severity* part feeds the health ranking; the *type*
  part (CLR latent) provides the interpretable explanation.
- The **acetylene** insight (C2H2 sparse but = arcing) directly motivates the acetylene-aware risk model.

## Hypotheses

- **H1 (established).** A concentration-based AE latent encodes gassing *severity* (R^2 ~ 0.61) — useful
  as the health/risk axis; magnitude/TCG are the ranking signal.
- **H2 (supported).** A CLR compositional representation recovers fault *type* (ARI vs Duval 0.47 +/-
  0.06 over 5 seeds, vs 0.14 for raw concentrations) — the interpretable axis. The adversarial
  independence term does not help (reported ablation), so the type-invariance comes from the CLR
  construction.
- **H3 (supported).** A label-free risk score (H2-led severity + acetylene-aware + temporal +
  self-supervised anomaly) ranks fault-note units above the fleet: **AUC 0.740** vs `fault_note` (vs
  0.521 for a TCG-only baseline), prec@10 = 50 % (x4.4), monotonic decile note-rate (D1 27 % -> D10 2 %).
  It matches the supervised CV upper bound, supporting the label-free claim. Acetylene is the dominant
  lever (validates the acetylene-aware model). NOTE: must use *physical gas* features only -
  `n_samples`/`span_years` are confounds (faulted units get re-sampled).
- **H4 (partly supported).** The **H2 generation rate** (not the TCG rate) adds signal: it lifts the
  index AUC (0.714 snapshot -> 0.733) and helps the riskiest decile; the gain is modest and needs >=3
  samples for a reliable C57.104 rate.

## Contributions (reframed 2026-06-24 to lead with the confound-free results)

A deep research campaign (5 adversarially-verified probes) reshaped the emphasis: the `fault_note`
label is surveillance-confounded, so we lead with the contributions whose targets are confound-free,
and report the risk ranking honestly with the confound floor shown.

1. **C1 (primary) — SD-CAE, a label-free severity-disentangled compositional representation of DGA.**
   It separates fault *type* from gassing *severity* by construction (CLR of gas proportions), recovers
   Duval fault-type structure (**ARI 0.47 +/- 0.06 vs 0.14 for raw concentrations**, 5 seeds), and the
   adversarial-independence term is a reported negative ablation. The target (Duval) is a deterministic
   rule, hence **confound-free** -- the most defensible result.
2. **C2 (primary, methodological) — DGA operator-note labels are surveillance-confounded.** We show
   that a trivial sampling-count (`n_samples`) out-predicts gas chemistry (**AUC 0.76 > the 0.74
   index**) and that a label-free risk model has little forward validity once `n_samples` is controlled
   (temporal hold-out 0.578 -> 0.514). This is a caution for the (label-hungry) DGA-ML literature, with
   a confound-free chemistry-target evaluation (arcing-onset AUC ~0.63) as the honest alternative.
3. **C3 (secondary, applied) — a label-free Health/Risk screening index** (severity + acetylene-aware +
   temporal + anomaly) that **beats conventional severity baselines** (0.74 vs 0.52-0.56 for
   TCG/C57.104/Duval), reported *with* the `n_samples`=0.76 confound floor and the forward-validity
   caveat. Anomaly-method and multi-seed robustness comparisons included.

## Scope & non-goals

Main-tank DGA only; offline decision-support ranking (not real-time monitoring). Field-event notes are
weak, non-exhaustive positives — validation is indicative. Temporal modelling starts from engineered
growth-rate features; deep sequence models are a stretch goal. We do not propose new mathematics for
CLR or autoencoders — the novelty is the label-free, temporal, fleet-ranking *application*.

## How each claim is evidenced (evaluation plan)

| Claim | Test | Status |
|------|------|--------|
| H1 severity axis | R²(severity~latent); magnitude/TCG distributions | done (W1) |
| H2 type recovery (SD-CAE) | ablation raw → proportions → CLR → adversary, 5 seeds; ARI/NMI vs Duval | done: CLR 0.47 vs raw 0.14; adversary hurts |
| H3 risk ranking vs notes | note rate by rank decile; precision@k; vs TCG-only | done: AUC 0.740 vs 0.521 TCG; prec@10 50%; monotonic deciles |
| H4 temporal vs snapshot | ranking quality with/without growth-rate features | done: H2-rate adds (0.714 -> 0.733); needs >=3 samples |
| anomaly methods | AE recon vs Isolation Forest vs Deep SVDD | done: IForest 0.60 > AE 0.58 > Deep SVDD 0.52 (collapses); deploy AE+IForest |
| C3 comparison | index vs conventional severity (TCG/C57.104/Duval); units the rules miss | done: index 0.740 vs conv. 0.52-0.56; flags 3 units rules miss |
| index robustness | multi-seed AUC stability | done: 0.741 +/- 0.001 over 8 seeds |

## Risks / credibility guardrails

- **Temporal signal: confirmed viable** (98 % of units have >= 2 samples, median 6, ~6-month interval,
  ~4.5-year span) — temporal features are well supported, not a stretch.
- **No ground-truth health label.** We validate the ranking indirectly (field-event notes + agreement
  with C57.104 severity) and present it as decision-support, not absolute accuracy.
- **Weak, sparse labels** (391 notes, 8.6 %): ranking validation is indicative; report note rate by
  decile and precision@k, not a single accuracy number.

See `paper/sections/` for the prose drafts (being realigned to this headline).
