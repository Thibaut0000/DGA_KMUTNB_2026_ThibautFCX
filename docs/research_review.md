# Research review and positioning (2026-06-24)

A deep-research pass to re-situate the project in the current state of the art (2024-2026) and to
test concrete new ideas it surfaced. Sources: the Ramarao et al. 2026 review (EAAI 181:115390, in
`docs/refs/`), plus web search of recent literature (links at the bottom).

## 1. State of the art (where the field is)

From the 2026 review and recent papers, DGA diagnostics fall into families with a clear trend:
- **Traditional heuristics** (Key Gas, Duval Triangle/Pentagon, IEC/Rogers ratios): ~60-75 % accuracy,
  expert-dependent, "not determined" cases.
- **Classical ML** (SVM, RF): ~85-95 % on ~300-1000 labelled samples; recurring issues = class
  imbalance and interpretability.
- **Deep learning** (CNN, LSTM, GAN): >97 % on >=5000 labelled samples, but opaque and data-hungry.
- **Hybrid / physics- or rule-informed AI**: ~95-98 %, more interpretable.
- **Frontier named by the review:** explainable AI, online sensors, **federated learning**,
  **digital twins**, and **prognostics / Remaining-Useful-Life** -- i.e. moving DGA "from a
  diagnostic tool to a predictive engine".

**Two facts that matter for us.** (a) The field is overwhelmingly **supervised classification**;
**unsupervised / label-free / anomaly** methods are thinly covered. (b) The hot "Health Index" (HI)
literature (2025-2026: LightGBM/CatBoost, SHAP, SMOTE) **predicts an expert-defined HI** -- it needs a
HI label. Even the closest *unsupervised* work (HI via clustering, 2024-2025) still targets a known HI
grade scale, not field events.

## 2. Where this project sits (relevance and novelty)

Our **label-free, self-supervised, temporal, fleet-level risk ranking, validated against real
field-event notes**, sits precisely in the under-served, forward-looking space (label-free + temporal
+ prognostic). To the best of our knowledge no prior work ranks a transformer fleet by incipient-fault
risk without any labels and validates it on recorded field events. This is a credible
"to-the-best-of-our-knowledge" novelty -- modest but real, and aligned with the review's stated frontier.
It also reframes our honesty about accuracy: we **do not chase 97-99 %** (those are supervised,
balanced, curated <5000-sample sets); we report ranking/agreement on 4,563 real, imbalanced, unlabelled
samples.

## 3. New ideas investigated and TESTED this pass

1. **Duval Pentagon (Duval & Lamarre 2014, 5 gases).** Newer than Triangle 1; the literature says it is
   better at PD and at **separating stray gassing** from real faults. We implemented the verified
   Pentagon *coordinate* math (axes H2 90 deg, C2H6 162, CH4 234, C2H4 306, C2H2 18; polygon centroid)
   and sanity-checked it (pure gases land on their axes). We did **not** hard-code the fault-zone
   polygons (could not source exact published coordinates safely -- would violate the "never invent"
   rule); the zone classification is left for when verified coordinates are available.
2. **Stray-gassing / "PD inflation" hypothesis (from Week 1).** Tested whether the Triangle-1 PD set is
   benign stray gassing. **Result tempers the Week-1 claim:** the PD inflation is a per-*sample*
   artifact; at the per-*unit* (latest-state) level only **97 units** are PD and they are **not benign**
   (fault rate 15.5 % vs fleet 11.3 %). So stray gassing is a minor issue for the *ranking* task, and
   the strong Week-1 wording should be softened.
3. **Depth of the temporal representation (the brief's "temporal representation learning" pillar).**
   Tested richer per-unit trajectory features (volatility, monotonic trend, peak gap, rise count) and
   the supervised ceiling. **Findings:** (a) physical trajectory-shape features (~0.59 AUC) do **not**
   beat the simple C57.104 **H2 generation rate** (0.62) we already use; (b) count-based features
   (`nrises`) look strong (0.76) but are the **sampling confound** in disguise (corr 0.87 with
   `n_samples`) -- excluded; (c) the weak `fault_note` label caps achievable AUC near **0.74**, so a
   heavier learned sequence model (GRU/LSTM autoencoder) would hit the same ceiling and is **not worth
   the complexity** for this validation target. This justifies the simple, robust engineered-rate design.

## 4. Implications

- **Keep the method simple and defensible:** engineered C57.104 generation rate + severity + acetylene
  + anomaly. Do not add a heavy temporal deep model just to satisfy the title; frame the contribution
  as *evaluating* temporal representations and identifying what actually helps for label-free ranking.
- **Soften the Week-1 stray-gassing/PD-inflation wording** in the docs (it is a per-sample artifact).
- **Related Work** can be structured along the review's families, ending on the label-free/temporal gap.
- **Future work (from the frontier):** prognostics/RUL, contrastive self-supervised temporal models,
  digital twins, federated learning -- connect our prototype to the field's direction.

## 5. Prognostic / RUL frontier — tested rigorously (verdict: weak, not this dataset)

We dug into the prognostic/early-warning frontier (predict a *future* event from the past DGA
trajectory). Honest, multi-method verdict:
- **Feasibility OK:** 86 % of events have prior DGA history; gases rise before some events (TCG +0.18
  log, 72 % rising) but **acetylene does not ramp** (arcing is sudden).
- **Per-sample 1-yr classification:** weak (AUC ~0.57-0.60 from H2 level/rate); top-2 % screen gave a
  x4.1 lift with ~5-month lead -- appealing, but see the confound below.
- **Survival analysis (time-varying Cox, the proper RUL frame):** **no significant signal** -- hazard
  ratios ~1.0 for logH2/logTCG/logC2H2/rateH2, all **p > 0.4**, **C-index 0.565**. Events are largely
  sudden and not preceded by significant, predictable gas trends.
- **Learned temporal model (GRU, group-CV):** looked better (0.638 vs 0.574 simple) **but it is the
  sampling confound**: sequence length alone predicts future events at 0.623, and simple+length (0.651)
  ~ GRU (0.638). Heavily-monitored units (long history) are more event-prone = a **surveillance bias**
  in the label, not a real DGA temporal signal.
- **Conclusion:** prognostics/RUL is **not well-supported by this dataset** (sudden events, weak
  precursors, surveillance-bias confound). This is a valuable honest negative; the surveillance bias is
  itself a useful methodological caveat for DGA prognostics. -> Keep the **current-risk ranking** (AUC
  0.74, robust, beats conventional) as the headline; report prognostics as a rigorously-tested
  limitation / future work (it would need event data without sampling bias, or larger fleets).

## 6. Multi-agent research campaign (2026-06-24) — the label is the real problem

Ran 5 parallel research probes + adversarial confound verification (Workflow). **All 5 negative,
all confound-free**, which is itself a rigorous result:
- **PyOD battery (9 detectors):** none beats the index; IForest best standalone (0.60); adding any
  detector does not exceed 0.737. Confound-free.
- **Self-supervised forecast-residual anomaly:** clean (decorrelated from n_samples and gas level) but
  near-chance (0.53); the only "good" variant (max-residual 0.638) was the n_samples confound.
- **VAE + contrastive representations:** both at chance for anomaly (0.50-0.56), ARI vs Duval ~0.09 -
  no better than the AE; per-sample recon/embedding anomalies do not encode unit-level event risk.
- **SD-CAE type latent added to the index:** robustly DEGRADES it (-0.023, every seed) - type is not
  severity, so it adds noise for risk ranking. (Confirms SD-CAE belongs to the *type* story, not risk.)
- **Label ceiling + temporal hold-out (the pivotal probe):** **the `fault_note` label is
  surveillance-confounded.** `n_samples` alone scores **AUC 0.76 > the 0.74 index**; on a strict
  temporal hold-out the risk score's forward AUC (0.578) collapses to **0.514 (chance) once n_samples is
  controlled**, while `n_samples`-in-first-half forward-predicts better (0.688). Mechanism:
  corr(future_event, n_future)=0.30-0.35, corr(n_first,n_future)=0.595 - operators sample worrisome
  units more, so the label encodes *attention*, not gas chemistry. Plus ~23% of notes are Thai-only
  (16 missed positive units).

**The hard, honest conclusion:** validating a gas-based risk model against operator-logged notes is
**fundamentally confounded** - sampling frequency (operator attention) out-predicts the physics, and
the index has little forward validity beyond it. This is a genuine methodological finding (a warning
for the DGA-ML literature, which routinely uses such labels), but it undermines the `fault_note`-based
risk-ranking *headline*.

**The one clean, confound-free signal found:** with a **chemistry-defined forward target** (does
acetylene/arcing *appear* in the next 2 years, given none now) - which is NOT surveillance-biased
(corr with n_samples ~ 0) - the trajectory is weakly predictive: **C2H4 (thermal) and H2 (incipient)
forecast arcing onset at AUC ~0.62-0.64**. Weak (45 positives) but real and survives every confound
check. A "future TCG doubling" target is flawed (regression-to-the-mean toward low-current units).

**Implications for the contribution (reshapes the emphasis):**
- The **SD-CAE fault-TYPE result (ARI 0.47 vs Duval)** is the more *defensible* contribution: Duval is a
  deterministic rule, not an attention-driven label, so that target is confound-free.
- The **risk-ranking** must be reported with the surveillance-bias caveat front and centre
  (co-report n_samples=0.76), not as a clean predictive claim.
- The **surveillance-bias demonstration** is itself a publishable methodological contribution.
- **Confound-free chemistry targets** (arcing onset) are the honest path for any prognostic claim, and
  a **better label** (Thai-note translation / real failure records from KMUTNB) is needed for strength.

## Sources
- Ramarao et al. 2026, EAAI 181:115390 (review; `docs/refs/ramarao2026review.pdf`).
- HI via interpretable ensembles 2026: https://doi.org/10.3390/technologies14010006
- HI prediction with SMOTE 2025: https://www.mdpi.com/1996-1073/18/9/2364
- HI via clustering (unsupervised) : https://www.researchgate.net/publication/379090928
- DGA ML systematic review 2025: https://www.mdpi.com/2076-3417/15/5/2395
- Benchmarking ML/DL for transformer fault detection 2025: https://arxiv.org/pdf/2505.06295
- Self-supervised transformer for multivariate time-series anomaly 2025: https://link.springer.com/article/10.1007/s10489-025-06481-7
- Duval Pentagon (combined/simplified): https://www.mdpi.com/1996-1073/13/11/2859 ; original: Duval & Lamarre 2014, IEEE EIM 30(6).
