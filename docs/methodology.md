# Methodology decisions (rationale log)

Record every methodological choice and *why*. When Claude or your supervisor asks
"why did you do X?", the answer lives here. Newest at the top.

| Date | Decision | Rationale | Alternatives considered |
|------|----------|-----------|-------------------------|
| 2026-06-15 | Features = H2, CH4, C2H2, C2H4, C2H6, CO, CO2 | the diagnostic fault gases | including O2/N2 (atmospheric), TCG (sum → leakage) |
| 2026-06-15 | log1p + standardise | DGA ppm are heavily right-skewed; log stabilises variance | robust/minmax scaling, no transform |
| 2026-06-15 | Clip per-column at q=0.999 | remove gross data-entry errors (e.g. O2 ≈ 7e8) | winsorise, drop rows |
| 2026-06-15 | Impute missing gas as 0 | ≈ below detection limit | median imputation, drop |

## Open methodological questions
- Does the AE latent encode fault **type** or overall **severity**? (first baseline suggests severity)
- Per-sample normalisation (gas proportions, à la Duval ratios) to factor out magnitude?
- Best latent dimension (2–5) by downstream cluster/anomaly usefulness, not reconstruction MSE alone.
- Validation strategy given only weak (note-based) labels.

## Positioning vs the 2026 EPSR paper (sent by the supervisor)
*"Detecting Incipient Faults in Power Transformers through Hybrid Model of DGA and ML"*,
Electric Power Systems Research, 2026 (`hybrid_aiscore2026`).
- **What they do:** fuse the outputs of conventional methods (Duval/Rogers/Doernenburg) into a
  single weighted **AI-Score Index**, then a **supervised ensemble classifier** → 99.17% accuracy
  on a **labelled** benchmark (386 train + 120 IEC TC10 test).
- **How we differ — keep this clear for the defense:** they are **supervised** and need labels;
  ours is **unsupervised** representation learning on ~4,563 **unlabelled** samples. We are *not*
  trying to beat their 99% accuracy (different problem) — our goal is **label-free pattern
  discovery + anomaly detection**. Cite them as the supervised state of the art and as evidence
  that conventional methods alone are insufficient ("out-of-code" cases).
- **Idea worth borrowing:** they use conventional-method outputs as **features**. We already use
  Duval as an **evaluation** label; we could also test feeding Duval/IEC ratios as extra inputs.
