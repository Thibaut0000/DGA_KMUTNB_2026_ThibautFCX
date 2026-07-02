# Liu et al. 2020 — Correlation-Coefficient DBSCAN (CC-DBSCAN) on DGA

- **Citation:** Y. Liu, B. Song, L. Wang, J. Gao, R. Xu, "Power Transformer Fault Diagnosis Based on
  Dissolved Gas Analysis by Correlation Coefficient-DBSCAN," *Applied Sciences*, vol. 10, no. 13,
  art. 4440, 2020. (Wuhan University.) `bib: dga_dbscan2020` · PDF `docs/refs/dga_dbscan2020.pdf`
  (open access).
- **Type:** primary method paper; **unsupervised clustering**. **Direct Week-3 comparison point.**

## TL;DR
Makes density clustering work on DGA by (a) converting to per-sample gas percentages,
(b) amplifying the sparse-but-discriminative gases, and (c) clustering with a correlation-coefficient
distance instead of Euclidean. The closest unsupervised-clustering paper to our Week-3 step — but
only **quasi**-unsupervised, which is exactly our differentiator.

## Problem
Plain DBSCAN fails on DGA: fault classes are not well separated in Euclidean space, and the
sparse discriminative gases (notably C2H2) get drowned out by the abundant ones.

## Method (CC-DBSCAN)
1. Convert each sample to **% content** of the 5 gases (per-sample normalisation; their Eq. 15).
2. Apply **amplification coefficients** to weak-but-characteristic gases (esp. C2H2), tuned by a
   **chaos-sequence optimiser** to maximise inter-fault separation.
3. Use the **correlation coefficient** (not Euclidean distance) as the DBSCAN neighbourhood metric.

## Data & results
- **5 gases** (H2, CH4, C2H6, C2H4, C2H2). From >2000 collected samples, **filtered to 60 balanced
  samples (10 × 6 fault types)** for clustering, + 30 for diagnosis.
- **Plain DBSCAN 58.3 % → CC-DBSCAN 90 %** (≈ +31–32.7 %); on the 30 diagnosis cases **26/30 correct
  vs IEC 60599's 20/30**.

## Limitations
- **Quasi-unsupervised:** the amplification weights and the cluster→fault mapping are **tuned using
  known labels** on a tiny, **balanced, curated** set — far from a real fleet.

## Relevance to our project
- The **direct comparison point** for our clustering step; cite as unsupervised-DGA prior art.
- **Our differentiation:** genuinely **label-free** on the **full imbalanced 4,563-sample** data, with
  **no per-fault tuning** and no curated balancing.
- **Reusable idea:** their **amplification of C2H2** is a published remedy for exactly our acetylene
  sparsity (97.4 % zeros) — worth an **ablation** (feature weighting before the AE/clustering).
- Adds to the per-sample-proportion = fault-type thread (their step 1).
