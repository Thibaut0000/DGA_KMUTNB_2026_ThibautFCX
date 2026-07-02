# Saleh et al. 2026 — Hybrid AI-Score Index for incipient-fault detection

- **Citation:** A. G. Saleh, A. E. Azzam, G. Attiya, E. Ibrahim, "Detecting Incipient Faults in Power
  Transformers through Hybrid Model of DGA and Machine Learning," *Electric Power Systems Research*,
  vol. 256, art. 112877, 2026. `bib: hybrid_aiscore2026` · PDF `docs/refs/hybrid_aiscore2026.pdf`.
- **Type:** primary method paper, **supervised**. Sent by the supervisor. **Same journal we target.**

## TL;DR
Fuses the outputs of the four classical interpretation methods into a single weighted "AI-Score"
feature, then trains an ensemble classifier on it — reaching ~99 % accuracy on a balanced 4-class
benchmark. It is the **state-of-the-art contrast paper**: the *opposite* philosophy to ours
(it needs labels and leans on the expert rules; we use neither).

## Problem
Conventional methods — Duval Triangle (DTM), Rogers Ratio (RRM), Doernenburg Ratio (DRM), IEC Ratio
(IRM) — frequently **disagree** or return **"out-of-code"** non-diagnoses, undermining confidence.

## Method
1. Run the **four** conventional methods; map each to a unified fault scheme.
2. Fuse their outputs into **one weighted score index** (weights from each method's accuracy; a
   scoring rule penalises invalid / out-of-code outputs).
3. Feed that score to an **ensemble of ML classifiers** (Random Forest, AdaBoost, Gradient Boosting,
   KNN, Decision Tree, BP neural net).
4. Handle imbalance with **SMOTE + class weights**.

## Data & results
- **506 samples = 386 train** (Holding Company of Electricity, Egypt + literature) **+ 120 IEC TC10
  test**, reduced to **4 classes** (Normal, PD, Arcing = D1+D2, Overheating = T1+T2+T3).
- **Hybrid Index: 99.17 % accuracy, MCC 98.76 %** (1/120 misclassified), vs **DTM 88.33 %,
  RRM 83.33 %, DRM 81.67 %**; reported to beat recent intelligent frameworks (87.7–98.53 %).

## Limitations
- Fully **supervised** and **rule-dependent** (its features *are* the conventional methods).
- Small, **balanced, curated** 4-class benchmark; performance on raw imbalanced operational fleets
  (like ours, 7 classes, 4,563 samples) is untested here.

## Relevance to our project (positioning paper — handle with care)
- **Do NOT** compare their 99 % accuracy to our ARI/AUC — different paradigm (balanced supervised
  4-class with SMOTE vs label-free discovery + ranking on imbalanced 7-class).
- Use it as the **supervised ceiling** and as the clean philosophical contrast: they **fuse expert
  rules and need labels**; we are **data-driven, rule-free, label-free**. Same journal + same
  "incipient fault" framing makes the contrast pointed and citable in §1 and §6.
- Their "out-of-code" motivation is excellent **Introduction** material (quantifies why rules alone
  are insufficient).
