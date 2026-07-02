# Wang et al. 2016 — Continuous Sparse Autoencoder (CSAE) on DGA

- **Citation:** L. Wang, X. Zhao, J. Pei, G. Tang, "Transformer fault diagnosis using continuous
  sparse autoencoder," *SpringerPlus*, vol. 5, art. 448, 2016. DOI 10.1186/s40064-016-2107-7.
  `bib: wang2016csae` · PDF `docs/refs/wang2016csae.pdf` (open access).
- **Type:** primary method paper; **representation learning** (semi-supervised). **Closest prior art.**

## TL;DR
Uses an autoencoder for **unsupervised feature learning on DGA**, then a supervised head for the
final diagnosis. It is the nearest neighbour to our idea — which lets us state precisely how we
differ: we keep the whole pipeline **label-free** and work on real operational data at scale.

## Problem
Single-layer neural nets learn weak DGA features; the paper seeks richer learned features for fault
recognition.

## Method
- **CSAE** = an autoencoder with a **Gaussian stochastic unit** added to the activation
  (manifold-learning flavour, reduces overfitting).
- Stacked into a **Deep Belief Network: 2 CSAE layers (unsupervised pre-training) + 1
  back-propagation (BP) layer (supervised classification)**.
- **Input = 3 IEC ratios** (CH4/H2, C2H2/C2H4, C2H4/C2H6), min–max scaled to [−1, 1] —
  i.e. magnitude is factored out *before* the network sees the data.

## Data & results
- **IEC TC10, only 134 samples** (125 train / 9 test, 5-fold CV), **5 classes** (PD, low-energy
  discharge, high-energy discharge, thermal <700 °C, thermal >700 °C).
- **CSAE 93.6 %** vs BP 84.1 % (p = 0.0195), SVM-RBF 79.9 %, KNN 90 %.

## Limitations
- The AE is only a **pre-training feature extractor**; the **diagnosis is supervised** (the BP head
  needs labels) → **not label-free end-to-end**.
- Tiny, curated benchmark (134 samples); uses ratios, not raw concentrations.

## Relevance to our project (cite & contrast — the key prior art)
- Confirms autoencoders are a sensible representation for DGA (supports our approach).
- **Our differentiation, made explicit:** (1) **fully unsupervised downstream** — clustering +
  anomaly detection, no supervised head; (2) **4,563 real operational samples** vs 134 curated;
  (3) we measure representation usefulness by **cluster/anomaly metrics**, not classification
  accuracy.
- Reinforces the **per-sample-ratio = fault-type** thread (they too feed ratios) → supports the
  Week-2 normalisation experiment.
