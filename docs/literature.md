# Annotated bibliography

One line per paper. Buckets: [C]onventional DGA · [S]upervised ML · [U]nsupervised/representation.
Seeded 2026-06-14 from the supervisor's list (verified) + the unsupervised gap. Full triage and
verification sources: `literature_triage.md`. Add your own takeaway as you read each one.

| # | Bucket | Citation key | Reference | Why it matters | Read |
|---|--------|--------------|-----------|----------------|------|
| 1 | C | `iec60599` | IEC 60599:2022 — interpretation of dissolved & free gases | source of the IEC ratio limits you implement | P1 |
| 2 | C | `ieee_c57104` | IEEE C57.104-2019 — interpretation of generated gases | the foundational DGA guide (concentrations, rates) | P1 |
| 3 | C | `duval2002review` | Duval 2002, IEEE EIM 18(3):8–17 | **Duval Triangle 1** + fault classes for your main-tank data | P1 |
| 4 | C | `duval2008ltc` | Duval 2008, IEEE EIM 24(6):22–29 | LTC / low-temp variant (supervisor's ref; secondary) | P1 (skim) |
| 5 | S | `zhang1996ann` | Zhang et al. 1996, IEEE TPWRD 11(4):1836–1841 | classic ANN-on-DGA; supervised baseline to contrast | P2 |
| 6 | S | `bacha2012svm` | Bacha et al. 2012, EPSR 83(1):73–79 | representative SVM data-driven DGA | P2 |
| 7 | S | `fei2009svmga` | Fei & Zhang 2009, ESWA | SVM+GA; the ESWA "ML techniques" slot | P2 |
| 8 | U | `wang2016csae` | Wang et al. 2016, SpringerPlus 5:448 | **closest prior art** — unsupervised autoencoder on DGA | P1 |
| 9 | U | `dga_dbscan2020` | DGA + DBSCAN 2020, Appl. Sci. 10(13):4440 | **unsupervised clustering** of DGA (your Week-3 step) | P1 |
| 10 | U | `idae2022` | IDAE 2022, EPSR | autoencoder architecture on DGA (imputation) | P3 |
| 11 | U | `kingma2014vae` | Kingma & Welling 2014, ICLR | foundational VAE (`models/vae.py`) | P3 |
| 12 | U | `hinton2006reducing` | Hinton & Salakhutdinov 2006, Science 313:504–507 | foundational autoencoder | P3 |
| 13 | S | `hybrid_aiscore2026` | **Hybrid DGA + ensemble ML, EPSR 2026** (sent by tutrice) | recent supervised SOTA, same journal; benchmark to cite/contrast (it needs labels, you don't) | **P1** |
| 14 | C | `duval2001interp` | Duval & dePablo 2001, IEEE EIM 17(2):31–41 | the IEC 60599 + IEC TC10 database interpretation paper (basis of Triangle 1) | P2 |
| 15 | U | `mirowski2012review` | Mirowski & LeCun 2012, IEEE TPWRD 27(4):1791–1799 | review of statistical ML on DGA; good Related-Work anchor | P2 |
| 16 | Rev | `ramarao2026review` | **DGA→AI review 2026, EAAI 181:115390** (sent by tutrice) | 34-pg survey traditional→AI/digital-twin; **Related-Work backbone** + accuracy ranges per family | **P1** |

> **Exhaustive per-paper notes now live in `docs/refs/notes/`** (one standalone `.md` each, more
> detailed than this index) — see `docs/refs/notes/README.md`.

**Target:** 15–20 quality entries by end of Week 1. Now at 16 in `references.bib`. Optional extras
to reach 18–20 (links in `docs/refs/README.md`; drop the PDFs and I will add reading notes):
- a recent DGA + AI review (arXiv:2304.11880, 2023) — to verify and cite in Related Work;
- a DGA review survey (e.g. *Energies* 11(4):913, 2018) — broad context.

Verify every entry's volume/pages against the publisher page before the paper goes out.
Entries 14 and 15 were taken from the reference lists of papers we hold (not from memory).

---

## P1 reading notes (read 2026-06-16; numbers taken from the PDFs in `docs/refs/`)

> **Cross-cutting thread (the single most useful finding for my method):** every method that
> recovers fault *type* factors out magnitude by working on **per-sample gas proportions/ratios** —
> Duval (% of CH4/C2H4/C2H2), Wang (IEC ratios CH4/H2, C2H2/C2H4, C2H4/C2H6), Liu (% of the 5
> gases, Eq. 15). My AE+KMeans baseline encodes *severity* (high silhouette, ARI≈0.015) precisely
> because it sees raw/log concentrations. → **strong prior-art support for the Week-2 per-sample
> normalisation experiment.** Second thread: acetylene is sparse but discriminative (my EDA: C2H2
> 97.4 % zero); Liu's amplification coefficient is a published fix for exactly this drowning-out.

### `duval2002review` — Duval Triangle 1 (IEEE EIM 18(3):8–17, 2002) matches bib
- **Problem:** turn IEC 60599's fault codes into a single user-friendly graphical interpretation
  for main-tank faults; review which faults are actually detectable by gas-in-oil.
- **Method:** Triangle 1 in **% of three gases only — %C2H2, %C2H4, %CH4** (ratios → magnitude is
  removed by construction). Seven zones: PD, D1, D2, T1, T2, T3 + **DT** (mixed thermal/electrical).
  Built from **179 inspected in-service faults + lab cases**. Notes: T3→hot spots in *oil*, T1/T2→
  *paper*; CO2/CO < 3 *suggests* paper involvement but "use with caution"; PDs often below DGA
  detection limit (~0.1 ppm), so PD may be missed.
- **Data:** IEC TC10 inspected-fault databases + laboratory simulations (ppm tables in-paper).
- **For my approach:** this *defines* the external label my pipeline is scored against
  (`conventional.py` Triangle 1). Crucially it confirms **why fault-type lives in proportions, not
  magnitude** — the core motivation for normalising per sample. Also: D1/D2/DT are intrinsically
  rare (matches my 70/4563 arcing cases) → ARI vs Duval will under-reward type structure; report
  internal metrics + per-cluster Duval cross-tab alongside ARI.

### `wang2016csae` — Continuous sparse autoencoder on DGA (SpringerPlus 5:448, 2016) 
- **Problem:** learn better DGA features than single-layer NNs for fault recognition.
- **Method:** **CSAE** = autoencoder with a Gaussian stochastic unit in the activation (manifold-
  learning flavour, anti-overfit). Stacked as a DBN: **2 CSAE layers (unsupervised pre-training) +
  1 BP layer (supervised classification)**. Input = **3 IEC ratios** (CH4/H2, C2H2/C2H4, C2H4/C2H6),
  min-max scaled to [−1,1].
- **Data:** IEC TC10, **only 134 samples** (125 train / 9 test, 5-fold CV), 5 classes (PD, LED, HED,
  TF1<700 °C, TF2>700 °C). Results: CSAE **93.6 %** vs BP 84.1 % (p=0.0195), SVM-RBF 79.9 %, KNN 90 %.
- **For my approach (closest prior art — cite & contrast):** the AE is only a **pre-training feature
  extractor; the actual diagnosis is supervised (BP needs labels)** → *not* label-free end-to-end.
  It also feeds **ratios**, not concentrations (again: magnitude factored out). My differentiation:
  (1) fully unsupervised downstream (clustering + anomaly, no BP head), (2) 4,563 real operational
  samples vs 134 curated, (3) I quantify representation usefulness by cluster/anomaly metrics, not
  classification accuracy.

### `dga_dbscan2020` — Correlation-coefficient DBSCAN (Appl. Sci. 10(13):4440, 2020) authors verified
- **Problem:** plain DBSCAN fails on DGA because fault classes aren't separable in Euclidean space
  (sparse discriminative gases get drowned).
- **Method:** **CCDBSCAN** = (1) convert each sample to **% content** (Eq. 15, per-sample
  normalisation); (2) **amplification coefficients** on weak-but-characteristic gases (esp. C2H2),
  tuned by a chaos-sequence optimiser to maximise inter-fault separation; (3) use the **correlation
  coefficient** (not Euclidean distance) as the DBSCAN neighbourhood metric.
- **Data:** 5 gases (H2, CH4, C2H6, C2H4, C2H2). From >2000 collected, **filtered to 60 balanced
  samples (10×6 faults)** for clustering + 30 for diagnosis. Accuracy: plain DBSCAN 58.3 % →
  CCDBSCAN **90 %** (+31 %); on the 30 cases, 26/30 correct vs IEC 60599's 20/30.
- **For my approach:** the **direct Week-3 comparison point** (unsupervised clustering on DGA). But
  it's only quasi-unsupervised — the **amplification weights and cluster→fault mapping are tuned
  using known labels** on a tiny, *balanced, curated* set. My differentiation: genuinely label-free
  on the full imbalanced 4,563-sample data; no per-fault tuning. Their amplification idea is a
  candidate fix for my C2H2 sparsity (feature weighting), worth an ablation.

### `hybrid_aiscore2026` — Saleh et al., Hybrid DGA + ensemble ML (EPSR 256:112877, 2026) supervisor's paper
- **Problem:** conventional methods (DTM/RRM/DRM/IRM) disagree or give "out-of-code" non-diagnoses,
  undermining confidence.
- **Method:** **Hybrid AI-Score Index** — run the **four** conventional methods, map each to a
  unified scheme, fuse their outputs into **one weighted score feature** (accuracy-based weights +
  a scoring rule that penalises invalid/out-of-code outputs), then feed that to **ensemble ML**
  (RF, AdaBoost, Gradient Boosting, KNN, DT, BPNN). **SMOTE + class weights** for imbalance.
- **Data:** **506 samples = 386 train** (Egypt Holding-Company lab + literature) **+ 120 IEC TC10
  test**, reduced to **4 classes** (Normal, PD, Arcing=D1+D2, OH=T1+T2+T3). Results: Hybrid Index
  **99.17 % acc, MCC 98.76 %, 1/120 misclassified**, vs DTM 88.33 % / RRM 83.33 % / DRM 81.67 %, and
  beating recent intelligent frameworks (87.7–98.53 %).
- **For my approach (the positioning paper):** this is **fully supervised** and, tellingly, **takes
  the conventional methods as *inputs*** — it *fuses expert rules*, the opposite of my data-driven,
  rule-free, **label-free** stance. **Do not compare their 99 % accuracy to my ARI** (different
  paradigm: balanced 4-class supervised w/ SMOTE vs unsupervised discovery on imbalanced 7-class).
  Same journal + same "incipient fault" framing → ideal contrast: they need labels and rules, I
  discover structure without either; report it as the supervised ceiling, not a head-to-head.

### `ieee_c57104` — IEEE C57.104-2019 (standard) — skimmed for the key tables
- **Problem / scope:** how to decide whether a transformer's gas levels are normal, from a large
  statistical population (>1M lab DGA results), not fixed legacy limits.
- **Key tables / norms:** **3 DGA Status levels** (1 probably normal · 2 suspicious · 3 serious),
  down from the 4 "conditions" of the 2008/1991 editions. **Table 1 = 90th-percentile gas levels,
  Table 2 = 95th-percentile**; **Table 3 = 95th-pct of deltas** between consecutive samples;
  **Table 4 = 95th-pct of multi-point generation *rates* (ppm/year)**. Norms are **stratified by
  transformer age and O2/N2 ratio** (sealed vs free-breathing). 2019 **embeds the Duval Triangle &
  Pentagon** (Annex D), **removes Doernenburg/key-gas from the main text**, drops TDCG/TCG
  interpretation, and adds Normalized Energy Intensity (NEI, Annex F).
- **For my approach:** rate-based, per-gas, population-percentile thinking — a useful framing for the
  **anomaly-detection** step (a reconstruction-error threshold is my data-driven analogue of their
  90th/95th-percentile "status"). Note their norms depend on **age & O2/N2**, which I currently drop;
  worth a sentence acknowledging that "normal" is condition-dependent. Cite-only for thresholds.

### `iec60599` — IEC 60599:2022 — **cite-only** (paid standard, not acquired)
- Source of the IEC ratio limits implemented in `conventional.py`. Its thresholds are reproduced in
  the papers we do hold (Saleh 2026 Tables 1–3, the Duval articles). No PDF; \todo{} verify any
  directly-quoted limit against a held source before it enters the paper.

### `ramarao2026review` — DGA→AI review (EAAI 181:115390, 2026) sent by supervisor
- **Type:** 34-page review/survey, not a primary method — the **Related-Work backbone**.
- **Taxonomy + reported accuracy:** heuristics (Key Gas/Duval) ~60–75 % (<1000 samples); classical ML
  (SVM/RF) ~85–95 % (300–1000, imbalance + interpretability issues); DL (GAN/CNN/LSTM) >97 % (≥5000,
  but more mixed-fault errors, opaque, costly); hybrid (neuro-fuzzy/Bayesian) 95–98 % (600–1500);
  digital twins → RUL ±10–15 %. Roadmap: hybrid XAI, low-cost online sensors, federated learning,
  cybersecurity.
- **For my approach:** structure §2 along these families; the review is dominated by *supervised*
  classification and names *prognostics/temporal/RUL* as the frontier → our label-free, self-supervised,
  temporal fleet-ranking sits in that under-served, forward-looking space ("to the best of our
  knowledge"). Use its ranges to justify *not* chasing a 97–99 % accuracy number. Cite for landscape,
  not primary results; verify any quoted figure. Full note: `docs/refs/notes/ramarao2026review.md`.
