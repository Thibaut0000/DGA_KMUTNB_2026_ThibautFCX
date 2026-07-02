# Project Plan — Deep Unsupervised Fault Detection in Transformer DGA

> **Direction refined (2026-06-18) to the supervisor's brief: fleet *health/risk ranking*
> (self-supervised + temporal + acetylene-aware), not fault-type clustering.** This file's
> 5-week calendar and process are still the working schedule, but the **current research
> framing, question and contributions live in `docs/problem_statement.md`** — read that first.
> Section 1 below describes the original (clustering-first) framing and is kept for history.

**Student:** Thibaut Faucheux (CESI, FISA4 INFO) · **Host:** KMUTNB, Bangkok ·
**Supervisor:** Dr. Rattanakorn Phadungthin
**Duration:** 5 weeks, 15 June → 17 July 2026 (defense/buffer week of 20 July)
**Final deliverable:** a short IEEE-format paper (~6 pages) + a reproducible code repository + defense slides.

---

## 1. The research in one paragraph

Conventional DGA interpretation (IEC 60599 ratios, Rogers, Duval Triangle) is rule-based, needs an expert, and leaves a share of cases "not determined". This project builds a **label-free** pipeline: an autoencoder (and a VAE) learns a compact latent code for each gas sample, clustering on that latent space discovers fault patterns, and reconstruction error flags incipient faults. We benchmark against the conventional methods using internal cluster metrics and external agreement (ARI/NMI), and sanity-check anomalies against the real field-event notes in the data.

**Dataset (already profiled):** 4,563 main-tank samples, 628 transformers, 42 manufacturers, 115–525 kV, 2019–2024. Gas concentrations are heavily right-skewed with one gross outlier and 391 rows carry a real field-event note (Buchholz trip, bushing explosion…). These notes are weak validation labels.

> **Already done in this scaffold:** the full pipeline runs end-to-end on your real data. A first AE+KMeans baseline gives a high silhouette but low ARI vs Duval — meaning the latent currently separates samples mostly by *overall gassing severity*, not *fault type*. Turning that into fault-type structure is your central research thread (see Week 2–3).

---

## 2. Calendar overview

| Week | Dates (Mon–Fri) | Theme | Study steps | Milestone (Definition of Done) |
|------|-----------------|-------|-------------|--------------------------------|
| **1** | 15–19 Jun | Foundations: literature, data, baselines | 1, 2, baseline of 7 | Clean dataset + working Duval/IEC baselines + literature synthesis + sharpened problem statement |
| **2** | 22–26 Jun | Representation learning (AE / VAE) | 3, 4 | Trained AE & VAE, latent codes saved, evidence-based choice of representation |
| **3** | 29 Jun–3 Jul | Clustering + anomaly detection | 5, 6 | Full pipeline gives clusters + anomalies with quantitative evaluation |
| **4** | 6–10 Jul | Evaluation, comparison, ablations, figures | 7, 8 | Frozen results, all paper figures/tables, full first draft |
| **5** | 13–17 Jul | Writing, review, dissemination | 9 | Submitted IEEE paper + reproducible repo + slides |
| buffer | 20 Jul+ | Defense / revisions | — | Defense delivered |

**Weekly cadence:** Monday — plan the week with the supervisor; Friday — 1-page progress note + demo. Keep `docs/lab_notebook.md` updated daily (5 min: what you did, what you found, what's next).

---

## 3. Week-by-week detail

### Week 1 — Foundations (15–19 Jun)
**Goal:** lock the problem, the data, and the baselines you'll compare against.

- **Mon** — Environment: clone repo, create the venv, run the smoke test (`prepare_data → train_ae → run_clustering`, see README). Meet Dr. Phadungthin: confirm scope, the target paper venue/format, and data-use/anonymisation rules. Start reading IEC 60599 + IEEE C57.104 + Duval (2002).
- **Tue–Wed** — **Literature review** (step 1): 15–20 papers across three buckets — conventional DGA, supervised ML for DGA, unsupervised/representation learning + anomaly detection. Capture each in `docs/literature.md` (1 line: problem / method / dataset / limitation). Write the precise research question and your 2–3 stated contributions.
- **Wed–Thu** — **Deep EDA** (step 2, `notebooks/01_eda.ipynb`): distributions, correlations, missingness, the field notes. Decide and document the final feature set and preprocessing. Lock the "Dataset" section facts.
- **Thu–Fri** — **Conventional baselines** (`notebooks/03_…`): run `conventional.diagnose`, reproduce the Duval/IEC distributions, eyeball the Duval triangle. These become your external labels.
- **Deliverables:** `literature.md` (15+ entries), finalized preprocessing, baseline label distributions, draft notes for paper sections 1–3.
- **Risks:** rabbit-holing in the literature. Time-box it; 20 good references beat 60 skimmed.

### Week 2 — Representation learning (22–26 Jun)
**Goal:** a latent representation that is *useful*, with evidence for why.

- **Mon–Tue** — Train and tune the **autoencoder** (`notebooks/04_…`): vary `latent_dim` (2–5) and hidden sizes; inspect training curves and reconstruction quality.
- **Wed** — Implement/compare the **VAE** (`dga.models.vae`); compare latent geometry and reconstruction.
- **Thu** — **Latent analysis (the key experiment):** colour the latent by Duval class. If classes don't separate, the latent encodes *severity*, not *type*. Test the fix: per-sample normalisation (gas proportions, the way Duval uses ratios) so magnitude is factored out. This is likely your main novelty/insight.
- **Fri** — Freeze the chosen representation; draft Method §4.1–4.2.
- **Deliverables:** saved AE & VAE, latent codes, a short written justification of the representation choice.
- **Risks:** chasing reconstruction loss for its own sake — what matters is downstream cluster/anomaly usefulness, not the lowest MSE.

### Week 3 — Clustering & anomaly detection (29 Jun–3 Jul)
**Goal:** turn the latent space into fault patterns + an anomaly flag, evaluated quantitatively.

- **Mon–Tue** — **Clustering** (`notebooks/05_…`): KMeans/GMM/HDBSCAN, model selection by silhouette, cluster×fault cross-tab. Interpret each cluster physically (which gases dominate, which Duval faults fall inside).
- **Wed–Thu** — **Anomaly detection** (`notebooks/06_…`): reconstruction error + Isolation Forest + One-Class SVM; threshold sweep; precision/recall against the field-event notes.
- **Fri** — Consolidate; lock Method §4.3–4.4; regenerate result tables.
- **Deliverables:** cluster metrics + fault-profile tables, anomaly precision/recall vs notes, interpreted clusters.
- **Risks:** treating the field notes as exhaustive ground truth — they are sparse positives, so report metrics as indicative.

### Week 4 — Evaluation, ablations, figures (6–10 Jul)
**Goal:** a defensible results section. Freeze the science this week.

- **Mon–Tue** — **Master comparison** (`notebooks/07_…`): AE+KMeans vs VAE+KMeans vs raw+KMeans vs Isolation Forest, with internal + external metrics. Ablations: latent dim, scaler, normalised vs raw features.
- **Wed** — Robustness (multiple seeds), sensitivity, error analysis → discussion points.
- **Thu–Fri** — Produce **all paper-quality figures/tables** (300 dpi, `results/figures`). Write Experiments + Results + Discussion in prose.
- **Deliverables:** frozen `results/`, complete figure/table set, full first draft of every section.
- **Rule:** after Friday, **no new experiments** — only writing and polish. Scope creep here is the #1 way 5-week projects fail.

### Week 5 — Writing & dissemination (13–17 Jul)
**Goal:** submit a clean paper + reproducible repo + slides.

- **Mon–Tue** — Complete the IEEE draft end-to-end (`paper/`); tighten abstract/intro/conclusion; fill every `\todo{}`.
- **Wed** — Self-review (use Claude as "Reviewer 2", see the Claude guide), verify **every citation by hand**, proofread, check IEEEtran length/format.
- **Thu** — Supervisor review + revisions; build defense slides.
- **Fri** — Final package: paper PDF, tagged repo, README, slides. Keep a buffer.
- **Deliverables:** submitted paper, reproducible repo, slides.

---

## 4. Milestones & artifacts checklist

- [ ] M1 (Fri W1): dataset locked, baselines reproduced, literature synthesised
- [ ] M2 (Fri W2): representation chosen with evidence
- [ ] M3 (Fri W3): clusters + anomalies evaluated
- [ ] M4 (Fri W4): results frozen, figures done, full draft
- [ ] M5 (Fri W5): paper + repo + slides submitted

## 5. Risk register (top 5)

1. **Over-scoping** → fixed weekly milestones, hard experiment-freeze after W4.
2. **Weak/sparse labels** → use notes only as indicative validation; lean on internal metrics + conventional-method agreement.
3. **Latent encodes severity not fault type** → planned normalisation experiment in W2; it's also a legitimate finding to report either way.
4. **Hallucinated citations / numbers** → every reference verified manually; every number in the paper comes from `results/`, never from a language model.
5. **Reproducibility drift** → fixed seeds, all outputs written to `results/`, config-driven runs, repo tagged at submission.

## 6. Academic integrity

Claude is a tool for planning, coding, debugging, drafting and reviewing. The scientific claims, the verified citations, the interpretation, and the final text are yours. Keep `docs/lab_notebook.md` as your own record, never paste unverified model output into the paper, and check whether CESI/KMUTNB require an AI-use disclosure — if so, write one.

> See `docs/GUIDE_CLAUDE_CODE.md` for exactly how to use Claude Code at each step, and `CLAUDE.md` for the project context Claude loads automatically.
