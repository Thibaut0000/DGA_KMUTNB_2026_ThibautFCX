# Briefing for Dr. Rattanakorn — decisions needed (prepared 2026-06-30)

Four items need your input before the paper is finalised. Supporting evidence:
`docs/deep_audit_2026-06-30.md` (full audit) and `results/tables/` (all numbers
reproducible by script). Suggested meeting length: 30 minutes.

## 1. A finding that changes the paper's method claim (decision needed)

While stress-testing our own headline result, we found that the neural network
in SD-CAE is unnecessary: a deterministic **linear** pipeline on the same
compositional features outperforms it.

| representation (same 5 gases, same evaluation) | ARI vs Duval (k=7) |
|---|---|
| raw log-concentrations + KMeans | 0.160 |
| CLR (centred log-ratio) + KMeans | 0.491 |
| **CLR + PCA-2 + KMeans (linear)** | **0.545 ± 0.002** |
| SD-CAE autoencoder (current paper headline) | 0.474 ± 0.055 |

The severity/type separation theory is unchanged (the CLR removes gassing
magnitude by construction); only the encoder is simpler. Robustness: the
structure holds on a strict temporal split (fit before 2022, evaluate 2022+,
ARI 0.454); it is sensitive to the zero-replacement constant (disclosed).

**Proposal:** reframe contribution 1 as "the compositional log-ratio geometry
recovers fault type — even a linear projection suffices"; deploy the linear
variant; report the autoencoder and the adversary as negative ablations
(consistent with the paper's honest-negatives style). The headline improves
(ARI 0.14 -> 0.545) and becomes deterministic and interpretable (the PCA
loadings give a log-ratio biplot an engineer can read).

**Why we cannot keep the current framing:** any reviewer can run PCA; if they
find it first, the paper is rejected for overclaimed novelty.

**Ask: approve the reframe (or discuss).**

## 2. Title (sign-off needed)

The June brief's working title emphasised self-supervised and temporal
representation learning; the evidence now supports neither emphasis (the
temporal/prognostic angle turned out to be dominated by a sampling confound —
item 3 of the paper). Current draft title:

> Label-Free Compositional DGA Representations and a Surveillance-Bias Caution
> for Transformer Risk Screening

It already fits the reframe of item 1. **Ask: approve or amend.**

## 3. Thai field-note classification (your validation needed)

69% of the field notes contain Thai text (127 notes are Thai-only); the
English-keyword event label misses them. We translated and classified all 211
distinct Thai notes; 15 describe genuine events (about 18 units), for example
"ตามผลเนื่องจาก Bkr. ระเบิด" (breaker explosion), "หม้อแปลงมีเสียงดัง"
(abnormal noise), "ค่า p.f สูง" (high insulation power factor). Follow-ups
triggered by gas findings (e.g. "ตามผล C2H2") were deliberately excluded to
avoid circularity.

File to review: `results/tables/thai_note_classification.csv` (15 fault
candidates + 4 documented exclusions, with English glosses and tiers).

Impact: adding the tier-A (own-unit electrical) events makes the sample-count
confound's advantage disappear entirely (AUC 0.734 vs index 0.735) — evidence
that part of the confound came from label incompleteness.

**Ask: confirm or correct the classifications (especially tier A), ~15 rows.**

## 4. Dataset provenance statement (blocking for submission)

Section III currently has no data-source statement. Reviewers will ask.
**Ask: what wording is permitted?** E.g. "operational DGA records from a
transmission utility in Thailand, anonymised; data not publicly available" —
or should the utility be named/acknowledged?

## Also worth mentioning (no decision needed)

- Statistics upgraded throughout: bootstrap CIs on all headline AUCs; the
  index-vs-count gap is not significant (p = 0.58); a likelihood-ratio test
  shows the gas signal is real but dominated by the sampling confound; the
  arcing-onset result now uses honest unit-blocked intervals (17 onset units).
- Literature: the surveillance-bias finding will be positioned against the
  epidemiology/EHR literature (Sackett 1979; Haut & Pronovost 2011; Goldstein
  2016), and the compositional method against Dukarm et al. 2020 (simplex
  without log-ratios) — our DGA-specific novelty survives both.
- An interactive dashboard of the whole framework is available
  (`streamlit run dashboard/app.py`).
