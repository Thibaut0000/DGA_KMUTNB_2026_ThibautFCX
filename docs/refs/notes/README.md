# Reading notes — exhaustive per-document summaries

One standalone `.md` per held reference, much more detailed than the one-liners in
`../../literature.md`. Purpose: a durable, precise knowledge base to pull from when writing the
paper (Related Work, Method positioning, Discussion) without re-reading the PDFs. Every number
is taken from the source PDF (in `docs/refs/`); claims not in the PDF are flagged.

**How this fits the research pipeline:** `references.bib` (citation) → this folder (what the paper
says + how we use it) → `literature.md` (one-line index) → `paper/sections/` (where it gets cited).

| Note | Document | Bucket | Use in our paper |
|------|----------|--------|------------------|
| [ramarao2026review](ramarao2026review.md) | Ramarao et al. 2026, EAAI 181:115390 — DGA→AI review (34 pp) | Review | Related-Work backbone; field taxonomy; positioning |
| [hybrid_aiscore2026](hybrid_aiscore2026.md) | Saleh et al. 2026, EPSR 256:112877 — Hybrid AI-Score Index | Supervised SOTA | The contrast paper (supervised, rule-fusing) |
| [wang2016csae](wang2016csae.md) | Wang et al. 2016, SpringerPlus 5:448 — continuous sparse AE | Representation | Closest prior art (AE on DGA) |
| [dga_dbscan2020](dga_dbscan2020.md) | Liu et al. 2020, Appl. Sci. 10(13):4440 — CC-DBSCAN | Unsupervised | Week-3 clustering comparison point |
| [duval2002review](duval2002review.md) | Duval 2002, IEEE EIM 18(3):8–17 — Triangle 1 | Conventional | Defines the external label (Duval) |
| [ieee_c57104_2019](ieee_c57104_2019.md) | IEEE C57.104-2019 — gas interpretation guide | Standard | Severity/anomaly framing; percentile limits |
| [iec60599](iec60599.md) | IEC 60599:2022 — ratio method (cite-only, no PDF) | Standard | Source of the IEC ratios in `conventional.py` |

Cross-cutting thread (see each note's "relevance"): every method that recovers fault **type**
factors out magnitude via **per-sample proportions/ratios** (Duval %, Wang IEC ratios, Liu %),
which is the published justification for our Week-2 per-sample normalisation experiment.
