# PDFs to read (drop them here)

Put the reference PDFs in this folder so Claude can read them and so they stay with
the repo notes (the folder is git-ignored if you add it to `.gitignore` — large PDFs
should not be committed). Suggested filenames match the citation keys in
`paper/references.bib` and the reading notes in `../literature.md`.

## P1 — read first (Week 1)

| Drop a file named… | Paper | Where to get it | Status |
|--------------------|-------|-----------------|--------|
| `hybrid_aiscore2026.pdf` | Saleh et al. 2026, EPSR 256:112877 — Hybrid DGA + ensemble ML (supervised SOTA, supervisor's ref) | uploaded by Thibaut | **present** |
| `ramarao2026review.pdf` | Ramarao et al. 2026, EAAI 181:115390 — DGA→AI review, 34 pp (supervisor's ref) | uploaded by Thibaut | **present** |
| `wang2016csae.pdf`       | Wang et al. 2016, SpringerPlus 5:448 — sparse autoencoder on DGA (closest prior art) | free (open access) | **present** |
| `dga_dbscan2020.pdf`     | Liu et al. 2020, Appl. Sci. 10(13):4440 — unsupervised DBSCAN clustering of DGA | free (open access) | **present** |
| `duval2002review.pdf`    | Duval 2002, IEEE EIM 18(3):8–17 — Duval Triangle 1 + fault classes | IEEE Xplore (KMUTNB login) | **present** |
| `ieee_c57104_2019.pdf`   | IEEE C57.104-2019 — interpretation of generated gases | IEEE Xplore / library (standard) | **present** |
| `iec60599_2022.pdf`      | IEC 60599:2022 — dissolved-gas interpretation (ratio limits) | paid standard — **not acquired** | **cite-only** |

> **IEC 60599** is intentionally not downloaded (expensive paid standard). It is still **cited**
> in the paper; its ratio thresholds are already implemented in `src/dga/conventional.py` and
> reproduced in the papers we do have (Saleh 2026, the Duval articles). No action needed.

### Direct links
- `wang2016csae.pdf` — open access: https://pmc.ncbi.nlm.nih.gov/articles/PMC4830783/ (PDF button)
- `dga_dbscan2020.pdf` — open access: https://www.mdpi.com/2076-3417/10/13/4440 (Download PDF)
- `duval2002review.pdf` — https://ieeexplore.ieee.org/document/1014963/ (via KMUTNB subscription)
- `ieee_c57104_2019.pdf` — https://ieeexplore.ieee.org/document/8890040/ (standard; library access)
- `iec60599_2022.pdf` — https://webstore.iec.ch/en/publication/66491 (paid; check the KMUTNB library first)

## P2 / optional — to reach 18--20 references (drop the PDFs, I will add reading notes)

| Drop a file named… | Paper | Where to get it |
|--------------------|-------|-----------------|
| `mirowski2012review.pdf` | Mirowski & LeCun 2012, IEEE TPWRD 27(4):1791--1799 — review of statistical ML on DGA | public PDF: http://www.mirowski.info/pub/dga ; or https://ieeexplore.ieee.org/document/6266722 |
| `duval2001interp.pdf`    | Duval & dePablo 2001, IEEE EIM 17(2):31--41 — IEC 60599 + IEC TC10 databases | https://ieeexplore.ieee.org/document/917529 (KMUTNB login) |
| `dga_ai_review2023.pdf`  | DGA + AI review, 2023 — to verify before citing | arXiv: https://arxiv.org/abs/2304.11880 |
| `dga_survey_energies2018.pdf` | DGA review survey — broad context (to verify) | https://www.mdpi.com/1996-1073/11/4/913 |

Mirowski 2012 and Duval 2001 are already in `references.bib` (details taken from the reference
lists of papers we hold). The 2023 review and 2018 survey still need their bibliographic details
verified before they go into `references.bib`.

Easiest order: the two **free** ones now, the uploaded EPSR paper is already here, then the IEEE/IEC
items through the KMUTNB library. Full reference list & priorities: `../literature.md`.

Once a PDF is here, tell Claude (e.g. "read docs/refs/hybrid_aiscore2026.pdf") and it
will produce a reading-note draft in `../literature.md` for you to verify.
