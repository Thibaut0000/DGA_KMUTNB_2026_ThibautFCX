# Reference triage — what your supervisor gave you, verified

Triaged on 2026-06-14. **Key finding: 3 of the 6 references your supervisor listed are
generic pointers (a journal + a topic), not specific papers.** Below, each is matched to a
real, verified paper, plus two corrections and the missing piece — unsupervised methods,
which is literally your topic and absent from the list.

**Status legend:** [verified] verified (edition, volume, pages confirmed) · [to confirm] identified, confirm
one detail on the publisher page before citing.

---

## Part A — Your supervisor's 6 references, resolved

| # | What she wrote | What it actually is | Status | Note |
|---|----------------|---------------------|--------|------|
| 1 | IEEE C57.104 — interpretation of gases | **IEEE Std C57.104-2019**, *Guide for the Interpretation of Gases Generated in Mineral Oil-Immersed Transformers* | [verified] | **Cite the 2019 edition**, not the 2008 "Oil-Immersed" one she may be thinking of. |
| 2 | IEC 60599 — interpretation of dissolved gases | **IEC 60599:2022 (Ed. 4.0)**, *Guidance on the interpretation of dissolved and free gases analysis* | [verified] | Latest is **2022**; 2015 (Ed. 3) is still widely cited. Source of the IEC ratio limits you implement. |
| 3 | Duval (2008), "Duval Triangle for Load Tap Changers" | Duval, *IEEE Electr. Insul. Mag.* **24(6):22–29, 2008** | [verified] | [to confirm] **Scope mismatch:** this paper is about **load tap changers, non-mineral oils, low-temp faults** — your data is the **main tank**. Useful, but not your primary Duval reference (see correction below). |
| 4 | Elsevier ESWA — "fault diagnosis using ML techniques" | **Fei & Zhang (2009)**, "Fault diagnosis of power transformer based on SVM with genetic algorithm," *Expert Systems with Applications* | [to confirm] | Real ESWA paper; confirm exact volume/pages on ScienceDirect. Generic slot → this is a strong representative. |
| 5 | IEEE TPWRD — "DGA fault diagnosis using neural networks" | **Zhang, Ding, Liu & Griffin (1996)**, "An artificial neural network approach to transformer fault diagnosis," *IEEE Trans. Power Delivery* **11(4):1836–1841** | [verified] | The **classic** NN-on-DGA paper in exactly that journal. Perfect fit for the slot. |
| 6 | Elsevier EPSR — "data-driven approaches" | **Bacha, Souahlia & Gossa (2012)**, "Power transformer fault diagnosis based on DGA by support vector machine," *Electric Power Systems Research* **83(1):73–79** | [verified] | Real EPSR data-driven paper. Exactly the slot. |

### Two corrections to make
1. **Add Duval's *main-tank* triangle reference.** Your data is main-tank, so cite the
   original **Duval (2002)**, "A review of faults detectable by gas-in-oil analysis in
   transformers," *IEEE Electr. Insul. Mag.* **18(3):8–17** [verified] — this defines Duval Triangle 1
   and the fault classes (PD, D1, D2, T1, T2, T3) you actually implement in `conventional.py`.
   Keep the 2008 LTC paper as secondary.
2. **Use current editions** of both standards (C57.104-**2019**, IEC 60599-**2022**).

---

## Part B — The missing bucket: unsupervised / representation learning (your topic)

Your supervisor's list is all *conventional* + *supervised*. Your contribution is **label-free**,
so you need prior work on autoencoders, clustering and anomaly detection for DGA. Verified, real:

| Key | Paper | Why it matters for you |
|-----|-------|------------------------|
| [verified] | **Wang, Zhao, Pei & Tang (2016)**, "Transformer fault diagnosis using continuous sparse autoencoder," *SpringerPlus* 5:448 | Closest prior art: **unsupervised** autoencoder feature learning on DGA. Cite + contrast. |
| [verified] | **DGA + DBSCAN (2020)**, "Power Transformer Fault Diagnosis Based on DGA by Correlation Coefficient-DBSCAN," *Applied Sciences* 10(13):4440 | **Unsupervised clustering** of DGA — directly your Week 3 step. Good comparison point. |
| [to confirm] | **IDAE (2022)**, "Missing data imputation using an iterative denoising autoencoder for DGA," *Electric Power Systems Research* | Autoencoder architecture applied to DGA (for imputation); useful design reference. |
| [verified] | **Kingma & Welling (2014)**, "Auto-Encoding Variational Bayes," ICLR | Foundational VAE — your `models/vae.py`. |
| [verified] | **Hinton & Salakhutdinov (2006)**, "Reducing the dimensionality of data with neural networks," *Science* 313:504–507 | Foundational autoencoder reference. |

Optional reviews for a strong Related Work paragraph: *"The State of the Art in transformer
fault diagnosis with AI and DGA: A Review"* (arXiv:2304.11880, 2023) and the *Energies* 11(4):913
(2018) survey.

---

## Part C — Recommended reading order

**P1 — Week 1, because you implement them as baselines:**
IEC 60599-2022 · IEEE C57.104-2019 · Duval 2002 (Triangle 1) · Duval 2008 (LTC, skim).

**P1 — Week 1–2, closest to your method:**
Wang 2016 (sparse autoencoder on DGA) · DGA+DBSCAN 2020 (unsupervised clustering).

**P2 — Related Work context:**
Zhang 1996 (NN) · Bacha 2012 (SVM) · Fei 2009 (SVM+GA) · one review.

**P3 — method depth:**
Kingma & Welling 2014 · Hinton & Salakhutdinov 2006 · IDAE 2022.

---

## Part D — Verify yourself before the paper goes out

- [to confirm] items: open the publisher page and copy exact volume/issue/pages into `references.bib`.
- Access: you're at KMUTNB — IEEE Xplore / ScienceDirect are very likely covered by the
  university subscription. Get the PDFs through the library, not random re-hosts.
- The verified entries are already in `paper/references.bib`; the running annotated list is in
  `docs/literature.md`. Add your own one-line takeaway per paper as you read.

## Sources (verification)
- IEEE C57.104-2019 — https://standards.ieee.org/ieee/C57.104/7476/ · https://ieeexplore.ieee.org/document/8890040/
- IEC 60599:2022 — https://webstore.iec.ch/en/publication/66491
- Duval 2008 (LTC) — https://ieeexplore.ieee.org/document/4665347/
- Duval 2002 (review/Triangle 1) — https://ieeexplore.ieee.org/document/1014963/
- Zhang et al. 1996 (TPWRD) — https://www.osti.gov/biblio/422769
- Bacha et al. 2012 (EPSR) — https://www.sciencedirect.com/science/article/abs/pii/S0378779611002288
- Fei & Zhang 2009 (ESWA) — ScienceDirect (Expert Systems with Applications, 2009)
- Wang et al. 2016 (sparse AE) — https://pmc.ncbi.nlm.nih.gov/articles/PMC4830783/
- DGA + DBSCAN 2020 — https://www.mdpi.com/2076-3417/10/13/4440
- IDAE 2022 (EPSR) — https://www.sciencedirect.com/science/article/abs/pii/S0378779622007155
