# Duval 2002 — A review of faults detectable by gas-in-oil analysis (Duval Triangle 1)

- **Citation:** M. Duval, "A review of faults detectable by gas-in-oil analysis in transformers,"
  *IEEE Electrical Insulation Magazine*, vol. 18, no. 3, pp. 8–17, 2002. `bib: duval2002review` ·
  PDF `docs/refs/duval2002review.pdf`.
- **Type:** foundational conventional-method paper. **Defines the external label we score against.**

## TL;DR
The reference that turns IEC 60599's fault codes into the **Duval Triangle 1** graphical method, and
reviews which faults DGA can actually detect. Our `conventional.py` implements exactly this; it also
*explains why fault type lives in gas proportions, not magnitude* — the conceptual core of our
normalisation experiment.

## What it contains
- **Fault codes** (from IEC 60599): **PD** (partial discharge), **D1** (low-energy discharge),
  **D2** (high-energy discharge), **T1** (<300 °C), **T2** (300–700 °C), **T3** (>700 °C), plus
  **DT** (mixed thermal/electrical).
- **Duval Triangle 1**: plot each sample in the **% of three gases only — %C2H2, %C2H4, %CH4**
  (because they are proportions, overall magnitude is removed by construction). Interior split into
  the seven fault zones.
- Built from **179 inspected in-service faults + laboratory cases** (IEC TC10 databases).
- Practical notes: **T3** → hot spots in *oil*; **T1/T2** often involve *paper*; **CO2/CO < 3**
  *suggests* paper involvement but "use with caution"; **PD** gas levels are often near the DGA
  detection limit (~0.1 ppm) and can be missed.

## Limitations
- A **rule built from a flawed/partial ground truth** (limited inspected faults); the triangle has
  no "normal" zone (every sample is forced into a fault type).

## Relevance to our project
- **This is the external label** our clusters/ranking are compared against (`conventional.py`
  Triangle 1; the Duval mix figure in the Week-1 synthesis).
- **Why we normalise per sample:** the triangle works on **proportions** → confirms fault *type* is a
  compositional property, while our raw-concentration AE encodes *severity*. Primary motivation for
  the Week-2 per-sample-proportion experiment.
- **Evaluation caveat:** D1/D2/DT are intrinsically **rare** (matches our ~70/4563 arcing cases), so
  **ARI vs Duval under-rewards** correct type structure → always report internal metrics + a
  **per-cluster × Duval cross-tab** beside ARI.
