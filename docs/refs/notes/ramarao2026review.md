# Ramarao et al. 2026 — DGA for transformer diagnostics: from traditional methods to AI-driven prognostics

- **Citation:** G. Ramarao, K. Dhananjay Rao, P. Chongala, P. Pavani, S. Dawn, T. S. Ustun,
  "Advances in dissolved gas analysis for power transformer diagnostics: From traditional methods
  to artificial intelligence driven prognostics," *Engineering Applications of Artificial
  Intelligence*, vol. 181, art. 115390, 2026. (34 pages.) `bib: ramarao2026review`
- **Type:** comprehensive **review / survey** (not a primary method paper). Sent by the supervisor.
- **PDF:** `docs/refs/ramarao2026review.pdf`.

## TL;DR
A 2026 state-of-the-field review that maps DGA diagnostics from rule-based heuristics to modern AI,
quantifies the accuracy/data trade-offs of each method family, and proposes a roadmap toward
explainable, online, predictive (prognostic) DGA. This is the **backbone for our Related Work** and
for **positioning** our contribution.

## What it contains (taxonomy + reported accuracy ranges)
- **Traditional heuristics** (Key Gas, Duval Triangle, ratios): simple but error-prone,
  **~60–75 %** accuracy on small utility datasets (<1000 samples).
- **Classical ML** (SVM, Random Forest, …): **~85–95 %** on ~300–1000 samples; recurring problems =
  **class imbalance** and **interpretability**.
- **Deep learning** (GAN, CNN, LSTM): can exceed **97 %** on **≥5000 samples**, but increases
  **false labelling of complex/mixed faults**, is **opaque**, and is **computationally heavy** →
  limited industrial uptake.
- **Hybrid AI** (neuro-fuzzy, Bayesian networks; physics-/rule-informed): **95–98 %** on ~600–1500
  samples, with **better interpretability and robustness**.
- **Digital twins** (DGA + thermal/electrical simulators): predict **Remaining Useful Life (RUL)**
  within **±10–15 %** error.
- **IoT-enabled online monitoring**: scalable but raises **interoperability, cybersecurity,
  standardisation** issues.
- **Roadmap proposed:** hybrid **explainable** models (XAI), low-cost **online sensors**,
  **federated learning** (privacy), built-in **cybersecurity** → move DGA "from a diagnostic tool to
  an AI-supported predictive engine for grid resilience." (Dedicated sections: 5.5 federated
  learning, 6.2 digital-twin frameworks, 9.2 federated/privacy-preserving ML.)

## Limitations (as a source)
- It is a **review**: accuracy figures are **aggregated from many heterogeneous datasets** and are
  **not directly comparable** to one another or to our unsupervised ranking. Cite it for the
  **landscape and trends**, not as primary evidence; verify any specific number against the original
  paper before quoting it as a result.

## Relevance to our project (how we use it)
- **Related Work skeleton:** structure §2 exactly along its families (conventional → classical ML →
  DL → hybrid → digital-twin/prognostics), then place our work.
- **Positioning / the gap:** the review is dominated by **supervised classification**; **unsupervised
  / anomaly / label-free** approaches are thinly covered, and **prognostics + temporal/RUL** is named
  as the frontier. Our **label-free, self-supervised, temporal fleet health-ranking** sits precisely
  in that under-served, forward-looking space → strong "to the best of our knowledge" framing.
- **Honest framing of accuracy:** use its ranges to explain why we **do not chase a 97–99 % accuracy
  number** (those are supervised, balanced, often <1000–5000 curated samples) and instead report
  ranking/agreement metrics on 4,563 real, imbalanced, unlabelled samples.
- **Future-work fuel:** digital twins, federated learning, online sensors, XAI → ready-made
  "future work" paragraph that connects our prototype to the field's direction.
