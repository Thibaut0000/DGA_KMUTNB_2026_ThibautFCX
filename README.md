# Label-Free Transformer Health Monitoring from DGA

Research-initiation project (CESI × KMUTNB, June–July 2026). A **label-free** framework
for transformer fleets from Dissolved Gas Analysis (DGA), with three contributions:

1. **Fault types without labels** — representing each sample by the centred log-ratio (CLR)
   of its gas composition separates fault *type* from gassing *severity* by construction;
   a simple linear projection (CLR → PCA → KMeans) agrees with the expert Duval classes at
   **ARI 0.55 vs 0.16** for raw concentrations. A neural autoencoder (SD-CAE) and an
   adversarial variant do **not** beat it — reported as negative results.
2. **A validation caution (surveillance bias)** — the operator-logged field events used to
   validate DGA risk models are confounded: a trivial *sample count* predicts them as well
   as the gas chemistry (AUC 0.76 vs 0.74, n.s.). We quantify the confound, repair part of
   the label by translating 211 Thai field notes, and propose a confound-free
   chemistry-defined target (arcing onset).
3. **A label-free health/risk index** — transparent components (severity, acetylene,
   H2 growth rate, anomaly); riskiest decile ≈ 27% event rate vs 2% for the safest —
   always reported against the confound floor above.

**Deliverables:** a 6-page IEEE paper (`paper/`), this reproducible repository, an
interactive dashboard (`dashboard/`), and the defense deck (`presentation/`).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_all.py                    # regenerate every number, table and figure
pytest tests/ -q                     # smoke tests (10)
streamlit run dashboard/app.py       # interactive fleet dashboard (localhost:8501)
```

The confidential raw dataset is **not** in this repository; the dashboard and most
analyses fall back to `data/public/dga_public.parquet`, an anonymised extract (hashed
unit ids, dates, gas concentrations, event flags). The paper builds with
`latexmk -pdf main.tex` (use a texlive Docker image if LaTeX is not installed).

## Layout

```
config/        experiment configuration (edit this, not the code)
data/          raw/ (private) · public/ (anonymised extract, tracked)
src/dga/       the package: data, compositional, conventional, temporal, health, models, ...
scripts/       one script per experiment; run_all.py chains them
tests/         pytest smoke suite
notebooks/     01_eda … 07_evaluation
dashboard/     Streamlit app (fleet ranking, inspector, fault map, surveillance bias)
paper/         IEEEtran LaTeX (main.tex + sections/ + references.bib)
presentation/  defense deck (self-contained HTML + PowerPoint + builders)
results/       figures/ and tables/ (all generated; regenerate with run_all.py)
docs/          problem statement, lab notebook, audits, final report, defense prep
```

## Start here

- **Final report:** [`docs/final_report_EN.md`](docs/final_report_EN.md)
- **Lab notebook (the honest history):** [`docs/lab_notebook.md`](docs/lab_notebook.md)
- **Deep audit that reframed the project:** [`docs/deep_audit_2026-06-30.md`](docs/deep_audit_2026-06-30.md)
- **Project context for Claude Code:** [`CLAUDE.md`](CLAUDE.md)

## Dataset

4,563 main-tank samples · 628 transformers · 42 manufacturers · 2019–2024, assembled from
multiple operational sources in Thailand and anonymised. Gases stored as text (heavy
right-skew → log1p; isolated outliers → clipped). 391 samples carry a free-text field-event
note (69% contain Thai); a curated translation of all 211 distinct Thai notes is released
for expert validation. Raw data stays out of version control.
