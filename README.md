# Deep Unsupervised Fault Detection in Transformer DGA Data

Research-initiation project (CESI × KMUTNB, June–July 2026). A label-free pipeline
that learns latent representations of Dissolved Gas Analysis (DGA) measurements with
an **autoencoder / VAE**, discovers fault patterns by **clustering** the latent space,
and detects incipient faults via **reconstruction-error anomaly detection** — then
benchmarks against the conventional Duval / IEC / Rogers methods.

**Deliverable:** a short IEEE-format paper (`paper/`) + this reproducible repository.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # smaller CPU build

python scripts/prepare_data.py      # clean data + conventional diagnoses
python scripts/train_ae.py          # train autoencoder, save latent codes
python scripts/run_clustering.py    # cluster + evaluate vs Duval labels
jupyter lab                          # explore notebooks 01–07
```

## Layout

```
config/      experiment configuration (edit this, not the code)
data/        raw/ (the .xlsx) + processed/
src/dga/     the package: data, preprocessing, conventional, models, clustering, anomaly, evaluation, viz
scripts/     prepare_data → train_ae → run_clustering
notebooks/   01_eda … 07_evaluation
paper/       IEEEtran LaTeX
results/     figures/, tables/, models/ (generated)
docs/        PROJECT_PLAN, GUIDE_CLAUDE_CODE, methodology, literature, lab_notebook
```

## Start here

- **Plan:** [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) — the 5-week schedule & milestones.
- **Working with Claude:** [`docs/GUIDE_CLAUDE_CODE.md`](docs/GUIDE_CLAUDE_CODE.md).
- **Project context for Claude Code:** [`CLAUDE.md`](CLAUDE.md).

## Dataset

4,563 main-tank samples · 628 transformers · 42 manufacturers · 115–525 kV · 2019–2024.
Gases stored as text (heavy right-skew → log1p; isolated outliers → clipped). 391 rows
carry a real field-event note used as weak validation labels. Keep raw data out of any
public repo unless cleared by the host lab.
