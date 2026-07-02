# DGA Fleet-Health Dashboard

An interactive Streamlit app over the label-free DGA framework. It is wired
**live** to the `dga` package — cleaning, conventional diagnoses, the SD-CAE
compositional representation, and the Health/Risk index are all computed from
the real pipeline (no hard-coded numbers). Change the component weights and the
fleet re-ranks instantly.

## Run

From the project root, with the virtual environment active:

```bash
pip install -r requirements.txt          # adds streamlit + plotly
streamlit run dashboard/app.py
```

It opens at http://localhost:8501. The first load trains the small anomaly AE
and the SD-CAE (a few seconds, then cached).

## Pages

- **Overview** — fleet KPIs, risk distribution, decile validation, riskiest units.
- **Fleet ranking** — every unit ranked, filterable, exportable; risk vs. acetylene.
- **Transformer inspector** — per-unit gas trajectory, risk breakdown (radar +
  gauge), Duval Triangle, conventional diagnoses, field notes.
- **SD-CAE fault map** — the learned, label-free diagnostic map + the ablation
  that justifies the log-ratio geometry.
- **Surveillance bias** — the honest finding: the field-event label is
  confounded by sample count; a confound-free chemistry target.
- **Conventional methods** — Duval / IEC / Rogers baselines on the same fleet.

## Notes

- The sidebar **Risk weights** recombine the standardised components
  (severity, acetylene, temporal, anomaly) and re-rank the fleet in real time.
  Defaults match the paper (acetylene up-weighted to 2).
- Some panels read precomputed tables from `results/tables/`. If a panel says a
  table is missing, run the matching script (`scripts/run_*.py`) once.
