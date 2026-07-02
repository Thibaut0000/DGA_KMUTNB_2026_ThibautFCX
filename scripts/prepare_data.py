"""Step 1 — load, clean and persist the DGA data + conventional diagnoses.

    python scripts/prepare_data.py
Outputs:
    data/processed/dga_clean.parquet      cleaned tidy frame (+ duval/iec/rogers)
    results/tables/gas_summary.csv         per-gas summary stats
"""
import _bootstrap  # noqa: F401
from pathlib import Path

from dga.config import PROJECT_ROOT, load_config
from dga import data as dga_data
from dga import conventional

cfg = load_config()


def main():
    df = dga_data.load_clean()
    print(f"loaded {len(df)} samples, {df['CODETX'].nunique()} transformers")

    diag = conventional.diagnose(df)
    df = df.join(diag)
    print("Duval label distribution:\n", df["duval"].value_counts(dropna=False))

    out = PROJECT_ROOT / cfg.paths.processed
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out)
        print("wrote", out)
    except Exception as exc:  # pyarrow missing -> fall back to csv
        csv = out.with_suffix(".csv")
        df.to_csv(csv, index=True)
        print(f"parquet unavailable ({exc}); wrote {csv}")

    tdir = PROJECT_ROOT / "results" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    dga_data.summary(df).to_csv(tdir / "gas_summary.csv")
    print("wrote", tdir / "gas_summary.csv")


if __name__ == "__main__":
    main()
