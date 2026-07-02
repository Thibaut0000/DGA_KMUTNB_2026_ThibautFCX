"""Export an anonymised dataset so the dashboard can run without the
confidential raw file (e.g. on Streamlit Community Cloud).

    python scripts/make_dashboard_data.py

Kept:    anonymised unit id (TX-0001...), sample date, all gas concentrations,
         and the two boolean note flags (has_note, fault_note).
Dropped: free-text notes (they contain substation/equipment identifiers),
         nameplate metadata (NAME, MFG, SER, KV, MVA, YEAR_Energized, LOC),
         and the tested-day/temperature/water columns.
The unit-id mapping is written to data/processed/ (git-ignored) for internal
reference; the public parquet goes to data/public/ (tracked by git).
"""
import _bootstrap  # noqa: F401
import pandas as pd

from dga.config import PROJECT_ROOT
from dga import data as dga_data

PUBLIC_DIR = PROJECT_ROOT / "data" / "public"
KEEP_GASES = ["O2", "N2", "CO2", "CO", "H2", "CH4",
              "C2H2", "C2H4", "C2H6", "C3H6", "C3H8", "TCG"]


def main():
    df = dga_data.load_clean()

    units = sorted(df["CODETX"].astype(str).unique())
    mapping = {u: f"TX-{i+1:04d}" for i, u in enumerate(units)}

    out = pd.DataFrame(index=df.index)
    out["CODETX"] = df["CODETX"].astype(str).map(mapping)
    out["Sample Day"] = pd.to_datetime(df["Sample Day"], errors="coerce")
    for g in KEEP_GASES:
        if g in df.columns:
            out[g] = pd.to_numeric(df[g], errors="coerce")
    out["has_note"] = df["has_note"].fillna(False).astype(bool)
    out["fault_note"] = df["fault_note"].fillna(False).astype(bool)
    out.index.name = "sample_id"

    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PUBLIC_DIR / "dga_public.parquet")

    pdir = PROJECT_ROOT / "data" / "processed"
    pdir.mkdir(parents=True, exist_ok=True)
    pd.Series(mapping).rename("public_id").to_csv(pdir / "unit_id_mapping.csv")

    print(f"public dataset: {out.shape[0]} samples, {out['CODETX'].nunique()} units, "
          f"{out.shape[1]} columns -> {PUBLIC_DIR / 'dga_public.parquet'}")
    print("columns:", ", ".join(out.columns))
    print("id mapping (git-ignored):", pdir / "unit_id_mapping.csv")


if __name__ == "__main__":
    main()
