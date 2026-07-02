"""Load and clean the raw transformer DGA Excel file into a tidy DataFrame.

The raw sheet stores every gas concentration as *text* (with '-' and '' used for
missing), dates as Excel datetimes, and a free-text note column (NB) that, for a
subset of rows, describes a real field event (Buchholz trip, bushing explosion,
high power factor, ...). Those notes are the closest thing we have to ground
truth and are exposed here as a boolean `has_note` flag plus the cleaned text.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, resolve

# Gas concentration columns (ppm) present in the file.
GAS_COLS = ["O2", "N2", "CO2", "CO", "H2", "CH4",
            "C2H2", "C2H4", "C2H6", "C3H6", "C3H8", "TCG"]
DATE_COLS = ["Sample Day", "Tested day"]
_MISSING_TOKENS = {"-": np.nan, "": np.nan, "N/A": np.nan,
                   "na": np.nan, "NA": np.nan, "None": np.nan}

# The NB column is a heterogeneous operational log (English + Thai abbreviations): mostly
# admin/maintenance (research, repeat, de-energize, overhaul/OH, HV test, oil purify, ...),
# only a minority real faults. `fault_note` is a keyword heuristic flagging genuine
# protection/damage events (trip, Buchholz, differential, sudden-pressure, bushing,
# overheat, ...) to use as a *weak* risk label. NOTE: ~24% of notes are Thai-only and are
# not captured; refine `_FAULT_NOTE_RE` if a Thai translation becomes available.
_FAULT_NOTE_RE = re.compile(
    r"\btrip|bou?chhol|buchhol|\bdiff(?:erential)?\b|tx\.?\s*diff|sudden\s*press|"
    r"press(?:ure)?\s*reli|relief\s*dev|oil\s*flow|over\s*current|overcurrent|lock\s*out|"
    r"\bflash|lightning|ground\s*fault|to\s*ground|\bbushing|over\s*heat|overheat|hydran|"
    r"surge\s*arest|\bfault\b|\balarm|87k|51g",
    re.IGNORECASE,
)


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read the .xlsx exactly as stored (no type coercion)."""
    cfg = load_config()
    path = resolve(path or cfg.paths.raw_xlsx)
    df = pd.read_excel(path, engine="openpyxl")
    # Drop the trailing all-empty column pandas names 'Unnamed: NN'.
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    return df


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().replace(_MISSING_TOKENS)
    return pd.to_numeric(cleaned, errors="coerce")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce gases to float, parse dates, normalise the NB note column."""
    df = df.copy()

    for col in GAS_COLS:
        if col in df.columns:
            df[col] = _to_numeric(df[col])

    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # NB = free-text field note. Treat '-'/'' as missing; flag real notes.
    if "NB" in df.columns:
        nb = df["NB"].astype("string").str.strip()
        nb = nb.mask(nb.isin(["-", ""]))
        df["NB"] = nb
        df["has_note"] = nb.notna()
        # Weak risk label: genuine protection/damage events only (see _FAULT_NOTE_RE).
        df["fault_note"] = nb.notna() & nb.str.contains(_FAULT_NOTE_RE, na=False)

    df = df.reset_index(drop=True)
    df.index.name = "sample_id"
    return df


def load_clean(path: str | Path | None = None) -> pd.DataFrame:
    """Convenience: load_raw + clean."""
    return clean(load_raw(path))


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Quick per-gas summary (count, missing, min/median/max) for the EDA."""
    rows = []
    for col in GAS_COLS:
        if col in df.columns:
            s = df[col]
            rows.append({
                "gas": col,
                "n": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "min": s.min(),
                "median": s.median(),
                "max": s.max(),
            })
    return pd.DataFrame(rows).set_index("gas")


if __name__ == "__main__":  # python -m dga.data
    frame = load_clean()
    print(frame.shape)
    print(summary(frame))
    print("rows with a field note:", int(frame.get("has_note", pd.Series()).sum()))
