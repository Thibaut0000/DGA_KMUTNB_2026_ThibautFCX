"""Thai field-note translation: extend the weak fault label with events the
English-keyword regex misses (~69% of notes contain Thai; 127 are Thai-only).

    python scripts/run_thai_label_extension.py

A curated table (below) classifies every Thai-note fault candidate with an
English gloss and a tier:
  A = own-unit electrical symptom or damage (noise, high tan-delta, impedance
      anomaly, OLTC damage, post-HV-test problem)
  B = nearby-equipment event stressing the unit (breaker/CVT explosion,
      wildlife flashover) - consistent with the existing label, which already
      includes external "Fault at PEA/MEA" through-faults
  C = mechanical/operational stress (transport impact, sustained overload)
Chemistry-triggered follow-ups (e.g. "follow-up because C2H2 found") are
deliberately EXCLUDED: labelling them would be circular with the gas features.

The table is a human classification (Thai read by the assistant, to be
validated by the supervisor); it is versioned here so the label remains
reproducible. Outputs results/tables/{thai_note_classification.csv,
thai_label_extension_effect.csv}.
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dga.config import PROJECT_ROOT, load_config
from dga import data as dga_data, health

cfg = load_config()
TDIR = PROJECT_ROOT / "results" / "tables"
TDIR.mkdir(parents=True, exist_ok=True)
W = {"severity": 1.0, "acetylene": 2.0, "temporal": 1.0, "anomaly": 1.0}

# ---- curated Thai fault-candidate notes (exact string after .strip()) -------
THAI_FAULT_NOTES: dict[str, tuple[str, str]] = {
    # tier A - own-unit electrical symptom / damage
    "มีเสียงดังผิดปกติ": ("A", "abnormal loud noise from the transformer"),
    "หม้อแปลงมีเสียงดัง": ("A", "transformer making noise"),
    "ค่า p.f สูง": ("A", "high insulation power factor (tan delta)"),
    "ตามผล Single phase Impedance เกิน 2%": ("A", "single-phase impedance deviation > 2% (winding deformation indicator)"),
    "พบค่าทดสอบ Resistance ไม่เรียงตาม Tap": ("A", "winding resistance not monotonic across taps (winding/OLTC anomaly)"),
    "หม้อแปลงไม่ได้ใช้งาน วัดทางไฟฟ้าผิดปกติ": ("A", "out of service; abnormal electrical measurements"),
    "เนื่องจากมี OLTC เสียหาย": ("A", "OLTC damaged"),
    "หลังทดสอบ HV TEST แล้วมีปัญหา": ("A", "problem found after HV test"),
    # tier B - nearby-equipment event (through-fault / external stress)
    "ตามผลเนื่องจาก Bkr. ระเบิด": ("B", "follow-up after circuit-breaker explosion"),
    "CVT ระเบิด": ("B", "capacitive voltage transformer exploded"),
    "งูขึ้นใบมีด69kV SB-6955 phase a,b": ("B", "snake on 69 kV disconnector blades (flashover event)"),
    # tier C - mechanical / operational stress
    "ตามผล 3 เดือนหลังตกกระแทก": ("C", "follow-up 3 months after impact/drop damage"),
    "ตามผล 1 เดือนหลังตกกระแทก": ("C", "follow-up 1 month after impact/drop damage"),
    "ติดตามผลหม้อแปลงตกกระแทกระหว่างติดตั้ง": ("C", "transformer dropped/impacted during installation"),
    "จ่ายโหลดเกิน 100%": ("C", "sustained loading above 100% rating"),
}

# Deliberate exclusions (documented for the supervisor):
EXCLUDED_NOTES = {
    "ตามผล C2H2": "chemistry-triggered follow-up - circular with gas features",
    "ตามผลเนื่องจากพบC2H2": "chemistry-triggered follow-up - circular",
    "เนื่องจากมีหม้อแปลงตัวใกล้เคียงระเบิดเกิดไฟลุกไหม้MMM-KT3A": "neighbouring transformer exploded - surveillance trigger, not own event",
    "ไม่น่าจะเป็นข้อมูลที่ได้จากเครื่อง DGA ที่ตรงกับใบแนบ (มั่วมา)": "operator flags the record itself as bogus - data-quality note",
}


def unit_label(df: pd.DataFrame, sample_flag: pd.Series) -> pd.Series:
    return df.assign(_f=sample_flag).groupby("CODETX")["_f"].any()


def main():
    df = dga_data.load_clean()
    nb = df["NB"].astype("string").str.strip()

    tier = nb.map({k: t for k, (t, _) in THAI_FAULT_NOTES.items()})
    print("Thai fault-candidate samples matched:", int(tier.notna().sum()))
    for t in ("A", "B", "C"):
        m = tier == t
        print(f"  tier {t}: {int(m.sum())} samples on {df.loc[m, 'CODETX'].nunique()} units")

    # classification table for supervisor review
    rows = [{"note": k, "tier": t, "english_gloss": g,
             "n_samples": int((nb == k).sum()),
             "n_units": df.loc[nb == k, "CODETX"].nunique()}
            for k, (t, g) in THAI_FAULT_NOTES.items()]
    rows += [{"note": k, "tier": "excluded", "english_gloss": why,
              "n_samples": int((nb == k).sum()),
              "n_units": df.loc[nb == k, "CODETX"].nunique()}
             for k, why in EXCLUDED_NOTES.items()]
    pd.DataFrame(rows).to_csv(TDIR / "thai_note_classification.csv",
                              index=False, encoding="utf-8-sig")

    # label variants
    base = df["fault_note"].fillna(False).astype(bool)
    variants = {
        "original (EN regex)": base,
        "+Thai tier A": base | (tier == "A"),
        "+Thai A+B": base | tier.isin(["A", "B"]),
        "+Thai A+B+C": base | tier.isin(["A", "B", "C"]),
    }

    feats = health.assemble_features(df, cfg, seed=42)
    risk = health.risk_index(feats, W).feats["risk_score"]
    n = feats["n_samples"].astype(float)

    out = {}
    for name, flag in variants.items():
        y = unit_label(df, flag).reindex(feats.index).fillna(False).astype(int).to_numpy()
        out[name] = {
            "positive_units": int(y.sum()),
            "rate_%": round(100 * y.mean(), 1),
            "AUC_index": round(float(roc_auc_score(y, risk)), 3),
            "AUC_n_samples": round(float(roc_auc_score(y, n)), 3),
            "gap (n - index)": round(float(roc_auc_score(y, n) - roc_auc_score(y, risk)), 3),
        }
    tab = pd.DataFrame(out).T
    tab.to_csv(TDIR / "thai_label_extension_effect.csv")
    print("\nEffect of the Thai-extended label (unit level):")
    print(tab.to_string())
    print("\nwrote thai_note_classification.csv, thai_label_extension_effect.csv")


if __name__ == "__main__":
    main()
