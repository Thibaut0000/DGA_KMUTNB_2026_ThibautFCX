"""Smoke tests: fast invariants of the dga package (no training, no figures).

    pytest tests/ -q

Data-dependent tests are skipped automatically when the raw dataset is absent
(it is confidential and not in version control).
"""
import numpy as np
import pandas as pd
import pytest

from dga.config import PROJECT_ROOT, load_config
from dga import conventional, evaluation, temporal
from dga.compositional import clr, multiplicative_replacement

RAW = PROJECT_ROOT / "data" / "raw" / "dga_main_tank.xlsx"
needs_data = pytest.mark.skipif(not RAW.exists(), reason="raw dataset not available")


def test_config_loads():
    cfg = load_config()
    assert cfg.seed == 42
    assert list(cfg.compositional.gases) == ["H2", "CH4", "C2H2", "C2H4", "C2H6"]
    assert cfg.sdcae.latent_dim == 2


def test_duval_known_cases():
    assert conventional.duval_triangle_1(98, 1, 1) == "PD"     # methane-dominant
    assert conventional.duval_triangle_1(50, 10, 40) == "D1"   # high C2H2, low C2H4
    assert conventional.duval_triangle_1(20, 75, 5) == "T3"    # C2H4-dominant thermal
    assert conventional.duval_triangle_1(90, 8, 2) == "T1"     # low-temp thermal
    assert conventional.duval_triangle_1(0, 0, 0) is None      # no gas, no diagnosis


def test_iec_partial_discharge():
    assert conventional.iec_60599(h2=100, ch4=1, c2h2=0, c2h4=1, c2h6=10) == "PD"


def test_clr_rows_sum_to_zero():
    P = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    C = clr(P)
    assert np.allclose(C.sum(axis=1), 0.0, atol=1e-9)


def test_clr_scale_invariance():
    x = np.array([[10.0, 20.0, 70.0]])
    p1 = x / x.sum()
    p2 = (100 * x) / (100 * x).sum()
    assert np.allclose(clr(p1), clr(p2))


def test_multiplicative_replacement_keeps_simplex():
    P = np.array([[0.0, 0.4, 0.6], [0.0, 0.0, 1.0]])
    R = multiplicative_replacement(P, delta=1e-3)
    assert (R > 0).all()
    assert np.allclose(R.sum(axis=1), 1.0)


def test_generation_rate_linear_trend():
    days = np.array([0, 182, 365, 548, 730])
    values = 10 + 20 * days / 365.25          # +20 ppm/year exactly
    rate = temporal.generation_rate(days, values)
    assert rate == pytest.approx(20.0, rel=1e-6)


def test_generation_rate_needs_min_points():
    assert np.isnan(temporal.generation_rate(np.array([0, 100]), np.array([1.0, 2.0])))


def test_external_metrics_perfect_agreement():
    labels = np.array([0, 1] * 20)
    truth = pd.Series(["a", "b"] * 20)
    m = evaluation.external_metrics(labels, truth)
    assert m["ARI"] == pytest.approx(1.0)


@needs_data
def test_dataset_loads_with_expected_shape():
    from dga import data as dga_data
    df = dga_data.load_clean()
    assert len(df) >= 4500                      # 4,563 samples expected
    assert df["CODETX"].nunique() >= 600        # 628 units expected
    for col in ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2"]:
        assert pd.api.types.is_numeric_dtype(df[col])
    assert bool(df["fault_note"].any())
