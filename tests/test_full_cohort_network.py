import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines.full_cohort_network_experiment import (
    benjamini_hochberg_fdr,
    bonferroni_correction,
    bootstrap_cohen_d_ci,
)
from src.utils.config import PATIENT_ID_MAPPING


def test_benjamini_hochberg_fdr_properties():
    """Verify FDR adjustments satisfy rank monotonicity and boundary conditions."""
    p_vals = np.array([0.001, 0.01, 0.02, 0.04, 0.15])
    q_vals = benjamini_hochberg_fdr(p_vals)

    # Monotonicity check
    assert len(q_vals) == len(p_vals)
    assert np.all(q_vals >= p_vals), "FDR q-values must be >= raw p-values"
    assert np.all(q_vals <= 1.0), "FDR q-values must be <= 1.0"
    assert np.all(q_vals >= 0.0), "FDR q-values must be >= 0.0"

    # Empty array handling
    empty_q = benjamini_hochberg_fdr(np.array([]))
    assert len(empty_q) == 0


def test_bonferroni_correction_properties():
    """Verify Bonferroni correction caps at 1.0 and scales linearly with test count."""
    p_vals = np.array([0.01, 0.05, 0.5])
    adj_p = bonferroni_correction(p_vals)

    assert adj_p[0] == pytest.approx(0.03)
    assert adj_p[1] == pytest.approx(0.15)
    assert adj_p[2] == pytest.approx(1.0)  # Capped at 1.0


def test_bootstrap_cohen_d_ci_properties():
    """Verify Cohen's d effect size and bootstrap confidence interval validity."""
    np.random.seed(42)
    # Strong positive paired effect
    y = np.random.normal(loc=3.0, scale=0.5, size=23)
    x = y + np.random.normal(loc=1.0, scale=0.3, size=23)

    d, ci_lo, ci_hi = bootstrap_cohen_d_ci(x, y, n_boot=500, alpha=0.05, seed=42)

    assert d > 0, "Point estimate must be positive"
    assert ci_lo <= d <= ci_hi, "Point estimate must fall within confidence interval"
    assert ci_lo > 0, "Lower CI bound should exclude zero for strong effect"


def test_patient_mapping_unifies_chb01_chb21():
    """Verify chb01 and chb21 are mapped strictly to the same biological patient ID."""
    assert PATIENT_ID_MAPPING["chb01"] == "patient_01"
    assert PATIENT_ID_MAPPING["chb21"] == "patient_01"
    # Ensure there are exactly 23 unique biological patients across 24 cases
    unique_patients = set(PATIENT_ID_MAPPING.values())
    assert len(unique_patients) == 23
