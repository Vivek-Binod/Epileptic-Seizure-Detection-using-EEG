"""Unit and regression tests for Full Cohort pipeline and patient grouping."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splitters import PatientGroupedSplitter
from src.utils.config import PATIENT_ID_MAPPING
from src.pipelines.full_cohort_experiment import SUBJECT_PAIRS, FullCohortExperiment


def test_full_cohort_patient_mapping_and_counts():
    """Verify all 24 cases map to exactly 23 unique biological patients."""
    assert len(SUBJECT_PAIRS) == 24
    patients = set(PATIENT_ID_MAPPING[c] for c in SUBJECT_PAIRS.keys())
    assert len(patients) == 23
    assert PATIENT_ID_MAPPING["chb01"] == "patient_01"
    assert PATIENT_ID_MAPPING["chb21"] == "patient_01"


def test_full_cohort_zero_leakage_splitter():
    """Verify that Leave-One-Patient-Out cross-validation NEVER splits patient_01."""
    records = []
    for c in SUBJECT_PAIRS.keys():
        records.append({"case_id": c, "patient_id": PATIENT_ID_MAPPING[c]})
    df = pd.DataFrame(records)

    splitter = PatientGroupedSplitter(patient_mapping=PATIENT_ID_MAPPING)
    splits = list(splitter.get_loso_splits(df))

    # Must be 23 splits (one for each biological patient)
    assert len(splits) == 23

    # For each split, verify zero patient overlap
    for tr, te, held_out in splits:
        tr_patients = set(tr["patient_id"].unique())
        te_patients = set(te["patient_id"].unique())
        assert len(te_patients) == 1
        assert held_out in te_patients
        assert held_out not in tr_patients
        assert len(tr_patients.intersection(te_patients)) == 0

        # Special verification for patient_01
        if held_out == "patient_01":
            assert set(te["case_id"].unique()) == {"chb01", "chb21"}
            assert "chb01" not in tr["case_id"].values
            assert "chb21" not in tr["case_id"].values
        else:
            assert "chb01" in tr["case_id"].values
            assert "chb21" in tr["case_id"].values
