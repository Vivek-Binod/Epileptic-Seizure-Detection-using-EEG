"""Unit tests for leakage-safe cross-validation splitters."""

import sys
from pathlib import Path
import pandas as pd
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splitters import PatientGroupedSplitter, WithinPatientChronologicalSplitter
from src.utils.config import DATA_DIR


@pytest.fixture
def manifest_recordings():
    parquet_path = DATA_DIR / "recordings_manifest.parquet"
    assert parquet_path.exists(), f"Manifest file missing: {parquet_path}"
    return pd.read_parquet(parquet_path)


def test_loso_splits_and_patient_identity(manifest_recordings):
    splitter = PatientGroupedSplitter()
    splits = list(splitter.get_loso_splits(manifest_recordings))

    # Total biological patients must be exactly 23
    assert len(splits) == 23, f"Expected 23 LOSO splits, got {len(splits)}"

    # Check each fold for zero patient leakage
    for train_df, test_df, held_out in splits:
        train_patients = set(train_df["patient_id"])
        test_patients = set(test_df["patient_id"])
        assert train_patients.isdisjoint(test_patients), f"Patient leakage in fold for {held_out}"
        assert len(test_patients) == 1
        assert list(test_patients)[0] == held_out

        # Verify chb01 and chb21 are NEVER split
        train_cases = set(train_df["case_id"])
        test_cases = set(test_df["case_id"])

        if held_out == "patient_01":
            assert "chb01" in test_cases and "chb21" in test_cases
            assert "chb01" not in train_cases and "chb21" not in train_cases
        else:
            assert "chb01" in train_cases and "chb21" in train_cases
            assert "chb01" not in test_cases and "chb21" not in test_cases

    print("test_loso_splits_and_patient_identity PASSED!")


def test_group_kfold_splits(manifest_recordings):
    splitter = PatientGroupedSplitter()
    splits = list(splitter.get_group_kfold_splits(manifest_recordings, n_splits=5))

    assert len(splits) == 5, f"Expected 5 folds, got {len(splits)}"

    all_test_patients = set()
    for train_df, test_df, fold_idx in splits:
        train_patients = set(train_df["patient_id"])
        test_patients = set(test_df["patient_id"])
        assert train_patients.isdisjoint(test_patients), f"Patient leakage in fold {fold_idx}"

        # chb01 and chb21 check
        train_cases = set(train_df["case_id"])
        test_cases = set(test_df["case_id"])
        both_in_train = ("chb01" in train_cases) and ("chb21" in train_cases)
        both_in_test = ("chb01" in test_cases) and ("chb21" in test_cases)
        assert both_in_train or both_in_test, f"chb01 and chb21 separated in fold {fold_idx}!"

        all_test_patients.update(test_patients)

    # All 23 patients must be evaluated across the 5 folds
    assert len(all_test_patients) == 23
    print("test_group_kfold_splits PASSED!")


def test_leakage_detection_catches_intentional_leak(manifest_recordings):
    splitter = PatientGroupedSplitter()
    # Create synthetic intentional leakage: put chb01 in train and chb21 in test
    bad_train = manifest_recordings[manifest_recordings["case_id"] != "chb21"].copy()
    bad_test = manifest_recordings[manifest_recordings["case_id"] == "chb21"].copy()

    with pytest.raises(ValueError, match="IDENTITY LEAKAGE|PATIENT LEAKAGE"):
        splitter.validate_patient_split(bad_train, bad_test)
    print("test_leakage_detection_catches_intentional_leak PASSED!")


def test_within_patient_chronological_splitter(manifest_recordings):
    chb01_df = manifest_recordings[manifest_recordings["case_id"] == "chb01"].copy()
    splitter = WithinPatientChronologicalSplitter(train_ratio=0.70)
    train_df, test_df = splitter.get_chronological_split(chb01_df)

    assert len(train_df) + len(test_df) == len(chb01_df)
    # Check chronological order (filenames are strictly earlier in train)
    max_train_file = train_df["file_name"].max()
    min_test_file = test_df["file_name"].min()
    assert max_train_file < min_test_file, f"Train {max_train_file} not before Test {min_test_file}"
    print("test_within_patient_chronological_splitter PASSED!")


if __name__ == "__main__":
    rec = pd.read_parquet(DATA_DIR / "recordings_manifest.parquet")
    test_loso_splits_and_patient_identity(rec)
    test_group_kfold_splits(rec)
    test_leakage_detection_catches_intentional_leak(rec)
    test_within_patient_chronological_splitter(rec)
    print("\nALL SPLITTER TESTS PASSED!")
