"""Regression tests for CNN benchmark forensic audit.

Tests:
1. Exact mathematical distinction between Macro Event Sensitivity and Pooled Event Sensitivity.
2. Reconciliation of baseline full-cohort metrics against ground-truth CSV artifacts.
3. Patient grouping, chb01/chb21 unification, and zero-leakage cross-validation folds.
4. False positive rate calculations and evaluation duration across the cohort.
5. Checkpoint weight reloading and parameter architecture integrity.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CNN_SUMMARY_CSV = PROJECT_ROOT / "outputs/experiments/exp_20261013_deep_learning_cnn/cnn_summary.csv"
CNN_CASE_CSV = PROJECT_ROOT / "outputs/experiments/exp_20261013_deep_learning_cnn/cnn_case_folds.csv"
CNN_PATIENT_CSV = PROJECT_ROOT / "outputs/experiments/exp_20261013_deep_learning_cnn/cnn_patient_folds.csv"
FULL_COHORT_SUMMARY_CSV = PROJECT_ROOT / "outputs/experiments/exp_20261013_full_cohort/full_cohort_summary.csv"
CHECKPOINTS_DIR = PROJECT_ROOT / "outputs/experiments/exp_20261013_deep_learning_cnn/checkpoints"


def test_event_sensitivity_macro_vs_pooled_distinction():
    """Verify that macro-case sensitivity and pooled event sensitivity are distinct and mathematically exact."""
    assert CNN_SUMMARY_CSV.exists(), f"Missing {CNN_SUMMARY_CSV}"
    df = pd.read_csv(CNN_SUMMARY_CSV)

    raw_row = df[df["config_id"] == "cnn_1d_raw"].iloc[0]
    ewma_row = df[df["config_id"] == "cnn_1d_ewma"].iloc[0]

    # Raw metrics
    raw_macro_sens = float(raw_row["macro_event_sensitivity"])
    raw_det = int(raw_row["pooled_detected_seizures"])
    raw_tot = int(raw_row["pooled_total_seizures"])
    raw_pooled_sens = raw_det / raw_tot

    assert np.isclose(raw_macro_sens, 0.8125, atol=1e-4), f"Raw macro sens mismatch: {raw_macro_sens}"
    assert raw_det == 21
    assert raw_tot == 28
    assert np.isclose(raw_pooled_sens, 0.7500, atol=1e-4), f"Raw pooled sens mismatch: {raw_pooled_sens}"
    # Macro must NOT equal pooled count ratio
    assert not np.isclose(raw_macro_sens, raw_pooled_sens)

    # EWMA metrics
    ewma_macro_sens = float(ewma_row["macro_event_sensitivity"])
    ewma_det = int(ewma_row["pooled_detected_seizures"])
    ewma_tot = int(ewma_row["pooled_total_seizures"])
    ewma_pooled_sens = ewma_det / ewma_tot

    assert np.isclose(ewma_macro_sens, 18.5 / 24, atol=1e-4), f"EWMA macro sens mismatch: {ewma_macro_sens}"
    assert ewma_det == 20
    assert ewma_tot == 28
    assert np.isclose(ewma_pooled_sens, 20 / 28, atol=1e-4), f"EWMA pooled sens mismatch: {ewma_pooled_sens}"
    assert not np.isclose(ewma_macro_sens, ewma_pooled_sens)


def test_baseline_full_cohort_metrics_integrity():
    """Verify ground truth baseline numbers from full_cohort_summary.csv (preventing historical memo copy errors)."""
    assert FULL_COHORT_SUMMARY_CSV.exists(), f"Missing {FULL_COHORT_SUMMARY_CSV}"
    df_fc = pd.read_csv(FULL_COHORT_SUMMARY_CSV)

    conv_row = df_fc[df_fc["config_id"] == "conventional"].iloc[0]
    # Authoritative Conventional baseline is 0.4972, NOT 0.5504 (which was the 10-subject sprint)
    assert np.isclose(float(conv_row["macro_case_pr_auc"]), 0.4972, atol=1e-3)
    assert not np.isclose(float(conv_row["macro_case_pr_auc"]), 0.5504, atol=1e-2)

    # Verified CNN + EWMA outperforms Conventional RF baseline on full cohort
    df_cnn = pd.read_csv(CNN_SUMMARY_CSV)
    cnn_ewma_pr = float(df_cnn[df_cnn["config_id"] == "cnn_1d_ewma"]["macro_case_pr_auc"].iloc[0])
    conv_pr = float(conv_row["macro_case_pr_auc"])

    assert cnn_ewma_pr > conv_pr, f"CNN EWMA PR-AUC ({cnn_ewma_pr}) should exceed Conventional ({conv_pr})"
    assert np.isclose(cnn_ewma_pr, 0.5571, atol=1e-3)


def test_case_folds_patient_01_unification():
    """Verify that chb01 and chb21 are mapped to patient_01 in case folds with no leakage."""
    assert CNN_CASE_CSV.exists(), f"Missing {CNN_CASE_CSV}"
    df_case = pd.read_csv(CNN_CASE_CSV)

    chb01_rows = df_case[df_case["case_id"] == "chb01"]
    chb21_rows = df_case[df_case["case_id"] == "chb21"]

    assert len(chb01_rows) == 2  # unsmoothed + ewma
    assert len(chb21_rows) == 2

    assert all(chb01_rows["patient_id"] == "patient_01")
    assert all(chb21_rows["patient_id"] == "patient_01")
    assert all(chb01_rows["fold_idx"] == 1)
    assert all(chb21_rows["fold_idx"] == 1)

    # Verify all 24 cases are present and account for 23 unique biological patients
    assert len(df_case["case_id"].unique()) == 24
    assert len(df_case["patient_id"].unique()) == 23


def test_false_alarm_rate_calculations():
    """Verify pooled and macro false alarm rate arithmetic and exact evaluation hours."""
    df_case = pd.read_csv(CNN_CASE_CSV)
    df_summary = pd.read_csv(CNN_SUMMARY_CSV)

    raw_cases = df_case[df_case["postprocessing"] == "unsmoothed"]
    total_hours = raw_cases["duration_hours"].sum()
    assert np.isclose(total_hours, 35.990, atol=1e-3)

    raw_fa_sum = raw_cases["false_alarms"].sum()
    assert raw_fa_sum == 975
    expected_pooled_fp_h = raw_fa_sum / total_hours
    assert np.isclose(expected_pooled_fp_h, 27.090859, atol=1e-4)

    summary_raw = df_summary[df_summary["config_id"] == "cnn_1d_raw"].iloc[0]
    assert np.isclose(float(summary_raw["pooled_fp_per_hour"]), expected_pooled_fp_h, atol=1e-4)
    assert np.isclose(float(summary_raw["macro_fp_per_hour"]), raw_cases["fp_per_hour"].mean(), atol=1e-4)

    ewma_cases = df_case[df_case["postprocessing"] == "ewma_0.30"]
    ewma_fa_sum = ewma_cases["false_alarms"].sum()
    assert ewma_fa_sum == 641
    expected_ewma_pooled_fp_h = ewma_fa_sum / total_hours
    assert np.isclose(expected_ewma_pooled_fp_h, 17.810503, atol=1e-4)

    summary_ewma = df_summary[df_summary["config_id"] == "cnn_1d_ewma"].iloc[0]
    assert np.isclose(float(summary_ewma["pooled_fp_per_hour"]), expected_ewma_pooled_fp_h, atol=1e-4)
    assert np.isclose(float(summary_ewma["macro_fp_per_hour"]), ewma_cases["fp_per_hour"].mean(), atol=1e-4)


def test_all_23_checkpoints_exist_and_loadable():
    """Verify that all 23 patient fold checkpoints exist and are loadable."""
    try:
        import torch
        from src.models.cnn1d import CompactEEGCNN
    except ImportError:
        pytest.skip("PyTorch not installed")

    unique_patients = [f"patient_{i:02d}" for i in range(1, 25) if i != 21]
    assert len(unique_patients) == 23

    for p in unique_patients:
        ckpt_path = CHECKPOINTS_DIR / f"best_model_fold_{p}.pt"
        assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path.name}"

    # Load fold 1 model
    model = CompactEEGCNN(in_channels=18, seq_len=1024, base_filters=16, dropout_rate=0.25)
    ckpt = torch.load(CHECKPOINTS_DIR / "best_model_fold_patient_01.pt", map_location="cpu")
    model.load_state_dict(ckpt)
    assert model.count_parameters() == 45409
