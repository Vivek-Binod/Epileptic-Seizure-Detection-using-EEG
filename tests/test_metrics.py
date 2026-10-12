"""Unit tests for evaluation metrics module."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import compute_event_metrics, compute_window_metrics


def test_compute_window_metrics():
    # 100 samples: 10 positives, 90 negatives
    y_true = np.zeros(100, dtype=int)
    y_true[:10] = 1

    # Perfect predictions
    y_prob_perf = np.zeros(100, dtype=float)
    y_prob_perf[:10] = 0.95
    y_prob_perf[10:] = 0.05

    m_perf = compute_window_metrics(y_true, y_prob_perf, threshold=0.5)
    assert m_perf["sensitivity"] == 1.0
    assert m_perf["specificity"] == 1.0
    assert m_perf["precision"] == 1.0
    assert m_perf["f1"] == 1.0
    assert m_perf["pr_auc"] == 1.0
    assert m_perf["roc_auc"] == 1.0
    assert m_perf["tp"] == 10
    assert m_perf["tn"] == 90
    assert m_perf["fp"] == 0
    assert m_perf["fn"] == 0

    # Imperfect predictions with 2 FPs and 2 FNs
    y_prob_imp = np.zeros(100, dtype=float)
    y_prob_imp[:8] = 0.8    # 8 TPs
    y_prob_imp[8:10] = 0.2  # 2 FNs
    y_prob_imp[10:12] = 0.7 # 2 FPs
    y_prob_imp[12:] = 0.1   # 88 TNs

    m_imp = compute_window_metrics(y_true, y_prob_imp, threshold=0.5)
    assert m_imp["tp"] == 8
    assert m_imp["fn"] == 2
    assert m_imp["fp"] == 2
    assert m_imp["tn"] == 88
    assert np.isclose(m_imp["sensitivity"], 0.8)
    assert np.isclose(m_imp["specificity"], 88.0 / 90.0)
    assert 0.0 < m_imp["pr_auc"] < 1.0
    print("test_compute_window_metrics PASSED!")


def test_compute_event_metrics():
    # Continuous recording of 3600 seconds (1.0 hour)
    total_duration_sec = 3600.0

    # 2 true seizures:
    # Seizure 1: 500s to 550s (50s duration)
    # Seizure 2: 2000s to 2040s (40s duration)
    annotated_seizures = [(500.0, 550.0), (2000.0, 2040.0)]

    # 1-second stride windows over 3600 seconds
    window_starts = np.arange(0, 3600, 1.0)
    window_preds = np.zeros(3600, dtype=int)

    # Simulate model predictions:
    # 1. Burst of 5 positive windows inside Seizure 1: t=510 to 514
    window_preds[510:515] = 1
    # 2. A second burst of 4 positive windows also inside Seizure 1: t=530 to 533
    window_preds[530:534] = 1

    # 3. False Alarm burst outside any seizure: t=1200 to 1203 (4 positive windows)
    window_preds[1200:1204] = 1

    # 4. Short burst (only 2 windows) inside Seizure 2: t=2010 to 2011 (should NOT trigger alarm with M=3)
    window_preds[2010:2012] = 1

    # Collar with M=3 consecutive windows
    metrics = compute_event_metrics(
        window_preds=window_preds,
        window_start_sec=window_starts,
        annotated_seizures_sec=annotated_seizures,
        total_duration_sec=total_duration_sec,
        collar_consecutive_windows=3,
        collar_tolerance_sec=10.0,
    )

    # Verification:
    # - Seizure 1 had 2 bursts >= 3. Due to single-count rule, Seizure 1 counts as 1 detected seizure!
    # - Seizure 2 only had 2 consecutive positive windows, so no alarm was triggered.
    #   Detected seizures = 1 out of 2.
    assert metrics["detected_seizures"] == 1
    assert metrics["total_seizures"] == 2
    assert np.isclose(metrics["event_sensitivity"], 0.5)

    # - False alarms: 1 false alarm at t=1200.
    assert metrics["false_alarms"] == 1
    # - Total hours = 1.0, so FP/h = 1.0.
    assert np.isclose(metrics["fp_per_hour"], 1.0)

    print("test_compute_event_metrics PASSED!")


if __name__ == "__main__":
    test_compute_window_metrics()
    test_compute_event_metrics()
    print("\nALL METRICS TESTS PASSED!")
