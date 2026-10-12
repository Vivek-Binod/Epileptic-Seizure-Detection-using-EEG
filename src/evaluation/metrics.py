"""Evaluation metrics module for CHB-MIT EEG Seizure Detection.

Provides both window-level and clinical event-level evaluation:
1. Window-Level Metrics:
   - PR-AUC (Average Precision Score): Primary metric under extreme class imbalance.
   - ROC-AUC: Literature baseline.
   - Sensitivity (Recall) & Specificity at specified decision threshold.
   - F1-Score & Precision for seizure class.
2. Event-Level Metrics:
   - Collar filter: Requires M consecutive positive windows to declare an alarm event.
   - Single-Count Rule: Matches alarms to ground-truth seizure events within +/- 10s tolerance.
   - Event Sensitivity: Detected Seizures / Total Annotated Seizures.
   - FP/h: False Positives per Hour of continuous EEG recording.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_window_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute comprehensive window-level classification metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1), shape (N,)
        y_prob: Predicted probabilities for the positive class (1), shape (N,)
        threshold: Decision threshold to convert probabilities to binary predictions.

    Returns:
        Dictionary of computed metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # Handle single-class cases safely
    has_both_classes = len(np.unique(y_true)) > 1

    pr_auc = float(average_precision_score(y_true, y_prob)) if has_both_classes else 0.0
    roc_auc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else 0.5

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "total_windows": len(y_true),
        "pos_windows": int(np.sum(y_true == 1)),
        "neg_windows": int(np.sum(y_true == 0)),
    }


def compute_event_metrics(
    window_preds: np.ndarray,
    window_start_sec: np.ndarray,
    annotated_seizures_sec: List[Tuple[float, float]],
    total_duration_sec: float,
    collar_consecutive_windows: int = 3,
    collar_tolerance_sec: float = 10.0,
) -> Dict[str, Union[float, int]]:
    """Compute clinical event-level seizure detection metrics.

    Args:
        window_preds: Binary window predictions (0 or 1) in chronological order.
        window_start_sec: Start time in seconds for each window.
        annotated_seizures_sec: List of (start_sec, end_sec) for annotated seizures.
        total_duration_sec: Total duration of the evaluated recording(s) in seconds.
        collar_consecutive_windows: Number of consecutive positive windows required
                                     to trigger an alarm (hysteresis filter).
        collar_tolerance_sec: Tolerance window around seizure boundaries (+/- sec).

    Returns:
        Dictionary with event-level sensitivity, false alarms, and FP/h.
    """
    window_preds = np.asarray(window_preds, dtype=int)
    window_start_sec = np.asarray(window_start_sec, dtype=float)

    # 1. Collar Filter: find alarm onset timestamps
    alarm_times_sec: List[float] = []
    consecutive = 0

    for pred, t_start in zip(window_preds, window_start_sec):
        if pred == 1:
            consecutive += 1
            if consecutive == collar_consecutive_windows:
                # Alarm triggered at this window's onset
                alarm_times_sec.append(float(t_start))
        else:
            consecutive = 0

    # 2. Match alarms to annotated seizure intervals
    n_seizures = len(annotated_seizures_sec)
    detected_seizures = set()
    false_alarms_count = 0

    for alarm_t in alarm_times_sec:
        matched_seizure_idx = None
        for s_idx, (s_start, s_end) in enumerate(annotated_seizures_sec):
            # Alarm matches if within [start - tolerance, end + tolerance]
            if (s_start - collar_tolerance_sec) <= alarm_t <= (s_end + collar_tolerance_sec):
                matched_seizure_idx = s_idx
                break

        if matched_seizure_idx is not None:
            # Single-count rule: marks this seizure as detected
            detected_seizures.add(matched_seizure_idx)
        else:
            false_alarms_count += 1

    total_hours = total_duration_sec / 3600.0 if total_duration_sec > 0 else 1.0
    event_sensitivity = len(detected_seizures) / float(n_seizures) if n_seizures > 0 else 1.0
    fp_per_hour = false_alarms_count / total_hours

    return {
        "event_sensitivity": float(event_sensitivity),
        "detected_seizures": len(detected_seizures),
        "total_seizures": n_seizures,
        "false_alarms": int(false_alarms_count),
        "total_alarms": len(alarm_times_sec),
        "total_hours": float(total_hours),
        "fp_per_hour": float(fp_per_hour),
    }
