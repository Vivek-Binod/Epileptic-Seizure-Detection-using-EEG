"""Temporal post-processing module for continuous EEG seizure detection.

Provides:
- Causal Exponentially Weighted Moving Average (EWMA) smoothing.
- Consecutive-window hysteresis / persistence filtering.
- Leak-free training-only parameter selection via nested Leave-One-Subject-Out CV.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score


def ewma_smooth(probabilities: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Apply causal exponentially weighted moving average (EWMA) to probability sequence.

    Formula:
        s[0] = p[0]
        s[t] = alpha * p[t] + (1 - alpha) * s[t-1]

    Args:
        probabilities: 1D array of predicted probabilities in chronological order.
        alpha: Smoothing factor in (0, 1]. alpha=1.0 corresponds to no smoothing.

    Returns:
        1D array of smoothed probabilities, same length and bounded in [0, 1].
    """
    p = np.asarray(probabilities, dtype=float)
    if len(p) == 0:
        return np.array([], dtype=float)
    if alpha >= 1.0:
        return np.clip(p, 0.0, 1.0)
    if alpha <= 0.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    smoothed = np.empty_like(p)
    smoothed[0] = p[0]
    for t in range(1, len(p)):
        smoothed[t] = alpha * p[t] + (1.0 - alpha) * smoothed[t - 1]

    return np.clip(smoothed, 0.0, 1.0)


def consecutive_window_persistence(
    binary_preds: np.ndarray,
    min_consecutive: int = 3,
) -> np.ndarray:
    """Apply persistence filter requiring K consecutive positive windows to sustain detection.

    Args:
        binary_preds: 1D binary array (0 or 1).
        min_consecutive: Minimum consecutive 1s required to activate alarm.

    Returns:
        Filtered binary array.
    """
    preds = np.asarray(binary_preds, dtype=int)
    out = np.zeros_like(preds)
    consec = 0

    for i, val in enumerate(preds):
        if val == 1:
            consec += 1
            if consec >= min_consecutive:
                out[i] = 1
        else:
            consec = 0

    return out


def tune_ewma_alpha_leak_free(
    inner_val_data: Dict[str, Dict],
    candidate_alphas: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
    criterion: str = "pr_auc",
) -> Tuple[float, Dict[float, float]]:
    """Select optimal EWMA alpha strictly using inner training-fold out-of-fold data.

    Args:
        inner_val_data: Dictionary mapping inner training subject IDs to:
            {
                'y_true': 1D array of ground truth labels,
                'raw_prob': 1D array of out-of-fold predicted probabilities,
            }
        candidate_alphas: List of alpha values to evaluate.
        criterion: Objective metric ('pr_auc' or 'f1').

    Returns:
        Tuple of (best_alpha, dict_of_alpha_to_mean_score).
    """
    alpha_scores: Dict[float, float] = {}

    for a in candidate_alphas:
        subj_scores = []
        for subj, d in inner_val_data.items():
            y_true = d["y_true"]
            raw_p = d["raw_prob"]
            p_sm = ewma_smooth(raw_p, alpha=a)

            if criterion == "pr_auc":
                score = float(average_precision_score(y_true, p_sm))
            elif criterion == "f1":
                y_pred = (p_sm >= 0.5).astype(int)
                score = float(f1_score(y_true, y_pred, zero_division=0))
            else:
                raise ValueError(f"Unknown criterion: {criterion}")

            subj_scores.append(score)

        alpha_scores[a] = float(np.mean(subj_scores))

    best_alpha = max(alpha_scores.keys(), key=lambda k: alpha_scores[k])
    return best_alpha, alpha_scores
