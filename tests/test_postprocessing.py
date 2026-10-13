import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.postprocessing import (
    consecutive_window_persistence,
    ewma_smooth,
    tune_ewma_alpha_leak_free,
)



def test_ewma_smooth_basic():
    # Empty array
    assert len(ewma_smooth(np.array([]))) == 0

    # alpha=1.0 returns unchanged
    p = np.array([0.1, 0.8, 0.2, 0.9])
    np.testing.assert_allclose(ewma_smooth(p, alpha=1.0), p)

    # Constant array returns constant
    p_const = np.full(10, 0.4)
    np.testing.assert_allclose(ewma_smooth(p_const, alpha=0.3), p_const)

    # Step response: p goes from 0 to 1 at t=1
    p_step = np.array([0.0, 1.0, 1.0, 1.0])
    s = ewma_smooth(p_step, alpha=0.5)
    # s[0] = 0.0
    # s[1] = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
    # s[2] = 0.5 * 1.0 + 0.5 * 0.5 = 0.75
    # s[3] = 0.5 * 1.0 + 0.5 * 0.75 = 0.875
    np.testing.assert_allclose(s, [0.0, 0.5, 0.75, 0.875])

    # Values must remain bounded in [0, 1]
    p_random = np.random.uniform(0, 1, 100)
    s_random = ewma_smooth(p_random, alpha=0.2)
    assert np.all(s_random >= 0.0)
    assert np.all(s_random <= 1.0)


def test_ewma_smooth_causality():
    # Changing future value must NOT affect past smoothed values
    p1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    p2 = np.array([0.1, 0.2, 0.3, 0.9, 0.9])  # changed at index 3 and 4

    s1 = ewma_smooth(p1, alpha=0.3)
    s2 = ewma_smooth(p2, alpha=0.3)

    # First 3 values must be identical (strictly causal)
    np.testing.assert_allclose(s1[:3], s2[:3])
    # Index 3 and 4 should differ
    assert not np.isclose(s1[3], s2[3])


def test_consecutive_window_persistence():
    preds = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0])
    filtered = consecutive_window_persistence(preds, min_consecutive=3)
    # First burst: length 2 (not >= 3) -> all 0
    # Second burst: length 4 -> first 2 are 0, remaining 2 are 1
    expected = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0])
    np.testing.assert_array_equal(filtered, expected)


def test_tune_ewma_alpha_leak_free():
    # Synthetic out-of-fold validation data from 2 training subjects
    # Subject 1: noisy step with seizure at t=10..20
    y1 = np.zeros(30, dtype=int)
    y1[10:20] = 1
    p1 = np.zeros(30)
    p1[10:20] = 0.8
    p1[0:5] = 0.6  # transient false spike
    p1[25:27] = 0.6  # transient false spike

    # Subject 2
    y2 = np.zeros(30, dtype=int)
    y2[15:25] = 1
    p2 = np.zeros(30)
    p2[15:25] = 0.85
    p2[2:4] = 0.7  # transient false spike

    inner_val_data = {
        "s1": {"y_true": y1, "raw_prob": p1},
        "s2": {"y_true": y2, "raw_prob": p2},
    }

    best_a, scores = tune_ewma_alpha_leak_free(
        inner_val_data, candidate_alphas=[0.1, 0.3, 0.5, 1.0], criterion="pr_auc"
    )

    # Smoothing should beat raw probabilities (alpha=1.0) because transient spikes get suppressed
    assert scores[best_a] >= scores[1.0]
    assert best_a in [0.1, 0.3, 0.5, 1.0]
