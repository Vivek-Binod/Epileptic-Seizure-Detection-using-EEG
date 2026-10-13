"""Unit and regression tests for FeatureSelector module."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.selection import FeatureSelector


def test_feature_selector_basic():
    # Synthetic data: 100 samples, 10 features
    np.random.seed(42)
    X = np.random.randn(100, 10)
    # Feature 2 and Feature 5 strongly correlated with y
    y = (X[:, 2] + 2.0 * X[:, 5] > 0.5).astype(int)

    feature_names = [f"feat_{i}" for i in range(10)]
    feature_names[2] = "CZ-PZ_variance"
    feature_names[5] = "nodal_plv_beta_FP1-F7"

    selector = FeatureSelector(method="mutual_info", k=4, random_state=42)
    selector.fit(X, y, feature_names=feature_names)

    assert selector.is_fitted
    assert len(selector.selected_indices_) == 4
    assert len(selector.selected_features_) == 4

    # Top features must include feat 5 or feat 2
    assert (2 in selector.selected_indices_) or (5 in selector.selected_indices_)

    # Transform
    X_trans = selector.transform(X)
    assert X_trans.shape == (100, 4)
    np.testing.assert_allclose(X_trans[:, 0], X[:, selector.selected_indices_[0]])


def test_feature_selector_family_breakdown():
    feature_names = [
        "CZ-PZ_variance",
        "FZ-CZ_rms",
        "nodal_plv_beta_FP1-F7",
        "global_mean_pli_theta",
        "P3-O1_line_length",
    ]
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, size=50)

    selector = FeatureSelector(method="mutual_info", k=3, random_state=42)
    selector.fit(X, y, feature_names=feature_names)

    breakdown = selector.get_family_breakdown()
    assert breakdown["total_selected"] == 3
    assert (
        breakdown["conventional_count"]
        + breakdown["plv_count"]
        + breakdown["pli_count"]
        == 3
    )
    assert breakdown["connectivity_count"] == breakdown["plv_count"] + breakdown["pli_count"]
    assert np.isclose(breakdown["conventional_pct"] + breakdown["connectivity_pct"], 100.0)


def test_feature_selector_leakage_isolation():
    # Verify selector fitted on X_train does not inspect X_test
    X_train = np.random.randn(80, 6)
    y_train = np.random.randint(0, 2, size=80)
    X_test = np.random.randn(20, 6)

    selector = FeatureSelector(method="mutual_info", k=2, random_state=42)
    selector.fit(X_train, y_train)

    X_test_trans = selector.transform(X_test)
    assert X_test_trans.shape == (20, 2)
    # Selected indices must match train-fit exactly
    np.testing.assert_allclose(X_test_trans[:, 0], X_test[:, selector.selected_indices_[0]])


def test_feature_selector_reproducibility():
    """Verify deterministic behavior across identical seeds."""
    np.random.seed(123)
    X = np.random.randn(100, 8)
    y = np.random.randint(0, 2, size=100)
    names = [f"f_{i}" for i in range(8)]

    sel1 = FeatureSelector(method="mutual_info", k=4, random_state=99)
    sel1.fit(X, y, feature_names=names)

    sel2 = FeatureSelector(method="mutual_info", k=4, random_state=99)
    sel2.fit(X, y, feature_names=names)

    np.testing.assert_array_equal(sel1.selected_indices_, sel2.selected_indices_)
    np.testing.assert_allclose(sel1.feature_scores_, sel2.feature_scores_)
    assert sel1.selected_features_ == sel2.selected_features_


def test_feature_selector_set_k_consistency():
    """Verify that set_k(k) matches a fresh selector fitted with k directly."""
    np.random.seed(456)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, size=100)
    names = [f"feat_{i}" for i in range(10)]

    # Fit with k=8
    sel_large = FeatureSelector(method="mutual_info", k=8, random_state=42)
    sel_large.fit(X, y, feature_names=names)

    # Slice to k=4 using set_k
    sel_large.set_k(4)
    indices_via_set_k = sel_large.selected_indices_.copy()

    # Direct fit with k=4
    sel_direct = FeatureSelector(method="mutual_info", k=4, random_state=42)
    sel_direct.fit(X, y, feature_names=names)
    indices_direct = sel_direct.selected_indices_.copy()

    np.testing.assert_array_equal(indices_via_set_k, indices_direct)
    assert sel_large.selected_features_ == sel_direct.selected_features_


def test_feature_selector_dataframe_and_errors():
    """Verify DataFrame input handling and input error checks."""
    df_X = pd.DataFrame(
        np.random.randn(50, 4),
        columns=["c1_var", "c2_rms", "nodal_plv_alpha", "pli_beta"],
    )
    y = np.random.randint(0, 2, size=50)

    # Test error before fit
    unfitted = FeatureSelector(method="mutual_info", k=2)
    with pytest.raises(RuntimeError):
        unfitted.transform(df_X.values)
    with pytest.raises(RuntimeError):
        unfitted.get_family_breakdown()
    with pytest.raises(RuntimeError):
        unfitted.set_k(3)

    # Test invalid method
    with pytest.raises(ValueError):
        FeatureSelector(method="invalid_method")

    # Test invalid k
    with pytest.raises(ValueError):
        FeatureSelector(method="mutual_info", k=-1)

    # Test valid fit on DataFrame values with feature names
    sel = FeatureSelector(method="mutual_info", k=2, random_state=42)
    sel.fit(df_X.values, y, feature_names=list(df_X.columns))
    assert sel.is_fitted
    summary_df = sel.get_selected_features_summary()
    assert len(summary_df) == 2
    assert "category" in summary_df.columns
