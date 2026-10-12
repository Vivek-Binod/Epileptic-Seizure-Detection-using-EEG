"""Unit tests for BaselineClassifier."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.baseline import BaselineClassifier


def test_baseline_classifier_models():
    # Synthetic dataset: 100 samples, 20 features
    rng = np.random.RandomState(42)
    X = rng.randn(100, 20).astype(np.float32)
    y = np.zeros(100, dtype=int)
    y[:20] = 1  # 20 positives, 80 negatives

    # Split 70/30
    X_train, X_test = X[:70], X[70:]
    y_train, y_test = y[:70], y[70:]

    for model_type in ["rf", "lgbm", "lr"]:
        clf = BaselineClassifier(model_type=model_type, random_state=42)
        clf.fit(X_train, y_train)

        # Predict proba
        proba = clf.predict_proba(X_test)
        assert proba.shape == (30,), f"Expected shape (30,), got {proba.shape} for {model_type}"
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

        # Predict binary
        pred = clf.predict(X_test, threshold=0.5)
        assert pred.shape == (30,)
        assert set(np.unique(pred)).issubset({0, 1})

        # Feature importances
        feat_names = [f"feat_{i}" for i in range(20)]
        fi_df = clf.get_feature_importances(feature_names=feat_names)
        assert fi_df is not None
        assert len(fi_df) == 20
        assert "feature" in fi_df.columns and "importance" in fi_df.columns
        assert fi_df["importance"].is_monotonic_decreasing

        print(f"BaselineClassifier ({model_type}) verified successfully!")


def test_scaler_encapsulation():
    """Verify that scaler mean and variance match X_train and do not touch X_test."""
    rng = np.random.RandomState(42)
    # Train data with mean 10.0, std 2.0
    X_train = rng.randn(100, 5).astype(np.float32) * 2.0 + 10.0
    y_train = rng.randint(0, 2, size=100)

    # Test data with totally different mean 100.0, std 20.0
    X_test = rng.randn(30, 5).astype(np.float32) * 20.0 + 100.0

    clf = BaselineClassifier(model_type="lr")
    clf.fit(X_train, y_train)

    # Scaler mean must match X_train mean
    assert np.allclose(clf.scaler.mean_, np.mean(X_train, axis=0), atol=1e-4)
    # Scaler must not reflect X_test mean
    assert not np.allclose(clf.scaler.mean_, np.mean(X_test, axis=0), atol=1.0)
    print("test_scaler_encapsulation PASSED!")


if __name__ == "__main__":
    test_baseline_classifier_models()
    test_scaler_encapsulation()
    print("\nALL BASELINE CLASSIFIER TESTS PASSED!")
