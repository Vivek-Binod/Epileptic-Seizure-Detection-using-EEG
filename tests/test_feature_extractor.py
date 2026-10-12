"""Unit tests for EEGFeatureExtractor."""

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.feature_extractor import EEGFeatureExtractor


def test_feature_extractor():
    extractor = EEGFeatureExtractor()
    assert len(extractor.feature_names) == 252, f"Expected 252 feature names, got {len(extractor.feature_names)}"

    # Generate synthetic batch of 10 windows (10, 18, 1024)
    N, n_ch, n_samples = 10, 18, 1024
    rng = np.random.RandomState(42)
    synthetic_windows = rng.randn(N, n_ch, n_samples).astype(np.float32)

    features = extractor.extract_features(synthetic_windows)

    # 1. Shape and type checks
    assert features.shape == (10, 252), f"Expected shape (10, 252), got {features.shape}"
    assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"

    # 2. Finite checks
    assert not np.isnan(features).any(), "Found NaNs in feature matrix"
    assert not np.isinf(features).any(), "Found Infs in feature matrix"

    # 3. Check that relative power features and spectral entropy are bounded
    # Inspect a few columns by name
    for col_idx, name in enumerate(extractor.feature_names):
        col_vals = features[:, col_idx]
        if "power" in name:
            assert np.all(col_vals >= -1e-6) and np.all(col_vals <= 1.0 + 1e-6), (
                f"Relative power out of bounds for {name}: min={col_vals.min()}, max={col_vals.max()}"
            )
        if "spectral_entropy" in name:
            assert np.all(col_vals >= -1e-6) and np.all(col_vals <= 1.0 + 1e-6), (
                f"Spectral entropy out of bounds for {name}: min={col_vals.min()}, max={col_vals.max()}"
            )

    # 4. Check that for every channel, sum of relative band powers across the 5 bands equals 1.0
    for ch_idx, ch in enumerate(extractor.channels):
        band_cols = [
            extractor.feature_names.index(f"{ch}_{b}_power")
            for b in ["delta", "theta", "alpha", "beta", "gamma"]
        ]
        sum_band_powers = np.sum(features[:, band_cols], axis=1)
        assert np.allclose(sum_band_powers, 1.0, atol=1e-4), (
            f"Band powers do not sum to 1.0 for {ch}: {sum_band_powers}"
        )

    # 5. Robustness check on all-zero windows
    zero_windows = np.zeros((2, n_ch, n_samples), dtype=np.float32)
    zero_features = extractor.extract_features(zero_windows)
    assert not np.isnan(zero_features).any()
    assert not np.isinf(zero_features).any()

    print("test_feature_extractor PASSED successfully!")


if __name__ == "__main__":
    test_feature_extractor()
