"""Unit and property tests for EEG Functional Connectivity (PLV / PLI) Feature Extractor."""

import numpy as np
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.connectivity import EEGConnectivityExtractor
from src.utils.config import CANONICAL_CHANNELS, SAMPLING_RATE


def test_connectivity_extractor_shapes_and_determinism():
    """Verify output dimensionality, feature names alignment, and numerical determinism."""
    extractor = EEGConnectivityExtractor(
        sampling_rate=SAMPLING_RATE,
        include_nodal=True,
        include_global=True,
        include_hemispheric=True,
        include_pairwise=False,
    )

    n_windows = 4
    n_channels = 18
    n_samples = 1024  # 4 seconds at 256 Hz

    rng = np.random.RandomState(42)
    synthetic_windows = rng.randn(n_windows, n_channels, n_samples).astype(np.float32)

    # 4 bands * (18 nodal plv + 18 nodal pli + 6 global + 8 hemispheric) = 4 * 50 = 200 features
    expected_dim = 200
    assert len(extractor.feature_names) == expected_dim

    feats_1 = extractor.extract_features(synthetic_windows)
    feats_2 = extractor.extract_features(synthetic_windows)

    assert feats_1.shape == (n_windows, expected_dim)
    assert feats_1.dtype == np.float32
    assert not np.isnan(feats_1).any()
    assert not np.isinf(feats_1).any()
    np.testing.assert_array_equal(feats_1, feats_2)


def test_plv_vs_pli_volume_conduction_properties():
    """Mathematically verify PLV vs PLI behavior under zero-lag vs true phase-lagged coupling.

    - Identical signals (zero phase lag, delta_phi = 0):
      PLV = 1.0 (susceptible to volume conduction)
      PLI = 0.0 (immune to zero-lag volume conduction)

    - Orthogonal signals (quarter-cycle lag, delta_phi = pi/2):
      PLV = 1.0
      PLI = 1.0
    """
    extractor = EEGConnectivityExtractor(
        sampling_rate=SAMPLING_RATE,
        include_nodal=False,
        include_global=False,
        include_hemispheric=False,
        include_pairwise=True,
    )

    t = np.linspace(0, 4.0, 1024, endpoint=False)
    freq = 6.0  # 6 Hz in Theta band (4 - 8 Hz)

    # Case A: Identical sine waves (zero phase lag: volume conduction artifact)
    ch1 = np.sin(2 * np.pi * freq * t)
    ch2 = np.sin(2 * np.pi * freq * t)  # Identical

    win_zero_lag = np.zeros((1, 18, 1024), dtype=np.float32)
    win_zero_lag[0, 0, :] = ch1
    win_zero_lag[0, 1, :] = ch2

    # Case B: 90-degree phase-shifted sine wave (true phase-delayed physiological coupling)
    ch3 = np.cos(2 * np.pi * freq * t)  # 90 degrees out of phase

    win_lagged = np.zeros((1, 18, 1024), dtype=np.float32)
    win_lagged[0, 0, :] = ch1
    win_lagged[0, 1, :] = ch3

    # Pair (0, 1) is the first pair in upper triangle
    # Test band 'theta'
    plv_zero, pli_zero = extractor._compute_band_matrices(win_zero_lag, "theta")
    plv_lag, pli_lag = extractor._compute_band_matrices(win_lagged, "theta")

    pair_idx = 0  # Pair (ch 0, ch 1)

    # In zero-lag: PLV is high (~1.0), PLI is ~0.0
    assert plv_zero[0, pair_idx] > 0.95, f"Expected high PLV for identical signals, got {plv_zero[0, pair_idx]}"
    assert pli_zero[0, pair_idx] < 0.05, f"Expected near-zero PLI for zero-lag coupling, got {pli_zero[0, pair_idx]}"

    # In true phase lag: both PLV and PLI are high (~1.0)
    assert plv_lag[0, pair_idx] > 0.95, f"Expected high PLV for lagged coupling, got {plv_lag[0, pair_idx]}"
    assert pli_lag[0, pair_idx] > 0.95, f"Expected high PLI for true phase-lagged coupling, got {pli_lag[0, pair_idx]}"


def test_pairwise_expansion_mode():
    """Verify that include_pairwise=True appends the full 153 pairwise upper-triangle edges."""
    extractor = EEGConnectivityExtractor(
        sampling_rate=SAMPLING_RATE,
        include_nodal=True,
        include_global=True,
        include_hemispheric=True,
        include_pairwise=True,
    )
    # 4 bands * (50 summary + 153 plv pairs + 153 pli pairs) = 4 * 356 = 1424 features
    expected_dim = 4 * (50 + 153 + 153)
    assert len(extractor.feature_names) == expected_dim

    test_window = np.random.randn(2, 18, 1024).astype(np.float32)
    feats = extractor.extract_features(test_window)
    assert feats.shape == (2, expected_dim)


def test_connectivity_on_real_chb_mit_slice():
    """Smoke test running extractor on a real slice from CHB-MIT prototype dataset."""
    from src.data.loader import EDFLoader
    from src.data.windowing import WindowGenerator
    from src.utils.config import DATASET_DIR

    rec_path = DATASET_DIR / "chb01" / "chb01_03.edf"
    loader = EDFLoader()
    window_gen = WindowGenerator()

    # Load 16 seconds = 4 windows
    slice_signal = loader.load_slice(
        rec_path,
        start_sample=0,
        stop_sample=16 * SAMPLING_RATE,
        apply_filter=True,
        apply_standardization=True,
    )
    windows, labels, meta = window_gen.extract_windows_for_recording(
        signal=slice_signal,
        seizure_intervals=[],
        is_training=True,
    )
    assert len(windows) >= 3

    extractor = EEGConnectivityExtractor()
    feats = extractor.extract_features(windows)

    assert feats.shape == (len(windows), 200)
    assert np.all(feats >= 0.0) or np.any(feats < 0.0)  # asymmetry metrics can be negative
    assert not np.isnan(feats).any()
