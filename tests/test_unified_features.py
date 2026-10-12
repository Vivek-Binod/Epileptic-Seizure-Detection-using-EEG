"""Unit tests for UnifiedFeatureExtractor and multimodal caching."""

import numpy as np
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.cache import FeatureCache
from src.features.unified import UnifiedFeatureExtractor
from src.utils.config import SAMPLING_RATE


def test_unified_extractor_shapes_and_slicing():
    """Verify UnifiedFeatureExtractor shapes and systematic ablation slicing."""
    extractor = UnifiedFeatureExtractor(sampling_rate=SAMPLING_RATE)

    assert len(extractor.conv_feature_names) == 252
    assert len(extractor.conn_feature_names) == 200
    assert len(extractor.plv_feature_names) == 112
    assert len(extractor.pli_feature_names) == 88
    assert len(extractor.feature_names) == 452

    n_windows = 3
    test_windows = np.random.randn(n_windows, 18, 1024).astype(np.float32)

    X = extractor.extract_features(test_windows)
    assert X.shape == (n_windows, 452)
    assert not np.isnan(X).any()

    # Test all ablation slices
    X_conv = extractor.slice_features(X, "conventional")
    assert X_conv.shape == (n_windows, 252)

    X_plv = extractor.slice_features(X, "plv_only")
    assert X_plv.shape == (n_windows, 112)

    X_pli = extractor.slice_features(X, "pli_only")
    assert X_pli.shape == (n_windows, 88)

    X_conn = extractor.slice_features(X, "connectivity")
    assert X_conn.shape == (n_windows, 200)

    X_comb = extractor.slice_features(X, "combined")
    assert X_comb.shape == (n_windows, 452)

    X_c_plv = extractor.slice_features(X, "conv_plus_plv")
    assert X_c_plv.shape == (n_windows, 364)

    X_c_pli = extractor.slice_features(X, "conv_plus_pli")
    assert X_c_pli.shape == (n_windows, 340)


def test_unified_feature_cache_roundtrip(tmp_path):
    """Verify FeatureCache saves and reloads unified multimodal feature DataFrames."""
    extractor = UnifiedFeatureExtractor(sampling_rate=SAMPLING_RATE)
    cache = FeatureCache(cache_dir=tmp_path, extractor=extractor)

    assert cache.extractor.feature_count == 452

    # Synthesize DataFrame with metadata + 452 features
    n_rows = 5
    meta_data = {
        "case_id": ["chb01"] * n_rows,
        "patient_id": ["patient_01"] * n_rows,
        "file_name": ["test.edf"] * n_rows,
        "mode": ["training"] * n_rows,
        "window_idx": list(range(n_rows)),
        "start_sample": list(range(0, n_rows * 256, 256)),
        "end_sample": list(range(1024, 1024 + n_rows * 256, 256)),
        "start_sec": [float(i) for i in range(n_rows)],
        "end_sec": [float(i + 4) for i in range(n_rows)],
        "overlap_ratio": [0.0] * n_rows,
        "seizure_idx": [-1] * n_rows,
        "category": ["interictal"] * n_rows,
        "label": [0] * n_rows,
    }
    feat_data = {col: np.random.randn(n_rows).astype(np.float32) for col in extractor.feature_names}
    import pandas as pd
    df = pd.DataFrame({**meta_data, **feat_data})

    # Save
    p = cache.save_recording_features(df, "chb01", "test.edf", "training")
    assert p.exists()

    # Load
    reloaded_df = cache.load_recording_features("chb01", "test.edf", "training")
    assert reloaded_df is not None
    assert len(reloaded_df) == n_rows
    assert len(reloaded_df.columns) == 13 + 452

    # Separate X, y, meta
    X, y, meta_df = cache.df_to_features_and_labels(reloaded_df)
    assert X.shape == (n_rows, 452)
    assert len(y) == n_rows
