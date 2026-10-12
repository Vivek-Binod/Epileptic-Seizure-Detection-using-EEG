"""Unit and integration tests for FeatureCache."""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.cache import FeatureCache, compute_config_fingerprint
from src.features.feature_extractor import EEGFeatureExtractor
from src.utils.config import DATASET_DIR, SAMPLING_RATE


def test_config_fingerprint_deterministic_and_sensitive():
    fp1 = compute_config_fingerprint()
    fp2 = compute_config_fingerprint()
    assert len(fp1) == 64, "Expected 64-char SHA256 hex string"
    assert fp1 == fp2, "Fingerprint must be deterministic"

    # Sensitivity check: change a configuration parameter
    custom_cfg = {
        "cache_version": "1.0.0",
        "sampling_rate": 256,
        "bandpass_low": 0.5,
        "bandpass_high": 40.0,  # Changed from 45.0
    }
    fp_custom = compute_config_fingerprint(custom_cfg)
    assert fp_custom != fp1, "Fingerprint must change when configuration changes"
    print("test_config_fingerprint_deterministic_and_sensitive PASSED!")


def test_feature_cache_creation_and_reload(tmp_path):
    cache = FeatureCache(cache_dir=tmp_path)
    extractor = EEGFeatureExtractor()

    # Generate synthetic DataFrame with metadata + 252 features
    N = 20
    meta = {
        "case_id": ["chb01"] * N,
        "patient_id": ["patient_01"] * N,
        "file_name": ["chb01_03.edf"] * N,
        "mode": ["training"] * N,
        "window_idx": list(range(N)),
        "start_sample": [i * 256 for i in range(N)],
        "end_sample": [i * 256 + 1024 for i in range(N)],
        "start_sec": [i * 1.0 for i in range(N)],
        "end_sec": [i * 1.0 + 4.0 for i in range(N)],
        "overlap_ratio": [0.75 if i < 5 else 0.0 for i in range(N)],
        "seizure_idx": [1 if i < 5 else -1 for i in range(N)],
        "category": ["ictal" if i < 5 else "interictal" for i in range(N)],
        "label": [1 if i < 5 else 0 for i in range(N)],
    }
    df_meta = pd.DataFrame(meta)

    rng = np.random.RandomState(42)
    fake_feats = rng.randn(N, 252).astype(np.float32)
    df_feats = pd.DataFrame(fake_feats, columns=extractor.feature_names)
    df_full = pd.concat([df_meta, df_feats], axis=1)

    # 1. Save to cache
    assert not cache.is_cached("chb01", "chb01_03.edf", "training")
    saved_path = cache.save_recording_features(df_full, "chb01", "chb01_03.edf", "training")
    assert saved_path.exists()
    assert cache.is_cached("chb01", "chb01_03.edf", "training")

    # 2. Reload from cache
    reloaded_df = cache.load_recording_features("chb01", "chb01_03.edf", "training")
    assert reloaded_df is not None
    assert len(reloaded_df) == N
    assert set(reloaded_df.columns) == set(df_full.columns)

    # 3. Verify numerical and metadata precision
    X, y, meta_out = cache.df_to_features_and_labels(reloaded_df)
    assert X.shape == (N, 252)
    assert X.dtype == np.float32
    assert np.allclose(X, fake_feats, atol=1e-7)
    assert np.array_equal(y, df_meta["label"].values)
    assert list(meta_out["patient_id"]) == ["patient_01"] * N

    # 4. Verify stats
    stats = cache.get_stats()
    assert stats["cached_files"] == 1
    assert stats["total_size_bytes"] > 0
    print("test_feature_cache_creation_and_reload PASSED!")


def test_cache_invalidation_on_fingerprint_change(tmp_path):
    cache_v1 = FeatureCache(cache_dir=tmp_path, config_fingerprint="1111111111111111")
    extractor = EEGFeatureExtractor()

    # Create dummy dataframe
    N = 5
    meta = {col: [0] * N for col in FeatureCache.METADATA_COLUMNS}
    feats = {col: [0.0] * N for col in extractor.feature_names}
    df = pd.DataFrame({**meta, **feats})

    cache_v1.save_recording_features(df, "chb01", "chb01_01.edf", "training")
    assert cache_v1.is_cached("chb01", "chb01_01.edf", "training")

    # Instantiate cache with different fingerprint
    cache_v2 = FeatureCache(cache_dir=tmp_path, config_fingerprint="2222222222222222")
    # Must NOT recognize cache from fingerprint v1
    assert not cache_v2.is_cached("chb01", "chb01_01.edf", "training")
    assert cache_v2.load_recording_features("chb01", "chb01_01.edf", "training") is None
    print("test_cache_invalidation_on_fingerprint_change PASSED!")


def test_handling_of_corrupted_cache_file(tmp_path):
    cache = FeatureCache(cache_dir=tmp_path)
    extractor = EEGFeatureExtractor()

    # Save a valid file first
    meta = {col: [0] * 5 for col in FeatureCache.METADATA_COLUMNS}
    feats = {col: [0.0] * 5 for col in extractor.feature_names}
    df = pd.DataFrame({**meta, **feats})
    path = cache.save_recording_features(df, "chb01", "corrupt_test.edf", "training")
    assert cache.is_cached("chb01", "corrupt_test.edf", "training")

    # Corrupt the file by overwriting with non-Parquet garbage bytes
    with open(path, "wb") as f:
        f.write(b"CORRUPTED_GARBAGE_DATA_NOT_PARQUET_HEADER")

    # is_cached must detect corruption and return False
    assert not cache.is_cached("chb01", "corrupt_test.edf", "training")
    # load_recording_features must gracefully return None and remove corrupted file
    loaded = cache.load_recording_features("chb01", "corrupt_test.edf", "training")
    assert loaded is None
    assert not path.exists(), "Corrupted file was not cleaned up"
    print("test_handling_of_corrupted_cache_file PASSED!")


def test_patient_identity_mapping_chb01_and_chb21(tmp_path):
    cache = FeatureCache(cache_dir=tmp_path)
    extractor = EEGFeatureExtractor()

    for cid in ["chb01", "chb21"]:
        meta = {col: [0] * 2 for col in FeatureCache.METADATA_COLUMNS}
        meta["case_id"] = [cid] * 2
        meta["patient_id"] = [cache.patient_mapping.get(cid, cid)] * 2
        feats = {col: [1.0] * 2 for col in extractor.feature_names}
        df = pd.DataFrame({**meta, **feats})
        cache.save_recording_features(df, cid, f"{cid}_01.edf", "training")

        reloaded = cache.load_recording_features(cid, f"{cid}_01.edf", "training")
        assert list(reloaded["patient_id"]) == ["patient_01", "patient_01"], (
            f"Expected patient_01 for {cid}, got {reloaded['patient_id'].iloc[0]}"
        )
    print("test_patient_identity_mapping_chb01_and_chb21 PASSED!")


def test_prototype_recording_cache_integration(tmp_path):
    """Integration test: extract a real prototype recording, verify cache write and fast reload."""
    cache = FeatureCache(cache_dir=tmp_path)
    edf_path = DATASET_DIR / "chb01/chb01_03.edf"
    assert edf_path.exists(), f"Raw recording missing: {edf_path}"

    seizure_intervals = [(2996 * SAMPLING_RATE, 3036 * SAMPLING_RATE)]

    # 1. First run: Cache MISS (loads raw EDF, filters, windows, extracts features)
    t0 = time.time()
    df_first = cache.extract_or_load(
        case_id="chb01",
        file_name="chb01_03.edf",
        mode="training",
        seizure_intervals=seizure_intervals,
        full_path=edf_path,
        neg_pos_ratio=10.0,
        random_seed=42,
    )
    t_miss = time.time() - t0

    assert len(df_first) > 0
    assert (df_first["label"] == 1).sum() == 41  # Exactly 41 ictal windows verified
    assert len(df_first.columns) == len(FeatureCache.METADATA_COLUMNS) + 252
    assert cache.is_cached("chb01", "chb01_03.edf", "training")

    # 2. Second run: Cache HIT (reads directly from Parquet on disk)
    t1 = time.time()
    df_second = cache.extract_or_load(
        case_id="chb01",
        file_name="chb01_03.edf",
        mode="training",
        seizure_intervals=seizure_intervals,
        full_path=edf_path,
    )
    t_hit = time.time() - t1

    assert len(df_second) == len(df_first)
    # Numerical equality
    X1, y1, _ = cache.df_to_features_and_labels(df_first)
    X2, y2, _ = cache.df_to_features_and_labels(df_second)
    assert np.allclose(X1, X2, atol=1e-6)
    assert np.array_equal(y1, y2)

    # Verification: Cache hit should be dramatically faster (orders of magnitude)
    print(f"  Extraction time (Cache Miss): {t_miss:.3f} s")
    print(f"  Reload time     (Cache Hit):  {t_hit:.4f} s")
    print(f"  Speedup factor:               {t_miss / max(t_hit, 1e-4):.1f}x")
    assert t_hit < t_miss, f"Cache reload ({t_hit}s) should be faster than raw extraction ({t_miss}s)"
    print("test_prototype_recording_cache_integration PASSED!")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        test_config_fingerprint_deterministic_and_sensitive()
        test_feature_cache_creation_and_reload(p)
        test_cache_invalidation_on_fingerprint_change(p)
        test_handling_of_corrupted_cache_file(p)
        test_patient_identity_mapping_chb01_and_chb21(p)
        test_prototype_recording_cache_integration(p)
    print("\nALL FEATURE CACHE TESTS PASSED!")
