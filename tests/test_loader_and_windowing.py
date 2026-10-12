"""Unit tests for EDFLoader and WindowGenerator."""

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import EDFLoader
from src.data.windowing import WindowGenerator
from src.utils.config import DATASET_DIR, SAMPLING_RATE, CANONICAL_CHANNELS


def test_loader_inspect_and_slice():
    loader = EDFLoader()
    edf_path = DATASET_DIR / "chb01/chb01_03.edf"
    assert edf_path.exists(), f"File {edf_path} does not exist"

    # Test header inspection
    info = loader.inspect_header(edf_path)
    assert info["sfreq"] == 256.0
    assert len(info["canonical_channels"]) == 18
    assert len(info["channel_map"]) == 18

    # Test slice loading (4 seconds = 1024 samples)
    slice_data = loader.load_slice(
        edf_path,
        start_sample=0,
        stop_sample=1024,
        apply_filter=True,
        apply_standardization=True,
    )
    assert slice_data.shape == (18, 1024)
    assert not np.isnan(slice_data).any(), "Found NaNs in loaded slice"
    assert not np.isinf(slice_data).any(), "Found Infs in loaded slice"
    # Channel-wise standardization check
    means = np.mean(slice_data, axis=-1)
    stds = np.std(slice_data, axis=-1)
    assert np.allclose(means, 0.0, atol=1e-5), f"Means not close to 0: {means}"
    assert np.allclose(stds, 1.0, atol=1e-4), f"Stds not close to 1: {stds}"
    print("test_loader_inspect_and_slice PASSED!")


def test_loader_duplicate_channel_policy():
    loader = EDFLoader()
    # chb01_01.edf has duplicate T8-P8
    edf_path = DATASET_DIR / "chb01/chb01_01.edf"
    info = loader.inspect_header(edf_path)
    raw_chs = [ch.replace("-0", "").replace("-1", "").strip().upper() for ch in info["raw_channels"]]
    t8_p8_raw_indices = [i for i, ch in enumerate(raw_chs) if ch == "T8-P8"]
    assert len(t8_p8_raw_indices) >= 2, "Expected duplicate T8-P8 in chb01_01"

    # Verify that the channel_map selected the FIRST index
    t8_p8_canonical_idx = CANONICAL_CHANNELS.index("T8-P8")
    selected_raw_idx = info["channel_map"][t8_p8_canonical_idx]
    assert selected_raw_idx == t8_p8_raw_indices[0], (
        f"Expected first index {t8_p8_raw_indices[0]}, got {selected_raw_idx}"
    )
    print("test_loader_duplicate_channel_policy PASSED!")


def test_windowing_synthetic():
    sfreq = 256
    window_samples = 1024  # 4s
    ictal_stride = 256     # 1s
    interictal_stride = 1024 # 4s
    generator = WindowGenerator(
        sampling_rate=sfreq,
        window_samples=window_samples,
        ictal_stride_samples=ictal_stride,
        interictal_stride_samples=interictal_stride,
        pre_ictal_guard_sec=60.0,
        post_ictal_guard_sec=60.0,
    )

    # Create synthetic signal: 300 seconds (76,800 samples)
    # Seizure from 120s to 160s (duration 40s): sample 30,720 to 40,960
    total_samples = 300 * sfreq
    n_channels = 18
    signal = np.random.randn(n_channels, total_samples).astype(np.float32)
    seizure_intervals = [(120 * sfreq, 160 * sfreq)]

    # 1. Training mode
    windows, labels, meta = generator.extract_windows_for_recording(
        signal=signal,
        seizure_intervals=seizure_intervals,
        is_training=True,
        neg_pos_ratio=10.0,
        random_seed=42,
    )

    assert len(windows) == len(labels) == len(meta)
    pos_count = np.sum(labels == 1)
    neg_count = np.sum(labels == 0)
    assert pos_count > 0, "No positive windows generated"
    assert neg_count > 0, "No negative windows generated"
    # Ratio should be at most 10:1
    assert neg_count <= 10 * pos_count

    # Guard-band verification:
    # No negative window should fall in [120 - 60 = 60s, 160 + 60 = 220s]
    # Sample range [15,360, 56,320]
    guard_start_sample = (120 - 60) * sfreq
    guard_end_sample = (160 + 60) * sfreq

    neg_meta = meta[meta["label"] == 0]
    for _, row in neg_meta.iterrows():
        w_s, w_e = row["start_sample"], row["end_sample"]
        # Ensure window is completely outside guard zone
        assert (w_e <= guard_start_sample) or (w_s >= guard_end_sample), (
            f"Negative window [{w_s}, {w_e}] violates guard band [{guard_start_sample}, {guard_end_sample}]"
        )

    # 2. Evaluation mode (default stride)
    eval_windows, eval_labels, eval_meta = generator.extract_windows_for_recording(
        signal=signal,
        seizure_intervals=seizure_intervals,
        is_training=False,
    )
    expected_eval_windows = (total_samples - window_samples) // interictal_stride + 1
    assert len(eval_windows) == expected_eval_windows
    print("test_windowing_synthetic PASSED!")


def test_windowing_multi_seizures_and_strides():
    """Test multiple closely-spaced seizures and configurable evaluation sliding strides."""
    sfreq = 256
    window_samples = 1024  # 4s
    generator = WindowGenerator(sampling_rate=sfreq, window_samples=window_samples)

    total_samples = 200 * sfreq
    signal = np.zeros((18, total_samples), dtype=np.float32)

    # Two closely spaced seizures:
    # Seizure 1: 50s to 52s (duration 2s = 512 samples)
    # Seizure 2: 53s to 55s (duration 2s = 512 samples)
    # Gap is 1s (256 samples).
    s1 = (50 * sfreq, 52 * sfreq)
    s2 = (53 * sfreq, 55 * sfreq)
    seizure_intervals = [s1, s2]

    # A window from 51s to 55s (4s duration) covers:
    # 1s of s1 (256 samples) + 1s gap + 2s of s2 (512 samples) = 768 samples total seizure!
    # 768 / 1024 = 75% overlap. Should be labeled positive (y=1)!
    w_start = 51 * sfreq
    w_end = w_start + window_samples
    total_o = sum(max(0, min(w_end, se) - max(w_start, ss)) for ss, se in seizure_intervals)
    assert total_o == 768

    # Test training extraction with deduplication
    train_w, train_y, train_meta = generator.extract_windows_for_recording(
        signal=signal,
        seizure_intervals=seizure_intervals,
        is_training=True,
    )
    # Verify no duplicate start_samples
    assert len(train_meta["start_sample"]) == len(train_meta["start_sample"].unique()), (
        "Duplicate window starts found in training extraction!"
    )
    assert np.all(train_y[train_meta["overlap_ratio"] >= 0.5] == 1)

    # Test evaluation mode with fine 1.0s sliding stride (256 samples)
    eval_stride = 256
    eval_w, eval_y, eval_meta = generator.extract_windows_for_recording(
        signal=signal,
        seizure_intervals=seizure_intervals,
        is_training=False,
        eval_stride_samples=eval_stride,
    )
    expected_fine_windows = (total_samples - window_samples) // eval_stride + 1
    assert len(eval_w) == expected_fine_windows
    # Some windows spanning the two seizures must be positive
    assert np.sum(eval_y == 1) > 0
    print("test_windowing_multi_seizures_and_strides PASSED!")


if __name__ == "__main__":
    test_loader_inspect_and_slice()
    test_loader_duplicate_channel_policy()
    test_windowing_synthetic()
    test_windowing_multi_seizures_and_strides()
    print("\nALL LOADER AND WINDOWING TESTS PASSED!")
