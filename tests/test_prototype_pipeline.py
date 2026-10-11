"""End-to-end prototype cohort validation test.

Validates the complete pipeline (loading, channel standardization, preprocessing,
window generation, guard-band filtering, and feature extraction) across the 5 permitted
prototype subjects: chb01, chb03, chb05, chb07, chb08.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import EDFLoader
from src.data.metadata_parser import CHBMITMetadataParser
from src.data.windowing import WindowGenerator
from src.features.feature_extractor import EEGFeatureExtractor
from src.utils.config import (
    DATASET_DIR,
    PROTOTYPE_COHORT,
    SAMPLING_RATE,
    WINDOW_SAMPLES,
)


def run_prototype_cohort_validation() -> Dict[str, Dict]:
    """Execute prototype pipeline checks across the 5 prototype patients."""
    print("=" * 70)
    print("STARTING PROTOTYPE COHORT VALIDATION (5 PATIENTS)")
    print("=" * 70)

    # 1. Load parsed manifests
    parser = CHBMITMetadataParser()
    recordings_df, seizures_df = parser.parse_all()

    loader = EDFLoader()
    window_gen = WindowGenerator()
    extractor = EEGFeatureExtractor()

    # Define test pairs: (seizure_file, non_seizure_file) per prototype subject
    test_cases = [
        ("chb01", "chb01_03.edf", "chb01_01.edf"),
        ("chb03", "chb03_01.edf", "chb03_05.edf"),
        ("chb05", "chb05_06.edf", "chb05_01.edf"),
        ("chb07", "chb07_19.edf", "chb07_01.edf"),
        ("chb08", "chb08_02.edf", "chb08_03.edf"),
    ]

    cohort_results: Dict[str, Dict] = {}

    for case_id, seiz_file, non_seiz_file in test_cases:
        print(f"\n---> Testing Subject: {case_id}")
        seiz_rec_path = DATASET_DIR / case_id / seiz_file
        non_seiz_rec_path = DATASET_DIR / case_id / non_seiz_file

        # Retrieve ground-truth seizure intervals for the seizure file
        file_seizures = seizures_df[
            (seizures_df["case_id"] == case_id) & (seizures_df["file_name"] == seiz_file)
        ]
        seizure_intervals_samples = [
            (row["start_sample"], row["end_sample"]) for _, row in file_seizures.iterrows()
        ]
        print(f"  [Seizure File: {seiz_file}] Annotated seizures: {len(seizure_intervals_samples)}")
        for _, row in file_seizures.iterrows():
            print(f"    - Interval: {row['start_sec']}s to {row['end_sec']}s ({row['duration_sec']}s)")

        # Inspect headers
        seiz_header = loader.inspect_header(seiz_rec_path)
        assert len(seiz_header["channel_map"]) == 18, f"Failed channel mapping for {seiz_file}"

        # Test Slice Loading around seizure (e.g. 100s slice covering seizure + guards)
        first_seiz = file_seizures.iloc[0]
        slice_start_sample = max(0, first_seiz["start_sample"] - 30 * SAMPLING_RATE)
        slice_end_sample = min(seiz_header["n_times"], first_seiz["end_sample"] + 30 * SAMPLING_RATE)

        slice_data = loader.load_slice(
            seiz_rec_path,
            start_sample=slice_start_sample,
            stop_sample=slice_end_sample,
            apply_filter=True,
            apply_standardization=True,
        )
        assert slice_data.shape[0] == 18, f"Expected 18 channels, got {slice_data.shape[0]}"
        assert not np.isnan(slice_data).any(), f"NaNs found in slice of {seiz_file}"

        # Test Windowing on this slice
        # Adjust seizure intervals relative to slice start
        local_seiz_intervals = [
            (s_start - slice_start_sample, s_end - slice_start_sample)
            for s_start, s_end in seizure_intervals_samples
        ]

        train_windows, train_labels, train_meta = window_gen.extract_windows_for_recording(
            signal=slice_data,
            seizure_intervals=local_seiz_intervals,
            is_training=True,
            neg_pos_ratio=10.0,
            random_seed=42,
        )

        n_pos = np.sum(train_labels == 1)
        n_neg = np.sum(train_labels == 0)
        print(f"  [Windowing on Seizure Slice] Total windows: {len(train_labels)} (Pos={n_pos}, Neg={n_neg})")
        assert n_pos > 0, f"Expected positive windows for {seiz_file}, got 0"

        # Test Feature Extraction on extracted windows
        features = extractor.extract_features(train_windows)
        assert features.shape == (len(train_windows), 252), (
            f"Expected shape ({len(train_windows)}, 252), got {features.shape}"
        )
        assert not np.isnan(features).any(), f"NaNs in features for {seiz_file}"
        assert not np.isinf(features).any(), f"Infs in features for {seiz_file}"
        print(f"  [Features Extracted] Matrix shape: {features.shape}, dtype: {features.dtype}")

        # Test Non-Seizure File: load 60 seconds slice
        non_seiz_header = loader.inspect_header(non_seiz_rec_path)
        assert len(non_seiz_header["channel_map"]) == 18
        ns_slice_data = loader.load_slice(
            non_seiz_rec_path,
            start_sample=0,
            stop_sample=60 * SAMPLING_RATE,
            apply_filter=True,
            apply_standardization=True,
        )
        ns_windows, ns_labels, ns_meta = window_gen.extract_windows_for_recording(
            signal=ns_slice_data,
            seizure_intervals=[],  # No seizures
            is_training=True,
        )
        assert np.sum(ns_labels == 1) == 0, f"False positive label generated for non-seizure file {non_seiz_file}"
        assert len(ns_labels) > 0, f"No windows generated for non-seizure file {non_seiz_file}"
        print(f"  [Non-Seizure File: {non_seiz_file}] Verified 0 positive windows (All {len(ns_labels)} are background)")

        cohort_results[case_id] = {
            "seizure_file": seiz_file,
            "non_seizure_file": non_seiz_file,
            "seizures_annotated": len(file_seizures),
            "slice_pos_windows": int(n_pos),
            "slice_neg_windows": int(n_neg),
            "features_extracted_shape": features.shape,
            "channels_verified": 18,
            "status": "PASSED",
        }

    # Test Event-Level Aggregation Logic
    print("\n---> Testing Event-Level Collar Alarm & Single-Counting Logic")
    test_event_aggregation()

    print("\n" + "=" * 70)
    print("PROTOTYPE COHORT VALIDATION COMPLETE: ALL 5 SUBJECTS PASSED")
    print("=" * 70)
    return cohort_results


def test_event_aggregation():
    """Verify that consecutive positive window predictions trigger event alarms without double-counting."""
    # Synthetic continuous timeline: 100 windows (each 4s, stride 1s = 100 seconds)
    # Annotated seizure from t=30s to t=50s (20s duration)
    annotated_seizure = (30.0, 50.0)

    # Simulated model predictions: windows 28 to 48 are predicted positive
    predictions = np.zeros(100, dtype=int)
    predictions[28:48] = 1

    # Collar filter: requires M=3 consecutive positive windows to declare an alarm
    M = 3
    consecutive_count = 0
    raw_alarms = []
    for i, p in enumerate(predictions):
        if p == 1:
            consecutive_count += 1
            if consecutive_count == M:
                raw_alarms.append(i)  # Alarm onset at window i
        else:
            consecutive_count = 0

    assert len(raw_alarms) == 1, f"Expected exactly 1 alarm onset for continuous burst, got {len(raw_alarms)}"

    # Single-count evaluation rule:
    # Match alarm to annotated seizure: true if alarm falls in [T_start - 10s, T_end + 10s]
    alarm_sec = raw_alarms[0]  # window index equals approximate start second with stride 1s
    is_detected = (annotated_seizure[0] - 10.0) <= alarm_sec <= (annotated_seizure[1] + 10.0)
    assert is_detected, "Seizure was not detected by collar alarm"
    print(f"  Collar alarm triggered at t={alarm_sec}s -> Matches seizure [30s, 50s]. True Positive = 1.")


if __name__ == "__main__":
    results = run_prototype_cohort_validation()
    for subj, res in results.items():
        print(f"Result for {subj}: {res['status']} ({res['slice_pos_windows']} ictal windows, features: {res['features_extracted_shape']})")
