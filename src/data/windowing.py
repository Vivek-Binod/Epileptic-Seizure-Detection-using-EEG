"""Window slicing, mathematical labeling, and guard-band filtering module."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    ICTAL_STRIDE_SAMPLES,
    INTERICTAL_STRIDE_SAMPLES,
    OVERLAP_THRESHOLD,
    POST_ICTAL_GUARD_SEC,
    PRE_ICTAL_GUARD_SEC,
    SAMPLING_RATE,
    WINDOW_SAMPLES,
)


class WindowGenerator:
    """Generates labeled EEG windows with strict mathematical overlap and guard-band logic."""

    def __init__(
        self,
        sampling_rate: int = SAMPLING_RATE,
        window_samples: int = WINDOW_SAMPLES,
        ictal_stride_samples: int = ICTAL_STRIDE_SAMPLES,
        interictal_stride_samples: int = INTERICTAL_STRIDE_SAMPLES,
        overlap_threshold: float = OVERLAP_THRESHOLD,
        pre_ictal_guard_sec: float = PRE_ICTAL_GUARD_SEC,
        post_ictal_guard_sec: float = POST_ICTAL_GUARD_SEC,
    ):
        self.sfreq = sampling_rate
        self.window_samples = window_samples
        self.ictal_stride_samples = ictal_stride_samples
        self.interictal_stride_samples = interictal_stride_samples
        self.overlap_threshold = overlap_threshold
        self.pre_guard_samples = int(pre_ictal_guard_sec * self.sfreq)
        self.post_guard_samples = int(post_ictal_guard_sec * self.sfreq)

    def extract_windows_for_recording(
        self,
        signal: np.ndarray,
        seizure_intervals: List[Tuple[int, int]],
        is_training: bool = True,
        neg_pos_ratio: Optional[float] = None,
        random_seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Extract windows from a continuous multi-channel signal array.

        Args:
            signal: np.ndarray of shape (channels, total_samples)
            seizure_intervals: List of (start_sample, end_sample) tuples for true seizures
            is_training: If True, applies training guard bands and boundary exclusions.
                         If False, evaluates full continuous recording without omissions.
            neg_pos_ratio: Ratio of negative to positive windows for training subsampling.
                           If None, retains all valid negative windows.
            random_seed: Seed for reproducible negative subsampling.

        Returns:
            Tuple of (windows_array, labels_array, metadata_df)
            windows_array shape: (N, channels, window_samples)
            labels_array shape: (N,) with integer values 0 or 1
            metadata_df: DataFrame with window intervals, overlaps, and categories
        """
        n_channels, total_samples = signal.shape

        if is_training:
            return self._extract_training_windows(
                signal=signal,
                seizure_intervals=seizure_intervals,
                neg_pos_ratio=neg_pos_ratio,
                random_seed=random_seed,
            )
        else:
            return self._extract_evaluation_windows(
                signal=signal,
                seizure_intervals=seizure_intervals,
            )

    def _extract_training_windows(
        self,
        signal: np.ndarray,
        seizure_intervals: List[Tuple[int, int]],
        neg_pos_ratio: Optional[float],
        random_seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Training mode: extracts augmented ictal windows and guard-banded interictal windows."""
        n_channels, total_samples = signal.shape
        pos_windows: List[np.ndarray] = []
        pos_meta: List[Dict] = []

        # 1. Extract Ictal (Seizure) Windows with fine stride (data augmentation)
        for s_idx, (s_start, s_end) in enumerate(seizure_intervals):
            # Start sliding slightly before seizure onset to capture transitions that meet threshold
            search_start = max(0, s_start - self.window_samples)
            search_end = min(total_samples - self.window_samples, s_end + self.window_samples)

            for w_start in range(search_start, search_end + 1, self.ictal_stride_samples):
                w_end = w_start + self.window_samples
                if w_end > total_samples:
                    break

                # Overlap calculation with this seizure
                overlap = max(0, min(w_end, s_end) - max(w_start, s_start))
                ratio = overlap / float(self.window_samples)

                # Positive label condition: r >= 0.5
                if ratio >= self.overlap_threshold:
                    window = signal[:, w_start:w_end]
                    pos_windows.append(window)
                    pos_meta.append(
                        {
                            "start_sample": w_start,
                            "end_sample": w_end,
                            "start_sec": w_start / float(self.sfreq),
                            "end_sec": w_end / float(self.sfreq),
                            "overlap_ratio": ratio,
                            "seizure_idx": s_idx + 1,
                            "label": 1,
                            "category": "ictal",
                        }
                    )

        # 2. Define Exclusion Zones (Seizures + Pre/Post Guard Bands)
        exclusion_intervals: List[Tuple[int, int]] = []
        for s_start, s_end in seizure_intervals:
            ex_start = max(0, s_start - self.pre_guard_samples)
            ex_end = min(total_samples, s_end + self.post_guard_samples)
            exclusion_intervals.append((ex_start, ex_end))

        # Merge overlapping exclusion intervals
        merged_exclusions = self._merge_intervals(exclusion_intervals)

        # 3. Extract Background (Interictal) Windows with non-overlapping stride
        neg_windows: List[np.ndarray] = []
        neg_meta: List[Dict] = []

        for w_start in range(0, total_samples - self.window_samples + 1, self.interictal_stride_samples):
            w_end = w_start + self.window_samples

            # Check if window intersects any exclusion zone
            in_exclusion = False
            for ex_start, ex_end in merged_exclusions:
                if not (w_end <= ex_start or w_start >= ex_end):
                    in_exclusion = True
                    break

            if not in_exclusion:
                window = signal[:, w_start:w_end]
                neg_windows.append(window)
                neg_meta.append(
                    {
                        "start_sample": w_start,
                        "end_sample": w_end,
                        "start_sec": w_start / float(self.sfreq),
                        "end_sec": w_end / float(self.sfreq),
                        "overlap_ratio": 0.0,
                        "seizure_idx": None,
                        "label": 0,
                        "category": "interictal",
                    }
                )

        # 4. Optional Negative Subsampling for Training Balance
        if neg_pos_ratio is not None and len(pos_windows) > 0:
            target_neg_count = int(len(pos_windows) * neg_pos_ratio)
            if target_neg_count < len(neg_windows):
                rng = np.random.RandomState(random_seed)
                selected_indices = np.sort(
                    rng.choice(len(neg_windows), size=target_neg_count, replace=False)
                )
                neg_windows = [neg_windows[i] for i in selected_indices]
                neg_meta = [neg_meta[i] for i in selected_indices]

        # Combine
        all_windows_list = pos_windows + neg_windows
        all_meta_list = pos_meta + neg_meta

        if not all_windows_list:
            empty_windows = np.empty((0, n_channels, self.window_samples), dtype=np.float32)
            empty_labels = np.empty((0,), dtype=np.int64)
            return empty_windows, empty_labels, pd.DataFrame(all_meta_list)

        windows_arr = np.stack(all_windows_list, axis=0).astype(np.float32)
        labels_arr = np.array([m["label"] for m in all_meta_list], dtype=np.int64)
        meta_df = pd.DataFrame(all_meta_list)

        return windows_arr, labels_arr, meta_df

    def _extract_evaluation_windows(
        self,
        signal: np.ndarray,
        seizure_intervals: List[Tuple[int, int]],
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Evaluation mode: slices full continuous recording sequentially without guard-band exclusion."""
        n_channels, total_samples = signal.shape
        eval_windows: List[np.ndarray] = []
        eval_meta: List[Dict] = []

        # Sequential non-overlapping or uniform sliding
        for w_start in range(0, total_samples - self.window_samples + 1, self.interictal_stride_samples):
            w_end = w_start + self.window_samples

            # Calculate overlap with any seizure
            max_overlap = 0
            best_s_idx = None
            for s_idx, (s_start, s_end) in enumerate(seizure_intervals):
                overlap = max(0, min(w_end, s_end) - max(w_start, s_start))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_s_idx = s_idx + 1

            ratio = max_overlap / float(self.window_samples)
            label = 1 if ratio >= self.overlap_threshold else 0
            category = "ictal" if label == 1 else "interictal"

            eval_windows.append(signal[:, w_start:w_end])
            eval_meta.append(
                {
                    "start_sample": w_start,
                    "end_sample": w_end,
                    "start_sec": w_start / float(self.sfreq),
                    "end_sec": w_end / float(self.sfreq),
                    "overlap_ratio": ratio,
                    "seizure_idx": best_s_idx if label == 1 else None,
                    "label": label,
                    "category": category,
                }
            )

        if not eval_windows:
            empty_windows = np.empty((0, n_channels, self.window_samples), dtype=np.float32)
            empty_labels = np.empty((0,), dtype=np.int64)
            return empty_windows, empty_labels, pd.DataFrame(eval_meta)

        windows_arr = np.stack(eval_windows, axis=0).astype(np.float32)
        labels_arr = np.array([m["label"] for m in eval_meta], dtype=np.int64)
        meta_df = pd.DataFrame(eval_meta)

        return windows_arr, labels_arr, meta_df

    @staticmethod
    def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping or adjacent intervals."""
        if not intervals:
            return []
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged = [sorted_intervals[0]]
        for current in sorted_intervals[1:]:
            prev_start, prev_end = merged[-1]
            if current[0] <= prev_end:
                merged[-1] = (prev_start, max(prev_end, current[1]))
            else:
                merged.append(current)
        return merged
