"""EDF loading, channel standardization, and preprocessing module."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import mne
import numpy as np
from scipy.signal import butter, filtfilt

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    BANDPASS_HIGH,
    BANDPASS_LOW,
    CANONICAL_CHANNELS,
    DATASET_DIR,
    FILTER_ORDER,
    SAMPLING_RATE,
)


class EDFLoader:
    """Memory-safe loader and standardizer for CHB-MIT EDF recordings."""

    def __init__(
        self,
        canonical_channels: List[str] = CANONICAL_CHANNELS,
        target_sfreq: int = SAMPLING_RATE,
        bandpass_low: float = BANDPASS_LOW,
        bandpass_high: float = BANDPASS_HIGH,
        filter_order: int = FILTER_ORDER,
    ):
        self.canonical_channels = list(canonical_channels)
        self.target_sfreq = target_sfreq
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.filter_order = filter_order

        # Precompute Butterworth filter coefficients
        nyquist = 0.5 * self.target_sfreq
        low = self.bandpass_low / nyquist
        high = self.bandpass_high / nyquist
        self.b, self.a = butter(self.filter_order, [low, high], btype="bandpass")

    def inspect_header(self, edf_path: Union[str, Path]) -> Dict:
        """Inspect EDF header without loading data into memory.

        Returns:
            Dictionary with sfreq, n_times, duration_sec, ch_names, channel_map.
        """
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
        sfreq = raw.info["sfreq"]
        n_times = raw.n_times
        duration_sec = n_times / sfreq

        channel_map = self._resolve_channel_indices(raw.ch_names)

        return {
            "sfreq": sfreq,
            "n_times": n_times,
            "duration_sec": duration_sec,
            "raw_channels": raw.ch_names,
            "channel_map": channel_map,
            "canonical_channels": self.canonical_channels,
        }

    def _resolve_channel_indices(self, raw_ch_names: List[str]) -> List[int]:
        """Map canonical 18 channels to exact raw channel indices.

        Deterministic duplicate policy:
        In CHB-MIT, T8-P8 appears twice in many recordings.
        The first occurrence sits within the Right Temporal longitudinal chain
        (FP2-F8 -> F8-T8 -> T8-P8 -> P8-O2).
        The second occurrence is an auxiliary channel in the modified inferior chain.
        We strictly select the FIRST occurrence.
        """
        # Clean raw channel names: uppercase, strip whitespace, remove running numbers like -0, -1
        cleaned_raw = [
            ch.replace("-0", "").replace("-1", "").replace("-2", "").strip().upper()
            for ch in raw_ch_names
        ]

        channel_indices: List[int] = []
        missing_channels: List[str] = []

        for target in self.canonical_channels:
            target_clean = target.strip().upper()
            found = False
            for idx, raw_ch in enumerate(cleaned_raw):
                if raw_ch == target_clean:
                    channel_indices.append(idx)
                    found = True
                    break  # Break on first occurrence (deterministic policy)
            if not found:
                missing_channels.append(target)

        if missing_channels:
            raise ValueError(
                f"Recording is missing required canonical channels: {missing_channels}. "
                f"Available channels: {raw_ch_names}"
            )

        return channel_indices

    def load_slice(
        self,
        edf_path: Union[str, Path],
        start_sample: int,
        stop_sample: int,
        apply_filter: bool = True,
        apply_standardization: bool = True,
    ) -> np.ndarray:
        """Load a specific sample slice for the canonical 18 channels without loading the full file.

        Args:
            edf_path: Path to EDF file
            start_sample: 0-indexed start sample index
            stop_sample: 0-indexed stop sample index (exclusive)
            apply_filter: Whether to apply zero-phase 0.5-45 Hz bandpass filter
            apply_standardization: Whether to apply per-channel z-score standardization

        Returns:
            np.ndarray of shape (18, stop_sample - start_sample)
        """
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
        ch_indices = self._resolve_channel_indices(raw.ch_names)

        # Slice only target channels and samples directly from disk
        data, _ = raw[ch_indices, start_sample:stop_sample]

        if apply_filter:
            data = self.filter_signal(data)

        if apply_standardization:
            data = self.standardize_signal(data)

        return data

    def load_full_recording(
        self,
        edf_path: Union[str, Path],
        apply_filter: bool = True,
        apply_standardization: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """Load and preprocess all canonical channels for a complete recording.

        Args:
            edf_path: Path to EDF file
            apply_filter: Whether to apply bandpass filter
            apply_standardization: Whether to apply z-score scaling

        Returns:
            Tuple of (preprocessed_data, info_dict)
            preprocessed_data shape: (18, total_samples)
        """
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
        ch_indices = self._resolve_channel_indices(raw.ch_names)

        # Read only canonical channels into memory
        data, _ = raw[ch_indices, :]

        if apply_filter:
            data = self.filter_signal(data)

        if apply_standardization:
            data = self.standardize_signal(data)

        info = {
            "sfreq": raw.info["sfreq"],
            "n_times": raw.n_times,
            "duration_sec": raw.n_times / raw.info["sfreq"],
            "channels": self.canonical_channels,
            "file_path": str(edf_path),
        }

        return data, info

    def filter_signal(self, data: np.ndarray) -> np.ndarray:
        """Apply zero-phase Butterworth bandpass filter (0.5 - 45 Hz) along time axis."""
        # filtfilt applies forward-backward filtering for zero phase distortion
        filtered = filtfilt(self.b, self.a, data, axis=-1)
        return filtered

    def standardize_signal(self, data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Apply channel-wise z-score standardization: (x - mean) / (std + eps)."""
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        standardized = (data - mean) / (std + eps)
        return standardized
