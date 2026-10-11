"""Feature extraction module for multi-channel EEG windows.

Computes 14 time and frequency domain features per channel across the 18 canonical channels,
yielding a 252-dimensional tabular feature vector per 4-second EEG window.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import CANONICAL_CHANNELS, SAMPLING_RATE


class EEGFeatureExtractor:
    """Extracts 252 time/frequency features from (N, 18, 1024) EEG window tensors."""

    def __init__(
        self,
        sampling_rate: int = SAMPLING_RATE,
        canonical_channels: List[str] = CANONICAL_CHANNELS,
    ):
        self.sfreq = sampling_rate
        self.channels = canonical_channels
        self.n_channels = len(self.channels)

        # Standard EEG frequency bands (Hz)
        self.bands = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 12.0),
            "beta": (12.0, 30.0),
            "gamma": (30.0, 45.0),
        }
        self.feature_names = self._generate_feature_names()

    def _generate_feature_names(self) -> List[str]:
        """Generate descriptive column names for all 252 features."""
        feat_types = [
            "mean",
            "variance",
            "skewness",
            "kurtosis",
            "rms",
            "line_length",
            "zero_crossings",
            "delta_power",
            "theta_power",
            "alpha_power",
            "beta_power",
            "gamma_power",
            "spectral_entropy",
            "spectral_edge_95",
        ]
        names = []
        for ch in self.channels:
            for ft in feat_types:
                names.append(f"{ch}_{ft}")
        return names

    def extract_features(self, windows: np.ndarray) -> np.ndarray:
        """Extract features from a batch of EEG windows.

        Args:
            windows: np.ndarray of shape (N, n_channels, n_samples)
                     e.g. (N, 18, 1024)

        Returns:
            np.ndarray of shape (N, n_channels * 14) = (N, 252)
        """
        if windows.ndim != 3:
            raise ValueError(f"Expected 3D tensor (N, channels, samples), got shape {windows.shape}")

        N, n_ch, n_samples = windows.shape
        if n_ch != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {n_ch}")

        features_list: List[np.ndarray] = []

        # 1. Time-Domain Features
        # Mean
        feat_mean = np.mean(windows, axis=-1)  # (N, n_ch)
        # Variance
        feat_var = np.var(windows, axis=-1)   # (N, n_ch)
        # Skewness
        feat_skew = skew(windows, axis=-1, nan_policy="omit")  # (N, n_ch)
        # Kurtosis
        feat_kurt = kurtosis(windows, axis=-1, nan_policy="omit")  # (N, n_ch)
        # RMS
        feat_rms = np.sqrt(np.mean(windows ** 2, axis=-1))  # (N, n_ch)
        # Line Length (total variation)
        feat_ll = np.sum(np.abs(np.diff(windows, axis=-1)), axis=-1)  # (N, n_ch)
        # Zero-Crossing Rate
        signs = np.sign(windows)
        signs[signs == 0] = 1  # handle exact zeros
        zero_crossings = np.sum(np.abs(np.diff(signs, axis=-1)) > 0, axis=-1) / float(n_samples)  # (N, n_ch)

        # 2. Frequency-Domain Features via Welch PSD
        # nperseg=256 gives 1.0s window for 1.0 Hz frequency resolution
        nperseg = min(256, n_samples)
        freqs, psd = welch(windows, fs=self.sfreq, nperseg=nperseg, axis=-1)  # psd shape: (N, n_ch, n_freqs)

        # Total power between 0.5 and 45 Hz
        valid_idx = (freqs >= 0.5) & (freqs <= 45.0)
        valid_freqs = freqs[valid_idx]
        valid_psd = psd[:, :, valid_idx]  # (N, n_ch, n_valid_freqs)
        total_power = np.sum(valid_psd, axis=-1, keepdims=True) + 1e-12

        # Relative Band Powers
        band_powers = []
        for band_name, (f_low, f_high) in self.bands.items():
            b_idx = (valid_freqs >= f_low) & (valid_freqs <= f_high)
            b_power = np.sum(valid_psd[:, :, b_idx], axis=-1) / total_power.squeeze(-1)
            band_powers.append(b_power)  # list of (N, n_ch)

        # Spectral Entropy
        psd_norm = valid_psd / total_power
        # Replace 0s with 1e-12 before log to avoid NaN
        psd_norm = np.clip(psd_norm, 1e-12, 1.0)
        spec_entropy = -np.sum(psd_norm * np.log(psd_norm), axis=-1) / np.log(valid_psd.shape[-1])  # (N, n_ch)

        # Spectral Edge Frequency (95% power frequency)
        cum_power = np.cumsum(valid_psd, axis=-1) / total_power
        # Find frequency where cumulative power exceeds 0.95
        # Vectorized across N and n_ch
        edge_idx = np.argmax(cum_power >= 0.95, axis=-1)
        spec_edge_95 = valid_freqs[edge_idx]  # (N, n_ch)

        # Stack per-channel features into (N, n_ch, 14)
        # Order: mean, var, skew, kurt, rms, ll, zcr, delta, theta, alpha, beta, gamma, sentropy, se95
        ch_features = np.stack(
            [
                feat_mean,
                feat_var,
                feat_skew,
                feat_kurt,
                feat_rms,
                feat_ll,
                zero_crossings,
                band_powers[0],  # delta
                band_powers[1],  # theta
                band_powers[2],  # alpha
                band_powers[3],  # beta
                band_powers[4],  # gamma
                spec_entropy,
                spec_edge_95,
            ],
            axis=-1,
        )  # shape: (N, n_ch, 14)

        # Flatten channel and feature dimensions: (N, n_ch * 14) = (N, 252)
        feature_matrix = ch_features.reshape(N, n_ch * 14).astype(np.float32)

        # Safety: replace any unexpected NaNs or Infs
        if np.isnan(feature_matrix).any() or np.isinf(feature_matrix).any():
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)

        return feature_matrix
