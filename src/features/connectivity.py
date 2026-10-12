"""EEG Functional Connectivity Feature Extractor.

Computes Phase Locking Value (PLV) and Phase Lag Index (PLI) across canonical
frequency bands for multi-channel scalp EEG windows.

Theoretical Foundation:
1. Phase Locking Value (PLV) (Lachaux et al., 1999; Wan Chen et al., 2024):
   Quantifies the consistency of phase differences between channel pairs:
   PLV_ij = | 1/N sum_{t=1}^N exp(j * (phi_i(t) - phi_j(t))) |
   Bounded in [0, 1]. Captures phase synchronization but is sensitive to
   instantaneous zero-lag volume conduction.

2. Phase Lag Index (PLI) (Stam et al., 2007; Chiarion et al., 2023):
   Quantifies the asymmetry of the phase difference distribution:
   PLI_ij = | 1/N sum_{t=1}^N sign(sin(phi_i(t) - phi_j(t))) |
   Bounded in [0, 1]. Invariant to zero-lag volume conduction artifacts.

3. Graph & Sub-network Topologies:
   Instead of solely exposing all 153 pairwise edges (which risks overfitting),
   this extractor computes:
   - Nodal connectivity strengths (mean connection weight per electrode)
   - Global network synchrony (mean, standard deviation, max connectivity)
   - Anatomical sub-network synchrony (left/right temporal, left/right parasagittal,
     inter-hemispheric homologous coupling, and hemispheric asymmetry)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.signal as signal

from src.utils.config import CANONICAL_CHANNELS, SAMPLING_RATE


class EEGConnectivityExtractor:
    """Extracts PLV and PLI functional connectivity features from EEG window batches."""

    # Default clinical frequency bands for connectivity analysis
    DEFAULT_BANDS = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
        "beta": (12.0, 30.0),
    }

    def __init__(
        self,
        sampling_rate: int = SAMPLING_RATE,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
        channels: Optional[List[str]] = None,
        filter_order: int = 3,
        include_nodal: bool = True,
        include_global: bool = True,
        include_hemispheric: bool = True,
        include_pairwise: bool = False,
    ):
        self.sfreq = float(sampling_rate)
        self.bands = bands if bands is not None else self.DEFAULT_BANDS
        self.channels = list(channels) if channels is not None else list(CANONICAL_CHANNELS)
        self.n_channels = len(self.channels)
        self.filter_order = filter_order

        self.include_nodal = include_nodal
        self.include_global = include_global
        self.include_hemispheric = include_hemispheric
        self.include_pairwise = include_pairwise

        # Precompute Upper-Triangle indices for all unique channel pairs
        self.triu_i, self.triu_j = np.triu_indices(self.n_channels, k=1)
        self.n_pairs = len(self.triu_i)  # 153 for 18 channels

        # Anatomical sub-network index masks based on 18 standard bipolar montage
        self._setup_subnetworks()

        # Precompute Second-Order Sections (SOS) bandpass filters for numerical stability
        self.sos_filters: Dict[str, np.ndarray] = {}
        nyquist = 0.5 * self.sfreq
        for band_name, (low, high) in self.bands.items():
            # Safeguard boundaries against Nyquist
            low_norm = max(0.001, low / nyquist)
            high_norm = min(0.999, high / nyquist)
            self.sos_filters[band_name] = signal.butter(
                self.filter_order,
                [low_norm, high_norm],
                btype="bandpass",
                output="sos",
            )

        # Build feature names
        self.feature_names = self._build_feature_names()

    def _setup_subnetworks(self):
        """Define anatomical channel group indices for subnetwork synchrony."""
        ch_to_idx = {ch: idx for idx, ch in enumerate(self.channels)}

        # 4 anatomical chains
        lt_names = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1"]
        lp_names = ["FP1-F3", "F3-C3", "C3-P3", "P3-O1"]
        rp_names = ["FP2-F4", "F4-C4", "C4-P4", "P4-O2"]
        rt_names = ["FP2-F8", "F8-T8", "T8-P8", "P8-O2"]

        self.lt_indices = [ch_to_idx[c] for c in lt_names if c in ch_to_idx]
        self.lp_indices = [ch_to_idx[c] for c in lp_names if c in ch_to_idx]
        self.rp_indices = [ch_to_idx[c] for c in rp_names if c in ch_to_idx]
        self.rt_indices = [ch_to_idx[c] for c in rt_names if c in ch_to_idx]

        # Homologous pairs across hemispheres
        homologous = [
            ("FP1-F7", "FP2-F8"), ("F7-T7", "F8-T8"), ("T7-P7", "T8-P8"), ("P7-O1", "P8-O2"),
            ("FP1-F3", "FP2-F4"), ("F3-C3", "F4-C4"), ("C3-P3", "C4-P4"), ("P3-O1", "P4-O2"),
        ]
        self.homologous_pairs = [
            (ch_to_idx[c1], ch_to_idx[c2]) for c1, c2 in homologous
            if c1 in ch_to_idx and c2 in ch_to_idx
        ]

    def _build_feature_names(self) -> List[str]:
        """Construct deterministic feature names matching extracted columns."""
        names: List[str] = []
        for band in self.bands:
            # 1. Nodal features
            if self.include_nodal:
                for ch in self.channels:
                    names.append(f"nodal_plv_{band}_{ch}")
                for ch in self.channels:
                    names.append(f"nodal_pli_{band}_{ch}")

            # 2. Global features
            if self.include_global:
                names.extend([
                    f"global_mean_plv_{band}",
                    f"global_std_plv_{band}",
                    f"global_max_plv_{band}",
                    f"global_mean_pli_{band}",
                    f"global_std_pli_{band}",
                    f"global_max_pli_{band}",
                ])

            # 3. Hemispheric sub-networks
            if self.include_hemispheric:
                names.extend([
                    f"intra_lt_plv_{band}",
                    f"intra_lp_plv_{band}",
                    f"intra_rp_plv_{band}",
                    f"intra_rt_plv_{band}",
                    f"interhemi_homologous_plv_{band}",
                    f"asym_temporal_plv_{band}",
                    f"asym_parasagittal_plv_{band}",
                    f"interhemi_homologous_pli_{band}",
                ])

            # 4. Pairwise features
            if self.include_pairwise:
                for i, j in zip(self.triu_i, self.triu_j):
                    names.append(f"plv_{band}_{self.channels[i]}_{self.channels[j]}")
                for i, j in zip(self.triu_i, self.triu_j):
                    names.append(f"pli_{band}_{self.channels[i]}_{self.channels[j]}")

        return names

    def _compute_band_matrices(
        self,
        windows: np.ndarray,
        band_name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized computation of PLV and PLI pairwise matrices for a single band.

        Args:
            windows: Batch array of shape (N, C, T)
            band_name: Name of frequency band

        Returns:
            Tuple of (plv_pairs, pli_pairs) each of shape (N, n_pairs)
        """
        sos = self.sos_filters[band_name]
        # Zero-phase IIR filtering along time axis
        filtered = signal.sosfiltfilt(sos, windows, axis=-1)

        # Analytic signal and instantaneous phase
        analytic = signal.hilbert(filtered, axis=-1)
        phases = np.angle(analytic).astype(np.float32)  # shape: (N, C, T)

        # Phase differences across all upper-triangle pairs: shape (N, n_pairs, T)
        diff = phases[:, self.triu_i, :] - phases[:, self.triu_j, :]

        # Numerical safeguard: trigonometric evaluation
        sin_diff = np.sin(diff)
        cos_diff = np.cos(diff)

        # PLV: magnitude of average phase vector: |1/T sum e^(j * diff)|
        mean_cos = np.mean(cos_diff, axis=-1)
        mean_sin = np.mean(sin_diff, axis=-1)
        plv_pairs = np.sqrt(mean_cos ** 2 + mean_sin ** 2)

        # PLI: asymmetry of phase difference distribution: |1/T sum sign(sin(diff))|
        pli_pairs = np.abs(np.mean(np.sign(sin_diff), axis=-1))

        # Clamp numerical boundaries to [0.0, 1.0]
        plv_pairs = np.clip(plv_pairs, 0.0, 1.0)
        pli_pairs = np.clip(pli_pairs, 0.0, 1.0)

        return plv_pairs, pli_pairs

    def _pairs_to_full_matrix(self, pairs: np.ndarray, N: int) -> np.ndarray:
        """Expand upper-triangle pair vector (N, n_pairs) into full symmetric (N, C, C) matrices."""
        mat = np.zeros((N, self.n_channels, self.n_channels), dtype=np.float32)
        mat[:, self.triu_i, self.triu_j] = pairs
        mat[:, self.triu_j, self.triu_i] = pairs
        return mat

    def extract_features(self, windows: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Extract all configured functional connectivity features for batch of windows.

        Splits processing into bounded chunks (default batch_size=256) to ensure
        memory consumption remains under 200 MB even on large continuous recordings
        (e.g., 14,000+ windows).

        Args:
            windows: NumPy array of shape (N, n_channels, n_samples)
            batch_size: Maximum number of windows to process in a single vectorized pass

        Returns:
            2D array of shape (N, n_features) with dtype float32.
        """
        if windows.ndim == 2:
            windows = np.expand_dims(windows, axis=0)

        N, C, T = windows.shape
        assert C == self.n_channels, (
            f"Expected {self.n_channels} channels, got {C} in input window batch"
        )

        if N > batch_size:
            chunks: List[np.ndarray] = []
            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                chunk_feats = self._extract_features_single_batch(windows[start_idx:end_idx])
                chunks.append(chunk_feats)
            return np.concatenate(chunks, axis=0)
        else:
            return self._extract_features_single_batch(windows)

    def _extract_features_single_batch(self, windows: np.ndarray) -> np.ndarray:
        """Process a single bounded batch of windows."""
        N, C, T = windows.shape
        all_band_features: List[np.ndarray] = []

        for band_name in self.bands:
            plv_pairs, pli_pairs = self._compute_band_matrices(windows, band_name)
            # Reconstruct full symmetric connectivity matrices for nodal/subnetwork aggregations
            full_plv = self._pairs_to_full_matrix(plv_pairs, N)
            full_pli = self._pairs_to_full_matrix(pli_pairs, N)

            band_feats: List[np.ndarray] = []

            # 1. Nodal connectivity strengths: mean connection per electrode
            if self.include_nodal:
                # Exclude self-loop by dividing by (C - 1)
                nodal_plv = np.sum(full_plv, axis=2) / (self.n_channels - 1.0)  # (N, C)
                nodal_pli = np.sum(full_pli, axis=2) / (self.n_channels - 1.0)  # (N, C)
                band_feats.append(nodal_plv)
                band_feats.append(nodal_pli)

            # 2. Global network synchrony
            if self.include_global:
                g_mean_plv = np.mean(plv_pairs, axis=1, keepdims=True)
                g_std_plv = np.std(plv_pairs, axis=1, keepdims=True)
                g_max_plv = np.max(plv_pairs, axis=1, keepdims=True)

                g_mean_pli = np.mean(pli_pairs, axis=1, keepdims=True)
                g_std_pli = np.std(pli_pairs, axis=1, keepdims=True)
                g_max_pli = np.max(pli_pairs, axis=1, keepdims=True)

                band_feats.extend([
                    g_mean_plv, g_std_plv, g_max_plv,
                    g_mean_pli, g_std_pli, g_max_pli,
                ])

            # 3. Anatomical subnetwork synchrony
            if self.include_hemispheric:
                # Subnetwork mean connectivity within chains
                intra_lt = self._subnetwork_mean(full_plv, self.lt_indices)
                intra_lp = self._subnetwork_mean(full_plv, self.lp_indices)
                intra_rp = self._subnetwork_mean(full_plv, self.rp_indices)
                intra_rt = self._subnetwork_mean(full_plv, self.rt_indices)

                # Inter-hemispheric homologous synchrony
                inter_plv = self._homologous_mean(full_plv)
                inter_pli = self._homologous_mean(full_pli)

                # Asymmetry indices: Left - Right
                asym_temp = intra_lt - intra_rt
                asym_para = intra_lp - intra_rp

                band_feats.extend([
                    intra_lt, intra_lp, intra_rp, intra_rt,
                    inter_plv, asym_temp, asym_para, inter_pli,
                ])

            # 4. Pairwise features
            if self.include_pairwise:
                band_feats.append(plv_pairs)
                band_feats.append(pli_pairs)

            all_band_features.append(np.concatenate(band_feats, axis=1))

        features = np.concatenate(all_band_features, axis=1).astype(np.float32)

        # Sanity validation
        assert features.shape[1] == len(self.feature_names), (
            f"Feature count mismatch: matrix has {features.shape[1]}, "
            f"expected {len(self.feature_names)}"
        )
        assert not np.isnan(features).any(), "NaN detected in extracted connectivity features"
        assert not np.isinf(features).any(), "Inf detected in extracted connectivity features"

        return features

    def _subnetwork_mean(self, full_mat: np.ndarray, indices: List[int]) -> np.ndarray:
        """Compute mean intra-subnetwork connection strength."""
        if len(indices) < 2:
            return np.zeros((full_mat.shape[0], 1), dtype=np.float32)
        sub = full_mat[:, indices, :][:, :, indices]
        k = len(indices)
        tri_i, tri_j = np.triu_indices(k, k=1)
        return np.mean(sub[:, tri_i, tri_j], axis=1, keepdims=True)

    def _homologous_mean(self, full_mat: np.ndarray) -> np.ndarray:
        """Compute mean connection strength across homologous inter-hemispheric pairs."""
        if not self.homologous_pairs:
            return np.zeros((full_mat.shape[0], 1), dtype=np.float32)
        vals = [full_mat[:, i, j] for i, j in self.homologous_pairs]
        return np.mean(np.column_stack(vals), axis=1, keepdims=True)
