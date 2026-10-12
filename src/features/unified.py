"""Unified Multimodal Feature Extractor.

Fuses:
1. Conventional Univariate Features (252 features) from EEGFeatureExtractor
2. Functional Connectivity Features (200 features: 112 PLV, 88 PLI) from EEGConnectivityExtractor

Total feature dimension: 452 features.

Provides clean column slicing utilities for systematic ablation studies:
- 'conventional': 252 features
- 'plv_only': 112 features
- 'pli_only': 88 features
- 'connectivity': 200 features (PLV + PLI)
- 'combined': 452 features (Conventional + PLV + PLI)
- 'conv_plus_plv': 364 features
- 'conv_plus_pli': 340 features
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.features.connectivity import EEGConnectivityExtractor
from src.features.feature_extractor import EEGFeatureExtractor
from src.utils.config import SAMPLING_RATE


class UnifiedFeatureExtractor:
    """Fuses univariate and functional connectivity feature extractors."""

    def __init__(self, sampling_rate: int = SAMPLING_RATE):
        self.sfreq = int(sampling_rate)
        self.conv_extractor = EEGFeatureExtractor(sampling_rate=self.sfreq)
        self.conn_extractor = EEGConnectivityExtractor(sampling_rate=self.sfreq)

        self.conv_feature_names: List[str] = list(self.conv_extractor.feature_names)  # 252
        self.conn_feature_names: List[str] = list(self.conn_extractor.feature_names)  # 200

        self.plv_feature_names: List[str] = [f for f in self.conn_feature_names if "plv" in f]  # 112
        self.pli_feature_names: List[str] = [f for f in self.conn_feature_names if "pli" in f]  # 88

        self.feature_names: List[str] = self.conv_feature_names + self.conn_feature_names  # 452
        self.feature_count = len(self.feature_names)

        # Index maps for instantaneous zero-copy slicing
        name_to_idx = {name: idx for idx, name in enumerate(self.feature_names)}
        self.subset_indices: Dict[str, np.ndarray] = {
            "conventional": np.array([name_to_idx[n] for n in self.conv_feature_names], dtype=int),
            "connectivity": np.array([name_to_idx[n] for n in self.conn_feature_names], dtype=int),
            "plv_only": np.array([name_to_idx[n] for n in self.plv_feature_names], dtype=int),
            "pli_only": np.array([name_to_idx[n] for n in self.pli_feature_names], dtype=int),
            "combined": np.arange(self.feature_count, dtype=int),
            "conv_plus_plv": np.array(
                [name_to_idx[n] for n in (self.conv_feature_names + self.plv_feature_names)], dtype=int
            ),
            "conv_plus_pli": np.array(
                [name_to_idx[n] for n in (self.conv_feature_names + self.pli_feature_names)], dtype=int
            ),
        }

    def extract_features(self, windows: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Extract both univariate and connectivity features for batch of windows.

        Args:
            windows: Array of shape (N, 18, 1024)
            batch_size: Window batch size for memory-safe connectivity extraction

        Returns:
            2D array of shape (N, 452)
        """
        conv_X = self.conv_extractor.extract_features(windows)
        conn_X = self.conn_extractor.extract_features(windows, batch_size=batch_size)
        fused = np.concatenate([conv_X, conn_X], axis=1).astype(np.float32)
        assert fused.shape[1] == self.feature_count, (
            f"Expected {self.feature_count} features, got {fused.shape[1]}"
        )
        return fused

    def slice_features(self, X: np.ndarray, subset: str) -> np.ndarray:
        """Slice feature matrix according to target ablation condition.

        Args:
            X: Matrix of shape (N, 452)
            subset: Condition key ('conventional', 'connectivity', 'plv_only', etc.)

        Returns:
            Sliced matrix of shape (N, subset_dim)
        """
        if subset not in self.subset_indices:
            raise ValueError(f"Unknown feature subset '{subset}'. Valid: {list(self.subset_indices.keys())}")
        indices = self.subset_indices[subset]
        return X[:, indices]

    def get_subset_names(self, subset: str) -> List[str]:
        """Return list of feature names for given subset."""
        if subset not in self.subset_indices:
            raise ValueError(f"Unknown feature subset '{subset}'. Valid: {list(self.subset_indices.keys())}")
        indices = self.subset_indices[subset]
        return [self.feature_names[i] for i in indices]
