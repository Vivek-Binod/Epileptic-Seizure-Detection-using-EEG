import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    BANDPASS_HIGH,
    BANDPASS_LOW,
    CANONICAL_CHANNELS,
    FILTER_ORDER,
    PATIENT_ID_MAPPING,
    SAMPLING_RATE,
)
from src.data.loader import EDFLoader
from src.features.connectivity import EEGConnectivityExtractor
from src.features.network_metrics import (
    clustering_coefficient_weighted,
    eigenvector_centrality,
    global_efficiency_weighted,
    node_strengths,
    threshold_proportional,
)


def test_config_consistency():
    """Verify central configuration invariants documented in the audit."""
    assert SAMPLING_RATE == 256
    assert BANDPASS_LOW == 0.5
    assert BANDPASS_HIGH == 45.0  # Confirms 45.0 Hz cutoff
    assert FILTER_ORDER == 4
    assert len(CANONICAL_CHANNELS) == 18
    # Biological patient unification
    assert PATIENT_ID_MAPPING["chb01"] == "patient_01"
    assert PATIENT_ID_MAPPING["chb21"] == "patient_01"
    assert len(set(PATIENT_ID_MAPPING.values())) == 23


def test_loader_duplicate_t8_p8_resolution():
    """Verify deterministic selection of first T8-P8 occurrence."""
    loader = EDFLoader()
    # Synthetic channel list with duplicate T8-P8 and dummy channels
    raw_channels = list(CANONICAL_CHANNELS) + ["T8-P8-1", "--0", "--1", "VNS"]
    indices = loader._resolve_channel_indices(raw_channels)
    assert len(indices) == 18
    # Index 16 is T8-P8 in CANONICAL_CHANNELS, should map to index 16 in raw_channels
    assert raw_channels[indices[16]].replace("-0", "") == "T8-P8"
    assert indices[16] == 16  # First occurrence, not index 18


def test_adjacency_matrix_mathematical_properties():
    """Verify symmetry, zero-diagonal, bounded range, and zero NaNs on synthetic signals."""
    extractor = EEGConnectivityExtractor()
    np.random.seed(42)
    # Synthetic batch of 2 windows, 18 channels, 1024 samples
    synthetic_windows = np.random.randn(2, 18, 1024).astype(np.float32)
    matrices = extractor.compute_full_matrices(synthetic_windows)
    
    for band in ["delta", "theta", "alpha", "beta"]:
        for conn_type in ["plv", "pli"]:
            batch_mat = matrices[band][conn_type]
            assert batch_mat.shape == (2, 18, 18)
            for w in range(2):
                A = batch_mat[w]
                # 1. Strict symmetry
                assert np.allclose(A, A.T, atol=1e-6)
                # 2. Zero diagonal
                assert np.allclose(np.diag(A), 0.0)
                # 3. Bounded [0, 1]
                assert np.all(A >= 0.0) and np.all(A <= 1.0)
                # 4. Zero NaNs or Infs
                assert not np.isnan(A).any()
                assert not np.isinf(A).any()


def test_edge_count_and_triu_indices():
    """Verify exactly 153 unique upper-triangular edge pairs for 18 channels."""
    extractor = EEGConnectivityExtractor()
    assert len(extractor.triu_i) == 153
    assert len(extractor.triu_j) == 153
    assert extractor.n_pairs == 153
    # Check all pairs have i < j
    assert np.all(extractor.triu_i < extractor.triu_j)


def test_graph_theoretical_metrics_contract():
    """Verify graph theoretical scalar metric calculation contracts."""
    # Synthetic symmetric 18x18 adjacency matrix with zero diagonal
    np.random.seed(42)
    A = np.random.uniform(0.1, 0.9, size=(18, 18)).astype(np.float32)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    
    strengths = node_strengths(A)
    assert strengths.shape == (18,)
    assert np.all(strengths > 0)
    
    clust = clustering_coefficient_weighted(A)
    assert clust.shape == (18,)
    assert np.all(clust >= 0.0) and np.all(clust <= 1.0)
    
    eff = global_efficiency_weighted(A)
    assert isinstance(eff, float)
    assert 0.0 <= eff <= 1.0
    
    cent = eigenvector_centrality(A)
    assert cent.shape == (18,)
    assert np.isclose(np.sum(cent), 1.0)
    
    B = threshold_proportional(A, density=0.20)
    assert np.allclose(B, B.T)
    assert np.allclose(np.diag(B), 0.0)
    # Check exact retained edge count: round(0.20 * 153) = 31 edges in upper triangle
    tri_i, tri_j = np.triu_indices(18, k=1)
    retained_edges = np.sum(B[tri_i, tri_j] > 0)
    assert retained_edges == int(np.round(0.20 * 153))
