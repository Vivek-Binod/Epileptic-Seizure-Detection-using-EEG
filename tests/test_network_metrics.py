"""Unit and regression tests for graph theoretical network metrics."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.connectivity import EEGConnectivityExtractor
from src.features.network_metrics import (
    clustering_coefficient_binary,
    clustering_coefficient_weighted,
    compute_network_summary,
    compute_window_graph_features,
    eigenvector_centrality,
    global_efficiency_binary,
    global_efficiency_weighted,
    node_strengths,
    threshold_proportional,
)


def test_synthetic_phase_relationships():
    """Verify PLV and PLI against ground-truth synthetic phase relationships."""
    extractor = EEGConnectivityExtractor(sampling_rate=256)
    t = np.linspace(0, 4, 1024, endpoint=False)
    f0 = 6.0  # Theta band (4-8 Hz)

    # 1. Zero-lag in-phase sine waves
    s1 = np.sin(2 * np.pi * f0 * t)
    s2 = np.sin(2 * np.pi * f0 * t)

    # 2. Quadrature sine waves (pi/2 phase lag)
    s_quad_1 = np.sin(2 * np.pi * f0 * t)
    s_quad_2 = np.cos(2 * np.pi * f0 * t)

    win = np.zeros((2, 18, 1024), dtype=np.float32)
    win[0, 0] = s1
    win[0, 1] = s2
    win[1, 0] = s_quad_1
    win[1, 1] = s_quad_2

    plv, pli = extractor._compute_band_matrices(win, "theta")

    # In-phase: PLV=1.0, PLI=0.0
    assert np.isclose(plv[0, 0], 1.0, atol=1e-3)
    assert np.isclose(pli[0, 0], 0.0, atol=1e-3)

    # Quadrature: PLV~1.0, PLI~1.0
    assert np.isclose(plv[1, 0], 1.0, atol=0.02)
    assert np.isclose(pli[1, 0], 1.0, atol=1e-3)


def test_graph_metrics_ideal_networks():
    """Verify graph theoretical measures on known canonical network topologies."""
    # Complete graph K_4: all edge weights = 1.0
    K4 = np.ones((4, 4), dtype=np.float32)
    np.fill_diagonal(K4, 0.0)

    # 1. Node strength: each node has degree 3
    s = node_strengths(K4)
    np.testing.assert_allclose(s, [3.0, 3.0, 3.0, 3.0])

    # 2. Clustering: in K4, every triangle is present -> C = 1.0
    c_bin = clustering_coefficient_binary(K4)
    c_wt = clustering_coefficient_weighted(K4)
    np.testing.assert_allclose(c_bin, 1.0)
    np.testing.assert_allclose(c_wt, 1.0)

    # 3. Global efficiency: every distance is 1.0 -> E_glob = 1.0
    e_bin = global_efficiency_binary(K4)
    e_wt = global_efficiency_weighted(K4)
    assert np.isclose(e_bin, 1.0)
    assert np.isclose(e_wt, 1.0)

    # 4. Eigenvector centrality: symmetric -> equal centrality = 0.25
    ev = eigenvector_centrality(K4)
    np.testing.assert_allclose(ev, 0.25)


def test_threshold_proportional_density():
    """Verify proportional thresholding preserves exact density."""
    N = 18
    A = np.random.rand(N, N).astype(np.float32)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)

    n_pairs = N * (N - 1) // 2  # 153
    target_density = 0.20
    expected_edges = int(np.round(target_density * n_pairs))  # 31

    B = threshold_proportional(A, density=target_density)
    tri_i, tri_j = np.triu_indices(N, k=1)
    actual_edges = int(np.sum(B[tri_i, tri_j]))
    assert actual_edges == expected_edges


def test_compute_full_matrices_and_summary():
    """Verify full matrix extraction and tabular summary extraction."""
    extractor = EEGConnectivityExtractor(sampling_rate=256)
    dummy_windows = np.random.randn(3, 18, 1024).astype(np.float32)

    mat_dict = extractor.compute_full_matrices(dummy_windows)
    assert "theta" in mat_dict
    assert mat_dict["theta"]["plv"].shape == (3, 18, 18)
    assert mat_dict["theta"]["pli"].shape == (3, 18, 18)

    feat_mat, feat_names = compute_window_graph_features(mat_dict, densities=[0.20])
    assert feat_mat.shape[0] == 3
    assert len(feat_names) == feat_mat.shape[1]
    assert not np.isnan(feat_mat).any()
    assert not np.isinf(feat_mat).any()
