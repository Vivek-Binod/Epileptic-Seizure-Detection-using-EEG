"""Graph Theoretical Network Metrics for Functional EEG Connectivity Matrices.

Computes:
1. Nodal Strength / Degree:
   - Weighted node strength s_i = sum_j A_ij
   - Mean network strength S = 1/N sum_i s_i
2. Clustering Coefficient:
   - Binary clustering C_i = (B^3)_ii / (k_i * (k_i - 1))
   - Weighted clustering (Onnela et al., 2005): C_i^w = (W^3)_ii / (k_i * (k_i - 1)), where W = A^(1/3)
3. Global Efficiency (Latora & Marchiori, 2001):
   - Harmonic mean of shortest path lengths: E_glob = 1/(N*(N-1)) sum_{i != j} 1 / d_ij
   - Computed via Dijkstra's shortest path on distance matrix L_ij = 1 / A_ij
4. Centrality Measures:
   - Eigenvector centrality (Bonacich, 1987) via principal eigenvector of A
5. Proportional Density Thresholding:
   - Preserves exact connection density across graphs to eliminate wiring cost confounds.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import shortest_path


def threshold_proportional(A: np.ndarray, density: float = 0.20) -> np.ndarray:
    """Threshold adjacency matrix to retain exact top proportion of connections.

    Args:
        A: Symmetric 2D adjacency matrix (N, N) with zeros on diagonal
        density: Proportion of upper-triangle edges to retain in (0.0, 1.0]

    Returns:
        Binary symmetric adjacency matrix B (N, N) with zero diagonal.
    """
    N = A.shape[0]
    tri_i, tri_j = np.triu_indices(N, k=1)
    edge_vals = A[tri_i, tri_j]
    n_edges = len(edge_vals)
    k_retain = max(1, int(np.round(density * n_edges)))

    # Find cutoff value
    cutoff = np.partition(edge_vals, -k_retain)[-k_retain]
    B = np.zeros_like(A, dtype=np.float32)

    # In case of ties, strictly enforce top k_retain
    sorted_order = np.argsort(edge_vals)[::-1][:k_retain]
    B[tri_i[sorted_order], tri_j[sorted_order]] = 1.0
    B[tri_j[sorted_order], tri_i[sorted_order]] = 1.0
    return B


def node_strengths(A: np.ndarray) -> np.ndarray:
    """Compute nodal connection strength (sum of incident weights)."""
    return np.sum(A, axis=-1)


def clustering_coefficient_weighted(A: np.ndarray) -> np.ndarray:
    """Compute nodal weighted clustering coefficient (Onnela et al., 2005).

    Formula:
        C_i^w = (W^3)_ii / (k_i * (k_i - 1))
        where W = A^(1/3), k_i = count of nonzero connections.
    """
    A_clean = np.copy(A)
    np.fill_diagonal(A_clean, 0.0)

    # Normalize weights by maximum edge if max > 1.0
    max_w = np.max(A_clean)
    if max_w > 1.0:
        A_clean /= max_w

    W = np.cbrt(A_clean)
    W3 = W @ W @ W
    triangles = np.diag(W3)

    k = np.sum(A_clean > 0, axis=-1)
    deg_denom = k * (k - 1)

    c = np.zeros(A.shape[0], dtype=np.float32)
    valid = deg_denom > 0
    c[valid] = triangles[valid] / deg_denom[valid]
    return np.clip(c, 0.0, 1.0)


def clustering_coefficient_binary(B: np.ndarray) -> np.ndarray:
    """Compute nodal binary clustering coefficient."""
    B_clean = np.copy(B)
    np.fill_diagonal(B_clean, 0.0)
    B_bin = (B_clean > 0).astype(np.float32)

    triangles = np.diag(B_bin @ B_bin @ B_bin)
    k = np.sum(B_bin, axis=-1)
    deg_denom = k * (k - 1)

    c = np.zeros(B.shape[0], dtype=np.float32)
    valid = deg_denom > 0
    c[valid] = triangles[valid] / deg_denom[valid]
    return np.clip(c, 0.0, 1.0)


def global_efficiency_weighted(A: np.ndarray) -> float:
    """Compute weighted global efficiency (Latora & Marchiori, 2001).

    Efficiency is the average inverse shortest path length across all node pairs:
        E_glob = 1 / (N * (N - 1)) * sum_{i != j} (1 / d_ij)
    where path distance L_ij = 1 / A_ij for positive weights.
    """
    N = A.shape[0]
    if N < 2:
        return 0.0

    A_clean = np.copy(A)
    np.fill_diagonal(A_clean, 0.0)

    # Length matrix: inversely proportional to coupling strength
    with np.errstate(divide="ignore"):
        L = np.where(A_clean > 0, 1.0 / A_clean, np.inf)
    np.fill_diagonal(L, 0.0)

    dists = shortest_path(L, directed=False)
    np.fill_diagonal(dists, np.inf)

    # Inverse shortest path distances (0 if disconnected)
    inv_dists = np.where(np.isinf(dists) | (dists == 0), 0.0, 1.0 / dists)
    return float(np.sum(inv_dists) / (N * (N - 1)))


def global_efficiency_binary(B: np.ndarray) -> float:
    """Compute binary global efficiency."""
    N = B.shape[0]
    if N < 2:
        return 0.0

    B_clean = np.copy(B)
    np.fill_diagonal(B_clean, 0.0)

    dists = shortest_path((B_clean > 0).astype(np.float32), directed=False, unweighted=True)
    np.fill_diagonal(dists, np.inf)

    inv_dists = np.where(np.isinf(dists) | (dists == 0), 0.0, 1.0 / dists)
    return float(np.sum(inv_dists) / (N * (N - 1)))


def eigenvector_centrality(A: np.ndarray) -> np.ndarray:
    """Compute eigenvector centrality via principal eigenvector."""
    A_clean = np.copy(A)
    np.fill_diagonal(A_clean, 0.0)

    # For symmetric matrices, eigh returns eigenvalues in ascending order
    eigvals, eigvecs = eigh(A_clean)
    principal_vec = np.abs(eigvecs[:, -1])
    vec_sum = np.sum(principal_vec)
    if vec_sum > 0:
        principal_vec /= vec_sum
    return principal_vec.astype(np.float32)


def compute_network_summary(
    A: np.ndarray,
    density: Optional[float] = None,
    weighted: bool = True,
) -> Dict[str, Union[float, np.ndarray]]:
    """Compute comprehensive graph theoretical measures for an adjacency matrix.

    Args:
        A: Symmetric 2D adjacency matrix (N, N)
        density: Optional proportional threshold density in (0.0, 1.0]
        weighted: If True, computes weighted metrics; if False, binarizes at density

    Returns:
        Dictionary of scalar network metrics and nodal metric vectors.
    """
    N = A.shape[0]
    A_clean = np.copy(A).astype(np.float32)
    np.fill_diagonal(A_clean, 0.0)

    if density is not None:
        B = threshold_proportional(A_clean, density=density)
        if weighted:
            # Weighted thresholded network
            W_mat = A_clean * B
            strengths = node_strengths(W_mat)
            clustering = clustering_coefficient_weighted(W_mat)
            efficiency = global_efficiency_weighted(W_mat)
            centrality = eigenvector_centrality(W_mat)
        else:
            # Binary thresholded network
            strengths = node_strengths(B)
            clustering = clustering_coefficient_binary(B)
            efficiency = global_efficiency_binary(B)
            centrality = eigenvector_centrality(B)
    else:
        # Full weighted network (unthresholded)
        strengths = node_strengths(A_clean)
        clustering = clustering_coefficient_weighted(A_clean)
        efficiency = global_efficiency_weighted(A_clean)
        centrality = eigenvector_centrality(A_clean)

    return {
        "mean_strength": float(np.mean(strengths)),
        "std_strength": float(np.std(strengths)),
        "max_strength": float(np.max(strengths)),
        "mean_clustering": float(np.mean(clustering)),
        "global_efficiency": float(efficiency),
        "max_centrality": float(np.max(centrality)),
        "std_centrality": float(np.std(centrality)),
        "nodal_strength": strengths,
        "nodal_clustering": clustering,
        "nodal_centrality": centrality,
    }


def compute_window_graph_features(
    full_matrices_by_band: Dict[str, Dict[str, np.ndarray]],
    densities: List[float] = [0.20],
) -> Tuple[np.ndarray, List[str]]:
    """Extract standard tabular graph features across bands and connectivity types.

    Args:
        full_matrices_by_band: Nested dict:
            band_name -> 'plv': (N, C, C), 'pli': (N, C, C)
        densities: List of proportional densities for thresholded metrics

    Returns:
        Tuple of (feature_matrix of shape (N, D), list of D feature names)
    """
    N_windows = next(iter(next(iter(full_matrices_by_band.values())).values())).shape[0]
    feature_cols: List[np.ndarray] = []
    feature_names: List[str] = []

    scalar_keys = [
        "mean_strength",
        "std_strength",
        "mean_clustering",
        "global_efficiency",
        "max_centrality",
    ]

    for band_name, type_dict in sorted(full_matrices_by_band.items()):
        for conn_type in ["plv", "pli"]:
            if conn_type not in type_dict:
                continue
            mat_batch = type_dict[conn_type]  # (N, C, C)

            # 1. Unthresholded Weighted Graph Metrics
            batch_scalars = {k: np.zeros(N_windows, dtype=np.float32) for k in scalar_keys}
            for w_idx in range(N_windows):
                summary = compute_network_summary(mat_batch[w_idx], density=None, weighted=True)
                for k in scalar_keys:
                    batch_scalars[k][w_idx] = summary[k]

            for k in scalar_keys:
                feature_cols.append(batch_scalars[k][:, np.newaxis])
                feature_names.append(f"graph_{conn_type}_{band_name}_{k}")

            # 2. Thresholded Network Metrics (density=0.20 default)
            for d in densities:
                d_str = f"d{int(d * 100)}"
                batch_d_scalars = {k: np.zeros(N_windows, dtype=np.float32) for k in scalar_keys}
                for w_idx in range(N_windows):
                    summary_d = compute_network_summary(mat_batch[w_idx], density=d, weighted=False)
                    for k in scalar_keys:
                        batch_d_scalars[k][w_idx] = summary_d[k]

                for k in scalar_keys:
                    feature_cols.append(batch_d_scalars[k][:, np.newaxis])
                    feature_names.append(f"graph_{conn_type}_{band_name}_{d_str}_{k}")

    feat_matrix = np.concatenate(feature_cols, axis=1) if feature_cols else np.empty((N_windows, 0))
    return feat_matrix, feature_names
