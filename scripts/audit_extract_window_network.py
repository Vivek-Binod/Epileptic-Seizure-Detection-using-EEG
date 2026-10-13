"""Extract real representative window from chb01_03.edf, compute full 18x18 PLV/PLI matrices, export edge table and plots."""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import CANONICAL_CHANNELS, DATASET_DIR, SAMPLING_RATE
from src.data.loader import EDFLoader
from src.features.connectivity import EEGConnectivityExtractor
from src.features.network_metrics import (
    compute_network_summary,
    compute_window_graph_features,
    threshold_proportional,
    node_strengths,
)

def classify_anatomical_relation(ch_i: str, ch_j: str) -> str:
    """Classify anatomical pair relation in 18-channel bipolar montage."""
    lt = {"FP1-F7", "F7-T7", "T7-P7", "P7-O1"}
    lp = {"FP1-F3", "F3-C3", "C3-P3", "P3-O1"}
    ml = {"FZ-CZ", "CZ-PZ"}
    rp = {"FP2-F4", "F4-C4", "C4-P4", "P4-O2"}
    rt = {"FP2-F8", "F8-T8", "T8-P8", "P8-O2"}
    
    homologous = {
        ("FP1-F7", "FP2-F8"), ("F7-T7", "F8-T8"), ("T7-P7", "T8-P8"), ("P7-O1", "P8-O2"),
        ("FP1-F3", "FP2-F4"), ("F3-C3", "F4-C4"), ("C3-P3", "C4-P4"), ("P3-O1", "P4-O2"),
    }
    
    if (ch_i, ch_j) in homologous or (ch_j, ch_i) in homologous:
        return "interhemispheric_homologous"
    
    if ch_i in lt and ch_j in lt:
        return "intra_left_temporal"
    if ch_i in lp and ch_j in lp:
        return "intra_left_parasagittal"
    if ch_i in rp and ch_j in rp:
        return "intra_right_parasagittal"
    if ch_i in rt and ch_j in rt:
        return "intra_right_temporal"
    if ch_i in ml and ch_j in ml:
        return "intra_midline"
        
    left = lt.union(lp)
    right = rt.union(rp)
    
    if ch_i in left and ch_j in left:
        return "intra_left_hemisphere_cross_chain"
    if ch_i in right and ch_j in right:
        return "intra_right_hemisphere_cross_chain"
    if (ch_i in left and ch_j in right) or (ch_i in right and ch_j in left):
        return "interhemispheric_heterologous"
    if ch_i in ml or ch_j in ml:
        return "midline_to_lateral"
        
    return "other"


def main():
    loader = EDFLoader()
    extractor = EEGConnectivityExtractor()
    
    edf_path = DATASET_DIR / "chb01" / "chb01_03.edf"
    assert edf_path.exists(), f"EDF path not found: {edf_path}"
    
    # Real ictal window: t in [3000.0, 3004.0]s (seizure is 2996s - 3036s)
    start_sec = 3000.0
    duration_sec = 4.0
    stop_sec = start_sec + duration_sec
    start_sample = int(start_sec * SAMPLING_RATE)
    stop_sample = int(stop_sec * SAMPLING_RATE)
    
    print(f"Loading real ictal EEG slice from {edf_path.name} at t=[{start_sec}, {stop_sec}] s...")
    raw_slice = loader.load_slice(
        edf_path,
        start_sample=start_sample,
        stop_sample=stop_sample,
        apply_filter=True,
        apply_standardization=True,
    )
    
    print(f"Extracted slice shape: {raw_slice.shape} (18 channels, 1024 samples)")
    assert raw_slice.shape == (18, 1024)
    
    # Window input shape for extractor: (N, C, T) -> (1, 18, 1024)
    window_batch = np.expand_dims(raw_slice, axis=0)
    
    # 1. Compute full matrices
    print("Computing full 18x18 PLV and PLI matrices across Delta, Theta, Alpha, Beta...")
    matrices_by_band = extractor.compute_full_matrices(window_batch)
    
    # Verify numerical sanity
    for band, m_dict in matrices_by_band.items():
        for conn_type in ["plv", "pli"]:
            A = m_dict[conn_type][0]  # (18, 18)
            diff_sym = np.max(np.abs(A - A.T))
            diag_val = np.max(np.abs(np.diag(A)))
            min_val = np.min(A)
            max_val = np.max(A)
            n_nan = np.isnan(A).sum()
            print(f"  {band.upper()} {conn_type.upper()}: sym_diff={diff_sym:.2e}, diag={diag_val:.2e}, "
                  f"range=[{min_val:.4f}, {max_val:.4f}], NaNs={n_nan}")
            assert diff_sym < 1e-6, f"Asymmetry in {band} {conn_type}"
            assert diag_val == 0.0, f"Nonzero diagonal in {band} {conn_type}"
            assert 0.0 <= min_val <= max_val <= 1.0, f"Out of bounds in {band} {conn_type}"
            assert n_nan == 0, f"NaNs in {band} {conn_type}"
            
    # 2. Build 153-pair edge table
    print("Building 153-pair labeled edge table...")
    tri_i, tri_j = extractor.triu_i, extractor.triu_j
    assert len(tri_i) == 153
    
    edge_rows = []
    for pair_idx in range(len(tri_i)):
        i, j = tri_i[pair_idx], tri_j[pair_idx]
        ch_i, ch_j = CANONICAL_CHANNELS[i], CANONICAL_CHANNELS[j]
        rel = classify_anatomical_relation(ch_i, ch_j)
        
        row_dict = {
            "pair_idx": pair_idx,
            "node_idx_i": i,
            "node_idx_j": j,
            "channel_i": ch_i,
            "channel_j": ch_j,
            "anatomical_relation": rel,
            "plv_delta": matrices_by_band["delta"]["plv"][0, i, j],
            "pli_delta": matrices_by_band["delta"]["pli"][0, i, j],
            "plv_theta": matrices_by_band["theta"]["plv"][0, i, j],
            "pli_theta": matrices_by_band["theta"]["pli"][0, i, j],
            "plv_alpha": matrices_by_band["alpha"]["plv"][0, i, j],
            "pli_alpha": matrices_by_band["alpha"]["pli"][0, i, j],
            "plv_beta": matrices_by_band["beta"]["plv"][0, i, j],
            "pli_beta": matrices_by_band["beta"]["pli"][0, i, j],
        }
        edge_rows.append(row_dict)
        
    edge_df = pd.DataFrame(edge_rows)
    out_edge_csv = PROJECT_ROOT / "outputs" / "audit" / "example_window_edge_table.csv"
    edge_df.to_csv(out_edge_csv, index=False)
    print(f"Saved {len(edge_df)} labeled edge rows to {out_edge_csv}")
    
    # 3. Extract and check 40 Graph Theoretical Features
    graph_feats, graph_names = compute_window_graph_features(matrices_by_band, densities=[0.20])
    print(f"Extracted graph theoretical features: shape={graph_feats.shape}, names={len(graph_names)}")
    assert graph_feats.shape == (1, 80) or graph_feats.shape == (1, 40) or len(graph_names) >= 40
    print(f"Graph feature names sample: {graph_names[:5]}")
    
    # 4. Generate Circular / Matrix Network Visualizations
    print("Generating comprehensive network visualization plot...")
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.2, 1.2, 1.0], height_ratios=[1, 1], hspace=0.35, wspace=0.3)
    
    ch_labels = [ch.replace("-", "\n") for ch in CANONICAL_CHANNELS]
    theta_plv = matrices_by_band["theta"]["plv"][0]
    theta_pli = matrices_by_band["theta"]["pli"][0]
    
    # Panel A: Theta PLV Matrix
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(theta_plv, cmap="viridis", vmin=0, vmax=1.0)
    ax_a.set_xticks(range(18))
    ax_a.set_yticks(range(18))
    ax_a.set_xticklabels(CANONICAL_CHANNELS, rotation=90, fontsize=7)
    ax_a.set_yticklabels(CANONICAL_CHANNELS, fontsize=7)
    ax_a.set_title("A. Theta Band PLV Matrix (4-8 Hz)\nchb01_03 Ictal Window [3000-3004s]", fontsize=10, fontweight="bold")
    cbar_a = fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
    cbar_a.set_label("Phase Locking Value (PLV)", fontsize=8)
    
    # Panel B: Theta PLI Matrix
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(theta_pli, cmap="magma", vmin=0, vmax=1.0)
    ax_b.set_xticks(range(18))
    ax_b.set_yticks(range(18))
    ax_b.set_xticklabels(CANONICAL_CHANNELS, rotation=90, fontsize=7)
    ax_b.set_yticklabels(CANONICAL_CHANNELS, fontsize=7)
    ax_b.set_title("B. Theta Band PLI Matrix (4-8 Hz)\nchb01_03 Ictal Window [3000-3004s]", fontsize=10, fontweight="bold")
    cbar_b = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cbar_b.set_label("Phase Lag Index (PLI)", fontsize=8)
    
    # Panel C: Circular Topographic Layout of Theta PLV (Thresholded top 20%)
    ax_c = fig.add_subplot(gs[1, 0])
    _draw_circular_graph(ax_c, theta_plv, density=0.20, cmap_name="viridis", 
                         title="C. Theta PLV Circular Graph (Top 20% Edges)\nSensitive to Phase Synchrony + Volume Conduction")
    
    # Panel D: Circular Topographic Layout of Theta PLI (Thresholded top 20%)
    ax_d = fig.add_subplot(gs[1, 1])
    _draw_circular_graph(ax_d, theta_pli, density=0.20, cmap_name="magma",
                         title="D. Theta PLI Circular Graph (Top 20% Edges)\nVolume Conduction Invariant (Phase Lags Only)")
    
    # Panel E: Nodal Strength Comparison Bar Chart
    ax_e = fig.add_subplot(gs[:, 2])
    str_plv = node_strengths(theta_plv)
    str_pli = node_strengths(theta_pli)
    y_pos = np.arange(18)
    h = 0.35
    ax_e.barh(y_pos - h/2, str_plv, height=h, color="#2b5c8f", label="PLV Strength (Coupling + Leakage)")
    ax_e.barh(y_pos + h/2, str_pli, height=h, color="#c44e52", label="PLI Strength (True Lagged Coupling)")
    ax_e.set_yticks(y_pos)
    ax_e.set_yticklabels(CANONICAL_CHANNELS, fontsize=8)
    ax_e.invert_yaxis()
    ax_e.set_xlabel("Nodal Connection Strength (Sum of Incident Weights)", fontsize=9)
    ax_e.set_title("E. Nodal Strength Comparison\n(18 Bipolar Channels, Theta Band)", fontsize=10, fontweight="bold")
    ax_e.legend(loc="lower right", fontsize=8)
    ax_e.grid(axis="x", linestyle="--", alpha=0.6)
    
    plt.suptitle("CHB-MIT Functional EEG Network Audit: Representative Real Window (chb01_03, t=3000-3004s)", 
                 fontsize=13, fontweight="bold", y=0.98)
    
    out_plot = PROJECT_ROOT / "outputs" / "audit" / "example_window_network_plot.png"
    plt.savefig(out_plot, bbox_inches="tight")
    plt.close()
    print(f"Saved publication network visualization to {out_plot}")


def _draw_circular_graph(ax, A, density=0.20, cmap_name="viridis", title=""):
    """Draw circular network graph with 18 nodes and top thresholded edges."""
    N = 18
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Circle layout
    x = np.cos(angles)
    y = np.sin(angles)
    
    # Threshold matrix
    B = threshold_proportional(A, density=density)
    tri_i, tri_j = np.triu_indices(N, k=1)
    
    # Plot edges
    cmap = plt.get_cmap(cmap_name)
    edge_weights = A[tri_i, tri_j]
    max_w = np.max(edge_weights) if np.max(edge_weights) > 0 else 1.0
    
    for i, j in zip(tri_i, tri_j):
        if B[i, j] > 0:
            w = A[i, j]
            norm_w = w / max_w
            ax.plot([x[i], x[j]], [y[i], y[j]], color=cmap(norm_w), alpha=0.7, linewidth=1.2 + 2.5 * norm_w, zorder=1)
            
    # Plot nodes
    # Color nodes by anatomical group
    group_colors = []
    for ch in CANONICAL_CHANNELS:
        if ch.startswith("FP1") or ch.startswith("F7") or ch.startswith("T7") or ch.startswith("P7"):
            group_colors.append("#387eb8")  # Left Temporal
        elif ch.startswith("FP1-F3") or ch.startswith("F3") or ch.startswith("C3") or ch.startswith("P3"):
            group_colors.append("#4daf4a")  # Left Parasagittal
        elif "Z" in ch:
            group_colors.append("#984ea3")  # Midline
        elif ch.startswith("FP2-F4") or ch.startswith("F4") or ch.startswith("C4") or ch.startswith("P4"):
            group_colors.append("#ff7f00")  # Right Parasagittal
        else:
            group_colors.append("#e41a1c")  # Right Temporal
            
    ax.scatter(x, y, s=180, c=group_colors, edgecolors="black", linewidth=1.2, zorder=2)
    
    # Node labels
    for i in range(N):
        ch = CANONICAL_CHANNELS[i]
        # Offset label outward
        x_lbl = x[i] * 1.22
        y_lbl = y[i] * 1.22
        ax.text(x_lbl, y_lbl, ch, fontsize=6.5, ha="center", va="center", fontweight="semibold")
        
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=12)


if __name__ == "__main__":
    main()
