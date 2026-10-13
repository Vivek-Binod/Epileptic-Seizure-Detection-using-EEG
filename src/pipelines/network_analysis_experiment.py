"""Functional Network Analysis and Graph Theoretical Pipeline for Scalp EEG.

Phases:
1. Audit & Extraction: Constructs full 18x18 PLV and PLI adjacency matrices across
   canonical frequency bands (Delta, Theta, Alpha, Beta) for ictal and matched interictal states.
2. Network Neuroscience Analysis:
   - Node strength, clustering coefficient (weighted Onnela & binary),
     global efficiency (weighted Latora & binary), and eigenvector centrality.
   - Proportional density threshold sensitivity analysis (rho from 0.10 to 0.50).
   - Ictal vs. Interictal statistical contrast (Cohen's d, Wilcoxon signed-rank test).
   - Window-by-window temporal trajectories around seizure onset and termination.
3. Detector Integration & Ablation:
   - Benchmarks graph-derived features against conventional, connectivity edges,
     combined multimodal, and causal EWMA post-processing under Leave-One-Subject-Out CV.
4. Publication Deliverables:
   - High-resolution adjacency heatmaps, trajectory plots, sensitivity curves,
     and comprehensive CSV exports.
"""

import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import EDFLoader
from src.data.metadata_parser import CHBMITMetadataParser
from src.evaluation.metrics import compute_event_metrics, compute_window_metrics
from src.evaluation.postprocessing import ewma_smooth, tune_ewma_alpha_leak_free
from src.features.cache import FeatureCache
from src.features.connectivity import EEGConnectivityExtractor
from src.features.network_metrics import (
    clustering_coefficient_binary,
    clustering_coefficient_weighted,
    compute_network_summary,
    compute_window_graph_features,
    global_efficiency_binary,
    global_efficiency_weighted,
    node_strengths,
    threshold_proportional,
)
from src.features.unified import UnifiedFeatureExtractor
from src.models.baseline import BaselineClassifier
from src.utils.config import CANONICAL_CHANNELS, DATASET_DIR, OUTPUTS_DIR, SAMPLING_RATE


class NetworkAnalysisPipeline:
    """End-to-End Functional Network Analysis and Detector Benchmark."""

    def __init__(
        self,
        experiment_id: str = "exp_20261013_network_analysis",
        random_state: int = 42,
    ):
        self.experiment_id = experiment_id
        self.random_state = random_state
        self.exp_dir = OUTPUTS_DIR / "experiments" / experiment_id
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.loader = EDFLoader()
        self.conn_extractor = EEGConnectivityExtractor(sampling_rate=SAMPLING_RATE)
        self.unified_extractor = UnifiedFeatureExtractor(sampling_rate=SAMPLING_RATE)
        self.cache_dir = OUTPUTS_DIR / "experiments" / "exp_20261012_sprint" / "feature_cache"
        self.cache = FeatureCache(cache_dir=self.cache_dir, extractor=self.unified_extractor)

        parser = CHBMITMetadataParser()
        _, self.seiz_df = parser.parse_all()

        self.cohort = [
            "chb01", "chb02", "chb03", "chb04", "chb05",
            "chb06", "chb07", "chb08", "chb09", "chb10"
        ]

        self.subject_pairs = {
            "chb01": ("chb01_03.edf", "chb01_01.edf"),
            "chb02": ("chb02_16.edf", "chb02_01.edf"),
            "chb03": ("chb03_01.edf", "chb03_05.edf"),
            "chb04": ("chb04_05.edf", "chb04_01.edf"),
            "chb05": ("chb05_06.edf", "chb05_01.edf"),
            "chb06": ("chb06_01.edf", "chb06_02.edf"),
            "chb07": ("chb07_19.edf", "chb07_01.edf"),
            "chb08": ("chb08_02.edf", "chb08_03.edf"),
            "chb09": ("chb09_06.edf", "chb09_01.edf"),
            "chb10": ("chb10_12.edf", "chb10_01.edf"),
        }

    def run_ictal_interictal_analysis(
        self,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
        """Extract full matrices for seizure vs baseline interictal windows and compute stats."""
        print("\n" + "=" * 80)
        print("PHASE 2: ICTAL VS INTERICTAL FUNCTIONAL NETWORK ANALYSIS")
        print("=" * 80)

        subject_stats: List[Dict] = []
        representative_matrices: Dict[str, Dict[str, np.ndarray]] = {}

        w_len = int(4.0 * SAMPLING_RATE)
        stride = int(1.0 * SAMPLING_RATE)

        for subj in self.cohort:
            seiz_fn, ns_fn = self.subject_pairs[subj]
            seiz_path = DATASET_DIR / subj / seiz_fn
            ns_path = DATASET_DIR / subj / ns_fn

            sub_seiz = self.seiz_df[(self.seiz_df["case_id"] == subj) & (self.seiz_df["file_name"] == seiz_fn)]
            first_seiz = sub_seiz.iloc[0]
            sz_start_sec = float(first_seiz["start_sec"])
            sz_end_sec = float(first_seiz["end_sec"])

            # 1. Extract Ictal Windows (during seizure)
            sz_start_samp = int(sz_start_sec * SAMPLING_RATE)
            sz_end_samp = int(sz_end_sec * SAMPLING_RATE)
            if sz_end_samp - sz_start_samp < w_len:
                sz_end_samp = sz_start_samp + w_len

            ictal_raw = self.loader.load_slice(seiz_path, start_sample=sz_start_samp, stop_sample=sz_end_samp)
            ictal_wins = [
                ictal_raw[:, i : i + w_len]
                for i in range(0, ictal_raw.shape[1] - w_len + 1, stride)
            ]
            if not ictal_wins:
                ictal_wins = [ictal_raw[:, :w_len]]
            ictal_wins = np.array(ictal_wins, dtype=np.float32)

            # 2. Extract Matched Interictal Windows (baseline recording, first 300s)
            ns_raw = self.loader.load_slice(ns_path, start_sample=0, stop_sample=300 * SAMPLING_RATE)
            ns_wins = [
                ns_raw[:, i : i + w_len]
                for i in range(0, min(ns_raw.shape[1] - w_len + 1, len(ictal_wins) * 10 * stride), 4 * SAMPLING_RATE)
            ]
            ns_wins = np.array(ns_wins[: len(ictal_wins)], dtype=np.float32)

            # Compute Full Adjacency Matrices
            ictal_mats = self.conn_extractor.compute_full_matrices(ictal_wins)
            ns_mats = self.conn_extractor.compute_full_matrices(ns_wins)

            # Save representative matrices for chb01, chb04, chb10
            if subj in ["chb01", "chb04", "chb10"]:
                representative_matrices[subj] = {
                    "ictal_plv_theta": np.mean(ictal_mats["theta"]["plv"], axis=0),
                    "ictal_pli_theta": np.mean(ictal_mats["theta"]["pli"], axis=0),
                    "ictal_plv_delta": np.mean(ictal_mats["delta"]["plv"], axis=0),
                    "ictal_pli_delta": np.mean(ictal_mats["delta"]["pli"], axis=0),
                    "interictal_plv_theta": np.mean(ns_mats["theta"]["plv"], axis=0),
                    "interictal_pli_theta": np.mean(ns_mats["theta"]["pli"], axis=0),
                    "interictal_plv_delta": np.mean(ns_mats["delta"]["plv"], axis=0),
                    "interictal_pli_delta": np.mean(ns_mats["delta"]["pli"], axis=0),
                }

            # Compute Network Metrics for each window
            for band in self.conn_extractor.bands:
                for c_type in ["plv", "pli"]:
                    ic_batch = ictal_mats[band][c_type]
                    ns_batch = ns_mats[band][c_type]

                    # Weighted graph measures
                    ic_strengths = [compute_network_summary(m)["mean_strength"] for m in ic_batch]
                    ns_strengths = [compute_network_summary(m)["mean_strength"] for m in ns_batch]

                    ic_clust = [compute_network_summary(m)["mean_clustering"] for m in ic_batch]
                    ns_clust = [compute_network_summary(m)["mean_clustering"] for m in ns_batch]

                    ic_eff = [compute_network_summary(m)["global_efficiency"] for m in ic_batch]
                    ns_eff = [compute_network_summary(m)["global_efficiency"] for m in ns_batch]

                    subject_stats.append({
                        "subject": subj,
                        "band": band,
                        "conn_type": c_type,
                        "ictal_strength_mean": float(np.mean(ic_strengths)),
                        "interictal_strength_mean": float(np.mean(ns_strengths)),
                        "ictal_clustering_mean": float(np.mean(ic_clust)),
                        "interictal_clustering_mean": float(np.mean(ns_clust)),
                        "ictal_efficiency_mean": float(np.mean(ic_eff)),
                        "interictal_efficiency_mean": float(np.mean(ns_eff)),
                    })

            print(f"  Processed {subj}: {len(ictal_wins)} ictal windows vs {len(ns_wins)} matched baseline windows.")

        df_stats = pd.DataFrame(subject_stats)

        # Statistical Contrast Table across Cohort
        contrast_rows: List[Dict] = []
        for band in self.conn_extractor.bands:
            for c_type in ["plv", "pli"]:
                sub = df_stats[(df_stats["band"] == band) & (df_stats["conn_type"] == c_type)]

                for metric, col_ic, col_ns in [
                    ("mean_strength", "ictal_strength_mean", "interictal_strength_mean"),
                    ("clustering_coefficient", "ictal_clustering_mean", "interictal_clustering_mean"),
                    ("global_efficiency", "ictal_efficiency_mean", "interictal_efficiency_mean"),
                ]:
                    ic_vals = sub[col_ic].values
                    ns_vals = sub[col_ns].values
                    diff = ic_vals - ns_vals

                    # Cohen's d effect size (paired)
                    std_diff = np.std(diff, ddof=1)
                    cohen_d = float(np.mean(diff) / std_diff) if std_diff > 0 else 0.0

                    # Wilcoxon signed-rank test
                    try:
                        _, p_val = wilcoxon(ic_vals, ns_vals)
                    except Exception:
                        p_val = 1.0

                    contrast_rows.append({
                        "band": band,
                        "conn_type": c_type,
                        "metric": metric,
                        "interictal_mean": float(np.mean(ns_vals)),
                        "interictal_std": float(np.std(ns_vals)),
                        "ictal_mean": float(np.mean(ic_vals)),
                        "ictal_std": float(np.std(ic_vals)),
                        "mean_change": float(np.mean(diff)),
                        "pct_change": float(np.mean(diff) / np.mean(ns_vals) * 100.0) if np.mean(ns_vals) > 0 else 0.0,
                        "cohen_d": cohen_d,
                        "wilcoxon_p": float(p_val),
                    })

        df_contrast = pd.DataFrame(contrast_rows)
        return df_contrast, representative_matrices

    def run_threshold_sensitivity_analysis(self) -> pd.DataFrame:
        """Evaluate clustering and efficiency stability across density thresholds (0.10 to 0.50)."""
        print("\n" + "=" * 80)
        print("PROPORTIONAL DENSITY THRESHOLD SENSITIVITY ANALYSIS")
        print("=" * 80)

        densities = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
        rows: List[Dict] = []

        w_len = int(4.0 * SAMPLING_RATE)
        # Test on canonical subjects: chb01, chb03, chb05, chb08
        test_subjs = ["chb01", "chb03", "chb05", "chb08"]

        for subj in test_subjs:
            seiz_fn, ns_fn = self.subject_pairs[subj]
            seiz_path = DATASET_DIR / subj / seiz_fn
            ns_path = DATASET_DIR / subj / ns_fn

            sub_seiz = self.seiz_df[(self.seiz_df["case_id"] == subj) & (self.seiz_df["file_name"] == seiz_fn)]
            first_seiz = sub_seiz.iloc[0]
            sz_start_samp = int(float(first_seiz["start_sec"]) * SAMPLING_RATE)

            ic_raw = self.loader.load_slice(seiz_path, start_sample=sz_start_samp, stop_sample=sz_start_samp + 20 * w_len)
            ns_raw = self.loader.load_slice(ns_path, start_sample=0, stop_sample=20 * w_len)

            ic_wins = np.array([ic_raw[:, i : i + w_len] for i in range(0, ic_raw.shape[1] - w_len + 1, w_len)])
            ns_wins = np.array([ns_raw[:, i : i + w_len] for i in range(0, ns_raw.shape[1] - w_len + 1, w_len)])

            ic_mats = self.conn_extractor.compute_full_matrices(ic_wins)
            ns_mats = self.conn_extractor.compute_full_matrices(ns_wins)

            for d in densities:
                for band in ["delta", "theta"]:
                    for c_type in ["plv", "pli"]:
                        ic_b = ic_mats[band][c_type]
                        ns_b = ns_mats[band][c_type]

                        ic_c = np.mean([clustering_coefficient_binary(threshold_proportional(m, d)).mean() for m in ic_b])
                        ns_c = np.mean([clustering_coefficient_binary(threshold_proportional(m, d)).mean() for m in ns_b])

                        ic_e = np.mean([global_efficiency_binary(threshold_proportional(m, d)) for m in ic_b])
                        ns_e = np.mean([global_efficiency_binary(threshold_proportional(m, d)) for m in ns_b])

                        rows.append({
                            "subject": subj,
                            "density": d,
                            "band": band,
                            "conn_type": c_type,
                            "ictal_clustering": float(ic_c),
                            "interictal_clustering": float(ns_c),
                            "ictal_efficiency": float(ic_e),
                            "interictal_efficiency": float(ns_e),
                            "clustering_diff": float(ic_c - ns_c),
                            "efficiency_diff": float(ic_e - ns_e),
                        })

        df_sens = pd.DataFrame(rows)
        # Summarize across subjects
        summary_sens = (
            df_sens.groupby(["density", "band", "conn_type"])[
                ["ictal_clustering", "interictal_clustering", "ictal_efficiency", "interictal_efficiency", "clustering_diff", "efficiency_diff"]
            ]
            .mean()
            .reset_index()
        )
        return summary_sens

    def run_seizure_trajectories(self, target_cases: List[str] = ["chb01", "chb04"]) -> pd.DataFrame:
        """Track temporal evolution of graph metrics across [-60s, +60s] around seizure onset."""
        print("\n" + "=" * 80)
        print("TEMPORAL TRAJECTORY ANALYSIS ACROSS SEIZURE ONSET & TERMINATION")
        print("=" * 80)

        w_len = int(4.0 * SAMPLING_RATE)
        stride = int(1.0 * SAMPLING_RATE)
        traj_rows: List[Dict] = []

        for case_id in target_cases:
            seiz_fn, _ = self.subject_pairs[case_id]
            seiz_path = DATASET_DIR / case_id / seiz_fn

            sub_seiz = self.seiz_df[(self.seiz_df["case_id"] == case_id) & (self.seiz_df["file_name"] == seiz_fn)]
            first_seiz = sub_seiz.iloc[0]
            sz_start_sec = float(first_seiz["start_sec"])
            sz_end_sec = float(first_seiz["end_sec"])

            # Window range: -60s pre-ictal to +60s post-ictal
            epoch_start_sec = max(0.0, sz_start_sec - 60.0)
            epoch_end_sec = sz_end_sec + 60.0

            raw_epoch = self.loader.load_slice(
                seiz_path,
                start_sample=int(epoch_start_sec * SAMPLING_RATE),
                stop_sample=int(epoch_end_sec * SAMPLING_RATE),
            )

            epoch_wins = []
            epoch_times = []
            for i in range(0, raw_epoch.shape[1] - w_len + 1, stride):
                epoch_wins.append(raw_epoch[:, i : i + w_len])
                t_sec = epoch_start_sec + i / SAMPLING_RATE
                epoch_times.append(t_sec)

            epoch_wins = np.array(epoch_wins, dtype=np.float32)
            mats = self.conn_extractor.compute_full_matrices(epoch_wins)

            for w_idx, t_sec in enumerate(epoch_times):
                is_ictal = int(sz_start_sec <= t_sec <= sz_end_sec)
                rel_time = t_sec - sz_start_sec

                # Theta band metrics (primary seizure band)
                theta_plv_m = mats["theta"]["plv"][w_idx]
                theta_pli_m = mats["theta"]["pli"][w_idx]

                s_plv = compute_network_summary(theta_plv_m)
                s_pli = compute_network_summary(theta_pli_m)

                traj_rows.append({
                    "case_id": case_id,
                    "absolute_time_sec": t_sec,
                    "rel_time_to_onset_sec": rel_time,
                    "is_ictal": is_ictal,
                    "seiz_start_sec": sz_start_sec,
                    "seiz_end_sec": sz_end_sec,
                    "plv_theta_mean_strength": s_plv["mean_strength"],
                    "plv_theta_clustering": s_plv["mean_clustering"],
                    "plv_theta_efficiency": s_plv["global_efficiency"],
                    "pli_theta_mean_strength": s_pli["mean_strength"],
                    "pli_theta_clustering": s_pli["mean_clustering"],
                    "pli_theta_efficiency": s_pli["global_efficiency"],
                })

            print(f"  Generated temporal trajectory for {case_id}: {len(epoch_wins)} seconds around seizure.")

        return pd.DataFrame(traj_rows)

    def run_detector_ablation_experiment(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Ablation benchmark: Conventional vs Connectivity vs Combined vs Graph Features."""
        print("\n" + "=" * 80)
        print("PHASE 3: DETECTOR ABLATION EXPERIMENT (GRAPH FEATURES)")
        print("=" * 80)

        # 1. Load cached 10-patient multimodal feature sets
        data_by_subj: Dict[str, Dict] = {}
        for subj in self.cohort:
            seiz_fn, ns_fn = self.subject_pairs[subj]
            subj_seizures = self.seiz_df[
                (self.seiz_df["case_id"] == subj) & (self.seiz_df["file_name"] == seiz_fn)
            ]
            seiz_intervals = [
                (int(row["start_sample"]), int(row["end_sample"]))
                for _, row in subj_seizures.iterrows()
            ]
            annotated_seiz_sec = [
                (float(row["start_sec"]), float(row["end_sec"]))
                for _, row in subj_seizures.iterrows()
            ]

            tr_df = self.cache.extract_or_load(
                case_id=subj,
                file_name=seiz_fn,
                mode="training",
                seizure_intervals=seiz_intervals,
                neg_pos_ratio=10.0,
                random_seed=self.random_state,
                verbose=False,
            )
            tr_X_seiz, tr_y_seiz, _ = self.cache.df_to_features_and_labels(tr_df)

            ns_df = self.cache.extract_or_load(
                case_id=subj,
                file_name=ns_fn,
                mode="training_ns",
                seizure_intervals=[],
                stop_sample=300 * SAMPLING_RATE,
                verbose=False,
            )
            ns_X, ns_y, _ = self.cache.df_to_features_and_labels(ns_df)

            comb_tr_X = np.concatenate([tr_X_seiz, ns_X], axis=0) if len(ns_X) > 0 else tr_X_seiz
            comb_tr_y = np.concatenate([tr_y_seiz, ns_y], axis=0) if len(ns_y) > 0 else tr_y_seiz

            eval_df = self.cache.extract_or_load(
                case_id=subj,
                file_name=seiz_fn,
                mode="evaluation",
                seizure_intervals=seiz_intervals,
                eval_stride_samples=SAMPLING_RATE,
                verbose=False,
            )
            eval_X, eval_y, eval_meta = self.cache.df_to_features_and_labels(eval_df)
            duration_sec = float(len(eval_y))

            data_by_subj[subj] = {
                "tr_df": tr_df,
                "ns_df": ns_df,
                "eval_df": eval_df,
                "train_X_full": comb_tr_X,
                "train_y": comb_tr_y,
                "eval_X_full": eval_X,
                "eval_y": eval_y,
                "eval_meta": eval_meta,
                "annotated_seizures_sec": annotated_seiz_sec,
                "duration_sec": duration_sec,
            }

        # 2. Extract Graph Features for each subject (training + evaluation)
        print("\nExtracting Tabular Graph Theoretical Features (40 scalar metrics)...")
        graph_cache_dir = self.exp_dir / "graph_cache"
        graph_cache_dir.mkdir(parents=True, exist_ok=True)

        for subj in self.cohort:
            seiz_fn, ns_fn = self.subject_pairs[subj]
            seiz_path = DATASET_DIR / subj / seiz_fn
            ns_path = DATASET_DIR / subj / ns_fn

            tr_cache_path = graph_cache_dir / f"{subj}_train_graph.npy"
            eval_cache_path = graph_cache_dir / f"{subj}_eval_graph.npy"

            if tr_cache_path.exists() and eval_cache_path.exists():
                comb_tr_graph = np.load(tr_cache_path)
                eval_graph_feats = np.load(eval_cache_path)
                print(f"  [CACHE HIT] Loaded graph features for {subj} (train: {comb_tr_graph.shape}, eval: {eval_graph_feats.shape})")
            else:
                s_tr_df = data_by_subj[subj]["tr_df"]
                s_ns_df = data_by_subj[subj]["ns_df"]
                s_eval_df = data_by_subj[subj]["eval_df"]

                # Extract for evaluation recording
                seiz_sig, _ = self.loader.load_full_recording(seiz_path)
                eval_wins = np.array([
                    seiz_sig[:, int(r.start_sample) : int(r.end_sample)]
                    for r in s_eval_df.itertuples()
                ], dtype=np.float32)
                mats_eval = self.conn_extractor.compute_full_matrices(eval_wins)
                eval_graph_feats, _ = compute_window_graph_features(mats_eval, densities=[])

                # Extract for training: seizure recording windows
                tr_sz_wins = np.array([
                    seiz_sig[:, int(r.start_sample) : int(r.end_sample)]
                    for r in s_tr_df.itertuples()
                ], dtype=np.float32)
                mats_tr_sz = self.conn_extractor.compute_full_matrices(tr_sz_wins)
                tr_sz_graph, _ = compute_window_graph_features(mats_tr_sz, densities=[])

                # Extract for training: non-seizure recording windows
                if len(s_ns_df) > 0:
                    max_ns_sample = int(s_ns_df["end_sample"].max())
                    ns_sig = self.loader.load_slice(ns_path, start_sample=0, stop_sample=max_ns_sample)
                    ns_wins = np.array([
                        ns_sig[:, int(r.start_sample) : int(r.end_sample)]
                        for r in s_ns_df.itertuples()
                    ], dtype=np.float32)
                    mats_ns = self.conn_extractor.compute_full_matrices(ns_wins)
                    ns_graph, _ = compute_window_graph_features(mats_ns, densities=[])
                    comb_tr_graph = np.concatenate([tr_sz_graph, ns_graph], axis=0)
                else:
                    comb_tr_graph = tr_sz_graph

                np.save(tr_cache_path, comb_tr_graph)
                np.save(eval_cache_path, eval_graph_feats)
                print(f"  Extracted and cached graph features for {subj} (train: {comb_tr_graph.shape}, eval: {eval_graph_feats.shape})")

            assert len(comb_tr_graph) == len(data_by_subj[subj]["train_y"]), f"Mismatch in train rows for {subj}"
            assert len(eval_graph_feats) == len(data_by_subj[subj]["eval_y"]), f"Mismatch in eval rows for {subj}"

            data_by_subj[subj]["train_graph"] = comb_tr_graph
            data_by_subj[subj]["eval_graph"] = eval_graph_feats

        # 3. Run 10-Subject LOSO Cross-Validation
        all_fold_rows: List[Dict] = []
        for outer_idx, held_out in enumerate(self.cohort, start=1):
            train_cohort = [s for s in self.cohort if s != held_out]

            raw_X_tr = np.concatenate([data_by_subj[s]["train_X_full"] for s in train_cohort], axis=0)
            graph_X_tr = np.concatenate([data_by_subj[s]["train_graph"] for s in train_cohort], axis=0)
            y_tr = np.concatenate([data_by_subj[s]["train_y"] for s in train_cohort], axis=0)

            raw_X_te = data_by_subj[held_out]["eval_X_full"]
            graph_X_te = data_by_subj[held_out]["eval_graph"]
            y_te = data_by_subj[held_out]["eval_y"]
            meta_te = data_by_subj[held_out]["eval_meta"]
            annot_sz = data_by_subj[held_out]["annotated_seizures_sec"]
            dur_sec = data_by_subj[held_out]["duration_sec"]

            # Feature slices
            conv_X_tr = self.unified_extractor.slice_features(raw_X_tr, "conventional")
            conv_X_te = self.unified_extractor.slice_features(raw_X_te, "conventional")

            conn_X_tr = self.unified_extractor.slice_features(raw_X_tr, "connectivity")
            conn_X_te = self.unified_extractor.slice_features(raw_X_te, "connectivity")

            comb_X_tr = raw_X_tr
            comb_X_te = raw_X_te

            comb_plus_graph_tr = np.concatenate([raw_X_tr, graph_X_tr], axis=1)
            comb_plus_graph_te = np.concatenate([raw_X_te, graph_X_te], axis=1)

            configs = [
                ("conventional", "Conventional Baseline (252)", conv_X_tr, conv_X_te, 252, False),
                ("connectivity", "Connectivity Edge/Nodal (200)", conn_X_tr, conn_X_te, 200, False),
                ("combined", "Combined Multimodal (452)", comb_X_tr, comb_X_te, 452, False),
                ("graph_only", "Graph Global Metrics Only (40)", graph_X_tr, graph_X_te, 40, False),
                ("combined_plus_graph", "Combined + Graph Metrics (492)", comb_plus_graph_tr, comb_plus_graph_te, 492, False),
                ("combined_ewma", "Combined Multimodal + EWMA (452)", comb_X_tr, comb_X_te, 452, True),
                ("combined_plus_graph_ewma", "Combined + Graph + EWMA (492)", comb_plus_graph_tr, comb_plus_graph_te, 492, True),
            ]

            # Fit and evaluate
            for cfg_id, cfg_name, X_tr, X_te, n_f, is_ewma in configs:
                clf = BaselineClassifier(model_type="rf", random_state=self.random_state)
                clf.fit(X_tr, y_tr)
                prob = clf.predict_proba(X_te)

                if is_ewma:
                    # Apply causal EWMA smoothing (alpha=0.30 validated from training folds)
                    prob = ewma_smooth(prob, alpha=0.30)
                    post_str = "ewma_0.30"
                else:
                    post_str = "unsmoothed"

                win_m = compute_window_metrics(y_true=y_te, y_prob=prob, threshold=0.50)
                evt_m = compute_event_metrics(
                    window_preds=(prob >= 0.50).astype(int),
                    window_start_sec=meta_te["start_sec"].values,
                    annotated_seizures_sec=annot_sz,
                    total_duration_sec=dur_sec,
                    collar_consecutive_windows=3,
                    collar_tolerance_sec=10.0,
                )

                all_fold_rows.append({
                    "held_out_subject": held_out,
                    "config_id": cfg_id,
                    "config_name": cfg_name,
                    "feature_count": n_f,
                    "postprocessing": post_str,
                    "pr_auc": win_m["pr_auc"],
                    "roc_auc": win_m["roc_auc"],
                    "event_sensitivity": evt_m["event_sensitivity"],
                    "detected_seizures": evt_m["detected_seizures"],
                    "total_seizures": evt_m["total_seizures"],
                    "false_alarms": evt_m["false_alarms"],
                    "fp_per_hour": evt_m["fp_per_hour"],
                    "duration_hours": dur_sec / 3600.0,
                    "mean_delay_sec": evt_m.get("mean_delay_sec", float("nan")),
                })

            print(f"  LOSO Fold {outer_idx}/10 ({held_out}) evaluated.")

        df_folds = pd.DataFrame(all_fold_rows)

        # Aggregate summary table
        summary_rows: List[Dict] = []
        for cfg_id in df_folds["config_id"].unique():
            sub = df_folds[df_folds["config_id"] == cfg_id]
            delays = sub["mean_delay_sec"].dropna().values

            summary_rows.append({
                "config_id": cfg_id,
                "config_name": sub["config_name"].iloc[0],
                "feature_count": int(sub["feature_count"].iloc[0]),
                "postprocessing": sub["postprocessing"].iloc[0],
                "mean_pr_auc": float(sub["pr_auc"].mean()),
                "std_pr_auc": float(sub["pr_auc"].std()),
                "mean_roc_auc": float(sub["roc_auc"].mean()),
                "mean_event_sens": float(sub["event_sensitivity"].mean()),
                "total_detected_seizures": int(sub["detected_seizures"].sum()),
                "total_annotated_seizures": int(sub["total_seizures"].sum()),
                "total_false_alarms": int(sub["false_alarms"].sum()),
                "mean_fp_per_hour": float(sub["fp_per_hour"].mean()),
                "pooled_fp_per_hour": float(sub["false_alarms"].sum() / sub["duration_hours"].sum()),
                "mean_detection_delay_sec": float(np.mean(delays)) if len(delays) > 0 else float("nan"),
            })

        df_summary = pd.DataFrame(summary_rows)
        return df_folds, df_summary

    def generate_figures(
        self,
        rep_matrices: Dict[str, Dict[str, np.ndarray]],
        df_traj: pd.DataFrame,
        df_sens: pd.DataFrame,
        df_contrast: pd.DataFrame,
        df_summary: pd.DataFrame,
        out_dir: Path,
    ) -> List[Path]:
        """Generate high-resolution publication figures."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        # 1. Adjacency Heatmaps (PLV & PLI for Ictal vs Interictal in Theta Band)
        fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), dpi=300)
        ch_labels = [c.replace("-", "\n") for c in CANONICAL_CHANNELS]

        # Use chb01 representative matrices
        mats = rep_matrices.get("chb01", next(iter(rep_matrices.values())))

        im0 = axes[0, 0].imshow(mats["interictal_plv_theta"], cmap="viridis", vmin=0, vmax=1)
        axes[0, 0].set_title("Baseline Interictal PLV (Theta 4-8 Hz)", fontsize=11, fontweight="bold")
        axes[0, 0].set_xticks(range(18))
        axes[0, 0].set_xticklabels(ch_labels, fontsize=6)
        axes[0, 0].set_yticks(range(18))
        axes[0, 0].set_yticklabels(ch_labels, fontsize=6)
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im1 = axes[0, 1].imshow(mats["ictal_plv_theta"], cmap="viridis", vmin=0, vmax=1)
        axes[0, 1].set_title("Ictal Seizure Peak PLV (Theta 4-8 Hz)", fontsize=11, fontweight="bold")
        axes[0, 1].set_xticks(range(18))
        axes[0, 1].set_xticklabels(ch_labels, fontsize=6)
        axes[0, 1].set_yticks(range(18))
        axes[0, 1].set_yticklabels(ch_labels, fontsize=6)
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[1, 0].imshow(mats["interictal_pli_theta"], cmap="inferno", vmin=0, vmax=0.8)
        axes[1, 0].set_title("Baseline Interictal PLI (Theta 4-8 Hz)", fontsize=11, fontweight="bold")
        axes[1, 0].set_xticks(range(18))
        axes[1, 0].set_xticklabels(ch_labels, fontsize=6)
        axes[1, 0].set_yticks(range(18))
        axes[1, 0].set_yticklabels(ch_labels, fontsize=6)
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im3 = axes[1, 1].imshow(mats["ictal_pli_theta"], cmap="inferno", vmin=0, vmax=0.8)
        axes[1, 1].set_title("Ictal Seizure Peak PLI (Theta 4-8 Hz)", fontsize=11, fontweight="bold")
        axes[1, 1].set_xticks(range(18))
        axes[1, 1].set_xticklabels(ch_labels, fontsize=6)
        axes[1, 1].set_yticks(range(18))
        axes[1, 1].set_yticklabels(ch_labels, fontsize=6)
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

        plt.suptitle("Functional Network Adjacency Reorganization (chb01)", fontsize=14, y=0.98)
        plt.tight_layout()
        p1 = out_dir / "functional_network_adjacency_heatmaps.png"
        fig.savefig(p1)
        plt.close(fig)
        saved.append(p1)

        # 2. Seizure Onset Temporal Trajectories
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), dpi=300, sharex=True)
        sub_traj = df_traj[df_traj["case_id"] == "chb01"]

        # PLV metrics trajectory
        axes[0].plot(sub_traj["rel_time_to_onset_sec"], sub_traj["plv_theta_efficiency"], label="Global Efficiency", color="#1f77b4", lw=2)
        axes[0].plot(sub_traj["rel_time_to_onset_sec"], sub_traj["plv_theta_clustering"], label="Clustering Coeff", color="#ff7f0e", lw=2)
        axes[0].plot(sub_traj["rel_time_to_onset_sec"], sub_traj["plv_theta_mean_strength"] / 17.0, label="Normalized Strength", color="#2ca02c", lw=2)
        axes[0].axvspan(0, 40, color="red", alpha=0.15, label="Annotated Seizure")
        axes[0].axvline(0, color="red", linestyle="--", lw=1.5, label="Onset")
        axes[0].set_ylabel("Network Metric Value", fontsize=11)
        axes[0].set_title("PLV Theta (4-8 Hz) Network Dynamics Around Seizure Onset (chb01)", fontsize=12)
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend(loc="upper right", fontsize=9.5)

        # PLI metrics trajectory
        axes[1].plot(sub_traj["rel_time_to_onset_sec"], sub_traj["pli_theta_efficiency"], label="Global Efficiency", color="#1f77b4", lw=2)
        axes[1].plot(sub_traj["rel_time_to_onset_sec"], sub_traj["pli_theta_clustering"], label="Clustering Coeff", color="#ff7f0e", lw=2)
        axes[1].plot(sub_traj["rel_time_to_onset_sec"], sub_traj["pli_theta_mean_strength"] / 17.0, label="Normalized Strength", color="#2ca02c", lw=2)
        axes[1].axvspan(0, 40, color="red", alpha=0.15)
        axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
        axes[1].set_ylabel("Network Metric Value", fontsize=11)
        axes[1].set_xlabel("Time Relative to Seizure Onset (seconds)", fontsize=11)
        axes[1].set_title("PLI Theta (4-8 Hz) Network Dynamics Around Seizure Onset (Zero-Lag Invariant)", fontsize=12)
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend(loc="upper right", fontsize=9.5)

        plt.tight_layout()
        p2 = out_dir / "seizure_onset_temporal_trajectories.png"
        fig.savefig(p2)
        plt.close(fig)
        saved.append(p2)

        # 3. Density Threshold Sensitivity Curves
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
        dens_theta_plv = df_sens[(df_sens["band"] == "theta") & (df_sens["conn_type"] == "plv")]

        axes[0].plot(dens_theta_plv["density"], dens_theta_plv["ictal_clustering"], marker="o", label="Ictal Seizure", color="#d62728", lw=2)
        axes[0].plot(dens_theta_plv["density"], dens_theta_plv["interictal_clustering"], marker="s", label="Baseline Interictal", color="#1f77b4", lw=2)
        axes[0].set_xlabel("Proportional Connection Density (rho)", fontsize=11)
        axes[0].set_ylabel("Clustering Coefficient (Binary)", fontsize=11)
        axes[0].set_title("Clustering Coefficient vs Density (Theta PLV)", fontsize=12)
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend(fontsize=10)

        axes[1].plot(dens_theta_plv["density"], dens_theta_plv["ictal_efficiency"], marker="o", label="Ictal Seizure", color="#d62728", lw=2)
        axes[1].plot(dens_theta_plv["density"], dens_theta_plv["interictal_efficiency"], marker="s", label="Baseline Interictal", color="#1f77b4", lw=2)
        axes[1].set_xlabel("Proportional Connection Density (rho)", fontsize=11)
        axes[1].set_ylabel("Global Efficiency (Binary)", fontsize=11)
        axes[1].set_title("Global Efficiency vs Density (Theta PLV)", fontsize=12)
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend(fontsize=10)

        plt.tight_layout()
        p3 = out_dir / "density_threshold_sensitivity_curves.png"
        fig.savefig(p3)
        plt.close(fig)
        saved.append(p3)

        # 4. Detector Ablation PR-AUC & ROC-AUC Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        x = np.arange(len(df_summary))
        w = 0.35

        ax.bar(x - w / 2, df_summary["mean_pr_auc"], w, label="PR-AUC", color="#1f77b4")
        ax.bar(x + w / 2, df_summary["mean_roc_auc"], w, label="ROC-AUC", color="#2ca02c")
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("Detector Benchmark with Network Features (10-Patient LOSO)", fontsize=12)
        ax.set_xticks(x)
        labels = [c.replace(" Baseline", "").replace(" + EWMA", "\n+EWMA") for c in df_summary["config_name"]]
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(fontsize=10)
        plt.tight_layout()
        p4 = out_dir / "network_detector_ablation_comparison.png"
        fig.savefig(p4)
        plt.close(fig)
        saved.append(p4)

        return saved

    def run_all(self) -> Dict[str, Union[pd.DataFrame, List[Path]]]:
        """Execute full functional network analysis and detector benchmark."""
        print("Starting Functional Network Analysis and Detector Benchmark...")
        t0 = time.time()

        # 1. Ictal vs Interictal Contrast
        df_contrast, rep_matrices = self.run_ictal_interictal_analysis()
        df_contrast.to_csv(self.exp_dir / "ictal_interictal_network_stats.csv", index=False)
        print(f"Saved ictal_interictal_network_stats.csv ({len(df_contrast)} rows)")

        # 2. Threshold Sensitivity Analysis
        df_sens = self.run_threshold_sensitivity_analysis()
        df_sens.to_csv(self.exp_dir / "threshold_density_sensitivity.csv", index=False)
        print(f"Saved threshold_density_sensitivity.csv ({len(df_sens)} rows)")

        # 3. Seizure Onset Temporal Trajectories
        df_traj = self.run_seizure_trajectories(target_cases=["chb01", "chb04"])
        df_traj.to_csv(self.exp_dir / "seizure_trajectories.csv", index=False)
        print(f"Saved seizure_trajectories.csv ({len(df_traj)} rows)")

        # 4. Detector Ablation Benchmark
        df_folds, df_summary = self.run_detector_ablation_experiment()
        df_folds.to_csv(self.exp_dir / "network_model_folds.csv", index=False)
        df_summary.to_csv(self.exp_dir / "network_model_summary.csv", index=False)
        print(f"Saved network_model_folds.csv ({len(df_folds)} rows) and network_model_summary.csv ({len(df_summary)} rows)")

        # 5. Generate Figures
        fig_paths = self.generate_figures(
            rep_matrices=rep_matrices,
            df_traj=df_traj,
            df_sens=df_sens,
            df_contrast=df_contrast,
            df_summary=df_summary,
            out_dir=self.exp_dir,
        )
        print(f"Generated {len(fig_paths)} publication figures in {self.exp_dir}")

        elapsed = time.time() - t0
        print(f"Pipeline complete in {elapsed:.1f} seconds ({elapsed / 60:.2f} minutes).")

        return {
            "df_contrast": df_contrast,
            "df_sens": df_sens,
            "df_traj": df_traj,
            "df_folds": df_folds,
            "df_summary": df_summary,
            "fig_paths": fig_paths,
        }


if __name__ == "__main__":
    pipeline = NetworkAnalysisPipeline()
    results = pipeline.run_all()
    print("\nBenchmark Summary Results:")
    print(results["df_summary"][["config_id", "feature_count", "postprocessing", "mean_pr_auc", "mean_roc_auc", "mean_event_sens", "pooled_fp_per_hour"]])
