"""Full-Cohort Functional Network Analysis and Detector Benchmark for Scalp EEG.

Evaluates:
- All 24 CHB-MIT recording cases across 23 independent biological patients (chb01 and chb21 unified under patient_01).
- 28 clinically annotated seizures (35.99 hours continuous evaluation EEG).
- 4 canonical frequency bands (Delta 0.5-4 Hz, Theta 4-8 Hz, Alpha 8-12 Hz, Beta 12-30 Hz).
- Full 18x18 symmetric adjacency matrices for Phase Locking Value (PLV) and Phase Lag Index (PLI).
- Mathematical graph metrics: Weighted nodal strength, weighted clustering coefficient (Onnela et al., 2005),
  weighted global efficiency (Latora & Marchiori, 2001), and eigenvector centrality.
- Biological patient-level statistical inference (N=23 independent biological replicates):
  Paired Wilcoxon signed-rank test, Cohen's d effect sizes with 95% bootstrap confidence intervals,
  Benjamini-Hochberg False Discovery Rate (FDR) q-values, and Bonferroni corrected p-values.
- Proportional connection density sensitivity analysis (rho in [0.10, 0.50]).
- Second-by-second temporal trajectory tracking ([-60s, +60s] around onset across multiple patients).
- 23-Fold Leave-One-Patient-Out Cross-Validation Detector Ablation benchmarking:
  1. Conventional Baseline (252)
  2. Connectivity Edge/Nodal (200)
  3. Combined Multimodal (452)
  4. Graph Global Metrics Only (40)
  5. Combined + Graph Metrics (492)
  6. Combined Multimodal + EWMA (452)
  7. Combined + Graph + EWMA (492)
"""

import json
from pathlib import Path
import shutil
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
from src.evaluation.postprocessing import ewma_smooth
from src.features.cache import FeatureCache
from src.features.connectivity import EEGConnectivityExtractor
from src.features.network_metrics import (
    clustering_coefficient_binary,
    compute_network_summary,
    compute_window_graph_features,
    global_efficiency_binary,
    threshold_proportional,
)
from src.features.unified import UnifiedFeatureExtractor
from src.models.baseline import BaselineClassifier
from src.pipelines.full_cohort_experiment import SUBJECT_PAIRS
from src.utils.config import CANONICAL_CHANNELS, DATASET_DIR, OUTPUTS_DIR, PATIENT_ID_MAPPING, SAMPLING_RATE


def benjamini_hochberg_fdr(p_values: np.ndarray) -> np.ndarray:
    """Compute Benjamini-Hochberg False Discovery Rate (FDR) adjusted q-values."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(p_values)
    ranked_p = p_values[order]
    q_values = np.zeros(n, dtype=float)
    cummin = float("inf")
    for i in range(n - 1, -1, -1):
        rank = i + 1
        q_raw = ranked_p[i] * n / rank
        cummin = min(cummin, q_raw)
        q_values[order[i]] = min(1.0, cummin)
    return q_values


def bonferroni_correction(p_values: np.ndarray) -> np.ndarray:
    """Compute Bonferroni adjusted p-values."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    return np.clip(p_values * n, 0.0, 1.0)


def bootstrap_cohen_d_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute paired Cohen's d with 95% percentile bootstrap confidence interval."""
    diff = x - y
    n = len(diff)
    std_diff = np.std(diff, ddof=1)
    d_pt = float(np.mean(diff) / std_diff) if std_diff > 0 else 0.0

    rng = np.random.default_rng(seed)
    boot_ds: List[float] = []
    for _ in range(n_boot):
        indices = rng.integers(0, n, size=n)
        b_diff = diff[indices]
        b_std = np.std(b_diff, ddof=1)
        if b_std > 0:
            boot_ds.append(float(np.mean(b_diff) / b_std))

    if len(boot_ds) > 0:
        ci_lower = float(np.percentile(boot_ds, 100.0 * (alpha / 2.0)))
        ci_upper = float(np.percentile(boot_ds, 100.0 * (1.0 - alpha / 2.0)))
    else:
        ci_lower, ci_upper = d_pt, d_pt

    return d_pt, ci_lower, ci_upper


class FullCohortNetworkExperiment:
    """End-to-End Functional Network Pipeline for All 24 CHB-MIT Cases / 23 Patients."""

    def __init__(
        self,
        experiment_id: str = "exp_20261013_full_cohort_networks",
        cache_experiment_id: str = "exp_20261012_sprint",
        random_state: int = 42,
    ):
        self.experiment_id = experiment_id
        self.random_state = random_state
        self.exp_dir = OUTPUTS_DIR / "experiments" / experiment_id
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.graph_cache_dir = self.exp_dir / "graph_cache"
        self.graph_cache_dir.mkdir(parents=True, exist_ok=True)

        # Seed graph cache from prior 10-patient experiment if present
        prior_graph_cache = OUTPUTS_DIR / "experiments" / "exp_20261013_network_analysis" / "graph_cache"
        if prior_graph_cache.exists():
            for p in prior_graph_cache.glob("*.npy"):
                target = self.graph_cache_dir / p.name
                if not target.exists():
                    shutil.copy(p, target)

        self.loader = EDFLoader()
        self.conn_extractor = EEGConnectivityExtractor(sampling_rate=SAMPLING_RATE)
        self.unified_extractor = UnifiedFeatureExtractor(sampling_rate=SAMPLING_RATE)
        self.cache_dir = OUTPUTS_DIR / "experiments" / cache_experiment_id / "feature_cache"
        self.cache = FeatureCache(cache_dir=self.cache_dir, extractor=self.unified_extractor)

        self.subject_pairs = SUBJECT_PAIRS
        self.patient_mapping = PATIENT_ID_MAPPING
        self.cases = sorted(list(self.subject_pairs.keys()))

        parser = CHBMITMetadataParser()
        _, self.seiz_df = parser.parse_all()

    def run_full_cohort_network_analysis(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
        """Extract full 18x18 matrices for all 28 annotated seizures vs matched baseline.

        Returns:
            Tuple of:
            - df_seizure_stats: per-seizure metrics table (28 rows)
            - df_case_stats: per-case aggregated metrics table (24 rows)
            - df_patient_stats: per-patient aggregated metrics table (23 biological patients)
            - df_contrast: statistical contrast inference table across 23 patients (24 rows)
            - rep_matrices: representative matrices for publication figures
        """
        print("\n" + "=" * 80)
        print("PHASE 2: FULL-COHORT FUNCTIONAL NETWORK NEUROSCIENCE ANALYSIS")
        print("=" * 80)

        w_len = int(4.0 * SAMPLING_RATE)
        stride = int(1.0 * SAMPLING_RATE)
        seizure_stat_rows: List[Dict] = []
        rep_matrices: Dict[str, Dict[str, np.ndarray]] = {}

        for case_id in self.cases:
            seiz_fn, ns_fn = self.subject_pairs[case_id]
            patient_id = self.patient_mapping.get(case_id, case_id)
            seiz_path = DATASET_DIR / case_id / seiz_fn
            ns_path = DATASET_DIR / case_id / ns_fn

            sub_seiz = self.seiz_df[(self.seiz_df["case_id"] == case_id) & (self.seiz_df["file_name"] == seiz_fn)]
            n_seiz = len(sub_seiz)

            # Load baseline interictal data (first 300s of matched non-seizure recording)
            ns_raw = self.loader.load_slice(ns_path, start_sample=0, stop_sample=300 * SAMPLING_RATE)

            for s_idx, (_, s_row) in enumerate(sub_seiz.iterrows(), start=1):
                sz_start_sec = float(s_row["start_sec"])
                sz_end_sec = float(s_row["end_sec"])
                sz_dur = sz_end_sec - sz_start_sec

                sz_start_samp = int(sz_start_sec * SAMPLING_RATE)
                sz_end_samp = int(sz_end_sec * SAMPLING_RATE)
                if sz_end_samp - sz_start_samp < w_len:
                    sz_end_samp = sz_start_samp + w_len

                # 1. Extract Ictal Windows
                ictal_raw = self.loader.load_slice(seiz_path, start_sample=sz_start_samp, stop_sample=sz_end_samp)
                ictal_wins = [
                    ictal_raw[:, i : i + w_len]
                    for i in range(0, ictal_raw.shape[1] - w_len + 1, stride)
                ]
                if not ictal_wins:
                    ictal_wins = [ictal_raw[:, :w_len]]
                ictal_wins = np.array(ictal_wins, dtype=np.float32)

                # 2. Extract Matched Interictal Windows
                ns_wins = [
                    ns_raw[:, i : i + w_len]
                    for i in range(0, min(ns_raw.shape[1] - w_len + 1, len(ictal_wins) * 10 * stride), 4 * SAMPLING_RATE)
                ]
                ns_wins = np.array(ns_wins[: len(ictal_wins)], dtype=np.float32)

                # Compute Full Adjacency Matrices
                ictal_mats = self.conn_extractor.compute_full_matrices(ictal_wins)
                ns_mats = self.conn_extractor.compute_full_matrices(ns_wins)

                # Save representative matrices for key diagnostic/canonical cases
                if case_id in ["chb01", "chb04", "chb15", "chb21"] and s_idx == 1:
                    rep_matrices[case_id] = {
                        "ictal_plv_theta": np.mean(ictal_mats["theta"]["plv"], axis=0),
                        "ictal_pli_theta": np.mean(ictal_mats["theta"]["pli"], axis=0),
                        "ictal_plv_delta": np.mean(ictal_mats["delta"]["plv"], axis=0),
                        "ictal_pli_delta": np.mean(ictal_mats["delta"]["pli"], axis=0),
                        "interictal_plv_theta": np.mean(ns_mats["theta"]["plv"], axis=0),
                        "interictal_pli_theta": np.mean(ns_mats["theta"]["pli"], axis=0),
                        "interictal_plv_delta": np.mean(ns_mats["delta"]["plv"], axis=0),
                        "interictal_pli_delta": np.mean(ns_mats["delta"]["pli"], axis=0),
                    }

                # Compute Network Metrics for each window and average across seizure epoch
                for band in self.conn_extractor.bands:
                    for c_type in ["plv", "pli"]:
                        ic_batch = ictal_mats[band][c_type]
                        ns_batch = ns_mats[band][c_type]

                        ic_strengths = [compute_network_summary(m)["mean_strength"] for m in ic_batch]
                        ns_strengths = [compute_network_summary(m)["mean_strength"] for m in ns_batch]

                        ic_clust = [compute_network_summary(m)["mean_clustering"] for m in ic_batch]
                        ns_clust = [compute_network_summary(m)["mean_clustering"] for m in ns_batch]

                        ic_eff = [compute_network_summary(m)["global_efficiency"] for m in ic_batch]
                        ns_eff = [compute_network_summary(m)["global_efficiency"] for m in ns_batch]

                        ic_cent = [compute_network_summary(m)["max_centrality"] for m in ic_batch]
                        ns_cent = [compute_network_summary(m)["max_centrality"] for m in ns_batch]

                        seizure_stat_rows.append({
                            "case_id": case_id,
                            "patient_id": patient_id,
                            "file_name": seiz_fn,
                            "seizure_idx": s_idx,
                            "seizure_start_sec": sz_start_sec,
                            "seizure_end_sec": sz_end_sec,
                            "seizure_duration_sec": sz_dur,
                            "n_ictal_windows": len(ictal_wins),
                            "band": band,
                            "conn_type": c_type,
                            "ictal_strength_mean": float(np.mean(ic_strengths)),
                            "interictal_strength_mean": float(np.mean(ns_strengths)),
                            "ictal_clustering_mean": float(np.mean(ic_clust)),
                            "interictal_clustering_mean": float(np.mean(ns_clust)),
                            "ictal_efficiency_mean": float(np.mean(ic_eff)),
                            "interictal_efficiency_mean": float(np.mean(ns_eff)),
                            "ictal_centrality_mean": float(np.mean(ic_cent)),
                            "interictal_centrality_mean": float(np.mean(ns_cent)),
                        })

            print(f"  Processed {case_id} ({patient_id}): {n_seiz} seizure(s) analyzed.")

        df_seizure_stats = pd.DataFrame(seizure_stat_rows)

        # 1. Per-Case Aggregated Table (24 cases)
        metric_cols = [
            "ictal_strength_mean", "interictal_strength_mean",
            "ictal_clustering_mean", "interictal_clustering_mean",
            "ictal_efficiency_mean", "interictal_efficiency_mean",
            "ictal_centrality_mean", "interictal_centrality_mean",
        ]
        df_case_stats = (
            df_seizure_stats.groupby(["case_id", "patient_id", "band", "conn_type"])[metric_cols]
            .mean()
            .reset_index()
        )

        # 2. Per-Patient Aggregated Table (23 biological patients, uniting chb01 and chb21)
        df_patient_stats = (
            df_seizure_stats.groupby(["patient_id", "band", "conn_type"])[metric_cols]
            .mean()
            .reset_index()
        )

        # 3. Statistical Inference Table across N=23 Biological Patients
        contrast_rows: List[Dict] = []
        raw_p_values: List[float] = []

        # We test 4 bands x 2 conn types x 3 primary metrics = 24 hypotheses
        metric_triplets = [
            ("mean_strength", "ictal_strength_mean", "interictal_strength_mean"),
            ("clustering_coefficient", "ictal_clustering_mean", "interictal_clustering_mean"),
            ("global_efficiency", "ictal_efficiency_mean", "interictal_efficiency_mean"),
        ]

        for band in self.conn_extractor.bands:
            for c_type in ["plv", "pli"]:
                sub = df_patient_stats[
                    (df_patient_stats["band"] == band) & (df_patient_stats["conn_type"] == c_type)
                ]

                for metric, col_ic, col_ns in metric_triplets:
                    ic_vals = sub[col_ic].values
                    ns_vals = sub[col_ns].values
                    diff = ic_vals - ns_vals

                    # Paired Cohen's d with 95% bootstrap CI
                    d_pt, d_ci_lo, d_ci_hi = bootstrap_cohen_d_ci(
                        ic_vals, ns_vals, n_boot=1000, alpha=0.05, seed=self.random_state
                    )

                    # Paired Wilcoxon signed-rank test
                    try:
                        w_res = wilcoxon(ic_vals, ns_vals)
                        p_val = float(w_res.pvalue)
                        stat_val = float(w_res.statistic)
                    except Exception:
                        p_val = 1.0
                        stat_val = 0.0

                    raw_p_values.append(p_val)

                    n_pos = int(np.sum(diff > 0))
                    contrast_rows.append({
                        "band": band,
                        "conn_type": c_type,
                        "metric": metric,
                        "n_patients": len(sub),
                        "interictal_mean": float(np.mean(ns_vals)),
                        "interictal_std": float(np.std(ns_vals, ddof=1)),
                        "ictal_mean": float(np.mean(ic_vals)),
                        "ictal_std": float(np.std(ic_vals, ddof=1)),
                        "mean_change": float(np.mean(diff)),
                        "pct_change": float(np.mean(diff) / np.mean(ns_vals) * 100.0) if np.mean(ns_vals) > 0 else 0.0,
                        "cohen_d": d_pt,
                        "cohen_d_ci_lower": d_ci_lo,
                        "cohen_d_ci_upper": d_ci_hi,
                        "wilcoxon_stat": stat_val,
                        "wilcoxon_p_raw": p_val,
                        "patients_with_increase": n_pos,
                        "pct_patients_increase": float(n_pos / len(sub) * 100.0),
                    })

        # Multiple Testing Correction across the 24 tests
        fdr_qs = benjamini_hochberg_fdr(np.array(raw_p_values))
        bonf_ps = bonferroni_correction(np.array(raw_p_values))

        for idx, row in enumerate(contrast_rows):
            row["fdr_q_value"] = float(fdr_qs[idx])
            row["bonferroni_p"] = float(bonf_ps[idx])
            row["is_fdr_significant"] = bool(fdr_qs[idx] < 0.05)
            row["is_bonf_significant"] = bool(bonf_ps[idx] < 0.05)

        df_contrast = pd.DataFrame(contrast_rows)
        return df_seizure_stats, df_case_stats, df_patient_stats, df_contrast, rep_matrices

    def run_threshold_sensitivity_analysis(
        self,
        test_cases: List[str] = ["chb01", "chb03", "chb05", "chb08", "chb12", "chb15", "chb21", "chb24"],
    ) -> pd.DataFrame:
        """Evaluate clustering and efficiency stability across density thresholds (0.10 to 0.50)."""
        print("\n" + "=" * 80)
        print("PROPORTIONAL DENSITY THRESHOLD SENSITIVITY ANALYSIS (COHORT-WIDE)")
        print("=" * 80)

        densities = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
        rows: List[Dict] = []
        w_len = int(4.0 * SAMPLING_RATE)

        for case_id in test_cases:
            seiz_fn, ns_fn = self.subject_pairs[case_id]
            seiz_path = DATASET_DIR / case_id / seiz_fn
            ns_path = DATASET_DIR / case_id / ns_fn

            sub_seiz = self.seiz_df[(self.seiz_df["case_id"] == case_id) & (self.seiz_df["file_name"] == seiz_fn)]
            first_seiz = sub_seiz.iloc[0]
            sz_start_samp = int(float(first_seiz["start_sec"]) * SAMPLING_RATE)

            ic_raw = self.loader.load_slice(seiz_path, start_sample=sz_start_samp, stop_sample=sz_start_samp + 20 * w_len)
            ns_raw = self.loader.load_slice(ns_path, start_sample=0, stop_sample=20 * w_len)

            ic_wins = np.array([ic_raw[:, i : i + w_len] for i in range(0, ic_raw.shape[1] - w_len + 1, w_len)], dtype=np.float32)
            ns_wins = np.array([ns_raw[:, i : i + w_len] for i in range(0, ns_raw.shape[1] - w_len + 1, w_len)], dtype=np.float32)

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
                            "case_id": case_id,
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
        summary_sens = (
            df_sens.groupby(["density", "band", "conn_type"])[
                ["ictal_clustering", "interictal_clustering", "ictal_efficiency", "interictal_efficiency", "clustering_diff", "efficiency_diff"]
            ]
            .mean()
            .reset_index()
        )
        return summary_sens

    def run_seizure_trajectories(
        self, target_cases: List[str] = ["chb01", "chb04", "chb15", "chb21"]
    ) -> pd.DataFrame:
        """Track temporal evolution of graph metrics across [-60s, +60s] around onset."""
        print("\n" + "=" * 80)
        print("TEMPORAL TRAJECTORY DYNAMICS ACROSS DIVERSE PATIENTS")
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
                epoch_times.append(epoch_start_sec + i / SAMPLING_RATE)

            epoch_wins = np.array(epoch_wins, dtype=np.float32)
            mats = self.conn_extractor.compute_full_matrices(epoch_wins)

            for w_idx, t_sec in enumerate(epoch_times):
                is_ictal = int(sz_start_sec <= t_sec <= sz_end_sec)
                rel_time = t_sec - sz_start_sec

                theta_plv_m = mats["theta"]["plv"][w_idx]
                theta_pli_m = mats["theta"]["pli"][w_idx]

                s_plv = compute_network_summary(theta_plv_m)
                s_pli = compute_network_summary(theta_pli_m)

                traj_rows.append({
                    "case_id": case_id,
                    "patient_id": self.patient_mapping.get(case_id, case_id),
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

            print(f"  Generated temporal trajectory for {case_id} ({len(epoch_wins)} windows).")

        return pd.DataFrame(traj_rows)

    def extract_and_cache_graph_features(self, data_by_case: Dict[str, Dict]) -> None:
        """Extract and cache 40 tabular graph features for each case."""
        print("\nExtracting & Caching 40 Tabular Graph Features for All 24 Cases...")

        for case_id in self.cases:
            seiz_fn, ns_fn = self.subject_pairs[case_id]
            seiz_path = DATASET_DIR / case_id / seiz_fn
            ns_path = DATASET_DIR / case_id / ns_fn

            tr_cache_path = self.graph_cache_dir / f"{case_id}_train_graph.npy"
            eval_cache_path = self.graph_cache_dir / f"{case_id}_eval_graph.npy"

            if tr_cache_path.exists() and eval_cache_path.exists():
                comb_tr_graph = np.load(tr_cache_path)
                eval_graph_feats = np.load(eval_cache_path)
                print(f"  [CACHE HIT] Loaded graph features for {case_id} (train: {comb_tr_graph.shape}, eval: {eval_graph_feats.shape})")
            else:
                s_tr_df = data_by_case[case_id]["tr_df"]
                s_ns_df = data_by_case[case_id]["ns_df"]
                s_eval_df = data_by_case[case_id]["eval_df"]

                # 1. Evaluation recording
                seiz_sig, _ = self.loader.load_full_recording(seiz_path)
                eval_wins = np.array([
                    seiz_sig[:, int(r.start_sample) : int(r.end_sample)]
                    for r in s_eval_df.itertuples()
                ], dtype=np.float32)
                mats_eval = self.conn_extractor.compute_full_matrices(eval_wins)
                eval_graph_feats, _ = compute_window_graph_features(mats_eval, densities=[])

                # 2. Training: seizure recording windows
                tr_sz_wins = np.array([
                    seiz_sig[:, int(r.start_sample) : int(r.end_sample)]
                    for r in s_tr_df.itertuples()
                ], dtype=np.float32)
                mats_tr_sz = self.conn_extractor.compute_full_matrices(tr_sz_wins)
                tr_sz_graph, _ = compute_window_graph_features(mats_tr_sz, densities=[])

                # 3. Training: non-seizure recording windows
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
                print(f"  Extracted and cached graph features for {case_id} (train: {comb_tr_graph.shape}, eval: {eval_graph_feats.shape})")

            assert len(comb_tr_graph) == len(data_by_case[case_id]["train_y"]), f"Mismatch in train rows for {case_id}"
            assert len(eval_graph_feats) == len(data_by_case[case_id]["eval_y"]), f"Mismatch in eval rows for {case_id}"

            data_by_case[case_id]["train_graph"] = comb_tr_graph
            data_by_case[case_id]["eval_graph"] = eval_graph_feats

    def run_detector_ablation_experiment(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run 23-Fold Leave-One-Patient-Out Cross-Validation Detector Ablation."""
        print("\n" + "=" * 80)
        print("PHASE 3: 23-FOLD LEAVE-ONE-PATIENT-OUT DETECTOR ABLATION BENCHMARK")
        print("=" * 80)

        # 1. Load Parquet feature tables for all 24 cases
        data_by_case: Dict[str, Dict] = {}
        for case_id in self.cases:
            seiz_fn, ns_fn = self.subject_pairs[case_id]
            patient_id = self.patient_mapping.get(case_id, case_id)

            subj_seizures = self.seiz_df[
                (self.seiz_df["case_id"] == case_id) & (self.seiz_df["file_name"] == seiz_fn)
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
                case_id=case_id,
                file_name=seiz_fn,
                mode="training",
                seizure_intervals=seiz_intervals,
                neg_pos_ratio=10.0,
                random_seed=self.random_state,
                verbose=False,
            )
            tr_X_seiz, tr_y_seiz, _ = self.cache.df_to_features_and_labels(tr_df)

            ns_df = self.cache.extract_or_load(
                case_id=case_id,
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
                case_id=case_id,
                file_name=seiz_fn,
                mode="evaluation",
                seizure_intervals=seiz_intervals,
                eval_stride_samples=SAMPLING_RATE,
                verbose=False,
            )
            eval_X, eval_y, eval_meta = self.cache.df_to_features_and_labels(eval_df)
            duration_sec = float(len(eval_y))

            data_by_case[case_id] = {
                "case_id": case_id,
                "patient_id": patient_id,
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

        # 2. Extract and cache graph features
        self.extract_and_cache_graph_features(data_by_case)

        # 3. 23-Fold Leave-One-Patient-Out Cross-Validation
        unique_patients = sorted(list(set(self.patient_mapping[c] for c in self.cases)))
        all_case_rows: List[Dict] = []

        for p_idx, held_out_patient in enumerate(unique_patients, start=1):
            train_cases = [c for c in self.cases if self.patient_mapping[c] != held_out_patient]
            eval_cases = [c for c in self.cases if self.patient_mapping[c] == held_out_patient]

            # Pool training data across all training patients
            raw_X_tr = np.concatenate([data_by_case[c]["train_X_full"] for c in train_cases], axis=0)
            graph_X_tr = np.concatenate([data_by_case[c]["train_graph"] for c in train_cases], axis=0)
            y_tr = np.concatenate([data_by_case[c]["train_y"] for c in train_cases], axis=0)

            # Feature slices for training
            conv_X_tr = self.unified_extractor.slice_features(raw_X_tr, "conventional")
            conn_X_tr = self.unified_extractor.slice_features(raw_X_tr, "connectivity")
            comb_X_tr = raw_X_tr
            comb_plus_graph_tr = np.concatenate([raw_X_tr, graph_X_tr], axis=1)

            # Define 7 benchmark model configurations
            configs = [
                ("conventional", "Conventional Baseline (252)", conv_X_tr, 252, False),
                ("connectivity", "Connectivity Edge/Nodal (200)", conn_X_tr, 200, False),
                ("combined", "Combined Multimodal (452)", comb_X_tr, 452, False),
                ("graph_only", "Graph Global Metrics Only (40)", graph_X_tr, 40, False),
                ("combined_plus_graph", "Combined + Graph Metrics (492)", comb_plus_graph_tr, 492, False),
                ("combined_ewma", "Combined Multimodal + EWMA (452)", comb_X_tr, 452, True),
                ("combined_plus_graph_ewma", "Combined + Graph + EWMA (492)", comb_plus_graph_tr, 492, True),
            ]

            # Fit 7 classifiers once per outer patient fold
            fitted_models = {}
            for cfg_id, cfg_name, X_tr, n_f, is_ewma in configs:
                clf = BaselineClassifier(model_type="rf", random_state=self.random_state)
                clf.fit(X_tr, y_tr)
                fitted_models[cfg_id] = clf

            # Evaluate each held-out case belonging to this patient
            for held_out_case in eval_cases:
                raw_X_te = data_by_case[held_out_case]["eval_X_full"]
                graph_X_te = data_by_case[held_out_case]["eval_graph"]
                y_te = data_by_case[held_out_case]["eval_y"]
                meta_te = data_by_case[held_out_case]["eval_meta"]
                annot_sz = data_by_case[held_out_case]["annotated_seizures_sec"]
                dur_sec = data_by_case[held_out_case]["duration_sec"]

                conv_X_te = self.unified_extractor.slice_features(raw_X_te, "conventional")
                conn_X_te = self.unified_extractor.slice_features(raw_X_te, "connectivity")
                comb_X_te = raw_X_te
                comb_plus_graph_te = np.concatenate([raw_X_te, graph_X_te], axis=1)

                test_feature_maps = {
                    "conventional": conv_X_te,
                    "connectivity": conn_X_te,
                    "combined": comb_X_te,
                    "graph_only": graph_X_te,
                    "combined_plus_graph": comb_plus_graph_te,
                    "combined_ewma": comb_X_te,
                    "combined_plus_graph_ewma": comb_plus_graph_te,
                }

                for cfg_id, cfg_name, _, n_f, is_ewma in configs:
                    clf = fitted_models[cfg_id]
                    prob = clf.predict_proba(test_feature_maps[cfg_id])

                    if is_ewma:
                        prob = ewma_smooth(prob, alpha=0.50)
                        post_str = "ewma_0.50"
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

                    all_case_rows.append({
                        "case_id": held_out_case,
                        "patient_id": held_out_patient,
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

            print(f"  LOPO Fold {p_idx}/23 ({held_out_patient} -> {eval_cases}) evaluated.")

        df_case_folds = pd.DataFrame(all_case_rows)

        # 4. Aggregate at Patient Level (23 biological patients)
        patient_group_rows: List[Dict] = []
        for p_id in unique_patients:
            for cfg_id in df_case_folds["config_id"].unique():
                sub = df_case_folds[(df_case_folds["patient_id"] == p_id) & (df_case_folds["config_id"] == cfg_id)]
                patient_group_rows.append({
                    "patient_id": p_id,
                    "config_id": cfg_id,
                    "config_name": sub["config_name"].iloc[0],
                    "feature_count": sub["feature_count"].iloc[0],
                    "postprocessing": sub["postprocessing"].iloc[0],
                    "pr_auc": float(sub["pr_auc"].mean()),
                    "roc_auc": float(sub["roc_auc"].mean()),
                    "event_sensitivity": float(sub["detected_seizures"].sum() / sub["total_seizures"].sum()) if sub["total_seizures"].sum() > 0 else 0.0,
                    "detected_seizures": int(sub["detected_seizures"].sum()),
                    "total_seizures": int(sub["total_seizures"].sum()),
                    "false_alarms": int(sub["false_alarms"].sum()),
                    "fp_per_hour": float(sub["false_alarms"].sum() / sub["duration_hours"].sum()),
                    "duration_hours": float(sub["duration_hours"].sum()),
                    "mean_delay_sec": float(sub["mean_delay_sec"].dropna().mean()) if len(sub["mean_delay_sec"].dropna()) > 0 else float("nan"),
                })
        df_patient_folds = pd.DataFrame(patient_group_rows)

        # 5. Cohort Summary Table
        summary_rows: List[Dict] = []
        for cfg_id in df_case_folds["config_id"].unique():
            c_sub = df_case_folds[df_case_folds["config_id"] == cfg_id]
            p_sub = df_patient_folds[df_patient_folds["config_id"] == cfg_id]
            c_delays = c_sub["mean_delay_sec"].dropna().values

            summary_rows.append({
                "config_id": cfg_id,
                "config_name": c_sub["config_name"].iloc[0],
                "feature_count": int(c_sub["feature_count"].iloc[0]),
                "postprocessing": c_sub["postprocessing"].iloc[0],
                "macro_case_pr_auc": float(c_sub["pr_auc"].mean()),
                "std_case_pr_auc": float(c_sub["pr_auc"].std()),
                "macro_case_roc_auc": float(c_sub["roc_auc"].mean()),
                "macro_case_event_sens": float(c_sub["event_sensitivity"].mean()),
                "macro_case_fp_per_hour": float(c_sub["fp_per_hour"].mean()),
                "macro_patient_pr_auc": float(p_sub["pr_auc"].mean()),
                "std_patient_pr_auc": float(p_sub["pr_auc"].std()),
                "macro_patient_roc_auc": float(p_sub["roc_auc"].mean()),
                "macro_patient_event_sens": float(p_sub["event_sensitivity"].mean()),
                "macro_patient_fp_per_hour": float(p_sub["fp_per_hour"].mean()),
                "pooled_detected_seizures": int(c_sub["detected_seizures"].sum()),
                "pooled_total_seizures": int(c_sub["total_seizures"].sum()),
                "pooled_event_sensitivity": float(c_sub["detected_seizures"].sum() / c_sub["total_seizures"].sum()),
                "pooled_false_alarms": int(c_sub["false_alarms"].sum()),
                "total_evaluation_hours": float(c_sub["duration_hours"].sum()),
                "pooled_fp_per_hour": float(c_sub["false_alarms"].sum() / c_sub["duration_hours"].sum()),
                "mean_detection_delay_sec": float(np.mean(c_delays)) if len(c_delays) > 0 else float("nan"),
                "median_detection_delay_sec": float(np.median(c_delays)) if len(c_delays) > 0 else float("nan"),
            })

        df_summary = pd.DataFrame(summary_rows)
        return df_case_folds, df_patient_folds, df_summary

    def generate_figures(
        self,
        rep_matrices: Dict[str, Dict[str, np.ndarray]],
        df_patient_stats: pd.DataFrame,
        df_traj: pd.DataFrame,
        df_sens: pd.DataFrame,
        df_summary: pd.DataFrame,
        out_dir: Path,
    ) -> List[Path]:
        """Generate high-resolution publication figures."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        ch_labels = [c.replace("-", "\n") for c in CANONICAL_CHANNELS]

        # 1. Full-Cohort Adjacency Heatmaps (chb01, chb21 longitudinal pair & chb04)
        fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), dpi=300)
        m_01 = rep_matrices.get("chb01", next(iter(rep_matrices.values())))

        im0 = axes[0, 0].imshow(m_01["interictal_plv_theta"], cmap="viridis", vmin=0, vmax=1)
        axes[0, 0].set_title("Interictal PLV (Theta 4-8 Hz) [chb01]", fontsize=11, fontweight="bold")
        axes[0, 0].set_xticks(range(18)); axes[0, 0].set_xticklabels(ch_labels, fontsize=6)
        axes[0, 0].set_yticks(range(18)); axes[0, 0].set_yticklabels(ch_labels, fontsize=6)
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im1 = axes[0, 1].imshow(m_01["ictal_plv_theta"], cmap="viridis", vmin=0, vmax=1)
        axes[0, 1].set_title("Ictal Seizure PLV (Theta 4-8 Hz) [chb01]", fontsize=11, fontweight="bold")
        axes[0, 1].set_xticks(range(18)); axes[0, 1].set_xticklabels(ch_labels, fontsize=6)
        axes[0, 1].set_yticks(range(18)); axes[0, 1].set_yticklabels(ch_labels, fontsize=6)
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[1, 0].imshow(m_01["interictal_pli_theta"], cmap="inferno", vmin=0, vmax=0.8)
        axes[1, 0].set_title("Interictal PLI (Theta 4-8 Hz) [chb01]", fontsize=11, fontweight="bold")
        axes[1, 0].set_xticks(range(18)); axes[1, 0].set_xticklabels(ch_labels, fontsize=6)
        axes[1, 0].set_yticks(range(18)); axes[1, 0].set_yticklabels(ch_labels, fontsize=6)
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im3 = axes[1, 1].imshow(m_01["ictal_pli_theta"], cmap="inferno", vmin=0, vmax=0.8)
        axes[1, 1].set_title("Ictal Seizure PLI (Theta 4-8 Hz) [chb01]", fontsize=11, fontweight="bold")
        axes[1, 1].set_xticks(range(18)); axes[1, 1].set_xticklabels(ch_labels, fontsize=6)
        axes[1, 1].set_yticks(range(18)); axes[1, 1].set_yticklabels(ch_labels, fontsize=6)
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

        plt.suptitle("Functional Network Adjacency Reorganization Across States", fontsize=14, y=0.98)
        plt.tight_layout()
        p1 = out_dir / "full_cohort_adjacency_heatmaps.png"
        fig.savefig(p1)
        plt.close(fig)
        saved.append(p1)

        # 2. Cohort-Wide Theta PLI Patient Distribution / Forest Plot (N=23)
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), dpi=300, sharey=True)
        theta_pli_sub = df_patient_stats[
            (df_patient_stats["band"] == "theta") & (df_patient_stats["conn_type"] == "pli")
        ].sort_values("patient_id")

        pts = theta_pli_sub["patient_id"].values
        y_pos = np.arange(len(pts))

        # Strength diff
        str_diff = theta_pli_sub["ictal_strength_mean"].values - theta_pli_sub["interictal_strength_mean"].values
        axes[0].barh(y_pos, str_diff, color=["#2ca02c" if v > 0 else "#d62728" for v in str_diff])
        axes[0].axvline(0, color="black", linestyle="--", lw=1)
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(pts, fontsize=8)
        axes[0].set_title("Theta PLI Connection Strength Δ\n(Ictal - Interictal)", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Mean Strength Change", fontsize=10)
        axes[0].grid(axis="x", linestyle="--", alpha=0.5)

        # Clustering diff
        clu_diff = theta_pli_sub["ictal_clustering_mean"].values - theta_pli_sub["interictal_clustering_mean"].values
        axes[1].barh(y_pos, clu_diff, color=["#1f77b4" if v > 0 else "#d62728" for v in clu_diff])
        axes[1].axvline(0, color="black", linestyle="--", lw=1)
        axes[1].set_title("Theta PLI Clustering Coefficient Δ\n(Ictal - Interictal)", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Clustering Coeff Change", fontsize=10)
        axes[1].grid(axis="x", linestyle="--", alpha=0.5)

        # Efficiency diff
        eff_diff = theta_pli_sub["ictal_efficiency_mean"].values - theta_pli_sub["interictal_efficiency_mean"].values
        axes[2].barh(y_pos, eff_diff, color=["#ff7f0e" if v > 0 else "#d62728" for v in eff_diff])
        axes[2].axvline(0, color="black", linestyle="--", lw=1)
        axes[2].set_title("Theta PLI Global Efficiency Δ\n(Ictal - Interictal)", fontsize=11, fontweight="bold")
        axes[2].set_xlabel("Global Efficiency Change", fontsize=10)
        axes[2].grid(axis="x", linestyle="--", alpha=0.5)

        plt.suptitle("Patient-Level Reorganization Consistency Across Full Cohort (N=23)", fontsize=13, y=0.98)
        plt.tight_layout()
        p2 = out_dir / "full_cohort_theta_pli_patient_distribution.png"
        fig.savefig(p2)
        plt.close(fig)
        saved.append(p2)

        # 3. Multi-Patient Temporal Trajectories around Onset
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=300, sharex=True)
        cases_to_plot = ["chb01", "chb04", "chb15", "chb21"]

        for idx, c_id in enumerate(cases_to_plot):
            ax = axes[idx // 2, idx % 2]
            c_traj = df_traj[df_traj["case_id"] == c_id]
            if len(c_traj) == 0:
                continue

            sz_dur = float(c_traj["seiz_end_sec"].iloc[0] - c_traj["seiz_start_sec"].iloc[0])
            ax.plot(c_traj["rel_time_to_onset_sec"], c_traj["pli_theta_efficiency"], label="Global Efficiency", color="#1f77b4", lw=2)
            ax.plot(c_traj["rel_time_to_onset_sec"], c_traj["pli_theta_clustering"], label="Clustering Coeff", color="#ff7f0e", lw=2)
            ax.plot(c_traj["rel_time_to_onset_sec"], c_traj["pli_theta_mean_strength"] / 17.0, label="Norm Strength", color="#2ca02c", lw=2)
            ax.axvspan(0, sz_dur, color="red", alpha=0.15, label="Ictal Seizure")
            ax.axvline(0, color="red", linestyle="--", lw=1.5)
            ax.set_title(f"Case {c_id} ({c_traj['patient_id'].iloc[0]}) - Duration: {sz_dur:.1f}s", fontsize=10, fontweight="bold")
            ax.set_ylabel("Network Metric", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.5)
            if idx == 0:
                ax.legend(loc="upper right", fontsize=8)

        axes[1, 0].set_xlabel("Time Relative to Seizure Onset (s)", fontsize=10)
        axes[1, 1].set_xlabel("Time Relative to Seizure Onset (s)", fontsize=10)
        plt.suptitle("Theta PLI Dynamics Around Seizure Onset Across Distinct Patients", fontsize=13, y=0.98)
        plt.tight_layout()
        p3 = out_dir / "full_cohort_temporal_trajectories.png"
        fig.savefig(p3)
        plt.close(fig)
        saved.append(p3)

        # 4. Threshold Density Sensitivity Curves
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
        p4 = out_dir / "full_cohort_threshold_density_sensitivity.png"
        fig.savefig(p4)
        plt.close(fig)
        saved.append(p4)

        # 5. Full-Cohort Detector Ablation Comparison
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
        x = np.arange(len(df_summary))
        w = 0.35

        # Bar 1: PR-AUC & ROC-AUC
        axes[0].bar(x - w / 2, df_summary["macro_patient_pr_auc"], w, label="Macro PR-AUC", color="#1f77b4")
        axes[0].bar(x + w / 2, df_summary["macro_patient_roc_auc"], w, label="Macro ROC-AUC", color="#2ca02c")
        axes[0].set_ylabel("Score", fontsize=11)
        axes[0].set_title("Patient-Level Discrimination (23 Folds)", fontsize=12)
        axes[0].set_xticks(x)
        labels = [c.replace(" Baseline", "").replace(" Multimodal", "").replace(" + EWMA", "\n+EWMA") for c in df_summary["config_name"]]
        axes[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
        axes[0].set_ylim(0, 1.05)
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)
        axes[0].legend(fontsize=10)

        # Bar 2: Pooled FP/h & Event Sensitivity
        ax2_twin = axes[1].twinx()
        p1_bar = axes[1].bar(x - w / 2, df_summary["pooled_fp_per_hour"], w, label="Pooled FP/h", color="#d62728")
        p2_bar = ax2_twin.bar(x + w / 2, df_summary["pooled_event_sensitivity"] * 100.0, w, label="Event Sens (%)", color="#9467bd")
        axes[1].set_ylabel("Pooled False Alarms / Hour", fontsize=11, color="#d62728")
        ax2_twin.set_ylabel("Event Sensitivity (%)", fontsize=11, color="#9467bd")
        axes[1].set_title("Clinical False Alarm Rate & Sensitivity", fontsize=12)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)
        axes[1].set_ylim(0, max(df_summary["pooled_fp_per_hour"]) * 1.15)
        ax2_twin.set_ylim(0, 105)

        plt.suptitle("Full-Cohort CHB-MIT Detector Benchmark (23 Biological Patients / 36.0h)", fontsize=13, y=0.98)
        plt.tight_layout()
        p5 = out_dir / "full_cohort_detector_ablation_comparison.png"
        fig.savefig(p5)
        plt.close(fig)
        saved.append(p5)

        return saved

    def run_all(self) -> Dict[str, Union[pd.DataFrame, List[Path]]]:
        """Execute complete full-cohort network neuroscience and detector experiment."""
        print("Starting Complete Full-Cohort Functional Network Analysis Experiment...")
        t0 = time.time()

        # 1. Full-Cohort Ictal vs. Interictal Network Contrast
        df_seiz, df_case, df_patient, df_contrast, rep_matrices = self.run_full_cohort_network_analysis()
        df_seiz.to_csv(self.exp_dir / "per_seizure_network_stats.csv", index=False)
        df_case.to_csv(self.exp_dir / "per_case_network_stats.csv", index=False)
        df_patient.to_csv(self.exp_dir / "per_patient_network_stats.csv", index=False)
        df_contrast.to_csv(self.exp_dir / "ictal_interictal_network_stats.csv", index=False)
        print(f"Saved network stats: {len(df_seiz)} seizures, {len(df_case)} cases, {len(df_patient)} patients, {len(df_contrast)} contrast rows.")

        # 2. Threshold Sensitivity Analysis
        df_sens = self.run_threshold_sensitivity_analysis()
        df_sens.to_csv(self.exp_dir / "threshold_density_sensitivity.csv", index=False)
        print(f"Saved threshold_density_sensitivity.csv ({len(df_sens)} rows)")

        # 3. Seizure Trajectory Dynamics
        df_traj = self.run_seizure_trajectories()
        df_traj.to_csv(self.exp_dir / "seizure_trajectories.csv", index=False)
        print(f"Saved seizure_trajectories.csv ({len(df_traj)} rows)")

        # 4. Detector Ablation Benchmark (23 Folds)
        df_case_folds, df_patient_folds, df_summary = self.run_detector_ablation_experiment()
        df_case_folds.to_csv(self.exp_dir / "full_cohort_network_model_folds.csv", index=False)
        df_patient_folds.to_csv(self.exp_dir / "full_cohort_network_patient_folds.csv", index=False)
        df_summary.to_csv(self.exp_dir / "full_cohort_network_model_summary.csv", index=False)
        print(f"Saved ablation results: {len(df_case_folds)} case folds, {len(df_patient_folds)} patient folds, {len(df_summary)} summary rows.")

        # 5. Generate Figures
        fig_paths = self.generate_figures(
            rep_matrices=rep_matrices,
            df_patient_stats=df_patient,
            df_traj=df_traj,
            df_sens=df_sens,
            df_summary=df_summary,
            out_dir=self.exp_dir,
        )
        print(f"Generated {len(fig_paths)} publication figures in {self.exp_dir}")

        elapsed = time.time() - t0
        print(f"\nFull-Cohort Network Experiment complete in {elapsed:.1f}s ({elapsed / 60:.2f} minutes).")

        return {
            "df_seiz": df_seiz,
            "df_case": df_case,
            "df_patient": df_patient,
            "df_contrast": df_contrast,
            "df_sens": df_sens,
            "df_traj": df_traj,
            "df_case_folds": df_case_folds,
            "df_patient_folds": df_patient_folds,
            "df_summary": df_summary,
            "fig_paths": fig_paths,
        }


if __name__ == "__main__":
    exp = FullCohortNetworkExperiment()
    results = exp.run_all()
    print("\nBenchmark Summary Results (Full Cohort):")
    print(results["df_summary"][[
        "config_id", "feature_count", "postprocessing",
        "macro_patient_pr_auc", "macro_patient_roc_auc",
        "pooled_event_sensitivity", "pooled_fp_per_hour"
    ]])
