"""Comprehensive Overnight Research Sprint Pipeline.

Orchestrates:
- Stage 1: Reproduction of baseline & connectivity comparison into timestamped experiment directory.
- Stage 2: Controlled ablations (A: Conventional, B: PLV-only, C: PLI-only, D: PLV+PLI, E: Combined,
  F: Conv+PLV, G: Conv+PLI) under fixed (0.5) and training-calibrated decision thresholds.
- Stage 3: Conditional 10-patient expansion (CHB01, 03, 05, 07, 08 + CHB02, 04, 06, 09, 10).
- Stage 4: Feature importance analysis, electrophysiological ranking, and publication-ready plots.
- Stage 5: Comprehensive deliverables, metrics summaries, and formal research report.
"""

import json
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import EDFLoader
from src.data.metadata_parser import CHBMITMetadataParser
from src.data.windowing import WindowGenerator
from src.evaluation.metrics import compute_event_metrics, compute_window_metrics
from src.features.cache import FeatureCache
from src.features.unified import UnifiedFeatureExtractor
from src.models.baseline import BaselineClassifier
from src.utils.config import DATASET_DIR, OUTPUTS_DIR, PROTOTYPE_COHORT, SAMPLING_RATE


class SprintPipeline:
    """End-to-end research sprint executor."""

    def __init__(
        self,
        experiment_id: str = "exp_20261012_sprint",
        random_state: int = 42,
    ):
        self.experiment_id = experiment_id
        self.random_state = random_state
        self.exp_dir = OUTPUTS_DIR / "experiments" / experiment_id
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = self.exp_dir / "feature_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.stage1_dir = self.exp_dir / "stage1_reproduction"
        self.stage1_dir.mkdir(parents=True, exist_ok=True)

        self.stage2_dir = self.exp_dir / "stage2_ablations"
        self.stage2_dir.mkdir(parents=True, exist_ok=True)

        self.stage3_dir = self.exp_dir / "stage3_expansion"
        self.stage3_dir.mkdir(parents=True, exist_ok=True)

        self.stage4_dir = self.exp_dir / "stage4_analysis"
        self.stage4_dir.mkdir(parents=True, exist_ok=True)

        self.extractor = UnifiedFeatureExtractor(sampling_rate=SAMPLING_RATE)
        self.cache = FeatureCache(cache_dir=self.cache_dir, extractor=self.extractor)
        self.loader = EDFLoader()
        self.window_gen = WindowGenerator()

        parser = CHBMITMetadataParser()
        self.recs_df, self.seiz_df = parser.parse_all()

        # Benchmark pair definitions
        self.prototype_pairs: Dict[str, Tuple[str, str]] = {
            "chb01": ("chb01_03.edf", "chb01_01.edf"),
            "chb03": ("chb03_01.edf", "chb03_05.edf"),
            "chb05": ("chb05_06.edf", "chb05_01.edf"),
            "chb07": ("chb07_19.edf", "chb07_01.edf"),
            "chb08": ("chb08_02.edf", "chb08_03.edf"),
        }

        self.expansion_pairs: Dict[str, Tuple[str, str]] = {
            "chb02": ("chb02_16.edf", "chb02_01.edf"),
            "chb04": ("chb04_05.edf", "chb04_01.edf"),
            "chb06": ("chb06_01.edf", "chb06_02.edf"),
            "chb09": ("chb09_06.edf", "chb09_01.edf"),
            "chb10": ("chb10_12.edf", "chb10_01.edf"),
        }

        # Ablation conditions
        self.ablation_conditions: Dict[str, Dict] = {
            "conventional": {
                "name": "Condition A: Conventional Univariate (252)",
                "subset": "conventional",
                "n_feats": 252,
            },
            "plv_only": {
                "name": "Condition B: PLV-only Connectivity (112)",
                "subset": "plv_only",
                "n_feats": 112,
            },
            "pli_only": {
                "name": "Condition C: PLI-only Connectivity (88)",
                "subset": "pli_only",
                "n_feats": 88,
            },
            "connectivity": {
                "name": "Condition D: Combined PLV+PLI (200)",
                "subset": "connectivity",
                "n_feats": 200,
            },
            "combined": {
                "name": "Condition E: Full Multimodal (Conventional + PLV + PLI, 452)",
                "subset": "combined",
                "n_feats": 452,
            },
            "conv_plus_plv": {
                "name": "Condition F: Conventional + PLV-only (364)",
                "subset": "conv_plus_plv",
                "n_feats": 364,
            },
            "conv_plus_pli": {
                "name": "Condition G: Conventional + PLI-only (340)",
                "subset": "conv_plus_pli",
                "n_feats": 340,
            },
        }

    def load_or_extract_subject_data(
        self,
        cohort: List[str],
        pairs_dict: Dict[str, Tuple[str, str]],
        verbose: bool = True,
    ) -> Dict[str, Dict]:
        """Extract or load multimodal features for all subjects in cohort."""
        data_by_subject: Dict[str, Dict] = {}

        for subj in cohort:
            seiz_fn, non_seiz_fn = pairs_dict[subj]
            seiz_path = DATASET_DIR / subj / seiz_fn

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

            # 1. Training features (seizure file subsampled + 5 min non-seizure background)
            tr_df = self.cache.extract_or_load(
                case_id=subj,
                file_name=seiz_fn,
                mode="training",
                seizure_intervals=seiz_intervals,
                neg_pos_ratio=10.0,
                random_seed=self.random_state,
                verbose=verbose,
            )
            tr_X_seiz, tr_y_seiz, _ = self.cache.df_to_features_and_labels(tr_df)

            ns_df = self.cache.extract_or_load(
                case_id=subj,
                file_name=non_seiz_fn,
                mode="training_ns",
                seizure_intervals=[],
                stop_sample=300 * SAMPLING_RATE,
                verbose=verbose,
            )
            ns_X, ns_y, _ = self.cache.df_to_features_and_labels(ns_df)

            comb_tr_X = np.concatenate([tr_X_seiz, ns_X], axis=0) if len(ns_X) > 0 else tr_X_seiz
            comb_tr_y = np.concatenate([tr_y_seiz, ns_y], axis=0) if len(ns_y) > 0 else tr_y_seiz

            # 2. Evaluation features (continuous 1.0s stride across seizure recording)
            eval_df = self.cache.extract_or_load(
                case_id=subj,
                file_name=seiz_fn,
                mode="evaluation",
                seizure_intervals=seiz_intervals,
                eval_stride_samples=SAMPLING_RATE,
                verbose=verbose,
            )
            eval_X, eval_y, eval_meta = self.cache.df_to_features_and_labels(eval_df)

            seiz_info = self.loader.inspect_header(seiz_path)
            duration_sec = float(seiz_info["duration_sec"])

            data_by_subject[subj] = {
                "train_X_full": comb_tr_X,  # (N_tr, 452)
                "train_y": comb_tr_y,
                "eval_X_full": eval_X,      # (N_eval, 452)
                "eval_y": eval_y,
                "eval_meta": eval_meta,
                "annotated_seizures_sec": annotated_seiz_sec,
                "duration_sec": duration_sec,
            }

            if verbose:
                n_pos = int(np.sum(comb_tr_y == 1))
                n_neg = int(np.sum(comb_tr_y == 0))
                print(
                    f"  [{subj} Ready] Train: {len(comb_tr_y)} (Ictal={n_pos}, Neg={n_neg}) | "
                    f"Eval Windows: {len(eval_y)} ({duration_sec/3600.0:.2f}h)"
                )

        return data_by_subject

    def evaluate_loso(
        self,
        subject_data: Dict[str, Dict],
        cohort: List[str],
        conditions_dict: Dict[str, Dict],
        calibrate_threshold: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute LOSO cross-validation across all conditions and subjects.

        Args:
            subject_data: Preloaded dictionary of extracted feature matrices
            cohort: List of subject IDs
            conditions_dict: Mapping of condition ID to metadata
            calibrate_threshold: If True, search threshold on training fold only

        Returns:
            Tuple of (detailed_folds_df, aggregate_summary_df)
        """
        all_fold_rows: List[Dict] = []

        for cond_id, cond_info in conditions_dict.items():
            subset_key = cond_info["subset"]
            n_feats = cond_info["n_feats"]

            for held_out_subj in cohort:
                train_subjs = [s for s in cohort if s != held_out_subj]

                # Assemble training set
                raw_X_train = np.concatenate(
                    [subject_data[s]["train_X_full"] for s in train_subjs], axis=0
                )
                y_train = np.concatenate(
                    [subject_data[s]["train_y"] for s in train_subjs], axis=0
                )

                # Assemble test set
                test_subj_data = subject_data[held_out_subj]
                raw_X_test = test_subj_data["eval_X_full"]
                y_test = test_subj_data["eval_y"]
                test_meta = test_subj_data["eval_meta"]

                # Slice to condition's target feature subset
                X_train = self.extractor.slice_features(raw_X_train, subset_key)
                X_test = self.extractor.slice_features(raw_X_test, subset_key)

                # Fit Random Forest (StandardScaler fit strictly on X_train)
                clf = BaselineClassifier(model_type="rf", random_state=self.random_state)
                clf.fit(X_train, y_train)

                # Predict test probabilities
                y_test_prob = clf.predict_proba(X_test)

                # Decision threshold
                if calibrate_threshold:
                    # Leak-free training threshold calibration: search theta in [0.10, 0.90] to maximize training F1
                    y_tr_prob = clf.predict_proba(X_train)
                    best_th = 0.50
                    best_f1 = -1.0
                    for th in np.arange(0.10, 0.90, 0.02):
                        th_f1 = f1_score(y_train, (y_tr_prob >= th).astype(int), zero_division=0)
                        if th_f1 > best_f1:
                            best_f1 = th_f1
                            best_th = float(th)
                    operating_threshold = best_th
                else:
                    operating_threshold = 0.50

                y_test_pred = (y_test_prob >= operating_threshold).astype(int)

                # Metrics
                win_m = compute_window_metrics(
                    y_true=y_test, y_prob=y_test_prob, threshold=operating_threshold
                )
                evt_m = compute_event_metrics(
                    window_preds=y_test_pred,
                    window_start_sec=test_meta["start_sec"].values,
                    annotated_seizures_sec=test_subj_data["annotated_seizures_sec"],
                    total_duration_sec=test_subj_data["duration_sec"],
                    collar_consecutive_windows=3,
                    collar_tolerance_sec=10.0,
                )

                fold_res = {
                    "condition": cond_id,
                    "condition_name": cond_info["name"],
                    "n_features": n_feats,
                    "threshold_mode": "training_calibrated" if calibrate_threshold else "fixed_0.5",
                    "decision_threshold": operating_threshold,
                    "held_out_subject": held_out_subj,
                    "train_samples": int(len(y_train)),
                    "test_windows": int(len(y_test)),
                    "ictal_windows": int(np.sum(y_test == 1)),
                    "prevalence_pct": float(np.mean(y_test == 1) * 100.0),
                    "pr_auc": float(win_m["pr_auc"]),
                    "roc_auc": float(win_m["roc_auc"]),
                    "sensitivity": float(win_m["sensitivity"]),
                    "specificity": float(win_m["specificity"]),
                    "f1": float(win_m["f1"]),
                    "event_sensitivity": float(evt_m["event_sensitivity"]),
                    "detected_seizures": int(evt_m["detected_seizures"]),
                    "total_seizures": int(evt_m["total_seizures"]),
                    "false_alarms": int(evt_m["false_alarms"]),
                    "fp_per_hour": float(evt_m["fp_per_hour"]),
                }
                all_fold_rows.append(fold_res)

        folds_df = pd.DataFrame(all_fold_rows)

        # Aggregate summary
        summary_rows: List[Dict] = []
        for cond_id in conditions_dict:
            sub_df = folds_df[folds_df["condition"] == cond_id]
            summary_rows.append({
                "condition": cond_id,
                "condition_name": conditions_dict[cond_id]["name"],
                "n_features": conditions_dict[cond_id]["n_feats"],
                "threshold_mode": "calibrated" if calibrate_threshold else "fixed_0.5",
                "mean_pr_auc": float(sub_df["pr_auc"].mean()),
                "std_pr_auc": float(sub_df["pr_auc"].std()),
                "mean_roc_auc": float(sub_df["roc_auc"].mean()),
                "mean_window_sens": float(sub_df["sensitivity"].mean()),
                "mean_window_spec": float(sub_df["specificity"].mean()),
                "mean_f1": float(sub_df["f1"].mean()),
                "mean_event_sens": float(sub_df["event_sensitivity"].mean()),
                "total_detected_seizures": int(sub_df["detected_seizures"].sum()),
                "total_annotated_seizures": int(sub_df["total_seizures"].sum()),
                "mean_fp_per_hour": float(sub_df["fp_per_hour"].mean()),
            })

        summary_df = pd.DataFrame(summary_rows)
        return folds_df, summary_df

    def run_feature_importance_analysis(
        self,
        subject_data: Dict[str, Dict],
        cohort: List[str],
    ) -> pd.DataFrame:
        """Train Random Forest across full cohort to inspect feature importance ranking."""
        X_all = np.concatenate([subject_data[s]["train_X_full"] for s in cohort], axis=0)
        y_all = np.concatenate([subject_data[s]["train_y"] for s in cohort], axis=0)

        clf = BaselineClassifier(model_type="rf", random_state=self.random_state)
        clf.fit(X_all, y_all)

        imp_df = clf.get_feature_importances(feature_names=self.extractor.feature_names)

        # Categorize features
        categories = []
        bands = []
        for feat in imp_df["feature"]:
            if "plv" in feat:
                categories.append("PLV Connectivity")
            elif "pli" in feat:
                categories.append("PLI Connectivity")
            else:
                categories.append("Conventional Univariate")

            # Detect band
            if "delta" in feat:
                bands.append("Delta (0.5-4Hz)")
            elif "theta" in feat:
                bands.append("Theta (4-8Hz)")
            elif "alpha" in feat:
                bands.append("Alpha (8-12Hz)")
            elif "beta" in feat:
                bands.append("Beta (12-30Hz)")
            elif "low_gamma" in feat:
                bands.append("Low Gamma (30-40Hz)")
            else:
                bands.append("Time-Domain")

        imp_df["category"] = categories
        imp_df["band"] = bands
        return imp_df

    def generate_figures(
        self,
        summary_fixed_df: pd.DataFrame,
        folds_fixed_df: pd.DataFrame,
        imp_df: pd.DataFrame,
        out_dir: Path,
    ) -> List[Path]:
        """Generate publication-quality charts for the Honors report."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_plots = []

        # 1. Bar Chart: PR-AUC & ROC-AUC across Ablations
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        x = np.arange(len(summary_fixed_df))
        width = 0.35

        ax.bar(x - width/2, summary_fixed_df["mean_pr_auc"], width, label="PR-AUC", color="#1f77b4")
        ax.bar(x + width/2, summary_fixed_df["mean_roc_auc"], width, label="ROC-AUC", color="#2ca02c")

        ax.set_ylabel("Metric Score", fontsize=12)
        ax.set_title("Ablation Study: PR-AUC and ROC-AUC by Feature Set (LOSO 5 Patients)", fontsize=13)
        ax.set_xticks(x)
        labels = [c.replace("Condition ", "").split(" (")[0] for c in summary_fixed_df["condition_name"]]
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(fontsize=11)
        plt.tight_layout()

        p1 = out_dir / "ablation_metrics_comparison.png"
        fig.savefig(p1)
        plt.close(fig)
        saved_plots.append(p1)

        # 2. Per-Subject PR-AUC Bar Chart across Conditions
        fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
        subjects = sorted(folds_fixed_df["held_out_subject"].unique())
        n_subj = len(subjects)
        x = np.arange(n_subj)

        target_conds = ["conventional", "connectivity", "combined"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        w = 0.25

        for i, (cond, col) in enumerate(zip(target_conds, colors)):
            vals = []
            for s in subjects:
                sub = folds_fixed_df[
                    (folds_fixed_df["condition"] == cond) & (folds_fixed_df["held_out_subject"] == s)
                ]
                vals.append(sub["pr_auc"].values[0] if len(sub) > 0 else 0.0)
            ax.bar(x + (i - 1) * w, vals, w, label=cond.capitalize(), color=col)

        ax.set_ylabel("PR-AUC", fontsize=12)
        ax.set_title("Per-Subject Generalization: Conventional vs Connectivity vs Combined", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(subjects, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(fontsize=11)
        plt.tight_layout()

        p2 = out_dir / "per_subject_pr_auc_breakdown.png"
        fig.savefig(p2)
        plt.close(fig)
        saved_plots.append(p2)

        # 3. Feature Importance Top 20
        fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
        top20 = imp_df.head(20).iloc[::-1]

        category_colors = {
            "Conventional Univariate": "#1f77b4",
            "PLV Connectivity": "#ff7f0e",
            "PLI Connectivity": "#9467bd",
        }
        bar_colors = [category_colors.get(c, "#7f7f7f") for c in top20["category"]]

        ax.barh(top20["feature"], top20["importance"], color=bar_colors)
        ax.set_xlabel("Gini Feature Importance", fontsize=12)
        ax.set_title("Top 20 Electrophysiological Features in Combined Multimodal Model", fontsize=13)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=c, label=l) for l, c in category_colors.items()
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
        plt.tight_layout()

        p3 = out_dir / "top_20_feature_importances.png"
        fig.savefig(p3)
        plt.close(fig)
        saved_plots.append(p3)

        return saved_plots
