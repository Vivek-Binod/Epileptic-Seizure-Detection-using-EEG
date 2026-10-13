"""Temporal post-processing experiment pipeline.

Evaluates causal EWMA smoothing vs unsmoothed predictions under strict
nested Leave-One-Subject-Out (LOSO) cross-validation with zero test leakage.
"""

from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_event_metrics, compute_window_metrics
from src.evaluation.postprocessing import ewma_smooth, tune_ewma_alpha_leak_free
from src.features.cache import FeatureCache
from src.features.unified import UnifiedFeatureExtractor
from src.models.baseline import BaselineClassifier
from src.utils.config import OUTPUTS_DIR, PROTOTYPE_COHORT, SAMPLING_RATE


class PostprocessingExperiment:
    """Bounded temporal post-processing evaluation runner."""

    def __init__(
        self,
        experiment_id: str = "exp_20261013_postprocessing",
        cache_experiment_id: str = "exp_20261012_sprint",
        random_state: int = 42,
    ):
        self.experiment_id = experiment_id
        self.random_state = random_state
        self.exp_dir = OUTPUTS_DIR / "experiments" / experiment_id
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = OUTPUTS_DIR / "experiments" / cache_experiment_id / "feature_cache"
        self.extractor = UnifiedFeatureExtractor(sampling_rate=SAMPLING_RATE)
        self.cache = FeatureCache(cache_dir=self.cache_dir, extractor=self.extractor)

        self.prototype_cohort = ["chb01", "chb03", "chb05", "chb07", "chb08"]
        self.expansion_cohort = ["chb02", "chb04", "chb06", "chb09", "chb10"]
        self.full_10_cohort = self.prototype_cohort + self.expansion_cohort

        self.subject_pairs = {
            "chb01": ("chb01_03.edf", "chb01_01.edf"),
            "chb03": ("chb03_01.edf", "chb03_05.edf"),
            "chb05": ("chb05_06.edf", "chb05_01.edf"),
            "chb07": ("chb07_19.edf", "chb07_01.edf"),
            "chb08": ("chb08_02.edf", "chb08_03.edf"),
            "chb02": ("chb02_16.edf", "chb02_01.edf"),
            "chb04": ("chb04_05.edf", "chb04_01.edf"),
            "chb06": ("chb06_01.edf", "chb06_02.edf"),
            "chb09": ("chb09_06.edf", "chb09_01.edf"),
            "chb10": ("chb10_12.edf", "chb10_01.edf"),
        }

        # Metadata on annotated seizures for event metrics
        from src.data.metadata_parser import CHBMITMetadataParser
        parser = CHBMITMetadataParser()
        _, self.seiz_df = parser.parse_all()

    def load_cohort_data(self, cohort: List[str]) -> Dict[str, Dict]:
        """Load cached Parquet features and evaluation arrays for specified cohort."""
        data_by_subject: Dict[str, Dict] = {}

        for subj in cohort:
            seiz_fn, non_seiz_fn = self.subject_pairs[subj]
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

            # 1. Training data
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
                file_name=non_seiz_fn,
                mode="training_ns",
                seizure_intervals=[],
                stop_sample=300 * SAMPLING_RATE,
                verbose=False,
            )
            ns_X, ns_y, _ = self.cache.df_to_features_and_labels(ns_df)

            comb_tr_X = np.concatenate([tr_X_seiz, ns_X], axis=0) if len(ns_X) > 0 else tr_X_seiz
            comb_tr_y = np.concatenate([tr_y_seiz, ns_y], axis=0) if len(ns_y) > 0 else tr_y_seiz

            # 2. Evaluation continuous data
            eval_df = self.cache.extract_or_load(
                case_id=subj,
                file_name=seiz_fn,
                mode="evaluation",
                seizure_intervals=seiz_intervals,
                eval_stride_samples=SAMPLING_RATE,
                verbose=False,
            )
            eval_X, eval_y, eval_meta = self.cache.df_to_features_and_labels(eval_df)
            duration_sec = float(len(eval_y))  # continuous 1.0s stride

            data_by_subject[subj] = {
                "train_X_full": comb_tr_X,
                "train_y": comb_tr_y,
                "eval_X_full": eval_X,
                "eval_y": eval_y,
                "eval_meta": eval_meta,
                "annotated_seizures_sec": annotated_seiz_sec,
                "duration_sec": duration_sec,
            }

        return data_by_subject

    def run_nested_loso_evaluation(
        self,
        cohort: List[str],
        subject_data: Dict[str, Dict],
        candidate_alphas: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
        operating_threshold: float = 0.50,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute nested LOSO cross-validation with training-only EWMA alpha tuning."""
        detailed_rows: List[Dict] = []

        print(f"\n--- Running Nested LOSO on Cohort ({len(cohort)} subjects: {cohort}) ---")

        for held_out_subj in cohort:
            train_cohort = [s for s in cohort if s != held_out_subj]

            # =================================================================
            # 1. Inner Cross-Validation across Training Cohort for Leak-Free Tuning
            # =================================================================
            inner_val_data: Dict[str, Dict] = {}
            for inner_val in train_cohort:
                inner_tr_subjs = [s for s in train_cohort if s != inner_val]

                raw_X_inner_tr = np.concatenate(
                    [subject_data[s]["train_X_full"] for s in inner_tr_subjs], axis=0
                )
                y_inner_tr = np.concatenate(
                    [subject_data[s]["train_y"] for s in inner_tr_subjs], axis=0
                )
                raw_X_inner_val = subject_data[inner_val]["eval_X_full"]

                # Combined features (452 dims)
                X_inner_tr = self.extractor.slice_features(raw_X_inner_tr, "combined")
                X_inner_val = self.extractor.slice_features(raw_X_inner_val, "combined")

                clf_inner = BaselineClassifier(model_type="rf", random_state=self.random_state)
                clf_inner.fit(X_inner_tr, y_inner_tr)
                p_inner_val = clf_inner.predict_proba(X_inner_val)

                inner_val_data[inner_val] = {
                    "y_true": subject_data[inner_val]["eval_y"],
                    "raw_prob": p_inner_val,
                }

            best_alpha, alpha_cv_scores = tune_ewma_alpha_leak_free(
                inner_val_data=inner_val_data,
                candidate_alphas=candidate_alphas,
                criterion="pr_auc",
            )

            # =================================================================
            # 2. Outer Model Fit on Full Training Cohort
            # =================================================================
            raw_X_train = np.concatenate(
                [subject_data[s]["train_X_full"] for s in train_cohort], axis=0
            )
            y_train = np.concatenate(
                [subject_data[s]["train_y"] for s in train_cohort], axis=0
            )
            raw_X_test = subject_data[held_out_subj]["eval_X_full"]
            y_test = subject_data[held_out_subj]["eval_y"]
            test_meta = subject_data[held_out_subj]["eval_meta"]
            annotated_seiz = subject_data[held_out_subj]["annotated_seizures_sec"]
            duration_sec = subject_data[held_out_subj]["duration_sec"]

            X_train = self.extractor.slice_features(raw_X_train, "combined")
            X_test = self.extractor.slice_features(raw_X_test, "combined")

            clf_outer = BaselineClassifier(model_type="rf", random_state=self.random_state)
            clf_outer.fit(X_train, y_train)
            raw_test_prob = clf_outer.predict_proba(X_test)

            # Apply leak-free selected EWMA
            smooth_test_prob = ewma_smooth(raw_test_prob, alpha=best_alpha)

            # =================================================================
            # 3. Evaluate Unsmoothed and Smoothed on Identical Windows
            # =================================================================
            for mode, p_series, alpha_val in [
                ("unsmoothed", raw_test_prob, 1.0),
                ("smoothed_ewma", smooth_test_prob, best_alpha),
            ]:
                win_m = compute_window_metrics(y_true=y_test, y_prob=p_series, threshold=operating_threshold)
                y_bin_pred = (p_series >= operating_threshold).astype(int)

                evt_m = compute_event_metrics(
                    window_preds=y_bin_pred,
                    window_start_sec=test_meta["start_sec"].values,
                    annotated_seizures_sec=annotated_seiz,
                    total_duration_sec=duration_sec,
                    collar_consecutive_windows=3,
                    collar_tolerance_sec=10.0,
                )

                row = {
                    "cohort_size": len(cohort),
                    "held_out_subject": held_out_subj,
                    "mode": mode,
                    "selected_alpha": alpha_val,
                    "operating_threshold": operating_threshold,
                    "test_windows": len(y_test),
                    "eval_duration_hr": duration_sec / 3600.0,
                    "total_seizures": evt_m["total_seizures"],
                    "detected_seizures": evt_m["detected_seizures"],
                    "event_sensitivity": evt_m["event_sensitivity"],
                    "false_alarms": evt_m["false_alarms"],
                    "fp_per_hour": evt_m["fp_per_hour"],
                    "detection_delays_sec": str(evt_m.get("detection_delays_sec", [])),
                    "mean_delay_sec": evt_m.get("mean_delay_sec", float("nan")),
                    "pr_auc": win_m["pr_auc"],
                    "roc_auc": win_m["roc_auc"],
                    "window_sens": win_m["sensitivity"],
                    "window_spec": win_m["specificity"],
                    "window_f1": win_m["f1"],
                    "prevalence_pct": float(np.mean(y_test == 1) * 100.0),
                }
                detailed_rows.append(row)

            print(
                f"  [{held_out_subj}] Selected alpha={best_alpha:.2f} | "
                f"Raw PR={detailed_rows[-2]['pr_auc']:.4f}, FP/h={detailed_rows[-2]['fp_per_hour']:.2f}, Sens={detailed_rows[-2]['detected_seizures']}/{detailed_rows[-2]['total_seizures']} | "
                f"Smooth PR={detailed_rows[-1]['pr_auc']:.4f}, FP/h={detailed_rows[-1]['fp_per_hour']:.2f}, Sens={detailed_rows[-1]['detected_seizures']}/{detailed_rows[-1]['total_seizures']}"
            )

        folds_df = pd.DataFrame(detailed_rows)

        # Aggregate summary
        summary_rows: List[Dict] = []
        for mode in ["unsmoothed", "smoothed_ewma"]:
            sub = folds_df[folds_df["mode"] == mode]
            delays = []
            for d_str in sub["detection_delays_sec"]:
                try:
                    d_list = eval(d_str)
                    delays.extend(d_list)
                except Exception:
                    pass

            summary_rows.append({
                "cohort_size": len(cohort),
                "mode": mode,
                "mean_pr_auc": float(sub["pr_auc"].mean()),
                "std_pr_auc": float(sub["pr_auc"].std()),
                "mean_roc_auc": float(sub["roc_auc"].mean()),
                "mean_window_sens": float(sub["window_sens"].mean()),
                "mean_window_spec": float(sub["window_spec"].mean()),
                "mean_window_f1": float(sub["window_f1"].mean()),
                "mean_event_sens": float(sub["event_sensitivity"].mean()),
                "total_detected_seizures": int(sub["detected_seizures"].sum()),
                "total_annotated_seizures": int(sub["total_seizures"].sum()),
                "total_false_alarms": int(sub["false_alarms"].sum()),
                "mean_fp_per_hour": float(sub["fp_per_hour"].mean()),
                "total_eval_hours": float(sub["eval_duration_hr"].sum()),
                "overall_fp_per_hour": float(sub["false_alarms"].sum() / sub["eval_duration_hr"].sum()),
                "mean_detection_delay_sec": float(np.mean(delays)) if len(delays) > 0 else float("nan"),
                "median_detection_delay_sec": float(np.median(delays)) if len(delays) > 0 else float("nan"),
            })

        summary_df = pd.DataFrame(summary_rows)
        return folds_df, summary_df

    def generate_figures(
        self,
        folds_df: pd.DataFrame,
        cohort_name: str,
        out_dir: Path,
    ) -> List[Path]:
        """Generate high-resolution comparison charts."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_plots = []

        sub_raw = folds_df[folds_df["mode"] == "unsmoothed"].set_index("held_out_subject")
        sub_sm = folds_df[folds_df["mode"] == "smoothed_ewma"].set_index("held_out_subject")
        subjects = list(sub_raw.index)
        x = np.arange(len(subjects))
        w = 0.35

        # 1. PR-AUC Comparison Plot
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        ax.bar(x - w / 2, sub_raw["pr_auc"], w, label="Unsmoothed Combined", color="#1f77b4")
        ax.bar(x + w / 2, sub_sm["pr_auc"], w, label="Smoothed EWMA (Leak-Free Tuned)", color="#2ca02c")
        ax.set_ylabel("PR-AUC", fontsize=12)
        ax.set_title(f"PR-AUC: Unsmoothed vs Smoothed EWMA ({cohort_name})", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(subjects, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(fontsize=11)
        plt.tight_layout()
        p1 = out_dir / f"pr_auc_smoothing_{cohort_name.lower().replace(' ', '_')}.png"
        fig.savefig(p1)
        plt.close(fig)
        saved_plots.append(p1)

        # 2. False Alarms / Hour Comparison Plot
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        ax.bar(x - w / 2, sub_raw["fp_per_hour"], w, label="Unsmoothed Combined", color="#d62728")
        ax.bar(x + w / 2, sub_sm["fp_per_hour"], w, label="Smoothed EWMA", color="#ff7f0e")
        ax.set_ylabel("False Alarms / Hour (FP/h)", fontsize=12)
        ax.set_title(f"False Alarm Rate: Unsmoothed vs Smoothed EWMA ({cohort_name})", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(subjects, fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(fontsize=11)
        plt.tight_layout()
        p2 = out_dir / f"false_alarms_{cohort_name.lower().replace(' ', '_')}.png"
        fig.savefig(p2)
        plt.close(fig)
        saved_plots.append(p2)

        return saved_plots
