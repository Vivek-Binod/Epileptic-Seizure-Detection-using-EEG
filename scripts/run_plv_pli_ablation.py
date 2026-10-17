"""Full-Cohort PLV-only vs PLI-only Connectivity Feature Ablation.

Evaluates:
1. Conventional Baseline (252)
2. Connectivity Baseline (PLV + PLI, 200)
3. Combined Multimodal (452)
4. PLV-only Connectivity (112)
5. PLI-only Connectivity (88)
6. Conventional + PLI (340)
7. Conventional + PLV (364)

Under the strict 23-fold Leave-One-Patient-Out (LOPO) cross-validation protocol
across all 24 cases (35.99 hours continuous evaluation EEG, 28 clinical seizures,
23 independent biological patients, chb01 and chb21 unified under patient_01).

Zero recomputation of features: reuses existing Parquet cache at
outputs/experiments/exp_20261012_sprint/feature_cache/fp_2dd1bac799f4/.
"""

import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.metadata_parser import CHBMITMetadataParser
from src.evaluation.metrics import compute_event_metrics, compute_window_metrics
from src.evaluation.postprocessing import ewma_smooth
from src.features.cache import FeatureCache
from src.features.unified import UnifiedFeatureExtractor
from src.models.baseline import BaselineClassifier
from src.pipelines.full_cohort_experiment import SUBJECT_PAIRS
from src.utils.config import OUTPUTS_DIR, PATIENT_ID_MAPPING, SAMPLING_RATE


class FullCohortPLVAblationExperiment:
    """Executes full-cohort PLV vs PLI ablation across 23 biological patient folds."""

    def __init__(
        self,
        experiment_id: str = "exp_20261015_plv_pli_ablation",
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

        self.subject_pairs = SUBJECT_PAIRS
        self.patient_mapping = PATIENT_ID_MAPPING
        self.cases = sorted(list(self.subject_pairs.keys()))

        parser = CHBMITMetadataParser()
        _, self.seiz_df = parser.parse_all()

    def load_cohort_data(self) -> Dict[str, Dict]:
        """Load Parquet feature arrays for all 24 cases from disk cache."""
        data_by_case: Dict[str, Dict] = {}
        t0 = time.time()
        print(f"Loading cached feature data across all {len(self.cases)} cases...")

        for case_id in self.cases:
            seiz_fn, non_seiz_fn = self.subject_pairs[case_id]
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

            # 1. Training features
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

            # 2. Non-seizure baseline
            ns_df = self.cache.extract_or_load(
                case_id=case_id,
                file_name=non_seiz_fn,
                mode="training_ns",
                seizure_intervals=[],
                stop_sample=300 * SAMPLING_RATE,
                verbose=False,
            )
            ns_X, ns_y, _ = self.cache.df_to_features_and_labels(ns_df)

            comb_tr_X = np.concatenate([tr_X_seiz, ns_X], axis=0) if len(ns_X) > 0 else tr_X_seiz
            comb_tr_y = np.concatenate([tr_y_seiz, ns_y], axis=0) if len(ns_y) > 0 else tr_y_seiz

            # 3. Continuous evaluation features
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
                "train_X_full": comb_tr_X,
                "train_y": comb_tr_y,
                "eval_X_full": eval_X,
                "eval_y": eval_y,
                "eval_meta": eval_meta,
                "annotated_seizures_sec": annotated_seiz_sec,
                "duration_sec": duration_sec,
            }

        t1 = time.time()
        print(f"Loaded all 24 cases in {t1 - t0:.2f} s.")
        return data_by_case

    def run_ablation(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute full 23-fold LOPO cross-validation across all ablation conditions."""
        data = self.load_cohort_data()
        unique_patients = sorted(list(set(self.patient_mapping[c] for c in self.cases)))

        # Define all 7 experimental feature configurations
        conditions = [
            ("conventional", "Conventional Baseline", "conventional", 252),
            ("connectivity", "Connectivity (PLV + PLI)", "connectivity", 200),
            ("combined", "Combined Multimodal", "combined", 452),
            ("plv_only", "PLV-only Connectivity", "plv_only", 112),
            ("pli_only", "PLI-only Connectivity", "pli_only", 88),
            ("conv_plus_pli", "Conventional + PLI-only", "conv_plus_pli", 340),
            ("conv_plus_plv", "Conventional + PLV-only", "conv_plus_plv", 364),
        ]

        all_case_rows: List[Dict] = []
        all_patient_rows: List[Dict] = []

        print("\n" + "=" * 80)
        print("RUNNING 23-FOLD LEAVE-ONE-PATIENT-OUT PLV vs PLI ABLATION")
        print(f"Conditions: {[c[0] for c in conditions]}")
        print("=" * 80)

        t0_total = time.time()

        for fold_idx, held_out_patient in enumerate(unique_patients, start=1):
            t0_fold = time.time()
            test_cases = [c for c in self.cases if self.patient_mapping[c] == held_out_patient]
            train_cases = [c for c in self.cases if self.patient_mapping[c] != held_out_patient]

            print(
                f"[Fold {fold_idx:02d}/23] Held-Out: {held_out_patient} (Cases: {test_cases}) | "
                f"Training on {len(train_cases)} cases...",
                flush=True
            )

            # Assemble training data
            raw_X_train = np.concatenate([data[c]["train_X_full"] for c in train_cases], axis=0)
            y_train = np.concatenate([data[c]["train_y"] for c in train_cases], axis=0)

            # Fit one Random Forest model per condition
            fitted_models = {}
            for cond_id, cond_name, subset_key, n_f in conditions:
                X_tr_slice = self.extractor.slice_features(raw_X_train, subset_key)
                clf = BaselineClassifier(model_type="rf", random_state=self.random_state)
                clf.fit(X_tr_slice, y_train)
                fitted_models[cond_id] = clf

            # Evaluate each test case
            for test_case in test_cases:
                eval_X = data[test_case]["eval_X_full"]
                eval_y = data[test_case]["eval_y"]
                meta = data[test_case]["eval_meta"]
                annot_seiz = data[test_case]["annotated_seizures_sec"]
                dur_sec = data[test_case]["duration_sec"]
                dur_hours = dur_sec / 3600.0

                for cond_id, cond_name, subset_key, n_f in conditions:
                    clf = fitted_models[cond_id]
                    X_te_slice = self.extractor.slice_features(eval_X, subset_key)
                    p_raw = clf.predict_proba(X_te_slice)
                    p_ewma = ewma_smooth(p_raw, alpha=0.30)

                    # 1. Unsmoothed metrics
                    w_raw = compute_window_metrics(y_true=eval_y, y_prob=p_raw, threshold=0.50)
                    e_raw = compute_event_metrics(
                        window_preds=(p_raw >= 0.50).astype(int),
                        window_start_sec=meta["start_sec"].values,
                        annotated_seizures_sec=annot_seiz,
                        total_duration_sec=dur_sec,
                        collar_consecutive_windows=3,
                        collar_tolerance_sec=10.0,
                    )

                    all_case_rows.append({
                        "fold_idx": fold_idx,
                        "held_out_patient": held_out_patient,
                        "case_id": test_case,
                        "config_id": cond_id,
                        "config_name": cond_name,
                        "feature_subset": subset_key,
                        "feature_count": n_f,
                        "postprocessing": "unsmoothed",
                        "pr_auc": w_raw["pr_auc"],
                        "roc_auc": w_raw["roc_auc"],
                        "window_sens": w_raw["sensitivity"],
                        "window_spec": w_raw["specificity"],
                        "window_f1": w_raw["f1"],
                        "event_sensitivity": e_raw["event_sensitivity"],
                        "detected_seizures": e_raw["detected_seizures"],
                        "total_seizures": e_raw["total_seizures"],
                        "false_alarms": e_raw["false_alarms"],
                        "fp_per_hour": e_raw["fp_per_hour"],
                        "duration_hours": dur_hours,
                        "mean_delay_sec": e_raw.get("mean_delay_sec", float("nan")),
                        "median_delay_sec": e_raw.get("median_delay_sec", float("nan")),
                    })

                    # 2. EWMA (alpha=0.30) metrics
                    w_ewma = compute_window_metrics(y_true=eval_y, y_prob=p_ewma, threshold=0.50)
                    e_ewma = compute_event_metrics(
                        window_preds=(p_ewma >= 0.50).astype(int),
                        window_start_sec=meta["start_sec"].values,
                        annotated_seizures_sec=annot_seiz,
                        total_duration_sec=dur_sec,
                        collar_consecutive_windows=3,
                        collar_tolerance_sec=10.0,
                    )

                    all_case_rows.append({
                        "fold_idx": fold_idx,
                        "held_out_patient": held_out_patient,
                        "case_id": test_case,
                        "config_id": f"{cond_id}_ewma",
                        "config_name": f"{cond_name} + EWMA (alpha=0.30)",
                        "feature_subset": subset_key,
                        "feature_count": n_f,
                        "postprocessing": "ewma_0.30",
                        "pr_auc": w_ewma["pr_auc"],
                        "roc_auc": w_ewma["roc_auc"],
                        "window_sens": w_ewma["sensitivity"],
                        "window_spec": w_ewma["specificity"],
                        "window_f1": w_ewma["f1"],
                        "event_sensitivity": e_ewma["event_sensitivity"],
                        "detected_seizures": e_ewma["detected_seizures"],
                        "total_seizures": e_ewma["total_seizures"],
                        "false_alarms": e_ewma["false_alarms"],
                        "fp_per_hour": e_ewma["fp_per_hour"],
                        "duration_hours": dur_hours,
                        "mean_delay_sec": e_ewma.get("mean_delay_sec", float("nan")),
                        "median_delay_sec": e_ewma.get("median_delay_sec", float("nan")),
                    })

            print(f"  Completed fold {fold_idx} in {time.time() - t0_fold:.1f} s.", flush=True)

        case_df = pd.DataFrame(all_case_rows)
        case_csv = self.exp_dir / "plv_pli_ablation_case_folds.csv"
        case_df.to_csv(case_csv, index=False)

        # Patient-Level Aggregations (23 independent biological patients)
        for cfg_id in case_df["config_id"].unique():
            sub_cfg = case_df[case_df["config_id"] == cfg_id]
            for p_id in unique_patients:
                p_sub = sub_cfg[sub_cfg["held_out_patient"] == p_id]
                delays = p_sub["mean_delay_sec"].dropna().values

                all_patient_rows.append({
                    "patient_id": p_id,
                    "cases": "+".join(p_sub["case_id"].unique()),
                    "config_id": cfg_id,
                    "config_name": p_sub["config_name"].iloc[0],
                    "feature_subset": p_sub["feature_subset"].iloc[0],
                    "feature_count": p_sub["feature_count"].iloc[0],
                    "postprocessing": p_sub["postprocessing"].iloc[0],
                    "pr_auc": float(p_sub["pr_auc"].mean()),
                    "roc_auc": float(p_sub["roc_auc"].mean()),
                    "window_sens": float(p_sub["window_sens"].mean()),
                    "window_spec": float(p_sub["window_spec"].mean()),
                    "window_f1": float(p_sub["window_f1"].mean()),
                    "event_sensitivity": float(p_sub["event_sensitivity"].mean()),
                    "detected_seizures": int(p_sub["detected_seizures"].sum()),
                    "total_seizures": int(p_sub["total_seizures"].sum()),
                    "false_alarms": int(p_sub["false_alarms"].sum()),
                    "total_hours": float(p_sub["duration_hours"].sum()),
                    "fp_per_hour": float(p_sub["false_alarms"].sum() / p_sub["duration_hours"].sum()),
                    "mean_delay_sec": float(np.mean(delays)) if len(delays) > 0 else float("nan"),
                })

        patient_df = pd.DataFrame(all_patient_rows)
        patient_csv = self.exp_dir / "plv_pli_ablation_patient_folds.csv"
        patient_df.to_csv(patient_csv, index=False)

        # Summary Table (Macro Case, Macro Patient, and Pooled Cohort)
        summary_rows: List[Dict] = []
        for cfg_id in case_df["config_id"].unique():
            sub_c = case_df[case_df["config_id"] == cfg_id]
            sub_p = patient_df[patient_df["config_id"] == cfg_id]
            delays_c = sub_c["mean_delay_sec"].dropna().values
            medians_c = sub_c["median_delay_sec"].dropna().values

            total_hours = float(sub_c["duration_hours"].sum())
            total_fa = int(sub_c["false_alarms"].sum())
            total_det = int(sub_c["detected_seizures"].sum())
            total_sz = int(sub_c["total_seizures"].sum())

            summary_rows.append({
                "config_id": cfg_id,
                "config_name": sub_c["config_name"].iloc[0],
                "feature_subset": sub_c["feature_subset"].iloc[0],
                "feature_count": int(sub_c["feature_count"].iloc[0]),
                "postprocessing": sub_c["postprocessing"].iloc[0],
                # Macro Case metrics (24 cases)
                "macro_case_pr_auc": float(sub_c["pr_auc"].mean()),
                "std_case_pr_auc": float(sub_c["pr_auc"].std()),
                "macro_case_roc_auc": float(sub_c["roc_auc"].mean()),
                "macro_case_event_sens": float(sub_c["event_sensitivity"].mean()),
                "macro_case_fp_per_hour": float(sub_c["fp_per_hour"].mean()),
                # Macro Patient metrics (23 patients)
                "macro_patient_pr_auc": float(sub_p["pr_auc"].mean()),
                "std_patient_pr_auc": float(sub_p["pr_auc"].std()),
                "macro_patient_roc_auc": float(sub_p["roc_auc"].mean()),
                "macro_patient_event_sens": float(sub_p["event_sensitivity"].mean()),
                "macro_patient_fp_per_hour": float(sub_p["fp_per_hour"].mean()),
                # Pooled Cohort metrics
                "pooled_detected_seizures": total_det,
                "pooled_total_seizures": total_sz,
                "pooled_event_sensitivity": float(total_det / total_sz) if total_sz > 0 else 0.0,
                "pooled_false_alarms": total_fa,
                "total_evaluation_hours": total_hours,
                "pooled_fp_per_hour": float(total_fa / total_hours) if total_hours > 0 else 0.0,
                "mean_detection_delay_sec": float(np.mean(delays_c)) if len(delays_c) > 0 else float("nan"),
                "median_detection_delay_sec": float(np.median(medians_c)) if len(medians_c) > 0 else float("nan"),
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = self.exp_dir / "plv_pli_ablation_summary.csv"
        summary_df.to_csv(summary_csv, index=False)

        print("\n" + "=" * 80)
        print(f"ABLATION COMPLETE IN {time.time() - t0_total:.1f} s. Outputs saved to {self.exp_dir}")
        print("=" * 80)
        return case_df, patient_df, summary_df


if __name__ == "__main__":
    exp = FullCohortPLVAblationExperiment()
    case_df, patient_df, summary_df = exp.run_ablation()
