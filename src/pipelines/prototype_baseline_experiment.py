"""Prototype cohort baseline experiment pipeline.

Executes a Leave-One-Subject-Out (LOSO) cross-validation across the 5 prototype patients
(chb01, chb03, chb05, chb07, chb08) using classical models (RandomForest and LightGBM).

Enforces:
1. Zero Patient Leakage: Scaler and classifier fit strictly on N-1 subjects, evaluated on unseen held-out subject.
2. Dual Evaluation: Window-level PR-AUC/ROC-AUC and Event-Level collar alarm sensitivity & FP/h.
3. Clean reporting: Saves detailed JSON and CSV summaries to outputs/.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import EDFLoader
from src.data.metadata_parser import CHBMITMetadataParser
from src.data.windowing import WindowGenerator
from src.evaluation.metrics import compute_event_metrics, compute_window_metrics
from src.features.cache import FeatureCache
from src.features.feature_extractor import EEGFeatureExtractor
from src.models.baseline import BaselineClassifier
from src.utils.config import DATASET_DIR, OUTPUTS_DIR, PROTOTYPE_COHORT, SAMPLING_RATE


def run_prototype_baseline_experiment(
    model_type: str = "rf",
    use_cache: bool = True,
    verbose_cache: bool = False,
    output_dir: Path = OUTPUTS_DIR,
    save_outputs: bool = True,
) -> Dict:
    """Run prototype LOSO cross-validation across the 5 prototype patients.

    Args:
        model_type: 'rf' (Random Forest) or 'lgbm' (LightGBM)
        use_cache: Whether to use on-disk FeatureCache for fast reproducible runs
        verbose_cache: Whether to print cache hit/miss status for each recording
        output_dir: Directory to save summary files
        save_outputs: Whether to write JSON/CSV reports

    Returns:
        Dictionary of results across all folds.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(f"RUNNING PROTOTYPE LOSO BASELINE EXPERIMENT (Model: {model_type.upper()})")
    print(f"Cohort: {PROTOTYPE_COHORT} (5 prototype patients)")
    print("=" * 75)

    # 1. Manifests & Initializers
    parser = CHBMITMetadataParser()
    recs_df, seiz_df = parser.parse_all()

    loader = EDFLoader()
    window_gen = WindowGenerator()
    extractor = EEGFeatureExtractor()

    # Benchmark test pairs per subject: (seizure_file, non_seizure_file)
    test_pairs: Dict[str, Tuple[str, str]] = {
        "chb01": ("chb01_03.edf", "chb01_01.edf"),
        "chb03": ("chb03_01.edf", "chb03_05.edf"),
        "chb05": ("chb05_06.edf", "chb05_01.edf"),
        "chb07": ("chb07_19.edf", "chb07_01.edf"),
        "chb08": ("chb08_02.edf", "chb08_03.edf"),
    }

    # 2. Pre-extract features per subject to make LOSO fast & modular
    subject_data: Dict[str, Dict] = {}
    feature_cache = FeatureCache() if use_cache else None

    print("\n[Step 1/3] Extracting/loading features for prototype subject recordings...")
    for subj in PROTOTYPE_COHORT:
        seiz_fn, non_seiz_fn = test_pairs[subj]
        seiz_path = DATASET_DIR / subj / seiz_fn
        non_seiz_path = DATASET_DIR / subj / non_seiz_fn

        # Annotations
        subj_seizures = seiz_df[(seiz_df["case_id"] == subj) & (seiz_df["file_name"] == seiz_fn)]
        seiz_intervals = [
            (row["start_sample"], row["end_sample"]) for _, row in subj_seizures.iterrows()
        ]
        annotated_seiz_sec = [
            (float(row["start_sec"]), float(row["end_sec"])) for _, row in subj_seizures.iterrows()
        ]

        if use_cache and feature_cache is not None:
            # Training windows: seizure recording
            tr_df = feature_cache.extract_or_load(
                case_id=subj,
                file_name=seiz_fn,
                mode="training",
                seizure_intervals=seiz_intervals,
                neg_pos_ratio=10.0,
                random_seed=42,
                verbose=verbose_cache,
            )
            tr_feats, tr_y, _ = feature_cache.df_to_features_and_labels(tr_df)

            # Non-seizure recording: load 5 minutes (300s = 75 negative windows)
            ns_df = feature_cache.extract_or_load(
                case_id=subj,
                file_name=non_seiz_fn,
                mode="training_ns",
                seizure_intervals=[],
                stop_sample=300 * SAMPLING_RATE,
                verbose=verbose_cache,
            )
            ns_feats, ns_y, _ = feature_cache.df_to_features_and_labels(ns_df)

            comb_feats = np.concatenate([tr_feats, ns_feats], axis=0) if len(ns_feats) > 0 else tr_feats
            comb_y = np.concatenate([tr_y, ns_y], axis=0) if len(ns_y) > 0 else tr_y

            # Evaluation windows: test on continuous seizure recording (sequential sliding)
            eval_df = feature_cache.extract_or_load(
                case_id=subj,
                file_name=seiz_fn,
                mode="evaluation",
                seizure_intervals=seiz_intervals,
                eval_stride_samples=SAMPLING_RATE,
                verbose=verbose_cache,
            )
            eval_feats, eval_y, eval_meta = feature_cache.df_to_features_and_labels(eval_df)
            seiz_info = loader.inspect_header(seiz_path)
            duration_sec = float(seiz_info["duration_sec"])
        else:
            # Load seizure recording full signal
            seiz_signal, seiz_info = loader.load_full_recording(seiz_path)
            # Training windows: seizure recording (with fine stride on ictal, coarse on background)
            tr_win, tr_y, tr_meta = window_gen.extract_windows_for_recording(
                signal=seiz_signal,
                seizure_intervals=seiz_intervals,
                is_training=True,
                neg_pos_ratio=10.0,
                random_seed=42,
            )

            # Non-seizure recording: load 5 minutes (300s = 75 negative windows)
            ns_slice = loader.load_slice(
                non_seiz_path,
                start_sample=0,
                stop_sample=300 * SAMPLING_RATE,
                apply_filter=True,
                apply_standardization=True,
            )
            ns_win, ns_y, ns_meta = window_gen.extract_windows_for_recording(
                signal=ns_slice,
                seizure_intervals=[],
                is_training=True,
            )

            # Combine training data for this subject
            comb_win = np.concatenate([tr_win, ns_win], axis=0) if len(ns_win) > 0 else tr_win
            comb_y = np.concatenate([tr_y, ns_y], axis=0) if len(ns_y) > 0 else tr_y
            comb_feats = extractor.extract_features(comb_win)

            # Evaluation windows: test on continuous seizure recording (sequential sliding)
            eval_win, eval_y, eval_meta = window_gen.extract_windows_for_recording(
                signal=seiz_signal,
                seizure_intervals=seiz_intervals,
                is_training=False,
                eval_stride_samples=SAMPLING_RATE,  # 1.0s stride for continuous event collar
            )
            eval_feats = extractor.extract_features(eval_win)
            duration_sec = float(seiz_info["duration_sec"])

        subject_data[subj] = {
            "train_X": comb_feats,
            "train_y": comb_y,
            "eval_X": eval_feats,
            "eval_y": eval_y,
            "eval_meta": eval_meta,
            "annotated_seizures_sec": annotated_seiz_sec,
            "duration_sec": duration_sec,
        }
        print(
            f"  Subj {subj}: Train samples={len(comb_y)} (Pos={np.sum(comb_y == 1)}, Neg={np.sum(comb_y == 0)}) | "
            f"Eval windows={len(eval_y)}"
        )

    # 3. Execute Leave-One-Subject-Out Cross-Validation
    print(f"\n[Step 2/3] Executing 5-Fold LOSO Cross-Validation ({model_type.upper()})...")
    loso_results: List[Dict] = []

    for held_out_subj in PROTOTYPE_COHORT:
        # Build training set from the other 4 subjects
        train_subjs = [s for s in PROTOTYPE_COHORT if s != held_out_subj]
        X_train = np.concatenate([subject_data[s]["train_X"] for s in train_subjs], axis=0)
        y_train = np.concatenate([subject_data[s]["train_y"] for s in train_subjs], axis=0)

        # Test set is purely the held-out subject
        test_data = subject_data[held_out_subj]
        X_test = test_data["eval_X"]
        y_test = test_data["eval_y"]
        test_meta = test_data["eval_meta"]

        # Train model (scaler fit strictly on X_train)
        clf = BaselineClassifier(model_type=model_type, random_state=42)
        clf.fit(X_train, y_train)

        # Predict on unseen held-out subject
        y_prob = clf.predict_proba(X_test)
        y_pred = (y_prob >= 0.5).astype(int)

        # Window metrics
        win_m = compute_window_metrics(y_true=y_test, y_prob=y_prob, threshold=0.5)

        # Event metrics (collar filter M=3 consecutive 1s windows = 3s burst)
        evt_m = compute_event_metrics(
            window_preds=y_pred,
            window_start_sec=test_meta["start_sec"].values,
            annotated_seizures_sec=test_data["annotated_seizures_sec"],
            total_duration_sec=test_data["duration_sec"],
            collar_consecutive_windows=3,
            collar_tolerance_sec=10.0,
        )

        fold_res = {
            "held_out_subject": held_out_subj,
            "train_subjects": train_subjs,
            "train_samples": int(len(y_train)),
            "test_windows": int(len(y_test)),
            "pr_auc": win_m["pr_auc"],
            "roc_auc": win_m["roc_auc"],
            "sensitivity": win_m["sensitivity"],
            "specificity": win_m["specificity"],
            "f1": win_m["f1"],
            "event_sensitivity": evt_m["event_sensitivity"],
            "detected_seizures": evt_m["detected_seizures"],
            "total_seizures": evt_m["total_seizures"],
            "false_alarms": evt_m["false_alarms"],
            "fp_per_hour": evt_m["fp_per_hour"],
        }
        loso_results.append(fold_res)
        print(
            f"  Fold [{held_out_subj} Held-Out]: PR-AUC={win_m['pr_auc']:.3f} | "
            f"ROC-AUC={win_m['roc_auc']:.3f} | Sens={win_m['sensitivity']:.3f} | "
            f"Spec={win_m['specificity']:.3f} | EventSens={evt_m['event_sensitivity']:.1f} | "
            f"FP/h={evt_m['fp_per_hour']:.1f}"
        )

    # 4. Summary Aggregation
    summary_df = pd.DataFrame(loso_results)
    mean_pr_auc = float(summary_df["pr_auc"].mean())
    mean_roc_auc = float(summary_df["roc_auc"].mean())
    mean_sens = float(summary_df["sensitivity"].mean())
    mean_spec = float(summary_df["specificity"].mean())
    mean_event_sens = float(summary_df["event_sensitivity"].mean())
    mean_fp_per_hour = float(summary_df["fp_per_hour"].mean())

    print("\n[Step 3/3] Aggregate Prototype LOSO Performance:")
    print(f"  Mean PR-AUC:           {mean_pr_auc:.4f}")
    print(f"  Mean ROC-AUC:          {mean_roc_auc:.4f}")
    print(f"  Mean Window Sens:      {mean_sens:.4f}")
    print(f"  Mean Window Spec:      {mean_spec:.4f}")
    print(f"  Mean Event Sens:       {mean_event_sens:.4f}")
    print(f"  Mean False Alarm/Hour: {mean_fp_per_hour:.2f} FP/h")

    # Feature Importance Analysis (trained on full prototype cohort)
    full_X = np.concatenate([subject_data[s]["train_X"] for s in PROTOTYPE_COHORT], axis=0)
    full_y = np.concatenate([subject_data[s]["train_y"] for s in PROTOTYPE_COHORT], axis=0)
    full_clf = BaselineClassifier(model_type=model_type, random_state=42)
    full_clf.fit(full_X, full_y)
    top_features = full_clf.get_feature_importances(feature_names=extractor.feature_names)
    if top_features is not None:
        print("\nTop 5 Predictive Electrophysiological Features:")
        for _, r in top_features.head(5).iterrows():
            print(f"  - {r['feature']}: {r['importance']:.4f}")

    results_payload = {
        "model_type": model_type,
        "cohort": PROTOTYPE_COHORT,
        "n_folds": len(loso_results),
        "mean_metrics": {
            "pr_auc": mean_pr_auc,
            "roc_auc": mean_roc_auc,
            "sensitivity": mean_sens,
            "specificity": mean_spec,
            "event_sensitivity": mean_event_sens,
            "fp_per_hour": mean_fp_per_hour,
        },
        "fold_results": loso_results,
    }

    if save_outputs:
        json_path = output_dir / f"prototype_baseline_{model_type}_results.json"
        csv_path = output_dir / f"prototype_baseline_{model_type}_summary.csv"
        with open(json_path, "w") as f:
            json.dump(results_payload, f, indent=2)
        summary_df.to_csv(csv_path, index=False)
        print(f"\nSaved benchmark outputs to {json_path} and {csv_path}")

    return results_payload


if __name__ == "__main__":
    res = run_prototype_baseline_experiment(model_type="rf")
