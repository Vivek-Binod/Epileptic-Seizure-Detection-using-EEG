"""Controlled 3-Way Experimental Comparison: Conventional vs Connectivity vs Combined.

Evaluates:
  Condition A: Conventional Univariate Features (252 features) + Random Forest
  Condition B: Functional Connectivity (PLV + PLI) Features (200 features) + Random Forest
  Condition C: Combined (Conventional + Connectivity, 452 features) + Random Forest

Protocol:
1. Identical patient-independent Leave-One-Subject-Out (LOSO) cross-validation across
   the 5 prototype patients (chb01, chb03, chb05, chb07, chb08).
2. Strict leakage prevention: StandardScaler fit strictly on training folds.
3. Dual evaluation:
   - Window-level PR-AUC, ROC-AUC, sensitivity, specificity, F1.
   - Event-level collar hysteresis alarm metrics (M=3 consecutive 1s windows = 3s burst,
     collar tolerance +-10s, single-count rule, FP/hour).
"""

import json
from pathlib import Path
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import EDFLoader
from src.data.metadata_parser import CHBMITMetadataParser
from src.data.windowing import WindowGenerator
from src.evaluation.metrics import compute_event_metrics, compute_window_metrics
from src.features.connectivity import EEGConnectivityExtractor
from src.features.feature_extractor import EEGFeatureExtractor
from src.models.baseline import BaselineClassifier
from src.utils.config import DATASET_DIR, OUTPUTS_DIR, PROTOTYPE_COHORT, SAMPLING_RATE


def run_connectivity_comparison_experiment(
    output_dir: Path = OUTPUTS_DIR,
    save_outputs: bool = True,
    random_state: int = 42,
) -> Dict:
    """Execute controlled 3-way experiment comparing Conventional, Connectivity, and Combined."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CONTROLLED 3-WAY EXPERIMENT: CONVENTIONAL vs CONNECTIVITY vs COMBINED")
    print(f"Cohort: {PROTOTYPE_COHORT} (5 prototype patients, LOSO)")
    print(f"Model: Random Forest (random_state={random_state}, class_weight='balanced')")
    print("=" * 80)

    # 1. Manifests & Initializers
    parser = CHBMITMetadataParser()
    recs_df, seiz_df = parser.parse_all()

    loader = EDFLoader()
    window_gen = WindowGenerator()
    conv_extractor = EEGFeatureExtractor()
    conn_extractor = EEGConnectivityExtractor(sampling_rate=SAMPLING_RATE)

    # Benchmark test pairs per subject: (seizure_file, non_seizure_file)
    test_pairs: Dict[str, Tuple[str, str]] = {
        "chb01": ("chb01_03.edf", "chb01_01.edf"),
        "chb03": ("chb03_01.edf", "chb03_05.edf"),
        "chb05": ("chb05_06.edf", "chb05_01.edf"),
        "chb07": ("chb07_19.edf", "chb07_01.edf"),
        "chb08": ("chb08_02.edf", "chb08_03.edf"),
    }

    # 2. Extract features per subject
    subject_data: Dict[str, Dict] = {}

    print("\n[Step 1/3] Extracting features for prototype recordings...")
    t_extract_start = time.time()

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

        # A. Load full seizure recording signal
        seiz_signal, seiz_info = loader.load_full_recording(seiz_path)
        tr_win, tr_y, _ = window_gen.extract_windows_for_recording(
            signal=seiz_signal,
            seizure_intervals=seiz_intervals,
            is_training=True,
            neg_pos_ratio=10.0,
            random_seed=42,
        )

        # B. Non-seizure recording (5 min background = 300s)
        ns_slice = loader.load_slice(
            non_seiz_path,
            start_sample=0,
            stop_sample=300 * SAMPLING_RATE,
            apply_filter=True,
            apply_standardization=True,
        )
        ns_win, ns_y, _ = window_gen.extract_windows_for_recording(
            signal=ns_slice,
            seizure_intervals=[],
            is_training=True,
        )

        # Training windows & labels
        comb_tr_win = np.concatenate([tr_win, ns_win], axis=0) if len(ns_win) > 0 else tr_win
        comb_tr_y = np.concatenate([tr_y, ns_y], axis=0) if len(ns_y) > 0 else tr_y

        # Training features: Conventional & Connectivity
        tr_X_conv = conv_extractor.extract_features(comb_tr_win)
        tr_X_conn = conn_extractor.extract_features(comb_tr_win, batch_size=256)
        tr_X_comb = np.concatenate([tr_X_conv, tr_X_conn], axis=1)

        # C. Evaluation windows (continuous 1s stride)
        eval_win, eval_y, eval_meta = window_gen.extract_windows_for_recording(
            signal=seiz_signal,
            seizure_intervals=seiz_intervals,
            is_training=False,
            eval_stride_samples=SAMPLING_RATE,
        )

        eval_X_conv = conv_extractor.extract_features(eval_win)
        eval_X_conn = conn_extractor.extract_features(eval_win, batch_size=256)
        eval_X_comb = np.concatenate([eval_X_conv, eval_X_conn], axis=1)

        # Release large signal and raw window tensors immediately
        del seiz_signal, tr_win, ns_win, comb_tr_win, eval_win
        import gc
        gc.collect()

        subject_data[subj] = {
            "tr_y": comb_tr_y,
            "eval_y": eval_y,
            "eval_meta": eval_meta,
            "annotated_seizures_sec": annotated_seiz_sec,
            "duration_sec": float(seiz_info["duration_sec"]),
            "conv": {"train_X": tr_X_conv, "eval_X": eval_X_conv},
            "conn": {"train_X": tr_X_conn, "eval_X": eval_X_conn},
            "comb": {"train_X": tr_X_comb, "eval_X": eval_X_comb},
        }

        print(
            f"  Subj {subj}: Train={len(comb_tr_y)} (Pos={np.sum(comb_tr_y == 1)}, Neg={np.sum(comb_tr_y == 0)}) | "
            f"Eval={len(eval_y)} | "
            f"Conv shape={tr_X_conv.shape[1]} | Conn shape={tr_X_conn.shape[1]} | Comb shape={tr_X_comb.shape[1]}"
        )

    t_extract_end = time.time()
    print(f"Feature extraction completed in {t_extract_end - t_extract_start:.2f}s")

    # 3. Evaluate each condition in LOSO
    conditions = {
        "conventional": {
            "name": "Condition A: Conventional Univariate (252 feats)",
            "key": "conv",
            "n_feats": 252,
        },
        "connectivity": {
            "name": "Condition B: Connectivity PLV+PLI (200 feats)",
            "key": "conn",
            "n_feats": 200,
        },
        "combined": {
            "name": "Condition C: Combined Conventional+Connectivity (452 feats)",
            "key": "comb",
            "n_feats": 452,
        },
    }

    comparison_results: Dict[str, Dict] = {}

    print("\n[Step 2/3] Executing LOSO Cross-Validation across 3 Conditions...")
    for cond_id, cond_info in conditions.items():
        print(f"\n---> Evaluating {cond_info['name']}...")
        cond_key = cond_info["key"]
        loso_folds: List[Dict] = []

        for held_out_subj in PROTOTYPE_COHORT:
            train_subjs = [s for s in PROTOTYPE_COHORT if s != held_out_subj]

            X_train = np.concatenate([subject_data[s][cond_key]["train_X"] for s in train_subjs], axis=0)
            y_train = np.concatenate([subject_data[s]["tr_y"] for s in train_subjs], axis=0)

            test_data = subject_data[held_out_subj]
            X_test = test_data[cond_key]["eval_X"]
            y_test = test_data["eval_y"]
            test_meta = test_data["eval_meta"]

            # Train classifier (StandardScaler fitted strictly inside BaselineClassifier on X_train)
            clf = BaselineClassifier(model_type="rf", random_state=random_state)
            clf.fit(X_train, y_train)

            y_prob = clf.predict_proba(X_test)
            y_pred = (y_prob >= 0.5).astype(int)

            # Window metrics
            win_m = compute_window_metrics(y_true=y_test, y_prob=y_prob, threshold=0.5)

            # Event collar metrics (M=3 consecutive windows = 3s burst, +-10s collar)
            evt_m = compute_event_metrics(
                window_preds=y_pred,
                window_start_sec=test_meta["start_sec"].values,
                annotated_seizures_sec=test_data["annotated_seizures_sec"],
                total_duration_sec=test_data["duration_sec"],
                collar_consecutive_windows=3,
                collar_tolerance_sec=10.0,
            )

            fold_res = {
                "condition": cond_id,
                "held_out_subject": held_out_subj,
                "n_features": cond_info["n_feats"],
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
            loso_folds.append(fold_res)
            print(
                f"  [{held_out_subj}]: PR-AUC={win_m['pr_auc']:.3f} | ROC-AUC={win_m['roc_auc']:.3f} | "
                f"Sens={win_m['sensitivity']:.3f} | Spec={win_m['specificity']:.3f} | "
                f"EvtSens={evt_m['event_sensitivity']:.1f} | FP/h={evt_m['fp_per_hour']:.1f}"
            )

        summary_df = pd.DataFrame(loso_folds)
        mean_metrics = {
            "pr_auc": float(summary_df["pr_auc"].mean()),
            "roc_auc": float(summary_df["roc_auc"].mean()),
            "sensitivity": float(summary_df["sensitivity"].mean()),
            "specificity": float(summary_df["specificity"].mean()),
            "f1": float(summary_df["f1"].mean()),
            "event_sensitivity": float(summary_df["event_sensitivity"].mean()),
            "fp_per_hour": float(summary_df["fp_per_hour"].mean()),
        }

        comparison_results[cond_id] = {
            "name": cond_info["name"],
            "n_features": cond_info["n_feats"],
            "mean_metrics": mean_metrics,
            "folds": loso_folds,
        }

    # 4. Construct Comparison Summary Table
    print("\n" + "=" * 80)
    print("[Step 3/3] AGGREGATE CONTROLLED COMPARISON RESULTS")
    print("=" * 80)

    rows = []
    for cond_id, res in comparison_results.items():
        m = res["mean_metrics"]
        rows.append({
            "Condition": cond_id.capitalize(),
            "Features": res["n_features"],
            "Mean PR-AUC": f"{m['pr_auc']:.4f}",
            "Mean ROC-AUC": f"{m['roc_auc']:.4f}",
            "Mean Window Sens": f"{m['sensitivity']:.4f}",
            "Mean Window Spec": f"{m['specificity']:.4f}",
            "Mean Event Sens": f"{m['event_sensitivity']:.2f}",
            "Mean FP/hour": f"{m['fp_per_hour']:.2f}",
        })

    table_df = pd.DataFrame(rows)
    print(table_df.to_string(index=False))

    # Per-subject comparison breakdown
    all_folds_list = []
    for cond_id, res in comparison_results.items():
        for f in res["folds"]:
            all_folds_list.append(f)
    all_folds_df = pd.DataFrame(all_folds_list)

    if save_outputs:
        json_path = output_dir / "connectivity_comparison_results.json"
        csv_path = output_dir / "connectivity_comparison_summary.csv"
        detailed_csv_path = output_dir / "connectivity_comparison_folds.csv"

        with open(json_path, "w") as f:
            json.dump(comparison_results, f, indent=2)
        table_df.to_csv(csv_path, index=False)
        all_folds_df.to_csv(detailed_csv_path, index=False)
        print(f"\nSaved comparison artifacts to {json_path}, {csv_path}, and {detailed_csv_path}")

    return {
        "summary_table": table_df.to_dict(orient="records"),
        "comparison_results": comparison_results,
    }


if __name__ == "__main__":
    run_connectivity_comparison_experiment()
