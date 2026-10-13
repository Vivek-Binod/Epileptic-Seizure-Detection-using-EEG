"""Leak-free Leave-One-Patient-Out post-processing selection on CHB-MIT.

In each fold:
1. Training cases are used to evaluate candidate post-processing configurations.
2. The candidate configuration that minimizes false alarms subject to zero sensitivity loss
   on training cases is selected.
3. The selected configuration is then evaluated on the held-out test patient.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.metadata_parser import CHBMITMetadataParser
from src.data.raw_window_dataset import SUBJECT_PAIRS
from src.evaluation.metrics import compute_window_metrics
from src.evaluation.postprocessing import ewma_smooth
from src.utils.config import OUTPUTS_DIR, PATIENT_ID_MAPPING
from scripts.run_postprocessing_ablation import compute_event_metrics_with_lockout


def run_leak_free_selection():
    exp_dir = OUTPUTS_DIR / "experiments" / "exp_20261013_postprocessing_ablation"
    pred_dir = OUTPUTS_DIR / "experiments" / "exp_20261013_deep_learning_cnn" / "predictions"

    parser = CHBMITMetadataParser()
    _, seiz_df = parser.parse_all()

    cases = sorted(list(SUBJECT_PAIRS.keys()))
    unique_patients = sorted(list(set(PATIENT_ID_MAPPING[c] for c in cases)))

    case_data = {}
    for c in cases:
        p = PATIENT_ID_MAPPING[c]
        seiz_fn, _ = SUBJECT_PAIRS[c]
        subj_seiz = seiz_df[(seiz_df["case_id"] == c) & (seiz_df["file_name"] == seiz_fn)]
        ann_seiz = [(float(r["start_sec"]), float(r["end_sec"])) for _, r in subj_seiz.iterrows()]
        pred_npz = np.load(pred_dir / f"{c}_predictions.npz")
        case_data[c] = {
            "patient_id": p,
            "probs_raw": pred_npz["probs_raw"],
            "y": pred_npz["y"],
            "ann_seiz": ann_seiz,
            "dur_sec": float(len(pred_npz["y"])),
            "dur_hours": float(len(pred_npz["y"])) / 3600.0,
        }

    # Candidate lockout options under standard benchmark EWMA (alpha=0.30, M=3)
    candidate_lockouts = [0.0, 15.0, 30.0, 60.0]

    outer_test_results = []
    fold_selections = []

    for fold_idx, held_out_patient in enumerate(unique_patients, start=1):
        test_cases = [c for c in cases if PATIENT_ID_MAPPING[c] == held_out_patient]
        train_cases = [c for c in cases if PATIENT_ID_MAPPING[c] != held_out_patient]

        # Inner evaluation on training cases:
        # Criterion: maintain maximum event sensitivity while minimizing false alarms
        best_lockout = 0.0
        best_train_fa = float("inf")
        baseline_train_det = None

        train_perf = {}
        for lk in candidate_lockouts:
            tot_det = 0
            tot_fa = 0
            for tc in train_cases:
                cd = case_data[tc]
                probs = ewma_smooth(cd["probs_raw"], alpha=0.30)
                win_start = np.arange(len(probs), dtype=float) * 1.0
                em = compute_event_metrics_with_lockout(
                    (probs >= 0.50).astype(int),
                    win_start,
                    cd["ann_seiz"],
                    cd["dur_sec"],
                    collar_consecutive_windows=3,
                    refractory_lockout_sec=lk,
                )
                tot_det += em["detected_seizures"]
                tot_fa += em["false_alarms"]

            train_perf[lk] = {"det": tot_det, "fa": tot_fa}
            if lk == 0.0:
                baseline_train_det = tot_det

        # Selection rule: strictly require zero loss of sensitivity on training cases
        valid_candidates = [
            lk for lk in candidate_lockouts
            if train_perf[lk]["det"] >= baseline_train_det
        ]
        # Among valid candidates, pick minimal false alarms
        best_lockout = min(valid_candidates, key=lambda lk: train_perf[lk]["fa"])

        fold_selections.append({
            "fold_idx": fold_idx,
            "held_out_patient": held_out_patient,
            "selected_lockout_sec": best_lockout,
            "train_perf": train_perf,
        })

        # Evaluate on held-out test cases
        for test_case in test_cases:
            cd = case_data[test_case]
            probs = ewma_smooth(cd["probs_raw"], alpha=0.30)
            win_start = np.arange(len(probs), dtype=float) * 1.0
            wm = compute_window_metrics(cd["y"], probs, threshold=0.50)
            em = compute_event_metrics_with_lockout(
                (probs >= 0.50).astype(int),
                win_start,
                cd["ann_seiz"],
                cd["dur_sec"],
                collar_consecutive_windows=3,
                refractory_lockout_sec=best_lockout,
            )
            outer_test_results.append({
                "fold_idx": fold_idx,
                "held_out_patient": held_out_patient,
                "case_id": test_case,
                "selected_lockout_sec": best_lockout,
                "pr_auc": wm["pr_auc"],
                "roc_auc": wm["roc_auc"],
                "event_sensitivity": em["event_sensitivity"],
                "detected_seizures": em["detected_seizures"],
                "total_seizures": em["total_seizures"],
                "false_alarms": em["false_alarms"],
                "fp_per_hour": em["fp_per_hour"],
                "duration_hours": cd["dur_hours"],
                "median_delay_sec": em["median_delay_sec"],
            })

    df_leak_free_cases = pd.DataFrame(outer_test_results)
    df_leak_free_cases.to_csv(exp_dir / "leak_free_postprocessing_case_folds.csv", index=False)

    macro_pr = df_leak_free_cases["pr_auc"].mean()
    macro_roc = df_leak_free_cases["roc_auc"].mean()
    macro_sens = df_leak_free_cases["event_sensitivity"].mean()
    macro_fp = df_leak_free_cases["fp_per_hour"].mean()
    tot_det = df_leak_free_cases["detected_seizures"].sum()
    tot_seiz = df_leak_free_cases["total_seizures"].sum()
    tot_fa = df_leak_free_cases["false_alarms"].sum()
    tot_hours = df_leak_free_cases["duration_hours"].sum()

    leak_free_summary = {
        "strategy": "leak_free_inner_lopo_lockout",
        "macro_case_pr_auc": macro_pr,
        "macro_case_roc_auc": macro_roc,
        "macro_case_sens": macro_sens,
        "pooled_detected_seizures": int(tot_det),
        "pooled_total_seizures": int(tot_seiz),
        "pooled_event_sens": tot_det / tot_seiz,
        "pooled_false_alarms": int(tot_fa),
        "pooled_duration_hours": float(tot_hours),
        "pooled_fp_per_hour": tot_fa / tot_hours,
        "macro_case_fp_per_hour": macro_fp,
        "selected_lockouts_distribution": {
            str(k): int(v) for k, v in df_leak_free_cases.groupby("held_out_patient")["selected_lockout_sec"].first().value_counts().items()
        },
    }

    with open(exp_dir / "leak_free_postprocessing_summary.json", "w") as f:
        json.dump(leak_free_summary, f, indent=2)

    print("Leak-free post-processing evaluation complete!")
    print(json.dumps(leak_free_summary, indent=2))


if __name__ == "__main__":
    run_leak_free_selection()
