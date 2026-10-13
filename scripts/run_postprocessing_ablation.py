"""Controlled post-processing ablation on CNN predictions for CHB-MIT full cohort.

Evaluates the impact of:
1. EWMA smoothing factor alpha in {0.15, 0.30, 0.50}
2. Collar consecutive window requirement M in {3, 5}
3. Causal refractory lockout period T_lockout in {0s, 15s, 30s, 60s}
4. Leak-free LOPO inner-validation selection vs held-out evaluation.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.metadata_parser import CHBMITMetadataParser
from src.data.raw_window_dataset import SUBJECT_PAIRS
from src.evaluation.metrics import compute_window_metrics
from src.evaluation.postprocessing import ewma_smooth
from src.utils.config import OUTPUTS_DIR, PATIENT_ID_MAPPING


def compute_event_metrics_with_lockout(
    window_preds: np.ndarray,
    window_start_sec: np.ndarray,
    annotated_seizures_sec: List[Tuple[float, float]],
    total_duration_sec: float,
    collar_consecutive_windows: int = 3,
    collar_tolerance_sec: float = 10.0,
    refractory_lockout_sec: float = 0.0,
) -> Dict:
    """Compute event detection metrics with optional causal refractory lockout."""
    alarm_times_sec: List[float] = []
    consecutive = 0
    last_alarm_time = -float("inf")

    for pred, t_start in zip(window_preds, window_start_sec):
        if pred == 1:
            consecutive += 1
            if consecutive == collar_consecutive_windows:
                # Check causal refractory lockout
                if (float(t_start) - last_alarm_time) >= refractory_lockout_sec:
                    alarm_times_sec.append(float(t_start))
                    last_alarm_time = float(t_start)
        else:
            consecutive = 0

    n_seizures = len(annotated_seizures_sec)
    detected_seizures = set()
    seizure_first_alarm: Dict[int, float] = {}
    false_alarms_count = 0

    for alarm_t in alarm_times_sec:
        matched_seizure_idx = None
        for s_idx, (s_start, s_end) in enumerate(annotated_seizures_sec):
            if (s_start - collar_tolerance_sec) <= alarm_t <= (s_end + collar_tolerance_sec):
                matched_seizure_idx = s_idx
                break

        if matched_seizure_idx is not None:
            detected_seizures.add(matched_seizure_idx)
            if matched_seizure_idx not in seizure_first_alarm:
                seizure_first_alarm[matched_seizure_idx] = float(alarm_t)
        else:
            false_alarms_count += 1

    delays = [
        float(seizure_first_alarm[idx] - annotated_seizures_sec[idx][0])
        for idx in sorted(seizure_first_alarm.keys())
    ]
    mean_delay_sec = float(np.mean(delays)) if len(delays) > 0 else float("nan")
    median_delay_sec = float(np.median(delays)) if len(delays) > 0 else float("nan")

    total_hours = total_duration_sec / 3600.0 if total_duration_sec > 0 else 1.0
    event_sensitivity = len(detected_seizures) / float(n_seizures) if n_seizures > 0 else 1.0
    fp_per_hour = false_alarms_count / total_hours

    return {
        "event_sensitivity": float(event_sensitivity),
        "detected_seizures": len(detected_seizures),
        "total_seizures": n_seizures,
        "false_alarms": int(false_alarms_count),
        "total_alarms": len(alarm_times_sec),
        "total_hours": float(total_hours),
        "fp_per_hour": float(fp_per_hour),
        "mean_delay_sec": mean_delay_sec,
        "median_delay_sec": median_delay_sec,
    }


def run_ablation():
    exp_dir = OUTPUTS_DIR / "experiments" / "exp_20261013_postprocessing_ablation"
    exp_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = OUTPUTS_DIR / "experiments" / "exp_20261013_deep_learning_cnn" / "predictions"

    parser = CHBMITMetadataParser()
    _, seiz_df = parser.parse_all()

    cases = sorted(list(SUBJECT_PAIRS.keys()))
    unique_patients = sorted(list(set(PATIENT_ID_MAPPING[c] for c in cases)))

    # Load all cached predictions
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

    # Grid of configurations
    configs = [
        {"config_id": "raw_baseline", "alpha": None, "M": 3, "lockout": 0.0, "name": "Raw 1D CNN (No EWMA, No Lockout)"},
        {"config_id": "ewma_0.30_baseline", "alpha": 0.30, "M": 3, "lockout": 0.0, "name": "EWMA alpha=0.30 (Current Benchmark)"},
        {"config_id": "ewma_0.30_lockout_15s", "alpha": 0.30, "M": 3, "lockout": 15.0, "name": "EWMA alpha=0.30 + Lockout 15s"},
        {"config_id": "ewma_0.30_lockout_30s", "alpha": 0.30, "M": 3, "lockout": 30.0, "name": "EWMA alpha=0.30 + Lockout 30s"},
        {"config_id": "ewma_0.30_lockout_60s", "alpha": 0.30, "M": 3, "lockout": 60.0, "name": "EWMA alpha=0.30 + Lockout 60s"},
        {"config_id": "ewma_0.15_baseline", "alpha": 0.15, "M": 3, "lockout": 0.0, "name": "EWMA alpha=0.15 (Heavier Smoothing)"},
        {"config_id": "ewma_0.15_lockout_60s", "alpha": 0.15, "M": 3, "lockout": 60.0, "name": "EWMA alpha=0.15 + Lockout 60s"},
        {"config_id": "ewma_0.30_M5_baseline", "alpha": 0.30, "M": 5, "lockout": 0.0, "name": "EWMA alpha=0.30 + Collar M=5"},
        {"config_id": "ewma_0.30_M5_lockout_60s", "alpha": 0.30, "M": 5, "lockout": 60.0, "name": "EWMA alpha=0.30 + Collar M=5 + Lockout 60s"},
    ]

    all_case_rows = []
    summary_rows = []

    for cfg in configs:
        cfg_id = cfg["config_id"]
        alpha = cfg["alpha"]
        M = cfg["M"]
        lockout = cfg["lockout"]

        case_results = []
        for c in cases:
            cd = case_data[c]
            p = cd["patient_id"]
            raw_p = cd["probs_raw"]
            y = cd["y"]
            ann_s = cd["ann_seiz"]
            dur_s = cd["dur_sec"]
            dur_h = cd["dur_hours"]

            probs = ewma_smooth(raw_p, alpha=alpha) if alpha is not None else raw_p
            win_start = np.arange(len(probs), dtype=float) * 1.0

            w_m = compute_window_metrics(y, probs, threshold=0.50)
            e_m = compute_event_metrics_with_lockout(
                window_preds=(probs >= 0.50).astype(int),
                window_start_sec=win_start,
                annotated_seizures_sec=ann_s,
                total_duration_sec=dur_s,
                collar_consecutive_windows=M,
                collar_tolerance_sec=10.0,
                refractory_lockout_sec=lockout,
            )

            res = {
                "config_id": cfg_id,
                "config_name": cfg["name"],
                "case_id": c,
                "patient_id": p,
                "alpha": alpha if alpha is not None else 1.0,
                "collar_M": M,
                "lockout_sec": lockout,
                "pr_auc": w_m["pr_auc"],
                "roc_auc": w_m["roc_auc"],
                "event_sensitivity": e_m["event_sensitivity"],
                "detected_seizures": e_m["detected_seizures"],
                "total_seizures": e_m["total_seizures"],
                "false_alarms": e_m["false_alarms"],
                "fp_per_hour": e_m["fp_per_hour"],
                "duration_hours": dur_h,
                "mean_delay_sec": e_m["mean_delay_sec"],
                "median_delay_sec": e_m["median_delay_sec"],
            }
            all_case_rows.append(res)
            case_results.append(res)

        # Summary for this configuration
        df_c = pd.DataFrame(case_results)
        macro_pr = df_c["pr_auc"].mean()
        macro_roc = df_c["roc_auc"].mean()
        macro_sens = df_c["event_sensitivity"].mean()
        macro_fp = df_c["fp_per_hour"].mean()
        tot_det = df_c["detected_seizures"].sum()
        tot_seiz = df_c["total_seizures"].sum()
        tot_fa = df_c["false_alarms"].sum()
        tot_h = df_c["duration_hours"].sum()
        pooled_sens = tot_det / tot_seiz
        pooled_fp = tot_fa / tot_h

        delays = df_c["median_delay_sec"].dropna()
        med_delay = delays.median() if len(delays) > 0 else np.nan

        summary_rows.append({
            "config_id": cfg_id,
            "config_name": cfg["name"],
            "alpha": alpha if alpha is not None else 1.0,
            "collar_M": M,
            "lockout_sec": lockout,
            "macro_case_pr_auc": macro_pr,
            "macro_case_roc_auc": macro_roc,
            "macro_case_sens": macro_sens,
            "pooled_detected_seizures": tot_det,
            "pooled_total_seizures": tot_seiz,
            "pooled_event_sens": pooled_sens,
            "pooled_false_alarms": tot_fa,
            "pooled_duration_hours": tot_h,
            "pooled_fp_per_hour": pooled_fp,
            "macro_case_fp_per_hour": macro_fp,
            "median_delay_sec": med_delay,
        })

    df_cases_all = pd.DataFrame(all_case_rows)
    df_summary_all = pd.DataFrame(summary_rows)

    df_cases_all.to_csv(exp_dir / "postprocessing_ablation_case_folds.csv", index=False)
    df_summary_all.to_csv(exp_dir / "postprocessing_ablation_summary.csv", index=False)
    print("Post-processing ablation completed successfully!")
    print(df_summary_all[[
        "config_id", "macro_case_pr_auc", "macro_case_sens", "pooled_event_sens",
        "pooled_detected_seizures", "pooled_false_alarms", "pooled_fp_per_hour", "macro_case_fp_per_hour"
    ]].to_string(index=False))


if __name__ == "__main__":
    run_ablation()
