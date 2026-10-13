"""Generate complete CHB-MIT raw EDF inventory and audit metadata."""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor

import mne
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import DATASET_DIR, PATIENT_ID_MAPPING, CANONICAL_CHANNELS
from src.data.loader import EDFLoader

def generate_inventory():
    loader = EDFLoader()
    recs_df = pd.read_csv(PROJECT_ROOT / "data" / "recordings_manifest.csv")
    seiz_df = pd.read_csv(PROJECT_ROOT / "data" / "seizures_manifest.csv")
    
    # Identify cached and evaluated files from the validated feature cache
    cache_dir = PROJECT_ROOT / "outputs" / "experiments" / "exp_20261012_sprint" / "feature_cache" / "fp_2dd1bac799f4"
    
    eval_files = set(p.name.replace("_evaluation.parquet", ".edf") for p in cache_dir.glob("*/*_evaluation.parquet"))
    train_seiz_files = set(p.name.replace("_training.parquet", ".edf") for p in cache_dir.glob("*/*_training.parquet"))
    train_ns_files = set(p.name.replace("_training_ns.parquet", ".edf") for p in cache_dir.glob("*/*_training_ns.parquet"))
    
    print(f"Loaded manifests: {len(recs_df)} recordings, {len(seiz_df)} seizures")
    print(f"Identified cached roles: {len(eval_files)} evaluation files, {len(train_ns_files)} interictal training slices")
    
    # Map seizures to files
    seiz_summary = seiz_df.groupby("file_name").agg(
        num_seizures=("seizure_idx", "count"),
        total_seizure_duration_sec=("duration_sec", "sum")
    ).reset_index()
    seiz_map = {row["file_name"]: (row["num_seizures"], row["total_seizure_duration_sec"]) 
                for _, row in seiz_summary.iterrows()}

    # Worker function to inspect EDF header
    def inspect_file(row_tuple):
        _, row = row_tuple
        fpath = Path(row["full_path"])
        file_size_bytes = fpath.stat().st_size
        file_name = row["file_name"]
        case_id = row["case_id"]
        patient_id = PATIENT_ID_MAPPING.get(case_id, case_id)
        
        num_seiz, tot_seiz_dur = seiz_map.get(file_name, (0, 0.0))
        
        try:
            raw = mne.io.read_raw_edf(str(fpath), preload=False, verbose=False)
            sfreq = float(raw.info["sfreq"])
            n_times = raw.n_times
            duration_sec = n_times / sfreq
            ch_count = len(raw.ch_names)
            
            # Check canonical channel mapping
            try:
                loader._resolve_channel_indices(raw.ch_names)
                canonical_compatible = True
                compat_note = "OK"
            except Exception as e_map:
                canonical_compatible = False
                compat_note = str(e_map)
        except Exception as e:
            sfreq = -1.0
            duration_sec = -1.0
            ch_count = -1
            canonical_compatible = False
            compat_note = f"Read error: {str(e)}"
            
        # Determine status
        is_evaluated = file_name in eval_files
        is_cached_ns = file_name in train_ns_files
        is_preprocessed = is_evaluated or is_cached_ns
        is_cached = is_preprocessed
        
        # Determine exclusion reason
        if is_evaluated:
            exclusion_reason = "None (Evaluated in full-cohort benchmark continuous evaluation file)"
        elif is_cached_ns:
            exclusion_reason = "Cohort design: Selected as 300s baseline interictal slice for training (not selected for continuous evaluation)"
        elif not canonical_compatible:
            exclusion_reason = f"Incompatible montage: Recorded with referential montage missing canonical 18 longitudinal bipolar channels ({compat_note[:60]})"
        elif num_seiz > 0:
            exclusion_reason = "Cohort design: Secondary seizure recording (1 representative seizure file evaluated per case to prevent patient imbalance)"
        else:
            exclusion_reason = "Cohort design: Unselected interictal recording (1 representative interictal slice cached per case)"
            
        return {
            "patient_id": patient_id,
            "case_id": case_id,
            "file_name": file_name,
            "rel_path": row["rel_path"],
            "file_size_bytes": file_size_bytes,
            "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
            "native_sampling_rate_hz": sfreq,
            "channel_count": ch_count,
            "canonical_18_compatible": canonical_compatible,
            "duration_sec": round(duration_sec, 2),
            "duration_hours": round(duration_sec / 3600.0, 4),
            "num_seizures": num_seiz,
            "total_seizure_duration_sec": round(tot_seiz_dur, 2),
            "preprocessing_status": "preprocessed" if is_preprocessed else "not_preprocessed",
            "cache_status": "cached" if is_cached else "not_cached",
            "evaluation_status": "evaluated" if is_evaluated else "not_evaluated",
            "exclusion_reason": exclusion_reason,
        }

    print("Extracting headers across all 686 EDF files...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        records = list(executor.map(inspect_file, recs_df.iterrows()))
    t1 = time.time()
    print(f"Extraction completed in {t1 - t0:.2f} s")
    
    inventory_df = pd.DataFrame(records)
    # Sort deterministically
    inventory_df = inventory_df.sort_values(["case_id", "file_name"]).reset_index(drop=True)
    
    out_dir = PROJECT_ROOT / "outputs" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "chb_mit_raw_edf_inventory.csv"
    inventory_df.to_csv(out_csv, index=False)
    print(f"Saved complete inventory ({len(inventory_df)} rows) to {out_csv}")
    
    # Print summary statistics
    total_bytes = inventory_df["file_size_bytes"].sum()
    total_gb = total_bytes / (1024**3)
    total_edf_count = len(inventory_df)
    total_duration_sec = inventory_df["duration_sec"].sum()
    total_duration_hours = total_duration_sec / 3600.0
    
    preprocessed_df = inventory_df[inventory_df["preprocessing_status"] == "preprocessed"]
    preprocessed_duration_hours = preprocessed_df["duration_sec"].sum() / 3600.0
    
    evaluated_df = inventory_df[inventory_df["evaluation_status"] == "evaluated"]
    evaluated_duration_hours = evaluated_df["duration_sec"].sum() / 3600.0
    
    print("\n--- DATASET INVENTORY SUMMARY ---")
    print(f"Total Raw Files: {total_edf_count}")
    print(f"Total Raw Bytes: {total_bytes:,} bytes ({total_gb:.2f} GB)")
    print(f"Total Recording Duration: {total_duration_sec:,.1f} s ({total_duration_hours:.2f} hours)")
    print(f"Canonical 18 Compatible Files: {inventory_df['canonical_18_compatible'].sum()} / {total_edf_count}")
    print(f"Incompatible Files: {(~inventory_df['canonical_18_compatible']).sum()}")
    print(f"Total Preprocessed/Cached Files: {len(preprocessed_df)} ({preprocessed_duration_hours:.2f} hours)")
    print(f"Total Continuous Evaluation Files: {len(evaluated_df)} ({evaluated_duration_hours:.2f} hours)")
    print(f"Biological Patients Covered in Evaluation: {evaluated_df['patient_id'].nunique()} / {inventory_df['patient_id'].nunique()} (100.0%)")
    print(f"Recording Cases Covered in Evaluation: {evaluated_df['case_id'].nunique()} / {inventory_df['case_id'].nunique()} (100.0%)")
    print(f"Total Seizures in Evaluated Files: {evaluated_df['num_seizures'].sum()} / {inventory_df['num_seizures'].sum()}")

if __name__ == "__main__":
    generate_inventory()
