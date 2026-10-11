"""CHB-MIT metadata and annotation parser.

Extracts recording manifests and seizure annotations from per-patient summary
files and cross-validates them against physical EDF files and WFDB .seizures files.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    DATASET_DIR,
    PATIENT_ID_MAPPING,
    PROTOTYPE_COHORT,
    SAMPLING_RATE,
)


class CHBMITMetadataParser:
    """Parser for CHB-MIT summary files, recording manifests, and seizure annotations."""

    def __init__(self, dataset_dir: Path = DATASET_DIR):
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

    def parse_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Parse all cases chb01 to chb24 into recordings and seizures DataFrames.

        Returns:
            Tuple of (recordings_df, seizures_df)
        """
        all_recordings: List[Dict] = []
        all_seizures: List[Dict] = []

        # Find all summary files
        summary_files = sorted(self.dataset_dir.glob("chb*/*-summary.txt"))

        for sf in summary_files:
            case_id = sf.parent.name
            patient_id = PATIENT_ID_MAPPING.get(case_id, case_id)
            is_prototype = case_id in PROTOTYPE_COHORT

            recs, seizures = self._parse_summary_file(sf, case_id, patient_id, is_prototype)
            all_recordings.extend(recs)
            all_seizures.extend(seizures)

        recordings_df = pd.DataFrame(all_recordings)
        seizures_df = pd.DataFrame(all_seizures)

        # Reconcile unlisted files (specifically non-seizure files in chb24)
        recordings_df = self._reconcile_with_records(recordings_df)

        # Sort and clean index
        recordings_df = recordings_df.sort_values(["case_id", "file_name"]).reset_index(drop=True)
        seizures_df = seizures_df.sort_values(["case_id", "file_name", "seizure_idx"]).reset_index(drop=True)

        return recordings_df, seizures_df

    def _parse_summary_file(
        self, summary_path: Path, case_id: str, patient_id: str, is_prototype: bool
    ) -> Tuple[List[Dict], List[Dict]]:
        """Parse a single chbXX-summary.txt file."""
        with open(summary_path, "r", errors="ignore") as f:
            text = f.read()

        # Regular expression to extract file blocks
        # Handles filename characters including '+', '-', '_'
        file_pattern = re.compile(
            r"File Name:\s*([a-zA-Z0-9_\-\.\+]+)(.*?)(?=(?:File Name:|$))",
            re.DOTALL | re.IGNORECASE,
        )

        recordings: List[Dict] = []
        seizures: List[Dict] = []

        matches = file_pattern.findall(text)
        for raw_filename, block in matches:
            fn = raw_filename.strip()
            if not fn.endswith(".edf"):
                fn = f"{fn}.edf"

            # Start and End Times
            start_time_m = re.search(r"File Start Time:\s*([0-9:]+)", block, re.IGNORECASE)
            end_time_m = re.search(r"File End Time:\s*([0-9:]+)", block, re.IGNORECASE)
            file_start_time = start_time_m.group(1).strip() if start_time_m else None
            file_end_time = end_time_m.group(1).strip() if end_time_m else None

            # Number of seizures
            num_seiz_m = re.search(r"Number of Seizures in File:\s*(\d+)", block, re.IGNORECASE)
            num_seizures = int(num_seiz_m.group(1)) if num_seiz_m else 0

            # File paths
            rel_path = f"{case_id}/{fn}"
            full_path = self.dataset_dir / rel_path
            seizure_file_path = self.dataset_dir / f"{rel_path}.seizures"

            # Parse seizure start and end timestamps
            seiz_times = re.findall(
                r"Seizure\s*(\d*)\s*Start Time:\s*(\d+)\s*seconds.*?"
                r"Seizure\s*(\d*)\s*End Time:\s*(\d+)\s*seconds",
                block,
                re.DOTALL | re.IGNORECASE,
            )

            rec_dict = {
                "case_id": case_id,
                "patient_id": patient_id,
                "file_name": fn,
                "rel_path": rel_path,
                "full_path": str(full_path),
                "file_start_time": file_start_time,
                "file_end_time": file_end_time,
                "num_seizures": len(seiz_times),
                "has_seizures": len(seiz_times) > 0,
                "has_seizure_file": seizure_file_path.exists(),
                "file_exists": full_path.exists(),
                "is_prototype": is_prototype,
            }
            recordings.append(rec_dict)

            for s_idx, (_, start_s, _, end_s) in enumerate(seiz_times, start=1):
                start_sec = int(start_s)
                end_sec = int(end_s)
                dur_sec = end_sec - start_sec

                seiz_dict = {
                    "case_id": case_id,
                    "patient_id": patient_id,
                    "file_name": fn,
                    "rel_path": rel_path,
                    "seizure_idx": s_idx,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": dur_sec,
                    "start_sample": int(start_sec * SAMPLING_RATE),
                    "end_sample": int(end_sec * SAMPLING_RATE),
                    "duration_samples": int(dur_sec * SAMPLING_RATE),
                    "is_prototype": is_prototype,
                }
                seizures.append(seiz_dict)

        return recordings, seizures

    def _reconcile_with_records(self, recordings_df: pd.DataFrame) -> pd.DataFrame:
        """Reconcile parsed recordings against RECORDS to capture files omitted from summaries (e.g. chb24)."""
        records_file = self.dataset_dir / "RECORDS"
        if not records_file.exists():
            return recordings_df

        with open(records_file, "r") as f:
            all_records = [line.strip() for line in f if line.strip()]

        existing_rel_paths = set(recordings_df["rel_path"])
        missing_records = [r for r in all_records if r not in existing_rel_paths]

        if missing_records:
            new_rows = []
            for rel in missing_records:
                case_id = rel.split("/")[0]
                fn = rel.split("/")[1]
                patient_id = PATIENT_ID_MAPPING.get(case_id, case_id)
                full_path = self.dataset_dir / rel
                seizure_file_path = self.dataset_dir / f"{rel}.seizures"

                new_rows.append(
                    {
                        "case_id": case_id,
                        "patient_id": patient_id,
                        "file_name": fn,
                        "rel_path": rel,
                        "full_path": str(full_path),
                        "file_start_time": None,
                        "file_end_time": None,
                        "num_seizures": 0,
                        "has_seizures": False,
                        "has_seizure_file": seizure_file_path.exists(),
                        "file_exists": full_path.exists(),
                        "is_prototype": case_id in PROTOTYPE_COHORT,
                    }
                )

            supplementary_df = pd.DataFrame(new_rows)
            recordings_df = pd.concat([recordings_df, supplementary_df], ignore_index=True)

        return recordings_df


def save_manifests(
    recordings_df: pd.DataFrame,
    seizures_df: pd.DataFrame,
    output_dir: Path = Path("/root/Honors-EEG/data"),
) -> Tuple[Path, Path]:
    """Save parsed manifests to parquet and CSV formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rec_csv = output_dir / "recordings_manifest.csv"
    rec_parquet = output_dir / "recordings_manifest.parquet"
    seiz_csv = output_dir / "seizures_manifest.csv"
    seiz_parquet = output_dir / "seizures_manifest.parquet"

    recordings_df.to_csv(rec_csv, index=False)
    recordings_df.to_parquet(rec_parquet, index=False)
    seizures_df.to_csv(seiz_csv, index=False)
    seizures_df.to_parquet(seiz_parquet, index=False)

    return rec_parquet, seiz_parquet


if __name__ == "__main__":
    parser = CHBMITMetadataParser()
    recs, seizures = parser.parse_all()
    print(f"Total recordings parsed: {len(recs)} (Seizure files: {recs['has_seizures'].sum()})")
    print(f"Total seizures parsed: {len(seizures)} (Total duration: {seizures['duration_sec'].sum()} s)")
    proto_recs = recs[recs["is_prototype"]]
    proto_seiz = seizures[seizures["is_prototype"]]
    print(f"Prototype recordings: {len(proto_recs)} (Seizures: {len(proto_seiz)}, duration: {proto_seiz['duration_sec'].sum()} s)")
    save_manifests(recs, seizures)
    print("Manifests successfully saved to /root/Honors-EEG/data/")
