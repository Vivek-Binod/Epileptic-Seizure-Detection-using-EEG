"""Unit tests for metadata and annotation parser."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.metadata_parser import CHBMITMetadataParser


def test_metadata_parser():
    parser = CHBMITMetadataParser()
    recordings_df, seizures_df = parser.parse_all()

    # 1. Total recordings verification
    assert len(recordings_df) == 686, f"Expected 686 recordings, got {len(recordings_df)}"
    assert recordings_df["file_exists"].all(), "Expected all 686 recordings to exist on disk"

    # 2. Total seizures and duration verification
    assert len(seizures_df) == 198, f"Expected 198 seizures, got {len(seizures_df)}"
    total_duration = seizures_df["duration_sec"].sum()
    assert total_duration == 11611, f"Expected 11611 seconds of seizures, got {total_duration}"

    # 3. Files with seizures
    num_seiz_recs = recordings_df["has_seizures"].sum()
    assert num_seiz_recs == 141, f"Expected 141 files with seizures, got {num_seiz_recs}"

    # 4. Patient ID mapping verification (chb01 and chb21 bound)
    chb01_patients = recordings_df[recordings_df["case_id"] == "chb01"]["patient_id"].unique()
    chb21_patients = recordings_df[recordings_df["case_id"] == "chb21"]["patient_id"].unique()
    assert len(chb01_patients) == 1 and chb01_patients[0] == "patient_01"
    assert len(chb21_patients) == 1 and chb21_patients[0] == "patient_01"

    # 5. CHB07 discrepancy verification
    chb07_18 = recordings_df[recordings_df["file_name"] == "chb07_18.edf"].iloc[0]
    chb07_19 = recordings_df[recordings_df["file_name"] == "chb07_19.edf"].iloc[0]
    assert chb07_18["num_seizures"] == 0
    assert not chb07_18["has_seizure_file"]
    assert chb07_19["num_seizures"] == 1
    assert chb07_19["has_seizure_file"]

    chb07_19_seiz = seizures_df[seizures_df["file_name"] == "chb07_19.edf"].iloc[0]
    assert chb07_19_seiz["start_sec"] == 13688
    assert chb07_19_seiz["end_sec"] == 13831
    assert chb07_19_seiz["duration_sec"] == 143

    # 6. Filename with '+' character
    chb02_plus = recordings_df[recordings_df["file_name"] == "chb02_16+.edf"]
    assert len(chb02_plus) == 1
    assert chb02_plus.iloc[0]["num_seizures"] == 1

    # 7. Prototype cohort verification
    proto_recs = recordings_df[recordings_df["is_prototype"]]
    proto_seiz = seizures_df[seizures_df["is_prototype"]]
    assert len(proto_recs) == 158, f"Expected 158 prototype recordings, got {len(proto_recs)}"
    assert len(proto_seiz) == 27, f"Expected 27 prototype seizures, got {len(proto_seiz)}"
    assert proto_seiz["duration_sec"].sum() == 2646, f"Expected 2646s prototype duration, got {proto_seiz['duration_sec'].sum()}"

    print("All metadata parser assertions passed successfully!")


if __name__ == "__main__":
    test_metadata_parser()
