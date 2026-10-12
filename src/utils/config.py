"""Central configuration and constants for CHB-MIT EEG Seizure Detection."""

from pathlib import Path
from typing import Dict, List

# Paths
DATASET_DIR = Path("/mnt/d/College/Honors/chb-mit-scalp-eeg-database-1.0.0")
WORKSPACE_DIR = Path("/root/Honors-EEG")
OUTPUTS_DIR = WORKSPACE_DIR / "outputs"
DATA_DIR = WORKSPACE_DIR / "data"
FEATURE_CACHE_DIR = DATA_DIR / "feature_cache"

# Cache configuration
CACHE_VERSION = "1.0.0"

# Standard 18 bipolar channels (longitudinal bipolar / double banana montage)
# Ordered logically by anatomical chains:
CANONICAL_CHANNELS: List[str] = [
    # Left Temporal Chain (4 channels)
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    # Left Parasagittal Chain (4 channels)
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    # Central/Midline Chain (2 channels)
    "FZ-CZ",
    "CZ-PZ",
    # Right Parasagittal Chain (4 channels)
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    # Right Temporal Chain (4 channels)
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
]

# Signal acquisition parameters
SAMPLING_RATE: int = 256  # Hz
BANDPASS_LOW: float = 0.5  # Hz
BANDPASS_HIGH: float = 45.0  # Hz
FILTER_ORDER: int = 4

# Windowing & Labeling parameters
WINDOW_DURATION_SEC: float = 4.0  # seconds
WINDOW_SAMPLES: int = int(WINDOW_DURATION_SEC * SAMPLING_RATE)  # 1024 samples

ICTAL_STRIDE_SEC: float = 1.0  # seconds (75% overlap for data augmentation)
ICTAL_STRIDE_SAMPLES: int = int(ICTAL_STRIDE_SEC * SAMPLING_RATE)  # 256 samples

INTERICTAL_STRIDE_SEC: float = 4.0  # seconds (non-overlapping)
INTERICTAL_STRIDE_SAMPLES: int = int(INTERICTAL_STRIDE_SEC * SAMPLING_RATE)  # 1024 samples

OVERLAP_THRESHOLD: float = 0.5  # >= 50% overlap required for positive label (y=1)
PRE_ICTAL_GUARD_SEC: float = 60.0  # seconds before seizure onset excluded from background
POST_ICTAL_GUARD_SEC: float = 60.0  # seconds after seizure offset excluded from background

# Prototype cohort for pipeline validation and unit tests
PROTOTYPE_COHORT: List[str] = ["chb01", "chb03", "chb05", "chb07", "chb08"]

# Mapping from case folder to unique biological patient ID
# Crucial: chb01 and chb21 are the same biological subject
PATIENT_ID_MAPPING: Dict[str, str] = {
    "chb01": "patient_01",
    "chb02": "patient_02",
    "chb03": "patient_03",
    "chb04": "patient_04",
    "chb05": "patient_05",
    "chb06": "patient_06",
    "chb07": "patient_07",
    "chb08": "patient_08",
    "chb09": "patient_09",
    "chb10": "patient_10",
    "chb11": "patient_11",
    "chb12": "patient_12",
    "chb13": "patient_13",
    "chb14": "patient_14",
    "chb15": "patient_15",
    "chb16": "patient_16",
    "chb17": "patient_17",
    "chb18": "patient_18",
    "chb19": "patient_19",
    "chb20": "patient_20",
    "chb21": "patient_01",  # Same individual as chb01 recorded 1.5 years later
    "chb22": "patient_22",
    "chb23": "patient_23",
    "chb24": "patient_24",  # Added Dec 2010
}
