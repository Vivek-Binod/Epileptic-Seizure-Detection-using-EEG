"""Raw EEG Window Loader, Cache, and Dataset for Deep Learning Seizure Detection.

Constructs 18-channel, 1024-sample (4-second at 256 Hz) time-domain tensors directly
corresponding to the verified manifest and metadata from the feature cache.
Ensures 100% equivalence in window definitions, labels, and splits with the existing pipeline.
"""

from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import EDFLoader
from src.utils.config import DATASET_DIR, OUTPUTS_DIR, PATIENT_ID_MAPPING, SAMPLING_RATE

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


SUBJECT_PAIRS: Dict[str, Tuple[str, str]] = {
    "chb01": ("chb01_03.edf", "chb01_01.edf"),
    "chb02": ("chb02_16.edf", "chb02_01.edf"),
    "chb03": ("chb03_01.edf", "chb03_05.edf"),
    "chb04": ("chb04_05.edf", "chb04_01.edf"),
    "chb05": ("chb05_06.edf", "chb05_01.edf"),
    "chb06": ("chb06_01.edf", "chb06_02.edf"),
    "chb07": ("chb07_19.edf", "chb07_01.edf"),
    "chb08": ("chb08_02.edf", "chb08_03.edf"),
    "chb09": ("chb09_06.edf", "chb09_01.edf"),
    "chb10": ("chb10_12.edf", "chb10_01.edf"),
    "chb11": ("chb11_82.edf", "chb11_01.edf"),
    "chb12": ("chb12_06.edf", "chb12_19.edf"),
    "chb13": ("chb13_19.edf", "chb13_02.edf"),
    "chb14": ("chb14_03.edf", "chb14_01.edf"),
    "chb15": ("chb15_06.edf", "chb15_01.edf"),
    "chb16": ("chb16_10.edf", "chb16_01.edf"),
    "chb17": ("chb17a_03.edf", "chb17a_05.edf"),
    "chb18": ("chb18_29.edf", "chb18_01.edf"),
    "chb19": ("chb19_28.edf", "chb19_01.edf"),
    "chb20": ("chb20_12.edf", "chb20_01.edf"),
    "chb21": ("chb21_19.edf", "chb21_01.edf"),
    "chb22": ("chb22_20.edf", "chb22_01.edf"),
    "chb23": ("chb23_06.edf", "chb23_07.edf"),
    "chb24": ("chb24_01.edf", "chb24_02.edf"),
}


class RawWindowCache:
    """Manages raw (18, 1024) time-domain EEG window extraction and on-disk caching."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        feature_cache_dir: Optional[Path] = None,
        dataset_dir: Path = DATASET_DIR,
    ):
        if cache_dir is None:
            cache_dir = OUTPUTS_DIR / "experiments" / "exp_20261013_deep_learning_cnn" / "window_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if feature_cache_dir is None:
            feature_cache_dir = (
                OUTPUTS_DIR / "experiments" / "exp_20261012_sprint" / "feature_cache" / "fp_2dd1bac799f4"
            )
        self.feature_cache_dir = Path(feature_cache_dir)
        self.dataset_dir = Path(dataset_dir)
        self.loader = EDFLoader()
        self.subject_pairs = SUBJECT_PAIRS
        self.patient_mapping = PATIENT_ID_MAPPING

    def get_or_extract_training_windows(
        self, case_id: str, verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load or extract raw training windows for a case.

        Returns:
            Tuple of:
            - X: np.ndarray of shape (N, 18, 1024), float32
            - y: np.ndarray of shape (N,), int64
            - meta: dictionary of metadata (patient_id, case_id, file_names)
        """
        npz_path = self.cache_dir / f"{case_id}_train_windows.npz"
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            return data["X"], data["y"], data["meta"].item()

        if verbose:
            print(f"Extracting raw training windows for {case_id}...")

        seiz_fn, non_seiz_fn = self.subject_pairs[case_id]
        patient_id = self.patient_mapping.get(case_id, case_id)

        # 1. Load window definitions from existing Parquet feature cache
        case_cache_dir = self.feature_cache_dir / case_id
        tr_seiz_meta_path = case_cache_dir / f"{seiz_fn.replace('.edf', '')}_training.parquet"
        tr_ns_meta_path = case_cache_dir / f"{non_seiz_fn.replace('.edf', '')}_training_ns.parquet"

        meta_cols = ["start_sample", "end_sample", "start_sec", "end_sec", "label"]
        df_seiz = pd.read_parquet(tr_seiz_meta_path, columns=meta_cols)
        df_ns = pd.read_parquet(tr_ns_meta_path, columns=meta_cols)

        # 2. Slice from preprocessed full recordings
        X_list: List[np.ndarray] = []
        y_list: List[int] = []

        # Load seizure recording once
        seiz_rec_data, _ = self.loader.load_full_recording(
            self.dataset_dir / case_id / seiz_fn,
            apply_filter=True,
            apply_standardization=True,
        )
        for _, row in df_seiz.iterrows():
            s0 = int(row["start_sample"])
            s1 = int(row["end_sample"])
            X_list.append(seiz_rec_data[:, s0:s1].astype(np.float32))
            y_list.append(int(row["label"]))

        # Load non-seizure recording once (only slice up to 300s needed)
        ns_rec_data, _ = self.loader.load_full_recording(
            self.dataset_dir / case_id / non_seiz_fn,
            apply_filter=True,
            apply_standardization=True,
        )
        for _, row in df_ns.iterrows():
            s0 = int(row["start_sample"])
            s1 = int(row["end_sample"])
            X_list.append(ns_rec_data[:, s0:s1].astype(np.float32))
            y_list.append(int(row["label"]))

        X = np.stack(X_list, axis=0)  # (N, 18, 1024)
        y = np.array(y_list, dtype=np.int64)

        meta = {
            "case_id": case_id,
            "patient_id": patient_id,
            "n_samples": len(y),
            "n_pos": int(np.sum(y == 1)),
            "n_neg": int(np.sum(y == 0)),
        }

        # Cache to disk
        np.savez_compressed(npz_path, X=X, y=y, meta=meta)
        if verbose:
            print(f"Cached {len(y)} training windows for {case_id} to {npz_path.name}")

        return X, y, meta

    def get_or_extract_evaluation_windows(
        self, case_id: str, verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load or extract raw continuous evaluation windows for a case.

        Returns:
            Tuple of:
            - X: np.ndarray of shape (N, 18, 1024), float32
            - y: np.ndarray of shape (N,), int64
            - meta: dictionary of metadata (annotated seizures, duration)
        """
        npz_path = self.cache_dir / f"{case_id}_eval_windows.npz"
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            return data["X"], data["y"], data["meta"].item()

        if verbose:
            print(f"Extracting raw evaluation windows for {case_id}...")

        seiz_fn, _ = self.subject_pairs[case_id]
        patient_id = self.patient_mapping.get(case_id, case_id)

        case_cache_dir = self.feature_cache_dir / case_id
        ev_meta_path = case_cache_dir / f"{seiz_fn.replace('.edf', '')}_evaluation.parquet"
        meta_cols = ["start_sample", "end_sample", "start_sec", "end_sec", "label"]
        df_ev = pd.read_parquet(ev_meta_path, columns=meta_cols)

        seiz_rec_data, info = self.loader.load_full_recording(
            self.dataset_dir / case_id / seiz_fn,
            apply_filter=True,
            apply_standardization=True,
        )

        X_list: List[np.ndarray] = []
        y_list: List[int] = []

        for _, row in df_ev.iterrows():
            s0 = int(row["start_sample"])
            s1 = int(row["end_sample"])
            X_list.append(seiz_rec_data[:, s0:s1].astype(np.float32))
            y_list.append(int(row["label"]))

        X = np.stack(X_list, axis=0)  # (N, 18, 1024)
        y = np.array(y_list, dtype=np.int64)

        meta = {
            "case_id": case_id,
            "patient_id": patient_id,
            "file_name": seiz_fn,
            "duration_sec": float(len(y)),
            "duration_hours": float(len(y) / 3600.0),
            "n_samples": len(y),
            "n_pos": int(np.sum(y == 1)),
            "n_neg": int(np.sum(y == 0)),
        }

        np.savez_compressed(npz_path, X=X, y=y, meta=meta)
        if verbose:
            print(f"Cached {len(y)} evaluation windows for {case_id} to {npz_path.name}")

        return X, y, meta


if TORCH_AVAILABLE:
    class EEGWindowDataset(Dataset):
        """PyTorch Dataset for (18, 1024) EEG window tensors."""

        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.from_numpy(X).float()
            self.y = torch.from_numpy(y).float()

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.X[idx], self.y[idx]
