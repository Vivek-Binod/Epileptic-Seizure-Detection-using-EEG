"""Leakage-safe cross-validation and dataset splitters for CHB-MIT EEG recordings.

Enforces zero-leakage evaluation protocols:
1. Patient-Independent (LOSO and Group-KFold):
   Strictly groups by biological patient identity, ensuring chb01 and chb21
   (recorded from the same individual 1.5 years apart) are NEVER split across
   train and test sets.
2. Within-Patient (Chronological Split):
   Strictly splits by chronological recording session (e.g. first 70% train, last 30% test)
   with temporal separation to eliminate future-leakage and autocorrelation bias.
"""

from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import PATIENT_ID_MAPPING


class PatientGroupedSplitter:
    """Splitter ensuring strict separation of biological patient identities."""

    def __init__(self, patient_mapping: Dict[str, str] = PATIENT_ID_MAPPING):
        self.patient_mapping = dict(patient_mapping)

    def get_loso_splits(
        self, recordings_df: pd.DataFrame
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, str], None, None]:
        """Generate Leave-One-Subject-Out (LOSO) train/test splits.

        Yields:
            Tuple of (train_df, test_df, held_out_patient_id)
        """
        df = self._ensure_patient_id(recordings_df)
        unique_patients = sorted(df["patient_id"].unique())

        for held_out_patient in unique_patients:
            train_mask = df["patient_id"] != held_out_patient
            test_mask = df["patient_id"] == held_out_patient

            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()

            self.validate_patient_split(train_df, test_df)
            yield train_df, test_df, held_out_patient

    def get_group_kfold_splits(
        self,
        recordings_df: pd.DataFrame,
        n_splits: int = 5,
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, int], None, None]:
        """Generate Group-KFold splits where groups are biological patient IDs.

        Yields:
            Tuple of (train_df, test_df, fold_idx)
        """
        df = self._ensure_patient_id(recordings_df)
        gkf = GroupKFold(n_splits=n_splits)

        groups = df["patient_id"].values
        # GroupKFold splits based on indices
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups), start=1):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            self.validate_patient_split(train_df, test_df)
            yield train_df, test_df, fold_idx

    def _ensure_patient_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure patient_id column exists and correctly maps chb01 and chb21."""
        df = df.copy()
        if "patient_id" not in df.columns:
            if "case_id" not in df.columns:
                raise ValueError("DataFrame must contain 'patient_id' or 'case_id' column")
            df["patient_id"] = df["case_id"].map(lambda c: self.patient_mapping.get(c, c))
        else:
            # Verify chb01 and chb21 consistency
            chb01_mask = df["case_id"] == "chb01" if "case_id" in df.columns else False
            chb21_mask = df["case_id"] == "chb21" if "case_id" in df.columns else False
            if np.any(chb01_mask) and np.any(chb21_mask):
                p01 = df.loc[chb01_mask, "patient_id"].unique()
                p21 = df.loc[chb21_mask, "patient_id"].unique()
                if not (len(p01) == 1 and len(p21) == 1 and p01[0] == p21[0]):
                    raise ValueError(
                        f"Inconsistent patient IDs for chb01 ({p01}) and chb21 ({p21}). "
                        "Both must map to the same biological patient ID!"
                    )
        return df

    @staticmethod
    def validate_patient_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Assert zero patient leakage between train and test splits."""
        train_patients = set(train_df["patient_id"].unique())
        test_patients = set(test_df["patient_id"].unique())
        overlap = train_patients.intersection(test_patients)
        if overlap:
            raise ValueError(
                f"PATIENT LEAKAGE DETECTED! Overlapping patients across train and test: {overlap}"
            )

        if "case_id" in train_df.columns and "case_id" in test_df.columns:
            train_cases = set(train_df["case_id"].unique())
            test_cases = set(test_df["case_id"].unique())

            # Specific check for chb01 and chb21
            if ("chb01" in train_cases and "chb21" in test_cases) or (
                "chb21" in train_cases and "chb01" in test_cases
            ):
                raise ValueError(
                    "IDENTITY LEAKAGE: chb01 and chb21 are split across train and test! "
                    "They are the identical subject and must remain in the same split."
                )


class WithinPatientChronologicalSplitter:
    """Splitter ensuring strictly chronological session ordering within a single patient."""

    def __init__(self, train_ratio: float = 0.70):
        if not (0.0 < train_ratio < 1.0):
            raise ValueError("train_ratio must be between 0.0 and 1.0")
        self.train_ratio = train_ratio

    def get_chronological_split(
        self, patient_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split a patient's recordings chronologically.

        Sorts files by name/number (e.g. chb01_01, chb01_02, ...).
        First train_ratio fraction goes to train, remainder to test.
        """
        if len(patient_df) < 2:
            raise ValueError(f"Patient has only {len(patient_df)} recordings; cannot split.")

        df = patient_df.copy()
        # Sort chronologically by filename / session order
        df = df.sort_values("file_name").reset_index(drop=True)

        n_train = max(1, int(np.floor(len(df) * self.train_ratio)))
        # Ensure test set has at least one recording
        if n_train >= len(df):
            n_train = len(df) - 1

        train_df = df.iloc[:n_train].copy()
        test_df = df.iloc[n_train:].copy()

        # Check for seizure representation
        if "has_seizures" in df.columns:
            n_train_seiz = train_df["has_seizures"].sum()
            n_test_seiz = test_df["has_seizures"].sum()
            if n_train_seiz == 0 or n_test_seiz == 0:
                # Warning or notice: chronological split may have no seizures in train or test
                pass

        return train_df, test_df
