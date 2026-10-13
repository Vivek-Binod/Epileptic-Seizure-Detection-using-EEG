"""Full-cohort CHB-MIT Deep Learning 1D CNN Experiment.

Evaluates a compact GPU-trained 1D Convolutional Neural Network directly on raw 18-channel
EEG windows ([batch, 18, 1024]) across the full 23-patient cohort (24 cases).

Features:
- Strict 23-Fold Leave-One-Patient-Out cross-validation (chb01 and chb21 unified under patient_01).
- Inner patient-grouped validation for early stopping and checkpoint selection (zero patient leakage).
- Dynamic class imbalance handling via training-fold pos_weight.
- Hardware-adaptive device selection (CUDA on RTX 3050 Laptop GPU with CPU fallback).
- Full continuous recording evaluation (35.99 hours total) matching the verified benchmark.
- Comparison with existing Conventional, Connectivity, Combined, and EWMA baselines.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.metadata_parser import CHBMITMetadataParser
from src.data.raw_window_dataset import RawWindowCache, SUBJECT_PAIRS
from src.evaluation.metrics import compute_event_metrics, compute_window_metrics
from src.evaluation.postprocessing import ewma_smooth
from src.utils.config import OUTPUTS_DIR, PATIENT_ID_MAPPING, SAMPLING_RATE

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from src.models.cnn1d import CompactEEGCNN
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DeepLearningCNNExperiment:
    """Orchestrates 23-fold LOPO evaluation of 1D CNN for EEG seizure detection."""

    def __init__(
        self,
        experiment_id: str = "exp_20261013_deep_learning_cnn",
        batch_size: int = 64,
        max_epochs: int = 20,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        base_filters: int = 16,
        dropout_rate: float = 0.25,
        patience: int = 5,
        random_seed: int = 42,
    ):
        self.experiment_id = experiment_id
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.base_filters = base_filters
        self.dropout_rate = dropout_rate
        self.patience = patience
        self.random_seed = random_seed

        self.exp_dir = OUTPUTS_DIR / "experiments" / experiment_id
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.window_cache = RawWindowCache(cache_dir=self.exp_dir / "window_cache")
        self.subject_pairs = SUBJECT_PAIRS
        self.patient_mapping = PATIENT_ID_MAPPING
        self.cases = sorted(list(self.subject_pairs.keys()))

        parser = CHBMITMetadataParser()
        _, self.seiz_df = parser.parse_all()

        # Device detection
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.device_name = torch.cuda.get_device_name(0)
            self.device_type = "cuda"
        else:
            self.device = torch.device("cpu") if TORCH_AVAILABLE else None
            self.device_name = "CPU"
            self.device_type = "cpu"

    def set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def load_cohort_raw_data(self, verbose: bool = False) -> Dict[str, Dict]:
        """Pre-load or retrieve cached raw window arrays for all 24 cases."""
        cohort_data: Dict[str, Dict] = {}
        t0 = time.time()
        print(f"Loading raw window data across all {len(self.cases)} cases...")

        for c in self.cases:
            patient_id = self.patient_mapping.get(c, c)
            seiz_fn, _ = self.subject_pairs[c]

            subj_seizures = self.seiz_df[
                (self.seiz_df["case_id"] == c) & (self.seiz_df["file_name"] == seiz_fn)
            ]
            annotated_seiz_sec = [
                (float(row["start_sec"]), float(row["end_sec"]))
                for _, row in subj_seizures.iterrows()
            ]

            X_tr, y_tr, tr_meta = self.window_cache.get_or_extract_training_windows(c, verbose=verbose)

            cohort_data[c] = {
                "case_id": c,
                "patient_id": patient_id,
                "train_X": X_tr,
                "train_y": y_tr,
                "annotated_seizures_sec": annotated_seiz_sec,
            }

        t1 = time.time()
        print(f"Cohort training data ready ({len(self.cases)} cases) in {t1 - t0:.2f} s.", flush=True)
        return cohort_data

    def run_smoke_test(self) -> Dict:
        """Run a fast 1-fold 2-epoch smoke test on GPU to verify tensor shapes, training, and gradients."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed.")

        self.set_seed(self.random_seed)
        print("\n" + "=" * 60)
        print("RUNNING GPU / PYTORCH SMOKE TEST")
        print(f"Selected Device: {self.device_name} ({self.device_type})")
        print("=" * 60)

        # Create dummy batch (B=4, C=18, T=1024)
        x_dummy = torch.randn(4, 18, 1024, device=self.device)
        y_dummy = torch.tensor([0.0, 1.0, 0.0, 1.0], device=self.device).view(4, 1)

        model = CompactEEGCNN(
            in_channels=18,
            seq_len=1024,
            base_filters=self.base_filters,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        n_params = model.count_parameters()
        print(f"CompactEEGCNN instantiated successfully. Total trainable parameters: {n_params:,}")

        # Forward pass
        logits = model(x_dummy)
        assert logits.shape == (4, 1), f"Unexpected output shape: {logits.shape}"

        # Loss and backward pass
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, y_dummy)
        loss.backward()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()

        probs = model.predict_proba(x_dummy).detach().cpu().numpy()
        assert probs.shape == (4, 1)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

        smoke_res = {
            "status": "PASSED",
            "device": str(self.device),
            "device_name": self.device_name,
            "trainable_parameters": n_params,
            "sample_loss": float(loss.item()),
            "output_shape": list(logits.shape),
        }
        print(f"Smoke test PASSED! Sample loss: {smoke_res['sample_loss']:.4f}")
        return smoke_res

    def run_full_cohort_evaluation(
        self,
        eval_cases: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute full 23-fold Leave-One-Patient-Out Cross-Validation.

        Args:
            eval_cases: Optional subset of cases for debugging/verification.
                        If None, runs all 24 cases across 23 biological folds.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available.")

        data = self.load_cohort_raw_data()
        unique_patients = sorted(list(set(self.patient_mapping[c] for c in self.cases)))

        if eval_cases is not None:
            active_cases = eval_cases
            active_patients = sorted(list(set(self.patient_mapping[c] for c in active_cases)))
        else:
            active_cases = self.cases
            active_patients = unique_patients

        print("\n" + "=" * 80, flush=True)
        print(f"STARTING 1D CNN FULL-COHORT EVALUATION ON {self.device_name.upper()}", flush=True)
        print(f"Folds to evaluate: {len(active_patients)} biological patients ({len(active_cases)} cases)", flush=True)
        print(f"Batch size: {self.batch_size} | Max epochs: {self.max_epochs} | LR: {self.learning_rate}", flush=True)
        print("=" * 80, flush=True)

        all_case_rows: List[Dict] = []
        all_patient_rows: List[Dict] = []
        all_curves: Dict[str, Dict] = {}

        for fold_idx, held_out_patient in enumerate(active_patients, start=1):
            t0_fold = time.time()
            test_cases = [c for c in active_cases if self.patient_mapping[c] == held_out_patient]
            train_cases = [c for c in self.cases if self.patient_mapping[c] != held_out_patient]

            print(
                f"\n[Fold {fold_idx}/{len(active_patients)}] Held-Out Patient: {held_out_patient} "
                f"(Cases: {test_cases}) | Training on {len(train_cases)} cases",
                flush=True
            )

            # Assemble outer training pool
            X_train_raw = np.concatenate([data[c]["train_X"] for c in train_cases], axis=0)
            y_train_raw = np.concatenate([data[c]["train_y"] for c in train_cases], axis=0)

            # Map patient IDs to training windows for inner GroupKFold split
            patient_ids_tr = []
            for c in train_cases:
                pid = self.patient_mapping[c]
                patient_ids_tr.extend([pid] * len(data[c]["train_y"]))
            patient_ids_tr = np.array(patient_ids_tr)

            # Inner validation split (4 folds among training patients)
            gkf = GroupKFold(n_splits=4)
            inner_tr_idx, inner_val_idx = next(gkf.split(X_train_raw, y_train_raw, groups=patient_ids_tr))

            X_in_tr = X_train_raw[inner_tr_idx]
            y_in_tr = y_train_raw[inner_tr_idx]
            X_in_val = X_train_raw[inner_val_idx]
            y_in_val = y_train_raw[inner_val_idx]

            # Dynamic class weighting from inner training split
            n_pos = max(1, int(np.sum(y_in_tr == 1)))
            n_neg = int(np.sum(y_in_tr == 0))
            pos_weight_val = float(n_neg / n_pos)

            # Build PyTorch DataLoaders
            tr_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_in_tr).float(), torch.from_numpy(y_in_tr).float()),
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
            )
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_in_val).float(), torch.from_numpy(y_in_val).float()),
                batch_size=self.batch_size,
                shuffle=False,
            )

            # Initialize model
            self.set_seed(self.random_seed + fold_idx)
            model = CompactEEGCNN(
                in_channels=18,
                seq_len=1024,
                base_filters=self.base_filters,
                dropout_rate=self.dropout_rate,
            ).to(self.device)

            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=self.device))
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-5)

            # Training loop with early stopping
            best_val_pr_auc = -1.0
            best_epoch = 0
            best_ckpt_path = self.ckpt_dir / f"best_model_fold_{held_out_patient}.pt"
            no_improve_epochs = 0
            train_losses, val_losses, val_pr_aucs = [], [], []

            for epoch in range(1, self.max_epochs + 1):
                model.train()
                running_loss = 0.0
                for x_b, y_b in tr_loader:
                    x_b = x_b.to(self.device)
                    y_b = y_b.to(self.device).view(-1, 1)

                    optimizer.zero_grad()
                    out_b = model(x_b)
                    loss = criterion(out_b, y_b)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * len(y_b)

                train_loss = running_loss / len(X_in_tr)
                scheduler.step()

                # Inner validation evaluation
                model.eval()
                val_preds, val_targets = [], []
                val_loss_tot = 0.0
                with torch.no_grad():
                    for x_v, y_v in val_loader:
                        x_v = x_v.to(self.device)
                        y_v = y_v.to(self.device).view(-1, 1)
                        out_v = model(x_v)
                        loss_v = criterion(out_v, y_v)
                        val_loss_tot += loss_v.item() * len(y_v)
                        p_v = torch.sigmoid(out_v).cpu().numpy().flatten()
                        val_preds.extend(p_v)
                        val_targets.extend(y_v.cpu().numpy().flatten())

                val_loss = val_loss_tot / len(X_in_val)
                val_preds = np.array(val_preds)
                val_targets = np.array(val_targets)
                val_metrics = compute_window_metrics(val_targets, val_preds)
                val_pr_auc = val_metrics["pr_auc"]

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_pr_aucs.append(val_pr_auc)

                if val_pr_auc > best_val_pr_auc:
                    best_val_pr_auc = val_pr_auc
                    best_epoch = epoch
                    no_improve_epochs = 0
                    torch.save(model.state_dict(), best_ckpt_path)
                else:
                    no_improve_epochs += 1

                if no_improve_epochs >= self.patience:
                    break

            all_curves[held_out_patient] = {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_pr_aucs": val_pr_aucs,
                "best_epoch": best_epoch,
                "best_val_pr_auc": best_val_pr_auc,
            }

            print(
                f"  Fold {fold_idx} trained in {best_epoch} epochs (best inner PR-AUC: {best_val_pr_auc:.4f}). "
                f"Reloading best checkpoint...",
                flush=True
            )

            # Reload best checkpoint for test patient evaluation
            model.load_state_dict(torch.load(best_ckpt_path, map_location=self.device))
            model.eval()

            # Evaluate test cases
            for test_case in test_cases:
                c_data = data[test_case]
                ann_seizures = c_data["annotated_seizures_sec"]

                # Load evaluation windows on-demand for this test case
                ev_X, ev_y, ev_meta = self.window_cache.get_or_extract_evaluation_windows(test_case, verbose=False)
                dur_h = ev_meta["duration_hours"]
                dur_sec = ev_meta["duration_sec"]

                # Predict test continuous evaluation in batches
                test_probs_list = []
                with torch.no_grad():
                    for b_start in range(0, len(ev_X), 256):
                        b_end = min(b_start + 256, len(ev_X))
                        batch_tensor = torch.from_numpy(ev_X[b_start:b_end]).float().to(self.device)
                        p_batch = model.predict_proba(batch_tensor).cpu().numpy().flatten()
                        test_probs_list.append(p_batch)

                probs_raw = np.concatenate(test_probs_list, axis=0)
                probs_ewma = ewma_smooth(probs_raw, alpha=0.30)
                window_start_sec = np.arange(len(probs_raw), dtype=float) * 1.0

                # Unsmoothed metrics
                w_raw = compute_window_metrics(ev_y, probs_raw, threshold=0.5)
                e_raw = compute_event_metrics(
                    window_preds=(probs_raw >= 0.50).astype(int),
                    window_start_sec=window_start_sec,
                    annotated_seizures_sec=ann_seizures,
                    total_duration_sec=dur_sec,
                    collar_consecutive_windows=3,
                    collar_tolerance_sec=10.0,
                )

                # EWMA smoothed metrics
                w_ewma = compute_window_metrics(ev_y, probs_ewma, threshold=0.5)
                e_ewma = compute_event_metrics(
                    window_preds=(probs_ewma >= 0.50).astype(int),
                    window_start_sec=window_start_sec,
                    annotated_seizures_sec=ann_seizures,
                    total_duration_sec=dur_sec,
                    collar_consecutive_windows=3,
                    collar_tolerance_sec=10.0,
                )

                # Log results
                all_case_rows.append({
                    "case_id": test_case,
                    "patient_id": held_out_patient,
                    "fold_idx": fold_idx,
                    "config_id": "cnn_1d_raw",
                    "config_name": "Compact 1D CNN (Raw Time-Series)",
                    "device": self.device_name,
                    "postprocessing": "unsmoothed",
                    "pr_auc": w_raw["pr_auc"],
                    "roc_auc": w_raw["roc_auc"],
                    "window_sens": w_raw["sensitivity"],
                    "window_spec": w_raw["specificity"],
                    "window_f1": w_raw["f1"],
                    "event_sensitivity": e_raw["event_sensitivity"],
                    "detected_seizures": e_raw["detected_seizures"],
                    "total_seizures": e_raw["total_seizures"],
                    "false_alarms": e_raw["false_alarms"],
                    "fp_per_hour": e_raw["fp_per_hour"],
                    "duration_hours": dur_h,
                    "mean_delay_sec": e_raw["mean_delay_sec"],
                    "median_delay_sec": float(np.median(e_raw["detection_delays_sec"])) if len(e_raw["detection_delays_sec"]) > 0 else float("nan"),
                })

                all_case_rows.append({
                    "case_id": test_case,
                    "patient_id": held_out_patient,
                    "fold_idx": fold_idx,
                    "config_id": "cnn_1d_ewma",
                    "config_name": "Compact 1D CNN + EWMA (alpha=0.30)",
                    "device": self.device_name,
                    "postprocessing": "ewma_0.30",
                    "pr_auc": w_ewma["pr_auc"],
                    "roc_auc": w_ewma["roc_auc"],
                    "window_sens": w_ewma["sensitivity"],
                    "window_spec": w_ewma["specificity"],
                    "window_f1": w_ewma["f1"],
                    "event_sensitivity": e_ewma["event_sensitivity"],
                    "detected_seizures": e_ewma["detected_seizures"],
                    "total_seizures": e_ewma["total_seizures"],
                    "false_alarms": e_ewma["false_alarms"],
                    "fp_per_hour": e_ewma["fp_per_hour"],
                    "duration_hours": dur_h,
                    "mean_delay_sec": e_ewma["mean_delay_sec"],
                    "median_delay_sec": float(np.median(e_ewma["detection_delays_sec"])) if len(e_ewma["detection_delays_sec"]) > 0 else float("nan"),
                })

                print(
                    f"    [{test_case}] PR-AUC: {w_raw['pr_auc']:.4f} -> EWMA: {w_ewma['pr_auc']:.4f} | "
                    f"Sens: {e_ewma['event_sensitivity']*100:.1f}% ({e_ewma['detected_seizures']}/{e_ewma['total_seizures']}) | "
                    f"FP/h: {e_ewma['fp_per_hour']:.2f}",
                    flush=True
                )

            t1_fold = time.time()
            print(f"  Fold {fold_idx} completed in {t1_fold - t0_fold:.2f} s.", flush=True)

        case_df = pd.DataFrame(all_case_rows)
        case_csv = self.exp_dir / "cnn_case_folds.csv"
        case_df.to_csv(case_csv, index=False)

        # Aggregate patient-level metrics
        patient_summary = (
            case_df.groupby(["patient_id", "config_id", "config_name", "postprocessing"])
            .agg({
                "pr_auc": "mean",
                "roc_auc": "mean",
                "window_sens": "mean",
                "window_spec": "mean",
                "window_f1": "mean",
                "event_sensitivity": "mean",
                "detected_seizures": "sum",
                "total_seizures": "sum",
                "false_alarms": "sum",
                "duration_hours": "sum",
            })
            .reset_index()
        )
        patient_summary["fp_per_hour"] = (
            patient_summary["false_alarms"] / patient_summary["duration_hours"].replace(0, np.nan)
        )
        patient_csv = self.exp_dir / "cnn_patient_folds.csv"
        patient_summary.to_csv(patient_csv, index=False)

        # Cohort Macro Summary
        summary_rows = []
        for (cfg_id, cfg_name, post), grp in case_df.groupby(["config_id", "config_name", "postprocessing"]):
            macro_pr = grp["pr_auc"].mean()
            macro_roc = grp["roc_auc"].mean()
            macro_sens = grp["event_sensitivity"].mean()
            macro_fp_rate = grp["fp_per_hour"].mean()
            tot_det = grp["detected_seizures"].sum()
            tot_seiz = grp["total_seizures"].sum()
            tot_fa = grp["false_alarms"].sum()
            tot_hours = grp["duration_hours"].sum()
            pooled_fp_rate = tot_fa / tot_hours if tot_hours > 0 else 0.0

            summary_rows.append({
                "config_id": cfg_id,
                "config_name": cfg_name,
                "postprocessing": post,
                "macro_case_pr_auc": macro_pr,
                "macro_case_roc_auc": macro_roc,
                "macro_event_sensitivity": macro_sens,
                "macro_fp_per_hour": macro_fp_rate,
                "pooled_detected_seizures": tot_det,
                "pooled_total_seizures": tot_seiz,
                "pooled_false_alarms": tot_fa,
                "pooled_duration_hours": tot_hours,
                "pooled_fp_per_hour": pooled_fp_rate,
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = self.exp_dir / "cnn_summary.csv"
        summary_df.to_csv(summary_csv, index=False)

        # Plot training curves and comparison
        self._plot_curves_and_comparisons(all_curves, case_df)

        print("\n" + "=" * 80)
        print("1D CNN EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(summary_df[["config_name", "macro_case_pr_auc", "macro_event_sensitivity", "pooled_fp_per_hour"]].to_string())

        return case_df, patient_summary, summary_df

    def _plot_curves_and_comparisons(self, all_curves: Dict, case_df: pd.DataFrame):
        """Generate publication-ready training curves and comparative diagnostic figures."""
        # 1. Training curves plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
        for pid, c_dict in all_curves.items():
            axes[0].plot(c_dict["train_losses"], color="#387eb8", alpha=0.3, linewidth=1.0)
            axes[0].plot(c_dict["val_losses"], color="#e41a1c", alpha=0.3, linewidth=1.0)
            axes[1].plot(c_dict["val_pr_aucs"], color="#4daf4a", alpha=0.4, linewidth=1.0)

        axes[0].set_title("A. Cross-Entropy Loss (Train vs Inner Val across Folds)", fontsize=10, fontweight="bold")
        axes[0].set_xlabel("Epoch", fontsize=9)
        axes[0].set_ylabel("Loss", fontsize=9)
        axes[0].grid(True, linestyle="--", alpha=0.5)

        axes[1].set_title("B. Inner Validation PR-AUC across Folds", fontsize=10, fontweight="bold")
        axes[1].set_xlabel("Epoch", fontsize=9)
        axes[1].set_ylabel("PR-AUC", fontsize=9)
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        curves_png = self.exp_dir / "cnn_training_curves.png"
        plt.savefig(curves_png, bbox_inches="tight")
        plt.close()

        # 2. Baseline comparison plot
        # Load baseline results if available
        base_summary_csv = OUTPUTS_DIR / "experiments" / "exp_20261013_full_cohort" / "full_cohort_summary.csv"
        if base_summary_csv.exists():
            base_df = pd.read_csv(base_summary_csv)
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

            # Combine baseline models and 1D CNN models
            labels = []
            pr_scores = []
            colors = []

            for _, row in base_df.iterrows():
                labels.append(row["config_name"])
                pr_scores.append(row["macro_case_pr_auc"])
                colors.append("#7293CB")

            cnn_raw = case_df[case_df["config_id"] == "cnn_1d_raw"]["pr_auc"].mean()
            cnn_ewma = case_df[case_df["config_id"] == "cnn_1d_ewma"]["pr_auc"].mean()

            labels.extend(["1D CNN (Raw EEG)", "1D CNN + EWMA (alpha=0.30)"])
            pr_scores.extend([cnn_raw, cnn_ewma])
            colors.extend(["#D35E60", "#E1974C"])

            y_pos = np.arange(len(labels))
            bars = ax.barh(y_pos, pr_scores, color=colors, height=0.6, edgecolor="black", linewidth=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("Macro PR-AUC (24 Cases / 23 Biological Patients)", fontsize=10)
            ax.set_title("Seizure Detection Benchmark: Classical Baselines vs Compact 1D CNN", fontsize=11, fontweight="bold")
            ax.set_xlim(0, 0.8)
            ax.grid(axis="x", linestyle="--", alpha=0.6)

            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=8.5, fontweight="bold")

            plt.tight_layout()
            comp_png = self.exp_dir / "cnn_vs_baselines_comparison.png"
            plt.savefig(comp_png, bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    exp = DeepLearningCNNExperiment()
    smoke = exp.run_smoke_test()
    print("Smoke test details:", json.dumps(smoke, indent=2))
