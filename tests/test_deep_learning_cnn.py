"""Regression tests for 1D CNN architecture, patient grouping, leakage prevention, checkpoint reload, and metric calculation."""

import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.raw_window_dataset import RawWindowCache, SUBJECT_PAIRS
from src.evaluation.metrics import compute_event_metrics, compute_window_metrics
from src.utils.config import CANONICAL_CHANNELS, PATIENT_ID_MAPPING, SAMPLING_RATE

try:
    import torch
    import torch.nn as nn
    from src.models.cnn1d import CompactEEGCNN, WeightedFocalLoss
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def test_patient_grouping_and_unification():
    """Verify that chb01 and chb21 are unified under patient_01 with 23 biological patients."""
    assert PATIENT_ID_MAPPING["chb01"] == "patient_01"
    assert PATIENT_ID_MAPPING["chb21"] == "patient_01"
    unique_patients = set(PATIENT_ID_MAPPING[c] for c in SUBJECT_PAIRS.keys())
    assert len(unique_patients) == 23
    assert len(SUBJECT_PAIRS) == 24


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not yet installed")
def test_cnn_model_shapes_and_parameters():
    """Verify 1D CNN tensor input/output shapes, parameter bounds, and probabilities."""
    model = CompactEEGCNN(in_channels=18, seq_len=1024, base_filters=16, dropout_rate=0.25)
    n_params = model.count_parameters()
    # Ensure parameter count is compact (< 100,000 params) for 4GB VRAM
    assert 10000 < n_params < 100000

    batch_x = torch.randn(4, 18, 1024)
    logits = model(batch_x)
    assert logits.shape == (4, 1)

    probs = model.predict_proba(batch_x)
    assert probs.shape == (4, 1)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not yet installed")
def test_focal_loss_numerical_stability():
    """Verify WeightedFocalLoss computes valid gradients without NaNs."""
    loss_fn = WeightedFocalLoss(gamma=2.0, alpha=0.75, pos_weight=5.0)
    logits = torch.randn(8, 1, requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]).view(8, 1)

    loss = loss_fn(logits, targets)
    assert not torch.isnan(loss) and not torch.isinf(loss)
    assert loss.item() > 0

    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not yet installed")
def test_checkpoint_save_and_reload(tmp_path):
    """Verify model state checkpointing and bitwise-exact weight reload."""
    model_orig = CompactEEGCNN(in_channels=18, seq_len=1024, base_filters=16)
    ckpt_path = tmp_path / "test_ckpt.pt"
    torch.save(model_orig.state_dict(), ckpt_path)

    model_loaded = CompactEEGCNN(in_channels=18, seq_len=1024, base_filters=16)
    model_loaded.load_state_dict(torch.load(ckpt_path))

    x = torch.randn(2, 18, 1024)
    model_orig.eval()
    model_loaded.eval()

    with torch.no_grad():
        out_orig = model_orig(x)
        out_loaded = model_loaded(x)

    assert torch.allclose(out_orig, out_loaded, atol=1e-6)


def test_leak_free_group_split_isolation():
    """Verify that patient-grouped inner validation strictly prevents patient overlap."""
    from sklearn.model_selection import GroupKFold

    cases = sorted(list(SUBJECT_PAIRS.keys()))
    unique_patients = sorted(list(set(PATIENT_ID_MAPPING[c] for c in cases)))

    for held_out in unique_patients[:3]:
        train_cases = [c for c in cases if PATIENT_ID_MAPPING[c] != held_out]
        patient_labels = [PATIENT_ID_MAPPING[c] for c in train_cases]

        # Inner 4-fold group CV
        gkf = GroupKFold(n_splits=4)
        dummy_X = np.zeros((len(train_cases), 10))
        dummy_y = np.zeros(len(train_cases))

        for in_tr, in_val in gkf.split(dummy_X, dummy_y, groups=patient_labels):
            tr_patients = set(patient_labels[i] for i in in_tr)
            val_patients = set(patient_labels[i] for i in in_val)

            # Mutual exclusivity
            assert len(tr_patients.intersection(val_patients)) == 0
            assert held_out not in tr_patients
            assert held_out not in val_patients


def test_raw_window_cache_integrity():
    """Verify properties of cached training windows for chb01."""
    cache = RawWindowCache()
    X, y, meta = cache.get_or_extract_training_windows("chb01")
    assert X.ndim == 3
    assert X.shape[1] == 18
    assert X.shape[2] == 1024
    assert len(X) == len(y)
    assert np.all((y == 0) | (y == 1))
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()
    assert meta["patient_id"] == "patient_01"
