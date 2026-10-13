"""Evaluation metrics and assessment subpackage."""

from .metrics import compute_event_metrics, compute_window_metrics
from .postprocessing import consecutive_window_persistence, ewma_smooth, tune_ewma_alpha_leak_free

__all__ = [
    "compute_window_metrics",
    "compute_event_metrics",
    "ewma_smooth",
    "consecutive_window_persistence",
    "tune_ewma_alpha_leak_free",
]

