"""Execution entry point for full-cohort 1D CNN deep learning experiment."""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.deep_learning_cnn_experiment import DeepLearningCNNExperiment

def main():
    print("=" * 80)
    print("HONORS EEG: DEEP LEARNING 1D CNN FULL-COHORT EXPERIMENT")
    print("=" * 80)

    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_avail}")
        if cuda_avail:
            print(f"Primary GPU: {torch.cuda.get_device_name(0)}")
            print(f"Device Capability: {torch.cuda.get_device_capability(0)}")
            print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        else:
            print("WARNING: CUDA is not available. Training will fall back to CPU.")
    except ImportError:
        print("ERROR: PyTorch is not installed in the current environment.")
        sys.exit(1)

    exp = DeepLearningCNNExperiment(
        experiment_id="exp_20261013_deep_learning_cnn",
        batch_size=64,
        max_epochs=20,
        learning_rate=1e-3,
        weight_decay=1e-4,
        base_filters=16,
        dropout_rate=0.25,
        patience=5,
        random_seed=42,
    )

    # 1. Run Smoke Test
    print("\n[Step 1/2] Executing Hardware & Model Smoke Test...")
    smoke_results = exp.run_smoke_test()
    print("Smoke test output:", json.dumps(smoke_results, indent=2))
    assert smoke_results["status"] == "PASSED"

    # 2. Run Full 23-Fold Evaluation
    print("\n[Step 2/2] Launching 23-Fold Leave-One-Patient-Out Evaluation...")
    t_start = time.time()
    case_df, patient_df, summary_df = exp.run_full_cohort_evaluation()
    t_end = time.time()

    print(f"\nExperiment finished in {(t_end - t_start) / 60.0:.2f} minutes.")
    print("Summary results:")
    print(summary_df.to_string())

if __name__ == "__main__":
    main()
