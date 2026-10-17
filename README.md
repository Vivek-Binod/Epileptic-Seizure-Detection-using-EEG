# EEG Seizure Detection & Functional Connectivity Pipeline

An end-to-end, reproducible scalp EEG seizure detection and functional connectivity framework evaluated on the **CHB-MIT Scalp EEG Database** under a strict **23-fold Leave-One-Patient-Out (LOPO)** cross-validation protocol.

---

## Key Features

- **Standardized Preprocessing (`src/data/`):**
  - Canonical 18-channel longitudinal bipolar montage ("double banana").
  - 4th-order zero-phase Butterworth bandpass filtering ($0.5 - 45.0\text{ Hz}$).
  - 4.0-second sliding windows ($1024$ samples at $256\text{ Hz}$) with pre- and post-ictal guard collars.
- **Multimodal Feature Extraction (`src/features/`):**
  - **252 Conventional Features:** Time-domain metrics (RMS, variance, skewness, kurtosis, line length, zero-crossings) and Welch spectral powers (Delta, Theta, Alpha, Beta, Gamma, spectral entropy, edge 95%).
  - **200 Functional Connectivity Features:**
    - **112 Phase Locking Value (PLV):** Pairwise phase consistency across canonical bands.
    - **88 Phase Lag Index (PLI):** Zero-lag volume-conduction invariant phase asymmetry.
    - Nodal strength, global synchrony (mean, std, max), anatomical sub-networks (temporal, parasagittal), and inter-hemispheric homologous coupling.
  - **Graph Network Topology:** Weighted local clustering coefficient, global efficiency, and eigenvector centrality.
- **Models & Classifiers (`src/models/`):**
  - **Classical Machine Learning:** Random Forest, LightGBM, and Logistic Regression with leakage-free feature standardization.
  - **Deep Learning Benchmark:** Compact 1D CNN with depthwise-separable temporal convolutions and batch normalization.
- **Temporal Postprocessing (`src/evaluation/`):**
  - Exponentially Weighted Moving Average (EWMA, $\alpha=0.30$) for smoothing continuous prediction time series.
  - Collar-based event detection ($\ge 3$ consecutive windows above threshold with collar tolerance).
- **Strict LOPO Cross-Validation (`src/data/splitters.py`):**
  - 23 independent biological folds across all 24 CHB-MIT cases.
  - Cases `chb01` and `chb21` (recorded 1.5 years apart from the same individual) are strictly unified into a single biological test fold to prevent patient identity leakage.
  - Feature normalization and class weighting are strictly fitted inside training splits with zero test leakage.

---

## Repository Layout

```
.
├── data/
│   ├── recordings_manifest.csv      # Manifest of all CHB-MIT recordings
│   └── seizures_manifest.csv        # Clinically annotated seizure intervals
├── scripts/
│   ├── run_plv_pli_ablation.py      # Full 23-fold LOPO PLV vs. PLI ablation
│   ├── run_deep_learning_cnn.py     # 1D CNN training and evaluation
│   ├── run_postprocessing_ablation.py # Postprocessing comparison (raw, collar, EWMA)
│   ├── run_leak_free_postprocessing_selection.py # Leakage-free postprocessing validation
│   ├── cache_all_training_windows.py # Precomputes training window features
│   └── cache_all_eval_windows.py    # Precomputes continuous evaluation features
├── src/
│   ├── data/                        # EDF loaders, window generators, LOPO splitters
│   ├── features/                    # Unified, conventional, and connectivity extractors
│   ├── models/                      # Baseline ML classifiers and 1D CNN architecture
│   ├── evaluation/                  # Window/event metrics and postprocessing functions
│   ├── pipelines/                   # Reusable experiment pipelines
│   └── utils/                       # Configuration, channel montages, and constants
├── tests/                           # Complete pytest unit and integration test suite
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.10+
- Linux / macOS / WSL2
- Optional: NVIDIA GPU with CUDA 12.0+ for 1D CNN acceleration

### Setup Virtual Environment
```bash
python3 -m venv honors_env
source honors_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configuration

Update dataset paths in [`src/utils/config.py`](src/utils/config.py):
```python
DATASET_DIR = Path("/path/to/chb-mit-scalp-eeg-database-1.0.0")
WORKSPACE_DIR = Path("/path/to/Honors-EEG")
```

---

## Running Experiments

### 1. Precompute Feature Caches
Feature extraction can be precomputed and cached atomically in Parquet format:
```bash
# Cache training windows (balanced background/seizure ratio)
python scripts/cache_all_training_windows.py

# Cache continuous evaluation windows
python scripts/cache_all_eval_windows.py
```

### 2. Full-Cohort PLV vs. PLI Ablation (Classical Models)
Executes the 23-fold LOPO ablation evaluating Conventional (252), PLV (112), PLI (88), Connectivity (200), and Multimodal configurations (340, 364, 452 features):
```bash
python scripts/run_plv_pli_ablation.py
```
Outputs:
- Per-case evaluation results: `plv_pli_ablation_case_folds.csv`
- Per-patient evaluation results: `plv_pli_ablation_patient_folds.csv`
- Cohort summary: `plv_pli_ablation_summary.csv`

### 3. Deep Learning 1D CNN Benchmark
Trains and evaluates the compact 1D CNN directly on normalized raw multi-channel EEG windows:
```bash
python scripts/run_deep_learning_cnn.py
```

### 4. Postprocessing & Collar Ablation
Evaluates the impact of raw predictions, collar voting, and EWMA filtering:
```bash
python scripts/run_postprocessing_ablation.py
```

---

## Running Tests

Run the comprehensive pytest suite verifying loader properties, mathematical invariants, channel mappings, feature extraction, and split isolation:
```bash
pytest tests/
```

---

## License

This project is licensed under the MIT License.
