# Research & ML Engineering Plan: Scalp EEG Epileptic Seizure Detection
**Project**: BTech CSE Honors Project  
**Dataset**: CHB-MIT Scalp EEG Database (PhysioNet v1.0.0)  
**Hardware Target**: NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) on WSL2  
**Date**: October 2026  
**Status**: Reconnaissance Phase Completed — Pending User Approval for Implementation

---

## 1. Executive Summary & Objective

The objective of this project is to build an automated, robust, and reproducible machine learning pipeline for detecting epileptic seizures from multi-channel scalp EEG recordings using the CHB-MIT Scalp EEG Database. 

During the initial reconnaissance phase, the dataset directory, file integrity, annotations, hardware environment, and literature conventions were thoroughly investigated. All dataset statistics and system configurations below have been directly inspected and verified from local files.

---

## 2. Environment & Hardware Inventory

### 2.1 Hardware Resources
* **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU
  * **Dedicated VRAM**: 4,096 MiB (4.0 GB)
  * **Driver / CUDA API**: Driver Version 615.65.07, CUDA driver API 13.4 supported
* **OS / Host**: Linux (WSL2 on Ubuntu 22.04 LTS), kernel 6.6.x

### 2.2 Python Environment & Package Audit
* **Python Executable**: `/usr/bin/python3` (v3.10.12)
* **Currently Installed & Available**:
  * `mne` (1.12.1) — Fully operational for EDF reading and signal processing
  * `numpy` (1.26.4), `scipy` (1.15.3), `pandas` (1.3.5)
  * `matplotlib` (3.10.9), `numba` (0.55.1), `tqdm` (4.70.1)
* **Packages Not Yet Installed**:
  * `scikit-learn` (Required for baseline classifiers, CV splitters, and metrics)
  * `torch` / `torchaudio` (Required for deep neural network architectures)
  * `lightgbm` / `xgboost` (Recommended for fast gradient boosted baseline)
* **Action Required (Pre-requisite for Phase 2)**:
  * In accordance with safety rules, no packages were installed. A clean virtual environment (`venv`) or pip installation for `scikit-learn` and CUDA-compatible `torch` will be requested prior to model training.

---

## 3. Dataset Reconnaissance & Verified Facts

### 3.1 Directory Structure & File Counts
* **Dataset Location**: `/mnt/d/College/Honors/chb-mit-scalp-eeg-database-1.0.0`
* **Total Size on Disk**: ~43 GB
* **Total EDF Recordings**: 686 files across 24 subject directories (`chb01` to `chb24`), verified matching `RECORDS`. All 686 files are present and readable.
* **Seizure-Containing Files**: 141 files, verified matching `RECORDS-WITH-SEIZURES`.
* **Total Recording Duration**: ~983 hours.

### 3.2 Subject Demographics & Critical Relationships
* `SUBJECT-INFO` details 23 cases (`chb01` to `chb23`):
  * **Age range**: 1.5 years to 22 years (pediatric cohort).
  * **Gender**: 5 Males (chb02, 04, 08, 10, 15), 17 Females (chb01, 03, 05, 06, 07, 09, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23).
* **Crucial Identity Link (`chb01` & `chb21`)**: 
  * As documented by PhysioNet, case `chb21` was recorded from the **same female patient** as `chb01` 1.5 years later.
  * **Design rule**: `chb01` and `chb21` must never be split across train and test sets in cross-subject evaluation.
* **Case `chb24`**: 
  * Added to PhysioNet in December 2010 post initial publication.
  * Contains 22 recordings and 16 seizures; demographic data is absent from `SUBJECT-INFO`.
  * `chb24-summary.txt` documents only files with seizures; all 16 seizures are verified.

### 3.3 Sampling Rate & Seizure Annotation Format
* **Sampling Frequency ($f_s$)**: Uniformly **256 Hz** across all recordings.
* **Seizure Annotations**:
  * Documented in per-patient `chbXX-summary.txt` files and companion binary `.edf.seizures` files.
  * Summaries provide exact integer seconds from recording start: `Seizure Start Time: X seconds`, `Seizure End Time: Y seconds`.
  * Sample alignment: `start_sample = int(X * 256)`, `end_sample = int(Y * 256)`.
* **Verified Seizure Statistics**:
  * **Total Seizure Events**: 198 seizures across all 24 subjects.
  * **Total Seizure (Ictal) Duration**: 11,611 seconds (~3.23 hours).
  * **Duration Range**: Minimum 6.0 s, Maximum 752.0 s (~12.5 minutes).
  * **Mean Duration**: 58.64 s (Median: 46.0 s).
* **Extreme Class Imbalance**:
  * Total seizure duration represents **~0.33%** of all recorded EEG data.
  * Class ratio is approximately **1 : 300** (Ictal vs. Inter-ictal).

### 3.4 Montage & Channel Inconsistencies
* Files have variable channel counts (ranging from 23 to 28 channels, with up to 153 channel listings in patients with montage modifications).
* Several files include non-EEG signals (`ECG`, `VNS`) or spacer dummy channels labeled `'-'` and `'.'`.
* Duplicate channel label quirk: In multiple files, channel `T8-P8` appears twice, causing MNE to label them `T8-P8-0` and `T8-P8-1`.
* **Standard 18-Channel Bipolar Derivation**:
  In accordance with clinical standards (longitudinal bipolar / double banana montage) and benchmark literature, the following 18 common bipolar channels are present across all subjects:
  1. **Left Temporal**: `FP1-F7`, `F7-T7`, `T7-P7`, `P7-O1`
  2. **Left Parasagittal**: `FP1-F3`, `F3-C3`, `C3-P3`, `P3-O1`
  3. **Right Parasagittal**: `FP2-F4`, `F4-C4`, `C4-P4`, `P4-O2`
  4. **Right Temporal**: `FP2-F8`, `F8-T8`, `T8-P8`, `P8-O2`
  5. **Midline / Central**: `FZ-CZ`, `CZ-PZ`
* Standardizing every file to this exact 18-channel subset eliminates montage discrepancy.

---

## 4. Methodological Rigor & Data Leakage Prevention

Data leakage is the most widespread source of invalidly optimistic claims in published EEG literature. We establish strict structural barriers against all forms of leakage:

### 4.1 Pitfalls of Random Window Splitting
* **Temporal Overlap Leakage**: If an EEG record is sliced into 4-second windows with 75% overlap (1-second step) and randomly shuffled into train/test sets, adjacent windows share 3 seconds of identical raw waveform. The test set evaluates memorization of signal autocorrelation rather than seizure pattern generalization.
* **Patient Identity Leakage**: Scalp EEG exhibits distinct baseline individual fingerprints (skull thickness, electrode resistance, alpha frequency, baseline impedance). A model trained and tested on windows from the same patient learns to recognize the patient rather than seizures.
* **Rule**: Random train/test splitting at the window level is **strictly prohibited**.

### 4.2 Patient-Independent vs. Within-Patient Paradigms

```
                     ┌──────────────────────────────────────────────┐
                     │          Evaluation Paradigms                │
                     └──────────────────────┬───────────────────────┘
                                            │
               ┌────────────────────────────┴───────────────────────────┐
               ▼                                                        ▼
┌───────────────────────────────┐                       ┌───────────────────────────────┐
│   Patient-Independent (LOSO)  │                       │   Within-Patient (Specific)   │
├───────────────────────────────┤                       ├───────────────────────────────┤
│ • Cross-Subject Generalization│                       │ • Personalized Tuning         │
│ • Train on N-1 Subjects       │                       │ • Chronological/Session Split │
│ • Test on Unseen Subject      │                       │ • Train on Earliest Seizures  │
│ • chb01 + chb21 Kept Together │                       │ • Test on Future Seizures     │
│ • Simulates Zero-Shot Clinical│                       │ • Simulates Adaptive Closed-  │
│   Deployment to New Patients  │                       │   Loop Neurostimulator        │
└───────────────────────────────┘                       └───────────────────────────────┘
```

1. **Patient-Independent (Leave-One-Subject-Out / Group K-Fold)**:
   * **Setup**: Train on $M-1$ patients, evaluate exclusively on the remaining held-out patient.
   * **Grouping**: Grouped by subject ID, explicitly binding `chb01` and `chb21` into the same fold.
   * **Clinical value**: Reflects real-world diagnostic software screening new patients without manual calibration.
2. **Within-Patient (Patient-Specific Chronological Split)**:
   * **Setup**: If evaluating individualized detection, splitting must be **strictly chronological by whole recording session** (e.g., first 70% of recordings with seizures for training, last 30% for testing).
   * **Constraint**: A temporal buffer (guard band) of at least 30 minutes must separate train and test sessions.

---

## 5. End-to-End Pipeline Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  MNE EDF Reader │────▶│ Standardize 18   │────▶│ Filtering        │────▶│ Sliding Window   │
│ (preload=False) │     │ Bipolar Channels │     │ (0.5 - 45 Hz)    │     │ Slicing (4s)     │
└─────────────────┘     └──────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                                                   │
                                                                                   ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Evaluation      │◀────│ Lightweight      │◀────│ Feature / Tensor │◀────│ Class Imbalance  │
│ PR-AUC, FP/h    │     │ Classifier       │     │ Extraction       │     │ Stratification   │
└─────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
```

### 5.1 Step 1: Memory-Safe EDF Loading
* The 43 GB dataset cannot be loaded into RAM at once.
* Use `mne.io.read_raw_edf(path, preload=False)` to parse headers in milliseconds.
* Slice target intervals on-demand (`raw.get_data(start=..., stop=...)`), releasing memory after extraction.

### 5.2 Step 2: Channel Selection & Preprocessing
* Select the canonical 18 channels. Handle the `T8-P8` duplicate gracefully by selecting the first occurrence.
* Apply a zero-phase bandpass filter: **0.5 Hz – 45.0 Hz** (order 4 Butterworth or FIR), removing low-frequency movement drift and high-frequency powerline/EMG noise.
* Per-channel z-score standardization ($z = \frac{x - \mu}{\sigma}$) computed per recording to account for day-to-day impedance shifts.

### 5.3 Step 3: Windowing & Label Assignment
* **Window Length ($W$)**: 4.0 seconds (1024 samples at 256 Hz) — standard temporal resolution capturing multiple seizure oscillation cycles.
* **Stride ($S$)**:
  * Training: 1.0 second (75% overlap) for seizure segments to capture phase variations.
  * Inference / Testing: 1.0 or 2.0 seconds sequential sliding window.
* **Label Rule**:
  * $y = 1$ (Seizure): Window overlaps annotated seizure boundary by $\ge 50\%$.
  * $y = 0$ (Background): Window has $0\%$ overlap with seizure, positioned at least 60 seconds outside any seizure boundary (guard band).

### 5.4 Step 4: Class Imbalance Strategy
* Seizure data is scarce (~3.23 hours), while background data is vast (~980 hours).
* Full background extraction yields millions of negative windows, exhausting RAM and compute.
* **Strategy**:
  * Inter-ictal under-sampling: Sample background segments from both pre-seizure baseline recordings and pure non-seizure files at a controlled ratio (e.g., 10:1 or 20:1 background-to-seizure ratio).
  * Cost-sensitive learning: Apply class weights inversely proportional to class frequencies (`class_weight='balanced'`).
  * Focal loss or custom threshold tuning during decision calibration.

---

## 6. Model Design & 4 GB VRAM Optimization

### 6.1 Phase 1: Lightweight Classical Baseline (Recommended First)
Before deploying GPU deep learning, an interpretable, resource-efficient classical ML baseline will be built:

* **Feature Extraction (18 channels $\times$ 14 features = 252 features per window)**:
  * *Time-Domain*: Mean, Variance, Skewness, Kurtosis, Zero-Crossing Rate, Line Length (total variation), Root Mean Square (RMS).
  * *Frequency-Domain (FFT / Welch PSD)*: Relative band power in Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–12 Hz), Beta (12–30 Hz), Gamma (30–45 Hz), Spectral Entropy, Spectral Edge Frequency (95%).
* **Classifier**:
  * **LightGBM / XGBoost** or **Random Forest** with balanced weights.
  * Linear SVM / Logistic Regression as an absolute lower bound.
* **Advantages for 4 GB VRAM Constraint**:
  * Requires 0 MB of VRAM (runs purely on multi-core CPU).
  * Train and cross-validate across multiple subjects in minutes.
  * Provides feature importance rankings to validate electrophysiological plausibility (e.g., rhythmic high-frequency power spikes).

### 6.2 Phase 2: Lightweight Deep Learning Architecture (GPU Target)
When progressing to neural architectures on the RTX 3050 (4 GB VRAM):

* **Model Choice**: **EEGNet** (Lawhern et al.) or a compact **1D-CNN + Temporal Convolutional Network (TCN)**.
  * Spatial depthwise separable convolutions to isolate spatial inter-channel patterns followed by temporal convolutions.
  * Parameter budget: $< 50,000$ parameters (model footprint $< 2$ MB).
* **VRAM Safeguards**:
  * Batch size: 32 or 64 windows of shape `(Batch, 18, 1024)`.
  * Mixed precision (`torch.cuda.amp.autocast`) to halve memory consumption and accelerate tensor operations on RTX 3050 Ampere tensor cores.
  * Memory footprint during training: $< 1.2$ GB VRAM, safely below the 4 GB ceiling.

---

## 7. Evaluation Metrics

Because background windows constitute $>99\%$ of continuous EEG, raw classification accuracy is deceptive and clinically meaningless (a trivial model predicting "no seizure" achieves $99.6\%$ accuracy).

The following primary and secondary metrics are adopted:

| Metric | Purpose & Clinical Interpretation |
| :--- | :--- |
| **PR-AUC (Average Precision)** | **Primary Metric**: Evaluates trade-off between precision and recall across all thresholds under severe class imbalance. |
| **Sensitivity (Recall)** | Percentage of seizure windows (or seizure events) correctly identified ($\ge 90\%$ target). |
| **Specificity** | Percentage of non-seizure windows correctly rejected. |
| **False Positive Rate per Hour (FP/h)** | **Clinical Metric**: Number of false seizure alarms per hour of continuous EEG (vital for alert fatigue prevention). |
| **F1-Score (Seizure Class)** | Harmonic mean of precision and recall for the minority class. |
| **ROC-AUC** | Area under receiver operating curve (reported for literature comparison). |

* **Event-Level vs. Window-Level Evaluation**:
  * In addition to window-level predictions, a post-processing collar filter (e.g., requiring 3 consecutive positive windows within a 5-second interval) will be applied to compute **event-based sensitivity** and **FP/h**.

---

## 8. Assumptions, Caveats & Unresolved Questions

### 8.1 Assumptions
1. The 18 standard bipolar channels provide sufficient spatial representation for the majority of focal and generalized seizure onsets in this cohort.
2. The integer-second seizure annotations in `chbXX-summary.txt` represent electrographic seizure onsets and terminations reviewed by clinical experts.
3. Pediatric EEG signals across different ages (1.5 to 22 years) exhibit significant background frequency variation (e.g., slower posterior dominant rhythm in infants), reinforcing the necessity of patient-stratified evaluation.

### 8.2 Unresolved Questions for Discussion
1. **Evaluation Focus**: Should the honors project prioritize **Patient-Independent (LOSO)** detection (higher difficulty, greater research contribution) or **Within-Patient** personalized detection (higher numerical metrics, suitable for implantable device design)? *(Recommendation: Implement LOSO as the main contribution, with a within-patient comparison).*
2. **Subset vs. Full Cohort for Initial Iterations**: To enable rapid prototyping, should Phase 1 validate on a representative 5-patient benchmark subset (e.g., chb01, chb03, chb05, chb07, chb08) before running the full 24-subject leave-one-out loop?
3. **Python Environment Setup**: Would you prefer a dedicated Python virtual environment (`python3 -m venv honors_env`) in the workspace, or installing missing libraries (`scikit-learn`, `torch`) in the existing user environment?

---

## 9. Next Steps (Pending Approval)

* [ ] **Step 1: Environment Setup**: Establish virtual environment and install `scikit-learn` and CUDA-enabled `torch`.
* [ ] **Step 2: Metadata & Indexing Utility**: Build a lightweight parser that parses all 24 summary files into a structured Pandas dataframe / manifest without touching raw EDF signals.
* [ ] **Step 3: Signal Preprocessing & Windowing Module**: Implement memory-safe chunk loading, 18-channel alignment, bandpass filtering, and balanced window sampling.
* [ ] **Step 4: Baseline Feature Extractor & Classical Classifier**: Implement time/frequency feature extractor and evaluate Random Forest / LightGBM under patient-grouped cross-validation.
* [ ] **Step 5: Metric Reporting & Visualization**: Generate PR curves, ROC curves, confusion matrices, and FP/h reports.
* [ ] **Step 6: Deep Learning Exploration (EEGNet / 1D-CNN)**: Once the baseline benchmark is secured, train lightweight PyTorch models with mixed precision on the RTX 3050.
