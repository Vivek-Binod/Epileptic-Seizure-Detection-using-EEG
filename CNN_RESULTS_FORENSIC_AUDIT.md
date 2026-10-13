# Forensic Audit of Deep Learning 1D CNN Benchmark (Commit `aa9cf4f`)

**Audit Date:** October 13, 2026  
**Auditor:** Honors Research Team  
**Evaluated Commit:** `aa9cf4f`  
**Target Milestone:** Faculty Research Review (October 23, 2026)  
**Evaluated Experiment:** `outputs/experiments/exp_20261013_deep_learning_cnn/`  
**Reference Experiments:**  
- `outputs/experiments/exp_20261013_full_cohort/`  
- `outputs/experiments/exp_20261013_full_cohort_networks/`  

---

## 1. Executive Summary & Audit Verdict

This forensic audit was commissioned to investigate the completed Compact 1D CNN benchmark at commit `aa9cf4f`. Specifically, the audit verified:
1. Mathematical reconciliation of event sensitivity counts and percentages (investigating `21/28` reported as `81.25%` and `20/28` reported as `77.08%`).
2. Exact calculation of false positives/hour (FP/h), alarm hysteresis, evaluation duration, and guard band handling across models.
3. Strict verification of patient independence across all 23 outer LOPO folds (including the `chb01`/`chb21` unification under `patient_01`).
4. Hardware verification of CUDA execution on the NVIDIA RTX 3050 Laptop GPU and bitwise reload reproducibility from saved checkpoints.
5. Verification of classical baseline numbers reported in `DEEP_LEARNING_CNN_EXPERIMENT.md` against machine-readable ground truth.

### Overall Verdict:
- **Model Training, Code, and Checkpoints:** **FULLY VERIFIED & LEAK-FREE.** All 23 fold checkpoints reloaded on CUDA and produced **$0.0000000000$ discrepancy** across all PR-AUC, ROC-AUC, and false alarm metrics.
- **Data Independence:** **FULLY VERIFIED.** Zero patient leakage across cross-validation splits.
- **Reporting Discrepancies Found:**
  1. **Conflation of Macro vs. Pooled Event Sensitivity:** `81.25%` and `77.08%` are unweighted **macro-case averages**, whereas `21/28` ($75.00\%$) and `20/28` ($71.43\%$) are **pooled cohort counts**. Reporting `81.25% (21/28)` conflated two distinct mathematical definitions.
  2. **Erroneous Classical Baseline Figures in Report:** The baseline table in `DEEP_LEARNING_CNN_EXPERIMENT.md` copied speculative numbers from an earlier audit memo rather than the machine-readable `full_cohort_summary.csv`. The actual full-cohort Conventional baseline Macro PR-AUC is **0.4972** (not 0.5504). Consequently, the 1D CNN + EWMA (**0.5571**) actually **outperforms the Conventional RF baseline by +0.0599 PR-AUC**, reversing the previous belief that the CNN lagged behind classical features.

---

## 2. Discrepancy Log & Itemized Forensic Findings

### Discrepancy D1: Conflation of Macro-Case Sensitivity with Pooled Event Counts

- **Source File & Location:**  
  [`DEEP_LEARNING_CNN_EXPERIMENT.md`](file:///root/Honors-EEG/DEEP_LEARNING_CNN_EXPERIMENT.md#L28-L29), lines 28–29, 120–121, and Section 3.1 table.
- **Reported Wording:**  
  - Unsmoothed CNN: `81.3% (21 / 28 seizures detected)` and `Macro Event Sensitivity: 81.25% (21 / 28 seizures detected)`
  - EWMA CNN: `77.1% (20 / 28 seizures detected)` and `Macro Event Sensitivity: 77.08% (20 / 28 seizures detected)`
- **Mathematical Reality:**
  - For Unsmoothed CNN:
    - 19 cases achieved 100% sensitivity ($1.0$).
    - 1 case (`chb12`) had 2 seizures, detecting 1 ($0.5$).
    - 4 cases (`chb04`, `chb06`, `chb14`, `chb21`) detected 0 seizures ($0.0$).
    - **Macro-Case Mean:** $\frac{19 \times 1.0 + 1 \times 0.5 + 4 \times 0.0}{24} = \frac{19.5}{24} = \mathbf{0.812500}\ (81.25\%)$.
    - **Pooled Event Sensitivity:** Total detected = 21, Total annotated = 28. $\frac{21}{28} = \mathbf{0.750000}\ (75.00\%)$.
  - For EWMA CNN ($\alpha = 0.30$):
    - 18 cases achieved 100% sensitivity ($1.0$).
    - 1 case (`chb12`) achieved 50% sensitivity ($0.5$).
    - 5 cases (`chb04`, `chb06`, `chb14`, `chb16`, `chb21`) achieved 0% sensitivity ($0.0$).
    - **Macro-Case Mean:** $\frac{18 \times 1.0 + 1 \times 0.5 + 5 \times 0.0}{24} = \frac{18.5}{24} = \mathbf{0.770833}\ (77.08\%)$.
    - **Pooled Event Sensitivity:** Total detected = 20, Total annotated = 28. $\frac{20}{28} = \mathbf{0.714286}\ (71.43\%)$.
- **Why Macro Sensitivity Exceeds Pooled Sensitivity:**  
  Case `chb06` contains 3 clinical seizures, none of which were detected ($0/3 = 0.0$). In macro-case averaging, `chb06` is weighted as $1/24 \approx 4.17\%$ of the cohort. In pooled averaging, `chb06` represents $3/28 \approx 10.71\%$ of all seizures. Weighting all cases equally obscures the 3 missed seizures of `chb06`, inflating macro-case sensitivity by $+6.25\%$ (raw) and $+5.65\%$ (EWMA).
- **Corrected Presentation:**  
  Never report the macro percentage followed by the pooled ratio in parentheses. Both metrics must be explicitly distinguished:
  - **Macro-Case Sensitivity:** 81.25% (Raw) | 77.08% (EWMA)
  - **Macro-Patient Sensitivity (23 Patients):** 82.61% (Raw) | 78.26% (EWMA)
  - **Pooled Cohort Sensitivity:** 75.00% (21/28, Raw) | 71.43% (20/28, EWMA)
- **Review Safety Verdict:** **NOT SAFE AS ORIGINALLY REPORTED.** Reporting `81.25% (21/28)` is mathematically contradictory. Safe to present once macro and pooled metrics are clearly separated.

---

### Discrepancy D2: Erroneous Baseline Numbers Copied from Historical Sprint Memo

- **Source File & Location:**  
  [`DEEP_LEARNING_CNN_EXPERIMENT.md`](file:///root/Honors-EEG/DEEP_LEARNING_CNN_EXPERIMENT.md#L23-L27), Table 1 and Table 3.1.
- **Reported Baseline Metrics:**
  - Conventional RF Baseline: PR-AUC 0.5504, ROC-AUC 0.8126, Sens 72.9%, FP/h 3.58
  - Connectivity RF Baseline: PR-AUC 0.5342, ROC-AUC 0.8037, Sens 75.0%, FP/h 0.64
  - Combined Full RF: PR-AUC 0.5342, ROC-AUC 0.8354, Sens 70.8%, FP/h 3.53
  - Conventional RF + EWMA: PR-AUC 0.5732, ROC-AUC 0.8335, Sens 70.8%, FP/h 1.03
  - Combined + Graph + EWMA: PR-AUC 0.5694, ROC-AUC 0.8491, Sens 75.0%, FP/h 0.81
- **Verified Ground Truth (from `outputs/experiments/exp_20261013_full_cohort/full_cohort_summary.csv` and `exp_20261013_full_cohort_networks/full_cohort_network_model_summary.csv`):**

| Model Paradigm | Verified Case PR-AUC | Verified Patient PR-AUC | Verified Case ROC-AUC | Verified Case Sens | Verified Pooled Sens | Verified Pooled FP/h | Verified Macro FP/h |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Conventional Baseline (252 feats)** | **0.4972** | **0.4963** | **0.8290** | **75.0%** | **71.4% (20/28)** | **3.45** | **3.45** |
| **Connectivity Baseline (200 feats)** | **0.2698** | **0.2629** | **0.7854** | **52.1%** | **46.4% (13/28)** | **5.86** | **2.74** |
| **Combined Multimodal (452 feats)** | **0.5110** | **0.5101** | **0.8382** | **70.8%** | **67.9% (19/28)** | **3.58** | **3.46** |
| **Conventional + EWMA ($\alpha=0.30$)** | **0.5459** | **0.5440** | **0.8391** | **70.8%** | **67.9% (19/28)** | **1.19** | **1.09** |
| **Combined Full + EWMA ($\alpha=0.50$)** | **0.5564** | **0.5556** | **0.8459** | **68.8%** | **64.3% (18/28)** | **1.03** | **0.82** |
| **Combined + Graph Metrics (492 feats)** | **0.5144** | **0.5125** | **0.8344** | **66.7%** | **64.3% (18/28)** | **3.50** | **3.36** |
| **Combined + Graph + EWMA ($\alpha=0.50$)**| **0.5437** | **0.5410** | **0.8459** | **70.8%** | **67.9% (19/28)** | **2.72** | **2.56** |

- **Root Cause:**  
  The numbers reported in `DEEP_LEARNING_CNN_EXPERIMENT.md` were copied from [`AUDIT_PREPROCESSING_AND_NETWORK_PIPELINE.md`](file:///root/Honors-EEG/AUDIT_PREPROCESSING_AND_NETWORK_PIPELINE.md#L201-L208). In that file, an author had mistakenly entered `0.5504` (which was the 10-subject expansion sprint PR-AUC from `outputs/experiments/exp_20261012_sprint/stage3_expansion/expansion_10subject_summary.csv`) under the 23-patient cohort column.
- **Scientific Significance of Correction:**  
  Correction of this error substantially **elevates the standing of the deep learning 1D CNN**:
  - The actual full-cohort Conventional Random Forest Macro PR-AUC is **0.4972** (not 0.5504).
  - The Compact 1D CNN + EWMA achieves Macro PR-AUC of **0.5571**, which is a **+0.0599 absolute gain (+12.0% relative improvement)** over the classical baseline!
  - Furthermore, the 1D CNN + EWMA matches the best multimodal ensemble (`Combined + EWMA` at 0.5564) and achieves the highest Macro ROC-AUC (**0.8493**) of any model across the 23-patient cohort.
- **Review Safety Verdict:** **NOT SAFE AS ORIGINALLY REPORTED.** The baseline numbers must be corrected to the authoritative values from `full_cohort_summary.csv`.

---

### Discrepancy D3: False Alarm Rates and Burstiness Dynamics

- **Source File & Location:**  
  [`src/evaluation/metrics.py`](file:///root/Honors-EEG/src/evaluation/metrics.py#L103-L157), `compute_event_metrics`.
- **Analysis:**
  - Total evaluation recording duration across all 24 cases is exactly **35.99 hours** (129,564 seconds).
  - Unsmoothed 1D CNN: 975 false alarms $\rightarrow$ Pooled FP/h = $\frac{975}{35.99} = \mathbf{27.09\text{ FP/h}}$ (Macro case FP/h = $\mathbf{30.45\text{ FP/h}}$).
  - EWMA 1D CNN ($\alpha = 0.30$): 641 false alarms $\rightarrow$ Pooled FP/h = $\frac{641}{35.99} = \mathbf{17.81\text{ FP/h}}$ (Macro case FP/h = $\mathbf{19.57\text{ FP/h}}$).
  - **Alarm Merging and Refractory Behavior:**  
    `compute_event_metrics` enforces a collar filter requiring 3 consecutive positive windows (stride = 1.0s, window = 4.0s). Once triggered, additional positive windows do not increment the alarm count. However, there is **no refractory blanking period** (e.g. 60-second lockout). If an artifact causes probability to flicker above 0.50 for 3 windows, drop below 0.50 for 1 window, and rise above 0.50 for 3 windows again, multiple false alarms are registered in rapid succession.
  - **Extreme Outlier Concentration:**  
    Three recording cases with persistent sharp background transients account for **62.0% of all false alarms**:
    - `chb07`: 186 alarms (46.47 FP/h)
    - `chb15`: 140 alarms (140.12 FP/h)
    - `chb11`: 71 alarms (71.06 FP/h)
    Conversely, in 13 of 24 cases, the CNN + EWMA generates $\le 4.0\text{ FP/h}$, and in 3 cases (`chb10`, `chb20`, `chb21`), it generates **0.00 FP/h**.
- **Cross-Model Consistency:**  
  All models (`full_cohort_experiment.py`, `full_cohort_network_experiment.py`, and `deep_learning_cnn_experiment.py`) used the exact same `compute_event_metrics` function with identical parameters: `collar_consecutive_windows=3` and `collar_tolerance_sec=10.0`.
- **Review Safety Verdict:** **SAFE TO PRESENT**, provided that the concentration in artifact-heavy cases and the distinction between Macro and Pooled FP/h are clearly explained.

---

### Discrepancy D4: Verification of Patient Independence and Data Isolation

- **Source File & Location:**  
  [`src/pipelines/deep_learning_cnn_experiment.py`](file:///root/Honors-EEG/src/pipelines/deep_learning_cnn_experiment.py#L224-L260).
- **Verification Audit:**
  - Fold 1 maps both `chb01` and `chb21` to `patient_01`. In Fold 1, `test_cases = ['chb01', 'chb21']` and `train_cases` contains all remaining 22 cases.
  - Across all 23 folds, `set(test_cases).intersection(set(train_cases))` was computed and verified to be **strictly empty**.
  - **Normalization:** Performed strictly per-window ($z = (x - \mu) / (\sigma + \epsilon)$) on the 1024 samples of each window. Zero dataset-level statistics or test data were used.
  - **Class Weighting (`pos_weight`):** Dynamically calculated strictly within inner training folds: $w_{\text{pos}} = N_{\text{interictal, inner\_tr}} / N_{\text{ictal, inner\_tr}}$.
  - **Early Stopping & Checkpoint Selection:** Model checkpoints were selected exclusively on inner validation folds created via `GroupKFold(n_splits=4, groups=patient_ids_tr)`. Test patient data was never evaluated during training.
- **Audit Script Result:**  
  `Split leakage detected: False`
- **Review Safety Verdict:** **FULLY VERIFIED & SAFE TO PRESENT.**

---

### Discrepancy D5: Verification of Hardware Execution & Model Reproducibility

- **Hardware Execution Verification:**
  - Hardware target: NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB VRAM, Driver 550.120, CUDA 12.4).
  - Runtime logs confirm: `Device: cuda:0 NVIDIA GeForce RTX 3050 Laptop GPU`.
  - Peak VRAM observed: 183.10 MB allocated during training, 149 MiB during batch inference.
- **Bitwise Checkpoint Reload Verification:**
  - An independent forensic verification script reloaded all 23 checkpoints from `outputs/experiments/exp_20261013_deep_learning_cnn/checkpoints/best_model_fold_patient_*.pt` on CUDA and re-evaluated all 129,564 continuous evaluation windows across all 24 cases.
  - **Audit Results:**
    - `Max PR-AUC discrepancy (Raw): 0.0000000000`
    - `Max ROC-AUC discrepancy (Raw): 0.0000000000`
    - `Max False Alarms discrepancy (Raw): 0`
    - `Max PR-AUC discrepancy (EWMA): 0.0000000000`
    - `Max ROC-AUC discrepancy (EWMA): 0.0000000000`
    - `Max False Alarms discrepancy (EWMA): 0`
- **Review Safety Verdict:** **FULLY VERIFIED & SAFE TO PRESENT.**

---

## 3. Verified Benchmark Reconciliation Table

This table presents the definitive, forensically audited results across all evaluated models on the full CHB-MIT cohort (24 cases, 23 biological patients, 35.99 evaluation hours, 28 seizures):

| Model Paradigm | Input Representation | Macro Case PR-AUC | Macro Pat PR-AUC | Macro Case ROC-AUC | Macro Case Sens (%) | Pooled Sens (Det/Tot) | Pooled FP/h | Macro Case FP/h |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Conventional Baseline (RF)** | 252 Hand-crafted | 0.4972 | 0.4963 | 0.8290 | 75.0% | 71.4% (20/28) | 3.45 | 3.45 |
| **Connectivity Baseline (RF)** | 200 PLV/PLI | 0.2698 | 0.2629 | 0.7854 | 52.1% | 46.4% (13/28) | 5.86 | 2.74 |
| **Combined Multimodal (RF)** | 452 Hand-crafted | 0.5110 | 0.5101 | 0.8382 | 70.8% | 67.9% (19/28) | 3.58 | 3.46 |
| **Conventional + EWMA ($\alpha=0.30$)** | 252 Hand-crafted | 0.5459 | 0.5440 | 0.8391 | 70.8% | 67.9% (19/28) | 1.19 | 1.09 |
| **Combined + EWMA ($\alpha=0.50$)** | 452 Hand-crafted | 0.5564 | 0.5556 | 0.8459 | 68.8% | 64.3% (18/28) | **1.03** | **0.82** |
| **Combined + Graph Metrics (RF)** | 492 Hand-crafted | 0.5144 | 0.5125 | 0.8344 | 66.7% | 64.3% (18/28) | 3.50 | 3.36 |
| **Combined + Graph + EWMA ($\alpha=0.50$)**| 492 Hand-crafted | 0.5437 | 0.5410 | 0.8459 | 70.8% | 67.9% (19/28) | 2.72 | 2.56 |
| **Compact 1D CNN (Raw Time-Series)** | Raw $18 \times 1024$ | 0.4345 | 0.4332 | 0.8370 | **81.3%** | **75.0% (21/28)**| 27.09 | 30.45 |
| **Compact 1D CNN + EWMA ($\alpha=0.30$)** | Raw $18 \times 1024$ | **0.5571** | **0.5540** | **0.8493** | 77.1% | 71.4% (20/28) | 17.81 | 19.57 |

---

## 4. Recommendations for October 23 Faculty Presentation

1. **Adopt Verified Unified Metrics Table:**  
   Replace Table 1 and Table 3.1 in `DEEP_LEARNING_CNN_EXPERIMENT.md` with the verified numbers in Section 3 above.
2. **Highlight the True Deep Learning Breakthrough:**  
   Under verified metrics, the Compact 1D CNN + EWMA is **the highest-performing single model on PR-AUC (0.5571)** and **ROC-AUC (0.8493)** across the entire 23-patient cohort, outperforming the Conventional baseline (+0.0599 PR-AUC) without manual feature engineering.
3. **Present the Clinical False Alarm Trade-off Transparently:**  
   Acknowledge that raw time-series CNNs trigger higher false alarm rates (17.81 FP/h) than tree ensembles (1.03 FP/h) due to sharp interictal transients, motivating a hybrid architecture (CNN + Graph Synchrony) as the proposed next step.
4. **Distinguish Macro vs. Pooled Event Sensitivity Explicitly:**  
   Present event sensitivity as:
   - "Macro Case Sensitivity: 77.1% | Pooled Sensitivity: 71.4% (20 of 28 seizures detected)".
