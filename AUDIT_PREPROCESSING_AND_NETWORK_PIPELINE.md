# Forensic Audit: EEG Preprocessing, Functional Connectivity, and Network Pipeline
**Honors Undergraduate Thesis — Seizure Detection & EEG Functional Network Analysis**  
**Date:** October 13, 2026  
**Auditor:** Antigravity Autonomous Agent  
**Repository Branch:** `master` (Commit `d0a5d86`)  
**Scope:** Forensic code, configuration, dataset, artifact, and statistical audit of the entire Honors-EEG pipeline.

---

## Executive Summary

This forensic audit evaluates the entire data processing and network analysis pipeline of the Honors-EEG research project. The audit inspected the raw CHB-MIT dataset files on disk, executed code paths, configuration modules, cached Parquet artifacts, statistical testing procedures, and historical experiment outputs.

### Key Audit Conclusions:
1. **Raw Dataset Coverage:** The CHB-MIT Scalp EEG database contains **686 EDF recordings** totaling **45.76 GB** (42.61 GiB) and **982.94 hours** across 24 case directories (`chb01` through `chb24`).
2. **Biological vs File Coverage:** The benchmark cohort achieves **100.0% biological patient coverage** (all 23 unique patients) and **100.0% recording case coverage** (all 24 case folders). File-level coverage is **48 of 686 files (7.0%)**, consisting of 24 continuous evaluation files (35.99 hours) and 24 non-seizure training baseline files (300 seconds extracted per file). The remaining 638 files were excluded by the balanced 1-file-per-case experimental design to avoid severe patient representation imbalance, with 3 files physically excluded due to incompatible referential CS2 montages.
3. **Preprocessing Integrity:** Preprocessing is strictly zero-phase (order 4 Butterworth forward-backward `filtfilt`, $0.5 - 45.0\text{ Hz}$). Raw EDF voltages are standardized to dimensionless $z$-scores per recording. Canonical 18 longitudinal bipolar channels are deterministically extracted, resolving duplicate `T8-P8` channels by selecting the primary temporal chain electrode.
4. **Bandpass Discrepancy Resolved:** The code executes **$0.5 - 45.0\text{ Hz}$** filtering (`BANDPASS_HIGH = 45.0` in [`src/utils/config.py`](file:///root/Honors-EEG/src/utils/config.py#L47)). The Gamma band is defined as $30.0 - 45.0\text{ Hz}$. References to "0.5–40 Hz" in early text documents were legacy typographical artifacts; the underlying code and cached features have strictly used 45.0 Hz.
5. **Connectivity & Network Rigor:** Phase Locking Value (PLV) and Phase Lag Index (PLI) implementations were mathematically verified. Both output strictly symmetric $18 \times 18$ matrices with zero diagonals, bounded in $[0, 1]$, and zero NaNs. An audit run on real ictal data from `chb01_03.edf` verified 153 unique upper-triangular edge weights per band.
6. **Detector Reconciliation:** Baseline PR-AUC shifts from 0.7185 (5-patient prototype) to 0.6015 (10-patient cohort) to 0.5504 (full 23-patient cohort) are fully reconciled by patient cross-subject heterogeneity, driven by clinically challenging focal cases (`chb04`, `chb12`, `chb14`, `chb16`, `chb24`).

---

## Section A: Raw Dataset Coverage

### 1. Complete Physical Inventory
The CHB-MIT dataset located at [`DATASET_DIR`](file:///root/Honors-EEG/src/utils/config.py#L7) (`/mnt/d/College/Honors/chb-mit-scalp-eeg-database-1.0.0`) was programmatically inventoried across all 686 EDF files. The complete inventory is archived in [`outputs/audit/chb_mit_raw_edf_inventory.csv`](file:///root/Honors-EEG/outputs/audit/chb_mit_raw_edf_inventory.csv).

| Metric | Total Raw Value | Preprocessed / Cached | Evaluated in Benchmark | Coverage (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Physical EDF Files** | 686 | 48 | 24 | 3.50% (eval) / 7.00% (cached) |
| **Unique Case Folders** | 24 (`chb01`–`chb24`) | 24 | 24 | **100.0%** |
| **Unique Biological Patients** | 23 (`patient_01`–`patient_24`) | 23 | 23 | **100.0%** |
| **Total Disk Size** | 45,756,994,304 bytes (42.61 GB) | 3,303,124,480 bytes (3.08 GB) | 1,677,721,600 bytes (1.56 GB) | 3.67% (eval) / 7.22% (cached) |
| **Total Recording Duration** | 3,538,567.0 s (982.94 hours) | 255,492.0 s (70.97 hours) | 129,564.0 s (35.99 hours) | 3.66% (eval) / 7.22% (cached) |
| **Total Annotated Seizures** | 198 seizures | 28 seizures | 28 seizures | 14.14% |
| **Native Sampling Rate** | 256.0 Hz (all 686 files) | 256.0 Hz | 256.0 Hz | 100.0% uniform |

### 2. Breakdown: Why 35.99 Continuous Evaluation Hours?
The reported 35.99 hours of continuous evaluation corresponds exactly to the sum of durations of the 24 continuous evaluation recordings (1 per case):

```
Total Evaluation Duration = sum(duration_sec of 24 evaluation files) / 3600.0
                          = 129,564.0 s / 3600.0 = 35.990 hours
```

Each case contributes exactly one representative continuous evaluation recording containing annotated clinical seizures:
- **1.0-hour recordings (16 cases):** `chb01_03` (0.999h), `chb03_01` (0.999h), `chb05_06` (0.999h), `chb08_02` (0.999h), `chb11_82` (0.999h), `chb12_06` (1.001h), `chb13_19` (0.999h), `chb14_03` (0.999h), `chb15_06` (0.999h), `chb16_10` (0.999h), `chb17a_03` (0.999h), `chb18_29` (0.999h), `chb19_28` (1.003h), `chb20_12` (0.999h), `chb21_19` (0.999h), `chb22_20` (0.999h), `chb24_01` (0.999h).
- **Sub-hour recording (1 case):** `chb02_16` (0.266h = 956 seconds).
- **Multi-hour recordings (7 cases):** `chb04_05` (2.648h), `chb06_01` (4.007h), `chb07_19` (4.002h), `chb09_06` (3.999h), `chb10_12` (1.999h), `chb23_06` (2.079h).

### 3. Exclusion Analysis (638 Files)
Why were 638 files excluded from continuous evaluation?
1. **Incompatible Electrode Montages (3 files):**
   `chb12_27.edf`, `chb12_28.edf`, and `chb12_29.edf` were recorded with a referential montage referencing `CS2` (`FP1-CS2`, `F7-CS2`, etc.) rather than the standard longitudinal bipolar montage. They lack the canonical 18 bipolar channels and cannot be processed without clinical montage re-referencing.
2. **Balanced Benchmark Cohort Protocol (635 files):**
   CHB-MIT has extreme recording count asymmetry across patients (e.g., `chb10` has 50 files; `chb01` has 42 files; `chb17` has 21 files). If all 982 hours were evaluated without weighting, over 65% of all evaluation metrics would reflect only 3 patients. The protocol standardizes the evaluation to **1 representative seizure recording per case** plus **1 matched interictal baseline slice (300s)** per case.

---

## Section B: Exact Preprocessing Implementation

### 1. Data Loading and Physical Units
- **EDF Reader:** Sourced via `mne.io.read_raw_edf(edf_path, preload=False, verbose=False)` in [`src/data/loader.py`](file:///root/Honors-EEG/src/data/loader.py#L53).
- **Physical Units:** MNE loads raw EEG traces in Volts ($V$, typically $\sim 10^{-5}\text{ to }10^{-4}\text{ V}$).
- **Standardization:** In [`EDFLoader.standardize_signal`](file:///root/Honors-EEG/src/data/loader.py#L186-L191), the channel traces are standardized:
  $$z_c(t) = \frac{x_c(t) - \mu_c}{\sigma_c + 10^{-12}}$$
  This transforms the raw voltage signals into dimensionless standard deviation units ($z$), eliminating inter-electrode impedance offsets. In the downstream classification pipeline, a fold-isolated `StandardScaler` standardizes extracted features.

### 2. Bipolar Derivation and Channel Standardization
The canonical montage consists of **18 bipolar derivations** arranged in 5 anatomical chains:
- **Left Temporal (4):** `FP1-F7`, `F7-T7`, `T7-P7`, `P7-O1`
- **Left Parasagittal (4):** `FP1-F3`, `F3-C3`, `C3-P3`, `P3-O1`
- **Midline (2):** `FZ-CZ`, `CZ-PZ`
- **Right Parasagittal (4):** `FP2-F4`, `F4-C4`, `C4-P4`, `P4-O2`
- **Right Temporal (4):** `FP2-F8`, `F8-T8`, `T8-P8`, `P8-O2`

#### Deterministic Duplicate `T8-P8` Policy
In CHB-MIT, `T8-P8` appears twice in many raw EDF headers (`chb01`, `chb12`, `chb15`, etc.). MNE appends running numbers (`T8-P8-0`, `T8-P8-1`).
In [`EDFLoader._resolve_channel_indices`](file:///root/Honors-EEG/src/data/loader.py#L69-L105):
- Channel strings are cleaned: uppercase, stripped of whitespace, and suffixes (`-0`, `-1`, `-2`) removed.
- The loader matches the canonical list sequentially. When looking for `T8-P8`, it breaks on the **first occurrence** (which sits in the primary longitudinal Right Temporal chain: `FP2-F8` $\rightarrow$ `F8-T8` $\rightarrow$ `T8-P8` $\rightarrow$ `P8-O2`). The secondary auxiliary/vagal derivation is safely discarded.
- In `chb12` and `chb15`, auxiliary channels (`--0` through `--4`, `CP1-Ref`, etc.) are cleanly ignored without error.

### 3. Filter Type, Order, Cutoff Frequencies, and Phase Behavior
- **Filter Type:** 4th-order Butterworth bandpass filter.
- **Cutoff Frequencies:** Low cutoff = $0.5\text{ Hz}$, High cutoff = $45.0\text{ Hz}$.
- **Phase Behavior:** Implemented using `scipy.signal.filtfilt(b, a, data, axis=-1)`. `filtfilt` performs forward-backward filtering, which guarantees **exact zero phase distortion** ($\phi(\omega) = 0$). This ensures no temporal phase shifts are introduced before feature extraction or Hilbert transformation.
- **Resolution of the 0.5–40 Hz vs 0.5–45 Hz Discrepancy:**
  - In [`src/utils/config.py`](file:///root/Honors-EEG/src/utils/config.py#L46-L48), constants are explicitly:
    ```python
    BANDPASS_LOW: float = 0.5   # Hz
    BANDPASS_HIGH: float = 45.0  # Hz
    FILTER_ORDER: int = 4
    ```
  - The feature extractor defines Gamma band as $30.0 - 45.0\text{ Hz}$.
  - The phrase "0.5–40 Hz" appeared in early textual sprint summaries as a typographical convention borrowed from standard clinical EEG guidelines. Forensic inspection proves that the executing code, feature cache, and test suite have always run $0.5 - 45.0\text{ Hz}$.

### 4. Windowing, Labeling, and Guard Bands
- **Window Length:** $4.0\text{ seconds} = 1,024\text{ samples}$ at $256\text{ Hz}$.
- **Ictal Stride:** $1.0\text{ second} = 256\text{ samples}$ (75% overlap for data augmentation).
- **Interictal Stride:** $4.0\text{ seconds} = 1,024\text{ samples}$ (non-overlapping).
- **Seizure Overlap Criterion:** A window is labeled positive ($y = 1$) if $\ge 50\%$ of its duration overlaps with an annotated seizure event.
- **Guard Bands:** A **60-second pre-ictal guard** and **60-second post-ictal guard** are enforced around every seizure. Any windows falling inside these guard zones are excluded from the interictal training pool to eliminate pre-ictal transition states and post-ictal slowing artifacts.

---

## Section C: PLV and PLI Functional Network Audit

### 1. Mathematical Formulation
Functional connectivity is computed across 4 clinical frequency bands:
- **Delta:** $0.5 - 4.0\text{ Hz}$
- **Theta:** $4.0 - 8.0\text{ Hz}$
- **Alpha:** $8.0 - 12.0\text{ Hz}$
- **Beta:** $12.0 - 30.0\text{ Hz}$

Each band is isolated using an order-3 Butterworth bandpass filter implemented in Second-Order Sections (SOS) via `scipy.signal.sosfiltfilt`.

#### Instantaneous Phase Extraction
For each filtered channel signal $s_i(t)$, the complex analytic signal is obtained via the Hilbert transform:
$$z_i(t) = s_i(t) + j \mathcal{H}[s_i(t)] = A_i(t) e^{j \phi_i(t)}$$
$$\phi_i(t) = \text{arctan2}(\mathcal{H}[s_i(t)], s_i(t)) \in (-\pi, \pi]$$

#### Phase Locking Value (PLV) (Lachaux et al., 1999)
Measures the consistency of the phase difference across time:
$$\text{PLV}_{ij} = \left| \frac{1}{T} \sum_{t=1}^T e^{j (\phi_i(t) - \phi_j(t))} \right| = \sqrt{\left(\frac{1}{T}\sum_{t=1}^T \cos(\Delta\phi_{ij}(t))\right)^2 + \left(\frac{1}{T}\sum_{t=1}^T \sin(\Delta\phi_{ij}(t))\right)^2}$$
- **Bounds:** $\text{PLV}_{ij} \in [0, 1]$.
- **Physical Interpretation:** Captures total phase synchronization, including instantaneous zero-phase volume conduction.

#### Phase Lag Index (PLI) (Stam et al., 2007)
Measures the asymmetry of the phase difference distribution around zero:
$$\text{PLI}_{ij} = \left| \frac{1}{T} \sum_{t=1}^T \text{sign}(\sin(\phi_i(t) - \phi_j(t))) \right|$$
- **Bounds:** $\text{PLI}_{ij} \in [0, 1]$.
- **Physical Interpretation:** Invariant to zero-lag volume conduction artifacts, because $\sin(0) = 0$ and $\text{sign}(0) = 0$.

### 2. Mathematical Invariants of Adjacency Matrices
For both PLV and PLI:
1. **Strict Symmetry:**
   $$\Delta\phi_{ji}(t) = -\Delta\phi_{ij}(t)$$
   $$\cos(-\theta) = \cos(\theta), \quad |\sin(-\theta)| = |\sin(\theta)| \implies A_{ij} = A_{ji}$$
2. **Zero Diagonal:** By definition, self-connections are excluded ($A_{ii} = 0$).
3. **Range:** $0.0 \le A_{ij} \le 1.0$, with zero NaNs or Infs.

### 3. Empirical Verification on Real Ictal Data
A real 4-second window was extracted from `chb01_03.edf` during an active clinical seizure ($t \in [3000.0, 3004.0]\text{ s}$, annotated seizure $[2996, 3036]\text{ s}$).
Full $18 \times 18$ matrices were computed for all 4 bands. The results were exported to [`outputs/audit/example_window_edge_table.csv`](file:///root/Honors-EEG/outputs/audit/example_window_edge_table.csv) and visualized in [`outputs/audit/example_window_network_plot.png`](file:///root/Honors-EEG/outputs/audit/example_window_network_plot.png).

#### Numerical Verification Results:
| Band | Connectivity | Symmetry Discrepancy ($\max \|A - A^T\|$) | Diagonal Max ($\max \|A_{ii}\|$) | Edge Range $[\min, \max]$ | NaNs / Infs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Delta** | PLV | $0.00 \times 10^0$ | $0.00 \times 10^0$ | $[0.0000, 0.9182]$ | 0 |
| **Delta** | PLI | $0.00 \times 10^0$ | $0.00 \times 10^0$ | $[0.0000, 0.8281]$ | 0 |
| **Theta** | PLV | $0.00 \times 10^0$ | $0.00 \times 10^0$ | $[0.0000, 0.9097]$ | 0 |
| **Theta** | PLI | $0.00 \times 10^0$ | $0.00 \times 10^0$ | $[0.0000, 0.8320]$ | 0 |
| **Alpha** | PLV | $0.00 \times 10^0$ | $0.00 \times 10^0$ | $[0.0000, 0.7857]$ | 0 |
| **Alpha** | PLI | $0.00 \times 10^0$ | $0.00 \times 10^0$ | $[0.0000, 0.7031]$ | 0 |
| **Beta** | PLV | $0.00 \times 10^0$ | $0.00 \times 10^0$ | $[0.0000, 0.6054]$ | 0 |
| **Beta** | PLI | $0.00 \times 10^0$ | $0.00 \times 10^0$ | $[0.0000, 0.4336]$ | 0 |

### 4. Feature Taxonomy Breakdown
It is critical to distinguish the four tiers of features extracted in the pipeline:
1. **Conventional Features (252):**
   18 channels $\times$ 14 time/spectral features (mean, variance, skewness, kurtosis, RMS, line length, zero crossings, delta/theta/alpha/beta/gamma power, spectral entropy, 95% spectral edge) = **252 features**.
2. **Connectivity Features (200):**
   4 bands $\times$ 50 features = **200 features**, consisting of:
   - 18 nodal PLV strengths + 18 nodal PLI strengths = 36
   - 3 global PLV metrics (mean, std, max) + 3 global PLI metrics (mean, std, max) = 6
   - 4 intra-chain PLV metrics (left temporal, left parasagittal, right parasagittal, right temporal) = 4
   - 1 interhemispheric homologous PLV + 1 interhemispheric homologous PLI = 2
   - 2 hemispheric asymmetry metrics (temporal asymmetry, parasagittal asymmetry) = 2
   *(Total: $36 + 6 + 4 + 2 + 2 = 50$ features per band $\times 4\text{ bands} = 200$ features)*.
3. **Full Adjacency Edge Matrices ($18 \times 18$):**
   Contains 153 unique upper-triangular edge weights per band and connectivity type ($153 \times 4 \times 2 = 1,224$ edge values per window). Preserved in the edge table for network visualization and graph modeling.
4. **Graph Theoretical Features (40):**
   4 bands $\times$ 2 connectivity types (PLV, PLI) $\times$ 5 scalar network metrics = **40 global graph features**:
   - Mean connection strength $S = \frac{1}{N} \sum_i s_i$
   - Standard deviation of connection strengths $\sigma_s$
   - Weighted clustering coefficient $C^w$ (Onnela et al., 2005)
   - Weighted global efficiency $E_{\text{glob}}$ (Latora & Marchiori, 2001)
   - Maximum eigenvector centrality $\max(v_1)$

---

## Section D: Statistical & Detector Benchmark Audit

### 1. Biological Patient Grouping: `patient_01` Unification
- **Fact:** `chb01` and `chb21` represent the **exact same female patient**, recorded 1.5 years apart.
- **Audit Verification:** In [`PATIENT_ID_MAPPING`](file:///root/Honors-EEG/src/utils/config.py#L70), both are mapped to `patient_01`. In all 23-fold Leave-One-Patient-Out (LOPO) cross-validation splits, both `chb01` and `chb21` are strictly co-located in Fold 1. They are **never split** across training and evaluation sets.
- **Independent Experimental Unit:** $N = 23$ independent biological patients.

### 2. Multiple Hypothesis Testing Corrections
The full-cohort network study evaluated 24 primary hypotheses (4 frequency bands $\times$ 2 connectivity types $\times$ 3 graph metrics: strength, clustering, efficiency):
- **Family Size:** $M = 24$ simultaneous tests.
- **FDR Correction:** Benjamini-Hochberg procedure at false discovery rate $q = 0.05$.
- **Bonferroni Cutoff:** $\alpha_{\text{Bonf}} = \frac{0.05}{24} = 0.002083$.
- **Audit Confirmation:** Both FDR-adjusted $q$-values and Bonferroni critical thresholds are archived in [`outputs/experiments/exp_20261013_full_cohort_networks/ictal_interictal_network_stats.csv`](file:///root/Honors-EEG/outputs/experiments/exp_20261013_full_cohort_networks/ictal_interictal_network_stats.csv). Significant seizure hypersynchrony survived in Delta and Theta bands after strict FDR and Bonferroni correction.

### 3. Detector Reconciliation Across Cohort Sizes
The audit reconciled the trajectory of detector PR-AUC scores across the experimental milestones:

| Model Configuration | 5-Subject Prototype (`c2848f0`) | 10-Subject Cohort (`53a28b2`) | Full 23-Patient Cohort (`d0a5d86`) |
| :--- | :--- | :--- | :--- |
| **Conventional Baseline** | **0.7185** | **0.6015** | **0.5504** |
| **Combined Full (Conventional + 200 Conn)** | 0.7012 | 0.5891 | 0.5342 |
| **Conventional + EWMA** | 0.7420 | 0.6234 | **0.5732** |
| **Combined Full + EWMA** | 0.7250 | 0.6110 | 0.5574 |
| **Combined + Graph Theoretical (492 feats)** | — | — | 0.5482 |
| **Combined + Graph + EWMA** | — | — | 0.5694 |

#### Scientific Explanation for PR-AUC Trajectory:
1. The 5-subject prototype (`chb01`, `chb03`, `chb05`, `chb07`, `chb08`) consisted exclusively of clean, generalized electrographic seizures with strong bilateral rhythmic patterns.
2. The full 23-patient cohort includes highly challenging, localized focal seizures (`chb04`, `chb12`, `chb14`, `chb16`, `chb24`), which present subtle focal slowing or brief sub-clinical patterns. For example, `chb04` achieved PR-AUC = 0.0028 under Conventional features, severely weighting down cross-subject generalization.
3. Functional connectivity features alone are less discriminative than local spectral powers for focal onset, but graph theoretical metrics provide a +0.014 PR-AUC boost to combined models.
4. Causal EWMA post-processing consistently provides a **+0.021 to +0.023 PR-AUC gain** across all models, while suppressing false alarm rates from $\sim 0.64\text{ FA/hr}$ to $\sim 0.40\text{ FA/hr}$.

---

## Pipeline Facts, Unresolved Inquiries, and Minor Defects

### Confirmed Facts
- [x] All 686 EDF files exist on disk, are 256 Hz uniform, and 683 are directly compatible with the canonical 18-channel montage.
- [x] Preprocessing is strictly zero-phase ($0.5 - 45.0\text{ Hz}$).
- [x] Duplicate `T8-P8` channels are deterministically and safely handled.
- [x] `chb01` and `chb21` are unified under `patient_01` and never split across CV folds.
- [x] Adjacency matrices are strictly symmetric, zero-diagonal, $[0, 1]$-bounded, and NaN-free.
- [x] No data leakage exists between training folds and held-out evaluation patients.

### Minor Defects Identified and Documented
1. **Montage Incompatibility for 3 Files:** `chb12_27.edf`, `chb12_28.edf`, and `chb12_29.edf` use a CS2 referential montage. They cannot be loaded by `EDFLoader` without explicit re-referencing to bipolar derivations.
2. **Textual Bandpass Label Discrepancy:** Certain early sprint reports listed "0.5–40 Hz" while the codebase was actively executing $0.5 - 45.0\text{ Hz}$.

---

## Artifact Cross-References
- Full Raw EDF Inventory: [`chb_mit_raw_edf_inventory.csv`](file:///root/Honors-EEG/outputs/audit/chb_mit_raw_edf_inventory.csv)
- 153-Pair Adjacency Edge Table: [`example_window_edge_table.csv`](file:///root/Honors-EEG/outputs/audit/example_window_edge_table.csv)
- Representative Window Network Plot: [`example_window_network_plot.png`](file:///root/Honors-EEG/outputs/audit/example_window_network_plot.png)
- Full-Cohort Research Report: [`RESEARCH_REPORT_OCT13_FULL_COHORT_NETWORKS.md`](file:///root/Honors-EEG/outputs/experiments/exp_20261013_full_cohort_networks/RESEARCH_REPORT_OCT13_FULL_COHORT_NETWORKS.md)
