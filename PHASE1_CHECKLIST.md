# Phase 1 Implementation Checklist: Scalp EEG Epileptic Seizure Detection

**Project**: BTech CSE Honors Project  
**Workspace**: `/root/Honors-EEG`  
**Target Environment**: `honors_env` (Python 3.10.12)  
**Prototyping Cohort**: `chb01`, `chb03`, `chb05`, `chb07`, `chb08` (5 patients, debugging only)  
**Status**: Phase 1 Fully Completed & Verified Across Prototype Cohort — Ready for Phase 2 Baseline Models

---

## 1. Environment & Package Audit

### 1.1 Virtual Environment Status
* **Location**: `/root/Honors-EEG/honors_env` (Created using `python3 -m venv`)
* **Base Python**: Python 3.10.12 (WSL2 Ubuntu, kernel 6.6.x)
* **Current State**: Clean venv containing only `pip 22.0.2` and `setuptools 59.6.0`.
* **Scikit-learn Check**: Verified **NOT installed** in either system Python or `honors_env`.
* **PyTorch Policy**: **Deferred**. Deep learning dependencies will not be installed until Phase 1 validation and classical baselines are complete.

### 1.2 Proposed Installation Commands (NOT EXECUTED)
To equip `honors_env` with the necessary data processing, feature extraction, and classical ML libraries without installing heavy PyTorch or CUDA packages:

```bash
/root/Honors-EEG/honors_env/bin/pip install --upgrade pip
/root/Honors-EEG/honors_env/bin/pip install numpy scipy pandas scikit-learn mne matplotlib tqdm
```

*Estimated download size*: ~120 MB (no GPU dependencies, installs in < 1 minute).

---

## 2. Verified Dataset Integrity & Documented Discrepancies

### 2.1 Full Dataset Ground-Truth Verification
Every file in `/mnt/d/College/Honors/chb-mit-scalp-eeg-database-1.0.0` was inspected programmatically against raw headers and summary text files.

| Dimension | Count / Metric | Verification Source & Status |
| :--- | :--- | :--- |
| **Total EDF Recordings** | 686 files | 100% verified present on disk; matches `RECORDS`. |
| **Files with Seizures (on disk)** | 141 files | Verified via companion `.edf.seizures` files on disk. |
| **Files with Seizures (in summaries)** | 141 files | Verified via `*-summary.txt` files (100% agreement with `.seizures` files). |
| **Total Seizure Events** | 198 seizures | Sum of all annotated seizure intervals across all 24 summary files. |
| **Total Ictal Duration** | 11,611 seconds (~3.23 hours) | Exact sum of all `(end - start)` intervals. |
| **Total Continuous EEG** | ~983 hours | Seizures represent **~0.33%** of recording time (~1:300 class imbalance). |
| **Sampling Rate ($f_s$)** | 256 Hz | Uniform across all 686 recordings. |

### 2.2 Deep Investigation of Documented Discrepancies

#### A. The `chb07_18.edf` vs. `chb07_19.edf` Clerical Typo in PhysioNet
* **`RECORDS`**: Lists both `chb07/chb07_18.edf` (line 233) and `chb07/chb07_19.edf` (line 234). Both exist on disk (142,024,704 bytes and 169,710,080 bytes respectively).
* **`RECORDS-WITH-SEIZURES`**: Lists `chb07/chb07_18.edf` (line 35). Does **not** list `chb07/chb07_19.edf`.
* **`chb07-summary.txt`**:
  * Lines 119–122: `File Name: chb07_18.edf` -> `Number of Seizures in File: 0`.
  * Lines 124–129: `File Name: chb07_19.edf` -> `Number of Seizures in File: 1`, Start: 13,688s, End: 13,831s (duration: 143s).
* **Disk Annotation Files**:
  * `chb07_19.edf.seizures` exists on disk (54 bytes, containing WFDB header `## time resolution: 256`).
  * `chb07_18.edf.seizures` does **not** exist anywhere on disk.
* **Conclusion**: `RECORDS-WITH-SEIZURES` has a known clerical typo in the original PhysioNet distribution. Ground truth must be parsed directly from `chbXX-summary.txt` and `.edf.seizures`, bypassing `RECORDS-WITH-SEIZURES`.

#### B. Audit of Duplicate Channel Labels (Particularly `T8-P8`)
* **Scope**: 655 of the 686 files contain duplicate labels for `T8-P8`.
* **Empirical Verification across all 686 files**:
  * In **605 files**, the two `T8-P8` channels are **bitwise identical** ($\max|\Delta| = 0.0$, correlation $r = 1.0$).
  * In **50 files** (11 files in `chb12`: `chb12_32`–`chb12_42`; 39 files in `chb15`: `chb15_02`–`chb15_63`), the two channels are **not identical**, although highly correlated ($r \approx 0.895$ to $0.999$).
* **Anatomical Root Cause**:
  * In the longitudinal bipolar ("double banana") montage, the Right Temporal chain is defined as:  
    $$\text{FP2-F8} \longrightarrow \text{F8-T8} \longrightarrow \text{T8-P8} \longrightarrow \text{P8-O2}$$
  * In the summary files for `chb12` and `chb15`, Channel 21/22 is the `T8-P8` situated directly between `F8-T8` and `P8-O2`.
  * Channel 28/29 is a secondary `T8-P8` appended at the end of the modified inferior temporal chain following `FT10-T8`.
* **Deterministic Policy**:
  * Always select the **first occurrence** of `T8-P8` (the one positioned in the standard Right Temporal longitudinal chain).
  * Explicitly reject dummy spacer channels (`'-'`, `'.'`) and non-EEG derivations (`ECG`, `VNS`, `LOC-ROC`, `FC*-Ref`).

#### C. Case `chb24`
* Added in December 2010 post-publication; omitted from `SUBJECT-INFO`.
* Verified: 16 seizures across 22 recordings. Non-seizure files were omitted from `chb24-summary.txt` but exist in `RECORDS`. Demographics recorded as `Unknown`.

#### D. Filename Formatting Quirk (`chb02_16+.edf`)
* Contains a literal `+` character; all parsers must use `[a-zA-Z0-9_\-\.\+]+`.

#### E. Repeated Subject Identity (`chb01` and `chb21`)
* `chb21` was recorded from the identical female subject as `chb01` 1.5 years later.
* **Leakage Barrier**: Must be bound to the **same subject group** in all cross-validation splits.

---

## 3. Prototype Cohort Specification (Debugging Only)

This 5-patient cohort (`chb01`, `chb03`, `chb05`, `chb07`, `chb08`) is strictly designated for **pipeline debugging and unit-testing**, not for claiming final generalization performance.

| Subject | Gender / Age | Total Files | Seizure Files | Total Seizures | Ictal Duration | Seizure Duration Range |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **chb01** | F / 11 yr | 42 | 7 | 7 | 442 s (~7.4 min) | 27 s – 101 s |
| **chb03** | F / 14 yr | 38 | 7 | 7 | 402 s (~6.7 min) | 47 s – 69 s |
| **chb05** | F / 7 yr | 39 | 5 | 5 | 558 s (~9.3 min) | 96 s – 120 s |
| **chb07** | F / 14.5 yr | 19 | 3 | 3 | 325 s (~5.4 min) | 86 s – 143 s |
| **chb08** | M / 3.5 yr | 20 | 5 | 5 | 919 s (~15.3 min) | 134 s – 264 s |
| **Total** | — | **158** | **27** | **27** | **2,646 s (~44.1 min)** | **27 s – 264 s** |

---

## 4. Finalized Signal Standardization, Windowing & Labeling Specification

### 4.1 Canonical 18-Channel Bipolar Montage
All EDF inputs will be projected to the following standard 18 bipolar derivations:
1. **Left Temporal**: `FP1-F7`, `F7-T7`, `T7-P7`, `P7-O1`
2. **Left Parasagittal**: `FP1-F3`, `F3-C3`, `C3-P3`, `P3-O1`
3. **Right Parasagittal**: `FP2-F4`, `F4-C4`, `C4-P4`, `P4-O2`
4. **Right Temporal**: `FP2-F8`, `F8-T8`, `T8-P8`, `P8-O2`
5. **Midline / Central**: `FZ-CZ`, `CZ-PZ`

*Preprocessing Filter*: Zero-phase Butterworth bandpass filter (**0.5 Hz – 45.0 Hz**) followed by per-recording channel-wise z-score standardization.

### 4.2 Mathematical Windowing & Overlap Formula
All calculations operate in integer sample indices at $f_s = 256\text{ Hz}$ to eliminate floating-point drift:
* Window duration: $W_{samp} = 4.0 \times 256 = 1024\text{ samples}$ ($W = 4.0\text{ s}$).
* Window $k$ covers sample interval $[s_k, e_k] = [k \cdot S_{samp}, k \cdot S_{samp} + W_{samp}]$.
* For a true seizure interval $[S_{start}, S_{end}] = [T_{start} \cdot f_s, T_{end} \cdot f_s]$:
  $$\text{Overlap Samples } O = \max(0, \min(e_k, S_{end}) - \max(s_k, S_{start}))$$
  $$\text{Overlap Ratio } r = \frac{O}{W_{samp}} = \frac{O}{1024}$$

```
      Pre-Ictal Guard                   Ictal (Seizure)                  Post-Ictal Guard
      [T_start - 60s, T_start]          [T_start, T_end]                 [T_end, T_end + 60s]
──────────┬─────────────────────────────┬────────────────────────────────┬─────────────────────────────┬──────────
          │   DISCARD FROM TRAINING     │  y = 1 if overlap >= 50%       │   DISCARD FROM TRAINING     │
 y = 0    │   (Transition / Pre-seizure)│  (Discard if 0% < r < 50%)     │   (Post-ictal suppression)  │  y = 0
 Valid BG │                             │                                │                             │ Valid BG
```

### 4.3 Classification & Guard-Band Rules
1. **Positive Class ($y = 1$)**: $r \ge 0.5$ (at least 50% of the window duration is ictal).
2. **Boundary Windows ($0 < r < 0.5$)**: Excluded from training to eliminate ambiguous label noise.
3. **Guard Bands ($60\text{ s}$ pre-ictal, $60\text{ s}$ post-ictal)**:
   * **Training Data**: Excluded from background ($y = 0$) to prevent pre-ictal epileptiform discharges or post-ictal EEG suppression from corrupting the baseline interictal model.
   * **Test Data**: **Guard bands are NOT excluded during test evaluation**. In real-time continuous evaluation, every window is classified. If a model fires during pre-ictal or post-ictal periods outside the annotated seizure, it is strictly counted as a False Positive or early onset detection.
4. **Recording Boundaries**: Windows are strictly contained within recording bounds ($e_k \le N_{total\_samples}$). No zero-padding across discontinuous recording boundaries.
5. **Adjacent Seizures**: If two seizures are separated by $< 120\text{ s}$, the intervening window is considered an inter-ictal transition zone and excluded from training background.

### 4.4 Event-Level Evaluation & Double-Counting Prevention
* **Window-Level to Event-Level Mapping**: A rolling collar filter requires **$M = 3$ consecutive positive windows** (spanning $\ge 6\text{ s}$) to declare a clinical Seizure Event Alarm.
* **Single-Count Rule**: An annotated seizure $[T_{start}, T_{end}]$ is marked **Detected (True Positive)** if at least one alarm triggers in $[T_{start} - 10\text{ s}, T_{end} + 10\text{ s}]$. Multiple alarms within the same seizure count as **exactly ONE True Positive**, never multiple TPs.
* **False Alarms**: Alarms occurring completely outside annotated seizure windows count toward the **False Positive Rate per Hour (FP/h)**.

---

## 5. Leakage-Safe Evaluation Architecture

### 5.1 Primary Evaluation: Patient-Independent (LOSO CV)
* **Objective**: Evaluate generalization to unseen patients.
* **Grouping**: Grouped by biological patient across all 24 cases ($N = 23$ unique biological subjects).
  * `chb01` and `chb21` are strictly kept together in the same cross-validation fold.
* **Strict Parameter Freeze**:
  * Filtering parameters, channel layout, windowing parameters, feature extraction definitions, scaler fitting ($\mu, \sigma$), and model hyperparameters are strictly determined using **training folds only**.
  * No test patient data is ever seen during feature selection or normalization fitting.

### 5.2 Within-Patient Evaluation (Secondary, Strictly Chronological)
* For individual subjects with sufficient seizures (e.g., $\ge 5$ seizures), split strictly by **chronological recording session**:
  * Train on the earliest $70\%$ of sessions; test on the subsequent $30\%$.
  * Never randomly shuffle windows or recordings.

### 5.3 Unaltered Test Prevalence
* While training negative windows may be subsampled (e.g., 10:1) to balance gradient updates, **test recordings are evaluated in full continuous duration** (unaltered natural prevalence) to guarantee clinically honest PR-AUC and FP/h metrics.

---

## 6. Phase 1 Implementation Sequence

- [x] **Task 1.0: Environment Reconnaissance & Virtual Environment Creation**
  - Verified Python 3.10, WSL2 CUDA/GPU availability, and clean `honors_env`.
- [x] **Task 1.1: Deep Evidence Audit & Discrepancy Verification**
  - Confirmed 686 EDF files, 141 seizure files, 198 seizures, 11,611s ictal time.
  - Resolved `chb07_18` vs `chb07_19` typo in `RECORDS-WITH-SEIZURES`.
  - Audited `T8-P8` duplicate channels across all 686 files.
  - Finalized mathematical windowing, labeling, and event-evaluation rules.
- [x] **Task 1.2: Dependency Installation**
  - Executed minimal pip installation in `honors_env` for `numpy`, `scipy`, `pandas`, `scikit-learn`, `mne`, `matplotlib`, `tqdm`, `pyarrow`.
  - Verified clean import in `honors_env`.
- [x] **Task 1.3: Metadata & Manifest Parser (`src/data/metadata_parser.py`)**
  - Implemented `CHBMITMetadataParser` parsing all 24 summary files into unified manifests (`recordings_manifest.parquet` and `seizures_manifest.parquet`).
  - Unit test `tests/test_metadata_parser.py` passed: 686 recordings, 141 seizure files, 198 seizures, 11,611s ictal time.
- [x] **Task 1.4: EDF Segment Loader & Preprocessing (`src/data/loader.py`)**
  - Implemented `EDFLoader` with memory-safe chunk loading, 18 canonical channels mapping, first-occurrence `T8-P8` de-duplication, 0.5–45 Hz zero-phase bandpass filtering, and z-score scaling.
  - Verified with unit tests in `tests/test_loader_and_windowing.py`.
- [x] **Task 1.5: Windowing & Label Generator (`src/data/windowing.py`)**
  - Implemented `WindowGenerator` with integer-sample overlap calculation ($r \ge 0.5$ for $y=1$), $0 < r < 0.5$ boundary exclusion during training, 60s pre/post-ictal guard bands, and full un-subsampled continuous evaluation mode.
  - Verified with synthetic and real signal unit tests.
- [x] **Task 1.6: Feature Extraction Module (`src/features/feature_extractor.py`)**
  - Implemented `EEGFeatureExtractor` computing 14 time/spectral features across 18 channels = 252 features per window.
  - Unit test `tests/test_feature_extractor.py` passed: output shape `(N, 252)`, finite values, bounded relative powers and entropy.
- [x] **Task 1.7: Prototype Validation Test (`tests/test_prototype_pipeline.py`)**
  - Executed end-to-end pipeline across all 5 prototype patients (`chb01`, `chb03`, `chb05`, `chb07`, `chb08`).
  - All 5 subjects passed:
    - `chb01_03.edf`: 41 ictal windows, features `(41, 252)`.
    - `chb03_01.edf`: 53 ictal windows, features `(53, 252)`.
    - `chb05_06.edf`: 116 ictal windows, features `(116, 252)`.
    - `chb07_19.edf`: 144 ictal windows, features `(144, 252)`.
    - `chb08_02.edf`: 172 ictal windows, features `(172, 252)`.
    - Non-seizure files verified producing 0 false positive labels.
    - Event-level collar hysteresis and single-count evaluation logic verified.
