# Faculty Review Presentation Outline (October 23, 2026)

**Title:** Cross-Patient Automated Seizure Detection in Scalp EEG: Evaluating Functional Connectivity Networks and End-to-End Deep Learning Under Strict Leak-Free Protocols  
**Candidate:** Honors Undergraduate Researcher  
**Format:** 12-Slide Progress Review (15-Minute Presentation + 10-Minute Q&A)  
**Deliverable Directory:** [`review_package_oct23/`](file:///root/Honors-EEG/review_package_oct23/)  

---

### Slide 1: Title & Research Motivation
- **Header:** Automated Cross-Patient Seizure Detection on Continuous Scalp EEG
- **Context:** BTech CSE Honors Research Project, Mid-Term Review (October 2026).
- **Core Challenge:** Detecting epileptic seizures across unobserved patients under severe class imbalance (~1:300 ictal-to-interictal ratio) on continuous multi-hour recordings.
- **Key Message:** Moving beyond artificially simplified clip classification to a rigorous, leak-free Leave-One-Patient-Out (LOPO) benchmark comparing domain-specific functional connectivity against end-to-end deep learning.

---

### Slide 2: Literature Pitfalls & The Need for Methodological Rigor
- **The Reproducibility Crisis in ML for EEG:**
  - Random window splitting inflates accuracy to >98% via signal autocorrelation leakage (adjacent 4-second windows share 3 seconds of identical waveforms).
  - Cross-window within-patient splitting tests patient identity re-identification rather than seizure generalization.
  - Relying exclusively on ROC-AUC under extreme imbalance obscures high false alarm rates.
- **Our Methodological Standards:**
  - Strict 23-fold Leave-One-Patient-Out (LOPO) cross-validation across all 24 recording cases.
  - Biological patient unification: `chb01` and `chb21` are unified under `patient_01` (recorded 1.5 years apart).
  - Primary evaluation metric: Precision-Recall Area Under Curve (PR-AUC) and False Positives per Hour (FP/h) over 35.99 continuous hours.

---

### Slide 3: Dataset Scope: Inventory vs. Evaluated Benchmark
- **The Raw Database:** 686 total raw EDF files (45.76 GB, ~983 hours, 256 Hz uniform) across 24 cases in the CHB-MIT Scalp EEG Database.
- **The Training Cohort:** 48 paired recordings (1 ictal recording + 1 baseline recording per case, generating 17,520 balanced 4-second training windows).
- **The Continuous Evaluation Benchmark:** 24 continuous clinical recordings (35.99 hours, 129,564 windows, 28 clinical seizures).
- **Honesty in Scoping:** We do *not* claim all 686 files were evaluated; the benchmark rigorously evaluates 35.99 continuous hours under clinically realistic continuous monitoring conditions.

---

### Slide 4: Signal Preprocessing & Bipolar Montage
- **Bandpass Filtering:** 0.5–45.0 Hz zero-phase 4th-order FIR filter (eliminates DC drift and 50/60 Hz powerline interference without introducing phase distortion).
- **Channel Standardization:** Derivation of the clinical 18-channel longitudinal bipolar montage ("Double Banana") across left temporal, left parasagittal, central midline, right parasagittal, and right temporal chains.
- **Robust Anomaly Handling:** Algorithmic resolution of duplicate channel labels (`T8-P8-0`, `T8-P8-1`) present in multiple PhysioNet recordings.

---

### Slide 5: Functional Connectivity & Graph Theoretical Modeling
- **Theoretical Basis:** Seizures recruit distributed neural assemblies into hypersynchronous firing.
- **Hilbert Analytic Phase:** Extracted across canonical bands (Delta 0.5–4 Hz, Theta 4–8 Hz, Alpha 8–13 Hz, Beta 13–30 Hz).
- **Two Distinct Connectivity Metrics:**
  - *Phase Locking Value (PLV):* Measures total phase synchronization, including zero-lag coupling.
  - *Phase Lag Index (PLI):* Discards zero-phase-lag differences; invariant to scalp volume conduction.
- **18-Node Functional Graph Metrics:** Node degree, graph density, global efficiency, clustering coefficient, and characteristic path length.
- **Caveat:** Nodes represent scalp bipolar derivations, and edges represent statistical phase locking—*not physical anatomical tracts*.

---

### Slide 6: End-to-End Deep Learning: Compact 1D CNN
- **Design Philosophy:** Eliminating manual feature engineering with a compact architecture suited for edge execution on laptop GPUs.
- **Architecture (`CompactEEGCNN`, 45,409 parameters):**
  - Multi-scale temporal filter bank (Conv1D: $k=33, 17, 9$, strides=1, padding=same).
  - Batch Normalization + GELU non-linear activations + MaxPool1D downsampling.
  - Global Average Pooling (GAP) + Dropout ($p=0.40$) + Linear classification head.
- **Hardware Profile:**
  - Trained on NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) using PyTorch 2.6+cu124.
  - Peak allocated VRAM: **183.1 MB** (4.5% of GPU memory).
  - Full 23-fold LOPO cross-validation completed in **10.08 minutes** (26.3 seconds per fold).

---

### Slide 7: Leak-Free Validation Controls
- **Normalization:** Per-window z-score standardization computed on each 1024-sample window independently; zero dataset-level statistics used.
- **Class Imbalance:** Positive class loss weight ($w_{\text{pos}} = N_{\text{interictal}} / N_{\text{ictal}}$) computed exclusively from inner training fold splits.
- **Inner-Validation Early Stopping:** In each fold, training patients are partitioned using `GroupKFold(n_splits=4)` grouped by patient. Checkpoint restoration evaluates only inner training patients.
- **Isolation Verification:** Outer test patient windows never enter training, validation, normalization, or early stopping.

---

### Slide 8: The Corrected Full-Cohort Benchmark Table
- **Display Benchmark Table:** Present verified results comparing Conventional (252), Connectivity (200), Multimodal (452), Graph (492), and 1D CNN (45k).
- **Core Results:**
  - *Compact 1D CNN + EWMA:* Highest Macro PR-AUC (**0.5571**) and highest Macro ROC-AUC (**0.8493**) in the project, beating the Conventional RF baseline (**0.4972**, $+0.0599$ gain).
  - *Macro vs. Pooled Event Sensitivity:* 1D CNN achieves **81.3% Macro / 75.0% Pooled (21/28)** unsmoothed, and **77.1% Macro / 71.4% Pooled (20/28)** with EWMA.
  - *The Specificity Trade-off:* Classical tree ensembles maintain superior false alarm rejection (**1.03 FP/h** for Combined+EWMA vs. **17.81 FP/h** for CNN+EWMA).

---

### Slide 9: Clinical Discovery: Feature Complementarity in Edge Cases
- **Electro-decremental & Focal Seizure Failure Modes:**
  - In `chb04` (electro-decremental seizure), conventional energy features collapse: **PR-AUC = 0.0028** (0/1 detected). Connectivity features achieve **PR-AUC = 0.5675** (1/1 detected, 0 false alarms).
  - In `chb15` (focal onset): Conventional PR-AUC = 0.0630 vs. Connectivity PR-AUC = 0.4072 ($+0.3442$ gain).
  - In `chb21` (longitudinal repeat): Conventional PR-AUC = 0.2070 vs. Connectivity PR-AUC = 0.4227 ($+0.2157$ gain).
- **Clinical Conclusion:** Phase synchrony captures physiological network reorganization during seizures even when local voltage amplitude drops.

---

### Slide 10: Dissecting Deep Learning False Alarms
- **The Empirical Findings:**
  - 1D CNN + EWMA produces 641 false alarms over 35.99 hours (17.81 FP/h).
  - **Burstiness Discovery:** **71.0% of false alarms occur within 60 seconds of a preceding alarm.**
  - **Outlier Concentration:** Top 3 cases (`chb07`, `chb15`, `chb11`) cause **61.9% of all false alarms**. Background non-seizure transients in these cases reach 2–4$\times$ ictal amplitude.
- **Causal Refractory Lockout Ablation:**
  - A 15-second lockout reduces false alarms by **19.3%** (down to 14.37 FP/h) with **zero loss of seizure sensitivity** (20 of 28 seizures retained).
  - Under strict leak-free inner-LOPO selection, false alarms drop to **13.61 FP/h** (19/28 seizures retained).

---

### Slide 11: Scientific Honesty: What Is Established vs. What Remains Work
- **Established Claims:**
  - End-to-end 1D CNN achieves superior ranking discrimination without feature engineering.
  - Functional connectivity rescues detector failure in electro-decremental seizures.
  - Causal EWMA provides a $+0.1226$ PR-AUC gain to deep learning.
- **Boundaries & Caveats:**
  - The model is **not ready for autonomous clinical deployment** (1.03–17.81 FP/h requires human-in-the-loop review).
  - Bipolar phase synchrony reflects statistical phase locking, not direct anatomical tract connectivity.
  - Evaluated on 24 recording cases (35.99 hours), not all 686 files.

---

### Slide 12: Honors Thesis Roadmap & Next Steps
- **Immediate Next Step (Post-Review):**
  - Implement a **Hybrid Spatial-Temporal Architecture** (coupling 1D CNN temporal kernels with a Graph Neural Network / PLI adjacency attention mechanism).
  - Goal: Retain the high ranking discrimination of CNNs while leveraging graph topology to reject localized non-ictal transients (<1 FP/h).
- **Thesis Completion Timeline:**
  - October 23: Faculty Progress Review.
  - November 15: Hybrid CNN-GNN implementation and benchmarking.
  - December 10: Complete Honors Thesis draft submission.
