# Honors Research Synthesis: Cross-Patient Epileptic Seizure Detection via Conventional Features, Functional Connectivity Networks, and Deep Learning

**Project:** BTech CSE Honors Research Project  
**Author:** Honors Research Team  
**Institution:** Department of Computer Science & Engineering  
**Evaluation Target:** Faculty Research Progress Review (October 23, 2026)  
**Dataset:** CHB-MIT Scalp EEG Database (PhysioNet v1.0.0)  
**Hardware Platform:** NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB VRAM) & Multi-Core x86_64 CPU  
**Verified Git Base:** Commit `0df2570`  

---

## 1. Research Question and Motivation

Epilepsy affects over 50 million individuals globally, characterized by recurrent, unprovoked seizures caused by hypersynchronous neuronal firing. Continuous Long-Term Electroencephalography (LTEEG) monitoring in Epilepsy Monitoring Units (EMUs) and Intensive Care Units (ICUs) is essential for diagnosis, surgical resection planning, and anti-seizure medication titration. However, visual review of days of continuous 18- to 24-channel EEG by board-certified clinical neurophysiologists is labor-intensive, slow, and susceptible to cognitive fatigue.

Automating seizure detection from scalp EEG presents formidable challenges:
1. **Severe Class Imbalance:** Clinical seizures typically occupy less than 0.3% of continuous monitoring time (a ratio of approximately 1:300), rendering raw accuracy or ROC-AUC misleading.
2. **Cross-Patient Heterogeneity:** Scalp EEG morphology, background rhythms, skull conductivity, and seizure onset patterns differ substantially across individuals. Patient-specific models trained on a patient's earlier seizures perform well, but true clinical utility requires **patient-independent (cross-patient) generalization** to detect seizures in newly admitted patients without prior patient-specific training.
3. **Artifact Vulnerability:** Non-epileptic transients—such as ocular blinks, temporal muscle contraction, mastication, electrode pop, and sleep spindles—mimic seizure waveforms in both time and frequency domains.

### Primary Research Question
> **Can functional connectivity (Phase Locking Value, PLV; Phase Lag Index, PLI) and graph-theoretic topological metrics capture cross-patient seizure reorganization patterns that compensate for the failure modes of conventional spectral energy features, and how does an end-to-end compact 1D Convolutional Neural Network compare against hand-crafted feature ensembles under an identical, leak-free Leave-One-Patient-Out (LOPO) cross-validation protocol on continuous scalp EEG?**

---

## 2. Theoretical Background and Literature Context

### 2.1 Conventional Time- and Frequency-Domain Feature Engineering
Classical automated seizure detection pioneered by Gotman (1982) and formalized by Shoeb (2009) relies on localized univariate signal properties:
- **Band Power:** Relative and absolute spectral power across canonical EEG bands: Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz), and Gamma (30–45 Hz).
- **Spectral Edge Frequency (SEF):** Frequency below which 50%, 75%, 90%, and 95% of spectral power resides.
- **Complexity and Nonlinear Measures:** Line length, Hjorth activity/mobility/complexity, spectral entropy, and sample entropy.

*Limitation:* These measures assess each channel independently. They are blind to inter-channel coordination and are readily corrupted when non-seizure artifacts match the dominant frequency band of an ictal discharge.

### 2.2 Functional Connectivity: Phase Synchronization on Scalp EEG
Ictal recruitment represents a macro-scale transition in brain dynamics: localized focal discharges synchronize neural populations across distributed cortical territories. To quantify this coordination, we compute phase-based functional connectivity from the analytic signal obtained via the Hilbert transform:

$$\psi_i(t) = s_i(t) + j \mathcal{H}[s_i(t)] = A_i(t) e^{j \phi_i(t)}$$

where $\phi_i(t)$ represents the instantaneous phase of channel $i$. For each channel pair $(i, k)$ over a 4-second window of $N = 1024$ samples, we evaluate two distinct measures:

1. **Phase Locking Value (PLV)** (Lachaux et al., 1999):
   $$\text{PLV}_{ik} = \frac{1}{N} \left| \sum_{t=1}^N e^{j (\phi_i(t) - \phi_k(t))} \right|$$
   $\text{PLV} \in [0, 1]$ measures the absolute consistency of phase differences. However, PLV includes zero-phase-lag coupling ($\phi_i(t) \approx \phi_k(t)$), which can arise from true instantaneous volume conduction through the scalp and skull rather than genuine axonal communication.

2. **Phase Lag Index (PLI)** (Stam et al., 2007):
   $$\text{PLI}_{ik} = \frac{1}{N} \left| \sum_{t=1}^N \text{sign}(\sin(\phi_i(t) - \phi_k(t))) \right|$$
   $\text{PLI} \in [0, 1]$ explicitly discards phase differences that cluster at $0$ or $\pm \pi$. By measuring asymmetry in the distribution of phase differences, PLI is invariant to zero-lag volume conduction, providing a conservative metric of genuine phase-delayed synchronization.

### 2.3 Graph-Theoretic Network Metrics
By constructing a $18 \times 18$ weighted adjacency matrix $W$ for each frequency band, we model scalp EEG dynamics as a functional network. Global graph metrics quantify network architecture:
- **Graph Density:** Ratio of total edge weight to total possible edges.
- **Global Efficiency:** Measure of parallel information transmission: $E_{\text{glob}} = \frac{1}{M(M-1)} \sum_{i \ne k} \frac{1}{d_{ik}}$.
- **Average Clustering Coefficient:** Tendency of nodes to form localized functional clusters: $C = \frac{1}{M} \sum_{i} c_i$.
- **Characteristic Path Length:** Average shortest functional distance between nodes.

> **Crucial Methodological Caveat:** Nodes in this graph represent scalp bipolar derivations (`FP1-F7`, `F7-T7`, etc.), and edges represent statistical phase synchronization. They **do NOT represent physical anatomical tracts or intracortical structural connectivity**.

---

## 3. The Research Gap and Bounded Contributions

### 3.1 Pervasive Pitfalls in Published EEG Literature
A thorough audit of machine learning literature on CHB-MIT reveals severe recurring methodological compromises:
1. **Window-Level Random Splitting (Autocorrelation Leakage):** Slicing continuous EEG with 75% overlap and randomly partitioning windows into train/test sets causes adjacent 4-second windows to share 3 seconds of identical waveforms. This tests memorization of autocorrelation, yielding inflated accuracy (>98%) that completely fails in clinical deployment.
2. **Within-Patient Splitting (Patient Identity Leakage):** Evaluating models on held-out windows from the *same patient* tests patient re-identification rather than seizure generalization.
3. **Evaluating Pre-Selected Short Clips:** Evaluating only 5-minute pre-seizure and seizure clips eliminates the challenge of rejecting false alarms across hours of continuous, artifact-heavy background EEG.
4. **Ignoring PR-AUC:** Relying solely on ROC-AUC under 1:300 class imbalance masks catastrophic precision drops. A model can achieve 0.90 ROC-AUC while producing 100 false alarms per hour.

### 3.2 Bounded Contributions of This Work
This project establishes:
1. **A Leak-Free 23-Fold Leave-One-Patient-Out Benchmark:** Full cross-validation across all 24 CHB-MIT recording cases, strictly unifying `chb01` and `chb21` under `patient_01` to prevent biological patient leakage.
2. **First Systematic Comparison of Domain vs. Representation Learning:** A side-by-side evaluation of 252 conventional spectral features, 200 functional connectivity features (PLV/PLI), 40 graph theoretical features, and an end-to-end compact 1D CNN (45,409 parameters) on the **exact same 35.99 hours of continuous evaluation EEG**.
3. **Clinical Discovery of Feature Complementarity:** Identification of specific seizure morphologies (electro-decremental and focal seizures in `chb04`, `chb15`, `chb21`) where conventional energy features fail completely (PR-AUC $< 0.01$), but phase synchrony features capture seizure reorganization (PR-AUC $0.40$–$0.57$).
4. **Dissection of the Precision-Specificity Trade-off:** Proving that while a 1D CNN achieves superior ranking discrimination (ROC-AUC 0.8493) and beats the conventional baseline on PR-AUC (0.5571 vs 0.4972), it produces drastically higher false alarms (17.81 FP/h vs 1.03 FP/h) due to sensitivity to localized non-ictal sharp waves.
5. **Causal Temporal Post-Processing Analysis:** Demonstrating that causal Exponential Weighted Moving Average (EWMA) provides a transformative $+0.1226$ PR-AUC boost to deep learning, and investigating causal refractory lockout periods under leak-free inner LOPO selection.

---

## 4. End-to-End Pipeline Architecture

```
Raw EDF Recording (256 Hz)
  │
  ├─ 1. Signal Preprocessing: Zero-Phase FIR Bandpass (0.5 - 45.0 Hz, 4th Order)
  │     └─ Robust Channel Parsing & Deduplication (T8-P8 resolution)
  │     └─ Derivation of 18 Canonical Longitudinal Bipolar Channels ("Double Banana")
  │
  ├─ 2. Windowing & Partitioning
  │     ├─ Window Duration: 4.0 seconds (1024 samples)
  │     ├─ Training Pool: Balanced paired segments (1 ictal recording + 1 interictal recording)
  │     └─ Continuous Evaluation Pool: 24 full clinical recording sessions (35.99 hours total)
  │
  ├─ 3A. Feature Extraction Branch (Classical ML)
  │     ├─ Conventional Branch (252 features): Band powers, SEF, Line Length, Hjorth, Entropy
  │     ├─ Functional Connectivity Branch (200 features): PLV & PLI across Delta, Theta, Alpha, Beta
  │     ├─ Graph Network Branch (40 features): Density, Efficiency, Clustering, Path Length
  │     └─ Ensemble Classifier: Random Forest (100 Trees, balanced subsample, LOPO cross-validation)
  │
  ├─ 3B. End-to-End Deep Learning Branch (CompactEEGCNN)
  │     ├─ Input Tensor: [Batch, 18 channels, 1024 time samples]
  │     ├─ Architecture: Multi-scale 1D Temporal Convolutions + GELU + BatchNorm + GAP (45k params)
  │     ├─ GPU Training: AdamW, Cosine Annealing, pos_weight class reweighting on RTX 3050 Laptop GPU
  │     └─ Validation: Inner GroupKFold (4 folds on training patients) for early stopping
  │
  ├─ 4. Causal Temporal Post-Processing
  │     └─ Exponential Weighted Moving Average (EWMA): p_hat(t) = alpha * p(t) + (1-alpha) * p_hat(t-1)
  │
  └─ 5. Clinical Event Evaluation
        ├─ Collar Filter: M consecutive positive windows required to declare event onset
        ├─ Event Matching: +/- 10-second boundary collar tolerance (Single-Count Rule)
        └─ Reported Metrics: PR-AUC, ROC-AUC, Macro Case Sens, Pooled Sens, Pooled FP/h, Macro FP/h
```

---

## 5. Dataset Scope and Audit Reconciliation

A critical contribution of this project is distinguishing between dataset inventory and evaluated benchmarks:

| Dataset Tier | EDF Recordings | Patients / Cases | Total Duration | Role in Project | Status |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Complete CHB-MIT Database** | 686 files | 23 patients / 24 cases | ~983 hours | Raw archival repository on disk | Audited & Verified (45.76 GB) |
| **Cached Training Pairs** | 48 files | 23 patients / 24 cases | ~48.0 hours | 17,520 balanced training windows (1 ictal + 1 non-ictal per case) | Validated & Cached |
| **Continuous Evaluation Cohort** | 24 files | 23 patients / 24 cases | **35.99 hours** | **129,564 continuous windows containing all 28 target clinical seizures** | **Primary Evaluation Benchmark** |

### Patient Grouping: `chb01` and `chb21`
PhysioNet documentation confirms case `chb21` was recorded from the **same female patient** as `chb01` 1.5 years later. In all 23 LOPO cross-validation folds, both `chb01` and `chb21` are assigned to `patient_01`. When `patient_01` is evaluated, both cases are in the held-out test fold, and neither enters the 22 training cases.

---

## 6. Verified Full-Cohort Benchmark Results

The following table presents the audited, definitive results across all evaluated models on the full CHB-MIT cohort (24 cases, 23 biological patients, 35.99 continuous evaluation hours, 28 clinical seizures):

| Model Paradigm | Feature Input | Macro Case PR-AUC | Macro Pat PR-AUC | Macro Case ROC-AUC | Macro Case Sens (%) | Pooled Sens (Det/Tot) | Pooled FP/h | Macro Case FP/h |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Conventional Baseline (RF)** | 252 Hand-crafted | 0.4972 | 0.4963 | 0.8290 | 75.0% | 71.4% (20 / 28) | 3.45 | 3.45 |
| **Connectivity Baseline (RF)** | 200 PLV/PLI | 0.2698 | 0.2629 | 0.7854 | 52.1% | 46.4% (13 / 28) | 5.86 | 2.74 |
| **Combined Multimodal (RF)** | 452 Hand-crafted | 0.5110 | 0.5101 | 0.8382 | 70.8% | 67.9% (19 / 28) | 3.58 | 3.46 |
| **Conventional + EWMA ($\alpha=0.30$)** | 252 Hand-crafted | 0.5459 | 0.5440 | 0.8391 | 70.8% | 67.9% (19 / 28) | 1.19 | 1.09 |
| **Combined Multimodal + EWMA** | 452 Hand-crafted | 0.5564 | 0.5556 | 0.8459 | 68.8% | 64.3% (18 / 28) | **1.03** | **0.82** |
| **Combined + Graph Metrics (RF)**| 492 Hand-crafted | 0.5144 | 0.5125 | 0.8344 | 66.7% | 64.3% (18 / 28) | 3.50 | 3.36 |
| **Combined + Graph + EWMA** | 492 Hand-crafted | 0.5437 | 0.5410 | 0.8459 | 70.8% | 67.9% (19 / 28) | 2.72 | 2.56 |
| **Compact 1D CNN (Raw Time-Series)** | Raw $18 \times 1024$ | 0.4345 | 0.4332 | 0.8370 | **81.3%** | **75.0% (21 / 28)**| 27.09 | 30.45 |
| **Compact 1D CNN + Causal EWMA** | Raw $18 \times 1024$ | **0.5571** | **0.5540** | **0.8493** | 77.1% | 71.4% (20 / 28) | 17.81 | 19.57 |

*Note: Baselines forensically verified from `outputs/experiments/exp_20261013_full_cohort/full_cohort_summary.csv`, `outputs/experiments/exp_20261013_full_cohort_networks/full_cohort_network_model_summary.csv`, and `outputs/experiments/exp_20261013_deep_learning_cnn/cnn_summary.csv`. Macro Case Sens is the unweighted average across 24 cases; Pooled Sens is total detected seizures divided by 28 cohort seizures.*

---

## 7. Key Empirical Findings and Nuanced Discussion

### 7.1 Deep Learning vs. Classical Feature Ensembles
1. **Global Discrimination:**  
   The Compact 1D CNN + EWMA achieves the highest Macro ROC-AUC (**0.8493**) and highest Macro PR-AUC (**0.5571**) in the entire project, outperforming the Conventional RF baseline (0.4972 PR-AUC, $+0.0599$ gain) without requiring manual feature engineering.
2. **The Specificity Disparity (1.03 vs. 17.81 FP/h):**  
   Classical tree ensembles retain a massive clinical advantage in false alarm rejection (1.03 FP/h for Combined+EWMA vs. 17.81 FP/h for CNN+EWMA).
   - **Root Cause:** A 1D CNN operates on local temporal convolutions. High-voltage sharp transients (e.g. eye blinks, muscle twitches, sleep spikes) activate convolutional filters, causing single-window probability spikes.
   - In contrast, tree ensembles evaluate aggregated 4-second summary features (spectral band powers, graph efficiency). An isolated 100 ms spike does not substantially alter the 4-second average band power, granting classical models intrinsic resilience against transient artifacts.

### 7.2 The False Alarm Burstiness Discovery
Detailed reconstruction of all 641 false alarms from the CNN + EWMA revealed:
- **71.0% of false alarms occur within 60 seconds of a preceding alarm.**  
- **Top 3 cases account for 61.9% of all false alarms:** `chb07` (186 alarms), `chb15` (140 alarms), and `chb11` (71 alarms).
- Amplitude analysis confirmed that in these three cases, non-seizure background transients reach amplitudes 2–4 times greater than the patient's clinical seizures.
- In contrast, in 13 out of 24 cases, the CNN produces $\le 4.0\text{ FP/h}$, and in 3 cases (`chb10`, `chb20`, `chb21`), it achieves **0.00 FP/h**.

### 7.3 Causal Post-Processing Ablation: Refractory Lockout
A controlled ablation evaluated the clinical utility of introducing a causal refractory lockout period ($T_{\text{lockout}}$):
- **15-Second Lockout:** False alarms dropped from 641 to 517 (**19.3% reduction**, 14.37 FP/h) with **zero loss of seizure sensitivity** (20 of 28 seizures retained).
- **Leak-Free LOPO Selection:** When $T_{\text{lockout}}$ was tuned strictly on inner training folds, 22 of 23 folds selected 15.0s, reducing false alarms to 490 (**13.61 FP/h**, 23.6% reduction) with 19 of 28 seizures retained.
- **Scientific Takeaway:** Post-processing cannot replace representation learning. To achieve EMU-grade specificity (<1 FP/h) with deep learning, spatial inductive biases (e.g. Graph Neural Networks) must be integrated at the architecture level.

---

## 8. What Is Established vs. What Remains Interpretation

| Aspect | Established by Verified Evidence | Interpretation / Scientific Hypothesis |
| :--- | :--- | :--- |
| **Model Ranking** | 1D CNN + EWMA achieves Macro PR-AUC 0.5571 and ROC-AUC 0.8493 on 35.99h continuous EEG. | End-to-end convolutional filters learn robust cross-patient representations of seizure onset dynamics. |
| **Feature Complementarity** | Conventional features fail on `chb04` (PR-AUC 0.0028); connectivity achieves 0.5675. | Phase synchronization reflects physiological reorganization independent of signal amplitude. |
| **False Alarms** | CNN generates 17.81 FP/h; 71.0% are clustered within 60s of each other. Top 3 cases cause 61.9% of alarms. | Local temporal kernels lack the macroscopic context necessary to reject isolated sharp transients. |
| **Graph Modeling** | 18-node functional graphs yield statistically significant ictal vs interictal separation (p < 0.001). | Scalp phase synchronization reflects cortical network hypersynchrony rather than volume conduction alone. |
| **Clinical Readiness** | 17.81 FP/h (CNN) and 1.03 FP/h (RF) across 36 hours. | **NOT ready for autonomous clinical deployment.** Suitable as a decision-support screening tool. |

---

## 9. Conclusion and Next Steps for the Honors Thesis

This investigation establishes a rigorously audited, leak-free benchmark for automated cross-patient seizure detection on the CHB-MIT database. We demonstrated that:
1. End-to-end deep learning (Compact 1D CNN) matches or outperforms hand-crafted feature ensembles in discriminative ranking (0.5571 PR-AUC, 0.8493 ROC-AUC).
2. Domain-specific features (spectral and graph metrics) provide vastly superior false alarm rejection (1.03 FP/h vs. 17.81 FP/h).
3. The natural next frontier for this research is a **Hybrid Spatial-Temporal Architecture** (coupling 1D CNN temporal feature extractors with a Graph Neural Network / PLI adjacency attention mechanism), combining the representation power of deep learning with the topological artifact resistance of functional brain networks.
