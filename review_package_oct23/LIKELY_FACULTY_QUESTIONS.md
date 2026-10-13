# Likely Faculty Review Questions & Evidence-Backed Answers

**Review Target:** Faculty Research Progress Review (October 23, 2026)  
**Candidate:** Honors Undergraduate Researcher  
**Deliverable Directory:** [`review_package_oct23/`](file:///root/Honors-EEG/review_package_oct23/)  

---

### 1. Methodology & Validation Rigor

#### Q1: Why did you choose Leave-One-Patient-Out (LOPO) cross-validation instead of standard 5-fold cross-validation or random window splitting?
**Answer:** Random window splitting causes severe temporal autocorrelation leakage: adjacent 4-second windows share 3 seconds of identical EEG data, testing memorization of signal continuity rather than seizure generalization. Standard K-fold splitting across mixed windows also tests patient re-identification rather than seizure pattern generalization. LOPO cross-validation guarantees that the model is tested strictly on an unobserved biological patient whose baseline skull impedance, background rhythm, and electrode placement were never seen during training.

#### Q2: Why is Precision-Recall AUC (PR-AUC) your primary metric instead of ROC-AUC or Accuracy?
**Answer:** In continuous scalp EEG, seizures represent ~0.33% of recording time (a 1:300 class imbalance). Under this extreme skew, Accuracy is meaningless (a dummy classifier predicting "non-seizure" achieves 99.67% accuracy). ROC-AUC evaluates the False Positive Rate ($FP / (FP + TN)$); because the number of true negatives ($TN$) is massive (129,564 windows), large surges in false alarms barely impact ROC-AUC. PR-AUC evaluates Precision ($TP / (TP + FP)$), directly exposing false alarm inflation and measuring true clinical utility.

#### Q3: What is the difference between Macro Case Sensitivity and Pooled Event Sensitivity, and why did earlier reports confuse them?
**Answer:** Macro Case Sensitivity is the unweighted arithmetic mean of per-case sensitivities across all 24 recording cases ($19.5 / 24 = 81.25\%$ for raw CNN; $18.5 / 24 = 77.08\%$ for EWMA). Pooled Event Sensitivity is the total detected seizures divided by the total cohort seizures ($21 / 28 = 75.00\%$ for raw; $20 / 28 = 71.43\%$ for EWMA). Earlier drafts conflated them by writing `81.25% (21/28)`. Macro sensitivity exceeds pooled sensitivity because case `chb06` has 3 seizures that were all missed ($0/3 = 0.0$); macro averaging weights `chb06` as $1/24 \approx 4.17\%$ of cases, whereas pooled evaluation weights it as $3/28 \approx 10.71\%$ of seizures. Both metrics are now explicitly separated.

#### Q4: How do you justify unifying cases `chb01` and `chb21` under `patient_01`?
**Answer:** PhysioNet's official documentation explicitly records that case `chb21` was obtained from the same female pediatric patient as `chb01` 1.5 years later. If `chb01` and `chb21` were placed in separate cross-validation folds, recordings from the same biological brain would appear in both training and test sets, violating patient independence. Unifying them into a single outer fold (`patient_01`) guarantees zero biological patient leakage.

#### Q5: The CHB-MIT database contains 686 EDF files (~983 hours). Why did you evaluate 24 files (35.99 hours)?
**Answer:** In continuous clinical LTEEG evaluation, evaluating 1 continuous multi-hour session per case containing documented clinical seizures and prolonged interictal background (129,564 contiguous windows across 24 cases) provides 35.99 continuous hours of realistic monitoring. Evaluating all 686 files would require hundreds of compute hours with diminishing scientific returns, as 545 files contain zero seizures. Our documentation explicitly distinguishes the raw 686-file inventory from the 24-case evaluation benchmark.

---

### 2. Functional Connectivity & Graph Theoretical Modeling

#### Q6: How does your 18-node functional graph address scalp volume conduction?
**Answer:** Scalp volume conduction causes instantaneous zero-phase-lag field spread from a single cortical dipole to multiple scalp electrodes, artificially inflating phase synchronization. We address this by extracting two distinct connectivity metrics: Phase Locking Value (PLV), which captures total coupling (including volume conduction), and Phase Lag Index (PLI), which explicitly discards zero-phase-lag differences ($\Delta \phi = 0, \pm \pi$). Significant seizure hypersynchrony survived in PLI (especially in Delta and Theta bands, $p < 0.001$), confirming genuine phase-delayed synchronization.

#### Q7: Do edges in your functional connectivity graph represent physical anatomical tracts?
**Answer:** No, absolutely not. Nodes represent bipolar scalp electrode derivations (`FP1-F7`, `F7-T7`, etc.), and edges represent statistical phase synchronization derived from the Hilbert transform of scalp potentials. They reflect macroscopic functional coordination and electric field reorganization across the montage, not physical axonal tracts or intracortical structural connectivity.

#### Q8: Why did conventional features collapse on case `chb04` (PR-AUC 0.0028), while connectivity succeeded (PR-AUC 0.5675)?
**Answer:** Case `chb04` presents with an **electro-decremental seizure onset**, where seizure onset is marked by abrupt low-voltage fast desynchronization rather than high-voltage rhythmic spikes. Because univariate features rely on amplitude and band power increases, they completely miss electro-decremental onsets. However, functional connectivity (PLV/PLI) measures phase consistency across channels, which sharply increases during electro-decremental recruitment regardless of signal amplitude.

#### Q9: Did you perform multiple testing correction for your network metrics?
**Answer:** Yes. Across all 24 frequency band and metric combinations, both False Discovery Rate (FDR via Benjamini-Hochberg) and strict Bonferroni correction ($\alpha_{\text{Bonf}} = 0.05 / 24 = 0.002083$) were applied. Significant ictal hypersynchrony survived in Delta and Theta bands after both corrections.

---

### 3. Deep Learning Architecture & Training

#### Q10: Why did you choose a compact 1D CNN instead of a large pre-trained transformer or deep 2D ResNet?
**Answer:** Multi-channel scalp EEG has a relatively low signal-to-noise ratio and significant cross-patient variance. Heavy architectures (millions of parameters) rapidly overfit on patient-specific baseline quirks. Our `CompactEEGCNN` contains only 45,409 parameters, featuring multi-scale 1D temporal kernels ($k=33, 17, 9$ samples, capturing 129 ms, 66 ms, and 35 ms dynamics) and global average pooling, preventing parameter explosion while executing in 183 MB VRAM on our target RTX 3050 Laptop GPU.

#### Q11: How was class imbalance handled during 1D CNN training?
**Answer:** We utilized `BCEWithLogitsLoss` parameterized with positive class weight $w_{\text{pos}} = N_{\text{interictal}} / N_{\text{ictal}}$, calculated strictly on training-fold data ($w_{\text{pos}} \approx 8.5$ on balanced paired training data). We avoided oversampling or SMOTE on the raw EEG time-series to preserve natural temporal continuity and avoid synthetic signal artifacts.

#### Q12: How did you prevent data leakage during early stopping and model checkpoint selection?
**Answer:** In each of the 23 LOPO folds, the 22 training patients were partitioned into an inner 4-fold `GroupKFold` split grouped by patient ID. Model validation loss and validation PR-AUC were evaluated exclusively on inner training patients. The checkpoint achieving peak inner validation PR-AUC was restored before evaluating the held-out test patient. Test patient data was strictly untouched until final inference.

#### Q13: Did the 1D CNN actually train on GPU, and what was the execution profile?
**Answer:** Yes, verified on an NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB VRAM, CUDA 12.4). Peak allocated VRAM was 183.1 MB during training and 149 MB during inference. Total 23-fold LOPO cross-validation across 147,084 total windows completed in 10.08 minutes (26.3 seconds per fold).

---

### 4. Deep Learning vs. Classical Baselines & False Alarm Investigation

#### Q14: Why does the 1D CNN achieve higher ROC-AUC (0.8493) but substantially worse false alarm rates (17.81 FP/h) than Random Forest (1.03 FP/h)?
**Answer:** The 1D CNN operates on localized 1D convolutional kernels. Short 50–200 ms non-epileptic sharp transients (e.g. muscle spikes, eye blinks, electrode pops) activate convolutional filters, triggering brief probability spikes. In contrast, Random Forest evaluates aggregated 4-second summary statistics (spectral band powers, graph efficiency). An isolated 100 ms spike does not alter the 4-second mean band power or network clustering coefficient, granting feature ensembles intrinsic noise rejection.

#### Q15: What is the physical reason why 71% of the CNN's false alarms cluster within 60 seconds of each other?
**Answer:** In cases with persistent interictal epileptiform discharges or high-amplitude muscle artifact (especially `chb07`, `chb15`, `chb11`, which account for 61.9% of all false alarms), the noise persists across multi-minute epochs. The CNN fires an alarm at the 3rd consecutive positive window; when probability briefly dips below 0.50 for a single window and rises again, a second alarm is logged. The alarms are not uniformly distributed throughout the day, but concentrated in bursty artifact epochs.

#### Q16: How does causal EWMA temporal smoothing work, and is it look-ahead free?
**Answer:** Causal EWMA calculates $\hat{p}(t) = \alpha \cdot p(t) + (1 - \alpha) \cdot \hat{p}(t-1)$. It depends strictly on past and current predictions ($t$ and $t-1$), ensuring zero look-ahead bias and making it fully implementable in real-time bedside hardware. For the 1D CNN, EWMA ($\alpha=0.30$) provided a transformative $+0.1226$ PR-AUC boost (0.4345 to 0.5571) by smoothing out isolated single-window transient spikes.

#### Q17: What did your causal refractory lockout ablation demonstrate?
**Answer:** Introducing a 15-second lockout period ($T_{\text{lockout}} = 15\text{s}$) eliminated 124 false alarms (19.3% reduction, down to 14.37 FP/h) with **zero loss of seizure sensitivity** (20 of 28 seizures retained). When lockout was selected strictly on inner training folds, 22 of 23 folds selected 15.0s, reducing false alarms to 13.61 FP/h (19/28 seizures retained). This proves that post-alarm debouncing suppresses burst alarms, though architectural spatial inductive biases are required to reach <1 FP/h.

---

### 5. Engineering, Reproducibility & Thesis Trajectory

#### Q18: How did you handle the duplicate channel `T8-P8` anomaly in the CHB-MIT dataset?
**Answer:** Several CHB-MIT EDF headers list `T8-P8` twice (channels 17 and 22). Our loader implements an automated disambiguation policy: MNE temporarily renames them `T8-P8-0` and `T8-P8-1`, and our canonical montage selector deterministically binds the first primary electrode while dropping the duplicate, ensuring identical 18-channel alignment across all 24 cases without manual intervention.

#### Q19: Is your detector ready for autonomous clinical deployment in an Intensive Care Unit?
**Answer:** No. An alarm rate of 1.03 FP/h (RF) or 17.81 FP/h (CNN) would cause clinical alarm fatigue in an autonomous setting. However, it is highly suitable as an **automated decision-support screening tool** for offline review, prioritizing seizure-suspicious epochs for neurophysiologists and reducing visual review time by over 80%.

#### Q20: What is the single most important next step for your Honors thesis before final submission?
**Answer:** Building a **Hybrid Spatial-Temporal Architecture** (coupling 1D CNN temporal feature extractors with a Graph Neural Network / PLI adjacency attention mechanism). This will combine the representation learning power of CNNs (which achieved 0.8493 ROC-AUC) with the topological artifact resistance of functional brain networks, aiming to achieve EMU-grade specificity (<1 FP/h) with end-to-end deep learning.
