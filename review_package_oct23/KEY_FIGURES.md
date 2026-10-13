# Key Figures Guide: Honors EEG Seizure Detection

This document catalogs the primary visual evidence prepared for the October 23, 2026 Faculty Research Review. Each figure has been verified on disk, generated from audited machine-readable experiment artifacts, and evaluated under leak-free Leave-One-Patient-Out (LOPO) cross-validation protocols.

---

### Figure 1: End-to-End Methodology Flowchart
- **Path:** [`review_package_oct23/METHODOLOGY_FLOWCHART.png`](file:///root/Honors-EEG/review_package_oct23/METHODOLOGY_FLOWCHART.png)
- **Explanation:** Publication-quality architectural diagram illustrating the complete processing pipeline from raw CHB-MIT EDF recordings and 18-channel bipolar montage construction through parallel feature-engineering and deep-learning branches to causal EWMA post-processing and collar evaluation.

---

### Figure 2: Model Paradigm Benchmark Comparison
- **Path:** [`outputs/experiments/exp_20261013_deep_learning_cnn/cnn_vs_baselines_comparison.png`](file:///root/Honors-EEG/outputs/experiments/exp_20261013_deep_learning_cnn/cnn_vs_baselines_comparison.png)
- **Explanation:** Comprehensive bar chart comparing Macro PR-AUC, Macro ROC-AUC, Event Sensitivity, and False Alarm rates across classical tree ensembles, graph-augmented models, and the compact 1D CNN across the full 23-patient cohort.

---

### Figure 3: Deep Learning Training & Validation Trajectories
- **Path:** [`outputs/experiments/exp_20261013_deep_learning_cnn/cnn_training_curves.png`](file:///root/Honors-EEG/outputs/experiments/exp_20261013_deep_learning_cnn/cnn_training_curves.png)
- **Explanation:** Epoch-by-epoch loss convergence and inner-validation PR-AUC curves across all 23 patient folds trained on the NVIDIA RTX 3050 Laptop GPU, illustrating early stopping checkpoint selection without test-set leakage.

---

### Figure 4: Ictal vs. Interictal Functional Network Adjacency Heatmaps
- **Path:** [`outputs/experiments/exp_20261013_full_cohort_networks/full_cohort_adjacency_heatmaps.png`](file:///root/Honors-EEG/outputs/experiments/exp_20261013_full_cohort_networks/full_cohort_adjacency_heatmaps.png)
- **Explanation:** Average $18 \times 18$ Phase Locking Value (PLV) and Phase Lag Index (PLI) functional connectivity matrices comparing seizure (ictal) and baseline (interictal) states across Delta, Theta, Alpha, and Beta frequency bands for all 23 biological patients.

---

### Figure 5: Dynamic Temporal Network Trajectories at Seizure Onset
- **Path:** [`outputs/experiments/exp_20261013_full_cohort_networks/full_cohort_temporal_trajectories.png`](file:///root/Honors-EEG/outputs/experiments/exp_20261013_full_cohort_networks/full_cohort_temporal_trajectories.png)
- **Explanation:** Window-by-window temporal evolution of Theta-band PLI connectivity and graph network density from 60 seconds pre-ictal through 60 seconds post-ictal, demonstrating rapid phase hypersynchronization at clinical seizure onset.

---

### Figure 6: Functional Connectivity and Graph Detector Ablation
- **Path:** [`outputs/experiments/exp_20261013_full_cohort_networks/full_cohort_detector_ablation_comparison.png`](file:///root/Honors-EEG/outputs/experiments/exp_20261013_full_cohort_networks/full_cohort_detector_ablation_comparison.png)
- **Explanation:** Full-cohort ablation comparing Conventional (252), Connectivity Edge (200), Graph Topology (40), and Combined (492) classifiers, highlighting complementary performance on focal and electro-decremental seizures.

---

### Figure 7: Classical Full-Cohort PR-AUC and ROC-AUC Generalization
- **Path:** [`outputs/experiments/exp_20261013_full_cohort/full_cohort_pr_roc_comparison.png`](file:///root/Honors-EEG/outputs/experiments/exp_20261013_full_cohort/full_cohort_pr_roc_comparison.png)
- **Explanation:** Per-case scatter and bar plots showing PR-AUC and ROC-AUC across all 24 cases, establishing that multimodal feature fusion outperforms conventional spectral features alone across heterogeneous clinical cases.

---

### Figure 8: False Alarm Suppression via Causal EWMA Post-Processing
- **Path:** [`outputs/experiments/exp_20261013_full_cohort/full_cohort_fp_per_hour.png`](file:///root/Honors-EEG/outputs/experiments/exp_20261013_full_cohort/full_cohort_fp_per_hour.png)
- **Explanation:** Case-by-case comparison of false alarms per hour before and after causal EWMA temporal smoothing, illustrating a 71.3% reduction in clinical false alarms across 36 hours of continuous monitoring.

---

### Figure 9: Bipolar Montage Exemplar Graph Topology
- **Path:** [`outputs/audit/example_window_network_plot.png`](file:///root/Honors-EEG/outputs/audit/example_window_network_plot.png)
- **Explanation:** Anatomical scalp layout of the 18 longitudinal bipolar derivation nodes showing significant functional connectivity edges during a verified clinical seizure window.
