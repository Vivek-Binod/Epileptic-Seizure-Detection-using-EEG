"""Generate publication-quality methodology flowchart for Honors EEG Seizure Detection."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "review_package_oct23" / "METHODOLOGY_FLOWCHART.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")

# Colors
c_data = "#2b5c8f"      # Slate blue
c_prep = "#1b7875"      # Teal
c_feat = "#d97724"      # Amber
c_cnn = "#8a3b8f"       # Purple
c_post = "#2e8b57"      # Sea green
c_eval = "#b22222"      # Firebrick
bg_box = "#f8f9fa"

def draw_box(ax, x, y, w, h, title, subtitle, color, ec=None):
    if ec is None:
        ec = color
    # Background shadow/fill
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor=bg_box, edgecolor=ec, linewidth=2.0
    )
    ax.add_patch(rect)
    # Header banner
    banner = patches.FancyBboxPatch(
        (x, y + h - 0.5), w, 0.5,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        facecolor=color, edgecolor=color, linewidth=1.0
    )
    ax.add_patch(banner)
    ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="center", color="white", fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + (h - 0.5) / 2, subtitle, ha="center", va="center", color="#222222", fontsize=8.5)

def draw_arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color="#444444", lw=1.8, mutation_scale=15)
    )
    if label:
        ax.text((x1 + x2) / 2 + 0.1, (y1 + y2) / 2, label, fontsize=8, color="#555555", style="italic")

# Row 1: Data & Preprocessing (Top)
draw_box(
    ax, 0.8, 7.8, 4.2, 1.8,
    "1. CHB-MIT Scalp EEG Dataset",
    "• 686 total raw EDFs (45.8 GB, 983 hrs)\n• 23 biological patients (24 cases)\n• chb01 & chb21 unified as patient_01\n• Strict 256 Hz uniform sampling",
    c_data
)

draw_box(
    ax, 5.9, 7.8, 4.2, 1.8,
    "2. Signal Preprocessing",
    "• 0.5 - 45.0 Hz zero-phase FIR bandpass\n• T8-P8 duplicate label disambiguation\n• Standard 18-channel bipolar montage\n  (Longitudinal 'Double Banana')",
    c_prep
)

draw_box(
    ax, 11.0, 7.8, 4.2, 1.8,
    "3. Windowing & Partitioning",
    "• 4.0-second non-overlapping windows (1024)\n• Training: 17,520 balanced paired windows\n• Continuous Eval: 24 full sessions\n  (35.99 hrs continuous, 129,564 windows)",
    c_prep
)

draw_arrow(ax, 5.0, 8.7, 5.9, 8.7)
draw_arrow(ax, 10.1, 8.7, 11.0, 8.7)

# Split from Step 3 down to Branches
draw_arrow(ax, 13.1, 7.8, 4.0, 6.3)
draw_arrow(ax, 13.1, 7.8, 11.5, 6.3)

# Row 2: Parallel Modeling Branches
# Branch A: Classical Feature Engineering
draw_box(
    ax, 0.8, 4.3, 6.5, 2.0,
    "4A. Multimodal Feature Engineering",
    "• Conventional (252): Band powers, SEF50-95, line length, Hjorth, spectral entropy\n• Functional Connectivity (200): PLV & PLI across Delta, Theta, Alpha, Beta\n• Graph Topology (40): Density, global efficiency, clustering, path length\n• Classifier: Random Forest (100 trees, balanced subsample, LOPO cross-validation)",
    c_feat
)

# Branch B: End-to-End Deep Learning
draw_box(
    ax, 8.7, 4.3, 6.5, 2.0,
    "4B. End-to-End Deep Learning (CompactEEGCNN)",
    "• Input: Raw bipolar tensor [Batch, 18 channels, 1024 samples]\n• Multi-scale temporal filter bank (Conv1D: k=33, 17, 9) + GELU + BatchNorm\n• Global Average Pooling + Dropout(0.40) + Linear Head (45,409 parameters)\n• GPU Optimization: AdamW, Cosine Annealing, pos_weight on RTX 3050 Laptop GPU",
    c_cnn
)

# Row 3: Temporal Post-Processing
draw_arrow(ax, 4.0, 4.3, 5.5, 2.9)
draw_arrow(ax, 11.9, 4.3, 10.5, 2.9)

draw_box(
    ax, 4.5, 1.6, 7.0, 1.4,
    "5. Causal Temporal Post-Processing",
    "• Exponential Weighted Moving Average (EWMA): p̂(t) = α·p(t) + (1-α)·p̂(t-1)\n• Eliminates high-frequency transient jitter without look-ahead bias\n• Causal Refractory Lockout (T_lockout): Suppresses repetitive alarm bursts in EMUs",
    c_post
)

# Row 4: Leak-Free Evaluation (Bottom)
draw_arrow(ax, 8.0, 1.6, 8.0, 0.8)

draw_box(
    ax, 1.5, 0.05, 13.0, 0.8,
    "6. Strict Leak-Free LOPO Evaluation Across Full 23-Patient Cohort",
    "23 Outer LOPO Folds • Zero Test Leakage (Normalization, pos_weight & Checkpoint Selection Isolated to Train) • 35.99 Continuous Hours • Collar Filter (±10s)",
    c_eval
)

# Add Title
fig.suptitle(
    "Honors EEG Seizure Detection: Complete Multimodal & Deep Learning Pipeline",
    fontsize=15, weight="bold", y=0.98, color="#1a202c"
)

plt.tight_layout()
fig.savefig(OUT_PATH, bbox_inches="tight")
plt.close(fig)
print(f"Flowchart saved successfully to {OUT_PATH}")
