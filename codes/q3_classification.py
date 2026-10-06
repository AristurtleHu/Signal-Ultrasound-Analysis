"""Q3: classify classroom ultrasound recordings with reference-set decision rules.

Method (details in the report):
  * Feature extraction is identical for every recording and every set.
  * The analytic-signal (Hilbert) envelope is smoothed with a centered
    five-point average. Let a_main be its global maximum.
  * a_early is the maximum envelope in [5, 22) us and t_early its time.
  * doublet_ratio is the maximum envelope in [t_early + 2.1, t_early + 3.4] us
    divided by a_early (detects the strong second surface echo of the acrylic).
  * late_ratio is the maximum envelope for t >= 22 us divided by a_early.

Decision rules chosen from the labeled reference set (explicit numbers):
  1. doublet_ratio > 0.15            -> class 1 (acrylic, double echo)
  2. else a_main < 0.70              -> class 3 (kitchen paper, weak echoes)
  3. otherwise                       -> class 2 (sandpaper, single strong echo)

Inputs : data/q3/{reference,stage1,stage2}
Outputs: predictions_stage1.csv, predictions.csv and Q3 figures in image/
Run    : conda run -n ai python codes/q3_classification.py
"""
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend so figures are written to image/
import matplotlib.pyplot as plt
from scipy.signal import hilbert, stft
from scipy.ndimage import uniform_filter1d
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_score

from hw1_common import ROOT, load_q3_set, save_fig, to_db

FS_Q3 = 26.5216e6
N_SAMPLES = 896
TIME_US = np.arange(N_SAMPLES) / FS_Q3 * 1e6

# Thresholds of the decision rules (chosen from the reference-set statistics).
DOUBLET_THRESHOLD = 0.15
A_MAIN_THRESHOLD = 0.70

# Echo-window definitions in microseconds.
EARLY_LO, EARLY_HI = 5.0, 22.0
LATE_LO = 22.0
DOUBLET_LO, DOUBLET_HI = 2.1, 3.4     # offsets from t_early


def echo_features(x):
    """Recording-level echo-structure features (same procedure for all signals)."""
    envelope = uniform_filter1d(np.abs(hilbert(x)), size=5, mode="reflect")
    early_mask = (TIME_US >= EARLY_LO) & (TIME_US < EARLY_HI)
    late_mask = TIME_US >= LATE_LO
    early = envelope[early_mask]
    late = envelope[late_mask]
    t_early = TIME_US[early_mask][np.argmax(early)]
    a_early = float(early.max())
    a_late = float(late.max())
    doublet_mask = (TIME_US >= t_early + DOUBLET_LO) & (TIME_US <= t_early + DOUBLET_HI)
    doublet_ratio = float(envelope[doublet_mask].max() / a_early)
    return {
        "a_main": max(a_early, a_late),
        "a_early": a_early,
        "a_late": a_late,
        "late_ratio": a_late / a_early,
        "t_early_us": float(t_early),
        "doublet_ratio": doublet_ratio,
    }


def apply_rules(features):
    """Apply the frozen decision rules to one feature dictionary."""
    if features["doublet_ratio"] > DOUBLET_THRESHOLD:
        return 1
    if features["a_main"] < A_MAIN_THRESHOLD:
        return 3
    return 2


# ---------------------------------------------------------------------------
# Load the labeled reference set and both task stages
# ---------------------------------------------------------------------------
reference_meta, reference_signals = load_q3_set("reference", "labels.csv")
stage1_meta, stage1_signals = load_q3_set("stage1", "index.csv")
stage2_meta, stage2_signals = load_q3_set("stage2", "index.csv")
reference_labels = np.array([int(row["true_class"]) for row in reference_meta])

ref_features = [echo_features(x) for x in reference_signals]
s1_features = [echo_features(x) for x in stage1_signals]
s2_features = [echo_features(x) for x in stage2_signals]

# ---------------------------------------------------------------------------
# Descriptive statistics of the features for each reference class
# ---------------------------------------------------------------------------
feature_names = ["a_main", "a_early", "a_late", "late_ratio", "t_early_us", "doublet_ratio"]
print("--- Reference-set feature statistics (mean +/- std) ---")
print(f'{"feature":14s} {"class 1":>18s} {"class 2":>18s} {"class 3":>18s}')
for key in feature_names:
    line = f"{key:14s} "
    for c in (1, 2, 3):
        values = np.array([f[key] for f, lab in zip(ref_features, reference_labels) if lab == c])
        line += f"{values.mean():8.3f} +/- {values.std():6.3f} "
    print(line)

print("\nThreshold margins (reference set)")
print(f"  doublet_ratio: class 1 min {min(f['doublet_ratio'] for f, l in zip(ref_features, reference_labels) if l == 1):.3f}, "
      f"classes 2+3 max {max(f['doublet_ratio'] for f, l in zip(ref_features, reference_labels) if l != 1):.3f}, "
      f"rule threshold {DOUBLET_THRESHOLD}")
print(f"  a_main       : class 3 max {max(f['a_main'] for f, l in zip(ref_features, reference_labels) if l == 3):.3f}, "
      f"classes 1+2 min {min(f['a_main'] for f, l in zip(ref_features, reference_labels) if l != 3):.3f}, "
      f"rule threshold {A_MAIN_THRESHOLD}")

# ---------------------------------------------------------------------------
# Validation on the labeled reference set
# ---------------------------------------------------------------------------
reference_predictions = np.array([apply_rules(f) for f in ref_features])
accuracy = float(np.mean(reference_predictions == reference_labels))
print(f"\nManual rules reference accuracy: {accuracy * 100:.1f}% "
      f"({int((reference_predictions == reference_labels).sum())}/{len(reference_labels)})")

# Optional cross-check: LDA on three echo features, leave-one-out.
X_ref = np.array([[f["a_main"], f["late_ratio"], f["doublet_ratio"]] for f in ref_features])
lda_accuracy = float(cross_val_score(
    LinearDiscriminantAnalysis(), X_ref, reference_labels, cv=LeaveOneOut()).mean())
print(f"LDA leave-one-out accuracy (cross-check): {lda_accuracy * 100:.1f}%")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# 1) Example signals and envelopes, four per class (reference set).
fig, axes = plt.subplots(3, 4, figsize=(15, 7.5), sharex=True, sharey=True,
                         constrained_layout=True)
for row, c in enumerate((1, 2, 3)):
    indices = [i for i, lab in enumerate(reference_labels) if lab == c][:4]
    for col, i in enumerate(indices):
        ax = axes[row, col]
        envelope = uniform_filter1d(np.abs(hilbert(reference_signals[i])), size=5, mode="reflect")
        ax.plot(TIME_US, reference_signals[i], lw=0.6, color="tab:blue", alpha=0.65)
        ax.plot(TIME_US, envelope, lw=1.1, color="tab:red")
        ax.set_ylim(-1.45, 1.45)
        if row == 0:
            ax.set_title(reference_meta[i]["sample_id"], fontsize=9)
        if col == 0:
            ax.set_ylabel(f"class {c}\namplitude")
axes[2, 0].set_xlabel(r"time ($\mu$s)")
axes[2, 3].set_xlabel(r"time ($\mu$s)")
fig.suptitle("Reference recordings: signal and smoothed Hilbert envelope")
save_fig(fig, "q3_reference_signals.png")

# 2) Feature scatter for the decision rule (all sets) and class boxplots.
def feature_array(features, key):
    return np.array([f[key] for f in features])


fig = plt.figure(figsize=(13, 5), constrained_layout=True)
grid = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1])

ax = fig.add_subplot(grid[0, 0])
colours = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
for c in (1, 2, 3):
    mask = reference_labels == c
    ax.scatter(feature_array(ref_features, "a_main")[mask],
               feature_array(ref_features, "doublet_ratio")[mask],
               c=colours[c], s=55, edgecolor="k", zorder=3, label=f"reference class {c}")
ax.scatter(feature_array(s1_features, "a_main"), feature_array(s1_features, "doublet_ratio"),
           marker="^", facecolor="none", edgecolor="red", s=80, label="stage 1")
ax.scatter(feature_array(s2_features, "a_main"), feature_array(s2_features, "doublet_ratio"),
           marker="s", facecolor="none", edgecolor="purple", s=45, alpha=0.7, label="stage 2")
ax.axhline(DOUBLET_THRESHOLD, color="gray", ls="--", lw=1.0)
ax.axvline(A_MAIN_THRESHOLD, color="gray", ls="--", lw=1.0)
ax.set_yscale("log")
ax.set_xlabel("a_main (envelope maximum)")
ax.set_ylabel("doublet_ratio (secondary / main echo)")
ax.set_title("Decision plane")
ax.grid(alpha=0.3)
ax.legend(fontsize=7, loc="lower right")

for panel, (key, ylabel) in enumerate([("a_main", "a_main"),
                                       ("doublet_ratio", "doublet_ratio")], start=1):
    ax = fig.add_subplot(grid[0, panel])
    data = [feature_array(ref_features, key)[reference_labels == c] for c in (1, 2, 3)]
    bp = ax.boxplot(data, tick_labels=["1", "2", "3"], widths=0.5, patch_artist=True)
    for patch, c in zip(bp["boxes"], (1, 2, 3)):
        patch.set_facecolor(colours[c])
        patch.set_alpha(0.55)
    for c in (1, 2, 3):
        ax.scatter(np.full(len(s1_features), c) + 0.12,
                   feature_array(s1_features, key), marker="^", facecolor="none",
                   edgecolor="red", s=25)
        ax.scatter(np.full(len(s2_features), c) - 0.12,
                   feature_array(s2_features, key), marker="s", facecolor="none",
                   edgecolor="purple", s=15, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("reference class")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{key} by class (open: stages)")
    ax.grid(alpha=0.3, axis="y")
save_fig(fig, "q3_features.png")

# 3) Example time-frequency views, one reference recording per class, dB display.
example_indices = [int(np.where(reference_labels == c)[0][0]) for c in (1, 2, 3)]
fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, sharey=True,
                         constrained_layout=True)
spectrograms = []
for i in example_indices:
    freq, time_s, z = stft(reference_signals[i], fs=FS_Q3, window="hann",
                           nperseg=64, noverlap=56, nfft=1024, detrend=False,
                           boundary=None, padded=False, scaling="spectrum")
    spectrograms.append((freq, time_s, np.abs(z)))
common_ref = max(s[2].max() for s in spectrograms)
for ax, (freq, time_s, magnitude), c, i in zip(axes, spectrograms, (1, 2, 3), example_indices):
    mesh = ax.pcolormesh(time_s * 1e6, freq / 1e6, to_db(magnitude, common_ref),
                         shading="auto", cmap="viridis", vmin=-80, vmax=0)
    ax.set_ylabel("frequency (MHz)")
    ax.set_title(f"class {c} example {reference_meta[i]['sample_id']} "
                 f"(STFT Hann 64, hop 8, NFFT 1024)", fontsize=9)
    ax.set_ylim(0, 13.26)
axes[2].set_xlabel(r"time ($\mu$s)")
axes[2].set_xlim(0, 34)
fig.colorbar(mesh, ax=axes, label="magnitude (dB, common reference)", shrink=0.9)
save_fig(fig, "q3_stft_examples.png")

# ---------------------------------------------------------------------------
# Predictions for both stages, written into the supplied templates
# ---------------------------------------------------------------------------
def write_predictions(template_name, metadata, features):
    """Fill predicted_class in a supplied template, keeping sample_ids in order."""
    template_path = ROOT / template_name
    with template_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header, template_ids = rows[0], [row[0] for row in rows[1:]]
    if template_ids != [row["sample_id"] for row in metadata]:
        raise ValueError(f"{template_name}: template IDs do not match the dataset")
    predictions = [apply_rules(f) for f in features]
    with template_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for sample_id, predicted in zip(template_ids, predictions):
            writer.writerow([sample_id, predicted])
    counts = {c: predictions.count(c) for c in (1, 2, 3)}
    print(f"[saved] {template_name}: {len(predictions)} predictions, counts {counts}")


write_predictions("predictions_stage1.csv", stage1_meta, s1_features)
write_predictions("predictions.csv", stage2_meta, s2_features)
