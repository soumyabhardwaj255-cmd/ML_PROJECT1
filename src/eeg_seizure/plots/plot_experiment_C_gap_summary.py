"""
Figure + table: the headline generalization-gap chart. Patient-independent
vs patient-dependent F1, untuned vs tuned, with error bars (std across
folds/seeds). This is the single most important chart in the project.

Ran from the repo root, after experiment_C_generalization.py AND
experiment_c_tuned.py:
     python src/eeg_seizure/plots/plot_experiment_C_gap_summary.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eeg_seizure.style import COLOR_DEPENDENT, COLOR_INDEPENDENT, apply_style, save_fig

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "tables"
FIGURES_DIR = Path(__file__).resolve().parents[3] / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

apply_style()

untuned = pd.read_csv(RESULTS_DIR / "experiment_c_generalization_gap.csv")
tuned = pd.read_csv(RESULTS_DIR / "experiment_c_tuned_summary.csv")

# Pull out the 4 numbers we need, regardless of exact split-label wording
def get_row(df, prefix):
    return df[df["split"].str.startswith(prefix, na=False)].iloc[0]

ind_untuned = get_row(untuned, "patient_independent")
dep_untuned = get_row(untuned, "patient_dependent")
ind_tuned = get_row(tuned, "patient_independent")
dep_tuned = get_row(tuned, "patient_dependent")

labels = ["Untuned", "Tuned"]
independent_f1 = [ind_untuned["f1"], ind_tuned["f1"]]
independent_std = [ind_untuned["f1_std"], ind_tuned["f1_std"]]
dependent_f1 = [dep_untuned["f1"], dep_tuned["f1"]]
dependent_std = [dep_untuned["f1_std"], dep_tuned["f1_std"]]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(
    x - width / 2, independent_f1, width, yerr=independent_std, capsize=5,
    label="Patient-independent (honest)", color=COLOR_INDEPENDENT,
)
ax.bar(
    x + width / 2, dependent_f1, width, yerr=dependent_std, capsize=5,
    label="Patient-dependent (leaked)", color=COLOR_DEPENDENT,
)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("F1 score (mean \u00b1 std)")
ax.set_ylim(0, 1.05)
ax.set_title("The generalization gap persists even after tuning")
ax.legend(loc="upper left", frameon=False)

for i, (ind, dep) in enumerate(zip(independent_f1, dependent_f1)):
    gap = dep - ind
    ax.annotate(
        f"gap = +{gap:.3f}",
        xy=(i, 0.55),
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

save_fig(fig, FIGURES_DIR / "experiment_c_generalization_gap_summary.png")

# --- Markdown table ---
summary = pd.DataFrame(
    {
        "Condition": ["Patient-independent", "Patient-dependent"],
        "Untuned F1 (mean \u00b1 std)": [
            f"{ind_untuned['f1']:.3f} \u00b1 {ind_untuned['f1_std']:.3f}",
            f"{dep_untuned['f1']:.3f} \u00b1 {dep_untuned['f1_std']:.3f}",
        ],
        "Tuned F1 (mean \u00b1 std)": [
            f"{ind_tuned['f1']:.3f} \u00b1 {ind_tuned['f1_std']:.3f}",
            f"{dep_tuned['f1']:.3f} \u00b1 {dep_tuned['f1_std']:.3f}",
        ],
    }
)
md_path = RESULTS_DIR / "experiment_c_summary_table.md"
md_path.write_text(summary.to_markdown(index=False))
print(f"Saved {md_path}")
print(summary.to_string(index=False))