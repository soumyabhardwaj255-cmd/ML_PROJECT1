"""
Figure + table: Experiment B feature ablation (time-domain, frequency-
domain, entropy+Hjorth, all combined).

Ran from the repo root, after experiments/experiment_b_features.py:
    python -m eeg_seizure.plots.plot_experiment_B
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from eeg_seizure.style import COLOR_HIGHLIGHT, PALETTE, apply_style, save_fig

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "tables"
FIGURES_DIR = Path(__file__).resolve().parents[3] / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

apply_style()

df = pd.read_csv(RESULTS_DIR / "experiment_b_feature_ablation.csv")
df = df.sort_values("f1", ascending=False).reset_index(drop=True)

family_labels = {
    "entropy_hjorth": "Entropy +\nHjorth",
    "all_combined": "All\ncombined",
    "time_domain": "Time\ndomain",
    "frequency_domain": "Frequency\ndomain",
}
df["label"] = df["feature_family"].map(family_labels).fillna(df["feature_family"])

colors = [COLOR_HIGHLIGHT if i == 0 else PALETTE[min(i, len(PALETTE) - 1)] for i in range(len(df))]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
bars = ax.bar(df["label"], df["f1"], color=colors, width=0.6)

for bar, f1, n in zip(bars, df["f1"], df["n_features"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f"F1={f1:.3f}\n({n} feat.)",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.set_ylabel("F1 score")
ax.set_ylim(0, max(df["f1"]) * 1.3)
ax.set_title("Experiment B: feature ablation (model=XGBoost, test=chb08)")

save_fig(fig, FIGURES_DIR / "experiment_b_feature_ablation.png")

# --- Markdown table ---
table_md = df[["feature_family", "n_features", "precision", "recall", "f1"]].copy()
table_md.columns = ["Feature family", "N features", "Precision", "Recall", "F1"]
for col in ["Precision", "Recall", "F1"]:
    table_md[col] = table_md[col].round(3)

md_path = RESULTS_DIR / "experiment_b_table.md"
md_path.write_text(table_md.to_markdown(index=False))
print(f"Saved {md_path}")
print(table_md.to_string(index=False))