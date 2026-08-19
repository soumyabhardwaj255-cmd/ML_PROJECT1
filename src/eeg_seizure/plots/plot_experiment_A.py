"""
Figure + table: Experiment A model comparison (Logistic Regression vs
Random Forest vs XGBoost).

Ran from the repo root:
    python -m eeg_seizure.plots.plot_experiment_A
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eeg_seizure.style import PALETTE, apply_style, save_fig

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "tables"
FIGURES_DIR = Path(__file__).resolve().parents[3] / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

apply_style()

df = pd.read_csv(RESULTS_DIR / "experiment_a_model_comparison.csv")
df = df.sort_values("f1", ascending=False).reset_index(drop=True)

model_labels = {
    "logistic_regression": "Logistic\nRegression",
    "random_forest": "Random\nForest",
    "xgboost": "XGBoost",
}
df["label"] = df["model"].map(model_labels).fillna(df["model"])

metrics = ["precision", "recall", "f1"]
x = np.arange(len(df))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5.5))
for i, metric in enumerate(metrics):
    ax.bar(x + (i - 1) * width, df[metric], width, label=metric.capitalize(), color=PALETTE[i])

ax.set_xticks(x)
ax.set_xticklabels(df["label"])
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_title("Experiment A: model comparison (patient-independent, test=chb08)")
ax.legend(loc="upper right", frameon=False)

best_idx = df["f1"].idxmax()
ax.annotate(
    "best F1",
    xy=(best_idx + width, df.loc[best_idx, "f1"] + 0.02),
    ha="center",
    fontsize=9,
    fontweight="bold",
)

save_fig(fig, FIGURES_DIR / "experiment_a_model_comparison.png")

# --- Markdown table ---
table_md = df[["model", "precision", "recall", "f1"]].copy()
table_md.columns = ["Model", "Precision", "Recall", "F1"]
for col in ["Precision", "Recall", "F1"]:
    table_md[col] = table_md[col].round(3)

md_path = RESULTS_DIR / "experiment_a_table.md"
md_path.write_text(table_md.to_markdown(index=False))
print(f"Saved {md_path}")
print(table_md.to_string(index=False))