"""
Figure: class imbalance. Simple, high-impact chart showing the ~1.6%
seizure rate from Day 2 -- the direct justification for using F1 instead
of accuracy throughout the project.

Ran from the repo root:
    python -m eeg_seizure.plots.plot_class_imbalance
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from eeg_seizure.style import COLOR_NONSEIZURE, COLOR_SEIZURE, apply_style, save_fig

PROCESSED_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"
FIGURES_DIR = Path(__file__).resolve().parents[3] / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

apply_style()

window_index = pd.read_csv(PROCESSED_DIR / "window_index.csv")
n_seizure = int(window_index["label"].sum())
n_total = len(window_index)
n_nonseizure = n_total - n_seizure
pct_seizure = 100 * n_seizure / n_total

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(
    ["Non-seizure", "Seizure"],
    [n_nonseizure, n_seizure],
    color=[COLOR_NONSEIZURE, COLOR_SEIZURE],
    width=0.55,
)
for bar, count in zip(bars, [n_nonseizure, n_seizure]):
    pct = 100 * count / n_total
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.02,
        f"{count:,}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

ax.set_ylabel("Number of 4-second windows")
ax.set_title(f"Class imbalance: only {pct_seizure:.1f}% of windows contain a seizure")
ax.set_ylim(0, n_nonseizure * 1.18)

save_fig(fig, FIGURES_DIR / "class_imbalance.png")