"""
Shared plotting style for all result figures, so every chart in the README
looks consistent. Import PALETTE and call apply_style() at the top of each
plotting script.
"""

import matplotlib.pyplot as plt

# A cohesive "sunset" palette used across every figure for categorical
# comparisons (models, feature families, etc.)
PALETTE = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]

# Reserved, fixed meanings used across multiple figures so color always
# means the same thing wherever it appears:
COLOR_SEIZURE = "#E76F51"       # coral — seizure / positive class
COLOR_NONSEIZURE = "#264653"    # deep teal — non-seizure / negative class
COLOR_UNTUNED = "#A8A8A8"       # neutral gray — "before" / untuned
COLOR_TUNED = "#2A9D8F"         # teal-green — "after" / tuned
COLOR_INDEPENDENT = "#E76F51"   # coral — patient-independent (the honest number)
COLOR_DEPENDENT = "#A8A8A8"     # gray — patient-dependent (the inflated number)
COLOR_HIGHLIGHT = "#E9C46A"     # gold — winner / best result


def apply_style():
    """Call once at the top of a plotting script, before creating figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_fig(fig, path):
    fig.savefig(path)
    print(f"Saved {path}")