"""
Load one EEG recording, confirm we can read it correctly, and
visually locate a labeled seizure in the raw signal.

Ran from the repo root:
    python notebooks/01_explore.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Allow importing our package from src/ without installing it
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eeg_seizure.data_loading import parse_summary_file, load_edf, summarize_recording

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Pick a known seizure file to start with
patient = "chb01"
edf_filename = "chb01_03.edf"

summary_path = DATA_DIR / patient / f"{patient}-summary.txt"
edf_path = DATA_DIR / patient / edf_filename

seizure_map = parse_summary_file(summary_path)
seizures_in_file = seizure_map[edf_filename]

raw = load_edf(edf_path)
summarize_recording(raw, seizures_in_file, fname=edf_filename)

# Plot the first few channels across the whole recording, so we can see
# the overall shape before zooming into the seizure itself.
n_channels_to_plot = 5
data, times = raw[:n_channels_to_plot, :]

fig, axes = plt.subplots(n_channels_to_plot, 1, figsize=(14, 8), sharex=True)
for i, ch_data in enumerate(data):
    axes[i].plot(times, ch_data, linewidth=0.5)
    axes[i].set_ylabel(raw.ch_names[i], rotation=0, labelpad=35, fontsize=8, va="center")
    for start, end in seizures_in_file:
        axes[i].axvspan(start, end, color="red", alpha=0.15)
axes[-1].set_xlabel("Time (s)")
fig.suptitle(f"{edf_filename} — full recording (seizure shaded red)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "day1_full_recording.png", dpi=150)
plt.show()

# Now zoom into just the seizure window, with some context before/after,
# so the change in amplitude/rhythm is actually visible.
start, end = seizures_in_file[0]
pad_seconds = 20
tmin = max(0, start - pad_seconds)
tmax = min(raw.times[-1], end + pad_seconds)

sfreq = raw.info["sfreq"]
data, times = raw[:n_channels_to_plot, int(tmin * sfreq):int(tmax * sfreq)]

fig, axes = plt.subplots(n_channels_to_plot, 1, figsize=(14, 8), sharex=True)
for i, ch_data in enumerate(data):
    axes[i].plot(times, ch_data, linewidth=0.6)
    axes[i].set_ylabel(raw.ch_names[i], rotation=0, labelpad=35, fontsize=8, va="center")
    axes[i].axvspan(start, end, color="red", alpha=0.15)
axes[-1].set_xlabel("Time (s)")
fig.suptitle(f"{edf_filename} — zoomed on seizure ({start}-{end}s, shaded)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "day1_seizure_zoom.png", dpi=150)
plt.show()

# Sanity check on a NON-seizure file from the same patient, to see what
# normal activity looks like for comparison.
nonseizure_filename = "chb01_01.edf"
nonseizure_seizures = seizure_map.get(nonseizure_filename, [])
raw_normal = load_edf(DATA_DIR / patient / nonseizure_filename)
summarize_recording(raw_normal, nonseizure_seizures, fname=nonseizure_filename)