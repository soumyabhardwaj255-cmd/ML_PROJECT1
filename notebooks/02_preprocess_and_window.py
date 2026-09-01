"""
Find the channel set common to all 5 patients, filter every
downloaded recording, and window each one into labeled 4-second segments.

Produces:
    data/processed/common_channels.json  — cached common channel list
    data/processed/window_index.csv      — one row per window (metadata only)

Ran from the repo root:
    python notebooks/02_preprocess_and_window.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eeg_seizure.channels import dedupe_channel_names, find_common_channels, scan_channel_sets
from eeg_seizure.data_loading import load_edf, parse_summary_file
from eeg_seizure.preprocessing import preprocess_raw
from eeg_seizure.windowing import build_window_index_for_file

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS = ["chb01", "chb02", "chb03", "chb05", "chb08"]
WINDOW_SEC = 4.0

# Step 1: find channels common to every downloaded file across all 5
# patients. This loads every file once just to check channel names —
# slow-ish the first time, so we cache the result.
common_channels_path = PROCESSED_DIR / "common_channels.json"

if common_channels_path.exists():
    common_channels = json.loads(common_channels_path.read_text())
    print(f"Loaded cached common channel list ({len(common_channels)} channels).")
else:
    print("Scanning channel sets across all downloaded files...")
    channel_sets = scan_channel_sets(DATA_DIR, PATIENTS)
    common_channels = find_common_channels(channel_sets)
    common_channels_path.write_text(json.dumps(common_channels, indent=2))
    print(f"\nFound {len(common_channels)} channels common to every file:")
    print(common_channels)

# Step 2: for every file, filter the full recording and build labeled
# window boundaries (metadata only, no signal arrays stored here).
all_rows = []

for patient in PATIENTS:
    patient_dir = DATA_DIR / patient
    summary_path = patient_dir / f"{patient}-summary.txt"
    seizure_map = parse_summary_file(summary_path)

    for edf_path in sorted(patient_dir.glob("*.edf")):
        fname = edf_path.name
        seizure_intervals = seizure_map.get(fname, [])

        raw = load_edf(edf_path)
        raw = dedupe_channel_names(raw)
        raw = preprocess_raw(raw, common_channels)

        duration_sec = raw.times[-1]
        rows = build_window_index_for_file(
            duration_sec, seizure_intervals, patient, fname, window_sec=WINDOW_SEC
        )
        all_rows.extend(rows)

        n_seizure = sum(r["label"] for r in rows)
        print(f"{patient}/{fname}: {len(rows)} windows, {n_seizure} labeled seizure")

# Step 3: save the window index and sanity-check class balance.
window_index = pd.DataFrame(all_rows)
window_index.to_csv(PROCESSED_DIR / "window_index.csv", index=False)

n_total = len(window_index)
n_seizure = int(window_index["label"].sum())

print(f"\nTotal windows: {n_total}")
print(f"Seizure windows:     {n_seizure} ({100 * n_seizure / n_total:.2f}%)")
print(f"Non-seizure windows: {n_total - n_seizure} ({100 * (n_total - n_seizure) / n_total:.2f}%)")
print(f"\nSaved window index to {PROCESSED_DIR / 'window_index.csv'}")