"""
Extract time-domain, frequency-domain, entropy, and Hjorth features
for every window in window_index.csv, producing one master feature table.

Produces:
    data/processed/feature_table.csv - one row per window, one column per
    (family, channel) feature pair, plus patient/file/label metadata.

Ran from the repo root:
    python notebooks/03_extract_features.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eeg_seizure.channels import dedupe_channel_names
from eeg_seizure.data_loading import load_edf
from eeg_seizure.features.feature_extraction import extract_window_features
from eeg_seizure.preprocessing import preprocess_raw

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

WINDOW_SEC = 4.0
TEST_MODE = False  # set False once you've validated the pipeline works

# Load Day 2 outputs
window_index = pd.read_csv(PROCESSED_DIR / "window_index.csv")
common_channels = json.loads((PROCESSED_DIR / "common_channels.json").read_text())

if TEST_MODE:
    # Sample some seizure AND non-seizure windows per file, so the test run
    # can actually confirm labels differ meaningfully in feature space —
    # group.head(20) alone would only grab early, pre-seizure windows.
    seizure_rows = window_index[window_index["label"] == 1].groupby(
        ["patient", "file"], group_keys=False
    ).head(5)
    nonseizure_rows = window_index[window_index["label"] == 0].groupby(
        ["patient", "file"], group_keys=False
    ).head(15)
    window_index = pd.concat([seizure_rows, nonseizure_rows]).sort_values(
        ["patient", "file", "start_sec"]
    )
    print(f"TEST_MODE on — using {len(window_index)} windows "
          f"({len(seizure_rows)} seizure, {len(nonseizure_rows)} non-seizure).\n")

# Process one (patient, file) group at a time: load + filter the recording
# ONCE, then slice out every window's samples from that single loaded copy.
#
# Checkpointing: after each file's windows are done, append them straight
# to disk. If the run gets interrupted (sleep, crash, power blip), rerun
# this script and it will skip files already checkpointed rather than
# starting over — important for a ~3 hour unattended run.
checkpoint_path = PROCESSED_DIR / (
    "feature_table_test_checkpoint.csv" if TEST_MODE else "feature_table_checkpoint.csv"
)

already_done = set()
if checkpoint_path.exists():
    done_df = pd.read_csv(checkpoint_path, usecols=["patient", "file"])
    already_done = set(map(tuple, done_df.drop_duplicates().values))
    print(f"Found existing checkpoint — {len(already_done)} files already done, skipping those.\n")

grouped = window_index.groupby(["patient", "file"])

for (patient, fname), group in tqdm(grouped, desc="Files"):
    if (patient, fname) in already_done:
        continue

    edf_path = DATA_DIR / patient / fname

    raw = load_edf(edf_path)
    raw = dedupe_channel_names(raw)
    raw = preprocess_raw(raw, common_channels)

    sfreq = raw.info["sfreq"]
    window_samples = int(WINDOW_SEC * sfreq)
    channel_names = raw.ch_names

    file_rows = []
    for _, row in group.iterrows():
        start_idx = int(round(row["start_sec"] * sfreq))
        end_idx = start_idx + window_samples

        window_data = raw.get_data(start=start_idx, stop=end_idx)

        features = extract_window_features(window_data, sfreq, channel_names)
        features["patient"] = patient
        features["file"] = fname
        features["start_sec"] = row["start_sec"]
        features["end_sec"] = row["end_sec"]
        features["label"] = row["label"]

        file_rows.append(features)

    # Append this file's rows to the checkpoint immediately — don't wait
    # until the whole run finishes to persist anything.
    file_df = pd.DataFrame(file_rows)
    write_header = not checkpoint_path.exists()
    file_df.to_csv(checkpoint_path, mode="a", header=write_header, index=False)

feature_table = pd.read_csv(checkpoint_path)

output_name = "feature_table_test.csv" if TEST_MODE else "feature_table.csv"
feature_table.to_csv(PROCESSED_DIR / output_name, index=False)

n_metadata_cols = 5  # patient, file, start_sec, end_sec, label
print(f"\nFeature table shape: {feature_table.shape}")
print(f"Feature columns (excluding metadata): {feature_table.shape[1] - n_metadata_cols}")
print(f"Label balance:\n{feature_table['label'].value_counts()}")
print(f"\nSaved to {PROCESSED_DIR / output_name}")

# Quick sanity check (TEST_MODE only): line length should be noticeably
# higher during seizure windows, per the source material's expectation
# that seizures produce larger, faster-changing signals. If seizure rows
# exist in this run, this confirms the features actually carry signal.
if TEST_MODE and (feature_table["label"] == 1).any():
    check_col = "time__line_length__FP1-F7"
    seizure_mean = feature_table.loc[feature_table["label"] == 1, check_col].mean()
    nonseizure_mean = feature_table.loc[feature_table["label"] == 0, check_col].mean()
    print(f"\nSanity check — {check_col}:")
    print(f"  seizure windows:     mean = {seizure_mean:.6f}")
    print(f"  non-seizure windows: mean = {nonseizure_mean:.6f}")
    print("  (expect seizure mean > non-seizure mean)")