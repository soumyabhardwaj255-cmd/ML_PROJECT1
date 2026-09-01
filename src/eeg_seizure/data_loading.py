"""
Load CHB-MIT EDF recordings and parse per-file seizure annotations from
each patient's summary.txt.

We deliberately parse the human-readable chbXX-summary.txt files rather
than the binary .edf.seizures files — the summary format is consistent,
easy to verify by eye, and gives us seizure start/end directly in seconds.
"""

import re
from pathlib import Path

import mne


def parse_summary_file(summary_path):
    """
    Parse a chbXX-summary.txt file.

    Returns
    -------
    dict[str, list[tuple[int, int]]]
        Maps edf filename (e.g. "chb01_03.edf") -> list of (start_sec, end_sec)
        seizure intervals. Files with no seizures map to an empty list.
    """
    text = Path(summary_path).read_text()

    # Split into one block per "File Name: ..." entry
    blocks = re.split(r"(?=File Name:)", text)

    seizure_info = {}
    for block in blocks:
        fname_match = re.search(r"File Name:\s*(\S+)", block)
        if not fname_match:
            continue
        fname = fname_match.group(1)

        # Handles both "Seizure Start Time:" (chb01-style) and
        # "Seizure 1 Start Time:" (multi-seizure files in other patients)
        starts = [int(x) for x in re.findall(r"Seizure(?:\s*\d*)?\s*Start Time:\s*(\d+)", block)]
        ends = [int(x) for x in re.findall(r"Seizure(?:\s*\d*)?\s*End Time:\s*(\d+)", block)]

        seizure_info[fname] = list(zip(starts, ends))

    return seizure_info


def load_edf(edf_path):
    """Load a single .edf recording into an MNE Raw object (full precision, preloaded)."""
    return mne.io.read_raw_edf(edf_path, preload=True, verbose=False)


def summarize_recording(raw, seizures, fname=""):
    """Print a quick sanity-check summary of a loaded recording."""
    print(f"--- {fname} ---")
    print(f"Channels ({len(raw.ch_names)}): {raw.ch_names}")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.1f} sec ({raw.times[-1] / 60:.1f} min)")
    print(f"Seizures in this file: {seizures if seizures else 'none'}")