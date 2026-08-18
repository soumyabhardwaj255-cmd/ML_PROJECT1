"""
Channel handling for CHB-MIT recordings.

Some files contain a duplicate channel name (e.g. two electrode pairs both
labeled "T8-P8" because the montage changed mid-recording), which MNE
auto-renames on load. We normalize those duplicates within a single file,
then find the channel set common to every file we downloaded so every
window in the final feature table has identical columns.
"""

import re
from collections import defaultdict
from pathlib import Path

from .data_loading import load_edf


def dedupe_channel_names(raw):
    """
    Collapse MNE's auto-renamed duplicate channels (e.g. "T8-P8-0",
    "T8-P8-1") back into a single channel per unique name. Keeps the first
    occurrence, drops the rest.

    Only touches names that are genuinely duplicated — a channel like
    "FT9-FT10" is left alone even though it ends in digits, because
    nothing else in the file shares the base "FT9-FT".
    """
    pattern = re.compile(r"^(.*)-(\d+)$")
    groups = defaultdict(list)
    for name in raw.ch_names:
        match = pattern.match(name)
        if match:
            groups[match.group(1)].append(name)

    to_drop = []
    rename_map = {}
    for base, duplicate_names in groups.items():
        if len(duplicate_names) >= 2:
            duplicate_names = sorted(duplicate_names)
            keep, *drop = duplicate_names
            rename_map[keep] = base
            to_drop.extend(drop)

    if to_drop:
        raw.drop_channels(to_drop)
    if rename_map:
        raw.rename_channels(rename_map)
    return raw


def scan_channel_sets(raw_dir: Path, patients: list) -> dict:
    """
    Load every downloaded .edf file for the given patients and record its
    (deduped) channel set.

    Returns
    -------
    dict[(str, str), set[str]]
        Maps (patient, filename) -> set of channel names in that file.
    """
    channel_sets = {}
    for patient in patients:
        patient_dir = raw_dir / patient
        edf_files = sorted(patient_dir.glob("*.edf"))
        for edf_path in edf_files:
            raw = load_edf(edf_path)
            raw = dedupe_channel_names(raw)
            channel_sets[(patient, edf_path.name)] = set(raw.ch_names)
            print(f"  scanned {patient}/{edf_path.name}: {len(raw.ch_names)} channels")
    return channel_sets


def find_common_channels(channel_sets: dict) -> list:
    """Intersect channel sets across every file scanned. Returns a sorted list."""
    all_sets = list(channel_sets.values())
    common = set.intersection(*all_sets)
    return sorted(common)