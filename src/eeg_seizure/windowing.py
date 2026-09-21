"""
Split a (preprocessed) recording into fixed-length windows and label each
one seizure / non-seizure based on overlap with the recording's labeled
seizure intervals.

This module only generates window BOUNDARIES and labels, not the signal
itself, keeps memory light. Day 3's feature extraction will reload each
file once and slice out real samples per window as needed.
"""
def label_window(start_sec, end_sec, seizure_intervals, overlap_threshold=0.5):
    """
    A window is labeled 1 (seizure) if at least `overlap_threshold` fraction
    of its duration falls inside a labeled seizure interval, else 0.

    This avoids labeling a window "seizure" just because it clips one
    second of a 40-second seizure at its very edge.
    """
    window_len = end_sec - start_sec
    if window_len <= 0 or not 0 < overlap_threshold <= 1:
        raise ValueError("Invalid window duration or overlap threshold")
    overlap = sum(max(0, min(end_sec, s_end) - max(start_sec, s_start))
                  for s_start, s_end in seizure_intervals)
    return int(overlap / window_len >= overlap_threshold)


def build_window_index_for_file(
    duration_sec, seizure_intervals, patient, fname, window_sec=4.0, overlap_ratio=0.0
):
    """
    Generate window boundaries and labels for one recording.

    Parameters
    ----------
    duration_sec : float
        Total recording length in seconds.
    seizure_intervals : list[tuple[int, int]]
        (start_sec, end_sec) pairs for this file, from parse_summary_file.
    overlap_ratio : float
        Fraction of overlap between consecutive windows. 0.0 = no overlap
        (default, simplest to reason about for a first pass).

    Returns
    -------
    list[dict]
        One row per window: patient, file, start_sec, end_sec, label.
    """
    if window_sec <= 0 or not 0 <= overlap_ratio < 1 or duration_sec < 0:
        raise ValueError("Invalid window configuration")
    step_sec = window_sec * (1 - overlap_ratio)
    rows = []
    start_sec = 0.0
    while start_sec + window_sec <= duration_sec:
        end_sec = start_sec + window_sec
        label = label_window(start_sec, end_sec, seizure_intervals)
        rows.append(
            {
                "patient": patient,
                "file": fname,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "label": label,
            }
        )
        start_sec += step_sec
    return rows
