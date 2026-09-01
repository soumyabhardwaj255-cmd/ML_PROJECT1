"""
Patient-level (z-score) normalization: rescale each patient's features
using ONLY that patient's own mean/std, computed across all of that
patient's windows regardless of label.

This does not leak label information (it's unsupervised -- no seizure
labels are used), so it's safe to apply to a held-out test patient using
that same patient's own statistics. The goal is to remove inter-patient
baseline differences (e.g. one child's resting entropy running generally
higher than another's) before the model sees the data, so the model
learns the SHAPE of a seizure relative to that patient's own baseline
rather than absolute values that vary patient to patient.
"""


def patient_normalize(feature_table, feature_cols):
    """
    Parameters
    ----------
    feature_table : pd.DataFrame
        Must contain a "patient" column plus the columns in feature_cols.
    feature_cols : list[str]
        Columns to normalize.

    Returns
    -------
    pd.DataFrame
        Copy of feature_table with feature_cols z-scored within each
        patient group. All other columns (patient, file, label, etc.)
        are left untouched.
    """
    normalized = feature_table.copy()

    group_mean = normalized.groupby("patient")[feature_cols].transform("mean")
    group_std = normalized.groupby("patient")[feature_cols].transform("std")
    group_std = group_std.replace(0, 1e-8)  # avoid divide-by-zero on constant columns

    normalized[feature_cols] = (normalized[feature_cols] - group_mean) / group_std

    return normalized