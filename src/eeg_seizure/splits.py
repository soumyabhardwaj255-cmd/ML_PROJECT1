"""
Train/test split strategies for the feature table.

- patient_independent_split: train and test come from DIFFERENT patients —
  the correct way to measure whether a model generalizes to new patients,
  and the default split used throughout this project.
- patient_dependent_split: a plain random split ignoring patient identity —
  used later in Experiment C specifically to demonstrate the leakage this
  causes (a model can partly memorize per-patient signal, inflating scores).
"""

from sklearn.model_selection import train_test_split

METADATA_COLS = ["patient", "file", "start_sec", "end_sec", "label"]


def _feature_cols(feature_table):
    return [c for c in feature_table.columns if c not in METADATA_COLS]


def patient_independent_split(feature_table, test_patients):
    """
    Parameters
    ----------
    feature_table : pd.DataFrame
        Must contain a "patient" column.
    test_patients : str or list[str]
        Patient(s) held out entirely for testing. Every other patient's
        windows go to training.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if isinstance(test_patients, str):
        test_patients = [test_patients]

    feature_cols = _feature_cols(feature_table)
    test_mask = feature_table["patient"].isin(test_patients)

    X_train = feature_table.loc[~test_mask, feature_cols]
    X_test = feature_table.loc[test_mask, feature_cols]
    y_train = feature_table.loc[~test_mask, "label"]
    y_test = feature_table.loc[test_mask, "label"]

    return X_train, X_test, y_train, y_test


def patient_dependent_split(feature_table, test_size=0.2, random_state=42):
    """
    Plain random row-level split, ignoring patient identity. Windows from
    the SAME patient/recording can end up in both train and test — this is
    the split Experiment C uses to demonstrate why patient-independent
    evaluation is necessary in the first place.
    """
    feature_cols = _feature_cols(feature_table)
    X = feature_table[feature_cols]
    y = feature_table["label"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)