"""Persisted model bundles and offline feature/EDF inference."""
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from . import config as C
from .modeling import student_network, student_scores


def save_bundle(path, bundle):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    joblib.dump(bundle, tmp)
    os.replace(tmp, path)


def load_bundle(path):
    bundle = joblib.load(path)
    if bundle.get("format_version") != 1:
        raise ValueError("Unsupported model bundle")
    if set(bundle["train_patients"]) & set(bundle["test_patients"]):
        raise ValueError("Bundle has overlapping training/test patients")
    return bundle


def predict_features(bundle, frame):
    columns = bundle["feature_columns"]
    if frame.columns.duplicated().any() or not set(columns).issubset(frame.columns):
        raise ValueError("Missing or duplicated feature columns")
    extra = set(frame.columns) - set(columns) - set(C.META)
    if extra:
        raise ValueError(f"Unexpected feature columns: {sorted(extra)}")
    X = frame.loc[:, columns]
    if bundle["kind"] == "student":
        architecture = bundle["architecture"]
        if len(architecture) != 4 or architecture[0] != len(bundle["selected_features"]) or architecture[-1] != 1:
            raise ValueError("Invalid saved student architecture")
        model = student_network(architecture[0], architecture[1:3])
        model.load_state_dict(bundle["state_dict"])
        z = bundle["transform"].transform(X[bundle["selected_features"]])
        scores = student_scores(model, z)
    else:
        scores = bundle["estimator"].predict_proba(X)[:, 1]
    return scores, (scores >= bundle["threshold"]).astype(int)


def extract_edf(bundle, path, patient="inference"):
    import mne
    from .channels import dedupe_channel_names
    from .preprocessing import preprocess_raw
    from .dataset import feature_rows, window_rows, extraction_spec
    spec = bundle["dataset_spec"]
    if spec != extraction_spec(spec["channels"]):
        raise ValueError("Installed feature/preprocessing implementation differs from bundle")
    with mne.io.read_raw_edf(path, preload=True, verbose="ERROR") as raw:
        if raw.info["sfreq"] != spec["sample_rate"]:
            raise ValueError("EDF sampling rate differs from bundle")
        raw = dedupe_channel_names(raw)
        if not set(spec["channels"]).issubset(raw.ch_names):
            raise ValueError("EDF is missing required channels")
        raw = preprocess_raw(raw, spec["channels"], **spec["preprocessing"])
        windows = window_rows(raw.n_times, raw.info["sfreq"], [], patient, Path(path).name)
        features = feature_rows(raw, windows, spec["channels"])
    # Labels are unknown during new-recording inference.
    return features.drop(columns="label")


def contiguous_events(frame, minimum_windows=2):
    events = []
    for (patient, file), g in frame.groupby(["patient", "file"]):
        run = None
        for row in g.sort_values("start_sec").itertuples():
            if row.prediction == 1:
                if run is not None and np.isclose(row.start_sec, run["end_sec"]):
                    run["end_sec"] = row.end_sec
                    run["windows"] += 1
                else:
                    if run is not None:
                        events.append(run)
                    run = dict(patient=patient, file=file, start_sec=row.start_sec,
                               end_sec=row.end_sec, windows=1)
            elif run is not None:
                events.append(run)
                run = None
        if run is not None:
            events.append(run)
    result = pd.DataFrame(events, columns=["patient", "file", "start_sec", "end_sec", "windows"])
    result["sustained"] = result.windows >= minimum_windows
    return result
