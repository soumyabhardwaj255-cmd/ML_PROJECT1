"""Read-only checks of real corrected data; write a compact validation receipt."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eeg_seizure import config as C
from eeg_seizure.artifacts import atomic_json, atomic_csv, digest, fingerprint
from eeg_seizure.dataset import load_dataset, read_table, checkpoint_valid
from eeg_seizure.modeling import FeatureTransform


def main():
    frame, manifest = load_dataset()
    old = read_table(ROOT / "data/processed/feature_table.csv")
    features = manifest["spec"]["feature_columns"]
    joined = frame.merge(old, on=C.KEYS, suffixes=("_new", "_old"), validate="one_to_one")
    assert len(joined) == len(old)
    assert np.array_equal(joined.label_new, joined.label_old)
    new_values = joined[[c + "_new" for c in features]].to_numpy()
    old_values = joined[[c + "_old" for c in features]].to_numpy()
    assert np.allclose(new_values, old_values, rtol=1e-12, atol=1e-25)
    for recording in manifest["records"]:
        patient, file = recording["patient"], recording["file"]
        source = C.RAW_DIR / patient / file
        assert digest(source) == recording["raw_sha256"]
        assert digest(source.parent / f"{patient}-summary.txt") == recording["summary_sha256"]
        windows = frame.loc[(frame.patient == patient) & (frame.file == file), C.META]
        identity = {k: recording[k] for k in ("patient", "file", "raw_sha256", "summary_sha256",
                                             "spec", "window_identity", "expected_windows")}
        path = C.DATA_DIR / "recordings" / patient / (file + ".csv")
        assert checkpoint_valid(path, path.with_suffix(".json"), identity, windows,
                                manifest["spec"]["channels"]) is not None
    # Check every outer fold's feature transform on genuine low-magnitude EEG features.
    scaling = []
    for patient in C.PATIENTS:
        train = frame.loc[frame.patient != patient, features]
        transform = FeatureTransform().fit(train)
        z = transform.transform(train)
        tiny = (train.std(ddof=0) < 1e-8) & (train.std(ddof=0) > 0)
        np.testing.assert_allclose(z[:, tiny].std(axis=0), 1, rtol=1e-5, atol=1e-5)
        before = transform.scaler_.mean_.copy()
        transform.transform(frame.loc[frame.patient == patient, features])
        np.testing.assert_array_equal(before, transform.scaler_.mean_)
        scaling.append(dict(held_out=patient, low_magnitude_nonconstant_features=int(tiny.sum()),
                            training_rows=len(train), test_rows=int((frame.patient == patient).sum())))
    changes = read_table(C.DATA_DIR / "window_count_changes.csv")
    report = dict(status="passed", dataset_sha256=manifest["feature_sha256"],
                  previous_windows=len(old), corrected_windows=len(frame),
                  previous_seizure=int(old.label.sum()), corrected_seizure=int(frame.label.sum()),
                  previous_nonseizure=int((old.label == 0).sum()),
                  corrected_nonseizure=int((frame.label == 0).sum()),
                  affected_recordings=int((changes.added_windows > 0).sum()),
                  recordings_checked=len(manifest["records"]), raw_and_summary_hashes_unchanged=True,
                  retained_legacy_rows_numerically_verified=len(joined),
                  legacy_reuse_scope="All retained values preserved; freshly extracted per-file samples checked during prepare",
                  scaling_checks=scaling)
    output = ROOT / "results/corrected/implementation-validation"
    atomic_json(output / "dataset_validation.json", report)
    atomic_csv(output / "window_count_changes.csv", changes)
    atomic_csv(output / "patient_counts.csv", frame.groupby("patient").label.agg(
        windows="size", seizure="sum").reset_index())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
