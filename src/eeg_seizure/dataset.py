"""Validated recording checkpoints and complete-window feature extraction."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from . import config as C
from .artifacts import atomic_csv, atomic_json, digest, environment, fingerprint
from .channels import dedupe_channel_names
from .data_loading import parse_summary_file
from .preprocessing import preprocess_raw
from .features.feature_extraction import extract_window_features
from .windowing import label_window


def read_table(path):
    # Preserve serialized binary64 values across checkpoint resumes.
    return pd.read_csv(path, float_precision="round_trip")


def window_rows(n_samples, sfreq, intervals, patient, filename):
    """Use sample count, not the last sample's timestamp."""
    size = int(round(C.WINDOW_SEC * sfreq))
    if size <= 0 or not np.isclose(size / sfreq, C.WINDOW_SEC):
        raise ValueError("Window duration must map to integer samples")
    rows = []
    for start in range(0, n_samples - size + 1, size):
        a, b = start / sfreq, (start + size) / sfreq
        rows.append(dict(patient=patient, file=filename, start_sec=a, end_sec=b,
                         label=label_window(a, b, intervals, C.OVERLAP_THRESHOLD)))
    return pd.DataFrame(rows, columns=C.META)


def validated_annotations(summary, filename, duration):
    mapping = parse_summary_file(summary)
    if filename not in mapping:
        raise ValueError(f"Missing annotations for {filename}")
    intervals = sorted(mapping[filename])
    previous_end = -1
    for start, end in intervals:
        if not 0 <= start < end <= duration or start < previous_end:
            raise ValueError(f"Invalid/overlapping seizure intervals: {filename}")
        previous_end = end
    return intervals


def validate_table(frame, windows, channels):
    expected_cols = C.feature_names(channels)
    if (len(expected_cols) != 330 or frame.columns.duplicated().any()
            or set(frame.columns) != set(C.META + expected_cols)):
        raise ValueError("Expected metadata plus 330 configured features")
    if frame.duplicated(C.KEYS).any():
        raise ValueError("Duplicate window identities")
    actual = frame[C.META].sort_values(C.KEYS).reset_index(drop=True)
    expected = windows[C.META].sort_values(C.KEYS).reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError("Window identity, counts or annotations do not match")
    if not np.isfinite(frame[expected_cols].to_numpy(dtype=float)).all():
        raise ValueError("Features contain NaN or infinity")


def load_channels():
    paths, channel_sets = [], []
    for patient in C.PATIENTS:
        files = sorted((C.RAW_DIR / patient).glob("*.edf"))
        if not files:
            raise FileNotFoundError(f"No EDF files for {patient}")
        for path in files:
            with mne.io.read_raw_edf(path, preload=False, verbose="ERROR") as raw:
                dedupe_channel_names(raw)
                if raw.info["sfreq"] != C.SAMPLE_RATE:
                    raise ValueError(f"Unsupported sample rate in {path.name}")
                channel_sets.append(set(raw.ch_names))
            paths.append((patient, path))
    channels = sorted(set.intersection(*channel_sets))
    if len(channels) != 22:
        raise ValueError(f"Expected 22 common channels, found {len(channels)}")
    return paths, channels


def extraction_spec(channels):
    source = Path(__file__).parent
    paths = sorted((source / "features").glob("*.py")) + [source / "preprocessing.py", source / "channels.py", source / "windowing.py", source / "data_loading.py", source / "dataset.py"]
    return dict(data_version=C.DATA_VERSION, feature_version=C.FEATURE_VERSION,
                preprocessing=C.PREPROCESS, window_sec=C.WINDOW_SEC,
                overlap_threshold=C.OVERLAP_THRESHOLD, channels=channels,
                feature_columns=C.feature_names(channels), sample_rate=C.SAMPLE_RATE,
                source_hashes={str(p.relative_to(source)): digest(p) for p in paths},
                packages={name: version for name, version in environment()["packages"].items()
                          if name in ("numpy", "pandas", "scipy", "mne", "antropy")})


def feature_rows(raw, windows, channels):
    result = []
    sfreq = raw.info["sfreq"]
    for row in windows.to_dict("records"):
        start, stop = int(round(row["start_sec"] * sfreq)), int(round(row["end_sec"] * sfreq))
        signal = raw.get_data(start=start, stop=stop)
        if signal.shape != (22, int(C.WINDOW_SEC * sfreq)):
            raise ValueError("Incomplete EEG window")
        result.append({**extract_window_features(signal, sfreq, raw.ch_names), **row})
    return pd.DataFrame(result, columns=C.feature_names(channels) + C.META)


def checkpoint_valid(csv_path, json_path, identity, windows, channels):
    if not csv_path.exists() or not json_path.exists():
        return None
    try:
        manifest = json.loads(json_path.read_text(encoding="utf-8"))
        if manifest["identity"] != identity or manifest["sha256"] != digest(csv_path):
            return None
        frame = read_table(csv_path)
        validate_table(frame, windows, channels)
        return frame
    except (ValueError, KeyError, OSError):
        return None


def prepare(reuse_legacy=False):
    paths, channels = load_channels()
    spec = extraction_spec(channels)
    legacy_path = C.ROOT / "data" / "processed" / "feature_table.csv"
    legacy = read_table(legacy_path) if reuse_legacy and legacy_path.exists() else None
    old_index = read_table(C.ROOT / "data" / "processed" / "window_index.csv") if legacy is not None else None
    if legacy is not None:
        validate_table(legacy, old_index, channels)
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    # A partial build must never retain an old complete-dataset marker.
    atomic_json(C.DATA_DIR / "manifest.json", {"status": "building", "spec": spec})
    tables, indices, changes, records = [], [], [], []
    for patient, path in paths:
        summary = path.parent / f"{patient}-summary.txt"
        with mne.io.read_raw_edf(path, preload=False, verbose="ERROR") as header:
            intervals = validated_annotations(summary, path.name, header.n_times / header.info["sfreq"])
            windows = window_rows(header.n_times, header.info["sfreq"], intervals, patient, path.name)
        identity = dict(patient=patient, file=path.name, raw_sha256=digest(path),
                        summary_sha256=digest(summary), spec=fingerprint(spec),
                        window_identity=fingerprint(windows.to_dict("records")), expected_windows=len(windows))
        csv_path = C.DATA_DIR / "recordings" / patient / (path.name + ".csv")
        json_path = csv_path.with_suffix(".json")
        frame = checkpoint_valid(csv_path, json_path, identity, windows, channels)
        old = legacy[(legacy.patient == patient) & (legacy.file == path.name)].copy() if legacy is not None else None
        mode = "validated_checkpoint"
        if frame is None:
            with mne.io.read_raw_edf(path, preload=True, verbose="ERROR") as raw:
                raw = preprocess_raw(dedupe_channel_names(raw), channels, **C.PREPROCESS)
                mode = "full_extraction"
                if old is not None and len(old):
                    retained = windows.merge(old[C.KEYS], on=C.KEYS, how="inner")
                    validate_table(old, retained, channels)
                    # Explicit legacy migration: validate a background and a seizure
                    # window (where available) from each recording before reuse.
                    sample = old.groupby("label", group_keys=False).head(1)[C.META]
                    check = feature_rows(raw, sample, channels)
                    actual = check[C.feature_names(channels)].to_numpy()
                    prior = old.merge(sample[C.KEYS], on=C.KEYS).sort_values(C.KEYS)
                    expected = prior[C.feature_names(channels)].to_numpy()
                    actual = check.sort_values(C.KEYS)[C.feature_names(channels)].to_numpy()
                    if np.allclose(actual, expected, rtol=1e-5, atol=1e-20):
                        pending = windows.merge(old[C.KEYS], on=C.KEYS, how="left", indicator=True)
                        pending = pending.loc[pending._merge == "left_only", C.META]
                        added = feature_rows(raw, pending, channels)
                        frame = pd.concat([old, added], ignore_index=True) if len(added) else old.copy()
                        mode = "legacy_reuse_after_per_recording_sample_check"
                    else:
                        print(f"{patient}/{path.name}: legacy verification failed; recomputing", flush=True)
                if frame is None:
                    frame = feature_rows(raw, windows, channels)
            frame = frame[C.feature_names(channels) + C.META].sort_values(C.KEYS).reset_index(drop=True)
            validate_table(frame, windows, channels)
            atomic_csv(csv_path, frame)
            atomic_json(json_path, dict(identity=identity, sha256=digest(csv_path), origin=mode))
        prior_count = len(old) if old is not None else None
        prior_positive = int(old.label.sum()) if old is not None else None
        changes.append(dict(patient=patient, file=path.name, previous_windows=prior_count,
                            corrected_windows=len(windows), previous_seizure=prior_positive,
                            corrected_seizure=int(windows.label.sum()),
                            added_windows=len(windows)-prior_count if prior_count is not None else None))
        tables.append(frame)
        indices.append(windows)
        records.append({**identity, "checkpoint_sha256": digest(csv_path), "origin": mode})
        print(f"{patient}/{path.name}: {len(frame)} windows ({mode})", flush=True)
    frame = pd.concat(tables, ignore_index=True)
    index = pd.concat(indices, ignore_index=True)
    validate_table(frame, index, channels)
    atomic_csv(C.DATA_DIR / "feature_table.csv", frame)
    atomic_csv(C.DATA_DIR / "window_index.csv", index)
    atomic_csv(C.DATA_DIR / "window_count_changes.csv", pd.DataFrame(changes))
    atomic_json(C.DATA_DIR / "common_channels.json", channels)
    manifest = dict(status="complete", spec=spec, records=records,
                    feature_sha256=digest(C.DATA_DIR / "feature_table.csv"),
                    window_sha256=digest(C.DATA_DIR / "window_index.csv"),
                    windows=len(frame), seizure_windows=int(frame.label.sum()),
                    previous_windows=len(legacy) if legacy is not None else None,
                    previous_seizure_windows=int(legacy.label.sum()) if legacy is not None else None,
                    legacy_sha256=digest(legacy_path) if legacy is not None else None)
    atomic_json(C.DATA_DIR / "manifest.json", manifest)
    return manifest


def load_dataset():
    manifest = json.loads((C.DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Dataset build is incomplete")
    if manifest["spec"] != extraction_spec(manifest["spec"]["channels"]):
        raise ValueError("Dataset configuration/code/environment changed; run prepare")
    for key, name in [("feature_sha256", "feature_table.csv"), ("window_sha256", "window_index.csv")]:
        if digest(C.DATA_DIR / name) != manifest[key]:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    frame = read_table(C.DATA_DIR / "feature_table.csv")
    validate_table(frame, read_table(C.DATA_DIR / "window_index.csv"), manifest["spec"]["channels"])
    if set(frame.patient) != set(C.PATIENTS):
        raise ValueError("Unexpected patient set")
    return frame, manifest
