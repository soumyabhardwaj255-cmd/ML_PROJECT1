"""Read canonical contracts; write only separately identified analysis outputs."""
from pathlib import Path
import numpy as np
import mne
from .. import config as C
from ..artifacts import digest, fingerprint, atomic_json, environment
from ..dataset import load_dataset, validated_annotations, window_rows
from ..channels import dedupe_channel_names
from ..preprocessing import preprocess_raw


def seed_for(seed, *identity):
    return int(fingerprint([seed, *identity])[:16], 16)


def load_windows(patient, recording, starts):
    if patient not in C.PATIENTS or Path(recording).name != recording:
        raise ValueError("Invalid recording identity")
    if not 1 <= len(starts) <= 32 or len(set(starts)) != len(starts):
        raise ValueError("Select 1-32 distinct explicit start times")
    frame, manifest = load_dataset()
    path = C.RAW_DIR / patient / recording
    record = next((r for r in manifest["records"] if r["patient"] == patient and r["file"] == recording), None)
    if record is None or digest(path) != record["raw_sha256"]:
        raise ValueError("EDF differs from the canonical dataset")
    summary = path.parent / f"{patient}-summary.txt"
    if digest(summary) != record["summary_sha256"]:
        raise ValueError("Annotation source changed")
    spec = manifest["spec"]
    with mne.io.read_raw_edf(path, preload=True, verbose="ERROR") as raw:
        if raw.info["sfreq"] != spec["sample_rate"]:
            raise ValueError("Sampling rate differs from canonical contract")
        intervals = validated_annotations(summary, recording, raw.n_times / raw.info["sfreq"])
        windows = window_rows(raw.n_times, raw.info["sfreq"], intervals, patient, recording)
        windows = windows[windows.start_sec.isin(starts)].sort_values("start_sec").reset_index(drop=True)
        if len(windows) != len(starts):
            raise ValueError("Requested start does not identify a complete canonical window")
        raw = preprocess_raw(dedupe_channel_names(raw), spec["channels"], **spec["preprocessing"])
        signals = np.stack([raw.get_data(start=int(r.start_sec*C.SAMPLE_RATE),
                                         stop=int(r.end_sec*C.SAMPLE_RATE)) for r in windows.itertuples()])
    reference = windows.merge(frame, on=C.META, validate="one_to_one")
    return windows, signals, reference, manifest, record


def new_run(run_id, configuration, dataset_manifest, record):
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("Run ID must be one directory name")
    path = C.ROOT / "results" / "analysis" / run_id
    path.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", analysis_version="psd-sr-cr-v1", exploratory=True,
                    configuration=configuration, dataset_sha256=dataset_manifest["feature_sha256"],
                    canonical_spec=dataset_manifest["spec"], recording=record,
                    environment=environment(),
                    source_hashes={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
                    fitting="none", input_stage="post-canonical-full-recording-filter; per-window analysis")
    atomic_json(path / "manifest.json", manifest)
    return path, manifest
