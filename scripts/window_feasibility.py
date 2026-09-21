"""Versioned, analysis-only 4 s/10 s feasibility; never changes canonical config."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eeg_seizure import config as C
from eeg_seizure.artifacts import atomic_csv, atomic_json, digest, environment
from eeg_seizure.dataset import validated_annotations, window_rows
from eeg_seizure.windowing import label_window
from eeg_seizure.channels import dedupe_channel_names
from eeg_seizure.preprocessing import preprocess_raw
from eeg_seizure.analysis.runner import analyse


def duration_windows(n_samples, sfreq, intervals, patient, filename, seconds):
    """Sample-count boundary logic matching canonical window_rows, explicit duration."""
    size = round(seconds * sfreq)
    if size <= 0 or not np.isclose(size / sfreq, seconds):
        raise ValueError("Duration must map to integer samples")
    rows = []
    for start in range(0, n_samples - size + 1, size):
        a, b = start / sfreq, (start + size) / sfreq
        overlap = sum(max(0, min(b, y) - max(a, x)) for x, y in intervals)
        rows.append(dict(patient=patient, file=filename, start_sec=a, end_sec=b,
                         label=label_window(a, b, intervals, C.OVERLAP_THRESHOLD),
                         overlap_sec=overlap, mixed=0 < overlap < seconds))
    return pd.DataFrame(rows, columns=C.META + ["overlap_sec", "mixed"])


def psd_validity(matrix, sfreq):
    """Same Welch settings as analysis.spectral.psd, vectorized over channels."""
    _, power = welch(matrix, fs=sfreq, window="hann", nperseg=256, noverlap=128,
                     nfft=256, detrend="constant", scaling="density", average="mean", axis=-1)
    valid = np.isfinite(power).all(axis=-1) & (power >= 0).all(axis=-1) & (power.sum(axis=-1) > 0)
    return valid


def protected_files():
    roots = [C.ROOT / x for x in ("src", "notebooks", "data/processed/corrected_v2", "results/corrected", "results/analysis")]
    return {str(p.relative_to(C.ROOT)): digest(p) for root in roots for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="window-feasibility-4s-10s-v1")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise ValueError("One directory name required")
    # Outside all existing analysis/canonical output roots; refuse overwrite.
    output = C.ROOT / "results" / "feasibility" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    before = protected_files()
    dm = json.loads((C.DATA_DIR / "manifest.json").read_text())
    spec = dm["spec"]
    canonical_index = pd.read_csv(C.DATA_DIR / "window_index.csv")
    manifest = dict(status="running", version="duration-feasibility-v1", environment=environment(),
                    durations=[4, 10], canonical_spec=spec, canonical_manifest_sha256=digest(C.DATA_DIR / "manifest.json"),
                    script_sha256=digest(__file__), source_hashes={k:v for k,v in before.items() if k.startswith("src/")},
                    fitting="none", selection="First annotated recording in chb01/chb02, start 0 and first common-20s-grid fully ictal context; historical chb08_21 starts 0/2140",
                    dynamics_scope="six contexts, FP1-F7 only; three trials, D=0/.1/.5/2, substeps=1/2; no parameter tuning",
                    sr_validity="Finite local spectral ratio and finite zero-noise comparison, no clipping; numerical availability, NOT evidence of SR",
                    psd_validity="All 22 channel PSDs finite, nonnegative and positive total power; census, NOT feature usefulness",
                    unchanged_files_before=before)
    atomic_json(output / "manifest.json", manifest)
    indices, record_counts, selections = [], [], []
    selected = {4: [], 10: []}
    signals = {4: [], 10: []}
    chosen_patients = set()
    try:
        for i, record in enumerate(dm["records"]):
            patient, filename = record["patient"], record["file"]
            path = C.RAW_DIR / patient / filename
            summary = path.parent / f"{patient}-summary.txt"
            if digest(path) != record["raw_sha256"] or digest(summary) != record["summary_sha256"]:
                raise ValueError("Raw data/annotation checksum mismatch")
            with mne.io.read_raw_edf(path, preload=True, verbose="ERROR") as raw:
                fs = raw.info["sfreq"]
                if fs != spec["sample_rate"]:
                    raise ValueError("Sampling rate mismatch")
                intervals = validated_annotations(summary, filename, raw.n_times / fs)
                tables = {d: duration_windows(raw.n_times, fs, intervals, patient, filename, d) for d in (4, 10)}
                pd.testing.assert_frame_equal(tables[4][C.META], window_rows(raw.n_times, fs, intervals, patient, filename))
                expected = canonical_index[(canonical_index.patient == patient) & (canonical_index.file == filename)][C.META].reset_index(drop=True)
                pd.testing.assert_frame_equal(tables[4][C.META], expected, check_dtype=False)
                starts = []
                cohort = "development"
                if patient in ("chb01", "chb02") and patient not in chosen_patients and intervals:
                    candidates = [20 * int(np.ceil(a / 20)) for a,b in intervals if 20 * int(np.ceil(a / 20)) + 10 <= b]
                    if candidates and label_window(0, 10, intervals) == 0:
                        starts = [0, candidates[0]]
                        chosen_patients.add(patient)
                if patient == "chb08" and filename == "chb08_21.edf":
                    starts, cohort = [0, 2140], "historical_comparator_not_for_tuning"
                raw = preprocess_raw(dedupe_channel_names(raw), spec["channels"], **spec["preprocessing"])
                for d, table in tables.items():
                    validity = []
                    for row in table.itertuples():
                        matrix = raw.get_data(start=round(row.start_sec * fs), stop=round(row.end_sec * fs))
                        valid = psd_validity(matrix, fs)
                        validity.append(int(valid.sum()))
                        if row.start_sec in starts:
                            selected[d].append({k:getattr(row,k) for k in C.META})
                            signals[d].append(matrix.copy())
                            selections.append(dict(duration_sec=d, cohort=cohort, **selected[d][-1]))
                    table = table.assign(duration_sec=d, psd_valid_channels=validity)
                    indices.append(table)
                    record_counts.append(dict(patient=patient, file=filename, duration_sec=d, windows=len(table),
                        seizure=int(table.label.sum()), nonseizure=int((table.label==0).sum()),
                        mixed=int(table.mixed.sum()), mixed_negative=int((table.mixed & (table.label==0)).sum()),
                        seizure_seconds_in_negative_windows=float(table.loc[table.label==0,"overlap_sec"].sum()),
                        discarded_tail_sec=raw.n_times/fs-len(table)*d,
                        psd_valid_windows=int((table.psd_valid_channels==22).sum()),
                        psd_valid_channel_windows=int(table.psd_valid_channels.sum())))
            print(f"Census {i+1}/{len(dm['records'])}: {patient}/{filename}", flush=True)
        if any(len(selected[d]) != 6 for d in (4,10)):
            raise AssertionError("Expected all six predefined contexts")
        atomic_csv(output / "window_index_psd.csv", pd.concat(indices, ignore_index=True))
        counts = pd.DataFrame(record_counts)
        atomic_csv(output / "recording_counts.csv", counts)
        numeric = [c for c in counts if c not in ("patient","file","duration_sec")]
        atomic_csv(output / "census_summary.csv", counts.groupby("duration_sec")[numeric].sum().reset_index())
        atomic_csv(output / "patient_counts.csv", counts.groupby(["duration_sec","patient"])[numeric].sum().reset_index())
        atomic_csv(output / "selected_contexts.csv", pd.DataFrame(selections))
        trial_tables = []
        for d in (4,10):
            folder = output / f"dynamics_{d}s"
            folder.mkdir()
            options = SimpleNamespace(channel="FP1-F7", substeps=[1,2], noise_levels=[0.,.1,.5,2.], trials=3, seed=42)
            analyse(pd.DataFrame(selected[d]), np.stack(signals[d]), spec, options, folder)
            trials = pd.read_csv(folder / "dynamics_trials.csv")
            trials["duration_sec"] = d
            trials["sr_numerically_valid"] = np.isfinite(trials.snr_db) & np.isfinite(trials.zero_noise_snr_db) & (trials.clipped_fraction == 0)
            trial_tables.append(trials)
            print(f"Dynamics {d}s complete", flush=True)
        trials = pd.concat(trial_tables, ignore_index=True)
        atomic_csv(output / "paired_dynamics_trials.csv", trials)
        grouped = trials.groupby(["duration_sec", "patient", "model", "drive"]).agg(
            trials=("valid","size"), valid_cr=("valid","sum"), insufficient=("reason",lambda s:int((s=="insufficient_intervals").sum())),
            events=("events","sum"), intervals=("intervals","sum"), min_events=("events","min"), max_events=("events","max"),
            sr_numerically_valid=("sr_numerically_valid","sum"), max_clipped_fraction=("clipped_fraction","max")).reset_index()
        atomic_csv(output / "validity_summary.csv", grouped)
        # Exact reproduction of the historical 4-second diagnostic before comparison.
        old = pd.read_csv(C.ROOT / "results/analysis/psd-sr-cr-validated-chb08/dynamics_trials.csv")
        keys = C.META + ["channel","model","drive","noise_intensity","substeps","trial"]
        new = trials[(trials.duration_sec==4)&(trials.patient=="chb08")]
        columns = keys + ["events","intervals","valid","reason","cv","coherence","snr_db"]
        pd.testing.assert_frame_equal(old[columns].sort_values(keys).reset_index(drop=True),
                                      new[columns].sort_values(keys).reset_index(drop=True), check_dtype=False)
        after = protected_files()
        if before != after:
            raise AssertionError("Protected canonical or historical artifact changed")
        manifest.update(status="complete", historical_4s_parity=True, protected_files_unchanged=len(before),
                        artifacts={str(p.relative_to(output)):digest(p) for p in output.rglob("*") if p.is_file() and p.name!="manifest.json"})
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        atomic_json(output / "manifest.json", manifest)
    print(output)


if __name__ == "__main__":
    main()
