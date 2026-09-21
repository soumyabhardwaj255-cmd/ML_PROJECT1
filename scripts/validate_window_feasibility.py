"""Read-only verification and concise summaries of the duration feasibility run."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eeg_seizure import config as C
from eeg_seizure.artifacts import digest
from eeg_seizure.dataset import validated_annotations


def main():
    root = C.ROOT / "results/feasibility/window-feasibility-4s-10s-v1"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["historical_4s_parity"]
    assert digest(C.ROOT / "scripts/window_feasibility.py") == manifest["script_sha256"]
    for relative, expected in manifest["unchanged_files_before"].items():
        assert digest(C.ROOT / relative) == expected, relative
    for relative, expected in manifest["artifacts"].items():
        assert digest(root / relative) == expected, relative
    windows = pd.read_csv(root / "window_index_psd.csv")
    trials = pd.read_csv(root / "paired_dynamics_trials.csv")
    assert len(trials) == 1152
    assert not trials.duplicated(["duration_sec"] + C.KEYS + ["model","drive","noise_intensity","substeps","trial"]).any()
    invalid = trials.reason == "insufficient_intervals"
    assert (~trials.loc[invalid,"valid"]).all()
    assert trials.loc[invalid,["cv","coherence"]].isna().all().all()
    assert (trials.events >= trials.intervals + 1).where(trials.events > 0, trials.intervals==0).all()
    assert np.isfinite(trials.loc[trials.valid,"coherence"]).all()
    counts = pd.read_csv(root / "census_summary.csv").set_index("duration_sec")
    for d,g in windows.groupby("duration_sec"):
        assert len(g) == counts.loc[d,"windows"]
        assert g.label.sum() == counts.loc[d,"seizure"]
        assert (g.psd_valid_channels==22).sum() == counts.loc[d,"psd_valid_windows"]
        assert (g.end_sec-g.start_sec==d).all()
    event_rows = []
    for (patient,file),g in windows.groupby(["patient","file"]):
        # Annotation bounds were already validated against actual EDF sample counts.
        intervals = validated_annotations(C.RAW_DIR / patient / f"{patient}-summary.txt", file, float(g.end_sec.max()+10))
        for a,b in intervals:
            for d in (4,10):
                w = g[(g.duration_sec==d)&(g.label==1)]
                overlaps = np.maximum(0, np.minimum(w.end_sec,b)-np.maximum(w.start_sec,a))
                positive = w[overlaps>0]
                event_rows.append(dict(duration_sec=d, patient=patient, file=file, onset=a, offset=b,
                    represented=bool(len(positive)), seizure_sec_in_positive=float(overlaps.sum()),
                    seizure_duration=b-a,
                    earliest_positive_window_end_lag=float(positive.end_sec.min()-a) if len(positive) else None))
    events = pd.DataFrame(event_rows)
    print("CENSUS\n" + counts.to_string())
    print("DYNAMICS\n" + trials.groupby(["duration_sec","model"]).agg(
        trials=("valid","size"),valid=("valid","sum"),insufficient=("reason",lambda s:(s=="insufficient_intervals").sum()),
        events=("events","sum"), intervals=("intervals","sum"), min_events=("events","min"),max_events=("events","max"),
        sr_numerically_valid=("sr_numerically_valid","sum"),clipped=("clipped_fraction","max")).to_string())
    print("HISTORICAL COMPARATOR\n" + trials[trials.patient=="chb08"].groupby(["duration_sec","model"]).agg(trials=("valid","size"),valid=("valid","sum"), events=("events","sum"),intervals=("intervals","sum")).to_string())
    print("ANNOTATED SEIZURES\n" + events.groupby("duration_sec").agg(events=("represented","size"), represented=("represented","sum"),
          total_seizure_sec=("seizure_duration","sum"),seizure_sec_in_positive=("seizure_sec_in_positive","sum"),
          median_label_end_lag=("earliest_positive_window_end_lag","median"),max_label_end_lag=("earliest_positive_window_end_lag","max")).to_string())
    print(json.dumps(dict(validated=True, artifacts=len(manifest["artifacts"]), protected_files=manifest["protected_files_unchanged"])))


if __name__ == "__main__":
    main()
