"""Read-only validation of completed bounded runs; write a separate receipt."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from eeg_seizure.artifacts import digest,atomic_json
from eeg_seizure.analysis.reporting import summarize,raw_optima


def main():
    base=ROOT/"results/analysis"
    protected=json.loads((base/"preservation_baseline.json").read_text())
    for path,checksum in protected.items():
        assert digest(ROOT/path)==checksum, f"Protected artifact changed: {path}"
    runs={}
    for name in ("psd-sr-cr-validated-chb08","frozen-kd-robustness-validated","frozen-lr-robustness-validation"):
        path=base/name
        manifest=json.loads((path/"manifest.json").read_text())
        assert manifest["status"]=="complete"
        for file,checksum in manifest["artifact_hashes"].items():
            assert digest(path/file)==checksum, f"Output checksum mismatch: {file}"
        for file,checksum in manifest["source_hashes"].items():
            assert digest(ROOT/"src/eeg_seizure/analysis"/file)==checksum, f"Run used different source: {file}"
        record=manifest["recording"]
        raw=ROOT/"data/raw"/record["patient"]/record["file"]
        assert digest(raw)==record["raw_sha256"]
        assert digest(raw.parent/f"{record['patient']}-summary.txt")==record["summary_sha256"]
        runs[name]=manifest
    dynamics=base/"psd-sr-cr-validated-chb08"
    trials=pd.read_csv(dynamics/"dynamics_trials.csv",float_precision="round_trip")
    summary=pd.read_csv(dynamics/"dynamics_summary.csv",float_precision="round_trip")
    expected=summarize(trials)
    pd.testing.assert_frame_equal(summary,expected,check_dtype=False,rtol=1e-12,atol=1e-12)
    measured=pd.read_csv(dynamics/"raw_grid_optima.csv",float_precision="round_trip")
    pd.testing.assert_frame_equal(measured,raw_optima(expected),check_dtype=False,rtol=1e-12,atol=1e-12)
    assert len(trials)==192 and summary.trials.sum()==192
    invalid=trials[~trials.valid]
    assert invalid.reason.eq("insufficient_intervals").all()
    assert invalid.coherence.isna().all() and invalid.cv.isna().all()
    assert invalid.events.notna().all() and invalid.intervals.notna().all()
    by_model=trials.groupby("model").valid.agg(total="size",valid="sum")
    by_model["invalid"]=by_model.total-by_model.valid
    robust=base/"frozen-kd-robustness-validated"
    predictions=pd.read_csv(robust/"predictions.csv",float_precision="round_trip")
    counts=pd.read_csv(robust/"robustness_summary.csv")
    assert len(predictions)==22
    for (condition,trial),group in predictions.groupby(["condition","trial"]):
        changed=(group.prediction!=group.clean_prediction)
        np.testing.assert_array_equal(group.decision_changed,changed)
        row=counts[(counts.condition==condition)&(counts.trial==trial)].iloc[0]
        assert row.decision_changes==changed.sum()
    clean=predictions[predictions.condition=="clean"]
    assert not clean.decision_changed.any()
    np.testing.assert_array_equal(clean.probability_seizure,clean.clean_probability)
    bundle=runs["frozen-kd-robustness-validated"]["configuration"]["bundle"]
    assert digest(ROOT/bundle)==runs["frozen-kd-robustness-validated"]["result"]["bundle_sha256"]
    lr=runs["frozen-lr-robustness-validation"]
    assert digest(ROOT/lr["configuration"]["bundle"])==lr["result"]["bundle_sha256"]
    lr_predictions=pd.read_csv(base/"frozen-lr-robustness-validation/predictions.csv")
    lr_clean=lr_predictions[lr_predictions.condition=="clean"]
    assert len(lr_predictions)==14 and not lr_clean.decision_changed.any()
    np.testing.assert_allclose(lr_clean.probability_seizure,lr_clean.clean_probability,rtol=1e-5,atol=1e-7)
    result=dict(status="passed",protected_artifacts_unchanged=len(protected),
                raw_edf_and_summary_unchanged=True,outputs_and_source_hashes_verified=True,
                trial_and_raw_optimum_reports_recomputed=True,dynamics=by_model.reset_index().to_dict("records"),
                invalid_trials_preserved=len(invalid),maximum_clipped_fraction=float(trials.clipped_fraction.max()),
                frozen_prediction_rows=len(predictions),classical_prediction_rows=len(lr_predictions),clean_prediction_parity=True,
                fitting="none",scope="two selected four-second windows; no population or resonance claim")
    atomic_json(base/"validation_receipt.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
