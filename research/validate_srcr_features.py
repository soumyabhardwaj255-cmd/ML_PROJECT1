"""Validate separate 334-feature artifacts and report coverage; never fit a model."""
import argparse
import json
import sys
from pathlib import Path
import mne
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from eeg_seizure import config as C
from eeg_seizure.artifacts import digest,atomic_json,atomic_csv
from eeg_seizure.dataset import load_dataset,read_table,validated_annotations
from eeg_seizure.analysis.coherence import fhn_events,regularity
from srcr_features import VALUE_COLUMNS,MASK_COLUMNS,EXTRA_COLUMNS,context_bounds,oscillator,innovations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",default="srcr-context-features-v1")
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in (".",".."):
        raise ValueError("One run ID required")
    root=C.ROOT/"results/feature_integration"/args.run_id
    m=json.loads((root/"manifest.json").read_text())
    assert m["status"]=="complete"
    assert m["source_sha256"]==digest(Path(__file__).with_name("srcr_features.py"))
    for path,expected in m["artifacts"].items(): assert digest(root/path)==expected,path
    for path,expected in m["protected_before"].items(): assert digest(C.ROOT/path)==expected,path
    canonical,dm=load_dataset(); assert m["canonical_dataset_sha256"]==dm["feature_sha256"]
    diagnostics=read_table(root/"diagnostics.csv")
    assert len(diagnostics)==2*len(canonical)
    assert not diagnostics.duplicated(["mode"]+C.KEYS).any()
    schemas=json.loads((root/"schemas.json").read_text())
    annotation=[]
    for rec in dm["records"]:
        patient,file=rec["patient"],rec["file"]
        path=C.RAW_DIR/patient/file
        with mne.io.read_raw_edf(path,preload=False,verbose="ERROR") as header:
            size=header.n_times
        intervals=validated_annotations(path.parent/f"{patient}-summary.txt",file,size/256)
        group=diagnostics[(diagnostics.patient==patient)&(diagnostics.file==file)]
        for row in group.itertuples():
            a,b,reason=context_bounds(row.start_sec,row.end_sec,size,row.mode)
            assert row.context_start_sec==a/256 and row.context_end_sec==b/256 and row.context_reason==reason
            if reason=="ok":
                overlap=sum(max(0,min(b/256,y)-max(a/256,x)) for x,y in intervals)
                target=sum(max(0,min(row.end_sec,y)-max(row.start_sec,x)) for x,y in intervals)
                future=sum(max(0,min(b/256,y)-max(row.end_sec,x)) for x,y in intervals)
                annotation.append(dict(mode=row.mode,label=row.label,context_has_seizure=overlap>0,
                    extra_context_seizure=overlap>target,future_seizure=future>0))
    valid_diagnostics=diagnostics[diagnostics.context_reason=="ok"]
    assert valid_diagnostics.max_clipped_fraction.notna().all()
    assert diagnostics.loc[diagnostics.cr_reason!="ok","cr_value"].isna().all()
    assert diagnostics.loc[diagnostics.sr_reason!="ok","sr_value"].isna().all()
    for mode in ("centered","trailing"):
        table=read_table(root/f"extended_{mode}.csv")
        assert list(table.columns)==C.META+schemas[mode]["feature_columns"]
        assert schemas[mode]["feature_columns"]==dm["spec"]["feature_columns"]+EXTRA_COLUMNS
        pd.testing.assert_frame_equal(table[canonical.columns],canonical,check_exact=True)
        diagnostic=diagnostics[diagnostics["mode"]==mode].reset_index(drop=True)
        pd.testing.assert_frame_equal(table[C.META],diagnostic[C.META],check_exact=True)
        for value,raw,mask in zip(VALUE_COLUMNS,["sr_value","cr_value"],MASK_COLUMNS):
            np.testing.assert_array_equal(table[value],diagnostic[raw])
            np.testing.assert_array_equal(table[mask],table[value].isna().astype(int))
            assert not np.isinf(table[value]).any()
        assert (table[VALUE_COLUMNS[1]].dropna()>0).all()
        # Preflight only: every outer training partition has observed values;
        # no imputer, scaler, model or feature selector is fitted here.
        for held_out in C.PATIENTS:
            train=table[table.patient!=held_out]
            assert train[VALUE_COLUMNS].notna().any().all()
    coverage=diagnostics.groupby(["mode","label"]).agg(windows=("label","size"),
        full_context=("context_reason",lambda s:int((s=="ok").sum())),sr_valid=("sr_value","count"),
        cr_valid=("cr_value","count"),joint_valid=("joint_valid","sum"),bistable_valid=("bistable_valid","sum"),
        bistable_events=("bistable_events","sum"),fhn_events=("fhn_events","sum")).reset_index()
    pd.testing.assert_frame_equal(coverage,read_table(root/"coverage.csv"),check_dtype=False)
    seeds=read_table(root/"seed_sensitivity.csv")
    seed_summary=seeds.groupby(["mode","seed"]).agg(contexts=("seed","size"),sr_valid=("sr_value","count"),cr_valid=("cr_value","count"),
        cr_mean=("cr_value","mean"),cr_std=("cr_value","std"),sr_mean=("sr_value","mean"),sr_std=("sr_value","std")).reset_index()
    correlations=[]
    for mode in ("centered","trailing"):
        for value in ("sr_value","cr_value"):
            pivot=seeds[seeds["mode"]==mode].pivot(index=["patient","file","start_sec"],columns="seed",values=value)
            for seed in (43,44):
                paired=pivot[[42,seed]].dropna()
                correlations.append(dict(mode=mode,feature=value,comparison=f"42 vs {seed}",paired_contexts=len(paired),spearman=paired.corr(method="spearman").iloc[0,1]))
    no_drive=[]
    for seed in (42,43,44):
        output,clipped=oscillator(np.zeros(2560),innovations(seed),.5,1)
        no_drive.append(dict(seed=seed,clipped_fraction=clipped,**regularity(fhn_events(output,256),256,minimum_intervals=3,ddof=0,minimum_interval_sec=.2)))
    summary=valid_diagnostics.groupby("mode").agg(contexts=("mode","size"),
        sr_min=("sr_value","min"),sr_max=("sr_value","max"),sr_std=("sr_value","std"),sr_unique=("sr_value","nunique"),
        cr_min=("cr_value","min"),cr_max=("cr_value","max"),cr_std=("cr_value","std"),cr_unique=("cr_value","nunique"),
        max_clipped=("max_clipped_fraction","max")).reset_index()
    review=root.parent/(args.run_id+"-validation")
    review.mkdir(parents=True,exist_ok=False)
    for name,table in (("feature_ranges.csv",summary),("seed_summary.csv",seed_summary),("seed_correlations.csv",pd.DataFrame(correlations)),
                       ("no_drive_fhn.csv",pd.DataFrame(no_drive)),("context_annotation_audit.csv",pd.DataFrame(annotation).groupby(["mode","label"]).sum().reset_index())):
        atomic_csv(review/name,table)
    atomic_json(review/"receipt.json",dict(validated=True,input_manifest_sha256=digest(root/"manifest.json"),
        validator_sha256=digest(__file__),protected_files=m["protected_files_unchanged"],artifact_hashes=len(m["artifacts"]),
        canonical_columns_exact=True,dimensions=334,imputation_and_model_fitting="none",
        artifacts={p.name:digest(p) for p in review.iterdir()}))
    print("COVERAGE\n"+coverage.to_string(index=False))
    print("REASONS\n"+read_table(root/"missing_reasons.csv").to_string(index=False))
    print("RANGES\n"+summary.to_string(index=False))
    print("SEEDS\n"+seed_summary.to_string(index=False))
    print("SEED CORRELATIONS\n"+pd.DataFrame(correlations).to_string(index=False))
    print("NO DRIVE\n"+pd.DataFrame(no_drive).to_string(index=False))
    print(json.dumps(dict(validated=True,protected_files=m["protected_files_unchanged"],parity_contexts=m["parity_contexts"],seconds=m["seconds"])))


if __name__=="__main__": main()
