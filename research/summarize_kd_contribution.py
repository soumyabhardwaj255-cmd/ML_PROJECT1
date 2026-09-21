"""Aggregate completed contribution seeds without treating repeated patients as independent."""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from eeg_seizure import config as C
from eeg_seizure.artifacts import atomic_csv,atomic_json,digest
from eeg_seizure.dataset import read_table
from kd_contribution import verify_complete,effects


def average_effects_per_patient(effect):
    if effect.duplicated(["comparison","seed","test_patient"]).any():
        raise ValueError("Duplicate paired observation")
    return effect.groupby(["comparison","test_patient"])[["f1","average_precision","recall"]].mean().reset_index()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",nargs="+",type=Path,required=True)
    parser.add_argument("--run-id",required=True)
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in (".",".."):
        raise ValueError("One directory name required")
    sources={}; rows=[]; macros=[]; pools=[]
    for run in args.runs:
        m=verify_complete(run)
        sources[str(run.resolve())]=digest(run/"manifest.json")
        rows.append(read_table(run/"per_patient_metrics.csv"))
        macros.append(read_table(run/"mean_patient_metrics.csv"))
        pools.append(read_table(run/"pooled_window_metrics.csv"))
    rows=pd.concat(rows,ignore_index=True).sort_values(["seed","test_patient","model"]).reset_index(drop=True)
    assert not rows.duplicated(["seed","test_patient","model"]).any()
    assert set(rows.seed)=={42,43,44} and len(rows)==60
    macro=pd.concat(macros,ignore_index=True)
    pooled=pd.concat(pools,ignore_index=True)
    effect=effects(rows)
    per_seed=effect.groupby(["comparison","seed"])[["f1","average_precision","recall"]].agg(["mean","std","min","max"])
    per_seed.columns=[f"{metric}_{stat}" for metric,stat in per_seed.columns]
    per_seed=per_seed.reset_index()
    wins=effect.assign(f1_wins=(effect.f1>0).astype(int)).groupby(["comparison","seed"]).f1_wins.sum().reset_index()
    per_seed=per_seed.merge(wins,on=["comparison","seed"],validate="one_to_one")
    # First average the paired seed effects per patient; patients remain the unit.
    patient_effect=average_effects_per_patient(effect)
    patient_summary=patient_effect.groupby("comparison")[["f1","average_precision","recall"]].agg(["mean","std","min","max"])
    patient_summary.columns=[f"{metric}_{stat}" for metric,stat in patient_summary.columns]
    seed_metrics=macro[["model","seed","f1_mean","f1_std"]].merge(pooled[["model","seed","precision","recall","f1","average_precision","roc_auc"]],on=["model","seed"],validate="one_to_one")
    seed_variation=seed_metrics.groupby("model")[["f1_mean","precision","recall","f1","average_precision","roc_auc"]].agg(["mean","std"])
    seed_variation.columns=[f"{metric}_{stat}" for metric,stat in seed_variation.columns]
    patient_f1=rows.groupby(["test_patient","model"]).f1.agg(["mean","std"]).reset_index()
    output=C.ROOT/"results/contribution"/args.run_id
    output.mkdir(parents=True,exist_ok=False)
    tables={"per_patient_metrics.csv":rows,"paired_effects.csv":effect,"effects_by_seed.csv":per_seed,
        "patient_effects_averaged_over_seeds.csv":patient_effect,"patient_effect_summary.csv":patient_summary.reset_index(),
        "metrics_by_seed.csv":seed_metrics,"seed_variation.csv":seed_variation.reset_index(),"patient_f1_seed_variation.csv":patient_f1}
    for name,table in tables.items():
        atomic_csv(output/name,table)
    atomic_json(output/"manifest.json",dict(status="complete",version="kd-contribution-summary-v1",sources=sources,
        source_sha256=digest(__file__),patient_count=5,student_seeds=[42,43,44],teacher_seed=42,
        uncertainty="descriptive SD, five patients; student seeds conditional on fixed teacher/features; no pooled concatenation over seeds",
        artifacts={name:digest(output/name) for name in tables}))
    print("METRICS BY SEED\n"+seed_metrics.to_string(index=False))
    print("PAIRED EFFECTS BY SEED\n"+per_seed.to_string(index=False))
    print("PATIENT EFFECTS, SEED AVERAGED\n"+patient_effect.to_string(index=False))
    print("SEED VARIATION\n"+seed_variation.to_string())
    print("PATIENT SUMMARY\n"+patient_summary.to_string())
    print("PATIENT F1\n"+patient_f1.to_string(index=False))


if __name__=="__main__":
    main()
