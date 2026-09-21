"""Read-only validation of saved ablation identities, transforms, predictions and metrics."""
import argparse
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from eeg_seizure import config as C
from eeg_seizure.artifacts import digest
from eeg_seizure.dataset import load_dataset, read_table
from eeg_seizure.evaluation import metrics, summarize
from psd_ablation import ARMS, MEASURES, arm_frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",default="psd-representation-v1")
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in (".",".."):
        raise ValueError("One run directory name required")
    run=C.ROOT/"results/ablations"/args.run_id
    manifest=json.loads((run/"manifest.json").read_text())
    assert manifest["status"]=="complete"
    assert manifest["source_sha256"]==digest(Path(__file__).with_name("psd_ablation.py"))
    for relative,expected in manifest["protected_before"].items():
        assert digest(C.ROOT/relative)==expected,relative
    for relative,expected in manifest["artifacts"].items():
        assert digest(run/relative)==expected,relative
    canonical,dm=load_dataset()
    schemas=json.loads((run/"feature_schemas.json").read_text())
    predictions=read_table(run/"predictions.csv")
    per_patient=read_table(run/"per_patient_metrics.csv")
    assert len(predictions)==3*len(canonical)
    assert not predictions.duplicated(["model"]+C.KEYS).any()
    assert set(predictions.model)==set(ARMS)
    audit=json.loads((run/"inner_fold_audit.json").read_text())
    assert len(audit)==60
    for fold in audit:
        expected=set(C.PATIENTS)-{fold["outer_patient"],fold["validation_patient"]}
        assert len(expected)==3 and set(fold["train_patients"])==expected
        inner=canonical[canonical.patient.isin(expected)]
        assert fold["training_windows"]==len(inner)
        assert fold["training_positive"]==int(inner.label.sum())
        assert fold["class_ratio"]==float((inner.label==0).sum()/(inner.label==1).sum())
    for arm in ARMS:
        values=None
        if arm!=ARMS[0]:
            feature_table=read_table(run/f"{arm}_features.csv")
            pd.testing.assert_frame_equal(feature_table[C.META],canonical[C.META],check_dtype=False)
            values=feature_table[[c for c in feature_table if c.startswith("freq")]].to_numpy()
            assert np.isfinite(values).all() and (values>=0).all() and (values<=1+1e-12).all()
        frame,columns=arm_frame(canonical,values,arm,dm["spec"]["feature_columns"])
        assert schemas[arm]==columns
        for patient in C.PATIENTS:
            train=frame[frame.patient!=patient]
            test=frame[frame.patient==patient]
            bundle=joblib.load(run/f"bundle_{arm}_{patient}.joblib")
            assert bundle["format_version"]=="psd-ablation-v1"
            assert bundle["feature_columns"]==columns and bundle["test_patients"]==[patient]
            assert set(bundle["train_patients"])==set(C.PATIENTS)-{patient}
            transform=bundle["estimator"].named_steps["features"]
            np.testing.assert_allclose(transform.scaler_.mean_,train[columns].to_numpy().mean(axis=0),rtol=1e-10,atol=1e-20)
            group=predictions[(predictions.model==arm)&(predictions.patient==patient)].reset_index(drop=True)
            pd.testing.assert_frame_equal(group[C.META],test[C.META].reset_index(drop=True),check_dtype=False)
            scores=bundle["estimator"].predict_proba(test[columns])[:,1]
            np.testing.assert_array_equal(scores,group.probability_seizure)
            np.testing.assert_array_equal(scores>=bundle["threshold"],group.prediction)
            inner=read_table(run/f"inner_{arm}_{patient}.csv")
            pd.testing.assert_frame_equal(inner[C.META],train[C.META].reset_index(drop=True),check_dtype=False)
            candidates=[f1_score(inner.label,inner.probability_seizure>=t,zero_division=0) for t in C.THRESHOLDS]
            assert bundle["threshold"]==C.THRESHOLDS[int(np.argmax(candidates))]
            actual=metrics(test.label,scores,bundle["threshold"])
            actual["specificity"]=actual["tn"]/(actual["tn"]+actual["fp"])
            actual["balanced_accuracy"]=(actual["specificity"]+actual["recall"])/2
            recorded=per_patient[(per_patient.model==arm)&(per_patient.test_patient==patient)].iloc[0]
            for key,value in actual.items():
                np.testing.assert_allclose(value,recorded[key],rtol=1e-12,atol=1e-15)
    _,pooled=summarize(predictions,per_patient)
    pooled["specificity"]=pooled.tn/(pooled.tn+pooled.fp)
    pooled["balanced_accuracy"]=(pooled.specificity+pooled.recall)/2
    stored=read_table(run/"pooled_window_metrics.csv")
    pd.testing.assert_frame_equal(pooled.sort_values("model").reset_index(drop=True),stored[pooled.columns].sort_values("model").reset_index(drop=True),check_exact=False,rtol=1e-12,atol=1e-15)
    print("PER PATIENT\n"+per_patient[["model","test_patient","threshold"]+MEASURES].to_string(index=False))
    print("POOLED\n"+stored[["model"]+MEASURES+["tp","fp","fn","tn"]].to_string(index=False))
    print("MACRO\n"+per_patient.groupby("model")[MEASURES].agg(["mean","std"]).to_string())
    delta=read_table(run/"paired_patient_deltas.csv")
    baseline=per_patient[per_patient.model==ARMS[0]].set_index("test_patient")
    for arm in ARMS[1:]:
        other=per_patient[per_patient.model==arm].set_index("test_patient")
        expected=other[MEASURES]-baseline[MEASURES]
        saved=delta[delta.comparison==f"{arm}-canonical_absolute"].set_index("test_patient")[MEASURES]
        pd.testing.assert_frame_equal(expected,saved,check_exact=False,rtol=1e-12,atol=1e-15)
    macro=per_patient.groupby("model")[MEASURES].agg(["mean","std","min","max"])
    macro.columns=[f"{m}_{s}" for m,s in macro.columns]
    pd.testing.assert_frame_equal(macro.reset_index(),read_table(run/"macro_patient_metrics.csv"),check_exact=False,rtol=1e-12,atol=1e-15)
    print("PAIRED DELTAS\n"+delta.groupby("comparison")[["f1","average_precision","recall"]].agg(["mean","std","min","max"]).to_string())
    print(json.dumps(dict(validated=True,protected_files=manifest["protected_files_unchanged"],artifact_hashes=len(manifest["artifacts"]),bundle_checks=15,zero_band_power_windows=manifest["zero_band_power_windows"])))


if __name__=="__main__":
    main()
