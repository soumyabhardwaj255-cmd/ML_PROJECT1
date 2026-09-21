"""Read-only saved contribution experiment validation; no fitting."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from eeg_seizure import config as C
from eeg_seizure.artifacts import digest,fingerprint
from eeg_seizure.dataset import load_dataset,read_table
from eeg_seizure.evaluation import metrics,summarize,patient_split
from eeg_seizure.modeling import select_features,class_ratio
from eeg_seizure.inference import load_bundle,predict_features
from kd_contribution import ARMS,random_features,initial_hash,effects,verify_complete


def validate(run):
    manifest=verify_complete(run)
    assert manifest["source_sha256"]==digest(Path(__file__).with_name("kd_contribution.py"))
    if manifest["primary_run"]:
        parent=Path(manifest["primary_run"])
        assert digest(parent/"manifest.json")==manifest["primary_manifest_sha256"]
        verify_complete(parent)
    for path,expected in manifest["protected_before"].items():
        assert digest(C.ROOT/path)==expected,path
    frame,dm=load_dataset()
    assert manifest["dataset_sha256"]==dm["feature_sha256"]
    predictions=read_table(run/"predictions.csv")
    rows=read_table(run/"per_patient_metrics.csv")
    assert len(predictions)==len(frame)*4*len(manifest["seeds"])
    assert not predictions.duplicated(C.KEYS+["model","seed"]).any()
    for patient in C.PATIENTS:
        train,test=patient_split(frame,patient)
        for seed in manifest["seeds"]:
            folder=run/f"seed{seed}"/patient
            audit=json.loads((folder/"fold.json").read_text())
            complete=json.loads((folder/"complete.json").read_text())
            assert complete["status"]=="complete"
            required={"fold.json","teacher_importance.csv","feature_selection.csv","predictions.csv","training_log.json"}|{f"{a}.joblib" for a in ARMS}
            assert required==set(complete["artifacts"])
            for path,expected in complete["artifacts"].items():
                assert digest(folder/path)==expected
            assert audit["train_patients"]==sorted(train.patient.unique()) and audit["test_patients"]==[patient]
            assert audit["training_windows"]==len(train) and audit["training_positive"]==int(train.label.sum())
            assert audit["class_ratio"]==class_ratio(train.label)
            assert audit["train_window_sha256"]==fingerprint(train[C.META].to_dict("records"))
            assert audit["test_window_sha256"]==fingerprint(test[C.META].to_dict("records"))
            assert audit["initialization_sha256"]==initial_hash(seed) and audit["loader_seed"]==seed
            random_names,random_seed=random_features(train,dm["spec"]["feature_columns"],patient)
            assert audit["random_features"]==random_names and audit["random_selection_seed"]==random_seed
            if manifest["primary_run"]:
                original=json.loads((parent/f"seed42/{patient}/fold.json").read_text())
                for key in ("selected_features","random_features","teacher_target_sha256","train_window_sha256","test_window_sha256"):
                    assert audit[key]==original[key]
            teacher=load_bundle(folder/"teacher.joblib")["estimator"]
            assert teacher.named_steps["model"].get_params()["scale_pos_weight"]==class_ratio(train.label)
            selected=select_features(teacher,dm["spec"]["feature_columns"]).feature.tolist()
            assert audit["selected_features"]==selected
            targets=teacher.predict_proba(train[dm["spec"]["feature_columns"]])[:,1]
            assert hashlib.sha256(targets.tobytes()).hexdigest()==audit["teacher_target_sha256"]
            feature_table=read_table(folder/"feature_selection.csv")
            logs=json.loads((folder/"training_log.json").read_text())
            assert {l["model"] for l in logs}==set(ARMS[1:])
            for log in logs:
                assert len(log["loss"])==40 and np.isfinite(log["loss"]).all() and log["parameters"]==2177
                assert log["initialization_sha256"]==audit["initialization_sha256"] and log["loader_seed"]==seed
            for name in ARMS:
                bundle=load_bundle(folder/f"{name}.joblib")
                assert bundle["train_patients"]==sorted(train.patient.unique()) and bundle["test_patients"]==[patient]
                assert bundle["threshold"]==.5 and bundle["feature_columns"]==dm["spec"]["feature_columns"]
                if name!="teacher":
                    names=random_names if name=="random_supervised" else selected
                    assert bundle["selected_features"]==names
                    assert feature_table[feature_table.model==name].feature.tolist()==names
                    assert bundle["architecture"]==[50,32,16,1] and bundle["seed"]==seed
                    assert bundle["training_configuration"]==C.STUDENT
                    transform=bundle["transform"]
                    np.testing.assert_allclose(transform.imputer_.statistics_,train[names].median().to_numpy(),rtol=1e-12,atol=1e-20)
                    np.testing.assert_allclose(transform.scaler_.mean_,train[names].to_numpy().mean(axis=0),rtol=1e-10,atol=1e-20)
                scores,decisions=predict_features(bundle,test)
                group=predictions[(predictions.seed==seed)&(predictions.patient==patient)&(predictions.model==name)].reset_index(drop=True)
                pd.testing.assert_frame_equal(group[C.META],test[C.META].reset_index(drop=True),check_dtype=False)
                np.testing.assert_allclose(scores,group.probability_seizure,rtol=1e-6,atol=1e-7)
                np.testing.assert_array_equal(decisions,group.prediction)
                actual=metrics(test.label,scores,.5)
                row=rows[(rows.seed==seed)&(rows.test_patient==patient)&(rows.model==name)].iloc[0]
                for key,value in actual.items():
                    np.testing.assert_allclose(value,row[key],rtol=1e-12,atol=1e-15)
    macro,pooled=summarize(predictions,rows)
    for actual,file in ((macro,"mean_patient_metrics.csv"),(pooled,"pooled_window_metrics.csv"),(effects(rows),"paired_effects.csv")):
        pd.testing.assert_frame_equal(actual,read_table(run/file),check_exact=False,rtol=1e-12,atol=1e-15)
    print(json.dumps(dict(run=str(run),validated=True,seeds=manifest["seeds"],seconds=manifest["seconds"],protected_files=manifest["protected_files_unchanged"],artifact_hashes=len(manifest["artifacts"]),bundles=20*len(manifest["seeds"]))))
    print(macro[["model","seed","f1_mean","f1_std"]].to_string(index=False))
    print(pooled[["model","seed","precision","recall","f1","average_precision","roc_auc"]].to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    validate(parser.parse_args().run)
