"""Isolated matched selection/KD contribution experiment; canonical code is read-only."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from eeg_seizure import config as C
from eeg_seizure.artifacts import atomic_csv, atomic_json, digest, fingerprint, environment
from eeg_seizure.dataset import load_dataset, read_table
from eeg_seizure.evaluation import patient_split, metrics, summarize
from eeg_seizure.modeling import make_model, select_features, FeatureTransform, train_student, student_scores, student_network, class_ratio
from eeg_seizure.inference import save_bundle, load_bundle, predict_features

ARMS=("teacher","selected_supervised","selected_kd","random_supervised")
COMPARISONS={"selection_effect":("selected_supervised","random_supervised"),
             "kd_effect":("selected_kd","selected_supervised"),
             "teacher_minus_kd":("teacher","selected_kd")}
METRICS=["accuracy","precision","recall","f1","average_precision","roc_auc"]


def random_features(train,columns,held_out,base_seed=42):
    patients=sorted(train.patient.unique())
    if held_out in patients or set(patients)!=set(C.PATIENTS)-{held_out}:
        raise ValueError("Random selection requires exactly outer-training patients")
    if len(columns)!=330 or len(set(columns))!=330 or not set(columns).issubset(train.columns):
        raise ValueError("Expected canonical 330-feature schema")
    seed=int(fingerprint(["random-50-v1",base_seed,patients])[:8],16)
    # Preserve schema order after uniform sampling without replacement.
    indices=np.sort(np.random.default_rng(seed).choice(len(columns),C.TOP_K,replace=False))
    return [columns[i] for i in indices],seed


def initial_hash(seed):
    torch.manual_seed(seed)
    model=student_network(C.TOP_K)
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def student_features(selected,random_names):
    if len(selected)!=50 or len(set(selected))!=50 or len(random_names)!=50 or len(set(random_names))!=50:
        raise ValueError("Each compact student needs exactly 50 unique inputs")
    return {"selected_supervised":list(selected),"selected_kd":list(selected),"random_supervised":list(random_names)}


def effects(rows):
    outputs=[]
    for name,(a,b) in COMPARISONS.items():
        left=rows[rows.model==a].set_index(["seed","test_patient"])
        right=rows[rows.model==b].set_index(["seed","test_patient"])
        if not left.index.equals(right.index):
            raise ValueError("Matched arms do not have identical patient/seed keys")
        outputs.append((left[METRICS]-right[METRICS]).reset_index().assign(comparison=name))
    return pd.concat(outputs,ignore_index=True)


def protect():
    roots=[C.ROOT/p for p in ("src","notebooks","data/processed/corrected_v2","results/corrected","results/analysis","results/feasibility","results/ablations")]
    paths=[p for root in roots for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    paths += [C.ROOT/p for p in ("research/psd_ablation.py","research/validate_psd_ablation.py","scripts/window_feasibility.py","PSD_ABLATION_REPORT.md","WINDOW_FEASIBILITY_10S.md")]
    return {str(p.relative_to(C.ROOT)):digest(p) for p in paths}


def verify_complete(run):
    m=json.loads((run/"manifest.json").read_text())
    if m["status"]!="complete":
        raise ValueError("Incomplete parent run")
    for path,expected in m["artifacts"].items():
        if digest(run/path)!=expected:
            raise ValueError(f"Artifact checksum mismatch: {path}")
    return m


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--seeds",nargs="+",type=int,default=[42])
    parser.add_argument("--primary-run",type=Path)
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in (".",".."):
        raise ValueError("One run directory name required")
    if (args.primary_run is None and args.seeds!=[42]) or (args.primary_run is not None and args.seeds!=[43,44]):
        raise ValueError("Predefined design: primary seed 42, optional replication seeds 43 and 44")
    run=C.ROOT/"results/contribution"/args.run_id
    run.mkdir(parents=True,exist_ok=False)
    protected=protect()
    frame,dm=load_dataset()
    columns=dm["spec"]["feature_columns"]
    parent=verify_complete(args.primary_run) if args.primary_run else None
    if parent and (parent["dataset_sha256"]!=dm["feature_sha256"] or parent["source_sha256"]!=digest(__file__) or parent["seeds"]!=[42]):
        raise ValueError("Parent experiment contract mismatch")
    manifest=dict(status="running",version="kd-contribution-v1",arms=ARMS,seeds=args.seeds,
        predefined_seeds=[42,43,44],teacher_seed=42,selection_seed_base=42,
        seed_scope="Student initialization and loader vary; teacher, selected features and random subset fixed per fold from primary",
        protocol="five-patient outer LOPO; fixed canonical threshold 0.5; no tuning or early stopping",
        student_configuration=C.STUDENT,teacher_configuration=C.XGB_PARAMS,threshold=C.DEFAULT_THRESHOLD,
        source_sha256=digest(__file__),dataset_sha256=dm["feature_sha256"],dataset_spec=dm["spec"],
        environment=environment(),protected_before=protected,
        primary_run=str(args.primary_run.resolve()) if args.primary_run else None,
        primary_manifest_sha256=digest(args.primary_run/"manifest.json") if args.primary_run else None,
        optional_random_kd="not run; no selection-by-KD interaction claim")
    atomic_json(run/"manifest.json",manifest)
    all_predictions,all_rows=[],[]
    start_all=time.perf_counter()
    try:
        for patient in C.PATIENTS:
            train,test=patient_split(frame,patient)
            common=dict(format_version=1,feature_columns=columns,train_patients=sorted(train.patient.unique()),
                test_patients=[patient],dataset_spec=dm["spec"],dataset_sha256=dm["feature_sha256"],threshold=C.DEFAULT_THRESHOLD)
            if args.primary_run:
                teacher_bundle=load_bundle(args.primary_run/f"seed42/{patient}/teacher.joblib")
                teacher=teacher_bundle["estimator"]
            else:
                teacher=make_model("xgboost",train.label,42)
                teacher.fit(train[columns],train.label)
            ranking=select_features(teacher,columns,k=330)
            selected=ranking.head(50).feature.tolist()
            random_names,selection_seed=random_features(train,columns,patient)
            selections=student_features(selected,random_names)
            transforms={key:FeatureTransform().fit(train[names]) for key,names in (("selected",selected),("random",random_names))}
            transformed={key:tf.transform(train[selected if key=="selected" else random_names]) for key,tf in transforms.items()}
            targets=teacher.predict_proba(train[columns])[:,1]
            teacher_scores=teacher.predict_proba(test[columns])[:,1]
            for seed in args.seeds:
                folder=run/f"seed{seed}"/patient
                folder.mkdir(parents=True)
                initial=initial_hash(seed)
                audit=dict(patient=patient,seed=seed,teacher_seed=42,random_selection_seed=selection_seed,
                    train_patients=sorted(train.patient.unique()),test_patients=[patient],training_windows=len(train),
                    training_positive=int(train.label.sum()),class_ratio=class_ratio(train.label),
                    selected_features=selected,random_features=random_names,threshold=C.DEFAULT_THRESHOLD,
                    initialization_sha256=initial,loader_seed=seed,student_configuration=C.STUDENT,
                    train_window_sha256=fingerprint(train[C.META].to_dict("records")),test_window_sha256=fingerprint(test[C.META].to_dict("records")),
                    teacher_target_sha256=hashlib.sha256(targets.tobytes()).hexdigest(),teacher_targets="in-sample outer-training probabilities")
                atomic_json(folder/"fold.json",audit)
                atomic_csv(folder/"teacher_importance.csv",ranking)
                atomic_csv(folder/"feature_selection.csv",pd.DataFrame([dict(model=name,rank=i+1,feature=f) for name,names in selections.items() for i,f in enumerate(names)]))
                bundles={"teacher":dict(common,kind="classical",estimator=teacher,model_name="teacher",seed=42)}
                logs=[]
                scores_by_arm={"teacher":teacher_scores}
                for name in ARMS[1:]:
                    key="random" if name=="random_supervised" else "selected"
                    before=time.perf_counter()
                    model,loss=train_student(transformed[key],train.label.to_numpy(),seed,
                        targets if name=="selected_kd" else None,epochs=C.STUDENT["epochs"])
                    bundle=dict(common,kind="student",selected_features=selections[name],transform=transforms[key],
                        state_dict=model.state_dict(),architecture=[50,*C.STUDENT["hidden"],1],model_name=name,seed=seed,
                        training_configuration=C.STUDENT,initialization_sha256=initial,loader_seed=seed)
                    bundles[name]=bundle
                    scores_by_arm[name]=student_scores(model,transforms[key].transform(test[selections[name]]))
                    logs.append(dict(model=name,seconds=time.perf_counter()-before,loss=loss,
                        parameters=sum(p.numel() for p in model.parameters()),initialization_sha256=initial,loader_seed=seed))
                fold_predictions=[]
                for name,bundle in bundles.items():
                    scores=scores_by_arm[name]
                    save_bundle(folder/f"{name}.joblib",bundle)
                    reloaded,decisions=predict_features(load_bundle(folder/f"{name}.joblib"),test)
                    np.testing.assert_array_equal(reloaded,scores)
                    np.testing.assert_array_equal(decisions,scores>=C.DEFAULT_THRESHOLD)
                    row=dict(metrics(test.label,scores,C.DEFAULT_THRESHOLD),model=name,seed=seed,test_patient=patient,threshold=C.DEFAULT_THRESHOLD)
                    all_rows.append(row)
                    fold_predictions.append(test[C.META].assign(model=name,seed=seed,threshold=C.DEFAULT_THRESHOLD,
                        probability_seizure=scores,prediction=decisions))
                fold_predictions=pd.concat(fold_predictions,ignore_index=True)
                all_predictions.append(fold_predictions)
                atomic_csv(folder/"predictions.csv",fold_predictions)
                atomic_json(folder/"training_log.json",logs)
                atomic_json(folder/"complete.json",dict(status="complete",artifacts={p.name:digest(p) for p in folder.iterdir() if p.is_file()}))
                print(f"seed {seed}/{patient}: "+", ".join(f"{r['model']} F1={r['f1']:.4f}" for r in all_rows[-4:]),flush=True)
        predictions=pd.concat(all_predictions,ignore_index=True)
        rows=pd.DataFrame(all_rows)
        macro,pooled=summarize(predictions,rows)
        atomic_csv(run/"predictions.csv",predictions)
        atomic_csv(run/"per_patient_metrics.csv",rows)
        atomic_csv(run/"mean_patient_metrics.csv",macro)
        atomic_csv(run/"pooled_window_metrics.csv",pooled)
        atomic_csv(run/"paired_effects.csv",effects(rows))
        if protected!=protect():
            raise AssertionError("Protected artifact changed")
        manifest.update(status="complete",seconds=time.perf_counter()-start_all,protected_files_unchanged=len(protected),
            bundle_parity=True,artifacts={str(p.relative_to(run)):digest(p) for p in run.rglob("*") if p.is_file() and p.name!="manifest.json"})
    except Exception as exc:
        manifest.update(status="failed",error=str(exc)); raise
    finally:
        atomic_json(run/"manifest.json",manifest)
    print(run,flush=True)


if __name__=="__main__":
    main()
