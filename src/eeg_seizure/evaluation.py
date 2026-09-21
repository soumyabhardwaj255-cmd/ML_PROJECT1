"""Patient-exclusive evaluation; thresholds selected using inner patients only."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
from . import config as C
from .artifacts import atomic_csv, atomic_json, digest, environment
from .dataset import load_dataset
from .modeling import (make_model, class_ratio, FeatureTransform, select_features,
                       train_student, student_scores)
from .inference import save_bundle, load_bundle, predict_features


def metrics(y, scores, threshold=0.5):
    pred = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
                roc_auc=roc_auc_score(y, scores) if len(np.unique(y)) == 2 else float("nan"),
                average_precision=average_precision_score(y, scores) if np.sum(y) else float("nan"),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), windows=len(y))


def patient_split(frame, test_patient):
    train = frame.loc[frame.patient != test_patient].copy()
    test = frame.loc[frame.patient == test_patient].copy()
    if train.empty or test.empty or set(train.patient) & set(test.patient):
        raise ValueError("Invalid patient-exclusive split")
    return train, test


def tune_threshold(train, columns, name, seed):
    """The caller passes only outer-training rows. Refit all learning per inner fold."""
    scores = pd.Series(index=train.index, dtype=float)
    audit = []
    for validation_patient in sorted(train.patient.unique()):
        inner, validation = patient_split(train, validation_patient)
        model = make_model(name, inner.label, seed)
        model.fit(inner[columns], inner.label)
        scores.loc[validation.index] = model.predict_proba(validation[columns])[:, 1]
        audit.append(dict(validation_patient=validation_patient,
                          train_patients=sorted(inner.patient.unique()),
                          training_windows=len(inner), training_positive=int(inner.label.sum()),
                          class_ratio=class_ratio(inner.label)))
    if scores.isna().any():
        raise ValueError("Incomplete inner-validation predictions")
    values = [f1_score(train.label, scores >= t, zero_division=0) for t in C.THRESHOLDS]
    threshold = C.THRESHOLDS[int(np.argmax(values))]
    oof = train[C.META].copy()
    oof["probability_seizure"] = scores
    return threshold, audit, oof


def summarize(predictions, per_patient):
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
    means = per_patient.groupby(["model", "seed"])[metric_names].agg(["mean", "std"])
    means.columns = [f"{m}_{aggregation}" for m, aggregation in means.columns]
    pooled = []
    for (model, seed), g in predictions.groupby(["model", "seed"]):
        # Fold-specific thresholds have already been applied in prediction.
        row = metrics(g.label, g.probability_seizure)
        for key in ("precision", "recall", "f1", "accuracy"):
            function = dict(precision=precision_score, recall=recall_score, f1=f1_score,
                            accuracy=accuracy_score)[key]
            row[key] = function(g.label, g.prediction, **({} if key == "accuracy" else {"zero_division": 0}))
        tn, fp, fn, tp = confusion_matrix(g.label, g.prediction, labels=[0, 1]).ravel()
        row.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), model=model, seed=seed)
        pooled.append(row)
    return means.reset_index(), pd.DataFrame(pooled)


def evaluate(run_id, kind="kd", test_patients=None, seeds=(C.SEED,), epochs=None,
             models=("logistic_regression", "random_forest", "xgboost"), threshold_tuning=False):
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("Run ID must be one directory name")
    frame, dataset_manifest = load_dataset()
    columns = dataset_manifest["spec"]["feature_columns"]
    patients = list(test_patients or C.PATIENTS)
    if not set(patients).issubset(C.PATIENTS) or len(set(patients)) != len(patients):
        raise ValueError("Invalid test-patient list")
    run = C.RESULTS_DIR / run_id
    run.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", kind=kind, test_patients=patients, seeds=list(seeds),
                    epochs=epochs if epochs is not None else C.STUDENT["epochs"],
                    models=list(models), threshold_tuning=threshold_tuning,
                    dataset_sha256=dataset_manifest["feature_sha256"], environment=environment(),
                    configuration=dict(preprocessing=C.PREPROCESS, student=C.STUDENT,
                                       xgboost=C.XGB_PARAMS, random_forest=C.RF_PARAMS, logistic_regression=C.LR_PARAMS,
                                       thresholds=C.THRESHOLDS, top_k=C.TOP_K),
                    source_hashes={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
                    protocol="outer LOPO; no test-patient fitting or model selection",
                    validation_only=set(patients) != set(C.PATIENTS) or
                    (kind == "kd" and epochs is not None and epochs != C.STUDENT["epochs"]))
    atomic_json(run / "manifest.json", manifest)
    predictions, rows, selections, training_log, importances = [], [], [], [], []
    try:
        for seed in seeds:
            for patient in patients:
                print(f"{kind}: seed={seed}, held out={patient}", flush=True)
                train, test = patient_split(frame, patient)
                common = dict(format_version=1, feature_columns=columns,
                              train_patients=sorted(train.patient.unique()), test_patients=[patient],
                              seed=seed, dataset_spec=dataset_manifest["spec"],
                              dataset_sha256=dataset_manifest["feature_sha256"])
                bundles = {}
                if kind == "ml":
                    for name in models:
                        threshold = C.DEFAULT_THRESHOLD
                        if threshold_tuning:
                            threshold, inner_audit, oof = tune_threshold(train, columns, name, seed)
                            if any(patient in fold["train_patients"] or patient == fold["validation_patient"] for fold in inner_audit):
                                raise AssertionError("Outer patient entered threshold selection")
                            atomic_json(run / f"{name}_{patient}_seed{seed}_inner_folds.json", inner_audit)
                            atomic_csv(run / f"{name}_{patient}_seed{seed}_inner_predictions.csv", oof)
                        model = make_model(name, train.label, seed)
                        model.fit(train[columns], train.label)
                        estimator = model.named_steps["model"]
                        values = getattr(estimator, "feature_importances_", None)
                        if values is None:
                            values = np.abs(estimator.coef_[0])
                        importances.append(pd.DataFrame(dict(feature=columns, importance=values)).assign(
                            model=name, seed=seed, test_patient=patient,
                            definition="absolute standardized coefficient" if name == "logistic_regression" else "tree importance"))
                        bundles[name] = {**common, "kind": "classical", "estimator": model,
                                         "threshold": threshold, "model_name": name}
                else:
                    teacher = make_model("xgboost", train.label, seed)
                    teacher.fit(train[columns], train.label)
                    importances.append(select_features(teacher, columns, k=len(columns)).assign(
                        model="teacher", seed=seed, test_patient=patient, definition="XGBoost gain importance"))
                    selected = select_features(teacher, columns)
                    if len(selected) != C.TOP_K:
                        raise ValueError("Teacher must select exactly 50 features")
                    selected_names = selected.feature.tolist()
                    selections.append(selected.assign(test_patient=patient, seed=seed))
                    transform = FeatureTransform().fit(train[selected_names])
                    z_train = transform.transform(train[selected_names])
                    teacher_targets = teacher.predict_proba(train[columns])[:, 1]
                    bundles["teacher"] = {**common, "kind": "classical", "estimator": teacher,
                                          "threshold": C.DEFAULT_THRESHOLD, "model_name": "teacher"}
                    for name, targets in [("student_baseline", None), ("kd_student", teacher_targets)]:
                        start = time.perf_counter()
                        student, loss = train_student(z_train, train.label.to_numpy(), seed, targets, epochs)
                        bundles[name] = {**common, "kind": "student", "selected_features": selected_names,
                                         "transform": transform, "state_dict": student.state_dict(),
                                         "threshold": C.DEFAULT_THRESHOLD, "model_name": name,
                                         "architecture": [C.TOP_K, *C.STUDENT["hidden"], 1],
                                         "training_configuration": dict(C.STUDENT,
                                             epochs=epochs if epochs is not None else C.STUDENT["epochs"])}
                        training_log.append(dict(model=name, seed=seed, test_patient=patient,
                                                 seconds=time.perf_counter()-start,
                                                 parameters=sum(p.numel() for p in student.parameters()),
                                                 loss=loss))
                for name, bundle in bundles.items():
                    scores, pred = predict_features(bundle, test)
                    path = run / "bundles" / f"{name}_{patient}_seed{seed}.joblib"
                    save_bundle(path, bundle)
                    saved_scores, saved_pred = predict_features(load_bundle(path), test)
                    if not np.array_equal(pred, saved_pred) or not np.allclose(scores, saved_scores, rtol=0, atol=0):
                        raise AssertionError("Saved bundle changed inference")
                    row = metrics(test.label, scores, bundle["threshold"])
                    row.update(model=name, seed=seed, test_patient=patient, threshold=bundle["threshold"])
                    rows.append(row)
                    predictions.append(test[C.META].assign(model=name, seed=seed,
                                                          threshold=bundle["threshold"],
                                                          probability_seizure=scores, prediction=pred))
                    print(f"  {name}: F1={row['f1']:.4f}; saved-bundle parity passed", flush=True)
        predictions = pd.concat(predictions, ignore_index=True)
        per_patient = pd.DataFrame(rows)
        means, pooled = summarize(predictions, per_patient)
        atomic_csv(run / "predictions.csv", predictions)
        atomic_csv(run / "per_patient_metrics.csv", per_patient)
        atomic_csv(run / "mean_patient_metrics.csv", means)
        atomic_csv(run / "pooled_window_metrics.csv", pooled)
        if selections:
            atomic_csv(run / "teacher_selected_features.csv", pd.concat(selections, ignore_index=True))
        if importances:
            atomic_csv(run / "feature_importance.csv", pd.concat(importances, ignore_index=True))
        atomic_json(run / "training_log.json", training_log)
        manifest.update(status="complete", predictions_sha256=digest(run / "predictions.csv"))
        atomic_json(run / "manifest.json", manifest)
        return run
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        atomic_json(run / "manifest.json", manifest)
        raise
