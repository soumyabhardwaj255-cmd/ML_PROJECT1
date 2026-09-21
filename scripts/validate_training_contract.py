"""Validate the documented chb08 integration artifacts without retraining."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eeg_seizure.dataset import load_dataset
from eeg_seizure.inference import load_bundle, predict_features
from eeg_seizure.modeling import select_features, class_ratio
from eeg_seizure.artifacts import atomic_json, environment, digest


def main():
    results = ROOT / "results/corrected"
    frame, manifest = load_dataset()
    train, test = frame[frame.patient != "chb08"], frame[frame.patient == "chb08"]
    run = results / "validation-kd-chb08-seed42-v2"
    previous = results / "validation-kd-chb08-seed42"
    repeated = None
    if previous.exists():
        repeated = digest(run / "predictions.csv") == digest(previous / "predictions.csv")
        assert repeated
    teacher = load_bundle(run / "bundles/teacher_chb08_seed42.joblib")
    assert teacher["estimator"].named_steps["model"].get_params()["scale_pos_weight"] == class_ratio(train.label)
    selected = select_features(teacher["estimator"], teacher["feature_columns"]).feature.tolist()
    predictions = pd.read_csv(run / "predictions.csv", float_precision="round_trip")
    checks = []
    for name in ("student_baseline", "kd_student"):
        bundle = load_bundle(run / "bundles" / f"{name}_chb08_seed42.joblib")
        assert bundle["training_configuration"]["epochs"] == 2
        assert bundle["architecture"] == [50, 32, 16, 1] and bundle["selected_features"] == selected
        assert set(bundle["train_patients"]) == set(train.patient) and bundle["test_patients"] == ["chb08"]
        transform, x = bundle["transform"], train[selected].to_numpy()
        np.testing.assert_allclose(transform.imputer_.statistics_, np.median(x, axis=0), rtol=1e-12, atol=1e-25)
        np.testing.assert_allclose(transform.scaler_.mean_, x.mean(axis=0), rtol=1e-12, atol=1e-25)
        z = transform.transform(train[selected])
        np.testing.assert_allclose(z[:, transform.scaler_.var_ > 0].std(axis=0), 1, rtol=1e-5, atol=1e-5)
        scores, decisions = predict_features(bundle, test)
        expected = predictions[predictions.model == name]
        # CSV stores float32 probabilities as decimal text; use the demo's tolerance.
        np.testing.assert_allclose(scores, expected.probability_seizure, rtol=1e-5, atol=1e-7)
        np.testing.assert_array_equal(decisions, expected.prediction)
        checks.append(dict(model=name, features=len(selected),
                           parameters=sum(v.numel() for v in bundle["state_dict"].values()),
                           selected_features_below_old_std_cutoff=int(((x.std(axis=0) < 1e-8) & (x.std(axis=0) > 0)).sum()),
                           training_only_imputation_and_scaling_verified=True,
                           saved_decisions_exact=True, csv_probability_tolerance=dict(rtol=1e-5, atol=1e-7), epochs=2))
    ml = results / "validation-ml-chb08-seed42"
    folds = json.loads((ml / "logistic_regression_chb08_seed42_inner_folds.json").read_text())
    for fold in folds:
        inner = train[train.patient != fold["validation_patient"]]
        assert set(inner.patient) == set(fold["train_patients"])
        assert "chb08" not in fold["train_patients"]
        assert class_ratio(inner.label) == fold["class_ratio"]
    demos = []
    for name in ("demo-kd-table-v2", "demo-kd-edf-v2", "demo-ml-table"):
        demo = json.loads((results / name / "manifest.json").read_text())
        assert demo["evaluation_parity"] and demo["status"] == "complete" and demo["windows"] == 900
        demos.append(dict(run=name, windows=demo["windows"], source=demo["source"], evaluation_parity=True))
    receipt = dict(status="passed", environment=environment(), dataset_sha256=manifest["feature_sha256"],
                   real_training_rows=len(train), held_out_windows=len(test), students=checks,
                   teacher_top50_verified=True, teacher_outer_training_weight_verified=True,
                   inner_fold_weights=[f["class_ratio"] for f in folds],
                   seeded_real_repeat_predictions_byte_identical=repeated, demos=demos)
    atomic_json(results / "implementation-validation/training_contract.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
