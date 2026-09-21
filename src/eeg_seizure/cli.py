"""Command-line interface shared by all current scripts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import config as C
from .artifacts import atomic_csv, atomic_json, environment


def main(argv=None):
    parser = argparse.ArgumentParser(description="Feature-guided EEG knowledge distillation")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("environment")
    prep = sub.add_parser("prepare")
    prep.add_argument("--reuse-legacy", action="store_true", help="Validate old per-file samples and reuse matching features")
    sub.add_parser("validate")
    run = sub.add_parser("evaluate")
    run.add_argument("--run-id", required=True)
    run.add_argument("--kind", choices=["ml", "kd"], default="kd")
    run.add_argument("--test-patients", nargs="+", choices=C.PATIENTS)
    run.add_argument("--seeds", nargs="+", type=int, default=[C.SEED])
    run.add_argument("--epochs", type=int)
    run.add_argument("--models", nargs="+", choices=["logistic_regression", "random_forest", "xgboost"],
                     default=["logistic_regression", "random_forest", "xgboost"])
    run.add_argument("--threshold-tuning", action="store_true")
    infer = sub.add_parser("infer")
    infer.add_argument("--bundle", type=Path, required=True)
    infer.add_argument("--patient", required=True)
    infer.add_argument("--recording", required=True, help="EDF filename; must identify one recording")
    infer.add_argument("--edf", type=Path, help="Extract features from this EDF instead of the validated table")
    infer.add_argument("--output", type=Path, required=True, help="New output directory")
    infer.add_argument("--compare-evaluation", type=Path)
    plot = sub.add_parser("report")
    plot.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "environment":
        # Actually import the runtime dependencies, not just inspect metadata.
        import mne, antropy, sklearn, xgboost, torch, scipy, matplotlib, joblib, tabulate
        print(json.dumps(environment(), indent=2))
    elif args.command == "prepare":
        from .dataset import prepare
        manifest = prepare(args.reuse_legacy)
        print(json.dumps({key: manifest[key] for key in ("status", "windows", "seizure_windows",
                         "previous_windows", "previous_seizure_windows")}, indent=2))
    elif args.command == "validate":
        from .dataset import load_dataset
        data, manifest = load_dataset()
        print(data.groupby("patient").label.agg(["size", "sum"]).to_string())
        print(f"Validated: {len(data)} windows, {len(manifest['spec']['feature_columns'])} features")
    elif args.command == "evaluate":
        from .evaluation import evaluate
        if args.epochs is not None and args.epochs < 1:
            parser.error("epochs must be positive")
        if args.kind == "kd" and args.threshold_tuning:
            parser.error("KD uses the fixed 0.5 comparison threshold; tuning is for classical ML")
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("seeds must be unique")
        print(evaluate(args.run_id, args.kind, args.test_patients, args.seeds, args.epochs,
                       args.models, args.threshold_tuning))
    elif args.command == "report":
        from .reporting import report
        report(args.run)
    else:
        from .inference import load_bundle, predict_features, extract_edf, contiguous_events
        from .dataset import load_dataset
        bundle = load_bundle(args.bundle)
        if args.patient in bundle["train_patients"]:
            parser.error("Demonstration patient must be held out from this bundle's training")
        if args.edf:
            if args.edf.name != args.recording:
                parser.error("EDF name must match --recording")
            data = extract_edf(bundle, args.edf, args.patient)
        else:
            data, manifest = load_dataset()
            if manifest["feature_sha256"] != bundle["dataset_sha256"]:
                raise ValueError("Bundle and feature table originate from different datasets")
            data = data[(data.patient == args.patient) & (data.file == args.recording)].copy()
        if data.empty:
            raise ValueError("No matching recording windows")
        scores, prediction = predict_features(bundle, data)
        output = data[[c for c in C.META if c in data]].assign(probability_seizure=scores, prediction=prediction)
        parity = None
        if args.compare_evaluation:
            expected = pd.read_csv(args.compare_evaluation)
            expected = expected[(expected.patient == args.patient) & (expected.file == args.recording)
                                & (expected.model == bundle["model_name"]) & (expected.seed == bundle["seed"])]
            expected = expected.sort_values(C.KEYS).reset_index(drop=True)
            actual = output.sort_values(C.KEYS).reset_index(drop=True)
            if not expected[C.KEYS].equals(actual[C.KEYS]):
                raise AssertionError("Evaluation and demo window identities differ")
            parity = bool(np.array_equal(expected.prediction, actual.prediction))
            if not parity or not np.allclose(expected.probability_seizure, actual.probability_seizure, rtol=1e-5, atol=1e-7):
                raise AssertionError("Inference differs from corresponding evaluation")
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_csv(args.output / "predictions.csv", output)
        atomic_csv(args.output / "predicted_events.csv", contiguous_events(output))
        from .reporting import timeline
        timeline(args.output, output, bundle["threshold"])
        atomic_json(args.output / "manifest.json", dict(status="complete", bundle=str(args.bundle),
                    model=bundle["model_name"], seed=bundle["seed"], threshold=bundle["threshold"],
                    windows=len(data), source="edf" if args.edf else "feature_table", evaluation_parity=parity,
                    mode="offline; full-recording filtering; events are descriptive, not event-level scores"))
        print(f"Inference complete: {len(data)} windows; evaluation parity={parity}")


if __name__ == "__main__":
    main()
