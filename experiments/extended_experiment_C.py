"""
Experiment C (extended): does the model generalize to new patients, or did
it just memorize them?

Fixes the model (XGBoost, winner of Experiment A) and the feature set
(entropy + Hjorth, winner of Experiment B), and compares two evaluation
strategies, each run repeatedly for a robust mean +/- std rather than a
single lucky/unlucky split:

  - patient_independent: LEAVE-ONE-PATIENT-OUT across all 5 patients (each
    patient held out as the test set once, trained on the other 4).
  - patient_dependent: random 80/20 row split ignoring patient identity,
    repeated across 5 random seeds.

The gap between the two means is the core "avoiding data leakage" finding
-- and reporting it as mean +/- std means it can't be dismissed as "you
just picked a favorable test patient."

Ran from the repo root, after experiment_a_models.py and
experiment_b_features.py:
    python experiments/experiment_c_generalization.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eeg_seizure.evaluate import evaluate_predictions
from eeg_seizure.splits import patient_dependent_split, patient_independent_split
from eeg_seizure.train import make_xgboost

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"

PATIENTS = ["chb01", "chb02", "chb03", "chb05", "chb08"]
FEATURE_PREFIXES = ["entropy__", "hjorth__"]  # winning feature set from Experiment B
DEPENDENT_SPLIT_SEEDS = [0, 1, 2, 3, 4]  # multiple seeds, for the same rigor as LOPO


def select_features(df):
    cols = [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)]
    return df[cols]


feature_table = pd.read_csv(PROCESSED_DIR / "feature_table.csv")

# --- Patient-independent: leave-one-patient-out across all 5 patients ---
lopo_results = []

for test_patient in PATIENTS:
    X_train, X_test, y_train, y_test = patient_independent_split(feature_table, test_patient)
    X_train, X_test = select_features(X_train), select_features(X_test)

    model = make_xgboost(y_train)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred, verbose=False)

    row = {"test_patient": test_patient, "n_seizure_windows": int(y_test.sum()), **metrics}
    lopo_results.append(row)
    print(
        f"held out {test_patient:>6} (n_seizure={row['n_seizure_windows']:>3}): "
        f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} f1={metrics['f1']:.3f}"
    )

lopo_df = pd.DataFrame(lopo_results)
lopo_df.to_csv(RESULTS_DIR / "experiment_c_lopo_per_patient.csv", index=False)

lopo_f1_mean = lopo_df["f1"].mean()
lopo_f1_std = lopo_df["f1"].std()
lopo_precision_mean = lopo_df["precision"].mean()
lopo_recall_mean = lopo_df["recall"].mean()

print(f"\nLeave-one-patient-out: F1 = {lopo_f1_mean:.3f} +/- {lopo_f1_std:.3f} (n=5 folds)")

# --- Patient-dependent: random 80/20 split, repeated across 5 seeds ---
dependent_results = []

for seed in DEPENDENT_SPLIT_SEEDS:
    X_train, X_test, y_train, y_test = patient_dependent_split(
        feature_table, test_size=0.2, random_state=seed
    )
    X_train, X_test = select_features(X_train), select_features(X_test)

    model = make_xgboost(y_train)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred, verbose=False)

    row = {"seed": seed, **metrics}
    dependent_results.append(row)
    print(f"seed={seed}: precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} f1={metrics['f1']:.3f}")

dependent_df = pd.DataFrame(dependent_results)
dependent_df.to_csv(RESULTS_DIR / "experiment_c_patient_dependent_repeats.csv", index=False)

dependent_f1_mean = dependent_df["f1"].mean()
dependent_f1_std = dependent_df["f1"].std()
dependent_precision_mean = dependent_df["precision"].mean()
dependent_recall_mean = dependent_df["recall"].mean()

print(f"\nPatient-dependent (5 seeds): F1 = {dependent_f1_mean:.3f} +/- {dependent_f1_std:.3f}")

# --- Final comparison ---
summary_df = pd.DataFrame(
    [
        {
            "split": "patient_independent (LOPO mean)",
            "precision": lopo_precision_mean,
            "recall": lopo_recall_mean,
            "f1": lopo_f1_mean,
            "f1_std": lopo_f1_std,
        },
        {
            "split": "patient_dependent (5-seed mean)",
            "precision": dependent_precision_mean,
            "recall": dependent_recall_mean,
            "f1": dependent_f1_mean,
            "f1_std": dependent_f1_std,
        },
    ]
)
summary_df.to_csv(RESULTS_DIR / "experiment_c_generalization_gap.csv", index=False)

gap = dependent_f1_mean - lopo_f1_mean

print("\n=== Final summary ===")
print(summary_df.to_string(index=False))
print(f"\nF1 gap (patient_dependent - patient_independent): {gap:+.3f}")
print("A positive gap shows patient-dependent evaluation overstates real-world performance,")
print("and is now backed by 5 folds / 5 seeds on each side rather than a single split.")
print("\nSaved per-fold results to:")
print(f"  {RESULTS_DIR / 'experiment_c_lopo_per_patient.csv'}")
print(f"  {RESULTS_DIR / 'experiment_c_patient_dependent_repeats.csv'}")
print(f"Saved summary to {RESULTS_DIR / 'experiment_c_generalization_gap.csv'}")