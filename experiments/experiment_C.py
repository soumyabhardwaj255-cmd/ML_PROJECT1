"""
Experiment C: does the model generalize to new patients, or did it just
memorize them?

Fixes the model (XGBoost, winner of Experiment A) and the feature set
(entropy + Hjorth, winner of Experiment B), and varies ONLY the split
strategy: patient-independent (train/test from different patients) vs.
patient-dependent (random row split, ignoring patient identity, so windows
from the same recording can land in both train and test).

The gap between the two F1 scores shows how much a naive random split
would have overstated real-world performance via patient-specific
memorization.

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

TEST_PATIENT = "chb08"
FEATURE_PREFIXES = ["entropy__", "hjorth__"]  # winning feature set from Experiment B


def select_features(df):
    cols = [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)]
    return df[cols]


feature_table = pd.read_csv(PROCESSED_DIR / "feature_table.csv")
results = []


# Patient-independent: train/test come from different patients (the
# default used throughout this project so far).
X_train, X_test, y_train, y_test = patient_independent_split(feature_table, TEST_PATIENT)
X_train, X_test = select_features(X_train), select_features(X_test)

print(f"=== patient_independent (test={TEST_PATIENT}) ===")
model = make_xgboost(y_train)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
metrics = evaluate_predictions(y_test, y_pred, label="patient_independent")
results.append({"split": "patient_independent", **metrics})

# Patient-dependent: random row split, ignoring patient identity.
X_train, X_test, y_train, y_test = patient_dependent_split(
    feature_table, test_size=0.2, random_state=42
)
X_train, X_test = select_features(X_train), select_features(X_test)

print("\n=== patient_dependent (random 80/20 split) ===")
model = make_xgboost(y_train)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
metrics = evaluate_predictions(y_test, y_pred, label="patient_dependent")
results.append({"split": "patient_dependent", **metrics})

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_DIR / "experiment_c_generalization_gap.csv", index=False)

print("\n=== Summary ===")
print(results_df.to_string(index=False))

f1_independent = results_df.loc[results_df["split"] == "patient_independent", "f1"].values[0]
f1_dependent = results_df.loc[results_df["split"] == "patient_dependent", "f1"].values[0]
gap = f1_dependent - f1_independent

print(f"\nF1 gap (patient_dependent - patient_independent): {gap:+.3f}")
print("A positive gap shows patient-dependent evaluation overstates real-world performance.")
print(f"Saved to {RESULTS_DIR / 'experiment_c_generalization_gap.csv'}")