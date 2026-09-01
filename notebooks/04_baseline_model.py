"""
Patient-independent split infrastructure + a baseline Random Forest,
to confirm the end-to-end pipeline (features -> split -> model -> metrics)
produces a sane result before full model comparison.

Ran from the repo root:
    python notebooks/04_baseline_model.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eeg_seizure.evaluate import evaluate_predictions
from eeg_seizure.splits import patient_independent_split

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

feature_table = pd.read_csv(PROCESSED_DIR / "feature_table.csv")

# Hold out one patient entirely for testing. chb08 has the most seizure
# windows (231), giving the most reliable F1 estimate for this first check.
TEST_PATIENT = "chb08"

X_train, X_test, y_train, y_test = patient_independent_split(feature_table, TEST_PATIENT)

train_patients = sorted(feature_table.loc[feature_table["patient"] != TEST_PATIENT, "patient"].unique())
print(f"Train: {X_train.shape[0]} windows from {train_patients}")
print(f"Test:  {X_test.shape[0]} windows from ['{TEST_PATIENT}']")
print(f"Train seizure windows: {y_train.sum()} ({100 * y_train.mean():.2f}%)")
print(f"Test seizure windows:  {y_test.sum()} ({100 * y_test.mean():.2f}%)")

# class_weight='balanced' is essential given ~1.6% positive class — without
# it, the model would learn to mostly predict "non-seizure" since that
# minimizes error on 98%+ of the training data.
model = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

results = evaluate_predictions(y_test, y_pred, label=f"Baseline RF, held-out patient={TEST_PATIENT}")

results_df = pd.DataFrame([{"experiment": "baseline_rf", "test_patient": TEST_PATIENT, **results}])
results_df.to_csv(RESULTS_DIR / "day4_baseline_results.csv", index=False)
print(f"\nSaved results to {RESULTS_DIR / 'day4_baseline_results.csv'}")