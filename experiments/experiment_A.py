"""
Experiment A: which model performs best?

Trains Logistic Regression, Random Forest, and XGBoost on the SAME
patient-independent split (chb08 held out, matching Day 4's baseline) and
the SAME full feature set (all 330 columns) — the only thing that varies
is the model itself.

Ran from the repo root:
    python experiments/experiment_a_models.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eeg_seizure.evaluate import evaluate_predictions
from eeg_seizure.splits import patient_independent_split
from eeg_seizure.train import get_models

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_PATIENT = "chb08"  # same held-out patient as Day 4's baseline, for a fair comparison

feature_table = pd.read_csv(PROCESSED_DIR / "feature_table.csv")
X_train, X_test, y_train, y_test = patient_independent_split(feature_table, TEST_PATIENT)

models = get_models(y_train)

results = []
for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred, label=name)
    results.append({"model": name, **metrics})

results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
results_df.to_csv(RESULTS_DIR / "experiment_a_model_comparison.csv", index=False)

print("\n=== Summary (sorted by F1) ===")
print(results_df.to_string(index=False))

best = results_df.iloc[0]
print(f"\nBest model: {best['model']} (F1={best['f1']:.3f})")
print(f"Saved to {RESULTS_DIR / 'experiment_a_model_comparison.csv'}")