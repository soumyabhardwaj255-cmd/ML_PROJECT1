"""
Figures: aggregated confusion matrix and pooled ROC curve for the FINAL
model (XGBoost, entropy+Hjorth features, patient-normalized, tuned
hyperparameters), evaluated across all 5 leave-one-patient-out folds.

Uses the exact hyperparameters found by RandomizedSearchCV in
experiment_c_tuned.py (hardcoded below) rather than re-running the search,
so this is fast and always reproduces the same figures.

Ran from the repo root, after experiment_C_tuned.py:
     python src/eeg_seizure/plots/generate_final_model_figures.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier

from eeg_seizure.normalization import patient_normalize
from eeg_seizure.splits import patient_independent_split
from eeg_seizure.style import COLOR_INDEPENDENT, COLOR_NONSEIZURE, apply_style, save_fig

PROCESSED_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"
FIGURES_DIR = Path(__file__).resolve().parents[3] / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

apply_style()

PATIENTS = ["chb01", "chb02", "chb03", "chb05", "chb08"]
FEATURE_PREFIXES = ["entropy__", "hjorth__"]

# Exact hyperparameters found by experiment_c_tuned.py's search -- update
# this if you rerun the search and get different values.
BEST_PARAMS = {
    "subsample": 0.6,
    "scale_pos_weight": 60,
    "reg_lambda": 5,
    "reg_alpha": 0,
    "n_estimators": 300,
    "min_child_weight": 5,
    "max_depth": 5,
    "learning_rate": 0.2,
    "colsample_bytree": 1.0,
}

feature_table = pd.read_csv(PROCESSED_DIR / "feature_table.csv")
feature_cols = [c for c in feature_table.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)]
feature_table_norm = patient_normalize(feature_table, feature_cols)

all_y_true, all_y_pred, all_y_proba = [], [], []

for test_patient in PATIENTS:
    X_train, X_test, y_train, y_test = patient_independent_split(feature_table_norm, test_patient)
    X_train, X_test = X_train[feature_cols], X_test[feature_cols]

    model = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1, **BEST_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    all_y_true.extend(y_test.tolist())
    all_y_pred.extend(y_pred.tolist())
    all_y_proba.extend(y_proba.tolist())

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)
all_y_proba = np.array(all_y_proba)

print(f"Pooled predictions across {len(PATIENTS)} LOPO folds: {len(all_y_true)} windows total")

# --- Confusion matrix ---
cm = confusion_matrix(all_y_true, all_y_pred)

fig, ax = plt.subplots(figsize=(6, 5.5))
im = ax.imshow(cm, cmap="Oranges")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Non-seizure", "Seizure"])
ax.set_yticklabels(["Non-seizure", "Seizure"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(
    "Confusion matrix — tuned model, pooled across 5 LOPO folds\n"
    "(39,480 windows tested)"
)

for i in range(2):
    for j in range(2):
        ax.text(
            j, i, f"{cm[i, j]:,}", ha="center", va="center",
            fontsize=13, fontweight="bold",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
        )

save_fig(fig, FIGURES_DIR / "final_model_confusion_matrix.png")

# --- ROC curve ---
fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6.5, 6))
ax.plot(fpr, tpr, color=COLOR_INDEPENDENT, linewidth=2.5, label=f"ROC (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color=COLOR_NONSEIZURE, linewidth=1, linestyle="--", label="Chance")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("Pooled ROC curve -- tuned model, patient-independent (LOPO)")
ax.legend(loc="lower right", frameon=False)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)

save_fig(fig, FIGURES_DIR / "final_model_roc_curve.png")

print(f"\nPooled AUC: {roc_auc:.3f}")