"""
Validate the Day 3 feature table before moving into modeling: cross-checks
against Day 2's window index, checks for NaN/Inf and zero-variance columns,
and extends the seizure-vs-non-seizure sanity check to the full dataset
across all four feature families.

Ran from the repo root:
    python scripts/validate_feature_table.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

feature_table = pd.read_csv(PROCESSED_DIR / "feature_table.csv")
window_index = pd.read_csv(PROCESSED_DIR / "window_index.csv")

metadata_cols = ["patient", "file", "start_sec", "end_sec", "label"]
feature_cols = [c for c in feature_table.columns if c not in metadata_cols]

print(f"Feature table: {feature_table.shape[0]} rows, {len(feature_cols)} feature columns\n")

# 1. Row count cross-check against window_index
assert len(feature_table) == len(window_index), "Row count mismatch vs window_index!"
print("[OK] Row count matches window_index.csv")

# 2. Label balance cross-check
ft_label_counts = feature_table["label"].value_counts().to_dict()
wi_label_counts = window_index["label"].value_counts().to_dict()
assert ft_label_counts == wi_label_counts, "Label counts mismatch vs window_index!"
print(f"[OK] Label balance matches window_index.csv: {ft_label_counts}")

# 3. NaN / Inf check
nan_counts = feature_table[feature_cols].isna().sum()
inf_counts = np.isinf(feature_table[feature_cols].select_dtypes(include=[np.number])).sum()

cols_with_nan = nan_counts[nan_counts > 0]
cols_with_inf = inf_counts[inf_counts > 0]

if len(cols_with_nan) == 0 and len(cols_with_inf) == 0:
    print("[OK] No NaN or Inf values found in any feature column")
else:
    if len(cols_with_nan) > 0:
        print(f"[WARN] {len(cols_with_nan)} columns contain NaN values:")
        print(cols_with_nan.sort_values(ascending=False).head(10))
    if len(cols_with_inf) > 0:
        print(f"[WARN] {len(cols_with_inf)} columns contain Inf values:")
        print(cols_with_inf.sort_values(ascending=False).head(10))

# 4. Zero-variance columns (harmless, but no predictive value)
stds = feature_table[feature_cols].std()
zero_var_cols = stds[stds == 0].index.tolist()
if zero_var_cols:
    preview = zero_var_cols[:10]
    suffix = "..." if len(zero_var_cols) > 10 else ""
    print(f"[WARN] {len(zero_var_cols)} zero-variance columns: {preview}{suffix}")
else:
    print("[OK] No zero-variance feature columns")

# 5. Per-patient counts, cross-checked against what was confirmed manually
print("\nPer-patient window counts:")
print(feature_table.groupby("patient").size())

print("\nPer-patient seizure window counts:")
print(feature_table[feature_table["label"] == 1].groupby("patient").size())
print("(expected: chb01=112, chb02=44, chb03=103, chb05=140, chb08=231)")

# 6. Sanity check across ALL feature families, on the full dataset
print("\nSanity check — seizure vs non-seizure means (full 39,480 rows):")
sample_features = [
    "time__line_length__FP1-F7",
    "freq__delta__FP1-F7",
    "entropy__perm__FP1-F7",
    "hjorth__mobility__FP1-F7",
]
for col in sample_features:
    if col not in feature_table.columns:
        print(f"  {col}: not found, skipping")
        continue
    seizure_mean = feature_table.loc[feature_table["label"] == 1, col].mean()
    nonseizure_mean = feature_table.loc[feature_table["label"] == 0, col].mean()
    print(f"  {col}: seizure={seizure_mean:.6f}, non-seizure={nonseizure_mean:.6f}")

# 7. Overall value range check
numeric_cols = feature_table[feature_cols].select_dtypes(include=[np.number])
print("\nFeature value ranges (across all columns):")
print(f"  min: {numeric_cols.min().min():.6f}")
print(f"  max: {numeric_cols.max().max():.6f}")

print("\nValidation complete.")