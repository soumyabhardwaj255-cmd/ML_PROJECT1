"""Authoritative settings for the corrected FYP pipeline (offline EEG analysis)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
DATA_DIR = ROOT / "data" / "processed" / "corrected_v2"
RESULTS_DIR = ROOT / "results" / "corrected"
PATIENTS = ("chb01", "chb02", "chb03", "chb05", "chb08")
META = ["patient", "file", "start_sec", "end_sec", "label"]
KEYS = META[:-1]
FEATURE_VERSION = "multidomain-330-v1"
DATA_VERSION = "complete-windows-v2"
PREPROCESS = dict(l_freq=1.0, h_freq=40.0, notch_freq=60.0)
WINDOW_SEC = 4.0
OVERLAP_THRESHOLD = 0.5
SAMPLE_RATE = 256.0
SEED = 42
THRESHOLDS = tuple(round(i / 100, 2) for i in range(5, 96))
DEFAULT_THRESHOLD = 0.5
TOP_K = 50
STUDENT = dict(hidden=(32, 16), epochs=40, batch_size=256,
               learning_rate=0.001, weight_decay=1e-4, temperature=3.0, alpha=0.5)
XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", n_jobs=2)
RF_PARAMS = dict(n_estimators=300, class_weight="balanced_subsample",
                 max_features="sqrt", min_samples_leaf=2, n_jobs=2)
LR_PARAMS = dict(class_weight="balanced", max_iter=2000, solver="liblinear")


def feature_names(channels):
    families = {"time": ("mean", "std", "rms", "ptp", "line_length"),
                "freq": ("delta", "theta", "alpha", "beta", "gamma"),
                "entropy": ("perm", "spectral"),
                "hjorth": ("activity", "mobility", "complexity")}
    return [f"{family}__{measure}__{ch}" for family, measures in families.items()
            for ch in channels for measure in measures]
