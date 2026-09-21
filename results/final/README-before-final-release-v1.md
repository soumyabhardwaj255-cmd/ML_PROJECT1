# Feature-Guided Knowledge Distillation for Interpretable and Lightweight Epileptic Seizure Detection

This FYP studies an XGBoost teacher using 330 engineered EEG features and a small neural student using the teacher's top 50 features. Each outer patient fold compares the teacher, a supervised baseline student, and a knowledge-distilled student. Feature importance supports interpretation of the inputs; it does not establish clinical explainability or causal mechanisms.

## Base paper and scope

The primary base paper is Ghosh, S., Singh, V., & Powar, O. S. (2026), **Seizure detection using EEG on the CHB-MIT dataset via multi-domain feature engineering and classical machine learning**, Discover Applied Sciences, 8, 312. [DOI: 10.1007/s42452-026-08306-9](https://doi.org/10.1007/s42452-026-08306-9).

This project is an **adaptation and extension**, not an exact reproduction. The paper uses one-second windows, 23 channels, 0.5–45 Hz preprocessing, a richer temporal/spectral/wavelet/spatial feature pool, MI followed by sequential feature selection, and classical KNN/SVM/RF evaluation. Here we use a selected five-patient subset, four-second windows, 22 common channels, 330 time/frequency/entropy/Hjorth features, patient-exclusive LOPO, and XGBoost-guided KD. The paper does not supply this KD architecture or the exploratory SR/CR methods. Scores across these protocols are not directly comparable.

## Structure and result status

- `src/eeg_seizure/config.py`: authoritative paths, preprocessing, feature schema, model parameters, seeds and threshold grid.
- `dataset.py`: complete-window generation, annotation checks, feature extraction and validated per-recording checkpoints.
- `modeling.py`, `evaluation.py`, `inference.py`: shared training transforms, nested thresholds, deterministic students and saved-bundle inference.
- `scripts/fyp.py`: supported command-line entry point. Numbered ML/KD scripts now delegate to it and accept the same required run/output arguments.
- `tests/`: targeted correctness and regression tests.
- `data/raw/`: immutable local CHB-MIT EDF files and annotation summaries.
- `data/processed/corrected_v2/`: corrected feature table, window index, channel list, recording checkpoints, checksums and count-change report. Ignored by Git.
- `results/corrected/<run-id>/`: separately named new runs, manifests, metrics, predictions, feature rankings, models and figures. Existing run directories are never overwritten. Bundles are ignored by Git.
- `results/exploratory/`: fresh exploratory notebook outputs.
- `research/`: readable preserved legacy implementations, pre-edit notebook snapshots, historical artifact inventory and status notes.
- Existing `results/tables`, `results/figures`, `data/results`, `data/figures`, and notebook CSV/PNG outputs are **historical**, not corrected final experiments. They retain their original paths and names; see `research/README.md`.

No current run is automatically designated final. A manifest marks shortened or partial-patient runs `validation_only: true`. Full new LOPO results still require scientific review before inclusion in the FYP report.

## Environment

Use Python 3.11, from this copied repository. PowerShell:

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -B scripts/fyp.py environment
.venv/Scripts/python.exe -B -m pytest -q
```

The requirements pin the versions used for validation, including PyTorch, plotting, Jupyter and tabulate. `requirements-lock.txt` records the installed Windows CPU environment, including transitive dependencies; use that file for exact platform-matched recreation. A different Python/platform or numerical dependency version requires re-preparation and new run IDs. The environment command imports the important runtime packages. Validation of this copy uses its existing environment; a fresh network installation is a separate check, not implied by import success.

## Dataset and preprocessing

The subset consists of 45 EDF recordings from chb01/chb02/chb03/chb05/chb08, plus each patient's summary text. `scripts/download_data.py --list` lists the exact subset. On a fresh checkout, run `python scripts/download_data.py` to download missing files. It preserves existing source files and stops on failed downloads. Binary `.seizures` files are not needed because the pipeline validates the text summaries. The old requests for unavailable chb08_01/chb08_06 are excluded.

The canonical pipeline requires 256 Hz EEG and 22 common channels, deduplicates channel names consistently, and applies 1–40 Hz bandpass plus the retained 60 Hz notch using MNE to each entire recording. This is **offline** preprocessing: zero-phase full-recording filtering is unsuitable for a causal real-time latency claim. No global or test-patient amplitude normalization is fitted.

Four-second non-overlapping windows contain 1,024 samples. The end bound uses sample count, including a complete final window. A window is positive when at least half its duration overlaps annotated seizure intervals. Incomplete tails are discarded. The corrected subset has **39,522 windows: 630 seizure and 38,892 non-seizure**, up from 39,480 (42 additional non-seizure windows across 42 recordings).

Per channel: mean/std/RMS/peak-to-peak/line length (110 total), five absolute bandpowers (110), permutation/spectral entropy (44), and Hjorth activity/mobility/complexity (66). Features follow the explicit 330-column schema. Units and algorithms remain those of the inherited feature definitions.

```powershell
# Optional migration of existing features: checks all metadata and finite values,
# re-extracts a background and seizure window per file where available,
# and computes missing windows. Files whose numerical checks fail are recomputed.
.venv/Scripts/python.exe -B scripts/fyp.py prepare --reuse-legacy
.venv/Scripts/python.exe -B scripts/fyp.py validate

# On a fresh checkout without legacy derived data:
.venv/Scripts/python.exe -B scripts/fyp.py prepare
```

Legacy reuse validates per-recording samples, not every old feature against fresh extraction. New checkpoints require raw/summary hashes, full expected window identities, 330 finite features, channels, preprocessing, feature version, relevant source hashes and numerical dependency versions. A CSV alone is never a complete checkpoint. Interrupted builds remain marked incomplete; rerunning prepare resumes valid recordings. CSV reads preserve floating-point values across resumes. Keep one prepare process active at a time.

## Training and evaluation

Classical ML shares one pipeline/factory: median imputation fitted on training rows; StandardScaler for logistic regression; configured RF or XGBoost without scaling. XGBoost class weight is recomputed from each actual training fold. Optional threshold selection uses inner leave-one-patient-out predictions from outer-training patients only, choosing the first maximum-F1 threshold on the configured 0.05–0.95 grid. Outer test rows never fit transforms, models, class weights or thresholds.

The XGBoost teacher has 300 depth-5 trees with learning rate 0.05. Each outer fold ranks its training-fitted teacher importance and selects exactly 50 features. The student uses float64 training-fitted imputation/StandardScaler before float32 tensors. There is no absolute `std < 1e-8` cutoff: small EEG powers are standardized rather than suppressed. Constant features remain finite, and standardization is checked.

Both students have 50→32→16→1 dimensions (2,177 parameters), ReLU hidden layers and a binary logit. The baseline uses weighted BCE. KD mixes weighted hard-label BCE with temperature-softened teacher/student BCE (temperature 3, alpha 0.5, temperature-squared weighting). The teacher supplies in-sample training probabilities. Adam, 40 epochs and batch size 256 are fixed. Baseline and KD reset the same seed, initialization and DataLoader generator within each fold/seed. CPU deterministic algorithms and two threads are used. Multiple seeds are supported but were not launched for implementation validation.

```powershell
# Small real integration runs: these are not final LOPO estimates.
.venv/Scripts/python.exe -B scripts/fyp.py evaluate --run-id validation-kd-chb08-seed42-v2 --kind kd --test-patients chb08 --epochs 2
.venv/Scripts/python.exe -B scripts/fyp.py evaluate --run-id validation-ml-chb08-seed42 --kind ml --test-patients chb08 --models logistic_regression --threshold-tuning

# Pending full experiments; run deliberately with fresh IDs.
.venv/Scripts/python.exe -B scripts/fyp.py evaluate --run-id kd-lopo-seed42-v2 --kind kd
.venv/Scripts/python.exe -B scripts/fyp.py evaluate --run-id ml-nested-lopo-seed42-v2 --kind ml --threshold-tuning
.venv/Scripts/python.exe -B scripts/fyp.py report --run results/corrected/kd-lopo-seed42-v2
```

Each run saves per-patient metrics, mean/patient-SD metrics and pooled-window metrics separately; pooled F1 is not mean patient F1. The report reads the identified run and generates PR curves, confusion matrices, feature rankings and clearly labelled F1 figures. Thresholds are stored per model/fold. Run manifests capture configuration, source hashes, dataset identity and environment. Model serialization is checked against pre-save predictions before a run is completed.

## Saved-model inference/demo

Inference never retrains. Use only trusted local joblib bundles. Bundles retain exact feature order, selected features, fitted imputation/scaling, architecture/weights or estimator, threshold, training patients and extraction specification.

```powershell
.venv/Scripts/python.exe -B scripts/fyp.py infer --bundle results/corrected/validation-kd-chb08-seed42-v2/bundles/kd_student_chb08_seed42.joblib --patient chb08 --recording chb08_21.edf --output results/corrected/demo-kd-table-v2 --compare-evaluation results/corrected/validation-kd-chb08-seed42-v2/predictions.csv

# Same saved model, extracting the raw EDF with the stored preprocessing contract:
.venv/Scripts/python.exe -B scripts/fyp.py infer --bundle results/corrected/validation-kd-chb08-seed42-v2/bundles/kd_student_chb08_seed42.joblib --patient chb08 --recording chb08_21.edf --edf data/raw/chb08/chb08_21.edf --output results/corrected/demo-kd-edf-v2 --compare-evaluation results/corrected/validation-kd-chb08-seed42-v2/predictions.csv
```

Use a new output directory each time. The demo saves window probabilities/decisions, a timeline and contiguous predicted intervals. The comparison flag checks recording/window identities, probabilities within numerical tolerance, and identical decisions against evaluation. Adjacent predictions never merge across recordings. Sustained predictions mean at least two consecutive positive windows; they are descriptive and are **not** validated event sensitivity, false alarms/hour or detection latency.

## Exploratory work and remaining science

The two original notebooks remain exploratory and retain historical outputs. Their path setup points to this repository and new outputs use `results/exploratory/`; original source snapshots are preserved. An undefined historical SR diagnostic and superseded plotting cells are explicitly excluded rather than used as evidence. SR, coherence resonance, compression, PSD and noise algorithms were not redesigned or scientifically validated here. The original compression/reconstruction meaning of CR and later coherence-resonance experiments must be distinguished in a dedicated review. Restart-and-run-all of the long scientific notebooks is not part of the canonical workflow.

The five selected patients/recordings limit generalization and prevalence estimates. Teacher importance is fold-dependent; KD improvement requires corrected full folds, paired multi-seed results and uncertainty analysis. The student parameter count excludes feature extraction and the teacher-training cost; end-to-end runtime/memory/size and clinical event metrics remain unmeasured. Historical experiments that normalize using the held-out patient's distribution or use non-nested tuning must not support leakage-free claims. No exact reproduction of the base paper or clinical deployment claim is made.

## Implementation validation receipts

`IMPLEMENTATION_REPORT.md` summarizes completed checks and deferred experiments. For this migrated copy, rerun `python scripts/validate_implementation.py` to check retained legacy rows, raw/checkpoint hashes and all-fold scaling. After the two documented integration runs and demos, `python scripts/validate_training_contract.py` verifies real fitted transforms, teacher selection, weights, saved decisions and demo receipts without retraining. `results/corrected/implementation-validation/pytest.xml` records the regression suite.

## Separate PSD/SR/CR and robustness analyses

See [ANALYSIS_VALIDATION.md](ANALYSIS_VALIDATION.md) for the bounded analysis CLI, validation results, invalid CR trial handling and minimum next experiments. These modules do not change the 330-feature pipeline, fit models or append SR/CR features. Full KD LOPO remains paused pending protocol review.
