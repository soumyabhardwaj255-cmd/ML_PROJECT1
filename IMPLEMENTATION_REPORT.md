# Corrective implementation and validation

Completed in `ML_PROJECT1 - Copy`. The separate original project was not accessed. Raw EDF files were not modified.

## Changes

- Added shared configuration, validated dataset/checkpoint handling, model factories, training-only transforms, evaluation, saved-bundle inference, reporting and CLI under `src/eeg_seizure/`; `scripts/fyp.py` is the entry point.
- Fixed final-window generation using sample counts. Preserved legacy tables and migrated matching rows into `data/processed/corrected_v2` after checking metadata, finite features and freshly extracted per-recording samples.
- Fixed small-magnitude scaling with fitted float64 imputation/StandardScaler, reused for inference. Preserved teacher top-50 selection and the 50→32→16→1, 2,177-parameter students. Baseline/KD use paired initialization and batch-order seeds.
- Nested threshold selection now computes weights from each inner training fold. Canonical estimators, feature ordering, preprocessing and thresholds are shared with evaluation and saved-model demos.
- Checkpoints validate recording/annotation hashes, complete window identities, feature schema, channels, configuration, source versions and numerical dependencies. Incomplete builds cannot load as complete. Resumes preserve numerical values and final table bytes.
- Numbered pipeline scripts delegate to the shared CLI. Preserved 25 old implementations and both original notebook snapshots under `research/`; 79 inherited result files remain identifiable and unchanged. Notebook paths/output names are corrected, new exploratory runs use unique directories, and undefined/superseded cells are excluded explicitly. SR/CR/PSD/noise methods were not redesigned.
- Updated downloader manifest/paths/failure handling, pinned requirements plus environment lock, and rewrote the README around the actual adaptation/extension of the specified Ghosh et al. base paper.

## Validated

- **19 tests passed** (see `results/corrected/implementation-validation/pytest.xml`): low-magnitude scaling, train-only transforms, annotation/window boundaries, malformed/partial/config-mismatched checkpoints, matching/nonmatching legacy migration, empty-tail reuse, byte-stable resume, inner-fold weights, patient separation, deterministic baseline/KD, teacher selection, saved architectures, classical bundles, feature ordering and recording-separated events.
- All 45 recording/checkpoint/source-hash checks passed. All 39,480 retained rows preserve their labels and feature values numerically. Freshly re-extracted migration samples matched for every recording. This is per-recording sample validation, not complete fresh recomputation of all retained features.
- A real resume reused all 45 checkpoints without extraction and produced a byte-identical feature CSV.
- Actual training-only scaling passed in all five outer folds. The real chb08 teacher selected 50 features; 22 had standard deviation below the old cutoff and now scale correctly. Stored medians/means match only the 33,222 training rows.
- Real chb08 validation: 300-tree teacher plus two-epoch baseline/KD; 6,300 held-out windows. A repeated seeded run produced byte-identical predictions. Nested logistic regression completed with four independently verified inner-training ratios. All models passed pre/post-serialization parity.
- KD table and raw-EDF demos, plus logistic-regression table demo, matched evaluation decisions for 900 windows each. CSV probabilities matched within the declared numerical tolerance. Plots generated successfully; representative figures were visually checked.
- Important runtime imports, `pip check`, and requirements dry-run passed. Python code syntax and notebook bootstrap from root/notebook working directories passed. The downloader manifest matches the 45 local EDFs and five summaries; live downloads were not rerun.

## Corrected dataset

| Item | Before | Corrected | Change |
|---|---:|---:|---:|
| Windows | 39,480 | 39,522 | +42 |
| Seizure windows | 630 | 630 | 0 |
| Non-seizure windows | 38,850 | 38,892 | +42 |

42 of 45 recordings gained one final window. Patient totals: chb01 9,581; chb02 5,639; chb03 9,900; chb05 8,102; chb08 6,300. Per-recording changes and validation receipts are in `results/corrected/implementation-validation/`.

## Limits and pending work

The completed experiments are validation-only, not final five-patient performance estimates. Pending: corrected full classical/nested LOPO; full 40-epoch teacher/baseline/KD LOPO; paired multi-seed uncertainty analysis; end-to-end size/runtime/memory benchmarking and event-level metrics. No new large multi-seed or full LOPO training was launched.

Five selected patients, previously explored data, full-recording offline filtering, in-sample teacher targets and importance-based interpretation remain methodological limits. No KD superiority, real-time readiness, clinical validity or exact base-paper reproduction is established. Historical metrics must be regenerated before reuse as corrected evidence.

The existing environment was validated; a clean environment reinstall was not performed. Full restart-and-run-all execution and scientific validation of the exploratory SR/CR/PSD/noise notebooks remain pending.

Recommended next deliberate experiment (from repository root):

```powershell
.venv/Scripts/python.exe -B scripts/fyp.py evaluate --run-id kd-lopo-seed42-v2 --kind kd
```
