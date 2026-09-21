# Bounded PSD/SR/CR integration: validation and next experiments

The canonical 330-feature ML/KD pipeline is unchanged. Analysis lives in `src/eeg_seizure/analysis/` and runs through `scripts/analysis.py`. No SR/CR columns are appended to the ML feature table, and no analysis command fits a model. Full KD LOPO and large multi-seed experiments remain paused.

## Implemented analysis roles

| Module | Role and retained definition |
|---|---|
| `spectral.py` | Explicit Welch PSD, local spectral-SNR ratio, and the notebook's FFT white/pink/black noise generator. PSD units are V²/Hz. |
| `sr.py` | Existing bistable update, fixed RNG, clipping diagnostics and optional integration substeps. Characteristic frequency follows one label-independent per-window peak rule. |
| `coherence.py` | Separate bistable hysteresis and FHN peak detectors. Corrected FHN SDE noise placement, with an explicitly named `legacy_extra_dt` convention for equation-parity checks. |
| `compression.py` | Legacy stride/interpolation and a separately named anti-aliased sample-reduction control. This is signal reconstruction, not neural-model compression or a measured byte-compression ratio. |
| `context.py` | Recording/annotation checksums, canonical full-recording preprocessing, exact four-second window metadata and stable seeds. |
| `runner.py` | Separate dynamics analysis and frozen-bundle robustness, with unique run directories and manifest/output checksums. |
| `reporting.py` | Raw grid maxima, invalid-trial counts, valid fractions and unsmoothed plots. No interpolation of missing CR values or smoothing-based optima. |

The optional SR robustness condition maps dimensionless oscillator output back to the original window's mean/SD before using the unchanged feature extractor. This is an explicitly experimental waveform perturbation, not a validated denoiser. It never uses the class label or test-cohort statistics. All current robustness perturbations are **post-filter**; they are not acquisition-noise simulations. Original EDFs and derived canonical tables remain unchanged.

FHN retains the notebook's mean-plus-SD peak threshold, minimum 0.2-second intervals, three-interval minimum and population-SD CV. Bistable regularity uses hysteresis, 0.1-second refractory separation, five intervals and sample-SD CV. Their validity counts and numerical scores must not be treated as interchangeable metrics. Refractory separation is not a continuous dwell-duration test. Zero-CV results are explicitly unbounded, not assigned a fabricated finite coherence.

## Completed validation

- **38 tests passed**, including the existing 19 canonical tests. Added checks cover notebook-equation parity, corrected FHN noise increments, original FHN peak detection, PSD scaling/units, coloured-noise slopes and dose, zero-noise/identity controls, seeds, no input mutation, invalid/all-invalid CR trials, raw optima, aliasing controls and rejection of training patients/incompatible datasets.
- Deterministic zero-noise solutions improved under 1/2/4 integration substeps against a 16-substep reference. This is a drift-integration check, **not** stochastic convergence validation.
- A synthetic 40 Hz signal aliases to 24 Hz under legacy 4:1 stride reduction; the anti-aliased control suppresses that alias. Anti-aliasing does not preserve frequencies above the reduced Nyquist limit (32 Hz at 4:1 reduction from 256 Hz).
- `scripts/validate_analysis.py` verified stored source/output checksums, regenerated summaries and raw maxima from trial rows, checked clean frozen predictions, and verified the raw EDF/summary hashes and all **49 protected canonical source/data/bundle/notebook artifacts**.
- The completed two-window KD robustness run includes seven conditions: clean; white, pink and black noise; both reconstruction variants; and experimental SR. It contains 22 prediction rows. Clean features/decisions match the canonical baseline. White noise and SR each changed one of the two decisions in each of their two trials; this tiny selection supports no general accuracy or benefit claim.
- Representative PSD and dynamics figures were visually inspected. Dynamics plots retain missing values and show validity separately.
- A further two-window frozen logistic-regression run completed all seven conditions (14 prediction rows), validating the classical bundle path and its saved 0.95 threshold. Clean decisions matched exactly; probabilities matched within the existing numerical tolerance. No model was retrained.

## Dynamics result: preserve insufficiency

The final run uses chb08_21 windows at 0–4 seconds (background) and 2140–2144 seconds (seizure), FP1–F7, D = 0/0.1/0.5/2, three trials, driven/no-drive controls and 1/2 substeps.

| Model-specific metric | Total trajectories | Valid | Insufficient |
|---|---:|---:|---:|
| Bistable transition regularity | 96 | 1 | 95 |
| FHN peak-interval regularity | 96 | 64 | 32 |
| Total | 192 | 65 | 127 |

All 127 insufficient trials remain in `dynamics_trials.csv` with their reason, event/interval counts and missing CV/coherence. They contribute to trial totals and validity fractions; finite medians are explicitly conditional on valid trials. No clipping occurred in this bounded D range. Finite FHN results are not evidence of resonance, and the four-second bistable estimates are mostly unavailable.

The earlier `psd-sr-cr-smoke-chb08` artifact reported 191 insufficient trials because the preliminary runner applied the bistable transition metric to FHN too. It is preserved as superseded, not used as the final FHN result. The final run restores FHN's separate original peak-based definition; it does not fill missing measurements. Both histories remain auditable.

## Reproduce the bounded checks

Run from this copied repository. Existing run directories cannot be overwritten; use fresh IDs for reruns.

```powershell
.venv/Scripts/python.exe -B -m pytest -q
.venv/Scripts/python.exe -B scripts/validate_analysis.py
.venv/Scripts/python.exe -B scripts/analysis.py analyse --run-id dynamics-next-check --patient chb08 --recording chb08_21.edf --starts 0 2140 --trials 3 --noise-levels 0 0.1 0.5 2 --substeps 1 2
.venv/Scripts/python.exe -B scripts/analysis.py robustness --run-id robustness-next-check --patient chb08 --recording chb08_21.edf --starts 0 2140 --trials 2 --bundle results/corrected/validation-kd-chb08-seed42-v2/bundles/kd_student_chb08_seed42.joblib --include-sr
```

The runner caps selection at 32 windows, trials at 20 and noise levels at 10. Dynamics requires a zero-noise control. Completed outputs are `results/analysis/psd-sr-cr-validated-chb08`, `results/analysis/frozen-kd-robustness-validated` and `results/analysis/frozen-lr-robustness-validation`. Earlier smoke directories are preliminary. `results/analysis/validation_receipt.json` and `pytest.xml` are validation receipts. The receipt validator expects these named local runs; it verifies existing artifacts rather than creating missing runs.

## Minimum next experiment set

1. **One small paired frozen-model comparison.** Predeclare a modest background/ictal window set from a few recordings of the held-out patient and run the already-saved teacher, baseline and KD bundles on exactly the same windows/noise seeds. Include clean/zero-dose controls and fixed modest noise/compression levels. Report per-window changes and feature distortion, not population performance. Do not choose SR parameters or thresholds using held-out labels. This is inference only, not new multi-seed model training.
2. **Only if SR/CR scientific claims will be retained:** a bounded duration/transient/time-step study on development data, including longer contiguous contexts, no-drive and zero-noise controls, valid-interval rates and clipping. Compare stochastic distributions across time steps, not paths generated by uncoupled noise. Keep four-second ML windows unchanged. Longer-context extraction requires an explicit analysis-only extension; the present CLI intentionally accepts canonical windows only. Do not lower event requirements or impute CR to manufacture usable values. Parameters must be fixed from development data before any held-out evaluation.
3. **Freeze the protocol before final ML/KD training.** Keep the 330 features and frozen inference contract; declare PSD/noise/reconstruction as robustness analyses and SR/CR as exploratory unless the preceding evidence warrants a separately versioned ablation. If no SR/CR predictive or resonance claim is made, completing a resonance study is not a prerequisite for evaluating the unchanged clean ML/KD baseline.

Full 40-epoch LOPO, model-training multi-seed uncertainty estimates, acquisition-noise robustness, deployment-size/runtime benchmarks and broader scientific SR/CR validation have not been run in this batch.
