# SR/CR feature integration, version 1

## Scope and decision rules

This is feature design/extraction and coverage validation only: no classifier, imputer, scaler, teacher or student was fitted. All 39,522 canonical four-second windows, original labels and 330 feature values are retained. Existing ML/KD, PSD, SR/CR, contribution and ten-second feasibility artifacts remain protected on `merge-eeg-project`.

Separate artifacts live in `results/feature_integration/srcr-context-features-v1/`. There are two explicitly versioned context variants, each with 334 ordered features. The primary proposed variant is **trailing**; centered is a separately labelled offline comparator. These tables do not replace the canonical 330-feature dataset or silently become accepted by its strict loader/bundle contract.

Parameters were fixed before the corpus pass: existing FP1-F7 channel, ten seconds at 256 Hz, D=0.5 (the existing SR default and a previous dynamics-grid value), one integration substep, initial states/clipping/drifts unchanged, and common fixed NumPy seed 42 innovations. There is no noise-intensity, channel, threshold or seed search. The same innovations are supplied to every context; identifiers and labels do not seed or otherwise enter feature computation. This makes each value a reproducible function of its waveform, but availability/stability is conditional on this stochastic realization. Three predefined seeds are also checked on a bounded label-independent diagnostic sample, not used to choose the best seed.

## Mapping four-second targets to ten-second contexts

For canonical target [s,e), e=s+4:

| Variant | Context | Consequence |
|---|---|---|
| Centered | [s−3,e+3) | Centered on the four-second midpoint; explicitly uses three seconds after the target ends. |
| Trailing | [e−10,e) = [s−6,e) | Includes the target and six preceding seconds; no added look-ahead beyond target end in the already-filtered signal. |

Contexts must contain exactly 2,560 real samples from the same EDF. If a complete context is unavailable, both values remain NaN and the missing flags are one. No padding, shortening, shifting, neighbouring-recording borrowing or row dropping occurs. Four-second targets keep their original >=50% overlap label; they are never relabelled from the ten-second context.

**Future information:** centered context is incompatible with claiming a prediction available at e without at least three extra seconds of look-ahead. It may include a later seizure onset after a negative target window. This is temporal look-ahead, even though it does not cross patient folds. The retained canonical full-recording zero-phase preprocessing is itself noncausal; trailing does not make this pipeline online/causal. Additional per-context zero-phase component filtering stays inside the selected context. A causal deployment design is outside this batch.

Neighbouring samples share EEG context. They are computed separately with reset oscillator states, but are not statistically independent. Outer patient-exclusive splits keep all such overlapping contexts within one partition. Random window splits would be inappropriate. No dataset-wide normalization or learned feature-extraction parameters are used.

## Minimum proposed schema

Each variant appends **two measurements and their two missingness indicators**, totaling **330 + 2 + 2 = 334** features. The two measurements alone would be 332, but silently imputing without missingness flags would conceal boundary/dynamics failures. Both masks remain explicit binary ML columns; reasons, event counts, CV diagnostics and context coordinates remain in a separate audit table.

| Appended column | Exact existing definition | Units / expected range | ML handling |
|---|---|---|---|
| `sr__snr_change_vs_zero_noise_db__FP1-F7__ctx10` | Existing bistable response local spectral ratio at D=.5 minus its D=0 ratio. Ratio = 10 log10(mean PSD within ±.2 Hz of target / mean PSD outside ±.2 but within ±2 Hz). Target is the existing label-independent 1–30 Hz context peak; existing ±.5 Hz component filtering/normalization is retained. | dB; any finite real value; negative values are valid. | Training-only median imputation and scaling; reject undefined ratios or clipped dynamics. This is not proof of stochastic resonance or broadband denoising. |
| `cr__fhn_interval_cv__FP1-F7__ctx10` | Existing FHN mean-plus-SD peak detector, 0.2-second separation rule; retained intervals >=.2 s; population SD(intervals)/mean(intervals), requiring at least three intervals and the existing valid result status. | Dimensionless, positive for retained values; no additional numerical cap. | Training-only median imputation and scaling. Insufficient intervals, existing zero-CV/unbounded status or clipped dynamics remain missing. |
| SR column plus `__missing` | 1 exactly when SR value is unavailable; otherwise 0. | Binary {0,1}. | Always defined; never infer it from labels or fit it from cohort statistics. |
| CR column plus `__missing` | 1 exactly when FHN CV is unavailable; otherwise 0. | Binary {0,1}. | Always defined; preserve it alongside imputation. |

Both measurements use the complete mapped ten-second context and are recomputed independently for each target. They inherit the future-information properties of the chosen mapping and preprocessing. Neither uses seizure annotations. Flat/nonfinite input is rejected rather than yielding a noise-only pseudo-measurement. No missing physical value is replaced by zero in the artifact.

**Quantities deliberately excluded from the ML vector:** bistable transition CV/inverse-CV (frequent insufficient intervals in existing studies); inverse FHN CV (redundant with CV and singular at zero); event counts, validity fractions, frequency, clipping rate and trial-grid maxima (retained as diagnostics rather than automatically adding them). FHN event counts can be numerically defined with too few intervals, but that does not make them equivalent to a coherence measure. PSD, noise dose and model identity are not appended again.

## Equation preservation and reproducibility

The new standalone runner imports the existing characteristic-component, PSD-ratio, event-detection and regularity definitions. A Numba evaluator accelerates only the existing one-substep bistable and FHN state updates using precomputed NumPy innovations. It uses float64, no fast-math and the same initial states, drift, dt=1/256, sqrt(2Ddt) noise and clipping. Canonical analysis modules are untouched. Synthetic waveform parity is checked at D=0/.5/2, and real-context parity is checked for the first valid context in each recording/mapping against the original implementation, including exact event counts and reasons.

The schemas explicitly identify `srcr-334-trailing-v1` and `srcr-334-centered-v1`. Manifests record dataset/source/environment identities, configuration, output hashes and protected-file hashes. Each complete extended table retains metadata and all 330 original columns, followed by the same four appended column names; context mode is an indispensable part of the version contract. Do not mix the two tables because their column names match.

Commands (existing output directories cannot be overwritten):

```powershell
.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider
.venv/Scripts/python.exe -B research/srcr_features.py --run-id srcr-context-features-v1
.venv/Scripts/python.exe -B research/validate_srcr_features.py --run-id srcr-context-features-v1
```

The validator writes only a fresh sibling validation directory, including feature ranges, seed sensitivity, a no-drive FHN control and an annotation-alignment audit. That last audit uses annotations only after extraction and is never included among model inputs.

## Measured coverage

Both physical features have identical availability in this run; both missingness indicators are defined for every row.

| Mapping / class | Canonical windows | Both measurements valid | Both missing |
|---|---:|---:|---:|
| Centered / all | 39,522 | 39,433 (99.775%) | 89 (0.225%) |
| Centered / non-seizure | 38,892 | 38,803 | 89 |
| Centered / seizure | 630 | 630 | 0 |
| Trailing / all | 39,522 | 39,432 (99.772%) | 90 (0.228%) |
| Trailing / non-seizure | 38,892 | 38,802 | 90 |
| Trailing / seizure | 630 | 630 | 0 |

Centered failures comprise 45 recording-start and 44 recording-end contexts; trailing failures comprise 90 recording-start contexts. Among complete contexts, neither proposed measurement was insufficient or clipped at the fixed configuration. Missing rates apply individually to SR and FHN CV, not just their intersection. Non-seizure missing rates are 0.229% centered and 0.231% trailing. No rows were discarded.

In contrast, **bistable transition regularity was insufficient in every complete context**: zero detected transitions, zero valid CVs, in both mappings at this fixed D/seed. These invalid trials remain recorded rather than converted to usable regularity values. Earlier multi-dose dynamics availability does not transfer automatically to this fixed configuration.

Centered contexts contain seizure activity after the target ends for 27 negative-labelled windows; trailing has zero such future-context cases. Context seizure activity occurs in 54 centered and 51 trailing negative-labelled windows. These are annotation-audit counts, not feature inputs, and target labels are unchanged. They illustrate why context and target labels must not be conflated.

## Scientific validity and bounded seed diagnostics

SR ranges are −38.601 to 3.507 dB centered and −38.999 to 8.546 dB trailing. FHN CV ranges are 0.706–1.227 centered and 0.701–1.213 trailing. Negative SR differences are retained: the chosen noise does not automatically improve the spectral ratio.

The label-independent first/middle/last complete context from each of 45 recordings gives 135 contexts per mapping, evaluated with seeds 42/43/44 (810 diagnostic rows). All proposed measurements were numerically valid in this sample for every seed.

| Mapping | SR rank correlation, seed 42 vs 43 / 44 | FHN CV rank correlation, seed 42 vs 43 / 44 |
|---|---:|---:|
| Centered | 0.702 / 0.692 | −0.159 / 0.181 |
| Trailing | 0.699 / 0.699 | −0.078 / 0.009 |

Trailing mean FHN CV changes from 0.952 to 0.826 to 0.601 across these seeds. A zero-drive FHN control yields CV 0.961 / 0.793 / 0.585, respectively. Thus the FHN assay is strongly conditional on the noise realization; numerical coverage does **not** establish an EEG-specific coherence-resonance feature. SR has moderate rank stability but also seed-dependent levels. A fixed common seed guarantees computational reproducibility, not scientific robustness. No seed was selected based on these results.

These are SR/CR-model-derived candidate measurements, not demonstrated stochastic/coherence resonance. A single noise intensity cannot establish a resonance curve, and oscillator peak regularity is not inter-channel EEG spectral coherence. Bistable CV remains unsuitable; FHN CV remains exploratory despite excellent availability.

## Completed validation and readiness

- Full test suite: **57 passed**, including seven focused SR/CR tests. Focused checks cover exact boundaries/mapping, deterministic computation, label rejection, identifier independence, original-equation parity, future-sample handling and preservation of insufficient/flat-input outcomes.
- Original versus accelerated implementation parity passed on **90 real contexts**, plus synthetic D=0/.5/2 checks. No clipping occurred in the corpus pass.
- Independent artifact validation passed: **79,044 mapping rows**, exact canonical column preservation in both CSVs, 334-feature schemas, diagnostics/value/mask consistency, coverage reconstruction and EDF-header mapping checks.
- Manifest/output hashes and **510 protected pre-existing file hashes** verified unchanged. The validator confirms observed training-partition values for each physical feature in every outer patient fold. No imputer, scaler, selection procedure or model was fitted.
- Test processes emitted native Windows access-violation diagnostics during imports but continued and exited successfully with passing tests; the cause is not established. This is an environment limitation, not a failed test assertion.

Validation receipts and supplementary tables are in `results/feature_integration/srcr-context-features-v1-validation/`; test receipt is `results/feature_integration/srcr-validation-v1/pytest.xml`.

**Readiness:** the separately versioned tables are structurally ready for a controlled offline ablation after adding an explicit extended-schema experiment/bundle adapter. They are not a drop-in replacement for the protected canonical loader. Training-only imputation/scaling must be fitted separately inside every inner/outer training partition; inference must reproduce context mode, channel, equations, seed, order and missingness. No predictive benefit is established. FHN CV is not yet scientifically stable enough to promote as a validated contribution.

## Exact next experiment (not run)

First perform a bounded, label-independent feature-stability study on the same diagnostic contexts: evaluate a predeclared small ensemble of existing oscillator trials, comparing repeatability of an explicitly documented replicate average and its variation against matched no-drive controls. Keep the current single-trial artifacts unchanged. Do not tune seed, D or aggregation against seizure labels; reserve disjoint contexts for confirming any chosen assay. This addresses the observed CR noise dependence before expensive training. If stability remains poor, retain the CR feature as explicitly exploratory rather than claiming a reliable physiological measure.

Once the feature contract is frozen, the smallest predictive test is a separately versioned single-seed, patient-exclusive classical-model ablation: (1) canonical 330, (2) canonical 330 plus the two missingness flags (332), (3) all 334 trailing features. Use identical patients, target rows, nested evaluation protocol and training-only transforms/threshold handling. The flags-only arm checks whether any apparent improvement comes from recording boundaries rather than SR/CR measurements. Report patient-level and pooled metrics separately. Centered context may be tested only as a labelled offline look-ahead comparator.

Only after that evidence should the extended teacher/student/KD experiment be considered. No new LOPO/KD training, multi-seed model sweep, deployment-cost experiment or interpretability experiment was launched in this batch.
