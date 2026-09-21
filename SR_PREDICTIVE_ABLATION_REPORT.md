# Frozen SR predictive ablation v1

**Result: this matched experiment does not demonstrate added predictive value from the reproducible SR representation.** Patient-mean F1 and pooled F1, precision, recall, AP and ROC-AUC all decreased. The SR average is reproducible and fully integrated as a separate artifact, but it should not replace the canonical 330-feature representation on this evidence. CR remains excluded; no KD retraining followed.

## Scope and exact feature definition

This is a separately versioned **330 versus 332** representation ablation. CR is excluded. Its completed exploratory conclusion and all earlier artifacts remain unchanged. This is not the final teacher/student/KD experiment.

The new column is `sr__mean10_snr_change_vs_zero_noise_db__FP1-F7__ctx10`, followed by the same name plus `__missing`. For each unchanged four-second/1,024-sample target [s,e), use the strict, same-recording trailing context [e−10,e), 2,560 samples at 256 Hz, from FP1-F7 after unchanged canonical preprocessing. No padding, shifting or samples after e are used. Existing full-recording zero-phase filtering remains noncausal; this is an offline evaluation, not a claim of causal deployment.

Use the existing label-independent characteristic component and normalization: the 1–30 Hz context spectral peak, existing ±.5 Hz bandpass with its existing frequency bounds and normalization. Apply the existing bistable equation at D=.5 and D=0, preserving drive coefficient .2, initial state −1, dt=1/256, one integration substep, noise sqrt(2Ddt) and clipping at ±3. For each seed **62–71**, compute the local spectral-ratio difference in dB. Each local ratio is 10 log10(mean PSD within ±.2 Hz of the characteristic frequency / mean PSD outside ±.2 Hz but within ±2 Hz). The feature is the arithmetic mean of those ten dB differences, not a ratio of averaged spectra. Negative finite values are retained. If context is unavailable or any trial is undefined/clipped, output NaN and missing flag1; otherwise flag0.

The implementation reuses the characteristic component and deterministic D=0 trajectory across seeds; parity was checked against ten calls to the frozen original measurement function. No scientific definition changed. Labels enter only target-index verification, coverage reporting and supervised evaluation, never SR calculation.

## Full-corpus coverage and extraction validation

| Target class | Windows | Valid SR | Missing SR |
|---|---:|---:|---:|
| Non-seizure | 38,892 | 38,802 | 90 |
| Seizure | 630 | 630 | 0 |
| All | 39,522 | 39,432 (99.772%) | 90 (0.228%) |

Every missing value comes from an incomplete context at the start of a recording. All complete contexts have ten valid trials. All rows remain present; missing physical values are not replaced with zero. The flag is defined for every window.

All 330 canonical columns and target metadata are preserved exactly, including after CSV serialization/reload. Target windows and labels were independently regenerated from the validated recording annotations for comparison; trailing context bounds were checked for every target. Original versus optimized ten-seed calculation and exact repeatability were checked in all **45 recordings**, and all **90 previously measured candidate contexts** matched. Raw EDF/annotation hashes, source/configuration and output hashes are recorded. Extraction completed in approximately 211 seconds in this environment.

The new feature table is `results/feature_integration/sr-mean10-full-v1/extended_features.csv`; the original canonical table was not overwritten. Its explicit schema contains exactly the original 330 columns followed by the SR value and its missingness indicator. Per-record checkpoints, mapping/validity diagnostics, class/patient coverage and manifest are saved alongside it.

## Matched predictive design

Arms: **A canonical330**, **B sr332**. Both use chb01, chb02, chb03, chb05 and chb08, identical four-second targets, seed42 and the existing logistic-regression model (`balanced` class weighting, `liblinear`, max_iter2000, otherwise existing defaults). No feature selection occurs: all 330 or 332 features enter the respective model. No teacher or neural student is trained.

Five outer patient-exclusive folds hold out one patient. Four inner patient-exclusive folds within each outer training partition generate out-of-fold probabilities; the existing procedure chooses the threshold maximizing pooled inner-window F1 on .05–.95 in .01 steps, using the first maximum. The procedure is identical in both arms, though fitted thresholds can differ. Median imputation and standard scaling are refitted exclusively on each inner/outer training partition. Class weights derive from that training partition through the unchanged model implementation. Outer test patients do not enter fitting or threshold selection.

There are **50 fits total** (two arms × five outer folds × [four inner fits + one outer fit]). Every fold saves its model/transform/schema bundle, threshold, inner predictions, outer predictions and split audit. Patient-mean F1 uses equal patient weights and sample SD across five patients; pooled metrics use all held-out windows, with each fold's selected threshold retained. AP and ROC-AUC use continuous probabilities.

## Results

| Metric | Canonical 330 | SR 332 | SR minus canonical |
|---|---:|---:|---:|
| **Mean patient F1 ± SD** | **.3981 ± .2333** | **.3881 ± .2440** | **−.0100** |
| Pooled precision | .3039 | .2727 | −.0312 |
| Pooled recall/sensitivity | .5349 | .5206 | −.0143 |
| Pooled F1 | .3876 | .3579 | −.0297 |
| Pooled average precision | .4314 | .4032 | −.0282 |
| Pooled ROC-AUC | .8739 | .8679 | −.0060 |

Patient-mean and pooled metrics are deliberately separate. The paired patient F1 differences have mean **−.0100**, sample SD **.0137**, and range **−.0317 to +.0026**.

| Held-out patient | Canonical F1 | SR F1 | F1 difference | Canonical AP | SR AP |
|---|---:|---:|---:|---:|---:|
| chb01 | .7282 | .7308 | +.0026 | .8084 | .8033 |
| chb02 | .1195 | .0878 | −.0317 | .2475 | .2191 |
| chb03 | .2946 | .2946 | .0000 | .4160 | .4302 |
| chb05 | .3262 | .3124 | −.0138 | .5987 | .5681 |
| chb08 | .5219 | .5148 | −.0072 | .5072 | .4946 |

All ten fitted thresholds were **.95**, selected from inner predictions without outer-test access. F1 rose slightly for one patient, was unchanged for one and declined for three. AP rose for chb03 and declined for the other four. This is not consistent evidence of an improvement.

## Model behavior

No selected-feature set exists in this experiment: all columns are used, and no teacher-based feature ranking was performed. In the SR arm the standardized SR coefficient is negative in every fold (−.488 to −.187); its missingness coefficient is also negative (−1.128 to −.120). These fitted associations are not independent evidence of usefulness, causal effects or a resonance mechanism.

At the selected thresholds, **166 predictions changed**: chb01 8, chb02 119, chb03 0, chb05 32, chb08 7. Continuous probabilities changed even in chb03. Pooled true positives fell **337→328**, false negatives rose **293→302**, and false positives rose **772→875**. Both arms were refitted, so these changes reflect the full fitted representation, not an isolated intervention on one coefficient.

## Verification and preserved artifacts

- **69 regression tests passed**, including four new focused tests for frozen-average parity/determinism, any-invalid-trial handling, schema/mask enforcement and training-only transforms. The focused SR subset passed 11 tests.
- Independent artifact validation passed for **39,522 mappings**, both feature schemas, missingness/coverage, all **10 saved bundles**, **40 inner folds**, training imputation/scaling statistics, threshold reconstruction, held-out predictions, patient/pooled aggregation and paired deltas.
- The rerun canonical baseline's predictions and thresholds match the prior matched PSD-ablation canonical LR baseline **exactly**. This is baseline reproducibility, not reuse of historical pre-correction numbers.
- Extraction verified **617 protected files** unchanged; evaluation and independent validation verified **671 protected files**, including the newly completed SR extraction, unchanged. Original 330-feature results, KD contribution, PSD, ten-second and SR/CR studies were not overwritten.
- Models, schemas, coefficients, all predictions and audits are in `results/ablations/sr-mean10-lr-v1/`. Independent validation receipt: `results/ablations/sr-mean10-lr-v1-validation.json`. Test receipts: `results/feature_integration/sr-mean10-tests-v1.xml` and `sr-mean10-regression-tests-v1.xml`.

Runner: `research/sr_predictive_ablation.py` with stages `extract` and `evaluate`; validator: `research/validate_sr_predictive_ablation.py`. New output directories refuse overwrite. Manifests record source/environment identities, exact schemas/settings/seeds and artifact hashes. The saved 332-feature bundles are explicitly separate ablation bundles, not a silent extension of the canonical inference contract.

## Interpretation and limitations

**Does reproducible SR add predictive information?** Under this fixed five-patient nested logistic-regression comparison, **no predictive benefit was demonstrated**; the observed changes were mostly adverse. Reproducibility of a feature does not establish incremental predictive value. This result does not prove the SR measurement contains no information under every model or population.

The study uses five patients, one model and one deterministic training seed, with substantial patient variation and correlated windows. No statistical superiority/significance claim is made. It evaluates the requested **SR plus missingness** representation jointly; no missingness-only arm was run, so their separate contributions cannot be identified. All missing windows are negative recording-start windows, making that distinction relevant. The unchanged threshold search selected its upper boundary in every fold; no post-result threshold expansion or hyperparameter tuning was performed. These are window-level metrics, not event sensitivity, false alarms per hour or clinical deployment evidence. Noncausal canonical preprocessing remains unchanged.

Keep the canonical 330-feature pipeline as the supported reference. Retain the 332-feature SR artifact and this negative matched ablation as reproducible FYP evidence. CR remains exploratory and outside ML inputs. This completes the authorized feature-engineering/representation experiment; **no final teacher/student/KD retraining was started**.
