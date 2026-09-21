# 4-second versus 10-second feasibility, version 1

## Decision

Ten seconds is feasible for PSD and gives more oscillator events, but this study does **not** justify replacing the validated four-second seizure-detection pipeline. Keep four seconds canonical and use ten seconds as a distinct, versioned analysis context. Ten seconds does not yet make bistable regularity dependable or establish stochastic/coherence resonance. No SR/CR columns were added to ML inputs, no models were trained, and no branch was created.

## Reproducible scope

Artifacts: `results/feasibility/window-feasibility-4s-10s-v1/`. The manifest records the environment, source/input hashes, explicit selection rule and all 19 output checksums. The new entry point is `scripts/window_feasibility.py`; the read-only verifier is `scripts/validate_window_feasibility.py`. The runner refuses to overwrite an existing run directory.

The census covers all 45 canonical EDFs and five patients. It uses the existing annotation parser/validator, >=50% overlap label rule, 22 canonical channels, 256 Hz and full-recording 1–40 Hz preprocessing with the existing notch filter. An analysis-only duration-parameterized boundary function reproduces canonical sample-count windowing, including the final complete window. Its four-second metadata was compared against both the canonical function and saved window index for every recording. The canonical config/function were not modified or monkey-patched. Partial-overlap windows are retained and labelled using the canonical rule, rather than discarded as in the early notebook.

PSD validity is a full census over every window and all 22 channels: finite, nonnegative density with positive summed power. Welch uses the same Hann/256-sample/128-overlap settings as the existing analysis. This verifies numerical availability, not discrimination. The spectral bin spacing remains 1 Hz at both durations; there are 7 versus 19 Welch segments. No complete 10-second 330-feature table was extracted, and individual relative-power feature discrimination was not tested.

Dynamics is deliberately bounded: six matched contexts on FP1-F7, durations 4/10 seconds, D=0/0.1/0.5/2, three trials, 1/2 integration substeps, driven/no-drive conditions and both models. This is 576 trajectories per duration, 1,152 total. The development contexts are chb01_03 at 0 and 3000 seconds, and chb02_16+ at 0 and 2980 seconds, selected by a predefined first-annotated-recording/common-grid rule. Historical chb08_21 contexts at 0 and 2140 seconds are included only for direct comparison, not parameter tuning. No broader population dynamics validity is inferred.

Existing oscillator equations, initial states, event thresholds and minimum interval requirements are unchanged. Seeds match across duration at fixed identity/model/trial/substeps, giving shared initial innovations. Different time steps use different streams; this is not coupled-path stochastic convergence validation. Longer EEG contexts can change the automatically selected characteristic frequency and normalization, so the driven comparison is not a pure extension of an identical four-second drive. No-drive controls help expose that distinction. Initialization transients remain included. Deterministic D=0 repetitions and paired controls are not independent patient observations.

## Full-corpus census

| Measure | 4 s | 10 s |
|---|---:|---:|
| Complete windows | 39,522 | 15,808 |
| Seizure-labelled | 630 | 250 |
| Non-seizure-labelled | 38,892 | 15,558 |
| All-channel PSD-valid windows | 39,522 | 15,808 |
| PSD-valid channel/windows | 869,484 | 347,776 |
| Mixed seizure/background windows | 37 | 49 |
| Mixed windows labelled negative | 13 | 24 |
| Annotated seizure seconds inside negative windows | 13 | 64 |
| Discarded incomplete tail seconds, total | 6 | 14 |
| Annotated seizure events represented by a positive window | 27/27 | 27/27 |
| Annotated seizure seconds inside positive windows | 2,480/2,493 | 2,429/2,493 |
| Median earliest positive-window end minus seizure onset | 4 s | 10 s |
| Maximum earliest positive-window end minus seizure onset | 5 s | 14 s |

The last two rows are annotation/window geometry, **not measured detector latency**. The existing full-recording filter is offline. Every seizure remains represented, but temporal precision worsens and positive training examples fall from 630 to 250. These counts do not establish unchanged sensitivity or accuracy. See `patient_counts.csv`, `recording_counts.csv` and `window_index_psd.csv` for the complete breakdown.

## SR and CR numerical availability

For SR, all **144/144 driven bistable trials at each duration** have finite output and zero-noise-reference local spectral ratios, with no clipping. There is therefore no demonstrated SR-availability improvement from 4 to 10 seconds. Of the 144 no-drive trials at each duration, 108 have a finite output ratio; the 36 D=0 constant-state controls have an undefined ratio. Consequently all no-drive differences against that undefined zero-noise ratio remain unavailable. These are retained, not filled. A computable local spectral ratio is not evidence of noise-enhanced information recovery or an SR optimum.

CR validity below means sufficient intervals and finite CV/inverse-CV, using the separate model-specific detectors. Events are oscillator crossings/peaks, **not EEG seizure events**. Aggregate event totals include all trajectories, including repeated controls; they are not independent biological events.

| Metric | 4 s | 10 s |
|---|---:|---:|
| Bistable valid trials | 15/288 (5.2%) | 59/288 (20.5%) |
| Bistable insufficient trials | 273 | 229 |
| Bistable transitions / retained intervals | 306 / 196 | 797 / 658 |
| Bistable events per trajectory, min–max | 0–9 | 0–14 |
| FHN valid trials | 192/288 (66.7%) | 221/288 (76.7%) |
| FHN insufficient trials | 96 | 67 |
| FHN peaks / retained intervals | 1,179 / 889 | 2,429 / 2,047 |
| FHN events per trajectory, min–max | 0–9 | 0–17 |
| Combined valid / insufficient | 207 / 369 | 280 / 296 |
| Maximum clipping fraction | 0 | 0 |

All 665 insufficient trajectories across both durations retain missing CV/coherence, reasons and event/interval counts. No event threshold was relaxed; no result was imputed or smoothed into a valid optimum.

The historical two-context comparison exactly reproduces the stored four-second diagnostic:

| Historical chb08 comparison | 4 s | 10 s |
|---|---:|---:|
| Bistable valid / insufficient | 1 / 95 | 15 / 81 |
| Bistable transitions / intervals | 82 / 46 | 232 / 188 |
| FHN valid / insufficient | 64 / 32 | 71 / 25 |
| FHN peaks / intervals | 406 / 309 | 752 / 638 |
| Combined valid / insufficient | 65 / 127 | 86 / 106 |

Development-only totals also improve: bistable validity 14/192 to 44/192; FHN 128/192 to 150/192. However, at ten seconds driven versus no-drive valid counts are very similar: bistable 30/144 versus 29/144; FHN 114/144 versus 107/144. Availability alone does not show that the response contains useful EEG information. Increasing duration mechanically gives more opportunities for events, and window-relative FHN peak thresholds also change with duration.

At ten seconds, bistable validity is 32/144 with one substep and 27/144 with two; FHN is 111/144 versus 110/144. Three trials and unmatched time-step noise streams cannot establish stochastic convergence or a stable interior-noise optimum. No reproducible SR/CR benefit, seizure discrimination or model improvement is claimed.

## Validation and preservation

- 40 tests passed, including two new duration/PSD tests. Tests cover exact-length and incomplete-tail boundaries, >=50% overlap, canonical metadata parity and zero-power PSD rejection.
- The independent verifier passed: all 19 generated artifact hashes, all **268 existing protected source/notebook/canonical-data/model/analysis files**, trial identities, counts and missing-result preservation.
- All 45 EDF and annotation checksums matched canonical records before use.
- The historical four-second dynamics event counts, validity/reasons, CV/coherence and SNR values matched the saved results.
- Existing four-second artifacts remain untouched on `merge-eeg-project`. Generated plots are incidental outputs; the conclusions above use checked CSV measurements.

Commands:

```powershell
.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider
.venv/Scripts/python.exe -B scripts/validate_window_feasibility.py
# Reproduce into a fresh directory; the default existing directory is protected.
.venv/Scripts/python.exe -B scripts/window_feasibility.py --run-id window-feasibility-4s-10s-replication
```

The verifier currently targets the named v1 run. The manifest remains the authority for that run's checksums; this report is a separate interpretation artifact.

## What a canonical 10-second switch would require

| Artifact/component | Required action |
|---|---|
| Original EDFs, annotations, channel set | Reuse verified sources; no redownload or annotation edits needed. |
| Full-recording preprocessing | Same algorithm can be retained. A correctly versioned cached continuous signal could be reused; current feature checkpoints cannot. |
| Window index and dataset contract | New duration/version, identities, labels, counts and checksums. Keep four-second version. |
| Feature table/checkpoints | Regenerate every feature on 10-second signals. Same 330 column names do not make values interchangeable; line length and other statistics change with duration. |
| Imputation, scaling, feature selection, thresholds | Refit using the new outer/inner training partitions only. Patient membership may stay fixed, but row indices change. |
| Classical models and teacher | Retrain and reevaluate under the new contract. |
| Supervised student and KD | Regenerate teacher targets, selected inputs, transforms and both paired training arms. |
| Robustness | New clean references and frozen-model perturbation evaluations using new bundles/windows. Four-second results remain separate. |
| Inference bundles/CLI | Versioned 10-second extraction and validation contract, new transforms/weights/thresholds. Existing canonical inference checks intentionally reject incompatible contracts; do not weaken them. |
| Reports and deployment benchmarks | Recompute accuracy, patient-level comparisons, event aggregation, timing, calibration, feature importance and figures. A two-window sustained rule becomes 20 seconds rather than 8 and needs an explicit duration-based decision. |

No item requiring training above was run in this feasibility batch.

## Smallest defensible next step

1. Retain four-second ML/KD and keep ten-second dynamics as a separate analysis with explicit event-availability reporting. A negative feasibility finding is valid FYP work; unreliable dynamics need not become classifier inputs.
2. To incorporate the independently implemented `02` PSD work into the predictive contribution, the smallest controlled candidate is a separately versioned **absolute-versus-relative band-power ablation on identical four-second windows**, preserving the canonical baseline. First specify the PSD settings and normalization denominator and verify its extraction. Train-only selection and matched folds are required; no feature-count increase is necessary. This study validates PSD availability, not that ablation's benefit.
3. Only if claiming SR/CR scientifically: predeclare a development-only duration/transient and noise-response study, retain no-drive controls, require reproducible interior-noise behavior and adequate valid-event rates, and check stochastic time-step sensitivity with adequate trials. Ten seconds alone has not met these conditions. Do not tune on chb08 or use seizure labels to pick oscillator frequencies.
4. Only after such evidence should a tiny predefined dynamics summary be considered for a separate predictive ablation. Account explicitly for missing values and whether missingness itself drives classification; evaluate against the unchanged baseline within patient-exclusive folds. The present findings support **analysis only**, not adding SR/CR to the ML vector.

No expensive LOPO/KD or multi-seed model training was launched. Switching canonical duration now would impose a complete regeneration cost without demonstrated seizure-detection or SR/CR benefit.
