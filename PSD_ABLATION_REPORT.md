# Absolute versus relative PSD representation: version 1

## Scope and provenance

This is a separate classical-model experiment in `results/ablations/psd-representation-v1/`. It does not replace or write to the canonical dataset, models, configurations, SR/CR or 10-second artifacts. It uses the same 39,522 corrected four-second windows (630 positive), five patients, 22 channels, annotations, >=50% overlap rule and full-recording preprocessing as the validated canonical dataset. No KD, teacher feature selection, model search or multi-seed training is included.

The inherited frequency extractor supplies canonical absolute powers. The FYP-side `02_preprocess_chb01.ipynb` independently implemented the standard relative-power calculation: band-integrated Welch PSD divided by the total integrated PSD of that channel/window. Its code uses different spectral settings as well as normalization. Two relative arms are therefore necessary to distinguish a representation-only test from a notebook-recipe comparison. The tests execute the notebook's extraction function in isolation and verify equation parity.

## Exact definitions and schema

Let P(f) be the per-channel Welch density, A_b its trapezoidal integral over the selected bins of band b, and T its trapezoidal integral over **all returned frequencies, 0–128 Hz**. Relative power is A_b/T when T>0, otherwise zero, matching the notebook. T is computed separately for each window/channel without labels or cohort statistics. It is not a patient-level normalization and is not the sum of the five band integrals. Values need not sum to one.

| Arm | Spectral definition | Units | Interpretation |
|---|---|---|---|
| `canonical_absolute` | Welch nperseg=256; inclusive lower/upper edges; delta 0.5–4, theta 4–8, alpha 8–13, beta 13–30, gamma 30–40 Hz | V² | Inherited canonical A_b |
| `matched_relative` | Identical PSD, integration bins and bands; A_b divided by T from that same PSD | Dimensionless | Primary representation-only contrast |
| `notebook_relative` | Welch nperseg=512; lower-inclusive/upper-exclusive bands; same lower four nominal bands, gamma 30–70 Hz; A_b/T | Dimensionless | Exact `02` spectral recipe on corrected canonical signals; secondary contrast |

All use SciPy Welch's existing defaults: Hann window, 50% segment overlap, constant detrending, density scaling and mean aggregation. The segment lengths yield 1 Hz versus 0.5 Hz bin spacing. The notebook arm deliberately retains its nominal 70 Hz gamma edge; the shared input remains filtered at 1–40 Hz. It therefore does not restore the original notebook's wider-band signals. Neither relative arm reproduces the original notebook's ten-second/23-channel end-to-end experiment.

**Every arm has 330 inputs.** Only the 110 band powers (22 channels × 5 bands) change; the other 220 time, entropy and Hjorth values are reused exactly. No features are appended. Feature-family order and channel/band positions follow the canonical schema. Spectral names are respectively `freq__<band>__<channel>`, `freq_relative__<band>__<channel>` and `freq02_relative__<band>__<channel>`. `feature_schemas.json` records all 330 ordered names for each arm. Saved relative tables contain metadata plus the 110 replacement columns; the canonical dataset checksum identifies the other 220 values.

The matched arm differs **only in spectral representation** at the extraction level. Separately fitted coefficients and threshold values can naturally change. The notebook arm additionally changes Welch resolution and band integration, so its difference from canonical cannot be attributed solely to normalization. Other unchanged features still retain amplitude information; this is not a claim that the entire classifier becomes amplitude invariant.

## Matched evaluation protocol

All arms use the unchanged canonical logistic-regression factory: balanced class weights, liblinear, max_iter=2000, seed=42 and the estimator's existing default regularization. Each of five outer folds holds out one complete patient. Each outer training set uses four inner patient-exclusive folds to select an F1 threshold on the canonical 0.05–0.95 grid (first maximum wins). Median imputation and float64 scaling are fitted only on the respective inner/outer training data. The threshold-selection procedure is identical across arms; numerical thresholds are allowed to differ.

There are 75 lightweight fits: 3 representations × 5 outer folds × (4 inner fits + 1 outer fit). No held-out labels select features, settings or thresholds. Each outer bundle is independently serialized and its predictions checked after reload. Ablation bundles have their own format identifier and schema; they are not drop-in canonical inference bundles.

Pooled classifications use each window's held-out-fold threshold. Macro metrics give each patient equal weight; patient sample SD/range and paired differences describe heterogeneity. Five patients and overlapping outer training sets do not support treating windows or folds as independent clinical replications. Pooled AUC/AP combine scores from separately fitted folds and are descriptive, not pooled calibration evidence.

## Reproduction

```powershell
.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider
.venv/Scripts/python.exe -B research/validate_psd_ablation.py
# Existing run directories are never overwritten.
.venv/Scripts/python.exe -B research/psd_ablation.py --run-id psd-representation-v1-replication
.venv/Scripts/python.exe -B research/validate_psd_ablation.py --run-id psd-representation-v1-replication
```

The experiment manifest records source/environment/input identities, full configuration, schemas and result hashes. Paired predictions, all inner predictions, inner-fold audits, per-patient metrics, pooled metrics, macro variation, paired patient deltas and all 15 outer bundles are saved separately.

## Results: completed single-seed LOPO

The relative representations improve the aggregate F1 operating point in this logistic-regression experiment, but **do not establish generally better seizure detection**. Average precision falls, improvements are concentrated in chb02, and several patients lose sensitivity/F1. Keep the canonical representation unchanged.

Patient-level held-out results (F1 / average precision):

| Patient | Canonical absolute | Matched relative | Notebook relative |
|---|---:|---:|---:|
| chb01 | 0.7282 / 0.8084 | 0.6989 / 0.7829 | 0.6526 / 0.7144 |
| chb02 | 0.1195 / 0.2475 | 0.5345 / 0.5327 | 0.5385 / 0.5320 |
| chb03 | 0.2946 / 0.4160 | 0.1667 / 0.2897 | 0.2958 / 0.2940 |
| chb05 | 0.3262 / 0.5987 | 0.2230 / 0.3305 | 0.2821 / 0.5512 |
| chb08 | 0.5219 / 0.5072 | 0.6067 / 0.4018 | 0.5458 / 0.3958 |

Held-out sensitivity / specificity:

| Patient | Canonical absolute | Matched relative | Notebook relative |
|---|---:|---:|---:|
| chb01 | 0.6696 / 0.9980 | 0.5804 / 0.9990 | 0.5536 / 0.9983 |
| chb02 | 0.4318 / 0.9544 | 0.7045 / 0.9927 | 0.6364 / 0.9943 |
| chb03 | 0.1845 / 0.9993 | 0.0971 / 0.9993 | 0.2039 / 0.9982 |
| chb05 | 0.7071 / 0.9538 | 0.4857 / 0.9495 | 0.6429 / 0.9488 |
| chb08 | 0.5411 / 0.9797 | 0.7013 / 0.9768 | 0.5801 / 0.9792 |

Equal-weight patient means and sample SD (n=5):

| Metric | Canonical absolute | Matched relative | Notebook relative |
|---|---:|---:|---:|
| F1 | 0.3981 ± 0.2333 | 0.4460 ± 0.2374 | 0.4630 ± 0.1652 |
| Average precision | 0.5156 ± 0.2089 | 0.4675 ± 0.1990 | 0.4975 ± 0.1604 |
| Sensitivity | 0.5068 ± 0.2106 | 0.5138 ± 0.2502 | 0.5234 ± 0.1825 |
| Specificity | 0.9770 ± 0.0223 | 0.9835 ± 0.0211 | 0.9837 ± 0.0211 |
| Balanced accuracy | 0.7419 ± 0.1019 | 0.7486 ± 0.1235 | 0.7536 ± 0.0867 |
| ROC-AUC | 0.8665 ± 0.0821 | 0.8969 ± 0.0939 | 0.9139 ± 0.0595 |

Pooled held-out windows (39,522 windows; 630 positives per arm):

| Metric | Canonical absolute | Matched relative | Notebook relative |
|---|---:|---:|---:|
| Accuracy | 0.9731 | 0.9774 | 0.9774 |
| Precision | 0.3039 | 0.3590 | 0.3583 |
| Sensitivity | 0.5349 | 0.5333 | 0.5317 |
| Specificity | 0.9802 | 0.9846 | 0.9846 |
| Balanced accuracy | 0.7575 | 0.7590 | 0.7582 |
| F1 | 0.3876 | 0.4291 | 0.4281 |
| ROC-AUC | 0.8739 | 0.8796 | 0.8966 |
| Average precision | 0.4314 | 0.3057 | 0.3632 |
| TP / FP / FN / TN | 337 / 772 / 293 / 38120 | 336 / 600 / 294 / 38292 | 335 / 600 / 295 / 38292 |

The pooled F1 increase is associated with fewer false-positive windows, not more detected positive windows. These false-positive counts are window counts, not false alarms per hour or merged alarm events. Accuracy is dominated by the negative majority. AP is reported as average precision, not trapezoidal PR-AUC.

Thresholds were 0.95 except matched-relative/chb01 and notebook-relative/chb02, which selected 0.94. Most thresholds are at the predefined grid ceiling. The grid was not extended after looking at held-out results. These are results under the existing threshold protocol, not evidence of globally optimal thresholds or probability calibration.

## Variation and inference limits

For the **primary, representation-only contrast**, paired patient F1 changes have mean **+0.0479**, sample SD **0.2213**, range **−0.1279 to +0.4150**, and improve in **2/5** patients. Paired AP changes average **−0.0480**, SD **0.2058**; AP improves only in chb02. Chb05 sensitivity falls from 0.7071 to 0.4857 and chb03 from 0.1845 to 0.0971.

For the secondary notebook recipe, paired F1 changes average **+0.0649**, SD **0.2017**, range **−0.0755 to +0.4190**, with improvements in **3/5** patients (chb03's increase is only 0.0012). Paired AP changes average **−0.0181**, SD **0.1715**, again improving only in chb02. The smaller between-patient F1 SD in this arm is descriptive and does not prove more reliable generalization.

As a descriptive sensitivity check on the displayed deltas, excluding chb02 would make mean paired F1 changes negative: approximately −0.0439 for matched relative and −0.0236 for the notebook recipe. No patients are excluded from the official results. This illustrates concentration of gains, not a new selection rule.

Patient SD and paired ranges quantify observed variation, not confidence intervals. No window bootstrap, significance claim, independent-cohort claim or multi-seed uncertainty estimate is made. The outer training sets overlap, and this is the project's five-patient subset. Neither arm was selected for canonical use based on these results.

## Validation and preserved state

- Four new focused tests passed; the full suite passed **44/44**.
- All 39,522 × 110 recomputed absolute powers matched the saved canonical table (rtol=1e-10, atol=1e-25). All window identities/labels matched, and EDF/annotation hashes were verified.
- All replacement values were finite, in [0,1] within tolerance, and there were zero all-band-zero channel/window cases counted by the extraction diagnostic.
- The independent read-only validator passed all **39 output hashes**, **15 saved-bundle prediction checks**, **60 inner-fold exclusion/count/class-ratio audits**, outer-training scaler-mean checks, threshold recomputation, confusion/metric reconstruction, pooled aggregation, macro variation and paired-delta reconstruction.
- All **288 protected pre-existing source, notebook, corrected dataset, model, analysis and 10-second feasibility files** retained identical hashes. The branch remains `merge-eeg-project`.
- The new files are the standalone experiment and validator in `research/`, `tests/test_psd_ablation.py`, this report, and the separate ablation output directory. Canonical code/configuration and behavior were not changed.

## Contribution assessment and next decision

This is a defensible **small applied predictive contribution from the FYP-side `02` work**: a clearly attributed relative-power representation has been integrated into a controlled patient-exclusive ablation without adding architecture or increasing feature count. Its contribution is the implementation, controlled comparison and documented patient-dependent trade-off, not invention of relative band power or proof of universal performance improvement.

The evidence supports a limited statement: in this matched logistic-regression run, relative spectral representations improved aggregate thresholded F1/precision while reducing average precision and harming some patients. The exact notebook recipe is a secondary analysis whose spectral-setting changes prevent attributing its results solely to normalization. Keep all three artifacts and retain canonical absolute features for the established ML/KD pipeline.

**Teacher-based selection was not included.** Every arm used all 330 inputs; there are no top-50 selection-change or KD-effect results to report. If representation is later considered for the final KD system, a predeclared comparison using the existing teacher architecture and training-only selection is needed before attributing any benefit to KD or replacing its inputs. No new architecture is necessary, and this batch does not justify launching or changing expensive KD experiments automatically. SR/CR and ten-second analyses remain unchanged.
