# Matched feature-selection and KD contribution validation, v1

## Design

This experiment reuses the protected canonical 4-second dataset (39,522 windows, 630 positives, five patients), 330 absolute-power multidomain features, patient-exclusive splitting, model factories, train-only transforms, canonical student training, metrics and saved-bundle inference. No canonical source/configuration/results, PSD ablation, SR/CR or 10-second study is modified. Work remains on `merge-eeg-project`.

| Arm | Input | Training |
|---|---|---|
| A `teacher` | All 330 canonical features | XGBoost, 300 depth-5 trees, canonical parameters, training-fold class ratio, seed 42 |
| B `selected_supervised` | Training-fitted teacher's top 50 | Weighted hard-label BCE |
| C `selected_kd` | Exactly B's ordered 50 features and fitted transform | Existing hard/soft output KD, T=3, alpha=0.5 and T² weighting |
| D `random_supervised` | Uniform random 50 without replacement | Same weighted hard-label BCE as B |

All students have 50→32→16→1 dimensions and 2,177 parameters, with the configured 40 epochs, Adam learning rate 0.001, weight decay 1e-4 and batch size 256. There is no early stopping or budget reduction. All four models use the existing KD protocol's fixed **0.5 threshold**; there is no threshold tuning or held-out parameter selection. This differs from the nested-threshold logistic-regression PSD experiment and should not be conflated with it.

Each outer fold excludes the held-out patient from teacher fitting, importance ranking, imputation, scaling, class ratios and teacher training targets. The random selector receives only the outer-training frame and canonical feature schema. Its recorded 32-bit seed is derived from a version tag, base seed 42 and the sorted training-patient identifiers; feature values/labels do not influence the draw. Chosen random columns retain canonical schema order. Teacher-selected columns retain canonical importance-ranking order, including its deterministic tie handling.

Baseline and KD share the same selected feature list, fitted transform, initialization seed and data-loader seed; random-50 uses identical capacity and training seed. Initial-state fingerprints and seeds are recorded. Teacher targets are in-sample outer-training probabilities, as in the existing implementation; cross-fitted distillation is not introduced.

The primary run uses student seed **42**. A bounded replication is predefined as student seeds **43 and 44**, conditional on a successful primary run with reasonable runtime (under ten minutes). Replication freezes each fold's seed-42 teacher, selected feature set and random subset, and varies student initialization and batch order. This isolates student-training variation; it does not estimate variation across teacher fits or random feature subsets. The repeated teacher rows are the same reference, not independent teacher replications. Random-50 KD is not included; no selection-by-KD interaction claim is supported.

## Comparisons and aggregation

- **Selection effect:** B minus D.
- **KD effect:** C minus B.
- **Teacher performance gap:** A minus C; a positive number favors the teacher.

Effects are paired within patient and seed. Patient means give each patient equal weight; pooled metrics combine held-out window predictions. Sample SD across five patients and ranges describe heterogeneity, not confidence intervals. For replication, seed summaries remain separate and patient averages across seeds do not create additional independent patients. Average precision is sklearn AP, not trapezoidal PR-AUC. No deployment-cost or student-interpretability experiment is part of this run.

## Artifacts and reproduction

The separate runner is `research/kd_contribution.py`; its read-only validator is `research/validate_kd_contribution.py`. Results use `results/contribution/`, with immutable fresh run IDs. Every patient/seed folder records all four bundles, predictions, full teacher ranking, selected/random feature lists, initialization/loader/selection seeds, train/test window fingerprints, class ratios, training losses and a completion receipt. The run manifest records configuration, source/environment/dataset identity, protected-file hashes and output hashes. Fold outputs are saved before advancing; failed runs are not marked complete.

```powershell
.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider
.venv/Scripts/python.exe -B research/kd_contribution.py --run-id kd-contribution-v1-seed42
.venv/Scripts/python.exe -B research/validate_kd_contribution.py results/contribution/kd-contribution-v1-seed42
# Only after primary validation and runtime review:
.venv/Scripts/python.exe -B research/kd_contribution.py --run-id kd-contribution-v1-seeds43-44 --seeds 43 44 --primary-run results/contribution/kd-contribution-v1-seed42
.venv/Scripts/python.exe -B research/validate_kd_contribution.py results/contribution/kd-contribution-v1-seeds43-44
```

Existing directories cannot be overwritten; reproductions require fresh run IDs and the corresponding primary-run path.

## Primary results: seed 42, five held-out patients

| Patient | A Teacher | B Selected supervised | C Selected KD | D Random supervised | Selection B−D | KD C−B | Teacher A−C |
|---|---:|---:|---:|---:|---:|---:|---:|
| chb01 | 0.5098 | 0.7846 | 0.8326 | 0.7214 | +0.0632 | +0.0480 | −0.3228 |
| chb02 | 0.6168 | 0.2826 | 0.5224 | 0.3214 | −0.0388 | +0.2398 | +0.0944 |
| chb03 | 0.0381 | 0.2190 | 0.1081 | 0.0561 | +0.1629 | −0.1109 | −0.0700 |
| chb05 | 0.1560 | 0.1383 | 0.1919 | 0.1790 | −0.0407 | +0.0536 | −0.0359 |
| chb08 | 0.6123 | 0.4659 | 0.4918 | 0.4818 | −0.0159 | +0.0259 | +0.1205 |

All values above are patient F1 or paired F1 differences. The teacher is better than KD in two patients and worse in three at the fixed threshold; there is no uniform teacher-to-student performance loss.

| Arm | Patient-mean F1 ± patient SD | Pooled precision | Pooled sensitivity | Pooled F1 | Pooled AP | Pooled ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| A Teacher | 0.3866 ± 0.2710 | 0.2084 | 0.4746 | 0.2896 | 0.3287 | 0.9344 |
| B Selected supervised | 0.3781 ± 0.2574 | 0.1652 | 0.7000 | 0.2673 | 0.4314 | 0.8995 |
| C Selected KD | 0.4294 ± 0.2893 | 0.2257 | 0.6127 | 0.3299 | 0.4112 | 0.9275 |
| D Random supervised | 0.3520 ± 0.2607 | 0.2004 | 0.5746 | 0.2972 | 0.3588 | 0.8735 |

Primary selection improves mean-patient F1 by **0.0261**, but only **2/5** patients improve and pooled F1 is **0.0299 lower** than random-50. Primary KD improves mean-patient F1 by **0.0513**, with **4/5** positive patient differences; pooled F1 increases **0.0626**, while sensitivity falls **0.0873** and pooled AP falls **0.0202**. Teacher-minus-KD is **−0.0427** in mean-patient F1 and **−0.0403** in pooled F1. These are different aggregation units, not interchangeable scores.

The primary run took **188.25 seconds** of recorded experiment time, passed independent validation and was saved before launching the two predefined replications. This timing is a run-budget check, not an inference/deployment benchmark.

## Three-seed results and variation

Both predefined replications completed at the full 40 epochs. Together they took **296.50 seconds** of recorded experiment time. In total, five teachers were trained and 45 student fits completed; repeated teacher references do not represent additional teacher training.

Mean-patient F1 for each seed (± SD across the five patients):

| Student seed | A Teacher (fixed) | B Selected supervised | C Selected KD | D Random supervised | Selection effect | KD effect | Teacher−KD |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.3866 ± 0.2710 | 0.3781 ± 0.2574 | 0.4294 ± 0.2893 | 0.3520 ± 0.2607 | +0.0261 | +0.0513 | −0.0427 |
| 43 | 0.3866 ± 0.2710 | 0.3900 ± 0.2033 | 0.4369 ± 0.2623 | 0.3739 ± 0.2156 | +0.0161 | +0.0469 | −0.0503 |
| 44 | 0.3866 ± 0.2710 | 0.4239 ± 0.2702 | 0.3794 ± 0.2383 | 0.3715 ± 0.2173 | +0.0524 | −0.0446 | +0.0073 |

The following table averages each seed's metric; it **does not concatenate repeated window predictions across seeds**. The ± values in its first column are SD across three student seeds, not patient SD.

| Arm | Mean of patient-mean F1 ± seed SD | Mean pooled precision | Mean pooled sensitivity | Mean pooled F1 | Mean pooled AP | Mean pooled ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| A Teacher (same reference) | 0.3866 (fixed) | 0.2084 | 0.4746 | 0.2896 | 0.3287 | 0.9344 |
| B Selected supervised | 0.3973 ± 0.0238 | 0.1626 | 0.6646 | 0.2612 | 0.4103 | 0.8943 |
| C Selected KD | 0.4152 ± 0.0313 | 0.2227 | 0.6164 | 0.3271 | 0.3947 | 0.9184 |
| D Random supervised | 0.3658 ± 0.0121 | 0.1955 | 0.6032 | 0.2951 | 0.3543 | 0.8955 |

Per-patient F1 averaged over the three student seeds (paired effects also averaged within patient first):

| Patient | A Teacher | B Selected supervised | C Selected KD | D Random supervised | Selection B−D | KD C−B | Teacher A−C |
|---|---:|---:|---:|---:|---:|---:|---:|
| chb01 | 0.5098 | 0.7361 | 0.7884 | 0.7018 | +0.0344 | +0.0523 | −0.2786 |
| chb02 | 0.6168 | 0.4466 | 0.4716 | 0.3960 | +0.0506 | +0.0250 | +0.1453 |
| chb03 | 0.0381 | 0.2166 | 0.1385 | 0.1320 | +0.0846 | −0.0781 | −0.1004 |
| chb05 | 0.1560 | 0.1347 | 0.1914 | 0.1795 | −0.0448 | +0.0568 | −0.0355 |
| chb08 | 0.6123 | 0.4527 | 0.4862 | 0.4198 | +0.0329 | +0.0335 | +0.1262 |

Full per-patient, per-seed F1/AP/ROC-AUC/precision/recall and paired changes are in the combined summary's `per_patient_metrics.csv` and `paired_effects.csv`. `patient_f1_seed_variation.csv` records per-patient seed SD; `metrics_by_seed.csv` and `seed_variation.csv` retain every pooled metric and its seed variation.

## Contribution findings

**Selection:** mean-patient F1 improves in all three seeds. After seed-averaging within each patient, the mean paired gain is **+0.0315**, patient SD **0.0475**, range **−0.0448 to +0.0846**, with positive differences in 4/5 patients. However, selected supervised has **lower pooled F1 in every seed** than random-50; the mean pooled difference is **−0.0339**. It increases pooled AP and sensitivity but loses precision. Thus there is limited, metric-dependent evidence for teacher-guided selection, not general superiority over random features. The random draw samples all 330 columns and may overlap the teacher-selected set; it is not constructed to be uninformative.

**KD:** the mean paired patient F1 gain, after seed-averaging per patient, is **+0.0179**, patient SD **0.0552**, range **−0.0781 to +0.0568**, with positive mean differences in 4/5 patients. But the overall mean-patient effect is positive at seeds 42/43 and negative at seed 44. KD loses F1 in chb03 in all seeds; it improves chb05 and chb08 in all seeds. Pooled F1 improves in every seed (differences +0.0626, +0.0789, +0.0562), averaging **+0.0659**. Across seeds, pooled precision increases about **0.0601**, sensitivity decreases **0.0481**, and AP decreases **0.0157**. The evidence supports an operating-point trade-off and a small, seed-sensitive patient-mean gain, not uniform improvement in detection or ranking quality.

**Teacher/student performance gap:** teacher-minus-KD averages **−0.0286** in patient-mean F1 and **−0.0375** in pooled F1 across student seeds. The teacher retains higher F1 on chb02/chb08, while KD has higher F1 on chb01/chb03/chb05. Teacher pooled ROC-AUC is higher than KD's seed average. There is no universal F1 compression penalty, but these comparisons do not establish actual runtime, memory or energy savings.

**Defensible claim:** this experiment provides controlled evidence that the feature-guided KD system can change the precision/sensitivity trade-off and improve some aggregate F1 measures at the fixed canonical threshold. It also exposes important failures and seed dependence. The FYP contribution can be the reproducible, explicitly qualified selection/KD comparison; a blanket claim that teacher selection or KD is superior is not supported.

## Validation completed

- **50 tests passed**, including six new focused checks for training-only random selection, identical selected schemas/capacity, paired initialization and loader seeds, deterministic repeat training, patient versus pooled aggregation, incomplete/tampered provenance, and correct handling of repeated seeds.
- All **60 saved bundles** passed reload prediction/decision checks. Independent validation recomputed metrics and paired effects, recovered teacher selections and deterministic random lists, checked training-only medians/scaler means and teacher class ratios, and verified all 40-epoch loss histories and 2,177-parameter students.
- The primary's **55 artifact hashes** and replication's **105 artifact hashes** passed. Replication verified the unchanged primary manifest/artifacts, teacher targets, selected/random lists and train/test identities. All **333 protected pre-existing files** remained unchanged.
- The completed runs are `kd-contribution-v1-seed42` and `kd-contribution-v1-seeds43-44`; their separate combined results are in `kd-contribution-v1-summary`. No historical pre-correction KD results enter these tables.
- New work is confined to three standalone research scripts, the focused test file, this report and separately versioned contribution results. Canonical source/configuration/results and the PSD/SR/CR/ten-second studies remain unchanged.

## Remaining limitations

Five selected, previously explored patients remain the only independent patient units. Three seeds are a small conditional replication, with overlapping outer training sets; no statistical-significance or external-generalization claim is made. One random subset per fold is held fixed across seeds, so variability across random feature draws is unmeasured. Teacher-seed variability, random-50 KD and a selection-by-KD interaction are also unmeasured. The supervised selected arm already depends on a teacher for input selection, even though it receives hard labels only.

Threshold 0.5 is fixed for fairness and preservation of the canonical KD protocol. F1 differences therefore include operating-point/calibration effects; for example, the teacher has high chb03 ROC-AUC but extremely low sensitivity at 0.5. No held-out tuning was used to repair such cases. In-sample teacher targets remain unchanged. Poor patient-specific sensitivity, seed reversals and AP declines prevent an unqualified superiority claim.

Student size is an architectural fact, not an end-to-end deployment benchmark: canonical inference still extracts all 330 features. Deployment cost, student interpretability, clinical alarm metrics and broader robustness were deliberately not investigated here. This batch completes the requested contribution experiment and does not authorize promotion of a new canonical model or further experiments.
