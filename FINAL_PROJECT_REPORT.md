# Final project report

Final frozen reference: **330 canonical features, four-second targets, no SR or CR inputs**. Completed corrected runs are designated final after renewed validation; they were not retrained merely to obtain another realization. This report supersedes earlier planning/status paragraphs, not their preserved measurements.

## 1. Problem statement
Feature-Guided Knowledge Distillation for Interpretable and Lightweight Epileptic Seizure Detection investigates whether training-fold teacher selection and output distillation help a compact EEG classifier generalize to held-out patients. The contribution is a controlled applied comparison, not a clinical deployment claim.

## 2. CHB-MIT dataset and cohort
Five patients: chb01/chb02/chb03/chb05/chb08; 45 recordings, 22 common channels, 256 Hz. Corrected census: 39,522 windows, 630 seizure and 38,892 non-seizure. This is a selected, repeatedly explored subset, not an independent external cohort. Primary base-paper identity and adaptation details remain in README; paper scores are not directly comparable to this protocol.

## 3. Preprocessing
Canonical channel deduplication/common-channel ordering, full-recording 1–40 Hz filtering and retained 60 Hz notch. No held-out cohort statistics fit normalization. Full-recording zero-phase filtering makes the pipeline offline/noncausal.

## 4. Four-second windowing
Non-overlapping 1,024-sample windows; complete final windows included; incomplete tails discarded. Positive labels require at least 50% seizure overlap. The correction added 42 previously omitted non-seizure windows. Annotation/channel/checkpoint validation and training-only transforms remain protected.

## 5. Frozen 330-feature representation
110 time-domain (five per channel), 110 absolute Welch bandpowers (delta/theta/alpha/beta/gamma), 44 entropy (permutation/spectral), 66 Hjorth (activity/mobility/complexity). Explicit order follows the canonical manifest. These definitions are inherited/imported; relative PSD and oscillator features are not final inputs.

## 6. XGBoost teacher
All330 inputs, 300 trees, max_depth5, learning_rate.05 and remaining established configuration. Class weighting derives solely from outer-training labels. Teacher seed42 is fixed across student replications; it was fitted five times, not fifteen independent times.

## 7. Teacher-guided top50 selection
Training-fitted importance selects exactly50 per outer fold with deterministic tie ordering. Selected supervised and KD students share the ordered subset and fitted transform. Random50 draws are deterministic, recorded, training-patient-derived and frozen across student seeds.

## 8. Compact student
50→32→16→1, ReLU, **2,177 parameters**. Training-only median imputation/standard scaling, float32 network, weighted hard-label BCE, Adam learning_rate.001, weight_decay1e-4, batch256, 40epochs. No smoke-test results substitute for this budget.

## 9. Knowledge distillation
Existing output KD: T=3, alpha=.5, T²-weighted soft BCE plus weighted hard BCE; teacher probabilities are in-sample outer-training targets. No cross-fitted distillation or new architecture was introduced. Initialization and loader seeds are paired between students.

## 10. Patient-exclusive LOPO evaluation
All five outer patient folds; no held-out patient affects selection, transforms, weighting or training targets. Final KD protocol uses fixed threshold **.5** for every arm. Inner patient-exclusive threshold tuning applies to the supporting LR representation experiments, not this frozen KD protocol. Seeds42/43/44 vary student initialization and batch order while keeping teacher and selected/random subsets fixed. Repeated predictions are not pooled across seeds as independent windows.

## 11. Final teacher/student/KD results
Primary seed42: the first metric is mean±sample SD over patients; all remaining metrics are pooled over windows.

| model               | Patient F1 ± SD   |   precision |   recall |     f1 |   average_precision |   roc_auc |
|:--------------------|:------------------|------------:|---------:|-------:|--------------------:|----------:|
| random_supervised   | 0.3520 ± 0.2607   |      0.2004 |   0.5746 | 0.2972 |              0.3588 |    0.8735 |
| selected_kd         | 0.4294 ± 0.2893   |      0.2257 |   0.6127 | 0.3299 |              0.4112 |    0.9275 |
| selected_supervised | 0.3781 ± 0.2574   |      0.1652 |   0.7000 | 0.2673 |              0.4314 |    0.8995 |
| teacher             | 0.3866 ± 0.2710   |      0.2084 |   0.4746 | 0.2896 |              0.3287 |    0.9344 |

Per-patient F1 for every student seed (teacher rows are the same reference):

| test_patient   |   seed |   random_supervised |   selected_kd |   selected_supervised |   teacher |
|:---------------|-------:|--------------------:|--------------:|----------------------:|----------:|
| chb01          |     42 |              0.7214 |        0.8326 |                0.7846 |    0.5098 |
| chb01          |     43 |              0.7148 |        0.8139 |                0.6619 |    0.5098 |
| chb01          |     44 |              0.6691 |        0.7188 |                0.7619 |    0.5098 |
| chb02          |     42 |              0.3214 |        0.5224 |                0.2826 |    0.6168 |
| chb02          |     43 |              0.4220 |        0.5128 |                0.4706 |    0.6168 |
| chb02          |     44 |              0.4444 |        0.3795 |                0.5865 |    0.6168 |
| chb03          |     42 |              0.0561 |        0.1081 |                0.2190 |    0.0381 |
| chb03          |     43 |              0.2016 |        0.1897 |                0.2667 |    0.0381 |
| chb03          |     44 |              0.1385 |        0.1176 |                0.1642 |    0.0381 |
| chb05          |     42 |              0.1790 |        0.1919 |                0.1383 |    0.1560 |
| chb05          |     43 |              0.1814 |        0.1840 |                0.1281 |    0.1560 |
| chb05          |     44 |              0.1779 |        0.1984 |                0.1377 |    0.1560 |
| chb08          |     42 |              0.4818 |        0.4918 |                0.4659 |    0.6123 |
| chb08          |     43 |              0.3499 |        0.4843 |                0.4228 |    0.6123 |
| chb08          |     44 |              0.4278 |        0.4825 |                0.4694 |    0.6123 |

Teacher-minus-KD averages −.0286 patient-mean F1 and −.0375 pooled F1 across student seeds. Teacher retains higher patient F1 on chb02/chb08; KD has higher seed-averaged F1 on the other three. Higher teacher ROC-AUC does not imply higher fixed-threshold F1. There is no universal compression penalty or universal student superiority.

## 12. Feature-selection control
Selected-supervised minus random-supervised: seed-averaged paired patient F1 **+.0315**, patient SD .0475, range −.0448 to +.0846; four of five patient averages positive. However, pooled F1 is lower in every seed, averaging **−.0339**. Selection improves some patient-mean/ranking/sensitivity measures at a precision cost; a blanket superiority claim is unsupported. Only one random subset per fold is tested.

## 13. Multi-seed reproducibility and KD effect
Full per-seed results: f1_mean/f1_std are across patients; precision/recall/f1/AP/AUC are pooled windows.

| model               |   seed |   f1_mean |   f1_std |   precision |   recall |     f1 |   average_precision |   roc_auc |
|:--------------------|-------:|----------:|---------:|------------:|---------:|-------:|--------------------:|----------:|
| random_supervised   |     42 |    0.3520 |   0.2607 |      0.2004 |   0.5746 | 0.2972 |              0.3588 |    0.8735 |
| selected_kd         |     42 |    0.4294 |   0.2893 |      0.2257 |   0.6127 | 0.3299 |              0.4112 |    0.9275 |
| selected_supervised |     42 |    0.3781 |   0.2574 |      0.1652 |   0.7000 | 0.2673 |              0.4314 |    0.8995 |
| teacher             |     42 |    0.3866 |   0.2710 |      0.2084 |   0.4746 | 0.2896 |              0.3287 |    0.9344 |
| random_supervised   |     43 |    0.3739 |   0.2156 |      0.1923 |   0.6206 | 0.2937 |              0.2963 |    0.9187 |
| random_supervised   |     44 |    0.3715 |   0.2173 |      0.1936 |   0.6143 | 0.2944 |              0.4077 |    0.8943 |
| selected_kd         |     43 |    0.4369 |   0.2623 |      0.2260 |   0.6063 | 0.3293 |              0.3467 |    0.9024 |
| selected_kd         |     44 |    0.3794 |   0.2383 |      0.2163 |   0.6302 | 0.3221 |              0.4261 |    0.9254 |
| selected_supervised |     43 |    0.3900 |   0.2033 |      0.1549 |   0.6540 | 0.2505 |              0.3785 |    0.8891 |
| selected_supervised |     44 |    0.4239 |   0.2702 |      0.1678 |   0.6397 | 0.2659 |              0.4211 |    0.8944 |
| teacher             |     43 |    0.3866 |   0.2710 |      0.2084 |   0.4746 | 0.2896 |              0.3287 |    0.9344 |
| teacher             |     44 |    0.3866 |   0.2710 |      0.2084 |   0.4746 | 0.2896 |              0.3287 |    0.9344 |

Average of the three seeds' metrics (not concatenated predictions):

| model               |   f1_mean |   precision |   recall |     f1 |   average_precision |   roc_auc |
|:--------------------|----------:|------------:|---------:|-------:|--------------------:|----------:|
| random_supervised   |    0.3658 |      0.1955 |   0.6032 | 0.2951 |              0.3543 |    0.8955 |
| selected_kd         |    0.4152 |      0.2227 |   0.6164 | 0.3271 |              0.3947 |    0.9184 |
| selected_supervised |    0.3973 |      0.1626 |   0.6646 | 0.2612 |              0.4103 |    0.8943 |
| teacher             |    0.3866 |      0.2084 |   0.4746 | 0.2896 |              0.3287 |    0.9344 |

KD minus selected-supervised: patient-mean F1 changes **+.0513,+.0469,−.0446** for seeds42/43/44. Seed-averaged paired patient gain **+.0179**, patient SD .0552; pooled F1 improves in all three seeds, mean **+.0659**, but recall decreases .0481 and AP decreases .0157 on average. Mean patient F1 across seeds is .3973±.0238 for supervised and .4152±.0313 for KD, where these SDs are **seed SDs**, not patient SDs. Evidence supports a seed-sensitive operating-point trade-off, not universally improved detection/ranking.

## 14. Lightweightness measurements
New CPU measurements use the final seed42 bundles and the first recording of each held-out patient; no retraining. Recordings have900–902 complete windows/3,600–3,608seconds of windowed EEG. Network timing uses100 warm repeats; feature-table API uses30 repeats. Single EDF passes include load/preprocessing, all330 feature extraction and inference with an already-loaded bundle. Imports/model-disk loading are excluded; caches/host activity are not controlled and validation/test activity overlapped part of the benchmark. Timing is a descriptive single-host benchmark, not a device-independent latency guarantee.

Serialized sizes in bytes (full joblib bundles include schemas/transforms/metadata; payload formats differ):

| arm                 |   bundle_bytes_min |   bundle_bytes_max |   model_payload_bytes_min |   model_payload_bytes_max |
|:--------------------|-------------------:|-------------------:|--------------------------:|--------------------------:|
| selected_kd         |              24415 |              24511 |                     11413 |                     11413 |
| selected_supervised |              24423 |              24519 |                     11413 |                     11413 |
| teacher             |             525917 |             625877 |                    502159 |                    602119 |

Ranges across patients of median milliseconds per call; batch is the number of windows, not milliseconds per window:

| arm                 | scope                                                            |   batch |    min |    max |
|:--------------------|:-----------------------------------------------------------------|--------:|-------:|-------:|
| selected_kd         | feature_table_API_including_transform_and_student_reconstruction |       1 | 1.6618 | 2.2716 |
| selected_kd         | feature_table_API_including_transform_and_student_reconstruction |     256 | 1.7998 | 2.2865 |
| selected_kd         | resident_network_including_tensor_conversion                     |       1 | 0.0430 | 0.0483 |
| selected_kd         | resident_network_including_tensor_conversion                     |     256 | 0.0642 | 0.1684 |
| selected_supervised | feature_table_API_including_transform_and_student_reconstruction |       1 | 1.6293 | 2.5762 |
| selected_supervised | feature_table_API_including_transform_and_student_reconstruction |     256 | 1.5882 | 2.6958 |
| selected_supervised | resident_network_including_tensor_conversion                     |       1 | 0.0400 | 0.0472 |
| selected_supervised | resident_network_including_tensor_conversion                     |     256 | 0.0673 | 0.2052 |
| teacher             | feature_table_API_including_transform_and_student_reconstruction |       1 | 1.4545 | 2.1743 |
| teacher             | feature_table_API_including_transform_and_student_reconstruction |     256 | 2.5537 | 3.3231 |

EDF timing ranges in seconds:

| arm                 |   feature_extraction_seconds_min |   feature_extraction_seconds_max |   end_to_end_seconds_min |   end_to_end_seconds_max |
|:--------------------|---------------------------------:|---------------------------------:|-------------------------:|-------------------------:|
| selected_kd         |                          16.7671 |                          23.4813 |                  17.9553 |                  25.0704 |
| selected_supervised |                          16.7671 |                          23.4813 |                  17.9542 |                  25.0695 |
| teacher             |                          16.7671 |                          23.4813 |                  17.9642 |                  25.0797 |

Whole-process memory snapshots, bytes:

| patient   |   rss_before_bytes |   rss_after_bytes |
|:----------|-------------------:|------------------:|
| chb01     |          575676416 |         935809024 |
| chb02     |          946319360 |         958480384 |
| chb03     |          946442240 |         959139840 |
| chb05     |          965865472 |         963035136 |
| chb08     |          963088384 |         962699264 |

RSS includes Python/libraries, loaded EEG, features and models; it is not isolated model memory or a measured peak. Network size/timing supports a **compact classifier** claim. Inference still computes all330 features, reads/filters a complete recording, and is noncausal; **end-to-end lightweight detector status is not established**, even if offline processing is faster than recording duration. No energy, embedded-device or clinical real-time measurement was made.

## 15. Interpretability and selection stability
Ten fold-pair Jaccards: mean 0.2007, range 0.1111–0.2987; 145 unique features among250 fold selections, 3 selected in all five folds. Overlapping training sets limit independence.

Feature-family counts across250 selections:

| family   |   selections |
|:---------|-------------:|
| entropy  |           10 |
| freq     |          102 |
| hjorth   |           40 |
| time     |           98 |

Channel counts:

| channel   |   selections |
|:----------|-------------:|
| C3-P3     |            8 |
| C4-P4     |            8 |
| CZ-PZ     |           24 |
| F3-C3     |           17 |
| F4-C4     |            5 |
| F7-T7     |            9 |
| F8-T8     |           12 |
| FP1-F3    |            7 |
| FP1-F7    |            7 |
| FP2-F4    |           12 |
| FP2-F8    |            1 |
| FT10-T8   |           14 |
| FT9-FT10  |           25 |
| FZ-CZ     |           17 |
| P3-O1     |           10 |
| P4-O2     |           16 |
| P7-O1     |           12 |
| P7-T7     |            8 |
| P8-O2     |           11 |
| T7-FT9    |            6 |
| T7-P7     |            6 |
| T8-P8     |           15 |

Frequency-band counts among selected frequency features only:

| band   |   selections |
|:-------|-------------:|
| alpha  |           15 |
| beta   |           10 |
| delta  |           27 |
| gamma  |           14 |
| theta  |           36 |

Per-fold distributions, exact ranked sets and feature occurrence counts are saved. These describe teacher-guided predictive feature reliance, not clinical causality or faithful explanations of the student. **Student-specific reliance/faithfulness is not established**: no existing implemented method supports it, and this final stage introduces no major new explanation methodology. The title's interpretability claim is limited accordingly.

## 16. PSD/relative-power ablation
Preserved matched LR experiment: pooled F1 absolute .3876→matched-relative .4291, while AP .4314→.3057; effects vary strongly by patient. The exact02 notebook recipe is a secondary, partly confounded spectral comparison. Relative power is supporting evidence, not a replacement for final absolute features. See PSD_ABLATION_REPORT.md.

## 17. Noise robustness
Preserved frozen-bundle diagnostics on two selected chb08 windows evaluated clean/white/pink/black perturbations. Clean parity passed; the tiny study does not establish population robustness. Perturbations are post-filter, not acquisition-noise simulations, and involved earlier validation bundles rather than a final-model population benchmark. No new robustness claim is inferred.

## 18. Compression/reconstruction
Existing stride/interpolation and anti-aliased sample-reduction controls are waveform reconstruction analyses. They are not measured bitstream compression and not teacher/student model compression. Keep their bounded metrics separate from neural classifier size measurements.

## 19. SR and negative predictive ablation
Ten-second trailing context retained four-second targets. Predeclared ten-seed averaging materially improved rank consistency (.608→.927) and reduced variability, but final matched LR comparison did not demonstrate incremental prediction: patient-mean F1 **.3981→.3881**, pooled F1 **.3876→.3579**, pooled AP **.4314→.4032**. Full extraction had39,432 valid/90 boundary-missing windows, including all630 positives. SR is a reproducible representation with unproven incremental benefit in this cohort/model, not 'useless'. It is excluded from final training.

## 20. CR and evidence-based exclusion
Current FHN inter-event-CV had near-zero/unstable context rank consistency, poor cross-seed agreement and inconsistent driven/no-drive differences; lowest pooled CV also occurred without drive. The fixed grid did not establish reproducible EEG-dependent predictive signal, so CR remains exploratory and excluded. No rescue scalar was created. This does **not** invalidate all possible coherence-resonance formulations. The10second feasibility census (15,808 windows,250 positives versus39,522/630 at4s) improved event availability but retained substantial insufficient trials; it did not justify changing canonical windows.

## 21. Limitations
Five selected patients, overlapping training folds and repeated exploration; no external/general-population claim. Three seeds condition on a fixed teacher/random draw. Fixed .5 thresholds confound F1 comparisons with operating point; KD uses in-sample teacher targets. KD reduces AP/sensitivity on average despite pooled F1 gains. No event-level alarm evaluation, clinical causality, faithful student explanation, isolated peak memory, embedded/energy benchmark or end-to-end lightweightness claim. Native Windows import access-violation diagnostics appeared during the final test invocation, which nevertheless exited0 with69 passing tests; the cause is not established.

## 22. Exact project contributions
Inherited: canonical330 extractor, classical EEG baseline concepts and imported exploration infrastructure. FYP-side: teacher-guided compact-student/output-KD system and independent02 PSD/SR/CR exploration. Corrective engineering: complete windows, checked annotations/channels/checkpoints, training-only scaling/class weights, explicit bundles/provenance and patient-exclusive comparisons. Validated evidence: matched random-feature control, paired small-seed KD evaluation, selection stability, measured classifier/pipeline costs and carefully retained negative/qualified PSD/SR/CR findings. No novelty claim is made for relative power, KD, oscillator equations or the inherited330 definitions themselves.

## 23. Reproducibility and final artifact set
Use the pinned existing Python environment/requirements-lock.txt. Final core evidence is the revalidated corrected contribution runs, not historical notebook or smoke-test outputs:
- results/contribution/kd-contribution-v1-seed42/ — primary bundles, transforms, thresholds, predictions, selections and training logs.
- results/contribution/kd-contribution-v1-seeds43-44/ — predefined paired replication bundles and predictions.
- results/contribution/kd-contribution-v1-summary/ — all per-patient/per-seed metrics and paired effects.
- results/final/final-evidence-v1/ — new size/runtime/memory and selection-stability measurements, source/environment/protection manifest.
- data/processed/corrected_v2/ — canonical dataset/index/schema/provenance; original raw EEG is also required for EDF reproduction.
- KD_CONTRIBUTION_REPORT.md and this report — core tables and qualified interpretation.
- PSD_ABLATION_REPORT.md, ANALYSIS_VALIDATION.md, WINDOW_FEASIBILITY_10S.md, SRCR_REPLICATE_STABILITY_REPORT.md, SRCR_CANDIDATE_STUDY_REPORT.md, SR_PREDICTIVE_ABLATION_REPORT.md — supporting evidence and exclusions.

Commands: `python -B research/validate_kd_contribution.py results/contribution/kd-contribution-v1-seed42`; repeat for kd-contribution-v1-seeds43-44; `python -B -m pytest -q -p no:cacheprovider`. Retraining reproduction uses research/kd_contribution.py with **fresh run IDs** and paired seeds as documented in KD_CONTRIBUTION_REPORT.md. No retraining was needed for this designation because existing corrected40epoch runs already exactly match the frozen design. research/final_evidence.py records the new measurements and refuses overwrite. Binary datasets/bundles may be gitignored: preserve the actual artifact directories, not just Git history.

## 24. Final conclusion
The final experiment is complete: a corrected, reproducible330-feature offline seizure-detection pipeline supports a compact2,177-parameter student. Teacher-guided selection and KD show conditional, metric-dependent effects; KD improves pooled F1 across three paired seeds while sacrificing sensitivity/AP and displaying patient/seed variation. SR/CR were investigated and excluded on evidence rather than forced into prediction. Report a compact classifier and audited experimental contribution, not a clinically validated or end-to-end lightweight detector. No further experiment or automatic KD run is scheduled.
