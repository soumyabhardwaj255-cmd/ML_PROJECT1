# SR replicate-average and CR drive-sensitivity study v1

## Predeclared design and scope

This separate diagnostic study preserves the canonical four-second ML sample and frozen trailing ten-second context [e−10,e). Existing datasets, ML/KD results and SR/CR studies are unchanged. No labels were loaded and no model, imputer, scaler, feature selector or decision threshold was fitted. Centered context was not reconsidered.

The sample contains 90 new contexts: two per recording, nearest the quarter and three-quarter positions among complete trailing contexts, excluding overlap with the previous first/middle/last diagnostic contexts in either mapping. These are new temporal contexts within the same 45 recordings/five patients, not new patients or statistically independent confirmation subjects. Contexts and configuration were saved before measurement; no adaptive stopping, grid refinement or seed selection was used.

**SR:** original per-trial spectral-ratio change at D=.5 versus D=0, with unchanged characteristic component, normalization, channel and oscillator settings. The candidate is the arithmetic mean of the ten dB values from seeds **62–71**. Three additional disjoint groups, 72–81, 82–91 and 92–101, assess repeatability. Seed42 supplies the current single-trial comparator. A block mean is missing if any constituent is missing; it never averages a favorable subset of valid trials. There are 3,690 SR trials (90 × 41).

**CR:** existing FHN interval CV, with drive coefficients **0, .175, .35, .70** and noise intensities **D=.25, .5, 1.0**. These are zero/half/default/double drive and half/default/double noise, fixed in advance. Twenty seeds **62–81** are shared across settings and contexts, yielding 21,600 trials including zero-drive controls. Drift, initial states, dt, clipping, peak detector and interval rules remain unchanged. Scaling the evaluator input is algebraically equivalent to the existing `input_scale` parameter, checked against the original implementation. No CR scalar is selected by maximizing this grid.

Within-context variability uses sample SD across replicates (or across four ten-replicate block means). Rank consistency uses Spearman correlation across the 90 contexts. SR single-trial statistics cover all 780 pairs among the 40 new seeds; block statistics cover all six pairs of ten-seed means. CR single-seed statistics cover 190 seed pairs per grid cell. CR split means use seeds62–71 versus72–81, with complete-replicate means only. Paired differences always use matched seeds and the same noise intensity. Repeated zero-drive controls are not independent EEG observations.

The current equations and missingness rules are tested rather than redesigned. No-drive SR remains mathematically undefined under the unchanged deterministic zero-drive baseline, as established previously; this study does not substitute a different baseline or claim an SR no-drive effect.

## A. SR results and recommendation

| Representation | Median within-context SD | Median rank correlation [range] |
|---|---:|---:|
| Individual trials, 40 new seeds | 3.723 dB | .608 [.026, .925] |
| Ten-replicate averages, four disjoint groups | .986 dB | .927 [.851, .968] |

The median within-context SD falls by about **74%**. The first two predeclared ten-seed groups have rank correlation **.933** and mean absolute difference **.967 dB**. Against the independent remaining 30-replicate mean, the preselected candidate has rank correlation **.966** and mean absolute difference **.744 dB**; current seed42 has **.669** and **4.955 dB**, respectively. The 30-replicate mean is a stochastic-reference estimate, not ground truth. These results support a material improvement in both rank and value reproducibility on new temporal contexts. Uncertainty remains: four blocks provide a limited estimate of average-feature variability, and residual differences near 1 dB persist.

All **3,690 SR trials** were valid; all four block means are valid for all 90 sampled contexts. This does not establish complete-corpus coverage for the new ten-replicate candidate. Boundary contexts must remain missing, and full-corpus validity for all ten seeds still needs measurement before dataset promotion.

**Recommendation: freeze the preselected ten-replicate SR candidate for a future controlled predictive test**, with an explicit new feature/version name. This is evidence of a more reproducible representation, not evidence of seizure discrimination, denoising or stochastic resonance. No seed group was chosen after observing results. The old single-trial SR and its exploratory conclusion remain unchanged.

## B. CR response and seed stability

Every cell contains 1,800 trials (90 contexts × 20 seeds). CV is dimensionless; smaller CV means more regular detected intervals. Delta is driven minus matched zero-drive CV.

| Noise D | Drive coefficient | Mean CV | Mean delta | Median within-context SD | Median single-seed rank | Split-ten rank |
|---|---:|---:|---:|---:|---:|---:|
| .25 | 0 | .832 | 0 | .509 | undefined | undefined |
| .25 | .175 | .811 | −.0207 | .499 | −.015 | −.351 |
| .25 | .350 | .803 | −.0286 | .488 | −.007 | −.376 |
| .25 | .700 | .804 | −.0281 | .469 | .032 | −.175 |
| .50 | 0 | .762 | 0 | .453 | undefined | undefined |
| .50 | .175 | .767 | +.0058 | .459 | .011 | .309 |
| .50 | .350 | .770 | +.0081 | .458 | .010 | .161 |
| .50 | .700 | .780 | +.0179 | .455 | .008 | −.039 |
| 1.0 | 0 | .913 | 0 | .579 | undefined | undefined |
| 1.0 | .175 | .901 | −.0122 | .578 | −.005 | .101 |
| 1.0 | .350 | .891 | −.0225 | .578 | .008 | .245 |
| 1.0 | .700 | .902 | −.0118 | .563 | .007 | .284 |

No-drive context ranks are undefined because each seed produces the same no-drive trajectory for every context. They must not be treated as zero correlation or as 90 independent control observations.

All **21,600 CR trials were valid**, including all 5,400 no-drive trials; no clipping occurred. The minimum retained interval count across the grid was three, exactly the existing validity threshold. No insufficient outcomes were filled or detector thresholds relaxed. Zero missingness here does not establish stable CR values.

There is a pooled dependence on noise intensity: the mean CV is lowest near D=.5, **including in the no-drive control**. This is not sufficient evidence of EEG-driven coherence resonance. Within-seed-block drive responses are inconsistent. At the original drive coefficient .35:

| Noise D | Driven-minus-no-drive mean, seeds62–71 | Same difference, seeds72–81 |
|---|---:|---:|
| .25 | −.0982 | +.0409 |
| .50 | +.0288 | −.0127 |
| 1.0 | −.0197 | −.0254 |

The small negative D=1 mean response persists across the two blocks, but context ranking remains weak (.245); this isolated population-average observation does not justify selecting D=1 as a feature. Across driven cells, single-seed median rank correlations range from −.015 to .032, while split-ten correlations range from −.376 to .309. Mean driven/no-drive differences are small compared with replicate SD (.45–.58), and the median paired difference is zero in eight of nine driven cells (−.0041 in the remaining cell). Mean absolute paired changes range .026–.200, showing that the drive can perturb event detection without supplying reproducible context discrimination.

**Recommendation: keep the existing FHN CR quantity exploratory.** This grid does not provide a sufficiently reproducible EEG-dependent response to justify freezing a scalar CR feature. Do not select an optimal cell, minimum CV across doses, fitted curve parameter or driven-minus-control feature after inspecting this grid. A new such scalar would require an independently predeclared definition and new validation; none was implemented. These conclusions are limited to the tested channel, ten-second context, existing component normalization, detector and small drive/noise range, not a claim that all possible FHN/CR formulations fail.

## Exact proposed freeze and next-stage readiness

Only the new **SR candidate** is recommended for freezing:

- Name: `sr__mean10_snr_change_vs_zero_noise_db__FP1-F7__ctx10`; paired flag: the same name plus `__missing`.
- Source: FP1-F7 from unchanged canonical preprocessing, strict trailing ten-second context [e−10,e) for each unchanged four-second target. Same 2,560 samples at 256 Hz, no padding or cross-record borrowing. The frozen mapping adds no look-ahead; canonical filtering remains noncausal.
- For each seed 62 through 71 inclusive, compute the existing characteristic component/normalization and bistable trajectories at D=.5 and D=0, with unchanged dt, drift, initial state and clipping. For each trajectory, the existing spectral ratio is 10 log10(mean PSD within ±.2 Hz of the context-derived frequency / mean PSD outside ±.2 Hz but within ±2 Hz). The per-trial feature is the D=.5 ratio minus the D=0 ratio, in dB. Average those **ten dB differences arithmetically**.
- Preserve all finite signed values. If the context is unavailable or any constituent measurement is undefined/clipped, output NaN and flag1; otherwise output the average and flag0. This is a new, explicitly versioned averaging contract, not a modification of the old single-trial column. Event detector or CR changes are not involved.
- No learned cohort transform is part of extraction. Any later imputation/scaling must be fitted only inside the appropriate training fold. No such transform was fitted here.

**CR has no new frozen scalar recommendation.** Its original single-trial column and missingness indicator remain intact in the old artifact. The guide's requirement does not supply missing reproducibility evidence. The combined SR/CR feature pair is therefore not ready to be described as scientifically validated; forcing CR into a classifier is not supported by this study.

The SR average is ready as a precisely specified candidate for subsequent full-corpus extraction/coverage checks and then a controlled predictive ablation. Neither step was run here. Before a combined predictive design is finalized, resolve whether CR remains a separately reported exploratory analysis or whether a distinct, explicitly authorized scientific reformulation is needed. There is no automatic substitution of a different coherence measure.

## Validation and provenance

**15 focused tests passed** (new six tests plus existing nine), including fixed seed groups, missing-value aggregation, deterministic disjoint-context selection and original FHN equation parity at all four predeclared scale/dose test pairs. Real-data parity checks passed on **90 contexts**, and exact repeated SR computations passed on all **90**. A separate read-only verification confirmed output hashes, trial uniqueness and **578 protected files unchanged**. Source, configuration and output hashes, environment, context IDs, raw-file verification and protection inventory are recorded in the new manifest. No prior report or result directory was overwritten.

Artifacts: `results/feature_integration/srcr-candidate-study-v1/`; configuration: `predeclared_config.json`; raw trials: `sr_trials.csv`, `cr_trials.csv`; summaries: `sr_stability.csv`, `sr_agreement.json`, `cr_summary.csv`, `cr_split_responses.csv`; reasons: `cr_reasons.csv`. Test receipt: `results/feature_integration/srcr-candidate-tests-v1.xml`. Runner: `research/srcr_candidate_study.py`. It refuses to overwrite an existing result directory.

No 330/332/334 ablation, LOPO, teacher, student or KD training was run. The study stops at the above recommendations.
