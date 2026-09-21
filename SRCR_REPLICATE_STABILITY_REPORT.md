# SR/CR replicate stability v1

## Decision

**Freeze trailing context as the mapping choice, but do not promote the current single-trial SR/CR pair to a scientifically validated frozen feature set.** SR is seed-sensitive; CR is dominated by stochastic variability and lacks reproducible context ranking. Numerical availability alone is insufficient. No equations, original definitions, canonical columns, existing artifacts or missingness indicators were changed. No LOPO, teacher, student, KD or predictive ablation was run.

## Bounded design

- Same previously selected first/middle/last complete context per recording: 45 recordings, 135 contexts per mapping, 270 in total. Selection uses only recording/window identifiers, never seizure labels. These are diagnostic samples, not an independent confirmation cohort or all 39,522 windows.
- Twenty predeclared seeds, 42–61 inclusive; 5,400 driven and 5,400 matched no-drive trials. Each control uses identical duration, oscillator settings, initial states and stochastic innovations, with external drive set to zero. No labels, trained transformations, thresholds or parameter searches were used.
- Definitions remain the existing D=.5 versus D=0 bistable spectral-ratio difference (SR) and FHN inter-event interval CV (CR). Both indicators remain exactly 1 for missing physical measurements and 0 otherwise.
- Single-replicate rank consistency is Spearman correlation across contexts for every one of the 190 seed pairs per mapping. Value variability is the median across contexts of sample SD across 20 replicates. Split means compare seeds 42–51 against 52–61; these are diagnostic averages, **not** a newly adopted feature definition.
- No inferential significance or patient-independent confidence intervals are claimed: contexts are correlated within recordings, and no-drive FHN trajectories repeat across contexts at each seed.

## Results

| Mapping / quantity | Median within-context SD | SD of context means | Median seed-pair rank correlation [min, max] | Rank correlation of two ten-seed means | Mean absolute difference of ten-seed means |
|---|---:|---:|---:|---:|---:|
| Centered SR, dB | 4.754 | 3.667 | .603 [−.086, .929] | .935 | 1.259 dB |
| Trailing SR, dB | 4.814 | 3.753 | .608 [−.134, .926] | .948 | 1.250 dB |
| Centered CR, CV | .314 | .034 | −.002 [−.239, .425] | .210 | .095 |
| Trailing CR, CV | .307 | .040 | −.005 [−.300, .365] | .235 | .088 |

All 5,400 driven trials have valid SR and CR values; both missingness indicators are zero in this full-context diagnostic sample. This does not remove the previously established recording-boundary missing cases: trailing has 90 missing windows among 39,522, centered 89; all are non-seizure windows. The current study did not recompute full-corpus coverage or use labels.

**No-drive SR:** all 5,400 values are undefined, explicitly retained as NaN with missing indicator 1 and reason `undefined_spectral_ratio`. With zero drive and initial bistable state −1, the D=0 trajectory stays constant, so its local spectral ratio is undefined. The context's original characteristic frequency is retained for the matched assay; no artificial frequency or denominator is substituted. Consequently a driven-minus-no-drive SR difference is **not estimable under the unchanged definition**. This is a control-domain limitation, not evidence of a large SR effect or a reason to silently change the equation. The SR feature's own driven D=0 comparator remains valid.

**No-drive CR:** all 5,400 values are valid. Matched driven-minus-no-drive differences are:

| Mapping | Mean CV difference | Median | Mean absolute difference | Fraction positive |
|---|---:|---:|---:|---:|
| Centered | .0448 | 0 | .0640 | 42.4% |
| Trailing | .0492 | 0 | .0676 | 43.1% |

The mean differences are small relative to within-context replicate SD (~.31), the median difference is zero, and rankings are unstable. Higher CV denotes less regular events; these results do not establish improved coherence or an EEG-specific resonance effect. A single fixed noise dose cannot establish a resonance curve. These quantities are oscillator-derived diagnostics, not inter-channel spectral coherence.

## Feature and context status

**SR: retain as an exploratory candidate, not yet a robust single-trial frozen feature.** Its rank stability is moderate and some seed pairs are negatively correlated. Ten-replicate means are more consistent, but adopting replicate averaging would change the extraction contract and requires a separately agreed validation on disjoint contexts. We did not change the current feature or select a favorable seed.

**CR: do not freeze as a validated ML feature at present.** Near-zero single-seed rank consistency, weak split-mean agreement and the matched no-drive results fail to support reproducible context discrimination. Even ten-replicate averaging does not resolve this diagnostic weakness. Insufficient-trial handling remains unchanged; high validity is not the problem here. The guide-required CR candidate is preserved in the existing 334-column artifact for transparency, without claiming that the requirement establishes its scientific validity.

**Mapping decision: trailing [e−10,e)** for a four-second target [s,e), e=s+4. This includes six seconds preceding the target and adds no EEG samples after its end. Centered [s−3,e+3) is retained only as an offline comparator because it uses three future seconds. Its stability results do not justify adopting that look-ahead. Existing full-recording zero-phase preprocessing is still noncausal; this choice does not establish online inference.

The unchanged candidate contract is: FP1-F7, 2,560 samples at 256 Hz; existing characteristic component/normalization, D=.5, existing oscillator equations, initialization, clipping and seed42; SR local spectral ratio difference in dB; CR population SD/mean of accepted FHN inter-event intervals, requiring at least three intervals; unavailable values NaN plus two binary missingness flags. Canonical 330 plus these four columns remains 334. **Only the context mapping is frozen by this recommendation; no new final feature contract is asserted for both measurements.**

## Validation, provenance and stopping point

The full suite passed **59 tests**, including nine focused SR/CR tests. The runner reproduced all **810 previously computed seed42–44 driven results** to numerical tolerance and checked trial uniqueness and reason/value/mask consistency. Existing protected artifacts were hash-checked before and after execution. The new manifest records exact source hash, environment, seed list, base-manifest identity, protection inventory and output hashes.

New artifacts: `results/feature_integration/srcr-replicate-stability-v1/` (`replicates.csv`, `stability.csv`, `reasons.csv`, `seed_summary.csv`, `manifest.json`). Test receipt: `results/feature_integration/srcr-replicate-tests-v1.xml`. Runner: `research/srcr_replicates.py`; focused tests: `tests/test_srcr_replicates.py`. Reproduction command: `.venv/Scripts/python.exe -B research/srcr_replicates.py`; it refuses to overwrite the existing result directory.

**Readiness for the next predictive ablation:** computationally the prior tables remain usable, but the pair is not scientifically ready as a validated, noise-stable SR/CR feature set. Stop before the 330/332/334 experiment. The next decision is whether to authorize a separately specified SR replicate-average candidate and an investigation of CR drive sensitivity, or retain the unchanged pair solely as explicitly exploratory guide-required features. No new CR equation, dose tuning, feature substitution or additional experiment has been performed here.
