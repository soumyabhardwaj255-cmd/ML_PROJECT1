"""Consolidate verified existing experiments and new cost evidence; no fitting."""
import json,sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eeg_seizure import config as C
from eeg_seizure.artifacts import digest,atomic_json

def main():
    root=C.ROOT;out=root/'results/final/final-evidence-v1'
    m=json.loads((out/'manifest.json').read_text());assert m['status']=='complete'
    for p,h in m['artifacts'].items():assert digest(out/p)==h
    for p,h in m['protected_before'].items():assert digest(root/p)==h,p
    summary=root/'results/contribution/kd-contribution-v1-summary'
    sm=json.loads((summary/'manifest.json').read_text())
    for p,h in sm['artifacts'].items():assert digest(summary/p)==h
    metrics=pd.read_csv(summary/'metrics_by_seed.csv')
    primary=metrics[metrics.seed==42].copy()
    primary['Patient F1 ± SD']=primary.apply(lambda r:f'{r.f1_mean:.4f} ± {r.f1_std:.4f}',axis=1)
    primary=primary[['model','Patient F1 ± SD','precision','recall','f1','average_precision','roc_auc']]
    seed_table=metrics[['model','seed','f1_mean','f1_std','precision','recall','f1','average_precision','roc_auc']]
    patient=pd.read_csv(summary/'per_patient_metrics.csv')
    patient_table=patient.pivot(index=['test_patient','seed'],columns='model',values='f1').reset_index()
    avg=metrics.groupby('model')[['f1_mean','precision','recall','f1','average_precision','roc_auc']].mean().reset_index()
    sizes=pd.read_csv(out/'sizes.csv');timing=pd.read_csv(out/'inference_timing.csv');edf=pd.read_csv(out/'edf_timing.csv');memory=pd.read_csv(out/'process_memory.csv')
    pairs=pd.read_csv(out/'selection_pairs.csv');selection=pd.read_csv(out/'selected_features.csv');freq=pd.read_csv(out/'feature_frequency.csv')
    def table(df):return df.to_markdown(index=False,floatfmt='.4f')
    costs=sizes.groupby('arm')[['bundle_bytes','model_payload_bytes']].agg(['min','max']);costs.columns=['_'.join(c) for c in costs.columns]
    lat=timing.groupby(['arm','scope','batch']).median_ms.agg(['min','max']).reset_index()
    totals=edf.groupby('arm')[['feature_extraction_seconds','end_to_end_seconds']].agg(['min','max']);totals.columns=['_'.join(c) for c in totals.columns]
    report=f'''# Final project report

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

{table(primary)}

Per-patient F1 for every student seed (teacher rows are the same reference):

{table(patient_table)}

Teacher-minus-KD averages −.0286 patient-mean F1 and −.0375 pooled F1 across student seeds. Teacher retains higher patient F1 on chb02/chb08; KD has higher seed-averaged F1 on the other three. Higher teacher ROC-AUC does not imply higher fixed-threshold F1. There is no universal compression penalty or universal student superiority.

## 12. Feature-selection control
Selected-supervised minus random-supervised: seed-averaged paired patient F1 **+.0315**, patient SD .0475, range −.0448 to +.0846; four of five patient averages positive. However, pooled F1 is lower in every seed, averaging **−.0339**. Selection improves some patient-mean/ranking/sensitivity measures at a precision cost; a blanket superiority claim is unsupported. Only one random subset per fold is tested.

## 13. Multi-seed reproducibility and KD effect
Full per-seed results: f1_mean/f1_std are across patients; precision/recall/f1/AP/AUC are pooled windows.

{table(seed_table)}

Average of the three seeds' metrics (not concatenated predictions):

{table(avg)}

KD minus selected-supervised: patient-mean F1 changes **+.0513,+.0469,−.0446** for seeds42/43/44. Seed-averaged paired patient gain **+.0179**, patient SD .0552; pooled F1 improves in all three seeds, mean **+.0659**, but recall decreases .0481 and AP decreases .0157 on average. Mean patient F1 across seeds is .3973±.0238 for supervised and .4152±.0313 for KD, where these SDs are **seed SDs**, not patient SDs. Evidence supports a seed-sensitive operating-point trade-off, not universally improved detection/ranking.

## 14. Lightweightness measurements
New CPU measurements use the final seed42 bundles and the first recording of each held-out patient; no retraining. Recordings have900–902 complete windows/3,600–3,608seconds of windowed EEG. Network timing uses100 warm repeats; feature-table API uses30 repeats. Single EDF passes include load/preprocessing, all330 feature extraction and inference with an already-loaded bundle. Imports/model-disk loading are excluded; caches/host activity are not controlled and validation/test activity overlapped part of the benchmark. Timing is a descriptive single-host benchmark, not a device-independent latency guarantee.

Serialized sizes in bytes (full joblib bundles include schemas/transforms/metadata; payload formats differ):

{table(costs.reset_index())}

Ranges across patients of median milliseconds per call; batch is the number of windows, not milliseconds per window:

{table(lat)}

EDF timing ranges in seconds:

{table(totals.reset_index())}

Whole-process memory snapshots, bytes:

{table(memory)}

RSS includes Python/libraries, loaded EEG, features and models; it is not isolated model memory or a measured peak. Network size/timing supports a **compact classifier** claim. Inference still computes all330 features, reads/filters a complete recording, and is noncausal; **end-to-end lightweight detector status is not established**, even if offline processing is faster than recording duration. No energy, embedded-device or clinical real-time measurement was made.

## 15. Interpretability and selection stability
Ten fold-pair Jaccards: mean {pairs.jaccard.mean():.4f}, range {pairs.jaccard.min():.4f}–{pairs.jaccard.max():.4f}; {len(freq)} unique features among250 fold selections, {int((freq.fold_count==5).sum())} selected in all five folds. Overlapping training sets limit independence.

Feature-family counts across250 selections:

{table(selection.groupby('family').size().reset_index(name='selections'))}

Channel counts:

{table(selection.groupby('channel').size().reset_index(name='selections'))}

Frequency-band counts among selected frequency features only:

{table(selection[selection.family=='freq'].groupby('band').size().reset_index(name='selections'))}

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
'''
    destination=root/'FINAL_PROJECT_REPORT.md'
    if destination.exists():raise ValueError('Final report already exists; review before updating')
    destination.write_text(report,encoding='utf-8')
    atomic_json(root/'results/final/final-release-v1.json',dict(status='complete',canonical_features=330,student_seeds=[42,43,44],
        primary_validation='passed this stage:20 bundles',replication_validation='passed this stage:40 bundles',tests='69 passed, exit0; native import diagnostics recorded in report',
        report_sha256=digest(destination),sources={str(p.relative_to(root)):digest(p) for p in [out/'manifest.json',summary/'manifest.json',root/'results/contribution/kd-contribution-v1-seed42/manifest.json',root/'results/contribution/kd-contribution-v1-seeds43-44/manifest.json']},
        supporting_artifact_hashes_verified=True,model_training_this_stage=False))
    print(destination)

if __name__=='__main__':main()
