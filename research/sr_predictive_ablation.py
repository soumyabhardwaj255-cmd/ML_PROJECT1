"""Frozen SR mean10 extraction and matched LR representation ablation, v1."""
import argparse
import json
import time
from pathlib import Path
import joblib
import mne
import numpy as np
import pandas as pd
import srcr_features as F
from eeg_seizure.artifacts import digest,atomic_json,atomic_csv,environment
from eeg_seizure.dataset import load_dataset,read_table,validated_annotations,window_rows
from eeg_seizure.evaluation import patient_split,tune_threshold,metrics,summarize
from eeg_seizure.modeling import make_model

C=F.C
VALUE='sr__mean10_snr_change_vs_zero_noise_db__FP1-F7__ctx10'
MASK=VALUE+'__missing'
SEEDS=tuple(range(62,72))
NOISE=tuple(F.innovations(seed) for seed in SEEDS)
DATA=C.ROOT/'results/feature_integration/sr-mean10-full-v1'
RUN=C.ROOT/'results/ablations/sr-mean10-lr-v1'
SPEC=dict(version='sr-mean10-trailing-v1',feature=VALUE,missingness=MASK,seeds=list(SEEDS),
    target_sec=4,target_samples=1024,context_sec=10,context_samples=2560,context='[e-10,e)',channel='FP1-F7',
    noise_intensities=[.5,0.],aggregation='arithmetic mean of ten dB differences; any invalid trial makes mean missing',
    component='existing characteristic_component and normalization',bistable='existing .2 drive, initial -1, dt1/256, clip3, one substep',
    boundary='strict same-record full context; no padding or shifting',cr='excluded',future='no context samples after e; canonical preprocessing remains noncausal')


def protection():
    roots=['src','research','tests','notebooks','data/processed/corrected_v2','results']
    files=[p for root in roots for p in (C.ROOT/root).rglob('*') if p.is_file() and '__pycache__' not in p.parts and DATA not in p.parents and RUN not in p.parents]
    files.extend(C.ROOT.glob('*.md'))
    return {str(p.relative_to(C.ROOT)):digest(p) for p in files}


def check_protected(before):
    for name,h in before.items(): assert digest(C.ROOT/name)==h,name


def sr_mean(signal):
    """No labels/identifiers. Reuse component and D=0 trajectory across seeds."""
    x=np.asarray(signal,dtype=float)
    if x.shape!=(2560,) or not np.isfinite(x).all() or x.std()==0:
        return dict(value=np.nan,missing=1,reason='invalid_or_constant_signal',valid_trials=0)
    drive,freq=F.characteristic_component(x,256)
    quiet,clip0=F.oscillator(drive,NOISE[0],0.,0)
    baseline=F.spectral_snr(quiet,256,freq)
    values=[]; reasons=[]
    for noise in NOISE:
        response,clip=F.oscillator(drive,noise,.5,0)
        value=F.spectral_snr(response,256,freq)-baseline
        reason='clipped_dynamics' if max(clip,clip0)>0 else 'ok' if np.isfinite(value) else 'undefined_spectral_ratio'
        values.append(value if reason=='ok' else np.nan);reasons.append(reason)
    valid=int(np.isfinite(values).sum())
    reason=next((r for r in reasons if r!='ok'),'ok')
    return dict(value=float(np.mean(values)) if valid==10 else np.nan,missing=int(valid!=10),reason=reason,valid_trials=valid)


def extended_table(canonical,extra):
    if list(extra.columns)!=C.KEYS+[VALUE,MASK]: raise ValueError('Explicit SR-only schema required')
    if len(extra)!=len(canonical) or extra.duplicated(C.KEYS).any(): raise ValueError('Incomplete/duplicate SR rows')
    if not np.array_equal(extra[MASK].to_numpy(),extra[VALUE].isna().astype(int).to_numpy()): raise ValueError('Bad missingness mask')
    result=canonical.merge(extra,on=C.KEYS,validate='one_to_one',sort=False)
    pd.testing.assert_frame_equal(result[canonical.columns],canonical,check_exact=True)
    return result


def extract():
    DATA.mkdir(parents=True,exist_ok=False)
    before=protection();frame,dm=load_dataset();start=time.perf_counter()
    manifest=dict(status='running',spec=SPEC,source_sha256=digest(__file__),environment=environment(),
        canonical_dataset_sha256=dm['feature_sha256'],protected_before=before,labels='used only for target verification and coverage; never extraction inputs')
    atomic_json(DATA/'manifest.json',manifest)
    rows=[];parities=0
    for i,rec in enumerate(dm['records']):
        patient,file=rec['patient'],rec['file'];path=C.RAW_DIR/patient/file
        annotation=path.parent/f'{patient}-summary.txt'
        assert digest(path)==rec['raw_sha256'] and digest(annotation)==rec['summary_sha256']
        subset=frame[(frame.patient==patient)&(frame.file==file)]
        record=[]
        with mne.io.read_raw_edf(path,preload=True,verbose='ERROR') as raw:
            assert raw.info['sfreq']==256
            expected=window_rows(raw.n_times,256,validated_annotations(annotation,file,raw.n_times/256),patient,file)
            pd.testing.assert_frame_equal(subset[C.META].reset_index(drop=True),expected,check_dtype=False)
            raw=F.preprocess_raw(F.dedupe_channel_names(raw),dm['spec']['channels'],**dm['spec']['preprocessing'])
            signal=raw.get_data(picks=[F.CHANNEL])[0]
            parity_done=False
            for key in subset[C.KEYS].to_dict('records'):
                a,b,reason=F.context_bounds(key['start_sec'],key['end_sec'],len(signal),'trailing')
                assert b==round(key['end_sec']*256) and b-a==2560
                value=sr_mean(signal[a:b]) if reason=='ok' else dict(value=np.nan,missing=1,reason=reason,valid_trials=0)
                if reason=='ok' and not parity_done:
                    reference=[F.measurements(signal[a:b],seed)['sr_value'] for seed in SEEDS]
                    expected_mean=np.mean(reference) if np.isfinite(reference).all() else np.nan
                    np.testing.assert_allclose(value['value'],expected_mean,rtol=1e-10,atol=1e-10,equal_nan=True)
                    np.testing.assert_equal(value,sr_mean(signal[a:b]));parities+=1;parity_done=True
                record.append(dict(key,context_start_sec=a/256,context_end_sec=b/256,**value))
        table=pd.DataFrame(record);rows.append(table)
        atomic_csv(DATA/f'checkpoint_{patient}_{file}.csv',table)
        print(f'SR extraction {i+1}/45: {patient}/{file}, {time.perf_counter()-start:.1f}s',flush=True)
    diagnostics=pd.concat(rows,ignore_index=True)
    extra=diagnostics[C.KEYS+['value','missing']].rename(columns={'value':VALUE,'missing':MASK})
    extended=extended_table(frame,extra)
    assert len(extended)==39522 and len(dm['spec']['feature_columns'])==330
    atomic_csv(DATA/'extended_features.csv',extended)
    reloaded=read_table(DATA/'extended_features.csv')
    pd.testing.assert_frame_equal(reloaded,extended,check_exact=True)
    atomic_csv(DATA/'diagnostics.csv',diagnostics)
    labeled=diagnostics.merge(frame[C.META],on=C.KEYS,validate='one_to_one')
    coverage=labeled.groupby('label').agg(windows=('value','size'),valid=('value','count'),missing=('missing','sum')).reset_index()
    atomic_csv(DATA/'coverage.csv',coverage)
    atomic_csv(DATA/'missing_reasons.csv',labeled.groupby(['label','reason']).size().reset_index(name='windows'))
    atomic_csv(DATA/'patient_coverage.csv',labeled.groupby(['patient','label']).agg(windows=('value','size'),valid=('value','count'),missing=('missing','sum')).reset_index())
    atomic_json(DATA/'schema.json',dict(feature_columns=dm['spec']['feature_columns']+[VALUE,MASK],spec=SPEC))
    # Compare against every predeclared candidate context/seed group already measured.
    prior=read_table(C.ROOT/'results/feature_integration/srcr-candidate-study-v1/sr_trials.csv')
    prior=prior[prior.seed.isin(SEEDS)].groupby(C.KEYS).value.agg(lambda v:float(v.mean()) if v.notna().all() and len(v)==10 else np.nan).reset_index(name='expected')
    matched=diagnostics.merge(prior,on=C.KEYS,validate='one_to_one'); assert len(matched)==90
    np.testing.assert_allclose(matched.value,matched.expected,rtol=1e-10,atol=1e-10,equal_nan=True)
    check_protected(before)
    manifest.update(status='complete',windows=len(frame),dimensions=332,canonical_columns_exact=True,parity_recordings=parities,
        previous_candidate_contexts=90,seconds=time.perf_counter()-start,protected_files_unchanged=len(before),
        artifacts={p.name:digest(p) for p in DATA.iterdir() if p.name!='manifest.json'})
    atomic_json(DATA/'manifest.json',manifest);print(coverage.to_string(index=False))


def load_extended():
    m=json.loads((DATA/'manifest.json').read_text());assert m['status']=='complete' and m['spec']==SPEC
    for name,h in m['artifacts'].items(): assert digest(DATA/name)==h,name
    canonical,dm=load_dataset();assert m['canonical_dataset_sha256']==dm['feature_sha256']
    table=read_table(DATA/'extended_features.csv')
    pd.testing.assert_frame_equal(table[canonical.columns],canonical,check_exact=True)
    assert list(table.columns)==list(canonical.columns)+[VALUE,MASK]
    assert (table[MASK]==table[VALUE].isna().astype(int)).all()
    assert not np.isinf(table[VALUE]).any()
    return table,dm


def evaluate():
    frame,dm=load_extended();RUN.mkdir(parents=True,exist_ok=False);before=protection()
    # Include the newly frozen extraction in evaluation's protection inventory.
    before.update({str(p.relative_to(C.ROOT)):digest(p) for p in DATA.iterdir() if p.is_file()})
    schemas={'canonical330':dm['spec']['feature_columns'],'sr332':dm['spec']['feature_columns']+[VALUE,MASK]}
    manifest=dict(status='running',source_sha256=digest(__file__),environment=environment(),protected_before=before,
        extraction_manifest_sha256=digest(DATA/'manifest.json'),schemas=schemas,seed=C.SEED,
        model='logistic_regression',parameters=C.LR_PARAMS,threshold_grid=C.THRESHOLDS,
        protocol='five outer patient-exclusive folds; four inner patient folds select threshold by pooled inner F1; all transforms refitted on training only',
        feature_selection='none',teacher_student_kd='none',sr_spec=SPEC)
    atomic_json(RUN/'manifest.json',manifest)
    predictions=[];rows=[];audits=[];coefficients=[]
    for arm,columns in schemas.items():
        for patient in C.PATIENTS:
            train,test=patient_split(frame,patient)
            threshold,audit,oof=tune_threshold(train,columns,'logistic_regression',C.SEED)
            for fold in audit:
                assert patient not in fold['train_patients'] and patient!=fold['validation_patient']
                audits.append(dict(arm=arm,outer_patient=patient,**fold))
            atomic_csv(RUN/f'inner_{arm}_{patient}.csv',oof)
            model=make_model('logistic_regression',train.label,C.SEED);model.fit(train[columns],train.label)
            transform=model.named_steps['features']
            np.testing.assert_allclose(transform.imputer_.statistics_,np.nanmedian(train[columns].to_numpy(),axis=0),rtol=0,atol=0)
            scores=model.predict_proba(test[columns])[:,1]
            bundle=dict(format_version='sr-ablation-v1',arm=arm,feature_columns=columns,estimator=model,
                train_patients=sorted(train.patient.unique()),test_patient=patient,threshold=threshold,seed=C.SEED,
                sr_spec=SPEC if arm=='sr332' else None,dataset_sha256=dm['feature_sha256'],extraction_manifest_sha256=manifest['extraction_manifest_sha256'])
            path=RUN/f'bundle_{arm}_{patient}.joblib';joblib.dump(bundle,path)
            np.testing.assert_array_equal(scores,joblib.load(path)['estimator'].predict_proba(test[columns])[:,1])
            rows.append(dict(metrics(test.label,scores,threshold),model=arm,seed=C.SEED,test_patient=patient,threshold=threshold))
            predictions.append(test[C.META].assign(model=arm,seed=C.SEED,threshold=threshold,probability_seizure=scores,prediction=(scores>=threshold).astype(int)))
            coefficients.append(pd.DataFrame(dict(feature=columns,coefficient=model.named_steps['model'].coef_[0])).assign(model=arm,test_patient=patient))
            print(f'{arm}/{patient}: F1={rows[-1]["f1"]:.6f}, AP={rows[-1]["average_precision"]:.6f}, threshold={threshold}',flush=True)
    pred=pd.concat(predictions,ignore_index=True);per=pd.DataFrame(rows);means,pooled=summarize(pred,per)
    a=per[per.model=='canonical330'].set_index('test_patient');b=per[per.model=='sr332'].set_index('test_patient')
    delta=(b[['f1','precision','recall','average_precision','roc_auc']]-a[['f1','precision','recall','average_precision','roc_auc']]).reset_index()
    paired=pred[pred.model=='canonical330'].merge(pred[pred.model=='sr332'],on=C.META,validate='one_to_one',suffixes=('_baseline','_sr'))
    behavior=paired.groupby('patient').apply(lambda g:pd.Series(dict(changed_predictions=int((g.prediction_baseline!=g.prediction_sr).sum()),mean_abs_probability_change=(g.probability_seizure_sr-g.probability_seizure_baseline).abs().mean())),include_groups=False).reset_index()
    for name,table in [('predictions',pred),('per_patient_metrics',per),('mean_patient_metrics',means),('pooled_window_metrics',pooled),('paired_patient_deltas',delta),('model_behavior',behavior),('standardized_coefficients',pd.concat(coefficients,ignore_index=True))]:atomic_csv(RUN/(name+'.csv'),table)
    atomic_json(RUN/'inner_fold_audit.json',audits)
    check_protected(before)
    manifest.update(status='complete',fits=50,protected_files_unchanged=len(before),bundle_reload_parity=True,
        artifacts={p.name:digest(p) for p in RUN.iterdir() if p.name!='manifest.json'})
    atomic_json(RUN/'manifest.json',manifest)
    print(means.to_string(index=False));print(pooled.to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['extract','evaluate']);args=parser.parse_args()
    extract() if args.stage=='extract' else evaluate()
