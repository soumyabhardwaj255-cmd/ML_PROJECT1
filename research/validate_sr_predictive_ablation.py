"""Independent saved-artifact validation; no training or canonical writes."""
import json
import numpy as np
import pandas as pd
import mne
import joblib
from sklearn.metrics import f1_score
import sr_predictive_ablation as S
from eeg_seizure.artifacts import digest,atomic_json
from eeg_seizure.dataset import read_table
from eeg_seizure.evaluation import metrics,summarize


def main():
    frame,dm=S.load_extended()
    data_manifest=json.loads((S.DATA/'manifest.json').read_text())
    run_manifest=json.loads((S.RUN/'manifest.json').read_text())
    assert run_manifest['status']=='complete'
    for root,m in [(S.DATA,data_manifest),(S.RUN,run_manifest)]:
        assert m['source_sha256']==digest(S.__file__)
        S.check_protected(m['protected_before'])
        for name,h in m['artifacts'].items(): assert digest(root/name)==h,name
    diag=read_table(S.DATA/'diagnostics.csv')
    pd.testing.assert_frame_equal(diag[S.C.KEYS],frame[S.C.KEYS],check_exact=True)
    np.testing.assert_array_equal(diag.value,frame[S.VALUE]);np.testing.assert_array_equal(diag.missing,frame[S.MASK])
    for rec in dm['records']:
        with mne.io.read_raw_edf(S.C.RAW_DIR/rec['patient']/rec['file'],preload=False,verbose='ERROR') as raw:
            size=raw.n_times
        for row in diag[(diag.patient==rec['patient'])&(diag.file==rec['file'])].itertuples():
            a,b,reason=S.F.context_bounds(row.start_sec,row.end_sec,size,'trailing')
            assert a/256==row.context_start_sec and b/256==row.context_end_sec
            if reason!='ok': assert row.reason==reason and row.missing==1 and np.isnan(row.value)
            if not row.missing: assert row.valid_trials==10
    coverage=diag.merge(frame[S.C.META],on=S.C.KEYS).groupby('label').agg(windows=('value','size'),valid=('value','count'),missing=('missing','sum')).reset_index()
    pd.testing.assert_frame_equal(coverage,read_table(S.DATA/'coverage.csv'))
    predictions=read_table(S.RUN/'predictions.csv');per=read_table(S.RUN/'per_patient_metrics.csv')
    assert len(predictions)==2*len(frame) and not predictions.duplicated(['model']+S.C.KEYS).any()
    assert set(predictions.model)=={'canonical330','sr332'}
    audit=json.loads((S.RUN/'inner_fold_audit.json').read_text());assert len(audit)==40
    for fold in audit:
        patients=set(S.C.PATIENTS)-{fold['outer_patient'],fold['validation_patient']}
        assert len(patients)==3 and set(fold['train_patients'])==patients
        inner=frame[frame.patient.isin(patients)]
        assert fold['training_windows']==len(inner) and fold['training_positive']==int(inner.label.sum())
        assert fold['class_ratio']==float((inner.label==0).sum()/(inner.label==1).sum())
    for arm,columns in run_manifest['schemas'].items():
        assert columns==dm['spec']['feature_columns']+([S.VALUE,S.MASK] if arm=='sr332' else [])
        for patient in S.C.PATIENTS:
            train,test=S.patient_split(frame,patient)
            bundle=joblib.load(S.RUN/f'bundle_{arm}_{patient}.joblib')
            assert bundle['test_patient']==patient and set(bundle['train_patients'])==set(S.C.PATIENTS)-{patient}
            assert bundle['feature_columns']==columns and bundle['seed']==42
            model=bundle['estimator'];transform=model.named_steps['features']
            expected=np.nanmedian(train[columns].to_numpy(),axis=0)
            np.testing.assert_array_equal(transform.imputer_.statistics_,expected)
            x=train[columns].to_numpy();x=np.where(np.isnan(x),expected,x)
            np.testing.assert_allclose(transform.scaler_.mean_,x.mean(axis=0),rtol=1e-12,atol=1e-20)
            np.testing.assert_allclose(transform.scaler_.var_,x.var(axis=0),rtol=1e-10,atol=1e-25)
            group=predictions[(predictions.model==arm)&(predictions.patient==patient)].reset_index(drop=True)
            pd.testing.assert_frame_equal(group[S.C.META],test[S.C.META].reset_index(drop=True))
            scores=model.predict_proba(test[columns])[:,1]
            np.testing.assert_array_equal(scores,group.probability_seizure)
            np.testing.assert_array_equal(scores>=bundle['threshold'],group.prediction)
            inner=read_table(S.RUN/f'inner_{arm}_{patient}.csv')
            pd.testing.assert_frame_equal(inner[S.C.META],train[S.C.META].reset_index(drop=True))
            thresholds=[f1_score(inner.label,inner.probability_seizure>=t,zero_division=0) for t in S.C.THRESHOLDS]
            assert bundle['threshold']==S.C.THRESHOLDS[int(np.argmax(thresholds))]
            actual=metrics(test.label,scores,bundle['threshold']);stored=per[(per.model==arm)&(per.test_patient==patient)].iloc[0]
            for key,value in actual.items():np.testing.assert_allclose(value,stored[key],rtol=1e-12,atol=1e-15)
    means,pooled=summarize(predictions,per)
    for name,actual in [('mean_patient_metrics',means),('pooled_window_metrics',pooled)]:
        pd.testing.assert_frame_equal(actual,read_table(S.RUN/(name+'.csv')),check_exact=False,rtol=1e-12,atol=1e-15)
    delta=read_table(S.RUN/'paired_patient_deltas.csv').set_index('test_patient')
    expected=per[per.model=='sr332'].set_index('test_patient')[delta.columns]-per[per.model=='canonical330'].set_index('test_patient')[delta.columns]
    pd.testing.assert_frame_equal(delta,expected,check_exact=False,rtol=1e-12,atol=1e-15)
    # Same canonical LR protocol as prior PSD ablation: independent baseline parity.
    old=read_table(S.C.ROOT/'results/ablations/psd-representation-v1/predictions.csv')
    old=old[old.model=='canonical_absolute'];new=predictions[predictions.model=='canonical330']
    pd.testing.assert_frame_equal(old[S.C.META+['probability_seizure','prediction','threshold']].reset_index(drop=True),
        new[S.C.META+['probability_seizure','prediction','threshold']].reset_index(drop=True),check_exact=True)
    receipt=S.RUN.parent/'sr-mean10-lr-v1-validation.json'
    if receipt.exists(): raise ValueError('Do not overwrite validation receipt')
    atomic_json(receipt,dict(validated=True,validator_sha256=digest(__file__),extraction_manifest_sha256=digest(S.DATA/'manifest.json'),
        experiment_manifest_sha256=digest(S.RUN/'manifest.json'),bundles=10,inner_folds=40,canonical_baseline_exact_parity=True,
        mapping_rows=len(diag),protected_files=run_manifest['protected_files_unchanged']))
    print(per[['model','test_patient','threshold','f1','average_precision']].to_string(index=False))
    print(means.to_string(index=False));print(pooled.to_string(index=False));print('Validation passed')


if __name__=='__main__':main()
