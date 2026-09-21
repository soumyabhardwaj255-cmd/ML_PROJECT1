"""Final frozen-bundle costs and teacher-selection description; no training."""
import json,time,itertools,io,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch,mne,joblib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eeg_seizure import config as C
from eeg_seizure.artifacts import digest,atomic_csv,atomic_json,environment
from eeg_seizure.dataset import load_dataset,feature_rows,window_rows
from eeg_seizure.inference import load_bundle,predict_features
from eeg_seizure.modeling import student_network,student_scores
from eeg_seizure.channels import dedupe_channel_names
from eeg_seizure.preprocessing import preprocess_raw

OUT=C.ROOT/'results/final/final-evidence-v1'
PRIMARY=C.ROOT/'results/contribution/kd-contribution-v1-seed42'

def timed(call,n):
    call();samples=[]
    for _ in range(n):
        start=time.perf_counter();call();samples.append(time.perf_counter()-start)
    return dict(median_ms=float(np.median(samples)*1000),p95_ms=float(np.percentile(samples,95)*1000),repeats=n)

def main():
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    protected={str(p.relative_to(C.ROOT)):digest(p) for root in ['src','data/processed/corrected_v2','results','research'] for p in (C.ROOT/root).rglob('*') if p.is_file() and '__pycache__' not in p.parts and OUT not in p.parents}
    frame,dm=load_dataset();assert len(dm['spec']['feature_columns'])==330
    manifest=dict(status='running',environment=environment(),source_sha256=digest(__file__),protected_before=protected,
        design='seed42 frozen teacher and both selected students; first recording per held-out patient; warm CPU classifier repeats100; EDF pipeline once per patient; no training',
        limitations='single host; warm caches; no energy or causal latency claim; RSS is whole-process resident memory, not isolated model peak')
    atomic_json(OUT/'manifest.json',manifest)
    selections=[];sizes=[];latencies=[];edf=[];memory=[];sets={}
    try:
        import psutil
        process=psutil.Process()
    except ImportError: process=None
    for patient in C.PATIENTS:
        folder=PRIMARY/'seed42'/patient
        selected=load_bundle(folder/'selected_kd.joblib')['selected_features'];sets[patient]=set(selected)
        for f in selected:
            family,measure,channel=f.split('__')
            selections.append(dict(patient=patient,feature=f,family=family,measure=measure,channel=channel,band=measure if family=='freq' else 'not_applicable'))
        record=sorted(frame[frame.patient==patient].file.unique())[0];path=C.RAW_DIR/patient/record
        original=frame[(frame.patient==patient)&(frame.file==record)].reset_index(drop=True)
        rec=next(r for r in dm['records'] if r['patient']==patient and r['file']==record);assert digest(path)==rec['raw_sha256']
        rss_before=process.memory_info().rss if process else None
        start=time.perf_counter()
        with mne.io.read_raw_edf(path,preload=True,verbose='ERROR') as raw:
            raw=preprocess_raw(dedupe_channel_names(raw),dm['spec']['channels'],**dm['spec']['preprocessing'])
            prep=time.perf_counter()-start
            windows=window_rows(raw.n_times,256,[],patient,record)
            begin=time.perf_counter();features=feature_rows(raw,windows,dm['spec']['channels']);extraction=time.perf_counter()-begin
        np.testing.assert_allclose(features[dm['spec']['feature_columns']],original[dm['spec']['feature_columns']],rtol=1e-5,atol=1e-20)
        for arm in ['teacher','selected_supervised','selected_kd']:
            bundle_path=folder/(arm+'.joblib');bundle=load_bundle(bundle_path)
            begin=time.perf_counter();scores,pred=predict_features(bundle,features);inference=time.perf_counter()-begin
            ref,refpred=predict_features(bundle,original)
            np.testing.assert_allclose(scores,ref,rtol=1e-5,atol=1e-6);np.testing.assert_array_equal(pred,refpred)
            edf.append(dict(patient=patient,recording=record,arm=arm,windows=len(features),eeg_seconds=len(features)*4,
                read_preprocess_seconds=prep,feature_extraction_seconds=extraction,inference_seconds=inference,
                end_to_end_seconds=prep+extraction+inference))
            if arm!='teacher':
                model=student_network(50);model.load_state_dict(bundle['state_dict']);model.eval()
                count=sum(p.numel() for p in model.parameters());assert count==2177
                payload=io.BytesIO();torch.save(bundle['state_dict'],payload);raw_bytes=len(payload.getvalue())
                z=bundle['transform'].transform(features[bundle['selected_features']])
                for batch in [1,min(256,len(z))]:
                    t=timed(lambda:student_scores(model,z[:batch]),100)
                    latencies.append(dict(patient=patient,arm=arm,scope='resident_network_including_tensor_conversion',batch=batch,**t))
            else:
                count=None;raw_bytes=len(bundle['estimator'].named_steps['model'].get_booster().save_raw(raw_format='ubj'))
            sizes.append(dict(patient=patient,arm=arm,parameters=count,bundle_bytes=bundle_path.stat().st_size,model_payload_bytes=raw_bytes,
                payload_format='torch_state_dict' if arm!='teacher' else 'xgboost_ubj'))
            for batch in [1,min(256,len(features))]:
                t=timed(lambda:predict_features(bundle,features.iloc[:batch]),30)
                latencies.append(dict(patient=patient,arm=arm,scope='feature_table_API_including_transform_and_student_reconstruction',batch=batch,**t))
        memory.append(dict(patient=patient,rss_before_bytes=rss_before,rss_after_bytes=process.memory_info().rss if process else None))
        print(f'Final benchmark {patient}/{record}: extraction {extraction:.2f}s, {len(features)} windows',flush=True)
    selection=pd.DataFrame(selections)
    pairs=pd.DataFrame([dict(patient_a=a,patient_b=b,intersection=len(sets[a]&sets[b]),jaccard=len(sets[a]&sets[b])/len(sets[a]|sets[b])) for a,b in itertools.combinations(sets,2)])
    tables={'selected_features':selection,'selection_pairs':pairs,'feature_frequency':selection.groupby('feature').size().reset_index(name='fold_count'),
        'sizes':pd.DataFrame(sizes),'inference_timing':pd.DataFrame(latencies),'edf_timing':pd.DataFrame(edf),'process_memory':pd.DataFrame(memory)}
    for axis in ['family','channel','band']:
        tables[axis+'_distribution']=selection.groupby(['patient',axis]).size().reset_index(name='count')
    for name,table in tables.items():atomic_csv(OUT/(name+'.csv'),table)
    for p,h in protected.items():assert digest(C.ROOT/p)==h,p
    manifest.update(status='complete',protected_files_unchanged=len(protected),edf_feature_and_decision_parity=True,
        student_specific_faithfulness='not established; no existing implemented method; not inferred from teacher importance',
        artifacts={p.name:digest(p) for p in OUT.iterdir() if p.name!='manifest.json'})
    atomic_json(OUT/'manifest.json',manifest)
    print(pd.DataFrame(sizes).groupby('arm')[['bundle_bytes','model_payload_bytes']].agg(['min','max']).to_string())
    print(pairs.to_string(index=False))

if __name__=='__main__':main()
