"""Predeclared SR replicate-average and FHN drive sensitivity diagnostics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import srcr_features as F
from eeg_seizure.artifacts import digest, atomic_csv, atomic_json, environment

OUT=F.C.ROOT/'results/feature_integration/srcr-candidate-study-v1'
BASE=OUT.parent/'srcr-context-features-v1'
SR_SEEDS=tuple(range(62,102))
CR_SEEDS=tuple(range(62,82))
SCALES=(0.,.175,.35,.7)
DOSES=(.25,.5,1.)
CONFIG=dict(version='srcr-candidate-study-v1',context='trailing',channel=F.CHANNEL,
    selection='nearest quarter and three-quarter full contexts per recording, excluding overlap with previous diagnostic contexts in either mapping',
    sr_seeds=list(SR_SEEDS),sr_blocks=[list(SR_SEEDS[i:i+10]) for i in range(0,40,10)],
    candidate_seeds=list(SR_SEEDS[:10]),current_single_seed=42,cr_seeds=list(CR_SEEDS),
    cr_input_scales=list(SCALES),cr_noise_intensities=list(DOSES),
    rationale='half/default/double original drive and noise; zero-drive control; no adaptive grid',
    sr_missing='aggregate NaN if any of the ten replicates is unavailable; no selective averaging',
    aggregation='arithmetic mean of existing per-trial dB SR values, not a ratio of averaged spectra',
    fitting='none',labels='never loaded',acceptance='descriptive stability evidence; no label-derived threshold or best-cell selection')


def strict_mean(values):
    values=np.asarray(values,dtype=float)
    return float(values.mean()) if np.isfinite(values).all() else np.nan


def cr_trial(drive,scale,dose,seed):
    # Existing evaluator has coefficient .35: scaling its input gives precisely
    # the existing fhn(input_scale=scale) equation, without other changes.
    response,clip=F.oscillator(drive*(scale/.35),F.innovations(seed),dose,1)
    r=F.regularity(F.fhn_events(response,256),256,minimum_intervals=3,ddof=0,minimum_interval_sec=.2)
    reason='clipped_dynamics' if clip>0 else r['reason']
    return dict(value=r['cv'] if reason=='ok' else np.nan,reason=reason,
                missing=int(reason!='ok'),events=r['events'],intervals=r['intervals'],clipped=clip)


def select_contexts():
    keys=pd.read_csv(BASE/'diagnostics.csv',usecols=F.C.KEYS+['mode','context_start_sec','context_end_sec','context_reason'])
    old=pd.read_csv(BASE/'seed_sensitivity.csv',usecols=['patient','file','start_sec','mode']).drop_duplicates()
    selected=[]
    for (patient,file),g in keys[keys['mode']=='trailing'].groupby(['patient','file'],sort=True):
        g=g[g.context_reason=='ok'].sort_values('start_sec')
        prior=old[(old.patient==patient)&(old.file==file)]
        prohibited=[]
        for row in prior.itertuples():
            prohibited.append((row.start_sec-6,row.start_sec+4) if row.mode=='trailing' else (row.start_sec-3,row.start_sec+7))
        for fraction in (.25,.75):
            target=g.iloc[round((len(g)-1)*fraction)].start_sec
            ordered=g.assign(distance=(g.start_sec-target).abs()).sort_values(['distance','start_sec'])
            for row in ordered.itertuples():
                a,b=row.context_start_sec,row.context_end_sec
                if all(b<=x or a>=y for x,y in prohibited):
                    selected.append(dict(patient=patient,file=file,start_sec=row.start_sec,end_sec=row.end_sec,context_start_sec=a,context_end_sec=b))
                    prohibited.append((a,b)); break
            else: raise AssertionError('No disjoint context available')
    result=pd.DataFrame(selected)
    assert len(result)==90 and not result.duplicated(F.C.KEYS).any()
    return result


def ranks(pivot):
    corr=pivot.corr(method='spearman').to_numpy()
    pairs=corr[np.triu_indices(corr.shape[0],1)]
    return dict(rank_median=float(np.nanmedian(pairs)),rank_min=float(np.nanmin(pairs)),rank_max=float(np.nanmax(pairs)),
                median_within_sd=float(pivot.std(axis=1).median()))


def summarize(sr,cr):
    pivot=sr[sr.seed.isin(SR_SEEDS)].pivot(index=F.C.KEYS,columns='seed',values='value')
    blocks=pd.DataFrame({i:pivot[list(SR_SEEDS[i*10:(i+1)*10])].apply(strict_mean,axis=1) for i in range(4)})
    single42=sr[sr.seed==42].set_index(F.C.KEYS).value
    sr_summary=[dict(representation='single_trial_40_seeds',**ranks(pivot)),dict(representation='ten_replicate_mean_4_blocks',**ranks(blocks))]
    details=dict(split_first_two_blocks_rank=blocks[[0,1]].corr(method='spearman').iloc[0,1],
        split_first_two_blocks_mae=(blocks[0]-blocks[1]).abs().mean(),
        candidate_vs_other30_rank=pd.concat([blocks[0],blocks[[1,2,3]].mean(axis=1)],axis=1).corr(method='spearman').iloc[0,1],
        current42_vs_other30_rank=pd.concat([single42,blocks[[1,2,3]].mean(axis=1)],axis=1).corr(method='spearman').iloc[0,1],
        current42_vs_other30_mae=(single42-blocks[[1,2,3]].mean(axis=1)).abs().mean(),
        candidate_vs_other30_mae=(blocks[0]-blocks[[1,2,3]].mean(axis=1)).abs().mean())
    summaries=[]; split_effects=[]
    for dose in DOSES:
        null=cr[(cr.dose==dose)&(cr.scale==0)].pivot(index=F.C.KEYS,columns='seed',values='value')
        for scale in SCALES:
            g=cr[(cr.dose==dose)&(cr.scale==scale)]
            p=g.pivot(index=F.C.KEYS,columns='seed',values='value')
            split=pd.DataFrame({i:p[list(CR_SEEDS[i*10:(i+1)*10])].apply(strict_mean,axis=1) for i in range(2)})
            delta=p-null
            effects=[delta[list(CR_SEEDS[i*10:(i+1)*10])].apply(strict_mean,axis=1) for i in range(2)]
            summary=dict(dose=dose,scale=scale,trials=len(g),valid=int(g.value.notna().sum()),
                mean=g.value.mean(),median=g.value.median(),median_within_sd=p.std(axis=1).median(),
                mean_delta=np.nanmean(delta.to_numpy()),median_delta=np.nanmedian(delta.to_numpy()),
                mean_abs_delta=np.nanmean(np.abs(delta.to_numpy())),
                split_mean0=split[0].mean(),split_mean1=split[1].mean(),
                paired_valid=int(delta.notna().sum().sum()),complete_contexts=int(split.notna().all(axis=1).sum()))
            if scale:
                paired_effects=pd.concat(effects,axis=1).dropna()
                summary.update(ranks(p),split_rank=split.corr(method='spearman').iloc[0,1],
                    split_mae=(split[0]-split[1]).abs().mean(),
                    effect_split_rank=pd.concat(effects,axis=1).corr(method='spearman').iloc[0,1],
                    effect_split_sign_agreement=(np.sign(paired_effects.iloc[:,0])==np.sign(paired_effects.iloc[:,1])).mean())
            summaries.append(summary)
            for idx in split.index:
                split_effects.append(dict(zip(F.C.KEYS,idx),dose=dose,scale=scale,mean0=split.loc[idx,0],mean1=split.loc[idx,1],effect0=effects[0].loc[idx],effect1=effects[1].loc[idx]))
    return pd.DataFrame(sr_summary),details,blocks.reset_index(),pd.DataFrame(summaries),pd.DataFrame(split_effects)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    previous=json.loads((OUT.parent/'srcr-replicate-stability-v1/manifest.json').read_text())
    protected=dict(previous['protected_before'])
    for root in (OUT.parent,F.C.ROOT/'research',F.C.ROOT/'tests'):
        for p in root.rglob('*'):
            if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts: protected[str(p.relative_to(F.C.ROOT))]=digest(p)
    for p in F.C.ROOT.glob('*.md'): protected[str(p.relative_to(F.C.ROOT))]=digest(p)
    for p,h in protected.items(): assert digest(F.C.ROOT/p)==h,p
    keys=select_contexts()
    atomic_json(OUT/'predeclared_config.json',CONFIG)
    atomic_csv(OUT/'contexts.csv',keys)
    manifest=dict(status='running',config_sha256=digest(OUT/'predeclared_config.json'),source_sha256=digest(__file__),
        environment=environment(),protected_before=protected,labels_loaded=False,models_fitted=False)
    atomic_json(OUT/'manifest.json',manifest)
    dm=json.loads((F.C.DATA_DIR/'manifest.json').read_text())
    sr=[];cr=[];parity=0;repeat=0
    for i,rec in enumerate(dm['records']):
        path=F.C.RAW_DIR/rec['patient']/rec['file']
        assert digest(path)==rec['raw_sha256']
        with mne.io.read_raw_edf(path,preload=True,verbose='ERROR') as raw:
            raw=F.preprocess_raw(F.dedupe_channel_names(raw),dm['spec']['channels'],**dm['spec']['preprocessing'])
            signal=raw.get_data(picks=[F.CHANNEL])[0]
            for key in keys[(keys.patient==rec['patient'])&(keys.file==rec['file'])].to_dict('records'):
                a,b,reason=F.context_bounds(key['start_sec'],key['end_sec'],len(signal),'trailing'); assert reason=='ok'
                x=signal[a:b];drive,frequency=F.characteristic_component(x,256)
                ident={k:key[k] for k in F.C.KEYS}
                for seed in (42,)+SR_SEEDS:
                    value=F.measurements(x,seed)
                    sr.append(dict(ident,seed=seed,value=value['sr_value'],reason=value['sr_reason'],missing=int(not np.isfinite(value['sr_value']))))
                np.testing.assert_equal(sr[-1]['value'],F.measurements(x,SR_SEEDS[-1])['sr_value']);repeat+=1
                for dose in DOSES:
                    for scale in SCALES:
                        for seed in CR_SEEDS:
                            cr.append(dict(ident,dose=dose,scale=scale,seed=seed,**cr_trial(drive,scale,dose,seed)))
                # One original-evaluator grid endpoint per recording/context.
                actual,clip=F.oscillator(drive*2,F.innovations(62),1.,1)
                expected,diag=F.fhn(drive,256,1.,np.random.default_rng(62),input_scale=.7)
                np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
                assert clip==diag['clipped_fraction'];parity+=1
        print(f'Candidate study {i+1}/45',flush=True)
    sr=pd.DataFrame(sr);cr=pd.DataFrame(cr)
    assert len(sr)==3690 and len(cr)==21600
    for table in (sr,cr):
        assert (table.missing==table.value.isna().astype(int)).all()
        assert table.loc[table.reason!='ok','value'].isna().all()
    a,b,c,d,e=summarize(sr,cr)
    for name,table in [('sr_trials',sr),('cr_trials',cr),('sr_stability',a),('sr_blocks',c),('cr_summary',d),('cr_split_responses',e)]: atomic_csv(OUT/(name+'.csv'),table)
    atomic_json(OUT/'sr_agreement.json',b)
    atomic_csv(OUT/'cr_reasons.csv',cr.groupby(['dose','scale','reason']).size().reset_index(name='trials'))
    for p,h in protected.items(): assert digest(F.C.ROOT/p)==h,p
    manifest.update(status='complete',protected_files_unchanged=len(protected),parity_contexts=parity,determinism_contexts=repeat,
        artifacts={p.name:digest(p) for p in OUT.iterdir() if p.name!='manifest.json'})
    atomic_json(OUT/'manifest.json',manifest)
    print(a.to_string(index=False)); print(json.dumps(b)); print(d.to_string(index=False))


if __name__=='__main__': main()
