"""Bounded label-independent stochastic assay validation; no model fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import srcr_features as F
from eeg_seizure.artifacts import digest, atomic_json, atomic_csv, environment

SEEDS = tuple(range(42, 62))
OUT = F.C.ROOT / 'results/feature_integration/srcr-replicate-stability-v1'
BASE = OUT.parent / 'srcr-context-features-v1'


def control(frequency, seed):
    """Zero external drive; retain context-derived frequency for matched SR assay.

    Do not call characteristic_component on zeros or redefine the feature.
    The deterministic zero-drive bistable trajectory can make SR undefined.
    """
    drive = np.zeros(2560)
    noise = F.innovations(seed)
    noisy, a = F.oscillator(drive, noise, .5, 0)
    quiet, b = F.oscillator(drive, noise, 0., 0)
    response, c = F.oscillator(drive, noise, .5, 1)
    value = F.spectral_snr(noisy, 256, frequency)-F.spectral_snr(quiet, 256, frequency)
    regularity = F.regularity(F.fhn_events(response, 256), 256,
                             minimum_intervals=3, ddof=0, minimum_interval_sec=.2)
    sr_reason = 'ok' if np.isfinite(value) else 'undefined_spectral_ratio'
    cr_reason = regularity['reason']
    if max(a, b)>0: sr_reason='clipped_dynamics'
    if c>0: cr_reason='clipped_dynamics'
    return dict(sr_value=value if sr_reason=='ok' else np.nan,
                cr_value=regularity['cv'] if cr_reason=='ok' else np.nan,
                sr_reason=sr_reason, cr_reason=cr_reason,
                fhn_events=regularity['events'], fhn_intervals=regularity['intervals'])


def summarize(rows):
    results=[]
    for mode in ('centered','trailing'):
        for feature in ('sr_value','cr_value'):
            group=rows[rows['mode']==mode]
            driven=group[group.condition=='driven'].pivot(index=['patient','file','start_sec'],columns='seed',values=feature)
            null=group[group.condition=='no_drive'].pivot(index=['patient','file','start_sec'],columns='seed',values=feature)
            corr=driven.corr(method='spearman').to_numpy()
            ranks=corr[np.triu_indices(len(SEEDS),1)]
            split=pd.concat([driven[list(SEEDS[:10])].mean(axis=1),driven[list(SEEDS[10:])].mean(axis=1)],axis=1)
            delta=(driven-null).to_numpy().ravel()
            finite=delta[np.isfinite(delta)]
            results.append(dict(mode=mode,feature=feature,contexts=len(driven),trials=driven.size,
                valid=int(driven.notna().sum().sum()),no_drive_valid=int(null.notna().sum().sum()),
                median_within_context_sd=driven.std(axis=1).median(),between_context_mean_sd=driven.mean(axis=1).std(),
                pair_rank_median=np.nanmedian(ranks),pair_rank_min=np.nanmin(ranks),pair_rank_max=np.nanmax(ranks),
                split10_rank=split.corr(method='spearman').iloc[0,1],split10_mae=(split[0]-split[1]).abs().mean(),
                paired_valid=len(finite),delta_mean=finite.mean() if len(finite) else np.nan,
                delta_median=np.median(finite) if len(finite) else np.nan,
                delta_abs_mean=np.abs(finite).mean() if len(finite) else np.nan,
                delta_positive_fraction=(finite>0).mean() if len(finite) else np.nan))
    return pd.DataFrame(results)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    baseline=json.loads((BASE/'manifest.json').read_text())
    protected=dict(baseline['protected_before'])
    for p in (F.C.ROOT/'results/feature_integration').rglob('*'):
        if p.is_file() and OUT not in p.parents: protected[str(p.relative_to(F.C.ROOT))]=digest(p)
    for p in (Path(F.__file__), F.C.ROOT/'SRCR_FEATURE_INTEGRATION_REPORT.md'):
        protected[str(p.relative_to(F.C.ROOT))]=digest(p)
    for p,h in protected.items(): assert digest(F.C.ROOT/p)==h,p
    dm=json.loads((F.C.DATA_DIR/'manifest.json').read_text())
    # Only identifiers from the predeclared first/middle/last sample are read.
    keys=pd.read_csv(BASE/'seed_sensitivity.csv',usecols=['patient','file','start_sec','mode']).drop_duplicates()
    assert len(keys)==270 and 'label' not in keys
    manifest=dict(status='running',seeds=list(SEEDS),contexts=270,selection='existing first/middle/last full context per recording and mapping',
        no_labels=True,fitting='none',definitions='unchanged srcr-context-features-v1',future_mapping='trailing selected; centered diagnostic only',
        controls='zero oscillator drive, matched innovations/settings; SR retains driven context frequency',
        aggregation='split 10-replicate means diagnostic only; not a changed feature',environment=environment(),
        source_sha256=digest(__file__),base_manifest_sha256=digest(BASE/'manifest.json'),protected_before=protected)
    atomic_json(OUT/'manifest.json',manifest)
    rows=[]
    for i,rec in enumerate(dm['records']):
        patient,file=rec['patient'],rec['file']
        path=F.C.RAW_DIR/patient/file
        assert digest(path)==rec['raw_sha256']
        with mne.io.read_raw_edf(path,preload=True,verbose='ERROR') as raw:
            raw=F.preprocess_raw(F.dedupe_channel_names(raw),dm['spec']['channels'],**dm['spec']['preprocessing'])
            signal=raw.get_data(picks=[F.CHANNEL])[0]
            for key in keys[(keys.patient==patient)&(keys.file==file)].to_dict('records'):
                a,b,reason=F.context_bounds(key['start_sec'],key['start_sec']+4,len(signal),key['mode'])
                assert reason=='ok'
                x=signal[a:b]
                for seed in SEEDS:
                    driven=F.measurements(x,seed)
                    for condition,values in [('driven',driven),('no_drive',control(driven['frequency'],seed))]:
                        rows.append(dict(key,seed=seed,condition=condition,**values,
                            sr_missing=int(not np.isfinite(values['sr_value'])),cr_missing=int(not np.isfinite(values['cr_value']))))
        print(f'Replicates {i+1}/45: {patient}/{file}',flush=True)
    table=pd.DataFrame(rows)
    assert len(table)==10800 and not table.duplicated(['patient','file','start_sec','mode','seed','condition']).any()
    for f in ('sr','cr'):
        assert (table[f+'_missing']==table[f+'_value'].isna().astype(int)).all()
        assert table.loc[table[f+'_reason']!='ok',f+'_value'].isna().all()
    # Independently reproduce the previous three-seed results on identical keys.
    previous=pd.read_csv(BASE/'seed_sensitivity.csv',float_precision='round_trip')
    paired=table[(table.condition=='driven')&table.seed.isin([42,43,44])].merge(previous,on=['patient','file','start_sec','mode','seed'],validate='one_to_one',suffixes=('_new','_old'))
    assert len(paired)==810
    for f in ('sr_value','cr_value'):
        np.testing.assert_allclose(paired[f+'_new'],paired[f+'_old'],rtol=1e-10,atol=1e-10,equal_nan=True)
    summary=summarize(table)
    atomic_csv(OUT/'replicates.csv',table)
    atomic_csv(OUT/'stability.csv',summary)
    atomic_csv(OUT/'reasons.csv',table.groupby(['mode','condition','sr_reason','cr_reason']).size().reset_index(name='trials'))
    atomic_csv(OUT/'seed_summary.csv',table.groupby(['mode','condition','seed']).agg(sr_mean=('sr_value','mean'),cr_mean=('cr_value','mean'),sr_valid=('sr_value','count'),cr_valid=('cr_value','count')).reset_index())
    for p,h in protected.items(): assert digest(F.C.ROOT/p)==h,p
    manifest.update(status='complete',previous_replicates_reproduced=810,protected_files_unchanged=len(protected),
        artifacts={p.name:digest(p) for p in OUT.iterdir() if p.name!='manifest.json'})
    atomic_json(OUT/'manifest.json',manifest)
    print(summary.to_string(index=False))


if __name__=='__main__': main()
