"""Separate SR/CR feature integration v1; full coverage, no ML fitting."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from eeg_seizure import config as C
from eeg_seizure.artifacts import atomic_csv,atomic_json,digest,environment
from eeg_seizure.dataset import load_dataset
from eeg_seizure.channels import dedupe_channel_names
from eeg_seizure.preprocessing import preprocess_raw
from eeg_seizure.analysis.sr import characteristic_component,bistable
from eeg_seizure.analysis.coherence import fhn,fhn_events,transitions,regularity
from eeg_seizure.analysis.spectral import spectral_snr

CHANNEL="FP1-F7"
VALUE_COLUMNS=["sr__snr_change_vs_zero_noise_db__FP1-F7__ctx10", "cr__fhn_interval_cv__FP1-F7__ctx10"]
MASK_COLUMNS=[c+"__missing" for c in VALUE_COLUMNS]
EXTRA_COLUMNS=VALUE_COLUMNS+MASK_COLUMNS


def context_bounds(start_sec,end_sec,n_samples,mode,sfreq=256):
    """Strict within-recording context. Never shift, pad or borrow another file."""
    if mode not in ("centered","trailing") or end_sec-start_sec!=4 or start_sec<0 or sfreq!=256:
        raise ValueError("Expected canonical 4-second window and known context mode")
    a,b=(start_sec-3,end_sec+3) if mode=="centered" else (end_sec-10,end_sec)
    start,stop=round(a*sfreq),round(b*sfreq)
    if not np.isclose(start/sfreq,a) or not np.isclose(stop/sfreq,b) or stop-start!=2560:
        raise ValueError("Context does not map to exactly 2560 samples")
    reason="ok" if start>=0 and stop<=n_samples else "recording_start" if start<0 else "recording_end"
    return start,stop,reason


@njit(cache=False,fastmath=False)
def oscillator(drive,innovations,noise_intensity,model):
    """Acceleration only: existing dt=1/256, one-substep equations and clipping."""
    dt=1/256
    response=np.empty(len(drive)); response[0]=-1.
    w=-.5; clipped=0
    for i in range(1,len(drive)):
        value=response[i-1]
        noise=np.sqrt(2*noise_intensity*dt)*innovations[i-1]
        if model==0:
            proposed=value+dt*(value-value**3+.2*drive[i-1])+noise
            clipped+=int(abs(proposed)>3)
            value=min(3.,max(-3.,proposed))
        else:
            proposed=value+dt*(value-value**3/3-w+.35*drive[i-1])+noise
            proposed_w=w+dt*.08*(value+.7-.8*w)
            clipped+=int(abs(proposed)>4 or abs(proposed_w)>4)
            value=min(4.,max(-4.,proposed)); w=min(4.,max(-4.,proposed_w))
        response[i]=value
    return response,clipped/(len(drive)-1)


def innovations(seed):
    return np.random.default_rng(seed).normal(size=2559)


def measurements(signal,seed=42,reference=False):
    """Only waveform enters: no annotations, IDs, patient data or fitted statistics."""
    x=np.asarray(signal,dtype=np.float64)
    if x.shape!=(2560,) or not np.isfinite(x).all() or x.std()==0:
        return dict(sr_value=np.nan,cr_value=np.nan,sr_reason="invalid_or_constant_signal",cr_reason="invalid_or_constant_signal",
                    bistable_valid=False,bistable_reason="invalid_or_constant_signal",bistable_events=0,bistable_intervals=0,
                    fhn_events=0,fhn_intervals=0,fhn_valid=False,frequency=np.nan,max_clipped_fraction=np.nan)
    drive,frequency=characteristic_component(x,256)
    if reference:
        noisy,a=bistable(drive,256,.5,np.random.default_rng(seed))
        quiet,b=bistable(drive,256,0.,np.random.default_rng(seed))
        output,c=fhn(drive,256,.5,np.random.default_rng(seed))
        clips=[a["clipped_fraction"],b["clipped_fraction"],c["clipped_fraction"]]
    else:
        noise=innovations(seed)
        noisy,a=oscillator(drive,noise,.5,0)
        quiet,b=oscillator(drive,noise,0.,0)
        output,c=oscillator(drive,noise,.5,1)
        clips=[a,b,c]
    gain=spectral_snr(noisy,256,frequency)-spectral_snr(quiet,256,frequency)
    bistable_measure=regularity(transitions(noisy,256),256)
    fhn_measure=regularity(fhn_events(output,256),256,minimum_intervals=3,ddof=0,minimum_interval_sec=.2)
    sr_reason="ok" if np.isfinite(gain) else "undefined_spectral_ratio"
    cr_reason=fhn_measure["reason"]
    if max(clips[:2])>0: sr_reason="clipped_dynamics"
    if clips[2]>0: cr_reason="clipped_dynamics"
    return dict(sr_value=float(gain) if sr_reason=="ok" else np.nan,
        cr_value=float(fhn_measure["cv"]) if cr_reason=="ok" else np.nan,
        sr_reason=sr_reason,cr_reason=cr_reason,frequency=frequency,max_clipped_fraction=max(clips),
        bistable_valid=bistable_measure["valid"],bistable_reason=bistable_measure["reason"],
        bistable_events=bistable_measure["events"],bistable_intervals=bistable_measure["intervals"],
        bistable_cv=bistable_measure["cv"],fhn_events=fhn_measure["events"],fhn_intervals=fhn_measure["intervals"],
        fhn_valid=fhn_measure["valid"])


def record_features(signal,keys,mode,seed=42):
    """Accept keys only; labels are joined by the caller after extraction."""
    if list(keys.columns)!=C.KEYS:
        raise ValueError("Pass only canonical identity columns; no labels")
    rows=[]
    for row in keys.to_dict("records"):
        start,stop,reason=context_bounds(row["start_sec"],row["end_sec"],len(signal),mode)
        values=measurements(signal[start:stop],seed) if reason=="ok" else dict(
            sr_value=np.nan,cr_value=np.nan,sr_reason=reason,cr_reason=reason,bistable_valid=False,
            bistable_reason=reason,bistable_events=np.nan,bistable_intervals=np.nan,fhn_events=np.nan,fhn_intervals=np.nan,fhn_valid=False)
        rows.append(dict(row,mode=mode,context_start_sec=start/256,context_end_sec=stop/256,context_reason=reason,**values))
    return pd.DataFrame(rows)


def protected_files():
    roots=[C.ROOT/p for p in ("src","notebooks","data/processed/corrected_v2","results/corrected","results/analysis","results/feasibility","results/ablations","results/contribution")]
    files=[p for root in roots for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += [p for p in (C.ROOT/"research").glob("*.py") if p.name!="srcr_features.py" and not p.name.startswith("validate_srcr")]
    files += [p for p in C.ROOT.glob("*.md") if p.name!="SRCR_FEATURE_INTEGRATION_REPORT.md"]
    return {str(p.relative_to(C.ROOT)):digest(p) for p in files}


def check_parity(signal):
    fast=measurements(signal)
    slow=measurements(signal,reference=True)
    for key in ("sr_reason","cr_reason","bistable_valid","bistable_events","bistable_intervals","fhn_events","fhn_intervals","fhn_valid"):
        if fast[key]!=slow[key]: raise AssertionError(f"Accelerated detector parity failed: {key}")
    for key in ("sr_value","cr_value"):
        np.testing.assert_allclose(fast[key],slow[key],rtol=1e-8,atol=1e-8,equal_nan=True)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",default="srcr-context-features-v1")
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in (".",".."):
        raise ValueError("One run ID required")
    out=C.ROOT/"results/feature_integration"/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    before=protected_files(); frame,dm=load_dataset()
    manifest=dict(status="running",version="srcr-context-features-v1",source_sha256=digest(__file__),environment=environment(),
        canonical_dataset_sha256=dm["feature_sha256"],canonical_spec=dm["spec"],protected_before=before,
        configuration=dict(channel=CHANNEL,context_sec=10,ml_window_sec=4,modes=["centered","trailing"],noise_intensity=.5,
            substeps=1,seed=42,seed_rule="common fixed innovations for all windows; function of waveform only",trials=1,
            candidate_values=VALUE_COLUMNS,missing_flags=MASK_COLUMNS,dimensions=334,
            scaling_imputation="not fitted; require outer/inner training-only median imputation and scaling later",
            boundaries="strict full context; no padding/shifting/cross-recording borrowing",future="centered +3s beyond target end; trailing none beyond end, but canonical full-recording filtering is noncausal",
            selection="predeclared SR gain and FHN CV; bistable regularity diagnostic only; no parameter search"),
        fitting="none",timing_scope="feature extraction and coverage only")
    atomic_json(out/"manifest.json",manifest)
    all_rows=[]; parity_count=0; seed_rows=[]; start_time=time.perf_counter()
    try:
        for i,rec in enumerate(dm["records"]):
            patient,file=rec["patient"],rec["file"]
            path=C.RAW_DIR/patient/file; summary=path.parent/f"{patient}-summary.txt"
            assert digest(path)==rec["raw_sha256"] and digest(summary)==rec["summary_sha256"]
            subset=frame[(frame.patient==patient)&(frame.file==file)]
            with mne.io.read_raw_edf(path,preload=True,verbose="ERROR") as raw:
                assert raw.info["sfreq"]==256
                raw=preprocess_raw(dedupe_channel_names(raw),dm["spec"]["channels"],**dm["spec"]["preprocessing"])
                signal=raw.get_data(picks=[CHANNEL])[0]
                for mode in ("centered","trailing"):
                    table=record_features(signal,subset[C.KEYS],mode)
                    table=table.merge(subset[C.META],on=C.KEYS,validate="one_to_one")
                    all_rows.append(table)
                    valid=table[table.context_reason=="ok"]
                    first=valid.iloc[0]
                    context=signal[round(first.context_start_sec*256):round(first.context_end_sec*256)]
                    check_parity(context); parity_count+=1
                    # Predefined bounded seed diagnostic: first/middle/last valid context
                    # per recording and mode; never selected using annotations or values.
                    for position in sorted(set([0,len(valid)//2,len(valid)-1])):
                        row=valid.iloc[position]
                        context=signal[round(row.context_start_sec*256):round(row.context_end_sec*256)]
                        for seed in (42,43,44):
                            seed_rows.append(dict(patient=patient,file=file,start_sec=row.start_sec,mode=mode,seed=seed,**measurements(context,seed)))
            print(f"Coverage {i+1}/{len(dm['records'])}: {patient}/{file}",flush=True)
        diagnostics=pd.concat(all_rows,ignore_index=True)
        diagnostics["joint_valid"]=diagnostics.sr_value.notna()&diagnostics.cr_value.notna()
        atomic_csv(out/"diagnostics.csv",diagnostics)
        atomic_csv(out/"seed_sensitivity.csv",pd.DataFrame(seed_rows))
        coverage=diagnostics.groupby(["mode","label"]).agg(windows=("label","size"),
            full_context=("context_reason",lambda s:int((s=="ok").sum())),sr_valid=("sr_value","count"),
            cr_valid=("cr_value","count"),joint_valid=("joint_valid","sum"),bistable_valid=("bistable_valid","sum"),
            bistable_events=("bistable_events","sum"),fhn_events=("fhn_events","sum")).reset_index()
        atomic_csv(out/"coverage.csv",coverage)
        reasons=diagnostics.groupby(["mode","label","context_reason","sr_reason","cr_reason"]).size().reset_index(name="windows")
        atomic_csv(out/"missing_reasons.csv",reasons)
        atomic_csv(out/"patient_coverage.csv",diagnostics.groupby(["mode","patient","label"]).agg(windows=("label","size"),sr_valid=("sr_value","count"),cr_valid=("cr_value","count"),joint_valid=("joint_valid","sum")).reset_index())
        schemas={}
        for mode in ("centered","trailing"):
            extra=diagnostics[diagnostics["mode"]==mode][C.KEYS+["sr_value","cr_value"]].rename(columns=dict(zip(["sr_value","cr_value"],VALUE_COLUMNS)))
            extended=frame.merge(extra,on=C.KEYS,validate="one_to_one",sort=False)
            pd.testing.assert_frame_equal(extended[frame.columns],frame)
            for value,mask in zip(VALUE_COLUMNS,MASK_COLUMNS): extended[mask]=extended[value].isna().astype(int)
            schema=dm["spec"]["feature_columns"]+EXTRA_COLUMNS
            assert len(schema)==334 and len(set(schema))==334 and len(extended)==39522
            atomic_csv(out/f"extended_{mode}.csv",extended[C.META+schema])
            schemas[mode]=dict(feature_columns=schema,context_mode=mode,version=f"srcr-334-{mode}-v1",canonical_values_unchanged=True)
        atomic_json(out/"schemas.json",schemas)
        if before!=protected_files(): raise AssertionError("Protected artifact changed")
        manifest.update(status="complete",seconds=time.perf_counter()-start_time,parity_contexts=parity_count,
            protected_files_unchanged=len(before),numba_version=__import__("numba").__version__,
            artifacts={p.name:digest(p) for p in out.iterdir() if p.name!="manifest.json"})
    except Exception as exc:
        manifest.update(status="failed",error=str(exc)); raise
    finally: atomic_json(out/"manifest.json",manifest)
    print(coverage.to_string(index=False)); print(out)


if __name__=="__main__": main()
