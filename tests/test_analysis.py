"""Cheap diagnostic tests; no model fitting or resonance claims."""
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from scipy.signal import welch, butter, filtfilt, find_peaks
from eeg_seizure import config as C
from eeg_seizure.analysis.spectral import psd, colored_noise, add_noise, spectral_slope
from eeg_seizure.analysis.sr import bistable, characteristic_component, voltage_referenced_transform
from eeg_seizure.analysis.coherence import fhn, fhn_events, transitions, regularity
from eeg_seizure.analysis.compression import reconstruct, preservation
from eeg_seizure.analysis.reporting import summarize, raw_optima
from eeg_seizure.analysis.context import seed_for, new_run


def notebook_function(cell, name):
    book=json.loads((C.ROOT/"notebooks/02_preprocess_chb01.ipynb").read_text(encoding="utf-8"))
    tree=ast.parse("".join(book["cells"][cell]["source"]))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
    env=dict(np=np,welch=welch,butter=butter,filtfilt=filtfilt,find_peaks=find_peaks,INPUT_SCALE=.2)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"<archived-equation-parity>","exec"),env)
    return env[name]


def test_psd_matches_canonical_welch_and_units():
    x=2e-6*np.sin(2*np.pi*10*np.arange(1024)/256)
    f,p=psd(x,256.)
    f0,p0=welch(x,fs=256.,nperseg=256)
    np.testing.assert_array_equal(f,f0)
    np.testing.assert_allclose(p,p0,rtol=1e-14,atol=1e-30)
    assert f[np.argmax(p)]==10
    np.testing.assert_allclose(np.trapezoid(p,f),np.var(x),rtol=1e-10)


@pytest.mark.parametrize("beta",[0,1,3])
def test_noise_units_reproducibility_and_slope(beta):
    n=colored_noise(16384,256,beta,np.random.default_rng(8))
    np.testing.assert_allclose(n.mean(),0,atol=1e-14)
    np.testing.assert_allclose(n.std(),1,atol=1e-14)
    assert abs(spectral_slope(n,256)+beta)<.25
    x=np.linspace(-1e-6,1e-6,16384)
    y=add_noise(x,256,beta,.3,np.random.default_rng(8))
    np.testing.assert_allclose((y-x).std()/x.std(),.3,rtol=1e-12)
    np.testing.assert_array_equal(add_noise(x,256,beta,0,np.random.default_rng(8)),x)


def test_bistable_matches_existing_equation_and_repeat():
    x=np.sin(np.arange(1024)/20)
    old=notebook_function(74,"bistable_sr")
    expected=old(x,256,.5,rng=np.random.default_rng(42))
    actual,diagnostic=bistable(x,256,.5,np.random.default_rng(42))
    np.testing.assert_array_equal(actual,expected)
    assert 0<=diagnostic["clipped_fraction"]<=1
    quiet,_=bistable(np.zeros(1024),256,0,np.random.default_rng(1))
    np.testing.assert_array_equal(quiet,np.full(1024,-1.))


def test_fhn_legacy_parity_and_correct_noise_increment():
    x=np.sin(np.arange(128)/20)
    old=notebook_function(80,"cr_fhn")
    expected=old(x,256,.1,rng=np.random.default_rng(1))
    actual,_=fhn(x,256,.1,np.random.default_rng(1),convention="legacy_extra_dt")
    np.testing.assert_allclose(actual,expected,rtol=1e-14,atol=1e-14)
    class Fixed:
        def normal(self): return 1.
    zero,_=fhn(np.zeros(2),256,0,Fixed())
    noisy,_=fhn(np.zeros(2),256,1,Fixed())
    np.testing.assert_allclose(noisy[1]-zero[1],np.sqrt(2/256),rtol=1e-14)


def test_characteristic_frequency_is_unlabelled_and_mapping_in_volts():
    x=4e-6*np.sin(2*np.pi*8*np.arange(1024)/256)+1e-6
    component,frequency=characteristic_component(x,256)
    assert frequency==8
    y,info=voltage_referenced_transform(x,256,.1,np.random.default_rng(2))
    np.testing.assert_allclose([y.mean(),y.std()],[x.mean(),x.std()],rtol=1e-12)
    assert info["target_frequency"]==8 and np.isfinite(y).all()


def test_events_keep_counts_and_regularity_edge_cases():
    short=regularity(np.array([1,5,9]),256)
    assert short["events"]==3 and short["intervals"]==2 and not short["valid"]
    regular=regularity(np.arange(7)*100,256)
    assert regular["cv"]==0 and np.isinf(regular["coherence"]) and regular["reason"]=="zero_cv_unbounded"
    events=transitions([0,0,.6,.6,-.6,-.6,.6],10,min_dwell_sec=0)
    np.testing.assert_array_equal(events,[4,6])
    assert len(transitions(np.zeros(10),256))==0
    with pytest.raises(ValueError): regularity([4,3,7],256)


def test_fhn_peak_detector_preserves_notebook_definition():
    x=np.sin(np.arange(1024)/20)
    old=notebook_function(80,"detect_cr_events")
    np.testing.assert_array_equal(fhn_events(x,256),old(x,256))
    measure=regularity([0,51,103,155,207],256,minimum_intervals=3,ddof=0,minimum_interval_sec=.2)
    assert measure["discarded_intervals"]==1 and measure["events"]==5 and measure["intervals"]==3


@pytest.mark.parametrize("method",["legacy_stride","antialiased"])
def test_compression_identity_and_metadata(method):
    x=np.sin(np.arange(1024)/10)*1e-6
    y,info=reconstruct(x,1,method)
    np.testing.assert_array_equal(x,y)
    assert preservation(x,y)["rmse_volts"]==0
    z,info=reconstruct(x,4,method)
    assert len(z)==len(x) and info["sample_ratio"]==4
    if method=="legacy_stride":
        np.testing.assert_array_equal(z,np.interp(np.arange(len(x)),np.arange(len(x[::4]))*4,x[::4]))


def test_invalid_trials_not_discarded_and_peak_uses_raw_data():
    rows=[]
    for level,values in [(0,[np.nan,np.nan]),(.1,[1.,np.nan]),(.5,[3.,4.]),(1.,[2.,2.])]:
        for trial,value in enumerate(values):
            rows.append(dict(patient="p",file="f",start_sec=0.,end_sec=4.,label=0,model="m",drive="eeg",
                             noise_intensity=level,substeps=1,trial=trial,valid=np.isfinite(value),coherence=value,
                             snr_db=value,events=2,clipped_fraction=0))
    result=summarize(pd.DataFrame(rows))
    assert result.valid_trials.tolist()==[0,1,2,2]
    assert result.events_mean.tolist()==[2]*4
    peak=raw_optima(result)
    assert (peak.noise_intensity==.5).all()
    assert (peak.status=="interior_grid_maximum").all()
    assert seed_for(42,"p","f",0,1)==seed_for(42,"p","f",0,1)
    assert seed_for(42,"p","f",0,1)!=seed_for(42,"p","f",0,2)


def test_analysis_outputs_never_reuse_run_directory(tmp_path,monkeypatch):
    monkeypatch.setattr(C,"ROOT",tmp_path)
    dataset=dict(feature_sha256="a",spec={})
    path,_=new_run("test",{},dataset,{})
    assert path.is_relative_to(tmp_path/"results/analysis")
    with pytest.raises(FileExistsError):new_run("test",{},dataset,{})


def test_robustness_transform_repeats_without_mutating_inputs():
    from eeg_seizure.analysis.runner import condition_signals
    matrix=np.stack([np.sin(np.arange(1024)/20)*1e-6,np.cos(np.arange(1024)/13)*2e-6])
    before=matrix.copy()
    args=SimpleNamespace(seed=42,noise_ratio=.3,factor=4,sr_noise=.5)
    for condition in ("clean","noise_white","noise_pink","noise_black","compression_legacy_stride","compression_antialiased","sr_voltage_referenced"):
        a,_=condition_signals(matrix,condition,args,["p","f",0.,4.],0)
        b,_=condition_signals(matrix,condition,args,["p","f",0.,4.],0)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(matrix,before)


def test_robustness_refuses_training_patient_and_wrong_dataset(monkeypatch,tmp_path):
    import eeg_seizure.analysis.runner as runner
    bundle=dict(train_patients=["p"],dataset_sha256="expected",dataset_spec={})
    monkeypatch.setattr(runner,"load_bundle",lambda path:bundle)
    args=SimpleNamespace(bundle=tmp_path/"fake",patient="p")
    with pytest.raises(ValueError,match="held out"):
        runner.robustness(None,None,None,{},dict(feature_sha256="expected"),args,tmp_path)
    args.patient="other"
    with pytest.raises(ValueError,match="differs"):
        runner.robustness(None,None,None,{},dict(feature_sha256="wrong"),args,tmp_path)


def test_all_invalid_trials_remain_missing_in_report():
    rows=[dict(patient="p",file="f",start_sec=0.,end_sec=4.,label=0,model="m",drive="none",
               noise_intensity=d,substeps=1,trial=t,valid=False,coherence=np.nan,
               snr_db=np.nan,events=2,clipped_fraction=0) for d in (0,.5) for t in range(3)]
    summary=summarize(pd.DataFrame(rows))
    assert summary.invalid_trials.sum()==6 and summary.valid_trials.sum()==0
    assert summary.coherence_median.isna().all()
    assert summary.events_mean.eq(2).all()
    assert raw_optima(summary).status.eq("no_valid_measurement").all()


@pytest.mark.parametrize("model",[bistable,fhn])
def test_zero_noise_time_step_refinement(model):
    # Deterministic integration check only; this does not establish SDE convergence.
    drive=np.sin(2*np.pi*2*np.arange(256)/256)
    reference,_=model(drive,256,0,np.random.default_rng(7),substeps=16)
    errors=[]
    for steps in (1,2,4):
        result,info=model(drive,256,0,np.random.default_rng(7),substeps=steps)
        errors.append(np.sqrt(np.mean((result-reference)**2)))
        assert np.isfinite(result).all() and info["clipped_fraction"]==0
    assert errors[0]>errors[1]>errors[2]


def test_stride_aliasing_and_antialias_control():
    # At 4:1 reduction, 40 Hz aliases to 24 Hz without an anti-alias filter.
    x=np.sin(2*np.pi*40*np.arange(4096)/256)
    legacy,_=reconstruct(x,4,"legacy_stride")
    filtered,_=reconstruct(x,4,"antialiased")
    f,p=psd(legacy[256:-256],256)
    _,q=psd(filtered[256:-256],256)
    assert f[np.argmax(p)]==24
    assert q[f==24][0] < .01*p[f==24][0]
