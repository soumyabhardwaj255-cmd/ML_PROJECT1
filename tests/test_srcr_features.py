import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from eeg_seizure import config as C
from eeg_seizure.analysis.sr import bistable
from eeg_seizure.analysis.coherence import fhn,regularity

spec=importlib.util.spec_from_file_location("srcr_features",Path(__file__).parents[1]/"research/srcr_features.py")
feature=importlib.util.module_from_spec(spec); spec.loader.exec_module(feature)


def test_context_mapping_and_boundaries():
    assert feature.context_bounds(8,12,256*20,"centered")== (5*256,15*256,"ok")
    assert feature.context_bounds(8,12,256*20,"trailing")== (2*256,12*256,"ok")
    assert feature.context_bounds(0,4,256*20,"centered")[2]=="recording_start"
    assert feature.context_bounds(4,8,256*20,"trailing")[2]=="recording_start"
    assert feature.context_bounds(16,20,256*20,"centered")[2]=="recording_end"
    assert feature.context_bounds(16,20,256*20,"trailing")[2]=="ok"
    with pytest.raises(ValueError): feature.context_bounds(0,10,256*20,"trailing")


@pytest.mark.parametrize("noise_level",[0.,.5,2.])
def test_acceleration_preserves_existing_equations(noise_level):
    drive=np.random.default_rng(33).normal(size=2560)
    noise=feature.innovations(42)
    for index,original in ((0,bistable),(1,fhn)):
        actual,clip=feature.oscillator(drive,noise,noise_level,index)
        expected,diagnostic=original(drive,256,noise_level,np.random.default_rng(42))
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
        assert clip==diagnostic["clipped_fraction"]


def test_deterministic_feature_and_detector_parity():
    t=np.arange(2560)/256
    signal=1e-5*(np.sin(2*np.pi*8*t)+.2*np.random.default_rng(6).normal(size=len(t)))
    original=signal.copy()
    assert feature.check_parity(signal)
    a,b=feature.measurements(signal),feature.measurements(signal)
    pd.testing.assert_series_equal(pd.Series(a),pd.Series(b))
    np.testing.assert_array_equal(signal,original)


def test_no_labels_or_other_patients_enter_feature_calculation():
    signal=np.random.default_rng(11).normal(size=20*256)*1e-5
    keys=pd.DataFrame([dict(patient="p",file="a",start_sec=8.,end_sec=12.)],columns=C.KEYS)
    a=feature.record_features(signal,keys,"trailing")
    # Identity and unrelated patient/recording metadata cannot change the values.
    changed=keys.assign(patient="held_out",file="other")
    b=feature.record_features(signal,changed,"trailing")
    pd.testing.assert_frame_equal(a[["sr_value","cr_value"]],b[["sr_value","cr_value"]])
    with pytest.raises(ValueError): feature.record_features(signal,keys.assign(label=1),"trailing")
    assert len(feature.EXTRA_COLUMNS)==4 and len(set(feature.EXTRA_COLUMNS))==4


def test_future_dependency_is_explicit_and_insufficiency_is_not_filled():
    signal=np.random.default_rng(12).normal(size=20*256)*1e-5
    keys=pd.DataFrame([dict(patient="p",file="a",start_sec=8.,end_sec=12.)],columns=C.KEYS)
    a=feature.record_features(signal,keys,"trailing")
    changed=signal.copy(); changed[12*256:]*=10
    b=feature.record_features(changed,keys,"trailing")
    pd.testing.assert_frame_equal(a,b)
    start,stop,_=feature.context_bounds(8,12,len(signal),"centered")
    assert stop>12*256 and not np.array_equal(signal[start:stop],changed[start:stop])
    zero=feature.measurements(np.zeros(2560))
    assert np.isnan(zero["sr_value"]) and np.isnan(zero["cr_value"])
    insufficient=regularity(np.array([1,100]),256,minimum_intervals=3,ddof=0)
    assert not insufficient["valid"] and insufficient["reason"]=="insufficient_intervals"
    assert np.isnan(insufficient["cv"])
    boundary=feature.record_features(signal,keys.assign(start_sec=0.,end_sec=4.),"trailing")
    assert boundary.sr_value.isna().all() and boundary.cr_value.isna().all()
