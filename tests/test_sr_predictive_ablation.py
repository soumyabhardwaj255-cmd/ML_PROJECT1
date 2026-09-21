import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'research'))
import sr_predictive_ablation as S


def test_frozen_average_parity_and_determinism():
    signal=np.random.default_rng(91).normal(size=2560)*1e-5
    expected=np.mean([S.F.measurements(signal,s)['sr_value'] for s in range(62,72)])
    a=S.sr_mean(signal)
    np.testing.assert_allclose(a['value'],expected,rtol=1e-10,atol=1e-10)
    assert a==S.sr_mean(signal.copy()) and a['valid_trials']==10
    assert S.SEEDS==tuple(range(62,72))
    assert S.sr_mean(np.zeros(2560))['missing']==1


def test_any_bad_replicate_makes_average_missing(monkeypatch):
    calls=[]
    def ratio(*args):
        calls.append(1)
        return np.nan if len(calls)==4 else 2.
    monkeypatch.setattr(S.F,'spectral_snr',ratio)
    result=S.sr_mean(np.random.default_rng(9).normal(size=2560))
    assert np.isnan(result['value']) and result['missing']==1 and result['valid_trials']==9


def test_schema_preserves_columns_and_rejects_masks_or_cr():
    frame=pd.DataFrame(dict(patient=['p'],file=['f'],start_sec=[0.],end_sec=[4.],label=[0],canonical=[1e-12]))
    extra=frame[S.C.KEYS].assign(**{S.VALUE:[np.nan],S.MASK:[1]})
    out=S.extended_table(frame,extra)
    pd.testing.assert_frame_equal(out[frame.columns],frame)
    with pytest.raises(ValueError): S.extended_table(frame,extra.assign(**{S.MASK:[0]}))
    with pytest.raises(ValueError): S.extended_table(frame,extra.assign(cr=1))


def test_training_transform_ignores_held_out_values():
    from eeg_seizure.modeling import FeatureTransform
    from eeg_seizure.evaluation import patient_split
    frame=pd.DataFrame({'patient':['a','a','b'],S.VALUE:[1.,np.nan,999.],S.MASK:[0,1,0]})
    train,test=patient_split(frame,'b')
    transform=FeatureTransform().fit(train[[S.VALUE,S.MASK]])
    assert transform.imputer_.statistics_[0]==1.
    before=transform.scaler_.mean_.copy()
    transform.transform(test[[S.VALUE,S.MASK]])
    np.testing.assert_array_equal(before,transform.scaler_.mean_)
