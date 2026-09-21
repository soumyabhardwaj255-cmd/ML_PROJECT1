import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'research'))
import srcr_candidate_study as S


def test_predeclared_groups_and_missing_aggregation():
    assert len(set(S.SR_SEEDS))==40
    assert S.CONFIG['candidate_seeds']==list(range(62,72))
    assert S.strict_mean([1.,3.])==2.
    assert np.isnan(S.strict_mean([1.,np.nan]))
    assert S.DOSES==(.25,.5,1.) and S.SCALES==(0.,.175,.35,.7)


@pytest.mark.parametrize('scale,dose',[(0.,.25),(.175,.25),(.35,.5),(.7,1.)])
def test_grid_preserves_original_equations(scale,dose):
    drive=np.sin(2*np.pi*8*np.arange(2560)/256)
    expected,diag=S.F.fhn(drive,256,dose,np.random.default_rng(62),input_scale=scale)
    actual,clip=S.F.oscillator(drive*(scale/.35),S.F.innovations(62),dose,1)
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
    assert clip==diag['clipped_fraction']
    a=S.cr_trial(drive,scale,dose,62)
    b=S.cr_trial(drive.copy(),scale,dose,62)
    np.testing.assert_equal(a,b)
    assert a['missing']==int(not np.isfinite(a['value']))


def test_selection_is_deterministic_label_free_and_disjoint():
    a=S.select_contexts();b=S.select_contexts()
    assert a.equals(b) and len(a)==90 and 'label' not in a.columns
    old=S.pd.read_csv(S.BASE/'seed_sensitivity.csv',usecols=['patient','file','start_sec','mode']).drop_duplicates()
    for row in a.itertuples():
        for prior in old[(old.patient==row.patient)&(old.file==row.file)].itertuples():
            x,y=(prior.start_sec-6,prior.start_sec+4) if prior.mode=='trailing' else (prior.start_sec-3,prior.start_sec+7)
            assert row.context_end_sec<=x or row.context_start_sec>=y
