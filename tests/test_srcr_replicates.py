import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).parents[1]/'research'))
import srcr_replicates as R


def test_no_drive_preserves_undefined_sr_and_deterministic_cr():
    a=R.control(8.,42)
    b=R.control(8.,42)
    pd.testing.assert_series_equal(pd.Series(a),pd.Series(b))
    assert np.isnan(a['sr_value'])
    assert a['sr_reason']=='undefined_spectral_ratio'
    assert np.isfinite(a['cr_value'])
    # SR assay frequency cannot alter the FHN no-drive trajectory.
    assert R.control(12.,42)['cr_value']==a['cr_value']


def test_summary_keeps_missing_controls_and_measures_seed_variation():
    rows=[]
    for mode in ('centered','trailing'):
        for i in range(5):
            for seed in R.SEEDS:
                for condition in ('driven','no_drive'):
                    rows.append(dict(mode=mode,patient='p',file='f',start_sec=i,seed=seed,condition=condition,
                        sr_value=i+seed if condition=='driven' else np.nan,
                        cr_value=i+seed if condition=='driven' else seed))
    summary=R.summarize(pd.DataFrame(rows))
    assert len(summary)==4
    assert np.allclose(summary.pair_rank_median,1)
    assert np.allclose(summary.split10_rank,1)
    assert (summary[summary.feature=='sr_value'].paired_valid==0).all()
    assert (summary[summary.feature=='cr_value'].delta_mean==2).all()
    assert (summary.median_within_context_sd>0).all()
