"""Duration extension boundary/label parity and PSD validity checks."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from eeg_seizure.dataset import window_rows
from eeg_seizure import config as C
from eeg_seizure.analysis.spectral import psd

spec = importlib.util.spec_from_file_location("feasibility", Path(__file__).parents[1]/"scripts/window_feasibility.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_duration_boundaries_annotations_and_canonical_parity():
    intervals = [(5, 10), (20, 26)]
    for n in (0, 2559, 2560, 2561, 7680):
        table = module.duration_windows(n, 256, intervals, "p", "r", 10)
        assert len(table) == n // 2560
        if len(table):
            assert table.iloc[-1].end_sec <= n / 256
        four = module.duration_windows(n, 256, intervals, "p", "r", 4)
        pd.testing.assert_frame_equal(four[C.META], window_rows(n, 256, intervals, "p", "r"))
    table = module.duration_windows(7680, 256, intervals, "p", "r", 10)
    assert table.label.tolist() == [1, 0, 1]
    assert table.overlap_sec.tolist() == [5, 0, 6]
    assert table.mixed.tolist() == [True, False, True]
    assert C.WINDOW_SEC == 4


def test_psd_validity_and_scalar_parity():
    t = np.arange(2560)/256
    matrix = np.stack([np.sin(2*np.pi*10*t), np.zeros_like(t)])
    assert module.psd_validity(matrix, 256).tolist() == [True, False]
    for x in matrix:
        _, p = psd(x, 256)
        assert module.psd_validity(x[None,:], 256)[0] == (np.isfinite(p).all() and (p>=0).all() and p.sum()>0)
