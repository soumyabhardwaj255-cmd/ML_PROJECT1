"""
Frequency-domain features: band power via Welch's method, computed per
channel. Seizures often shift energy toward slower frequencies.

Band edges follow standard EEG convention, EXCEPT gamma is narrowed to
30-40Hz (instead of the usual 30-45Hz) to match our preprocessing bandpass
filter, which cuts everything above 40Hz — there's no signal left above
that to measure.
"""

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}


def _band_power(freqs, psd, band):
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    if not np.any(idx):
        return 0.0
    return trapezoid(psd[idx], freqs[idx])


def extract_frequency_domain_features(window_data, sfreq, channel_names, nperseg=256):
    """
    Parameters
    ----------
    window_data : np.ndarray, shape (n_channels, n_samples)
    sfreq : float
        Sampling rate in Hz.
    channel_names : list[str]
    nperseg : int
        Welch segment length. 256 samples = 1 second at 256Hz, giving 1Hz
        frequency resolution — fine-grained enough for our band edges.

    Returns
    -------
    dict
        Keys like "freq__delta__C3-P3", one entry per channel per band.
    """
    features = {}
    for ch_idx, ch_name in enumerate(channel_names):
        x = window_data[ch_idx]
        freqs, psd = welch(x, fs=sfreq, nperseg=min(nperseg, len(x)))
        for band_name, band_range in BANDS.items():
            features[f"freq__{band_name}__{ch_name}"] = _band_power(freqs, psd, band_range)
    return features