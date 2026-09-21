"""
Time-domain features: simple statistics describing the shape of the raw
signal over time, computed per channel. Seizures often produce larger,
more rapidly-changing signals, so these tend to shift during seizures.
"""

import numpy as np


def extract_time_domain_features(window_data, channel_names):
    """
    Parameters
    ----------
    window_data : np.ndarray, shape (n_channels, n_samples)
    channel_names : list[str]

    Returns
    -------
    dict
        Keys like "time__mean__C3-P3", one entry per channel per statistic.
    """
    features = {}
    for ch_idx, ch_name in enumerate(channel_names):
        x = window_data[ch_idx]
        features[f"time__mean__{ch_name}"] = np.mean(x)
        features[f"time__std__{ch_name}"] = np.std(x)
        features[f"time__rms__{ch_name}"] = np.sqrt(np.mean(x ** 2))
        features[f"time__ptp__{ch_name}"] = np.ptp(x)
        features[f"time__line_length__{ch_name}"] = np.sum(np.abs(np.diff(x)))
    return features