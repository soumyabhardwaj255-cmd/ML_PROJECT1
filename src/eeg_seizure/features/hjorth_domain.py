"""
Hjorth parameters: Activity (signal variance), Mobility (rate of change),
and Complexity (how much more complicated the waveform is than a simple
repeating signal). Simple to compute, still commonly used decades later.
"""

import numpy as np
import antropy as ant


def extract_hjorth_features(window_data, channel_names):
    """
    Parameters
    ----------
    window_data : np.ndarray, shape (n_channels, n_samples)
    channel_names : list[str]

    Returns
    -------
    dict
        Keys like "hjorth__mobility__C3-P3", one entry per channel per param.
    """
    features = {}
    for ch_idx, ch_name in enumerate(channel_names):
        x = window_data[ch_idx]
        mobility, complexity = ant.hjorth_params(x)
        features[f"hjorth__activity__{ch_name}"] = np.var(x)
        features[f"hjorth__mobility__{ch_name}"] = mobility
        features[f"hjorth__complexity__{ch_name}"] = complexity
    return features