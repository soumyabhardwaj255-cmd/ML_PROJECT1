"""
Entropy features: how predictable or irregular each channel's signal is.
Rhythmic seizure activity often has LOWER entropy than normal background
activity, since the waveform becomes more repetitive.
"""

import antropy as ant


def extract_entropy_features(window_data, sfreq, channel_names):
    """
    Parameters
    ----------
    window_data : np.ndarray, shape (n_channels, n_samples)
    sfreq : float
    channel_names : list[str]

    Returns
    -------
    dict
        Keys like "entropy__perm__C3-P3", one entry per channel per measure.
    """
    features = {}
    for ch_idx, ch_name in enumerate(channel_names):
        x = window_data[ch_idx]
        features[f"entropy__perm__{ch_name}"] = ant.perm_entropy(x, normalize=True)
        features[f"entropy__spectral__{ch_name}"] = ant.spectral_entropy(
            x, sf=sfreq, method="welch", normalize=True
        )
    return features