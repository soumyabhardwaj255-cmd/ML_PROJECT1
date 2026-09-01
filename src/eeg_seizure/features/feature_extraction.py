"""
Combine all four feature families into one flat dict per window. Column
names are prefixed by family ("time__", "freq__", "entropy__", "hjorth__")
so feature ablation experiment can select a family just by
filtering columns on this prefix — no re-extraction needed.
"""

from .entropy_domain import extract_entropy_features
from .frequency_domain import extract_frequency_domain_features
from .hjorth_domain import extract_hjorth_features
from .time_domain import extract_time_domain_features


def extract_window_features(window_data, sfreq, channel_names):
    """
    Parameters
    ----------
    window_data : np.ndarray, shape (n_channels, n_samples)
        Already-filtered signal for one window.
    sfreq : float
    channel_names : list[str]

    Returns
    -------
    dict
        All features for this window, family-tagged keys, no collisions
        between families since each uses a distinct prefix.
    """
    features = {}
    features.update(extract_time_domain_features(window_data, channel_names))
    features.update(extract_frequency_domain_features(window_data, sfreq, channel_names))
    features.update(extract_entropy_features(window_data, sfreq, channel_names))
    features.update(extract_hjorth_features(window_data, channel_names))
    return features