"""
Signal cleaning: restrict to the common channel set, remove power-line
interference, and band-limit to the frequency range where seizure-relevant
EEG activity lives.
"""


def preprocess_raw(raw, channels, l_freq=1.0, h_freq=40.0, notch_freq=60.0):
    """
    Return a filtered copy of `raw` restricted to `channels`.

    Applied to the FULL recording before windowing (not per-window) to
    avoid filter edge artifacts at every window boundary.

    - notch_freq=60Hz removes US power-line interference (CHB-MIT was
      recorded at Boston Children's Hospital).
    - l_freq/h_freq band-limits to 1-40Hz, covering delta through gamma
      band activity relevant to seizures while dropping slow drift and
      high-frequency muscle/electrical noise.
    """
    raw = raw.copy()
    raw.pick(channels)
    raw.notch_filter(freqs=notch_freq, verbose=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    return raw