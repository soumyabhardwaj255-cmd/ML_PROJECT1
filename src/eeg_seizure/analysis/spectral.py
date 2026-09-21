"""Explicit Welch PSD diagnostics and the notebook's FFT coloured noise."""
import numpy as np
from scipy.signal import welch


def signal_array(signal):
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite one-dimensional signal with at least two samples")
    return x


def psd(signal, sfreq, nperseg=256):
    x = signal_array(signal)
    if sfreq <= 0 or nperseg < 2:
        raise ValueError("Invalid PSD configuration")
    size = min(int(nperseg), len(x))
    return welch(x, fs=sfreq, window="hann", nperseg=size, noverlap=size // 2,
                 nfft=size, detrend="constant", scaling="density", average="mean")


def spectral_snr(signal, sfreq, target_frequency):
    """Notebook local mean-PSD ratio, not broadband reconstruction SNR."""
    f, p = psd(signal, sfreq, len(signal))
    signal_band = np.abs(f - target_frequency) <= .2
    noise_band = (np.abs(f - target_frequency) > .2) & (np.abs(f - target_frequency) <= 2.)
    if not signal_band.any() or not noise_band.any():
        return float("nan")
    numerator, denominator = p[signal_band].mean(), p[noise_band].mean()
    return float(10 * np.log10(numerator / denominator)) if numerator > 0 and denominator > 0 else float("nan")


def spectral_slope(signal, sfreq):
    f, p = psd(signal, sfreq, min(len(signal), int(sfreq * 2)))
    keep = (f >= 1) & (f <= min(70, sfreq / 2)) & (p > 0)
    return float(np.polyfit(np.log10(f[keep]), np.log10(p[keep]), 1)[0]) if keep.sum() >= 3 else float("nan")


def colored_noise(n_samples, sfreq, beta, rng):
    """Notebook FFT generator: beta 0/1/3; zero mean and unit standard deviation."""
    if n_samples < 4 or sfreq <= 0 or beta not in (0, 1, 3):
        raise ValueError("Unsupported coloured-noise configuration")
    f = np.fft.rfftfreq(n_samples, d=1 / sfreq)
    spectrum = rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))
    spectrum[1:] *= f[1:] ** (-beta / 2)
    spectrum[0] = 0
    noise = np.fft.irfft(spectrum, n=n_samples)
    noise -= noise.mean()
    return noise / noise.std()


def add_noise(signal, sfreq, beta, ratio, rng):
    """Post-filter additive perturbation; ratio is noise SD / input SD."""
    x = signal_array(signal)
    if ratio < 0:
        raise ValueError("Noise ratio cannot be negative")
    return x + ratio * x.std() * colored_noise(len(x), sfreq, beta, rng)
