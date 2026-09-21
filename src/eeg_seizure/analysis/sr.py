"""Notebook bistable SR, with explicit drive, RNG and integration diagnostics."""
import numpy as np
from scipy.signal import butter, filtfilt
from .spectral import signal_array, psd


def normalize(signal):
    x = signal_array(signal)
    scale = x.std()
    return (x - x.mean()) / scale if scale > 0 else np.zeros_like(x)


def characteristic_component(signal, sfreq, frequency=None, bandwidth=.5):
    """One label-independent rule; no seizure/non-seizure frequency constants."""
    x = signal_array(signal)
    if frequency is None:
        f, p = psd(x, sfreq, len(x))
        keep = (f >= 1) & (f <= min(30, sfreq / 2 - 1))
        frequency = float(f[keep][np.argmax(p[keep])])
    low, high = max(.2, frequency - bandwidth), min(sfreq / 2 - 1, frequency + bandwidth)
    if not 0 < low < high < sfreq / 2:
        raise ValueError("Invalid characteristic component band")
    b, a = butter(3, [low, high], btype="bandpass", fs=sfreq)
    return normalize(filtfilt(b, a, normalize(x))), float(frequency)


def bistable(signal, sfreq, noise_intensity, rng, input_scale=.2, substeps=1):
    """dx=(x-x^3+drive)dt+sqrt(2Ddt)N(0,1), legacy clip at +/-3.

    substeps=1 preserves the notebook update for the same supplied drive/RNG.
    The drive is held constant over each sample during substeps.
    """
    drive = signal_array(signal)
    if sfreq <= 0 or noise_intensity < 0 or int(substeps) != substeps or substeps < 1:
        raise ValueError("Invalid stochastic integration configuration")
    dt = 1 / (sfreq * substeps)
    x = np.empty(len(drive)); x[0] = -1.
    clipped = 0
    for i in range(1, len(x)):
        value = x[i-1]
        for _ in range(substeps):
            proposal = value + dt * (value - value**3 + input_scale * drive[i-1]) + np.sqrt(2 * noise_intensity * dt) * rng.normal()
            clipped += int(abs(proposal) > 3)
            value = np.clip(proposal, -3., 3.)
        x[i] = value
    return x, dict(clipped_fraction=clipped / ((len(x)-1)*substeps), substeps=substeps,
                   noise_convention="sqrt(2Ddt)", initial_state=-1., clip=3.)


def voltage_referenced_transform(signal, sfreq, noise_intensity, rng, frequency=None):
    """Exploratory waveform ablation, NOT a validated denoiser.

    Narrowband SR output is normalized and mapped to the original window's
    mean/SD so dimensionless oscillator state is not mislabelled as volts.
    No training/test cohort statistics or class labels enter this mapping.
    """
    original = signal_array(signal)
    component, frequency = characteristic_component(original, sfreq, frequency)
    response, diagnostic = bistable(component, sfreq, noise_intensity, rng)
    mapped = original.mean() + original.std() * normalize(response)
    return mapped, {**diagnostic, "target_frequency": frequency,
                    "mapping": "per-window original mean and SD; exploratory"}
