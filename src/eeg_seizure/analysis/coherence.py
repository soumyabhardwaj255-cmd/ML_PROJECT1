"""Driven bistable regularity and separately named FHN coherence diagnostics."""
import numpy as np
from scipy.signal import find_peaks
from .spectral import signal_array


def fhn(signal, sfreq, noise_intensity, rng, input_scale=.35, substeps=1, convention="sde"):
    """Preserve FHN drift/initial states; correct noise placement by default.

    legacy_extra_dt retains the notebook's extra dt for explicit comparisons.
    Neither variant is automatically evidence of coherence resonance.
    """
    drive = signal_array(signal)
    if sfreq <= 0 or noise_intensity < 0 or substeps < 1 or int(substeps) != substeps:
        raise ValueError("Invalid FHN configuration")
    if convention not in ("sde", "legacy_extra_dt"):
        raise ValueError("Unknown FHN noise convention")
    dt = 1 / (sfreq * substeps)
    v = np.empty(len(drive)); v[0] = -1.
    w = -.5
    clipped = 0
    for i in range(1, len(v)):
        value = v[i-1]
        for _ in range(substeps):
            noise = np.sqrt(2 * noise_intensity * dt) * rng.normal()
            if convention == "legacy_extra_dt":
                noise *= dt
            proposed = value + dt * (value - value**3 / 3 - w + input_scale * drive[i-1]) + noise
            proposed_w = w + dt * .08 * (value + .7 - .8 * w)
            clipped += int(abs(proposed) > 4 or abs(proposed_w) > 4)
            value, w = np.clip(proposed, -4., 4.), np.clip(proposed_w, -4., 4.)
        v[i] = value
    return v, dict(clipped_fraction=clipped/((len(v)-1)*substeps), substeps=substeps, noise_convention=convention)


def transitions(output, sfreq, threshold=.5, min_dwell_sec=.1):
    """Hysteresis plus refractory separation, not proof of sustained dwell."""
    x = signal_array(output)
    if threshold <= 0 or sfreq <= 0 or min_dwell_sec < 0:
        raise ValueError("Invalid event detector")
    dwell = max(1, int(min_dwell_sec * sfreq))
    reached = np.flatnonzero(np.abs(x) >= threshold)
    if not len(reached):
        return np.array([], dtype=int)
    first = reached[0]
    state = 1 if x[first] >= threshold else -1
    last = -dwell
    events = []
    # Start after the actual initial-state observation, never scan backwards.
    for i in range(first + 1, len(x)):
        crossed = (state == -1 and x[i] >= threshold) or (state == 1 and x[i] <= -threshold)
        if crossed and i-last >= dwell:
            events.append(i); state *= -1; last = i
    return np.asarray(events, dtype=int)


def fhn_events(output, sfreq):
    """Notebook FHN peak detector, distinct from bistable hysteresis crossings."""
    x=signal_array(output)
    return find_peaks(x,height=x.mean()+x.std(),distance=max(1,int(.2*sfreq)))[0]


def regularity(events, sfreq, minimum_intervals=5, ddof=1, minimum_interval_sec=0.):
    events = np.asarray(events, dtype=int)
    if sfreq <= 0 or minimum_intervals < 2 or ddof not in (0,1) or minimum_interval_sec < 0:
        raise ValueError("Invalid interval metric configuration")
    intervals = np.diff(events) / sfreq
    if np.any(intervals <= 0):
        raise ValueError("Invalid transition intervals")
    discarded=int((intervals<minimum_interval_sec).sum())
    intervals=intervals[intervals>=minimum_interval_sec]
    result = dict(events=len(events), intervals=len(intervals), discarded_intervals=discarded, ddof=ddof,
                  cv=float("nan"), coherence=float("nan"), valid=False)
    if len(intervals) < minimum_intervals:
        return {**result, "reason": "insufficient_intervals"}
    cv = float(intervals.std(ddof=ddof) / intervals.mean())
    if cv == 0:
        return {**result, "cv": 0., "coherence": float("inf"), "reason": "zero_cv_unbounded"}
    return {**result, "cv": cv, "coherence": 1/cv, "valid": True, "reason": "ok"}
