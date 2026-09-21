"""Sample-reduction/reconstruction baselines; distinct from model compression."""
import numpy as np
from scipy.signal import resample_poly
from .spectral import signal_array


def reconstruct(signal, factor, method="legacy_stride"):
    x = signal_array(signal)
    if int(factor) != factor or factor < 1 or factor > len(x):
        raise ValueError("Invalid sample-reduction factor")
    if method == "legacy_stride":
        compressed = x[::factor]
    elif method == "antialiased":
        compressed = resample_poly(x, up=1, down=factor)
    else:
        raise ValueError("Unknown reconstruction method")
    restored = np.interp(np.arange(len(x)), np.arange(len(compressed))*factor, compressed)
    return restored, dict(method=method, factor=factor, original_samples=len(x),
                          retained_samples=len(compressed), sample_ratio=len(x)/len(compressed),
                          endpoint_policy="linear interpolation; final unobserved tail held constant")


def preservation(original, restored):
    x, y = signal_array(original), signal_array(restored)
    if x.shape != y.shape:
        raise ValueError("Signal lengths differ")
    rmse = float(np.sqrt(np.mean((x-y)**2)))
    return dict(rmse_volts=rmse, normalized_rmse=rmse/x.std() if x.std() else float("nan"),
                correlation=float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else float("nan"))
