"""Sample-rate conversion of a whole decoded file."""

import numpy as np


def resample_linear(samples: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    """Linear interpolation of ``samples`` [frames, channels] to ``target_sample_rate``.

    Output frame k is the input read at position k * source/target, up to the
    last input frame. Done on the whole array at once: a file is decoded into
    memory before ACE reads it.
    """
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    src = max(int(source_sample_rate), 1)
    dst = max(int(target_sample_rate), 1)
    frames, channels = int(arr.shape[0]), int(arr.shape[1])
    if frames == 0 or src == dst:
        return np.ascontiguousarray(arr)
    step = float(src) / float(dst)
    count = int(np.floor((frames - 1) / step + 1e-9)) + 1
    positions = np.arange(count, dtype=np.float64) * step
    x_old = np.arange(frames, dtype=np.float64)
    out = np.empty((count, channels), dtype=np.float32)
    for ch in range(channels):
        out[:, ch] = np.interp(positions, x_old, arr[:, ch].astype(np.float64, copy=False))
    return out
