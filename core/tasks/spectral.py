"""Spectrum, masking and level helpers shared by tasks."""

import math
from typing import Optional, Tuple
import numpy as np
from ... import utils
from .base import _LAYER3_THREAD_LOCAL
from .chunk import AudioChunk, SAMPLE_RATE, _analysis_signal_for_chunk, _finite_copy, _preview_signal_for_chunk


def _aligned_signal_length(signal: np.ndarray, target_len: int) -> np.ndarray:
    target = max(int(target_len or 0), 0)
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim == 1:
        src = arr.reshape(-1)
        if src.shape[0] == target:
            return src.astype(np.float32, copy=False)
        if src.shape[0] > target:
            return src[:target].astype(np.float32, copy=False)
        out = np.zeros(target, dtype=np.float32)
        if src.shape[0] > 0:
            out[: src.shape[0]] = src
        return out

    if arr.ndim == 2:
        if arr.shape[1] == target:
            return arr.astype(np.float32, copy=False)
        if arr.shape[1] > target:
            return arr[:, :target].astype(np.float32, copy=False)
        out = np.zeros((arr.shape[0], target), dtype=np.float32)
        if arr.shape[1] > 0:
            out[:, : arr.shape[1]] = arr
        return out

    flat = arr.reshape(-1)
    return _aligned_signal_length(flat, target)


def _analysis_signals_normalized(
    chunk: AudioChunk,
    prefer_analysis_track: bool = True,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if prefer_analysis_track:
        mono_sig, stereo_sig = _analysis_signal_for_chunk(chunk)
    else:
        mono_sig, stereo_sig = _preview_signal_for_chunk(chunk)
    mono = np.asarray(mono_sig, dtype=np.float32).reshape(-1).astype(np.float32, copy=False)
    if mono.size > 0:
        mono = _finite_copy(mono)

    stereo: Optional[np.ndarray] = None
    if isinstance(stereo_sig, np.ndarray):
        arr = np.asarray(stereo_sig, dtype=np.float32)
        if arr.ndim == 2:
            if arr.shape[1] == mono.shape[0]:
                stereo = arr.astype(np.float32, copy=False)
            elif arr.shape[0] == mono.shape[0]:
                stereo = arr.T.astype(np.float32, copy=False)
            if stereo is not None and stereo.shape[1] != mono.shape[0]:
                target = min(int(stereo.shape[1]), int(mono.shape[0]))
                if target <= 0:
                    stereo = None
                    mono = np.zeros(0, dtype=np.float32)
                else:
                    stereo = stereo[:, :target].astype(np.float32, copy=False)
                    mono = mono[:target].astype(np.float32, copy=False)
            if stereo is not None and stereo.size > 0:
                stereo = _finite_copy(stereo)
    return mono, stereo


def _safe_rms(signal: np.ndarray) -> float:
    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    arr = _finite_copy(arr, dtype=np.float64)
    peak = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
    if (not np.isfinite(peak)) or peak <= 1e-20:
        return 0.0
    scaled = arr / peak
    mean_square = float(np.mean(np.square(scaled, dtype=np.float64)))
    if (not np.isfinite(mean_square)) or mean_square <= 0.0:
        return 0.0
    return float(math.sqrt(mean_square) * peak)


def _zero_crossing_rate(mono: np.ndarray, epsilon: float = 1e-6) -> float:
    signal = np.asarray(mono, dtype=np.float32).reshape(-1)
    n = int(signal.shape[0] or 0)
    if n < 2:
        return 0.0

    if signal.size > 0:
        signal = _finite_copy(signal)
    eps = float(max(epsilon, 0.0))
    signs = np.zeros(n, dtype=np.int8)
    if eps > 0.0:
        signs[signal > eps] = 1
        signs[signal < -eps] = -1
    else:
        signs[signal > 0.0] = 1
        signs[signal < 0.0] = -1
    zero_mask = signs == 0
    if np.any(zero_mask):
        nz = np.flatnonzero(~zero_mask)
        if nz.size == 0:
            return 0.0
        first = int(nz[0])
        # Forward-fill sign values entirely in NumPy (no Python loop).
        fill_idx = np.zeros(n, dtype=np.intp)
        fill_idx[nz] = nz
        if first > 0:
            fill_idx[:first] = first
        np.maximum.accumulate(fill_idx, out=fill_idx)
        signs = signs[fill_idx]

    crossings = int(np.count_nonzero(signs[1:] != signs[:-1]))
    return float(crossings / float(max(n - 1, 1)))


def _chunk_has_mask_constraints(chunk: AudioChunk) -> bool:
    if getattr(chunk, "band_hz_bounds", None) is not None:
        return True
    return getattr(chunk, "band_indices", None) is not None


# Layer 2 only: inverse-FFT reconstruction, never reached from the preview path.
def _masked_mono_signal(chunk: AudioChunk, fft_size: Optional[int] = None) -> np.ndarray:
    # Runtime guard: _LAYER3_THREAD_LOCAL.in_preview is True when called from
    # PreviewSinkTask.process(). Reaching here from Layer 3 is a guardrail violation.
    if getattr(_LAYER3_THREAD_LOCAL, "in_preview", False):
        utils.log(
            "_masked_mono_signal called from Layer-3 preview path — "
            "iFFT reconstruction must not run in PreviewSinkTask. "
            "This is a guardrail violation; returning silence.",
            "ERROR",
        )
        n = int(fft_size) if fft_size is not None else int(getattr(chunk.mono, "shape", (0,))[-1] or 0)
        return np.zeros(max(n, 1), dtype=np.float32)
    analysis_mono, _ = _analysis_signals_normalized(chunk)
    if analysis_mono.size == 0:
        return np.zeros(0, dtype=np.float32)

    n = int(fft_size) if fft_size is not None else int(analysis_mono.shape[-1] or 0)
    n = max(n, 1)
    idx = _resolve_effective_bin_indices(chunk, n)
    has_constraints = _chunk_has_mask_constraints(chunk)
    if idx is None:
        if has_constraints:
            return np.zeros(n, dtype=np.float32)
        if int(analysis_mono.shape[-1] or 0) == n:
            return analysis_mono.astype(np.float32, copy=False)
        return _aligned_signal_length(analysis_mono, n)

    idx_arr = np.asarray(idx, dtype=np.int64).ravel()
    if idx_arr.size == 0:
        return np.zeros(n, dtype=np.float32)

    src = _aligned_signal_length(analysis_mono, n)
    spec = np.fft.rfft(src, n=n)
    masked = np.zeros_like(spec)
    masked[idx_arr] = spec[idx_arr]
    return np.fft.irfft(masked, n=n).astype(np.float32, copy=False)


def _masked_spectrum(chunk: AudioChunk, fft_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    analysis_mono, _ = _analysis_signals_normalized(chunk)
    n = int(fft_size) if fft_size is not None else int(analysis_mono.shape[-1] or 0)
    n = max(n, 1)
    mono = _aligned_signal_length(analysis_mono, n)
    spec = np.fft.rfft(mono, n=n)
    mag = np.abs(spec).astype(np.float32, copy=False)
    sr = float(max(getattr(chunk, "sample_rate", SAMPLE_RATE), 1.0))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr).astype(np.float32, copy=False)

    idx = _resolve_effective_bin_indices(chunk, n)
    has_constraints = _chunk_has_mask_constraints(chunk)
    if idx is None:
        if has_constraints:
            return np.zeros_like(mag, dtype=np.float32), freqs
        return mag, freqs

    idx_arr = np.asarray(idx, dtype=np.int64).ravel()
    if idx_arr.size == 0:
        return np.zeros_like(mag, dtype=np.float32), freqs
    masked_mag = np.zeros_like(mag, dtype=np.float32)
    masked_mag[idx_arr] = mag[idx_arr]
    return masked_mag, freqs


def _autocorr_pitch_estimate(
    mono: np.ndarray,
    sample_rate: int,
    min_hz: float = 40.0,
    max_hz: float = 2000.0,
) -> float:
    signal = np.asarray(mono, dtype=np.float32).reshape(-1)
    n = int(signal.shape[0] or 0)
    if n < 8:
        return 0.0

    signal = signal - float(np.mean(signal))
    energy = float(np.sum(signal * signal))
    if energy <= 1e-8:
        return 0.0

    sr = float(max(int(sample_rate), 1))
    min_lag = max(int(sr / max(float(max_hz), 1.0)), 1)
    max_lag = min(int(sr / max(float(min_hz), 1.0)), n - 1)
    if max_lag <= min_lag:
        return 0.0

    corr = np.correlate(signal, signal, mode="full")[n - 1:]
    corr = corr.astype(np.float64, copy=False)
    if corr.shape[0] <= max_lag:
        return 0.0
    corr[:min_lag] = 0.0
    lag_slice = corr[min_lag:max_lag + 1]
    if lag_slice.size == 0:
        return 0.0
    best_lag = int(np.argmax(lag_slice)) + min_lag
    peak = float(corr[best_lag])
    if peak <= 0.0:
        return 0.0
    if peak < (0.12 * float(corr[0])):
        return 0.0

    pitch = sr / float(best_lag)
    if not np.isfinite(pitch):
        return 0.0
    return float(np.clip(pitch, 20.0, 5000.0))


def _get_rfft(chunk: AudioChunk, fft_size: int):
    key = f"rfft_{fft_size}"
    if key in chunk.features:
        return chunk.features[key]

    mono_sig, stereo_sig = _analysis_signal_for_chunk(chunk)
    mono_fft = np.fft.rfft(mono_sig, n=fft_size)
    stereo_fft = None
    if stereo_sig is not None and stereo_sig.shape[0] >= 2:
        stereo_fft = np.fft.rfft(stereo_sig, n=fft_size, axis=1)

    freq_bins = np.fft.rfftfreq(fft_size, d=1.0 / chunk.sample_rate)
    chunk.features[key] = (mono_fft, stereo_fft, freq_bins)
    return chunk.features[key]


def _remap_indices_fft_domain(
    indices: np.ndarray,
    src_fft_size: int,
    dst_fft_size: int,
    sample_rate: int,
) -> np.ndarray:
    src_n = max(int(src_fft_size), 1)
    dst_n = max(int(dst_fft_size), 1)
    if src_n == dst_n:
        return np.unique(np.asarray(indices, dtype=np.int64).ravel())

    arr = np.asarray(indices, dtype=np.int64).ravel()
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)

    src_max_bin = (src_n // 2) + 1
    arr = arr[(arr >= 0) & (arr < src_max_bin)]
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)

    sr = float(max(int(sample_rate), 1))
    src_freqs = np.fft.rfftfreq(src_n, d=1.0 / sr)
    dst_freqs = np.fft.rfftfreq(dst_n, d=1.0 / sr)
    mapped_freqs = src_freqs[arr]

    right = np.searchsorted(dst_freqs, mapped_freqs, side="left")
    right = np.clip(right, 0, len(dst_freqs) - 1)
    left = np.clip(right - 1, 0, len(dst_freqs) - 1)
    right_err = np.abs(dst_freqs[right] - mapped_freqs)
    left_err = np.abs(dst_freqs[left] - mapped_freqs)
    mapped = np.where(left_err <= right_err, left, right).astype(np.int64, copy=False)
    return np.unique(mapped)


def _resolve_chunk_fft_size(chunk: AudioChunk, fallback_size: int) -> int:
    provider = getattr(chunk, "provider", None)
    analyzer = getattr(provider, "analysis", None) if provider is not None else None
    fft_len = getattr(analyzer, "fft_length", None)
    if fft_len:
        with utils.logged_errors("spectral._resolve_chunk_fft_size"):
            return max(int(fft_len), 1)
    return max(int(fallback_size), 1)


def _resolve_effective_bin_indices(chunk: AudioChunk, fft_size: int) -> Optional[np.ndarray]:
    """Resolve the effective FFT-bin mask represented by a chunk: intersection of explicit `band_indices` and `band_hz_bounds` when both exist."""
    idx = None
    raw_idx = getattr(chunk, "band_indices", None)
    if raw_idx is not None:
        try:
            arr = np.asarray(raw_idx, dtype=np.int64).ravel()
        except Exception as exc:
            utils.note_swallowed("spectral._resolve_effective_bin_indices", exc)
            arr = np.zeros(0, dtype=np.int64)
        src_fft_size = getattr(chunk, "band_index_fft_size", None)
        if src_fft_size is None:
            src_fft_size = max(int(getattr(chunk, "mono", np.zeros(1, dtype=np.float32)).shape[-1] or 1), 1)
        try:
            src_fft_size = max(int(src_fft_size), 1)
        except Exception as exc:
            utils.note_swallowed("spectral._resolve_effective_bin_indices#2", exc)
            src_fft_size = max(int(fft_size), 1)
        idx = _remap_indices_fft_domain(
            arr,
            src_fft_size=src_fft_size,
            dst_fft_size=max(int(fft_size), 1),
            sample_rate=int(max(getattr(chunk, "sample_rate", SAMPLE_RATE), 1)),
        )

    bounds = getattr(chunk, "band_hz_bounds", None)
    if bounds is not None:
        try:
            low_hz = float(bounds[0])
            high_hz = float(bounds[1])
        except Exception as exc:
            utils.note_swallowed("spectral._resolve_effective_bin_indices#3", exc)
            low_hz = 0.0
            high_hz = 0.0
        if high_hz > low_hz:
            sr = float(max(getattr(chunk, "sample_rate", SAMPLE_RATE), 1.0))
            freqs = np.fft.rfftfreq(max(int(fft_size), 1), d=1.0 / sr)
            hz_idx = np.nonzero((freqs >= low_hz) & (freqs < high_hz))[0].astype(np.int64, copy=False)
            if idx is None:
                idx = hz_idx
            else:
                idx = np.intersect1d(idx, hz_idx, assume_unique=False)

    if idx is None:
        return None
    return idx.astype(np.int64, copy=False)
