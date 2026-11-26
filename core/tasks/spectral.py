"""Spectrum, masking and level helpers shared by tasks."""

import math
from typing import Optional, Tuple
import numpy as np
from ... import utils
from .base import _LAYER3_THREAD_LOCAL
from .chunk import AudioChunk, SAMPLE_RATE, _analysis_signal_for_chunk, _apply_analysis_ops, _finite_copy, _normalized_analysis_ops, _preview_signal_for_chunk
from .frontend import SourceFrontendProcessor


# Pitch: the newest samples are compared with the signal one period earlier (a 21 ms window at
# 48 kHz); the frame is long enough for periods down to 20 Hz.
_PITCH_WINDOW = 1024
_PITCH_FRAME = 4096
_PITCH_THRESHOLD = 0.15


# Band level: the fade at each end of the analysed frame, in samples (5 ms at 48 kHz).
_BAND_FADE = 256


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


def _analysis_history(chunk: AudioChunk) -> Optional[np.ndarray]:
    """The analysis signal before the chunk's block, processed like the block, or None."""
    if "analysis_history" in chunk.features:
        return chunk.features["analysis_history"]
    mono = chunk.history_mono
    stereo = chunk.history_stereo
    if mono is not None:
        if chunk.analysis_mono is not None and chunk.frontend_config is not None:
            # A fresh filter, so the result depends only on the audio, not on what played before.
            mono, stereo = SourceFrontendProcessor().process(mono, stereo, int(chunk.sample_rate), chunk.frontend_config)
        ops = _normalized_analysis_ops(chunk)
        if ops:
            mono, stereo = _apply_analysis_ops(mono, stereo, ops)
        mono = _finite_copy(mono)
    chunk.features["analysis_history"] = mono
    return mono


def _analysis_frame(chunk: AudioChunk, length: int) -> Tuple[np.ndarray, int]:
    """The last *length* samples of the analysis signal, ending with this block.

    Also returns how many of them are audio: a chunk without history only has
    its block, and the rest of the frame is zero padding.
    """
    mono, _ = _analysis_signals_normalized(chunk)
    mono = mono[-length:]
    need = length - mono.size
    if need <= 0:
        return mono, length
    frame = np.zeros(length, dtype=np.float32)
    frame[need:] = mono
    history = _analysis_history(chunk)
    if history is None:
        return frame, int(mono.size)
    tail = history[-need:]
    frame[need - tail.size:need] = tail
    return frame, length


def _yin_pitch(frame: np.ndarray, sample_rate: int, min_hz: float = 20.0, max_hz: float = 5000.0) -> float:
    """Fundamental frequency at the end of *frame* in Hz, 0.0 when it has none.

    YIN (de Cheveigne and Kawahara, 2002): the shortest lag at which the newest
    samples repeat the earlier signal, by the cumulative mean normalized
    difference. Only the newest samples are compared, so the estimate stays
    close to the end of the frame.
    """
    x = np.asarray(frame, dtype=np.float64).reshape(-1)
    window = min(_PITCH_WINDOW, x.size // 2)
    sr = float(max(int(sample_rate), 1))
    tau_min = max(int(sr / max_hz), 2)
    tau_max = min(int(sr / min_hz) + 1, x.size - window)
    if tau_max <= tau_min + 1:
        return 0.0
    head = x[x.size - window:]
    head_energy = float(np.dot(head, head))
    if head_energy <= 1e-8 * window:  # below -80 dBFS
        return 0.0
    seg = x[x.size - window - tau_max:]
    size = 1 << int(math.ceil(math.log2(seg.size)))
    # corr[k] = sum(head * seg[k:k + window]); the lag of offset k is tau_max - k.
    corr = np.fft.irfft(np.conj(np.fft.rfft(head, size)) * np.fft.rfft(seg, size), size)[:tau_max + 1]
    energy = np.concatenate(([0.0], np.cumsum(seg * seg)))
    seg_energy = energy[window:window + tau_max + 1] - energy[:tau_max + 1]
    diff = np.maximum(head_energy + seg_energy - 2.0 * corr, 0.0)[::-1]  # by lag, 0 first
    normalized = np.ones_like(diff)
    normalized[1:] = diff[1:] * np.arange(1, diff.size) / np.maximum(np.cumsum(diff[1:]), 1e-20)
    below = np.nonzero(normalized[tau_min:tau_max] < _PITCH_THRESHOLD)[0]
    if below.size == 0:
        return 0.0
    tau = int(below[0]) + tau_min
    while tau + 1 < tau_max and normalized[tau + 1] < normalized[tau]:
        tau += 1
    before, at, after = diff[tau - 1], diff[tau], diff[tau + 1]
    curve = before - 2.0 * at + after
    shift = float(np.clip(0.5 * (before - after) / curve, -1.0, 1.0)) if curve > 0.0 else 0.0
    return float(sr / (tau + shift))


def _chunk_pitch(chunk: AudioChunk) -> float:
    """Pitch of the chunk's analysis signal, inside its band when it has one."""
    frame, _ = _analysis_frame(chunk, _PITCH_FRAME)
    sr = int(getattr(chunk, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
    if _chunk_has_mask_constraints(chunk):
        idx = _resolve_effective_bin_indices(chunk, frame.size)
        if idx is None or idx.size == 0:
            return 0.0
        spec = np.fft.rfft(frame)
        masked = np.zeros_like(spec)
        masked[idx] = spec[idx]
        frame = np.fft.irfft(masked, n=frame.size)
    return _yin_pitch(frame, sr, max_hz=min(5000.0, sr / 2.0 - 10.0))


def _band_rms(chunk: AudioChunk) -> float:
    """RMS of the part of the block's signal inside the chunk's band (Parseval).

    The frame starts a fade before the block and fades out at its end. The
    squares of the two fades add up to 1, so across consecutive blocks every
    sample counts exactly once: a click on a block boundary is not lost, and
    the bands of a split add up to the full level. A chunk without history is
    measured on its block alone.
    """
    mono, _ = _analysis_signals_normalized(chunk)
    block = int(mono.size)
    if block == 0:
        return 0.0
    fade = min(_BAND_FADE, block // 2) if chunk.history_mono is not None else 0
    frame = _analysis_frame(chunk, block + fade)[0].astype(np.float64)
    if fade:
        rise = np.sin(0.5 * np.pi * (np.arange(fade) + 0.5) / fade)
        frame[:fade] *= rise
        frame[-fade:] *= rise[::-1]
    n = int(frame.size)
    sr = float(max(getattr(chunk, "sample_rate", SAMPLE_RATE), 1.0))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    inside = np.ones(freqs.size, dtype=bool)
    if chunk.band_indices is not None:
        # The indices are bins of another FFT size: a bin here is inside when its nearest one there is.
        source_n = int(chunk.band_index_fft_size or block)
        inside &= np.isin(np.rint(freqs * source_n / sr).astype(np.int64), np.asarray(chunk.band_indices, dtype=np.int64))
    if chunk.band_hz_bounds is not None:
        low, high = float(chunk.band_hz_bounds[0]), float(chunk.band_hz_bounds[1])
        if high > low:
            inside &= (freqs >= low) & (freqs < high)

    power = np.square(np.abs(np.fft.rfft(frame)))
    power[1:(n + 1) // 2] *= 2.0  # the negative frequencies
    # The fades hold one block of energy between them, so the block length normalizes.
    return float(math.sqrt(float(np.sum(power[inside])) / (n * block)))


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
