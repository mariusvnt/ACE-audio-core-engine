"""
DSP task registry for the AudioCore Engine Blender add-on.

Defines every audio processing node (tasks) used by the engine pipeline:

  - **Infrastructure**: ``ACEAudioTask``, ``ACEAudioGraph``, ``AudioChunk``,
    ``TaskConnection``, and the ``AudioFluxAnalyzer`` pool-managed BFT wrapper.
  - **Source nodes**: ``SourceTask``, ``SourceSpecsTask``.
  - **Splitter nodes**: ``FrequencySplitterTask``, ``MidsideSplitterTask``,
    ``AttackReleaseSplitterTask``, ``ChromagramTask``,
    ``AutoComponentSplitterTask``.
  - **Stream operators**: ``RMSTask``, ``LUFSTask``, ``PeakTask``, ``PitchTask``,
    ``SpectralCentroidTask``, ``SpectralFluxTask``, ``OnsetTask``,
    ``ZeroCrossTask``, ``StereoWidthTask``, ``StereoBalanceTask``,
    ``StereoVectorTask``.
  - **Layer-3 preview**: ``PreviewSinkTask`` — time-domain-only biquad filter
    chain; strictly no iFFT/iBFT.
  - **Utility**: ``RollingAvgTask``, ``StreamAccumulatorTask``, ``AudioMathTask``,
    ``SourceFrontendProcessor``.

All bpy access is forbidden in this module: tasks are evaluated on Stage-B
worker processes (via ``parallel_dsp``) that are intentionally bpy-free.
"""
import math
import os
import hashlib
import heapq
import threading
import time
from dataclasses import dataclass
from collections import deque, OrderedDict
from queue import Queue, Empty, Full
from typing import Dict, List, Any, Optional, Tuple, NamedTuple, Callable

import numpy as np

try:
    import audioflux  # type: ignore
    from audioflux import type as af_type  # type: ignore
    AUDIOFLUX_AVAILABLE = True
except ImportError:
    AUDIOFLUX_AVAILABLE = False

try:
    from numba import njit  # type: ignore
    NUMBA_AVAILABLE = True
except Exception:  # noqa: BLE001 boundary: a broken native library raises more than ImportError
    NUMBA_AVAILABLE = False

try:
    from scipy.signal import lfilter as scipy_lfilter  # type: ignore
    SCIPY_AVAILABLE = True
except ImportError:
    scipy_lfilter = None
    SCIPY_AVAILABLE = False

# One-time warning flag: emitted the first time the Tier-3 Python biquad loop
# is entered (Numba absent AND scipy absent).  Suppresses log spam while still
# surfacing the dropout risk once per engine session.
# Reset by reset_biquad_fallback_warned(), called from ACEAudioEngine.shutdown_all_instances().
_BIQUAD_PYTHON_FALLBACK_WARNED: bool = False


def reset_biquad_fallback_warned() -> None:
    """Reset the one-time Tier-3 biquad fallback warning flag.

    Called by ``ACEAudioEngine.shutdown_all_instances()`` so the warning fires
    once per engine session rather than once per Python interpreter lifetime.
    This ensures users are re-warned on a fresh session if Numba and scipy are
    still absent.
    """
    global _BIQUAD_PYTHON_FALLBACK_WARNED
    _BIQUAD_PYTHON_FALLBACK_WARNED = False

from .audio_influx import AudioReception, build_live_flow_identifier
from .. import utils

SAMPLE_RATE = 44100
MIN_BLOCK_SIZE = 512
MAX_BLOCK_SIZE = 4096

# Thread-local flag set by PreviewSinkTask.process() to mark the Layer-3 preview
# call path. _masked_mono_signal() checks this and logs an ERROR on violation since
# calling iFFT reconstruction from Layer 3 would contaminate the time-domain output.
_LAYER3_THREAD_LOCAL: threading.local = threading.local()

# Per-filepath open locks: prevents simultaneous AudioReception.open() calls on the
# same file from different threads (SourceSpecsTask BPM thread + SourceTask prefetch),
# which on Windows can hit exclusive-lock failures.
_FILE_OPEN_LOCKS: Dict[str, threading.Lock] = {}
_FILE_OPEN_LOCKS_MUTEX = threading.Lock()
_FILE_OPEN_LOCKS_MAX = 128  # Max dict entries before evicting idle (unlocked) entries.


def _get_file_open_lock(filepath: str) -> threading.Lock:
    """Return (creating if needed) a per-filepath Lock to serialize open() calls.

    When the dict exceeds _FILE_OPEN_LOCKS_MAX entries, idle (unlocked) locks are
    evicted to prevent unbounded growth in long sessions with many different files.
    """
    with _FILE_OPEN_LOCKS_MUTEX:
        lock = _FILE_OPEN_LOCKS.get(filepath)
        if lock is None:
            lock = threading.Lock()
            _FILE_OPEN_LOCKS[filepath] = lock
        if len(_FILE_OPEN_LOCKS) > _FILE_OPEN_LOCKS_MAX:
            # Evict only locks not currently held to avoid disrupting active I/O.
            stale = [k for k, v in _FILE_OPEN_LOCKS.items() if k != filepath and not v.locked()]
            for k in stale[: max(len(stale) // 2, 1)]:
                _FILE_OPEN_LOCKS.pop(k, None)
        return lock


def _clear_file_open_locks():
    """Remove all idle filepath locks. Called on engine shutdown to reclaim memory."""
    with _FILE_OPEN_LOCKS_MUTEX:
        idle = [k for k, v in _FILE_OPEN_LOCKS.items() if not v.locked()]
        for k in idle:
            _FILE_OPEN_LOCKS.pop(k, None)


def _finite_copy(arr, dtype="float32"):
    """Copy of `arr` with NaN and infinities set to 0 (np.nan_to_num's result, ~4x cheaper).

    Always a new array, so callers may modify it in place. dtype=None keeps
    the input dtype.
    """
    out = np.array(arr, dtype=dtype, copy=True)
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = 0.0
    return out


def _make_silence_chunk(block_size: int, sample_rate: int = SAMPLE_RATE) -> "AudioChunk":
    """Return a zero-filled AudioChunk used as a safe placeholder for broken sources.

    Downstream DSP tasks always receive a well-formed chunk (not ``None``) so
    Stage-B workers never encounter a missing input.  All analysis on the
    returned chunk yields zero/silence values.

    Args:
        block_size: Number of samples in the silence block.
        sample_rate: Sample rate to embed in the chunk (default: ``SAMPLE_RATE``).

    Returns:
        An ``AudioChunk`` whose ``mono`` array is all zeros.
    """
    silence = np.zeros(max(int(block_size), 1), dtype=np.float32)
    return AudioChunk(silence, None, int(max(sample_rate, 1)), latency_ms=0.0)


_ADAPTIVE_FILE_BLOCK_BUCKETS = (
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
)
_ADAPTIVE_LIVE_BLOCK_BUCKETS = (
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
)
_PREVIEW_BLOCK_BUCKETS = (
    128,
    256,
    512,
)


if NUMBA_AVAILABLE:
    @njit(cache=True)  # pragma: no cover - compiled at runtime
    def _signed_smooth_block(data: np.ndarray, attack_c: float, release_c: float, smooth_state: float):
        out = np.empty_like(data)
        smoothed = smooth_state
        for i in range(data.shape[0]):
            x = float(data[i])
            coeff = attack_c if abs(x) > abs(smoothed) else release_c
            smoothed = coeff * smoothed + (1.0 - coeff) * x
            out[i] = smoothed
        return out, smoothed

    @njit(cache=True)  # pragma: no cover - compiled at runtime
    def _biquad_filter_block(
        data: np.ndarray,
        b0: float,
        b1: float,
        b2: float,
        a1: float,
        a2: float,
        z1: float,
        z2: float,
    ):
        out = np.empty_like(data)
        s1 = z1
        s2 = z2
        for i in range(data.shape[0]):
            x = float(data[i])
            y = (b0 * x) + s1
            s1 = (b1 * x) - (a1 * y) + s2
            s2 = (b2 * x) - (a2 * y)
            out[i] = y
        return out, s1, s2

    @njit(cache=True)  # pragma: no cover - compiled at runtime
    def _iir1_filter_block(
        data: np.ndarray,
        b0: float,
        b1: float,
        a1: float,
        x_prev: float,
        y_prev: float,
    ):
        out = np.empty_like(data)
        x1 = x_prev
        y1 = y_prev
        for i in range(data.shape[0]):
            x = float(data[i])
            y = (b0 * x) + (b1 * x1) - (a1 * y1)
            out[i] = y
            x1 = x
            y1 = y
        return out, x1, y1


_KEEP = object()


class SourceFrontendProcessor:
    """Stateful source-side preprocessing for analysis-domain audio.

    Shared by the main-thread source reader and the worker-side source bridge.
    Applies, in order:
      1. Optional 1st-order Butterworth high-pass filter (``freq_min``).
      2. Optional 1st-order Butterworth low-pass filter (``freq_max``).
      3. Optional signed attack/release envelope smoother.

    Filter coefficients are recomputed only when the configuration signature
    (sample_rate, frontend_config, channel_count) changes.
    """

    def __init__(self):
        self._iir_states: Dict[tuple[str, str], Dict[str, float]] = {}
        self._env_states: Dict[str, float] = {}
        self._config_sig: Optional[Tuple[int, Optional[Tuple[float, float, float, float]], int]] = None
        self._coeff_cache = {
            "hpf": None,
            "lpf": None,
            "attack_c": 0.0,
            "release_c": 0.0,
        }

    def reset(self):
        """Clear all filter states and force coefficient recomputation on the next call."""
        self._iir_states.clear()
        self._env_states.clear()
        self._config_sig = None
        self._coeff_cache = {
            "hpf": None,
            "lpf": None,
            "attack_c": 0.0,
            "release_c": 0.0,
        }

    @staticmethod
    def _butterworth_coeffs(cutoff_hz: float, sample_rate: float, mode: str):
        """Compute 1st-order Butterworth IIR coefficients (b0, b1, a1).

        Returns None when the cutoff would be outside the valid frequency range
        (DC or Nyquist boundary), in which case the caller skips filtering.

        Args:
            cutoff_hz: Filter corner frequency in Hz.
            sample_rate: Signal sample rate in Hz.
            mode: ``"low"`` for low-pass, any other string for high-pass.
        """
        if cutoff_hz <= 0.0 or sample_rate <= 0.0 or cutoff_hz >= (sample_rate * 0.49):
            return None
        omega = math.tan(math.pi * cutoff_hz / sample_rate)
        norm = 1.0 / (1.0 + omega)
        if mode == "low":
            b0 = omega * norm
            b1 = omega * norm
            a1 = (omega - 1.0) * norm
        else:
            b0 = 1.0 * norm
            b1 = -1.0 * norm
            a1 = (omega - 1.0) * norm
        return float(b0), float(b1), float(a1)

    def _ensure_config(
        self,
        sample_rate: int,
        frontend_config: Optional[Tuple[float, float, float, float]],
        channel_count: int,
    ):
        """Recompute filter coefficients when the configuration has changed.

        Args:
            sample_rate: Signal sample rate in Hz.
            frontend_config: (freq_min, freq_max, attack_ms, release_ms) or None.
            channel_count: Number of audio channels (used in the config signature).
        """
        sig = (
            int(max(sample_rate, 1)),
            tuple(frontend_config) if frontend_config is not None else None,
            int(max(channel_count, 1)),
        )
        if sig == self._config_sig:
            return

        self._config_sig = sig
        self._iir_states.clear()
        self._env_states.clear()

        freq_min = 0.0
        freq_max = 0.0
        attack_ms = 0.0
        release_ms = 0.0
        if frontend_config is not None and len(frontend_config) >= 4:
            try:
                freq_min = float(frontend_config[0] or 0.0)
                freq_max = float(frontend_config[1] or 0.0)
                attack_ms = float(frontend_config[2] or 0.0)
                release_ms = float(frontend_config[3] or 0.0)
            except Exception as exc:
                utils.note_swallowed("audio_tasks_registry.SourceFrontendProcessor._ensure_config", exc)
                freq_min = 0.0
                freq_max = 0.0
                attack_ms = 0.0
                release_ms = 0.0

        sr = float(max(sample_rate, 1))
        dt = 1.0 / sr
        self._coeff_cache["hpf"] = self._butterworth_coeffs(freq_min, sr, "high") if freq_min > 0.0 else None
        self._coeff_cache["lpf"] = self._butterworth_coeffs(freq_max, sr, "low") if (freq_max > 0.0 and freq_max < (sr * 0.5)) else None
        attack_tc = max(attack_ms, 0.0) / 1000.0
        release_tc = max(release_ms, 0.0) / 1000.0
        self._coeff_cache["attack_c"] = math.exp(-dt / max(attack_tc, 1e-6)) if attack_tc > 0.0 else 0.0
        self._coeff_cache["release_c"] = math.exp(-dt / max(release_tc, 1e-6)) if release_tc > 0.0 else 0.0

    def _run_iir(self, data: np.ndarray, coeffs: Optional[Tuple[float, float, float]], state_key: tuple[str, str]) -> np.ndarray:
        src = np.asarray(data, dtype=np.float32).reshape(-1).astype(np.float32, copy=False)
        if coeffs is None or src.size == 0:
            return src

        state = self._iir_states.setdefault(state_key, {"x1": 0.0, "y1": 0.0})
        b0, b1, a1 = coeffs
        x_prev = float(state.get("x1", 0.0))
        y_prev = float(state.get("y1", 0.0))

        if NUMBA_AVAILABLE:
            with utils.logged_errors("audio_tasks_registry.SourceFrontendProcessor._run_iir"):
                out, x_last, y_last = _iir1_filter_block(
                    src,
                    float(b0),
                    float(b1),
                    float(a1),
                    float(x_prev),
                    float(y_prev),
                )
                state["x1"] = float(x_last)
                state["y1"] = float(y_last)
                return out.astype(np.float32, copy=False)

        out = np.empty_like(src, dtype=np.float32)
        x1 = float(x_prev)
        y1 = float(y_prev)
        for idx in range(src.shape[0]):
            x = float(src[idx])
            y = (float(b0) * x) + (float(b1) * x1) - (float(a1) * y1)
            out[idx] = y
            x1 = x
            y1 = y
        state["x1"] = float(x1)
        state["y1"] = float(y1)
        return out

    def _run_envelope(self, data: np.ndarray, attack_c: float, release_c: float, state_key: str) -> np.ndarray:
        src = np.asarray(data, dtype=np.float32).reshape(-1).astype(np.float32, copy=False)
        if src.size == 0:
            return src
        if attack_c <= 0.0 and release_c <= 0.0:
            return src

        env_state = float(self._env_states.get(state_key, 0.0) or 0.0)
        if NUMBA_AVAILABLE:
            with utils.logged_errors("audio_tasks_registry.SourceFrontendProcessor._run_envelope"):
                env_out, env_last = _signed_smooth_block(src, float(attack_c), float(release_c), float(env_state))
                self._env_states[state_key] = float(env_last)
                return env_out.astype(np.float32, copy=False)

        out = np.empty_like(src, dtype=np.float32)
        env = float(env_state)
        for idx in range(src.shape[0]):
            x = float(src[idx])
            coeff = float(attack_c) if abs(x) > abs(env) else float(release_c)
            env = (coeff * env) + ((1.0 - coeff) * x)
            out[idx] = env
        self._env_states[state_key] = float(env)
        return out

    def _process_channel(self, signal: np.ndarray, channel_key: str) -> np.ndarray:
        data = np.asarray(signal, dtype=np.float32).reshape(-1).astype(np.float32, copy=False)
        data = self._run_iir(data, self._coeff_cache.get("hpf"), (channel_key, "hpf"))
        data = self._run_iir(data, self._coeff_cache.get("lpf"), (channel_key, "lpf"))
        data = self._run_envelope(
            data,
            float(self._coeff_cache.get("attack_c", 0.0) or 0.0),
            float(self._coeff_cache.get("release_c", 0.0) or 0.0),
            channel_key,
        )
        return _finite_copy(data)

    def process(
        self,
        mono: np.ndarray,
        stereo: Optional[np.ndarray],
        sample_rate: int,
        frontend_config: Optional[Tuple[float, float, float, float]],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply the full frontend processing chain and return (analysis_mono, analysis_stereo).

        When ``frontend_config`` is None the input is returned unchanged.  When
        stereo is present each channel is processed independently and
        ``analysis_mono`` is computed as the channel mean.

        Args:
            mono: Raw mono signal (1-D float32).
            stereo: Optional stereo signal (2-D float32, shape (channels, samples)).
            sample_rate: Signal sample rate in Hz.
            frontend_config: (freq_min_hz, freq_max_hz, attack_ms, release_ms) tuple,
                             or None to bypass all processing.

        Returns:
            Tuple of (analysis_mono, analysis_stereo) arrays.
        """
        mono_arr = np.asarray(mono, dtype=np.float32).reshape(-1).astype(np.float32, copy=False)
        stereo_arr: Optional[np.ndarray] = None
        if isinstance(stereo, np.ndarray):
            arr = np.asarray(stereo, dtype=np.float32)
            if arr.ndim == 2:
                if arr.shape[1] == mono_arr.shape[0]:
                    stereo_arr = arr.astype(np.float32, copy=False)
                elif arr.shape[0] == mono_arr.shape[0]:
                    stereo_arr = arr.T.astype(np.float32, copy=False)

        if frontend_config is None:
            return mono_arr, stereo_arr

        channel_count = int(stereo_arr.shape[0]) if isinstance(stereo_arr, np.ndarray) and stereo_arr.ndim == 2 else 1
        self._ensure_config(sample_rate=sample_rate, frontend_config=frontend_config, channel_count=channel_count)

        if stereo_arr is not None and stereo_arr.shape[1] == mono_arr.shape[0]:
            processed_channels = []
            for ch in range(int(stereo_arr.shape[0])):
                processed_channels.append(self._process_channel(stereo_arr[ch], f"ch{int(ch)}"))
            analysis_stereo = np.stack(processed_channels, axis=0).astype(np.float32, copy=False)
            analysis_mono = np.mean(analysis_stereo, axis=0).astype(np.float32, copy=False)
            return analysis_mono, analysis_stereo

        analysis_mono = self._process_channel(mono_arr, "mono")
        return analysis_mono, None


class AudioChunk:
    """Lightweight wrapper around one audio block passed between DSP task nodes.

    Carries the raw mono/stereo signal, an optional preprocessed analysis track,
    spectral metadata (band-mask indices, Hz bounds, lineage token), and a lazy
    feature cache so derived quantities (RMS, centroid, …) are computed at most
    once per frame per chunk.

    The ``provider`` back-reference to the owning ``SourceTask`` allows downstream
    analysis nodes to delegate to the source's long-lived ``AudioFluxAnalyzer``
    rather than allocating a new BFT object on every call.

    Slot layout is fixed (``__slots__``) to minimize per-instance memory overhead
    in multi-source sessions where hundreds of chunks may be alive simultaneously.
    """
    __slots__ = (
        "mono",
        "stereo",
        "analysis_mono",
        "analysis_stereo",
        "frontend_config",
        "sample_rate",
        "latency_ms",
        "provider",
        "frame_index",
        "features",
        "band_indices",
        "band_index_fft_size",
        "band_hz_bounds",
        "lineage",
        "spectrum_token",
        "analysis_ops",
    )

    def __init__(self, mono: np.ndarray, stereo: Optional[np.ndarray], sample_rate: int,
                 provider: Optional['SourceTask'] = None, frame_index: int = -1,
                 analysis_mono: Optional[np.ndarray] = None,
                 analysis_stereo: Optional[np.ndarray] = None,
                 frontend_config: Optional[Tuple[float, float, float, float]] = None,
                 band_indices: Optional[np.ndarray] = None,
                 band_index_fft_size: Optional[int] = None,
                 band_hz_bounds: Optional[Tuple[float, float]] = None,
                 lineage: Optional[Tuple[str, ...]] = None,
                 spectrum_token: Optional[Any] = None,
                 analysis_ops: Optional[Tuple[Tuple[str, str], ...]] = None,
                 latency_ms: float = 0.0):
        self.mono = mono.astype(np.float32, copy=False)
        self.stereo = stereo.astype(np.float32, copy=False) if stereo is not None else None
        self.analysis_mono = analysis_mono.astype(np.float32, copy=False) if isinstance(analysis_mono, np.ndarray) else None
        self.analysis_stereo = analysis_stereo.astype(np.float32, copy=False) if isinstance(analysis_stereo, np.ndarray) else None
        self.frontend_config = tuple(frontend_config) if isinstance(frontend_config, (tuple, list)) and len(frontend_config) >= 4 else None
        self.sample_rate = sample_rate
        self.latency_ms = float(max(float(latency_ms), 0.0))
        self.provider = provider
        self.frame_index = frame_index
        self.features: Dict[str, Any] = {}
        self.band_indices = band_indices
        if band_index_fft_size is not None:
            try:
                self.band_index_fft_size = max(int(band_index_fft_size), 1)
            except Exception as exc:
                utils.note_swallowed("audio_tasks_registry.AudioChunk.__init__", exc)
                self.band_index_fft_size = None
        elif band_indices is not None:
            # Default behavior: assume indices target the chunk FFT domain.
            self.band_index_fft_size = max(int(self.mono.shape[-1] or 1), 1)
        else:
            self.band_index_fft_size = None
        self.band_hz_bounds = band_hz_bounds
        self.lineage = lineage or ()
        self.analysis_ops = tuple(analysis_ops or ())
        if spectrum_token is not None:
            self.spectrum_token = spectrum_token
        elif lineage:
            # Stable token for continuity across frames on real signal branches.
            self.spectrum_token = tuple(lineage)
        else:
            # Fallback token for ad-hoc chunks.
            self.spectrum_token = id(self.mono)

    def __array__(self, dtype=None):
        if dtype is not None:
            return np.asarray(self.mono, dtype=dtype)
        return self.mono

    def get_feature(self, key: str, compute_fn=None):
        """Return a cached feature, optionally computing it on the first call.

        Args:
            key: Unique string key identifying the feature.
            compute_fn: Zero-argument callable that computes the feature value.
                        Called once and cached when the key is not already present.

        Returns:
            The cached (or freshly computed) feature value, or None if absent
            and no ``compute_fn`` was supplied.
        """
        if key in self.features:
            return self.features[key]
        if compute_fn:
            self.features[key] = compute_fn()
            return self.features[key]
        return None


def _spawn_audio_chunk(
    source: AudioChunk,
    *,
    mono: Any = _KEEP,
    stereo: Any = _KEEP,
    analysis_mono: Any = _KEEP,
    analysis_stereo: Any = _KEEP,
    frontend_config: Any = _KEEP,
    sample_rate: Any = _KEEP,
    provider: Any = _KEEP,
    frame_index: Any = _KEEP,
    band_indices: Any = _KEEP,
    band_index_fft_size: Any = _KEEP,
    band_hz_bounds: Any = _KEEP,
    lineage: Any = _KEEP,
    spectrum_token: Any = _KEEP,
    analysis_ops: Any = _KEEP,
    latency_ms: Any = _KEEP,
) -> AudioChunk:
    """Create a new AudioChunk by overriding selected fields from *source*.

    Unspecified keyword arguments default to the sentinel ``_KEEP``, which
    causes the corresponding field to be copied verbatim from *source*.  This
    avoids boilerplate in splitter tasks that only mutate the lineage or
    band-mask fields while keeping all other metadata unchanged.

    Returns:
        A new AudioChunk with the merged field values.
    """
    return AudioChunk(
        mono=(source.mono if mono is _KEEP else mono),
        stereo=(source.stereo if stereo is _KEEP else stereo),
        sample_rate=(source.sample_rate if sample_rate is _KEEP else sample_rate),
        provider=(source.provider if provider is _KEEP else provider),
        frame_index=(source.frame_index if frame_index is _KEEP else frame_index),
        analysis_mono=(source.analysis_mono if analysis_mono is _KEEP else analysis_mono),
        analysis_stereo=(source.analysis_stereo if analysis_stereo is _KEEP else analysis_stereo),
        frontend_config=(source.frontend_config if frontend_config is _KEEP else frontend_config),
        band_indices=(source.band_indices if band_indices is _KEEP else band_indices),
        band_index_fft_size=(source.band_index_fft_size if band_index_fft_size is _KEEP else band_index_fft_size),
        band_hz_bounds=(source.band_hz_bounds if band_hz_bounds is _KEEP else band_hz_bounds),
        lineage=(source.lineage if lineage is _KEEP else lineage),
        spectrum_token=(source.spectrum_token if spectrum_token is _KEEP else spectrum_token),
        analysis_ops=(source.analysis_ops if analysis_ops is _KEEP else analysis_ops),
        latency_ms=(source.latency_ms if latency_ms is _KEEP else latency_ms),
    )


def _normalized_analysis_ops(chunk: AudioChunk) -> Tuple[Tuple[str, str], ...]:
    """Return the chunk's ``analysis_ops`` as a normalised immutable tuple.

    Each element is a ``(kind, mode)`` pair of lower-cased strings.  Malformed
    entries (wrong type, missing fields, empty strings) are silently dropped.
    """
    raw_ops = getattr(chunk, "analysis_ops", ()) or ()
    if isinstance(raw_ops, tuple):
        items = raw_ops
    elif isinstance(raw_ops, list):
        items = tuple(raw_ops)
    else:
        return ()

    normalized: list[Tuple[str, str]] = []
    for op in items:
        if not isinstance(op, (tuple, list)) or len(op) < 2:
            continue
        kind = str(op[0] or "").strip().lower()
        mode = str(op[1] or "").strip().lower()
        if not kind or not mode:
            continue
        normalized.append((kind, mode))
    return tuple(normalized)


def _append_analysis_op(chunk: AudioChunk, kind: str, mode: str) -> Tuple[Tuple[str, str], ...]:
    """Return the chunk's existing analysis ops with a new (kind, mode) entry appended."""
    ops = _normalized_analysis_ops(chunk)
    return ops + ((str(kind).strip().lower(), str(mode).strip().lower()),)


def _signal_for_chunk(
    chunk: AudioChunk,
    *,
    prefer_analysis_track: bool = True,
    apply_analysis_ops: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Resolve the effective (mono, stereo) signal arrays for a chunk.

    Applies any recorded ``analysis_ops`` (mid/side matrix, attack/release gate)
    in sequence.  Results are memoised in ``chunk.features`` keyed by
    ``("analysis_signal", prefer_analysis_track, ops_tuple)`` to avoid redundant
    computation when the same chunk is consumed by multiple downstream tasks.

    Args:
        chunk: Source AudioChunk.
        prefer_analysis_track: When True, uses the preprocessed ``analysis_mono``
            / ``analysis_stereo`` arrays if present; falls back to raw ``mono``/``stereo``.
        apply_analysis_ops: When False, ops are skipped and the base signal is
            returned directly (used by PreviewSinkTask to separate band-limiting
            from temporal ops).

    Returns:
        Tuple of (mono_1d, stereo_2d_or_None).
    """
    ops = _normalized_analysis_ops(chunk)
    if not apply_analysis_ops:
        ops = ()

    mono_base = (
        getattr(chunk, "analysis_mono", None)
        if prefer_analysis_track and isinstance(getattr(chunk, "analysis_mono", None), np.ndarray)
        else getattr(chunk, "mono", None)
    )
    stereo_base = (
        getattr(chunk, "analysis_stereo", None)
        if prefer_analysis_track and isinstance(getattr(chunk, "analysis_stereo", None), np.ndarray)
        else getattr(chunk, "stereo", None)
    )
    mono_arr = np.asarray(mono_base if isinstance(mono_base, np.ndarray) else np.zeros(0, dtype=np.float32), dtype=np.float32).reshape(-1)
    if not ops:
        return mono_arr, stereo_base

    cache_key = ("analysis_signal", bool(prefer_analysis_track), ops)
    cached = chunk.features.get(cache_key)
    if isinstance(cached, tuple) and len(cached) == 2:
        mono_cached, stereo_cached = cached
        return mono_cached, stereo_cached

    mono = mono_arr.astype(np.float32, copy=True)
    stereo_src = stereo_base
    stereo: Optional[np.ndarray] = None
    if isinstance(stereo_src, np.ndarray) and stereo_src.ndim == 2 and stereo_src.shape[1] == mono.shape[0]:
        stereo = np.asarray(stereo_src, dtype=np.float32).astype(np.float32, copy=True)

    for kind, mode in ops:
        if kind == "midside":
            if stereo is not None and stereo.shape[0] >= 2:
                left = stereo[0]
                right = stereo[1]
            else:
                # Mono sources behave as perfectly centered stereo for Mid/Side.
                left = mono
                right = mono

            if mode == "side":
                mono = (0.5 * (left - right)).astype(np.float32, copy=False)
            else:
                mono = (0.5 * (left + right)).astype(np.float32, copy=False)
            # Mid/Side branches collapse to mono descriptors for analysis.
            stereo = None
            continue

        if kind == "attackrelease":
            if mono.size == 0:
                continue
            env = np.abs(mono)
            diff = np.diff(env, prepend=env[0])
            if mode == "release":
                gate = (diff < 0.0).astype(np.float32, copy=False)
            else:
                gate = (diff >= 0.0).astype(np.float32, copy=False)
            mono = (mono * gate).astype(np.float32, copy=False)
            if stereo is not None and stereo.shape[1] == gate.shape[0]:
                stereo = (stereo * gate[np.newaxis, :]).astype(np.float32, copy=False)
            continue

    chunk.features[cache_key] = (mono, stereo)
    return mono, stereo


def _analysis_signal_for_chunk(chunk: AudioChunk) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return analysis-domain (mono, stereo) with ops applied.  Prefers preprocessed track."""
    return _signal_for_chunk(chunk, prefer_analysis_track=True, apply_analysis_ops=True)


def _preview_signal_for_chunk(chunk: AudioChunk) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return raw-domain (mono, stereo) with ops applied.  Prefers unprocessed track."""
    return _signal_for_chunk(chunk, prefer_analysis_track=False, apply_analysis_ops=True)


class AudioFluxAnalyzer:
    """Pool-managed wrapper around long-lived audioflux BFT / Spectral / Pitch objects.

    Creating a BFT object is expensive (~few ms) so instances are never thrown
    away: ``borrow()`` retrieves a ready-to-use analyzer from the pool (keyed by
    sample_rate × block_size) and ``release()`` returns it after each DSP frame.

    A class-level ``_GLOBAL_SPECTRUM_CACHE`` is shared across all instances so
    that multi-band sources (FrequencySplitter with N outputs) pay exactly one
    BFT call per audio frame instead of N.  The cache is keyed by
    ``(chunk_token, "full_spectrum", frame_index)`` to prevent spectra from
    bleeding across frames.

    When ``audioflux`` is not installed the analyzer degrades gracefully:
    ``compute_feature()`` returns None for spectrum-dependent features (centroid,
    flux, pitch) and falls back to a pure-numpy ZCR and RMS path.
    """
    _POOL: Dict[Tuple[int, int], list['AudioFluxAnalyzer']] = {}
    _MAX_POOL_PER_KEY = 4
    _MAX_POOL_KEYS = 16
    _MAX_PREV_MASK_ENTRIES = 128
    # Cross-instance BFT cache: keyed by (chunk_token, "full_spectrum", frame_index).
    # Shared across all AudioFluxAnalyzer instances so multi-band FrequencySplitter
    # sources pay exactly 1 BFT call per chunk instead of N (one per band).
    # THREAD-SAFETY NOTE: Each Stage-B worker process is single-threaded (no
    # internal BLAS thread-pool enabled), so this class-level dict is safe without
    # a Lock under the current architecture. If worker-internal threading is ever
    # introduced, protect all mutations with a threading.Lock here.
    _GLOBAL_SPECTRUM_CACHE: "OrderedDict[Any, np.ndarray]" = OrderedDict()
    _GLOBAL_SPECTRUM_CACHE_MAX_DEFAULT = 64

    def __init__(self, sample_rate: int, block_size: int):
        self.sample_rate = sample_rate
        self.block_size = block_size

        self.radix2_exp = max(6, int(math.ceil(math.log2(max(block_size, 1)))))
        self.fft_length = 1 << self.radix2_exp
        self.num_bins = self.fft_length // 2 + 1
        self.slide_length = block_size

        self.bft = None
        self.spectral = None
        self.pitch = None
        self.pitch_buffer = None
        self.freq_bins = np.fft.rfftfreq(self.fft_length, d=1.0 / self.sample_rate)
        self.prev_masked_mag: Dict[Tuple[Any, Optional[Tuple[int, ...]]], np.ndarray] = {}
        self._last_error_log: Dict[str, float] = {}

        self.cache_frame = -1
        self.cache: Dict[str, Any] = {}

        if AUDIOFLUX_AVAILABLE:
            try:
                self.bft = audioflux.BFT(
                    num=self.num_bins,
                    radix2_exp=self.radix2_exp,
                    samplate=self.sample_rate,
                    low_fre=0.0,
                    high_fre=self.sample_rate / 2,
                    window_type=af_type.WindowType.HANN,
                    slide_length=self.slide_length,
                    scale_type=af_type.SpectralFilterBankScaleType.LINEAR,
                    style_type=af_type.SpectralFilterBankStyleType.SLANEY,
                    normal_type=af_type.SpectralFilterBankNormalType.NONE,
                    data_type=af_type.SpectralDataType.MAG,
                    is_reassign=False,
                    is_temporal=True,
                )
                self.spectral = audioflux.Spectral(
                    num=self.num_bins,
                    fre_band_arr=self.bft.get_fre_band_arr()
                )

                pitch_radix = max(self.radix2_exp, 12)
                pitch_len = max(self.fft_length * 2, 2048)
                self.pitch = audioflux.PitchYIN(
                    samplate=self.sample_rate,
                    low_fre=20.0,
                    high_fre=min(5000.0, self.sample_rate / 2 - 10),
                    radix2_exp=pitch_radix,
                    slide_length=max(self.block_size, 256),
                    auto_length=pitch_len
                )
                self.pitch_buffer = np.zeros(self.pitch.auto_length, dtype=np.float32)
            except Exception as exc:
                utils.note_swallowed("audio_tasks_registry.AudioFluxAnalyzer.__init__", exc)
                self.bft = None
                self.spectral = None
                self.pitch = None
                self.pitch_buffer = None

    @classmethod
    def borrow(cls, sample_rate: int, block_size: int) -> 'AudioFluxAnalyzer':
        """Retrieve an analyzer from the pool or create a new one.

        The returned instance has its per-frame state cleared via ``reset_state()``
        so it is safe to use immediately.
        """
        key = (int(sample_rate), int(block_size))
        pool = cls._POOL.get(key)
        if pool:
            analyzer = pool.pop()
            analyzer.reset_state()
            return analyzer
        return cls(sample_rate, block_size)

    @classmethod
    def release(cls, analyzer: 'AudioFluxAnalyzer'):
        """Return an analyzer to the pool, evicting overflow entries for other keys."""
        key = (int(analyzer.sample_rate), int(analyzer.block_size))
        analyzer.reset_state()
        pool = cls._POOL.setdefault(key, [])
        if len(pool) < cls._MAX_POOL_PER_KEY:
            pool.append(analyzer)
        if len(cls._POOL) > cls._MAX_POOL_KEYS:
            overflow = len(cls._POOL) - cls._MAX_POOL_KEYS
            for stale_key in list(cls._POOL.keys())[:overflow]:
                if stale_key == key:
                    continue
                cls._POOL.pop(stale_key, None)

    @classmethod
    def _global_spectrum_cache_limit(cls) -> int:
        """Return the global cross-instance spectrum cache capacity."""
        try:
            raw = int(
                os.environ.get(
                    "ACE_GLOBAL_SPECTRUM_CACHE_MAX",
                    cls._GLOBAL_SPECTRUM_CACHE_MAX_DEFAULT,
                )
                or cls._GLOBAL_SPECTRUM_CACHE_MAX_DEFAULT
            )
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.AudioFluxAnalyzer._global_spectrum_cache_limit", exc)
            raw = int(cls._GLOBAL_SPECTRUM_CACHE_MAX_DEFAULT)
        return int(min(max(raw, 16), 1024))

    def reset_state(self):
        """Clear all per-frame caches and flux-history state."""
        self.prev_masked_mag.clear()
        self.cache_frame = -1
        self.cache.clear()

    def _prepare_frame_cache(self, frame_index: int):
        """Invalidate the feature cache when a new frame begins."""
        if self.cache_frame != frame_index:
            self.cache_frame = frame_index
            self.cache.clear()

    @staticmethod
    def _chunk_token(chunk: AudioChunk):
        """Return a hashable cache key for the chunk's audio content and ops."""
        token = getattr(chunk, "spectrum_token", None)
        if token is None:
            if getattr(chunk, "lineage", None):
                token = tuple(chunk.lineage)
            else:
                token = id(chunk.mono)

        ops_sig = _normalized_analysis_ops(chunk)
        if ops_sig:
            return (token, ("ops", ops_sig))
        return token

    def _resolve_chunk_mask_indices(self, chunk: AudioChunk) -> Optional[np.ndarray]:
        """Resolve the effective FFT-bin mask for this analyzer's FFT domain.

        Remaps any ``band_indices`` from the chunk's source FFT domain to this
        analyzer's domain, then intersects with indices derived from
        ``band_hz_bounds`` when both are present.

        Returns None when no mask constraints are set on the chunk.
        """
        indices = getattr(chunk, "band_indices", None)
        if indices is not None:
            if not isinstance(indices, np.ndarray):
                indices = np.asarray(indices, dtype=np.int64)
            if indices.size == 0:
                indices = None
            else:
                src_fft_size = getattr(chunk, "band_index_fft_size", None)
                if src_fft_size is None:
                    src_fft_size = max(int(getattr(chunk, "mono", np.zeros(1, dtype=np.float32)).shape[-1] or 1), 1)
                try:
                    src_fft_size = max(int(src_fft_size), 1)
                except Exception as exc:
                    utils.note_swallowed("audio_tasks_registry.AudioFluxAnalyzer._resolve_chunk_mask_indices", exc)
                    src_fft_size = self.fft_length
                indices = _remap_indices_fft_domain(
                    np.asarray(indices, dtype=np.int64),
                    src_fft_size=src_fft_size,
                    dst_fft_size=self.fft_length,
                    sample_rate=int(max(getattr(chunk, "sample_rate", SAMPLE_RATE), 1)),
                )
                if indices.size == 0:
                    indices = None

        bounds = getattr(chunk, "band_hz_bounds", None)
        if bounds is None:
            return indices
        try:
            low_hz, high_hz = float(bounds[0]), float(bounds[1])
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.AudioFluxAnalyzer._resolve_chunk_mask_indices#2", exc)
            return indices
        if high_hz <= low_hz:
            return indices
        mask = (self.freq_bins >= low_hz) & (self.freq_bins < high_hz)
        hz_indices = np.nonzero(mask)[0]
        if hz_indices.size == 0:
            return None
        if indices is None:
            return hz_indices
        merged = np.intersect1d(indices, hz_indices, assume_unique=False)
        return merged if merged.size > 0 else None

    def _ensure_spectrum(self, chunk: AudioChunk):
        if not AUDIOFLUX_AVAILABLE or self.bft is None or self.spectral is None:
            return None
        chunk_token = self._chunk_token(chunk)
        spec_key = ("spec_mag", chunk_token)
        if spec_key in self.cache:
            return self.cache[spec_key]

        # Check the cross-instance global cache before running BFT.
        # This reduces multi-band FrequencySplitter sources from N BFT calls to 1 per frame.
        # frame_index is included so cached spectra never bleed across frames — without it,
        # the lineage-derived chunk_token is stable across frames and the first frame's
        # spectrum would be returned indefinitely, causing constant RMS/centroid outputs.
        global_key = (chunk_token, "full_spectrum", int(chunk.frame_index))
        global_mag = AudioFluxAnalyzer._GLOBAL_SPECTRUM_CACHE.get(global_key)
        if global_mag is not None:
            AudioFluxAnalyzer._GLOBAL_SPECTRUM_CACHE.move_to_end(global_key)
            self.cache[spec_key] = global_mag
            return global_mag

        data, _ = _analysis_signal_for_chunk(chunk)
        # Guard against empty chunks (e.g. from _make_silence_chunk on broken sources)
        # to prevent null/invalid spectrum entries from entering the cache.
        if data is None or data.size == 0:
            return None
        if data.shape[-1] < self.fft_length:
            padded = np.zeros(self.fft_length, dtype=np.float32)
            padded[:data.shape[-1]] = data
            data = padded

        try:
            spec = self.bft.bft(data)
        except Exception as e:
            self._log_error("bft", f"AudioFlux BFT failed: {e}")
            return None

        mag = np.abs(spec)
        time_len = mag.shape[-1]
        with utils.logged_errors("audio_tasks_registry.AudioFluxAnalyzer._ensure_spectrum"):
            self.spectral.set_time_length(time_len)

        self.cache[spec_key] = mag
        # Populate global cross-instance cache with LRU eviction.
        AudioFluxAnalyzer._GLOBAL_SPECTRUM_CACHE[global_key] = mag
        AudioFluxAnalyzer._GLOBAL_SPECTRUM_CACHE.move_to_end(global_key)
        max_entries = AudioFluxAnalyzer._global_spectrum_cache_limit()
        while len(AudioFluxAnalyzer._GLOBAL_SPECTRUM_CACHE) > int(max_entries):
            try:
                AudioFluxAnalyzer._GLOBAL_SPECTRUM_CACHE.popitem(last=False)
            except Exception as exc:
                utils.note_swallowed("audio_tasks_registry.AudioFluxAnalyzer._ensure_spectrum#2", exc)
                break
        if self.bft.is_temporal:
            try:
                _, rms_arr, zcr_arr = self.bft.get_temporal_data()
                self.cache[("rms_arr", chunk_token)] = rms_arr
                self.cache[("zcr_arr", chunk_token)] = zcr_arr
            except Exception as e:
                self._log_error("temporal", f"AudioFlux temporal feature extraction failed: {e}")
                pass
        return mag

    def _get_mask_key(self, indices: Optional[np.ndarray]) -> Optional[Tuple[int, int]]:
        """Return a compact (count, digest) cache key for a bin-index array.

        Uses a blake2b-8 digest so per-frame feature cache keys remain small
        even for large masks with many bin entries.
        """
        if indices is None:
            return None
        if not isinstance(indices, np.ndarray):
            indices = np.asarray(indices, dtype=np.int64)
        arr = np.asarray(indices, dtype=np.int64).ravel()
        if arr.size == 0:
            return (0, 0)
        arr = np.unique(arr)
        # Keep feature-cache keys compact: avoid expanding potentially long
        # Python tuples of bin indices in hot per-frame paths.
        digest = hashlib.blake2b(arr.view(np.uint8), digest_size=8).digest()
        return (int(arr.size), int.from_bytes(digest, byteorder="little", signed=False))

    def _get_masked_mag(self, mag: np.ndarray, indices: Optional[np.ndarray]):
        if indices is None:
            return mag
        if not isinstance(indices, np.ndarray):
            indices = np.asarray(indices, dtype=np.int64)
        return mag[indices]

    def _log_error(self, key: str, message: str, min_interval: float = 2.0):
        now = time.time()
        last = self._last_error_log.get(key, 0.0)
        if (now - last) < min_interval:
            return
        self._last_error_log[key] = now
        utils.log(message, "ERROR")

    def _apply_spectral_mask(self, mask_indices: Optional[np.ndarray]):
        if self.spectral is None:
            return
        if mask_indices is None:
            if self.num_bins > 1:
                self.spectral.set_edge(0, self.num_bins - 1)
            return
        idx = np.asarray(mask_indices, dtype=np.int32).ravel()
        if idx.size == 0:
            if self.num_bins > 1:
                self.spectral.set_edge(0, self.num_bins - 1)
            return
        idx = np.unique(idx)
        idx = idx[(idx >= 0) & (idx < self.num_bins)]
        if idx.size == 0:
            if self.num_bins > 1:
                self.spectral.set_edge(0, self.num_bins - 1)
            return
        self.spectral.set_edge_arr(np.ascontiguousarray(idx, dtype=np.int32))

    def _compute_flux(self, mag: np.ndarray, chunk_token: Any, mask_indices: Optional[np.ndarray] = None):
        if mag is None or mag.size == 0 or self.spectral is None:
            return 0.0

        current_frame = mag[..., -1]
        mask_key = self._get_mask_key(mask_indices)
        store_key = (chunk_token, mask_key)
        prev_store = self.prev_masked_mag.get(store_key)

        if prev_store is None:
            self.prev_masked_mag[store_key] = current_frame
            if len(self.prev_masked_mag) > self._MAX_PREV_MASK_ENTRIES:
                self.prev_masked_mag.pop(next(iter(self.prev_masked_mag)), None)
            return 0.0

        # Compute positive spectral flux directly from magnitude arrays.
        # The audioFlux spectral.flux() API requires set_time_length() to match
        # the input's time dimension; when called after _ensure_spectrum() has
        # already set a different time_length the library can return a per-bin
        # array (not a scalar), causing float() to raise and the except to return
        # 0.0 every frame.  Computing from first principles avoids the API contract
        # entirely while producing the identical mathematical result:
        #   positive_flux = sum(max(current - previous, 0)) / sum(previous + ε)
        # This is consistent with the fallback path in SpectralFluxTask._fallback_flux.
        try:
            prev_f = prev_store.astype(np.float64)
            curr_f = current_frame.astype(np.float64)
            flux_raw = float(np.sum(np.maximum(curr_f - prev_f, 0.0), dtype=np.float64))
            denom = float(np.sum(np.abs(prev_f), dtype=np.float64)) + 1e-9
            val = float(np.clip(flux_raw / denom, 0.0, 10.0))
        except Exception as e:
            self._log_error("flux", f"flux computation failed: {e}")
            val = 0.0

        self.prev_masked_mag[store_key] = current_frame
        if len(self.prev_masked_mag) > self._MAX_PREV_MASK_ENTRIES:
            self.prev_masked_mag.pop(next(iter(self.prev_masked_mag)), None)
        return val

    def compute_feature(self, name: str, chunk: AudioChunk):
        """
        Compute (and cache) a feature for the given frame.
        """
        self._prepare_frame_cache(chunk.frame_index)
        chunk_token = self._chunk_token(chunk)
        mask_indices = self._resolve_chunk_mask_indices(chunk)
        mask_key = self._get_mask_key(mask_indices)
        cache_key = (name, chunk_token, mask_key)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if name == "zcr":
            analysis_mono, _ = _analysis_signals_normalized(chunk)
            val = _zero_crossing_rate(analysis_mono)
            self.cache[cache_key] = val
            return val

        if not AUDIOFLUX_AVAILABLE:
            return None

        if name in ("rms", "centroid", "flux"):
            mag = self._ensure_spectrum(chunk)
            if mag is None:
                return None

            if name == "rms":
                arr = self.cache.get(("rms_arr", chunk_token)) if mask_indices is None else None
                if arr is not None and arr.size > 0:
                    val = float(arr[..., -1])
                else:
                    if mask_indices is None:
                        analysis_mono, _ = _analysis_signals_normalized(chunk)
                        val = _safe_rms(analysis_mono)
                    else:
                        idx = np.asarray(mask_indices, dtype=np.int64).ravel()
                        if idx.size == 0:
                            val = 0.0
                        else:
                            idx = np.unique(idx)
                            idx = idx[(idx >= 0) & (idx < self.num_bins)]
                            if idx.size == 0:
                                val = 0.0
                            else:
                                frame_mag = np.abs(mag[..., -1]).reshape(-1)
                                masked_power = np.square(frame_mag[idx], dtype=np.float64)
                                n = max(int(self.fft_length), 1)
                                weights = np.full(masked_power.shape, 2.0, dtype=np.float64)
                                weights[idx == 0] = 1.0
                                if (n % 2) == 0:
                                    weights[idx == (n // 2)] = 1.0
                                mean_square = float(np.sum(masked_power * weights) / (n * n))
                                val = float(np.sqrt(max(mean_square, 0.0)))
                self.cache[cache_key] = val
                return val
            elif name == "centroid":
                try:
                    self.spectral.set_time_length(mag.shape[-1])
                    self._apply_spectral_mask(mask_indices)
                    centroid_arr = self.spectral.centroid(mag)
                    val = float(centroid_arr[..., -1])
                    self.cache[cache_key] = val
                    return val
                except Exception as e:
                    self._log_error("centroid", f"AudioFlux centroid failed: {e}")
                    return None
            elif name == "flux":
                val = self._compute_flux(mag, chunk_token, mask_indices)
                self.cache[cache_key] = val
                return val
            # "onset" intentionally not handled here: the adaptive envelope-based
            # detection in OnsetTask._fallback_onset() is the correct implementation.
            # Returning raw flux here would produce a [0, 10] value contradicting the
            # node description "0 = smooth, 1 = sharp transient". Returning None causes
            # OnsetTask.process() to fall through to _fallback_onset().

        if name == "pitch":
            if not self.pitch or self.pitch_buffer is None:
                return None

            data, _ = _analysis_signal_for_chunk(chunk)
            target = self.pitch_buffer.shape[-1] if self.pitch_buffer is not None else data.shape[-1]
            if target > data.shape[-1]:
                buf = np.zeros(target, dtype=np.float32)
                buf[-data.shape[-1]:] = data
            else:
                buf = data[-target:]
            try:
                fre_arr, _, _ = self.pitch.pitch(buf)
                if fre_arr.size > 0:
                    val = float(np.asarray(fre_arr).reshape(-1)[-1])
                    self.cache[cache_key] = val
                    return val
            except Exception as e:
                self._log_error("pitch", f"AudioFlux pitch failed: {e}")
                return None

        return None


def _quantize_adaptive_block_size(requested_samples: int, source_type: str = "FILE") -> int:
    req = max(int(requested_samples or 0), 1)
    src_type = str(source_type or "").strip().upper()
    buckets = _ADAPTIVE_LIVE_BLOCK_BUCKETS if src_type == "LIVE" else _ADAPTIVE_FILE_BLOCK_BUCKETS
    arr = np.asarray(buckets, dtype=np.int64)
    idx = int(np.argmin(np.abs(arr - req)))
    return int(arr[idx])


def resolve_preview_block_size(requested_samples: int) -> int:
    req = max(int(requested_samples or 0), 1)
    arr = np.asarray(_PREVIEW_BLOCK_BUCKETS, dtype=np.int64)
    idx = int(np.argmin(np.abs(arr - req)))
    return int(arr[idx])


def resolve_context_block_size(
    sample_rate: int,
    fps: float,
    source_type: str = "FILE",
    timeslice_mode: str = "fixed",
    delta_frames: float = 1.0,
) -> int:
    sr = max(int(sample_rate or SAMPLE_RATE), 1)
    fps_val = float(fps) if float(fps) > 0.0 else 24.0
    src_type = str(source_type or "").strip().upper()
    mode = str(timeslice_mode or "fixed").strip().lower()

    nominal = int(round(sr / fps_val)) if fps_val > 0.0 else sr
    nominal = max(MIN_BLOCK_SIZE, min(nominal, MAX_BLOCK_SIZE))

    if mode != "adaptive":
        if src_type == "LIVE":
            if nominal <= 768:
                return 512
            if nominal <= 1536:
                return 1024
            return 2048
        return nominal

    try:
        delta_mag = abs(float(delta_frames))
    except Exception as exc:
        utils.note_swallowed("audio_tasks_registry.resolve_context_block_size", exc)
        delta_mag = 1.0
    if not math.isfinite(delta_mag):
        delta_mag = 1.0

    requested = int(round((float(sr) * float(delta_mag)) / max(fps_val, 1e-6))) if delta_mag > 1e-9 else 0
    if requested <= 0:
        requested = nominal
    requested = max(MIN_BLOCK_SIZE, min(requested, MAX_BLOCK_SIZE))
    return _quantize_adaptive_block_size(requested, source_type=src_type)


def _resolve_block_size_default(
    sample_rate: int,
    fps: float,
    source_type: str = "FILE",
    timeslice_mode: str = "fixed",
    delta_frames: float = 1.0,
) -> int:
    return resolve_context_block_size(
        sample_rate,
        fps,
        source_type=source_type,
        timeslice_mode=timeslice_mode,
        delta_frames=delta_frames,
    )


def _block_duration_ms(sample_rate: int | float, block_size: int | float) -> float:
    sr = max(float(sample_rate), 1.0)
    bs = max(float(block_size), 0.0)
    return float((bs / sr) * 1000.0)


class TaskConnection(NamedTuple):
    task: 'ACEAudioTask'
    socket_index: int


class ACEAudioTask:
    """Base class for all DSP processing nodes in the ACE pipeline.

    Subclasses implement ``process(inputs)`` to transform one or more
    ``AudioChunk`` inputs into output values.  Each task owns its internal
    filter state, silence buffer, and block-size configuration so that
    state is preserved across frames without external management.

    Subclasses should NOT import or call bpy — tasks are evaluated on
    Stage-B worker processes that are intentionally bpy-free.
    """
    def __init__(self, node_id: str, parameters: Dict[str, Any], block_size: int):
        self.node_id = node_id
        self.parameters = parameters
        self.block_size = block_size
        self.engine_ref = None
        self.undo_generation: int = 0
        self.resolve_block_size_fn = None
        
        self.inputs: Dict[int, TaskConnection] = {} 
        self.cached_output: Any = None
        self.last_frame_processed = -1
        self._current_chunk: Optional[AudioChunk] = None
        self._current_chunk_frame: int = -1
        
        self.silence_buffer = np.zeros(block_size, dtype=np.float32)
        
        self.source_context = None
        self.scene_fps: float = 24.0
        self.scene_frame_start: int = 1
        self.engine_settings = None

    def connect_input(self, input_idx: int, target_task: 'ACEAudioTask', target_output_idx: int):
        self.inputs[input_idx] = TaskConnection(target_task, target_output_idx)

    def _evaluate_frame_iterative(self, frame_index: int):
        """
        Evaluate upstream dependencies for this task without Python recursion.
        Keeps per-task frame caching semantics unchanged.
        """
        frame_idx = int(frame_index)
        if int(getattr(self, "last_frame_processed", -1)) == frame_idx:
            return

        stack: List[Tuple['ACEAudioTask', bool]] = [(self, False)]
        queued: set[int] = set()

        while stack:
            task, ready = stack.pop()
            if task is None:
                continue
            if int(getattr(task, "last_frame_processed", -1)) == frame_idx:
                continue

            if ready:
                setattr(task, "_current_frame_request", frame_idx)
                input_data: Dict[int, Any] = {}
                for in_sock_idx, connection in (getattr(task, "inputs", {}) or {}).items():
                    upstream = getattr(connection, "task", None)
                    socket_idx = int(getattr(connection, "socket_index", 0) or 0)
                    if upstream is None:
                        input_data[in_sock_idx] = None
                        continue
                    if int(getattr(upstream, "last_frame_processed", -1)) != frame_idx:
                        try:
                            input_data[in_sock_idx] = upstream.get_output(frame_idx, socket_idx)
                        except Exception as exc:
                            utils.note_swallowed("audio_tasks_registry.ACEAudioTask._evaluate_frame_iterative", exc)
                            input_data[in_sock_idx] = None
                        continue
                    extract_fn = getattr(upstream, "_extract_result", None)
                    if callable(extract_fn):
                        with utils.logged_errors("audio_tasks_registry.ACEAudioTask._evaluate_frame_iterative#2"):
                            input_data[in_sock_idx] = extract_fn(socket_idx)
                            continue
                    try:
                        input_data[in_sock_idx] = upstream.get_output(frame_idx, socket_idx)
                    except Exception as exc:
                        utils.note_swallowed("audio_tasks_registry.ACEAudioTask._evaluate_frame_iterative#3", exc)
                        input_data[in_sock_idx] = None

                try:
                    task.cached_output = task.process(input_data)
                    task._tag_output_frame(task.cached_output, frame_idx)
                    task.last_frame_processed = frame_idx
                except Exception as exc:
                    utils.note_swallowed("audio_tasks_registry.ACEAudioTask._evaluate_frame_iterative#4", exc)
                    continue
                continue

            task_key = id(task)
            if task_key in queued:
                # Cycle safeguard: process with whatever upstream data is currently available.
                stack.append((task, True))
                continue
            queued.add(task_key)
            stack.append((task, True))
            for connection in (getattr(task, "inputs", {}) or {}).values():
                upstream = getattr(connection, "task", None)
                if upstream is None:
                    continue
                if int(getattr(upstream, "last_frame_processed", -1)) == frame_idx:
                    continue
                stack.append((upstream, False))

    def get_output(self, frame_index: int, output_socket_idx: int = 0):
        frame_idx = int(frame_index)
        if self.last_frame_processed == frame_idx:
            return self._extract_result(output_socket_idx)

        self._evaluate_frame_iterative(frame_idx)

        # Fallback: preserve legacy recursive behavior for edge cases where
        # iterative scheduling could not complete this task.
        if self.last_frame_processed != frame_idx:
            self._current_frame_request = frame_idx
            input_data = {}
            for in_sock_idx, connection in self.inputs.items():
                input_data[in_sock_idx] = connection.task.get_output(frame_idx, connection.socket_index)
            self.cached_output = self.process(input_data)
            self._tag_output_frame(self.cached_output, frame_idx)
            self.last_frame_processed = frame_idx

        return self._extract_result(output_socket_idx)

    def _extract_result(self, socket_idx):
        if self.cached_output is None:
            return self.silence_buffer

        if isinstance(self.cached_output, tuple):
            if len(self.cached_output) == 0:
                return self.silence_buffer
            
            if 0 <= socket_idx < len(self.cached_output):
                val = self.cached_output[socket_idx]
                return val if val is not None else self.silence_buffer
            
            return self.silence_buffer

        return self.cached_output

    def _tag_output_frame(self, value, frame_index: int):
        current_chunk = self._tag_output_frame_recursive(value, frame_index)
        self._current_chunk = current_chunk
        self._current_chunk_frame = int(frame_index) if current_chunk is not None else -1

    def _tag_output_frame_recursive(self, value, frame_index: int) -> Optional[AudioChunk]:
        if isinstance(value, AudioChunk):
            value.frame_index = frame_index
            return value
        if isinstance(value, (list, tuple)):
            for v in value:
                chunk = self._tag_output_frame_recursive(v, frame_index)
                if chunk is not None:
                    return chunk
        return None

    def process(self, inputs: Dict[int, Any]):
        """Override this."""
        return self.silence_buffer


class SourceTask(ACEAudioTask):
    """Audio source bridge task — supplies raw audio chunks to downstream nodes.

    Holds a reference to an ``AudioReception`` instance that owns the actual
    I/O (file prefetch or live ring buffer).  Each ``process()`` call returns
    the current audio chunk together with lightweight metadata (sample rate,
    latency estimate) for downstream tasks.
    """
    def __init__(self, node_id, params, block_size):
        super().__init__(node_id, params, block_size)
        self.read_cursor = 0
        self.reader_path: Optional[str] = None
        self.reader_channels = 1
        self.sample_rate = SAMPLE_RATE
        self.analysis: Optional[AudioFluxAnalyzer] = None
        self._frontend_processor = SourceFrontendProcessor()
        self.initialized = False
        self._env_state = 0.0
        self.reception: Optional[AudioReception] = None
        self._last_seek_sample = -1
        self._hpf_state = {"x1": 0.0, "y1": 0.0}
        self._lpf_state = {"x1": 0.0, "y1": 0.0}
        self._frontend_buffer: Optional[np.ndarray] = None
        self._env_buffer: Optional[np.ndarray] = None
        self._iir_prev_input: Optional[np.ndarray] = None
        self._iir_u_buffer: Optional[np.ndarray] = None
        self._iir_pow_buffer: Optional[np.ndarray] = None
        self._iir_accum_buffer: Optional[np.ndarray] = None
        self._frontend_index_buffer: Optional[np.ndarray] = None
        self._workspace_pool: Dict[int, Dict[str, np.ndarray]] = {}
        self._last_live_retry_ts: float = 0.0
        self.engine_settings = None
        self._live_block_buffer: deque[Tuple[np.ndarray, Optional[np.ndarray], int, int, int]] = deque(maxlen=16)
        self._last_frame_request: Optional[int] = None
        self._live_clock_anchor_frame: Optional[int] = None
        self._live_clock_anchor_sample: Optional[float] = None
        self._live_clock_last_target_sample: Optional[int] = None
        self._filter_cache = {
            "sr": None,
            "freq_min": None,
            "freq_max": None,
            "attack_ms": None,
            "release_ms": None,
            "hpf_coeffs": None,
            "lpf_coeffs": None,
            "attack_c": 0.0,
            "release_c": 0.0,
        }
        self.live_ready: bool = True
        self._last_config_signature: Optional[Tuple[Any, ...]] = None
        self.source_ready: bool = True
        self.last_error: str = ""
        # Pending bpy PropertyGroup writes deferred from non-main threads.
        self._pending_ctx_writes: dict[str, Any] = {}

        # --- Prefetch double-buffer (FILE sources only) ---
        # A daemon thread pre-reads the next sequential block while the main
        # thread processes the current one, eliminating file I/O from the hot path.
        self._prefetch_enabled: bool = False
        self._prefetch_chunk: Optional[Any] = None
        self._prefetch_for_frame: Optional[int] = None
        self._prefetch_target_frame: Optional[int] = None
        self._prefetch_ready: threading.Event = threading.Event()
        self._prefetch_trigger: threading.Event = threading.Event()
        self._prefetch_stop: threading.Event = threading.Event()
        self._prefetch_thread: Optional[threading.Thread] = None

    def _start_prefetch_thread(self):
        """Start the background pre-reader thread (FILE sources only)."""
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            return
        self._prefetch_stop.clear()
        self._prefetch_ready.clear()
        self._prefetch_enabled = True
        t = threading.Thread(
            target=self._prefetch_loop,
            daemon=True,
            name=f"ACEPrefetch-{str(self.node_id)[:20]}",
        )
        self._prefetch_thread = t
        t.start()

    def _stop_prefetch_thread(self):
        """Stop the background pre-reader thread and drain its state."""
        self._prefetch_enabled = False
        self._prefetch_stop.set()
        self._prefetch_trigger.set()   # Unblock a waiting thread
        t = self._prefetch_thread
        self._prefetch_thread = None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=1.0)
        self._prefetch_ready.clear()
        self._prefetch_chunk = None
        self._prefetch_for_frame = None

    def _prefetch_loop(self):
        """
        Worker body: waits for a trigger, reads one block via _read_next_block_sync,
        stores it, and signals ready.  Only active for FILE sources.
        """
        while not self._prefetch_stop.is_set():
            self._prefetch_trigger.wait(timeout=0.5)
            if self._prefetch_stop.is_set():
                break
            self._prefetch_trigger.clear()
            target_frame = self._prefetch_target_frame
            if target_frame is None:
                continue
            # Temporarily set the frame request so _read_next_block_sync uses
            # the correct timestamp for cursor alignment.
            old_request = getattr(self, "_current_frame_request", None)
            self._current_frame_request = target_frame
            try:
                chunk = self._read_next_block_sync(track_perf_stage=False)
            except Exception as exc:
                utils.note_swallowed("audio_tasks_registry.SourceTask._prefetch_loop", exc)
                chunk = None
            finally:
                # Only restore the previous value if the main thread hasn't
                # written a newer request in the meantime; avoids clobbering it.
                if self._current_frame_request == target_frame:
                    self._current_frame_request = old_request
            self._prefetch_chunk = chunk
            self._prefetch_for_frame = target_frame
            self._prefetch_ready.set()

    def process(self, inputs):
        self._flush_pending_ctx_writes()   # drain RNA writes from prefetch thread
        chunk = self._read_next_block()
        if chunk:
            return chunk
        if not self.source_ready:
            # Return a zero-filled chunk rather than None so Stage-B workers always
            # receive a well-formed input.  The UI reads ace_source_error for detail.
            return _make_silence_chunk(self.block_size, self.sample_rate)
        return self.silence_buffer

    def _flush_pending_ctx_writes(self) -> None:
        """Apply deferred PropertyGroup writes. Call only from the main thread."""
        if not self._pending_ctx_writes:
            return
        engine = getattr(self, "engine_ref", None)
        engine_generation = int(getattr(engine, "undo_generation", getattr(self, "undo_generation", 0)) or 0)
        task_generation = int(getattr(self, "undo_generation", engine_generation) or engine_generation)
        if task_generation != engine_generation:
            self._pending_ctx_writes.clear()
            return
        ctx = self.source_context
        if ctx is None:
            self._pending_ctx_writes.clear()
            return
        for attr, val in list(self._pending_ctx_writes.items()):
            with utils.logged_errors("audio_tasks_registry.SourceTask._flush_pending_ctx_writes"):
                setattr(ctx, attr, val)
        self._pending_ctx_writes.clear()

    def _timeslice_mode_hint(self) -> str:
        settings = getattr(self, "engine_settings", None)
        try:
            mode = str(getattr(settings, "ace_timeslice_mode", "fixed") or "fixed").strip().lower()
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.SourceTask._timeslice_mode_hint", exc)
            mode = "fixed"
        if mode not in {"fixed", "adaptive"}:
            mode = "fixed"
        return mode

    def _config_signature(self) -> Tuple[Any, ...]:
        ctx = self.source_context
        source_type = str(getattr(ctx, "source_type", "") or "").strip().upper() if ctx else ""
        filepath = str(getattr(ctx, "filepath", "") or "") if ctx else ""
        flow_id = build_live_flow_identifier(ctx) if ctx else ""
        live_latency_ms = int(getattr(ctx, "live_latency_ms", 10) or 10) if ctx else 10
        requested_sr = int(getattr(ctx, "live_udp_sample_rate", 0) or 0) if ctx else 0
        live_transport = str(getattr(ctx, "live_transport", "") or "").strip().upper() if ctx else ""
        live_device = str(getattr(ctx, "live_device_selector", "") or "") if ctx else ""
        live_udp_port = int(getattr(ctx, "live_udp_port", 0) or 0) if ctx else 0
        live_udp_channels = int(getattr(ctx, "live_udp_channels", 0) or 0) if ctx else 0
        return (
            source_type,
            filepath,
            flow_id,
            live_latency_ms,
            requested_sr,
            live_transport,
            live_device,
            live_udp_port,
            live_udp_channels,
        )

    def _frontend_runtime_config(self) -> Optional[Tuple[float, float, float, float]]:
        ctx = self.source_context
        if not ctx or not bool(getattr(ctx, "enable_preprocessing", False)):
            return None
        try:
            freq_min = float(max(getattr(ctx, "freq_min", 0.0) or 0.0, 0.0))
            freq_max = float(max(getattr(ctx, "freq_max", 0.0) or 0.0, 0.0))
            attack_ms = float(max(getattr(ctx, "envelope_attack", 0.0) or 0.0, 0.0))
            release_ms = float(max(getattr(ctx, "envelope_release", 0.0) or 0.0, 0.0))
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.SourceTask._frontend_runtime_config", exc)
            return None
        return (freq_min, freq_max, attack_ms, release_ms)

    def _should_materialize_local_frontend(self) -> bool:
        engine = getattr(self, "engine_ref", None)
        resolver = getattr(getattr(engine, "parallel", None), "_source_task_requires_local_frontend", None)
        if callable(resolver):
            with utils.logged_errors("audio_tasks_registry.SourceTask._should_materialize_local_frontend"):
                return bool(resolver(self))
        return True

    @staticmethod
    def _coerce_stereo_block(stereo: Any, frame_count: int) -> Optional[np.ndarray]:
        if not isinstance(stereo, np.ndarray):
            return None
        arr = np.asarray(stereo, dtype=np.float32)
        if arr.ndim != 2:
            return None
        if arr.shape[1] == int(frame_count):
            return arr.astype(np.float32, copy=False)
        if arr.shape[0] == int(frame_count):
            return arr.T.astype(np.float32, copy=False)
        return None

    def _analysis_frontend_tracks(
        self,
        mono: np.ndarray,
        stereo: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Tuple[float, float, float, float]]]:
        mono_arr = np.asarray(mono, dtype=np.float32).reshape(-1).astype(np.float32, copy=False)
        stereo_arr = self._coerce_stereo_block(stereo, mono_arr.shape[0])
        frontend_config = self._frontend_runtime_config()
        if frontend_config is None:
            return mono_arr, stereo_arr, None
        analysis_mono, analysis_stereo = self._frontend_processor.process(
            mono=mono_arr,
            stereo=stereo_arr,
            sample_rate=int(self.sample_rate),
            frontend_config=frontend_config,
        )
        return analysis_mono, analysis_stereo, frontend_config

    @staticmethod
    def _workspace_bucket_sizes() -> Tuple[int, ...]:
        sizes = set(_ADAPTIVE_FILE_BLOCK_BUCKETS) | set(_ADAPTIVE_LIVE_BLOCK_BUCKETS)
        sizes.add(int(MIN_BLOCK_SIZE))
        sizes.add(int(MAX_BLOCK_SIZE))
        return tuple(sorted(int(v) for v in sizes if int(v) > 0))

    def _workspace_for_size(self, length: int) -> Dict[str, np.ndarray]:
        n = max(int(length), 1)
        cached = self._workspace_pool.get(n)
        if cached is not None:
            return cached
        cached = {
            "silence": np.zeros(n, dtype=np.float32),
            "frontend": np.empty(n, dtype=np.float32),
            "env": np.empty(n, dtype=np.float32),
            "iir_prev_input": np.empty(n + 1, dtype=np.float32),
            "iir_u": np.empty(n, dtype=np.float32),
            "iir_pow": np.empty(n, dtype=np.float64),
            "iir_accum": np.empty(n, dtype=np.float64),
            "frontend_idx": np.arange(n, dtype=np.float64),
        }
        self._workspace_pool[n] = cached
        return cached

    def _adopt_workspace_for_size(self, length: int):
        ws = self._workspace_for_size(length)
        self.silence_buffer = ws["silence"]
        self._frontend_buffer = ws["frontend"]
        self._env_buffer = ws["env"]
        self._iir_prev_input = ws["iir_prev_input"]
        self._iir_u_buffer = ws["iir_u"]
        self._iir_pow_buffer = ws["iir_pow"]
        self._iir_accum_buffer = ws["iir_accum"]
        self._frontend_index_buffer = ws["frontend_idx"]

    def _prime_workspace_buckets(self):
        for n in self._workspace_bucket_sizes():
            self._workspace_for_size(n)

    def _set_block_size_from_engine(self, new_size: int):
        size = max(int(new_size), 1)
        if size == int(self.block_size):
            self._adopt_workspace_for_size(size)
            return
        if self.analysis:
            AudioFluxAnalyzer.release(self.analysis)
        self.block_size = size
        self._adopt_workspace_for_size(size)
        self._frontend_processor.reset()
        if self.source_ready:
            self.analysis = AudioFluxAnalyzer.borrow(self.sample_rate, size)
        else:
            self.analysis = None
        self._reset_sample_clocks()

    def _reopen_for_config_change(self):
        if self.reception is not None:
            with utils.logged_errors("audio_tasks_registry.SourceTask._reopen_for_config_change"):
                self.reception.close()
        self.reception = None
        if self.analysis is not None:
            with utils.logged_errors("audio_tasks_registry.SourceTask._reopen_for_config_change#2"):
                AudioFluxAnalyzer.release(self.analysis)
        self.analysis = None
        self._reset_sample_clocks()
        self.cached_output = None
        self.last_frame_processed = -1
        self.initialized = False
        self._frontend_processor.reset()
        self.source_ready = True
        self.live_ready = True
        self.last_error = ""

    def _load_source(self):
        """Load source audio into memory and prepare analyzers."""
        self.initialized = True
        ctx = self.source_context

        source_type = getattr(ctx, "source_type", "") if ctx else ""
        filepath = getattr(ctx, "filepath", "") if ctx else ""
        flow_id = build_live_flow_identifier(ctx) if ctx else ""
        if ctx is not None:
            self._pending_ctx_writes["flow_identifier"] = flow_id
        live_latency_ms = int(getattr(ctx, "live_latency_ms", 10) or 10) if ctx else 10
        target_sample_rate = None
        engine = getattr(self, "engine_ref", None)
        target_sr_resolver = getattr(getattr(engine, "compiler", None), "_source_task_target_sample_rate", None)
        if callable(target_sr_resolver):
            try:
                target_sample_rate = target_sr_resolver(self)
            except Exception as exc:
                utils.note_swallowed("audio_tasks_registry.SourceTask._load_source", exc)
                target_sample_rate = None

        self.reader_path = filepath
        self.reception = AudioReception(source_type, filepath, flow_id, live_latency_ms=live_latency_ms)
        opened = self.reception.open(
            block_size=self.block_size,
            target_sample_rate=target_sample_rate,
        )
        self._live_block_buffer.clear()
        if opened:
            self.reader_channels = self.reception.channels
            self.sample_rate = self.reception.sample_rate
            self.source_ready = True
            self.last_error = ""
            if ctx and getattr(ctx, "source_type", "") == "FILE":
                self._pending_ctx_writes["ace_source_error"] = ""
            if ctx and getattr(ctx, "source_type", "") == "LIVE":
                self._pending_ctx_writes["live_status"] = ""
            self.live_ready = True
        else:
            self.sample_rate = SAMPLE_RATE
            self.source_ready = False
            rec = self.reception
            self.last_error = (
                rec.format_error_summary()
                if rec is not None and hasattr(rec, "format_error_summary")
                else (getattr(rec, "last_error", "") if rec is not None else "Source open failed")
            ) or "Source open failed"
            if ctx and getattr(ctx, "source_type", "") == "LIVE":
                self._pending_ctx_writes["live_status"] = (
                    getattr(self.reception, "last_error", "") if self.reception else "Live input unavailable"
                )
            self.reception = None
            if ctx and getattr(ctx, "source_type", "") == "LIVE":
                self.live_ready = False
            if ctx and getattr(ctx, "source_type", "") == "FILE":
                self._pending_ctx_writes["ace_source_error"] = self.last_error

        # Recompute block size now that the real sample rate is known
        resolve_fn = self.resolve_block_size_fn or _resolve_block_size_default
        try:
            delta_frames = float(getattr(getattr(self, "engine_ref", None), "_current_delta_frames", 1.0) or 1.0)
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.SourceTask._load_source#2", exc)
            delta_frames = 1.0
        timeslice_mode = self._timeslice_mode_hint()
        try:
            new_block_size = int(
                resolve_fn(
                    self.sample_rate,
                    self.scene_fps,
                    source_type=source_type,
                    timeslice_mode=timeslice_mode,
                    delta_frames=delta_frames,
                )
            )
        except TypeError:
            new_block_size = int(resolve_fn(self.sample_rate, self.scene_fps))
        self._apply_block_size(new_block_size)
        self._reset_sample_clocks()

        if ctx:
            with utils.logged_errors("audio_tasks_registry.SourceTask._load_source#3"):
                duration_val = float(getattr(self.reception, "duration", 0.0) or 0.0)
                native_sample_rate = int(getattr(self.reception, "native_sample_rate", 0) or 0)
                if native_sample_rate <= 0:
                    native_sample_rate = int(self.sample_rate)
                self._pending_ctx_writes["sample_rate"] = int(self.sample_rate)
                self._pending_ctx_writes["native_sample_rate"] = int(native_sample_rate)
                self._pending_ctx_writes["duration"] = duration_val
                if hasattr(ctx, "live_status") and opened:
                    self._pending_ctx_writes["live_status"] = ""

        if not self.source_ready:
            self.analysis = None
            self._last_config_signature = self._config_signature()
            self._last_live_retry_ts = time.time()
            return

        self.analysis = AudioFluxAnalyzer.borrow(self.sample_rate, self.block_size)
        self._last_config_signature = self._config_signature()
        self._hpf_state = {"x1": 0.0, "y1": 0.0}
        self._lpf_state = {"x1": 0.0, "y1": 0.0}
        self._env_state = 0.0
        self._frontend_processor.reset()
        self._adopt_workspace_for_size(self.block_size)
        self._ensure_frontend_workspace(self.block_size)
        self._last_live_retry_ts = time.time()

    def _apply_block_size(self, new_size: int):
        """Update local buffers and ask the engine to retarget the graph size."""
        if new_size <= 0 or new_size == self.block_size:
            return
        self._set_block_size_from_engine(int(new_size))
        engine = getattr(self, "engine_ref", None)
        if engine and hasattr(getattr(engine, "frame", None), "adjust_graph_block_size_for_task"):
            engine.frame.adjust_graph_block_size_for_task(self, new_size)

    def _ensure_frontend_workspace(self, length: int):
        """Allocate/reuse working buffers for frontend filtering."""
        if length <= 0:
            return
        if self._frontend_buffer is None or self._frontend_buffer.shape[0] != int(length):
            self._adopt_workspace_for_size(int(length))

    @staticmethod
    def _butterworth_coeffs(cutoff_hz: float, sample_rate: float, mode: str):
        if cutoff_hz <= 0.0 or sample_rate <= 0.0 or cutoff_hz >= (sample_rate * 0.49):
            return None
        omega = math.tan(math.pi * cutoff_hz / sample_rate)
        norm = 1.0 / (1.0 + omega)
        if mode == "low":
            b0 = omega * norm
            b1 = omega * norm
            a1 = (omega - 1.0) * norm
        else:
            b0 = 1.0 * norm
            b1 = -1.0 * norm
            a1 = (omega - 1.0) * norm
        return float(b0), float(b1), float(a1)

    def _run_iir(self, data: np.ndarray, coeffs: Tuple[float, float, float], state: Dict[str, float], out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Vectorized single-pole IIR (biquad degenerated to 1st order) using a closed-form cumsum.
        Keeps stateful x1/y1 between calls.
        """
        b0, b1, a1 = coeffs
        n = data.shape[0]
        if out is None or out.shape != data.shape:
            out = np.empty_like(data, dtype=np.float32)
        if n == 0:
            return out

        self._ensure_frontend_workspace(n)

        x_prev = float(state.get("x1", 0.0))
        y_prev = float(state.get("y1", 0.0))
        if NUMBA_AVAILABLE:
            with utils.logged_errors("audio_tasks_registry.SourceTask._run_iir"):
                src = data.astype(np.float32, copy=False)
                filtered, x_last, y_last = _iir1_filter_block(
                    src,
                    float(b0),
                    float(b1),
                    float(a1),
                    float(x_prev),
                    float(y_prev),
                )
                np.copyto(out, filtered)
                state["x1"] = float(x_last)
                state["y1"] = float(y_last)
                return out

        prev_input = self._iir_prev_input
        if prev_input is None:
            prev_input = np.empty(n + 1, dtype=np.float32)
            self._iir_prev_input = prev_input

        prev_input[0] = x_prev
        np.copyto(prev_input[1:n + 1], data)

        u = self._iir_u_buffer[:n] if self._iir_u_buffer is not None else np.empty(n, dtype=np.float32)
        np.multiply(data, b0, out=u)
        if b1 != 0.0:
            u += b1 * prev_input[:n]

        a = -a1
        if abs(a) < 1e-8:
            out[:] = u
            state["x1"] = float(data[-1])
            state["y1"] = float(out[-1])
            return out

        pow_buf = self._iir_pow_buffer[:n] if self._iir_pow_buffer is not None else np.empty(n, dtype=np.float64)
        idx = self._frontend_index_buffer[:n] if self._frontend_index_buffer is not None else np.arange(n, dtype=np.float64)
        np.power(a, idx, out=pow_buf)

        scaled = self._iir_accum_buffer[:n] if self._iir_accum_buffer is not None else np.empty(n, dtype=np.float64)
        scaled[:] = u
        np.divide(scaled, pow_buf + 1e-12, out=scaled)
        np.cumsum(scaled, out=scaled)
        scaled += y_prev * a
        np.multiply(pow_buf, scaled, out=scaled)
        out[:] = scaled.astype(np.float32, copy=False)

        state["x1"] = float(data[-1])
        state["y1"] = float(out[-1])
        return out

    def _apply_frontend(self, block: np.ndarray) -> np.ndarray:
        """Backward-compatible mono wrapper around the shared frontend processor."""
        analysis_mono, _, _ = self._analysis_frontend_tracks(block, None)
        return analysis_mono

    def _reset_sample_clocks(self):
        self.read_cursor = 0
        self._last_seek_sample = -1
        self._last_frame_request = None
        self._live_clock_anchor_frame = None
        self._live_clock_anchor_sample = None
        self._live_clock_last_target_sample = None
        self._live_block_buffer.clear()

    def _requested_frame(self) -> int:
        try:
            return int(getattr(self, "_current_frame_request", 0) or 0)
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.SourceTask._requested_frame", exc)
            return 0

    def _samples_per_frame(self, sample_rate: int) -> float:
        sr = max(float(sample_rate), 1.0)
        fps = float(self.scene_fps) if float(self.scene_fps) > 0.0 else 24.0
        return sr / max(fps, 1e-6)

    def _file_sample_clock_target(self, frame_index: int) -> int:
        frame_start = int(getattr(self, "scene_frame_start", 1) or 1)
        frame_offset = max(int(frame_index) - frame_start, 0)
        target = float(frame_offset) * self._samples_per_frame(self.sample_rate)
        return max(int(round(target)), 0)

    def _live_sample_clock_target(self, frame_index: int, latest_capture_sample: int) -> int:
        sr = max(int(self.sample_rate), 1)
        spf = self._samples_per_frame(sr)
        live_latency_ms = float(getattr(self.source_context, "live_latency_ms", 0.0) or 0.0) if self.source_context else 0.0
        latency_samples = (live_latency_ms / 1000.0) * float(sr)
        desired_target = float(latest_capture_sample) - latency_samples

        if self._live_clock_anchor_frame is None or self._live_clock_anchor_sample is None:
            self._live_clock_anchor_frame = int(frame_index)
            self._live_clock_anchor_sample = float(desired_target)
        else:
            elapsed = float(int(frame_index) - int(self._live_clock_anchor_frame))
            predicted = float(self._live_clock_anchor_sample) + (elapsed * spf)
            error = desired_target - predicted

            # Hard realign after large jumps/stalls to avoid stale frame-to-sample mapping.
            if abs(error) > float(self.block_size * 8):
                self._live_clock_anchor_frame = int(frame_index)
                self._live_clock_anchor_sample = float(desired_target)
            else:
                # Soft phase correction keeps A/V alignment stable without visible jitter.
                max_correction = float(self.block_size) * 0.5
                correction = max(-max_correction, min(max_correction, error * 0.2))
                self._live_clock_anchor_sample = float(self._live_clock_anchor_sample) + correction

        elapsed = float(int(frame_index) - int(self._live_clock_anchor_frame or frame_index))
        target = float(self._live_clock_anchor_sample or 0.0) + (elapsed * spf)
        target_i = int(round(target))
        if self._live_clock_last_target_sample is not None:
            target_i = max(target_i, int(self._live_clock_last_target_sample))
        self._live_clock_last_target_sample = int(target_i)
        return max(target_i, 0)

    def _push_live_block(
        self,
        mono: np.ndarray,
        stereo: Optional[np.ndarray],
        sample_rate: int,
        capture_end_sample: int,
        capture_span_samples: int,
    ):
        cap_end = max(int(capture_end_sample), 0)
        cap_span = max(int(capture_span_samples), 1)
        mono_arr = np.asarray(mono, dtype=np.float32).reshape(-1)
        stereo_arr = None
        if isinstance(stereo, np.ndarray):
            stereo_arr = np.asarray(stereo, dtype=np.float32)
        entry = (
            mono_arr,
            stereo_arr,
            int(sample_rate),
            cap_end,
            cap_span,
        )
        if self._live_block_buffer and int(self._live_block_buffer[-1][3]) == cap_end:
            self._live_block_buffer[-1] = entry
            return
        self._live_block_buffer.append(entry)

    def _select_live_block_for_target(self, target_sample: int) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], int]]:
        if not self._live_block_buffer:
            return None
        best_entry = self._live_block_buffer[-1]
        best_distance = float("inf")
        target = int(target_sample)
        for entry in reversed(self._live_block_buffer):
            _, _, _, cap_end, cap_span = entry
            block_end = int(cap_end)
            block_start = max(block_end - int(cap_span), 0)
            if target < block_start:
                distance = float(block_start - target)
            elif target > block_end:
                distance = float(target - block_end)
            else:
                distance = 0.0
            if distance < best_distance:
                best_distance = distance
                best_entry = entry
                if distance <= 0.0:
                    break
        mono, stereo, sr, _end, _span = best_entry
        return mono, stereo, int(sr)

    def _sync_reader_cursor(self, desired_sample: int, force: bool = False):
        if self.reception is None or self.reception.reader is None:
            return
        if desired_sample < 0:
            return
        try:
            current_cursor = int(getattr(self.reception, "_cursor_sample", self.read_cursor) or self.read_cursor)
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.SourceTask._sync_reader_cursor", exc)
            current_cursor = int(self.read_cursor)
        self.read_cursor = current_cursor
        delta = int(desired_sample) - int(current_cursor)
        if delta == 0:
            return
        if not force:
            tolerance = max(8, int(self.block_size) // 4)
            if abs(delta) <= tolerance:
                return
            if delta >= 0 and delta <= self.block_size * 2:
                return
        try:
            if self.reception.seek(int(desired_sample)):
                self.read_cursor = int(desired_sample)
                self._last_seek_sample = int(desired_sample)
        except Exception as e:
            utils.log(f"Source cursor sync failed: {e}", "ERROR")

    def _read_next_block(self) -> Optional["AudioChunk"]:
        """
        Main-thread entry point for reading the next audio block.

        For FILE sources with an active prefetch thread: returns the pre-read
        chunk immediately (zero I/O) and triggers the next speculative read.
        Falls back to _read_next_block_sync on scrub/seek or first call.
        For LIVE sources: delegates directly to _read_next_block_sync.
        """
        source_type = str(
            getattr(getattr(self, "source_context", None), "source_type", "") or ""
        ).upper()
        frame_request = getattr(self, "_current_frame_request", None)

        if self._prefetch_enabled and source_type == "FILE" and frame_request is not None:
            if self._prefetch_ready.is_set() and self._prefetch_for_frame == frame_request:
                # Cache hit: take pre-read chunk, clear slot, trigger next read.
                self._prefetch_ready.clear()
                chunk = self._prefetch_chunk
                self._prefetch_chunk = None
                self._prefetch_target_frame = frame_request + 1
                self._prefetch_trigger.set()
                return chunk
            else:
                # Cache miss (scrub or first frame): stop prefetch, do sync read,
                # then restart prefetch for next sequential frame.
                self._stop_prefetch_thread()
                chunk = self._read_next_block_sync()
                if chunk is not None:
                    self._prefetch_target_frame = (
                        (frame_request + 1) if frame_request is not None else None
                    )
                    self._start_prefetch_thread()
                    self._prefetch_trigger.set()
                return chunk

        return self._read_next_block_sync()

    def _read_next_block_sync(self, track_perf_stage: bool = True) -> Optional["AudioChunk"]:
        """Synchronous block read.  May perform file I/O or queue dequeue."""
        t0 = time.perf_counter()
        try:
            current_signature = self._config_signature()
            if self.initialized and self._last_config_signature is not None and current_signature != self._last_config_signature:
                # Source IO params changed at runtime: reopen cleanly so readers/analyzers follow.
                self._reopen_for_config_change()
            if not self.initialized:
                self._load_source()
            if self._last_config_signature is None:
                self._last_config_signature = current_signature

            source_type = str(getattr(self.source_context, "source_type", "") or "").upper() if self.source_context else ""
            if source_type != "LIVE" and not self.source_ready:
                return None

            # Arm prefetch thread on the first successful FILE source open.
            if source_type == "FILE" and not self._prefetch_enabled:
                self._start_prefetch_thread()

            resolve_fn = self.resolve_block_size_fn or _resolve_block_size_default
            try:
                delta_frames = float(getattr(getattr(self, "engine_ref", None), "_current_delta_frames", 1.0) or 1.0)
            except Exception as exc:
                utils.note_swallowed("audio_tasks_registry.SourceTask._read_next_block_sync", exc)
                delta_frames = 1.0
            timeslice_mode = self._timeslice_mode_hint()
            try:
                target_block = int(
                    resolve_fn(
                        self.sample_rate,
                        self.scene_fps,
                        source_type=source_type,
                        timeslice_mode=timeslice_mode,
                        delta_frames=delta_frames,
                    )
                )
            except TypeError:
                try:
                    target_block = int(resolve_fn(self.sample_rate, self.scene_fps, source_type=source_type))
                except TypeError:
                    target_block = int(resolve_fn(self.sample_rate, self.scene_fps))
            target_block = max(MIN_BLOCK_SIZE, min(target_block, MAX_BLOCK_SIZE))
            if target_block != self.block_size:
                self._apply_block_size(target_block)

            if source_type == "LIVE":
                if (self.reception is None or getattr(self.reception, "kind", "") == "live_stub"):
                    now = time.time()
                    if hasattr(self.source_context, "live_status"):
                        message = (
                            (getattr(self.reception, "last_error", "") if self.reception else "")
                            or str(getattr(self, "last_error", "") or "")
                            or str(getattr(self.source_context, "live_status", "") or "")
                            or "Live input unavailable"
                        )
                        self.source_context.live_status = message or "Live input unavailable"
                    if (now - self._last_live_retry_ts) > 2.0:
                        self._load_source()
                    self.live_ready = False
                    return None
                elif hasattr(self.source_context, "live_status") and getattr(self.source_context, "live_status", ""):
                    # Stream recovered, clear any stale warning
                    self.source_context.live_status = ""
                    self.live_ready = True

            mono = None
            stereo = None
            frame_request = self._requested_frame()

            if source_type == "LIVE" and self._last_frame_request is not None:
                frame_jump = int(frame_request) - int(self._last_frame_request)
                if frame_jump <= -1 or frame_jump > 3:
                    self._live_clock_anchor_frame = None
                    self._live_clock_anchor_sample = None
                    self._live_clock_last_target_sample = None

            if self.reception:
                if source_type == "LIVE":
                    block = self.reception.read_block(self.block_size)
                    if block is not None:
                        mono_live, stereo_live, sr_live = block
                        self.sample_rate = int(sr_live)
                        self.live_ready = True
                        capture_end = int(getattr(self.reception, "_live_last_capture_sample", 0) or 0)
                        capture_span = int(
                            getattr(self.reception, "_live_last_capture_block_samples", mono_live.shape[-1]) or mono_live.shape[-1]
                        )
                        self._push_live_block(mono_live, stereo_live, sr_live, capture_end, capture_span)
                    elif getattr(self.reception, "_eof", False):
                        return None

                    if self._live_block_buffer:
                        latest_capture = int(self._live_block_buffer[-1][3])
                        live_target = self._live_sample_clock_target(frame_request, latest_capture)
                        selected = self._select_live_block_for_target(live_target)
                        if selected is not None:
                            mono, stereo, sr = selected
                            self.sample_rate = int(sr)
                            self.read_cursor = int(live_target)
                            self.live_ready = True
                        # If selected is None, retain current live_ready so the engine
                        # keeps calling process() and the buffer can fill on the next frame.
                    # If buffer is empty (stream just opened, first callback not yet fired),
                    # retain live_ready = True so the engine does not deadlock by skipping
                    # process() indefinitely before any audio arrives.
                else:
                    desired_sample = self._file_sample_clock_target(frame_request)
                    file_force_seek = False
                    if self._last_frame_request is None:
                        file_force_seek = True
                    else:
                        frame_jump = int(frame_request) - int(self._last_frame_request)
                        # Hard-seek only on scrubs/backward jumps or large frame skips.
                        if frame_jump <= 0 or frame_jump > 3:
                            file_force_seek = True
                    self._sync_reader_cursor(desired_sample, force=file_force_seek)
                    block = self.reception.read_block(self.block_size)
                    if block is not None:
                        mono, stereo, sr = block
                        self.sample_rate = int(sr)
                        try:
                            self.read_cursor = int(
                                getattr(self.reception, "_cursor_sample", desired_sample + int(mono.shape[-1]))
                                or (desired_sample + int(mono.shape[-1]))
                            )
                        except Exception as exc:
                            utils.note_swallowed("audio_tasks_registry.SourceTask._read_next_block_sync#2", exc)
                            self.read_cursor = desired_sample + int(mono.shape[-1])
                    elif getattr(self.reception, "_eof", False):
                        return None

            self._last_frame_request = int(frame_request)

            live_kind = str(getattr(self.reception, "kind", "")) if self.reception else ""
            if source_type == "LIVE" and self.reception and live_kind.endswith("sounddevice"):
                # Skip extra processing when the live queue has not delivered a block yet.
                if mono is None:
                    return None

            if mono is None:
                mono = np.copy(self.silence_buffer)

            mono_raw = np.asarray(mono, dtype=np.float32).reshape(-1).astype(np.float32, copy=False)
            stereo_raw = self._coerce_stereo_block(stereo, mono_raw.shape[0])
            analysis_mono = None
            analysis_stereo = None
            frontend_config = self._frontend_runtime_config()
            if self._should_materialize_local_frontend():
                analysis_mono, analysis_stereo, frontend_config = self._analysis_frontend_tracks(
                    mono_raw,
                    stereo_raw,
                )
            live_latency_ms = float(getattr(self.source_context, "live_latency_ms", 0.0) or 0.0) if source_type == "LIVE" else 0.0
            sr = max(float(self.sample_rate), 1.0)
            block_latency_ms = (float(self.block_size) / sr) * 1000.0
            total_latency_ms = float(max(0.0, live_latency_ms + block_latency_ms))
            return AudioChunk(
                mono_raw,
                stereo_raw,
                self.sample_rate,
                provider=self,
                analysis_mono=analysis_mono,
                analysis_stereo=analysis_stereo,
                frontend_config=frontend_config,
                lineage=(self.node_id,),
                latency_ms=total_latency_ms,
            )
        finally:
            engine = getattr(self, "engine_ref", None)
            elapsed = time.perf_counter() - t0
            if bool(track_perf_stage):
                perf_add = getattr(getattr(engine, "diagnostics", None), "_perf_add", None)
                if callable(perf_add):
                    with utils.logged_errors("audio_tasks_registry.SourceTask._read_next_block_sync#3"):
                        perf_add("source_read", elapsed)
            else:
                prefetch_add = getattr(getattr(engine, "diagnostics", None), "_record_source_prefetch", None)
                if callable(prefetch_add):
                    with utils.logged_errors("audio_tasks_registry.SourceTask._read_next_block_sync#4"):
                        prefetch_add(elapsed)

    def compute_feature(self, name: str, chunk: AudioChunk):
        if not self.analysis:
            return None
        return self.analysis.compute_feature(name, chunk)

    def close(self):
        # Stop prefetch thread before closing reception so it cannot call
        # reception.read_block() after the underlying resource is freed.
        self._stop_prefetch_thread()
        if self.reception:
            with utils.logged_errors("audio_tasks_registry.SourceTask.close"):
                self.reception.close()
            self.reception = None
        if self.analysis:
            AudioFluxAnalyzer.release(self.analysis)
            self.analysis = None
        self._last_config_signature = None
        self._reset_sample_clocks()
        self.live_ready = True
        self.source_ready = True
        self.last_error = ""
        self._frontend_processor.reset()
        self._workspace_pool.clear()
        self.initialized = False


class SourceSpecsTask(ACEAudioTask):
    """
    Outputs static metadata about the source with lightweight long-term analysis.
    Passes the incoming audio through unchanged on the last output.
    """
    _GLOBAL_LONG_TERM_CACHE: Dict[Tuple[str, float], Dict[str, Any]] = {}
    _GLOBAL_ACTIVE_ANALYSIS: Dict[Tuple[str, float], int] = {}
    _GLOBAL_CACHE_TTL = 120.0  # seconds
    _GLOBAL_CACHE_MAX = 128
    _LONG_TERM_MAX_SECONDS = 15.0
    _LONG_TERM_BLOCKS_PER_TICK = 3
    _LONG_TERM_TICK_INTERVAL_S = 0.05
    _GLOBAL_CACHE_LOCK = threading.Lock()
    _BG_QUEUE_MAX = 16
    _BG_TASK_QUEUE: Queue = Queue(maxsize=_BG_QUEUE_MAX)
    _BG_PENDING: set[Tuple[str, float]] = set()
    _BG_QUEUED: set[Tuple[str, float]] = set()
    _BG_THREAD_LOCK = threading.Lock()
    _BG_STOP_EVENT = threading.Event()
    _BG_THREAD: Optional[threading.Thread] = None
    # Pending ctx writes: (target_key, bpm, beats_per_bar). Drained on main thread.
    _PENDING_CTX_WRITES: "Queue[Tuple[Tuple[str, float], float, int]]" = Queue()

    def __init__(self, node_id, params, block_size):
        super().__init__(node_id, params, block_size)
        self._static_specs: Optional[Tuple[float, int, float, float, float, float]] = None
        self._long_term_cache: Dict[str, Any] = {}
        self._long_term_target: Optional[Tuple[str, float]] = None
        self._long_term_done: bool = False
        self._auto_bpm_key = "ace_bpm_auto"
        self._auto_meter_key = "ace_meter_auto"
        self._analysis_rec: Optional[AudioReception] = None
        self._analysis_analyzer = None
        self._analysis_block: int = 0
        self._analysis_sr: int = SAMPLE_RATE
        self._analysis_flux_vals: list[float] = []
        self._analysis_total_samples: int = 0
        self._analysis_frame_idx: int = 0
        self._analysis_next_tick_ts: float = 0.0

    def process(self, inputs):
        # Flush any detected-tempo results from the background worker.
        # Guarded to main-thread execution only so worker/bake threads never mutate bpy state.
        if threading.current_thread() is threading.main_thread():
            SourceSpecsTask.flush_pending_ctx_writes(
                getattr(getattr(self, "engine_ref", None), "_source_specs_task_registry", {})
            )
        # Preserve the audio path
        chunk_in = _ensure_audio_chunk(inputs.get(0), self.block_size)
        ctx = self.source_context
        if ctx:
            # If the user changed BPM or meter after an automatic estimate, honor
            # that override and stop auto-updates for this source.
            with utils.logged_errors("audio_tasks_registry.SourceSpecsTask.process"):
                auto_val = ctx.get(self._auto_bpm_key, None) if hasattr(ctx, "get") else None
                if auto_val is not None:
                    current = float(getattr(ctx, "bpm", 120.0))
                    if abs(current - float(auto_val)) > 1e-3:
                        ctx["ace_bpm_user_override"] = True
                        self._long_term_done = True
                        self._reset_long_term_analysis_state(clear_target=False)
            with utils.logged_errors("audio_tasks_registry.SourceSpecsTask.process#2"):
                auto_meter = ctx.get(self._auto_meter_key, None) if hasattr(ctx, "get") else None
                if auto_meter is not None:
                    current_meter = int(getattr(ctx, "beats_per_bar", 4) or 4)
                    if current_meter != int(auto_meter):
                        ctx["ace_meter_user_override"] = True
                        self._long_term_done = True
                        self._reset_long_term_analysis_state(clear_target=False)
        fps = getattr(self, "scene_fps", 24.0)
        self._maybe_run_long_term_analysis(ctx)

        bpm = float(self._long_term_cache.get("bpm", getattr(ctx, "bpm", 120.0) if ctx else 120.0) or 120.0)
        beats_per_bar = int(getattr(ctx, "beats_per_bar", 4) or 4) if ctx else 4
        sample_rate = float(getattr(ctx, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE) if ctx else float(SAMPLE_RATE)
        duration = float(getattr(ctx, "duration", 0.0) or 0.0) if ctx else 0.0

        fps_val = float(fps) if float(fps) > 0.0 else 24.0
        if (
            self._static_specs is None
            or self._static_specs[2] != sample_rate
            or self._static_specs[1] != beats_per_bar
            or self._static_specs[4] != bpm
            or self._static_specs[5] != fps_val
        ):
            beats_per_frame = float(bpm / max(60.0 * fps_val, 1e-6))
            self._static_specs = (beats_per_frame, beats_per_bar, sample_rate, duration, bpm, fps_val)
        else:
            beats_per_frame, beats_per_bar, sample_rate, duration, _, _ = self._static_specs

        frames_per_beat = float(1.0 / beats_per_frame) if beats_per_frame > 1e-6 else 0.0
        engine = getattr(self, "engine_ref", None)
        current_frame = getattr(engine, "current_frame", 0) if engine else 0
        measure_len_frames = float(beats_per_bar) / beats_per_frame if beats_per_frame > 0 else 0.0
        if measure_len_frames > 0.0:
            pos_in_measure = (float(current_frame) * beats_per_frame) % float(beats_per_bar)
        else:
            pos_in_measure = 0.0

        beat_phase = pos_in_measure / float(beats_per_bar) if beats_per_bar > 0 else 0.0
        # Output order matches ACE_NODES: Frames per Beat, Beat Phase, Audio passthrough
        return (frames_per_beat, beat_phase, chunk_in)

    def close(self):
        self._reset_long_term_analysis_state(clear_target=True)


    @staticmethod
    def _user_overrode_tempo(ctx: Optional[Any]) -> bool:
        if not ctx:
            return False
        getter = getattr(ctx, "get", None)
        if callable(getter):
            try:
                if getter("ace_bpm_user_override", False) or getter("ace_meter_user_override", False):
                    return True
            except Exception as exc:
                utils.note_swallowed("audio_tasks_registry.SourceSpecsTask._user_overrode_tempo", exc)
                return False
        return False

    def _reset_long_term_analysis_state(self, clear_target: bool = False):
        self._release_global_analysis_slot(self._long_term_target)
        rec = self._analysis_rec
        if rec is not None:
            with utils.logged_errors("audio_tasks_registry.SourceSpecsTask._reset_long_term_analysis_state"):
                rec.close()
        self._analysis_rec = None

        analyzer = self._analysis_analyzer
        if analyzer is not None:
            with utils.logged_errors("audio_tasks_registry.SourceSpecsTask._reset_long_term_analysis_state#2"):
                AudioFluxAnalyzer.release(analyzer)
        self._analysis_analyzer = None

        self._analysis_block = 0
        self._analysis_sr = SAMPLE_RATE
        self._analysis_flux_vals = []
        self._analysis_total_samples = 0
        self._analysis_frame_idx = 0
        self._analysis_next_tick_ts = 0.0
        if clear_target:
            self._long_term_target = None

    def _claim_global_analysis_slot(self, target: Tuple[str, float]) -> bool:
        owner = SourceSpecsTask._GLOBAL_ACTIVE_ANALYSIS.get(target)
        me = int(id(self))
        if owner is None or int(owner) == me:
            SourceSpecsTask._GLOBAL_ACTIVE_ANALYSIS[target] = me
            return True
        return False

    def _release_global_analysis_slot(self, target: Optional[Tuple[str, float]] = None):
        active_target = target if target is not None else self._long_term_target
        if active_target is None:
            return
        owner = SourceSpecsTask._GLOBAL_ACTIVE_ANALYSIS.get(active_target)
        if owner is None:
            return
        if int(owner) == int(id(self)):
            SourceSpecsTask._GLOBAL_ACTIVE_ANALYSIS.pop(active_target, None)

    @staticmethod
    def resolve_analysis_target(ctx: Optional[Any]) -> Optional[Tuple[str, float]]:
        if not ctx or str(getattr(ctx, "source_type", "") or "").upper() != "FILE":
            return None
        filepath = str(getattr(ctx, "filepath", "") or "").strip()
        if not filepath or not os.path.isfile(filepath):
            return None
        try:
            return (filepath, float(os.path.getmtime(filepath)))
        except OSError:
            return None

    @classmethod
    def get_cached_analysis(cls, target: Optional[Tuple[str, float]]) -> Optional[Dict[str, Any]]:
        if target is None:
            return None
        with cls._GLOBAL_CACHE_LOCK:
            raw = cls._GLOBAL_LONG_TERM_CACHE.get(target)
            cached = dict(raw) if isinstance(raw, dict) else None
        if not cached:
            return None
        if (time.time() - float(cached.get("ts", 0.0) or 0.0)) > cls._GLOBAL_CACHE_TTL:
            with cls._GLOBAL_CACHE_LOCK:
                cls._GLOBAL_LONG_TERM_CACHE.pop(target, None)
            return None
        return cached

    @classmethod
    def _store_global_analysis_result(
        cls,
        target: Tuple[str, float],
        bpm_est: float,
        beats_per_bar: int,
    ):
        with cls._GLOBAL_CACHE_LOCK:
            cls._GLOBAL_LONG_TERM_CACHE[target] = {
                "bpm": float(bpm_est),
                "beats_per_bar": int(max(1, beats_per_bar)),
                "ts": time.time(),
            }
            if len(cls._GLOBAL_LONG_TERM_CACHE) > cls._GLOBAL_CACHE_MAX:
                overflow = len(cls._GLOBAL_LONG_TERM_CACHE) - cls._GLOBAL_CACHE_MAX
                stale_items = heapq.nsmallest(
                    overflow,
                    cls._GLOBAL_LONG_TERM_CACHE.items(),
                    key=lambda kv: float(kv[1].get("ts", 0.0)) if isinstance(kv[1], dict) else 0.0,
                )
                for stale_key, _ in stale_items:
                    cls._GLOBAL_LONG_TERM_CACHE.pop(stale_key, None)

    @classmethod
    def _store_global_bpm_cache(cls, target: Tuple[str, float], bpm_est: float, beats_per_bar: int = 4):
        cls._store_global_analysis_result(target, bpm_est, beats_per_bar)

    @classmethod
    def _store_global_analysis_failure(cls, target: Tuple[str, float], error: str = ""):
        with cls._GLOBAL_CACHE_LOCK:
            cls._GLOBAL_LONG_TERM_CACHE[target] = {
                "bpm": None,
                "beats_per_bar": None,
                "error": str(error or "Tempo analysis failed"),
                "ts": time.time(),
            }
            if len(cls._GLOBAL_LONG_TERM_CACHE) > cls._GLOBAL_CACHE_MAX:
                overflow = len(cls._GLOBAL_LONG_TERM_CACHE) - cls._GLOBAL_CACHE_MAX
                stale_items = heapq.nsmallest(
                    overflow,
                    cls._GLOBAL_LONG_TERM_CACHE.items(),
                    key=lambda kv: float(kv[1].get("ts", 0.0)) if isinstance(kv[1], dict) else 0.0,
                )
                for stale_key, _ in stale_items:
                    cls._GLOBAL_LONG_TERM_CACHE.pop(stale_key, None)

    @classmethod
    def _ensure_background_worker(cls):
        thread = cls._BG_THREAD
        if thread is not None and thread.is_alive():
            return
        with cls._BG_THREAD_LOCK:
            thread = cls._BG_THREAD
            if thread is not None and thread.is_alive():
                return
            cls._BG_STOP_EVENT.clear()
            cls._BG_THREAD = threading.Thread(
                target=cls._background_worker_loop,
                name="ACE_SourceSpecsLongTermWorker",
                daemon=True,
            )
            cls._BG_THREAD.start()

    @classmethod
    def shutdown_background_worker(cls):
        cls._BG_STOP_EVENT.set()
        thread = cls._BG_THREAD
        if thread is not None and thread.is_alive():
            with utils.logged_errors("audio_tasks_registry.SourceSpecsTask.shutdown_background_worker"):
                thread.join(timeout=0.25)
        cls._BG_THREAD = None
        with cls._GLOBAL_CACHE_LOCK:
            cls._BG_PENDING.clear()
            cls._BG_QUEUED.clear()
            cls._GLOBAL_ACTIVE_ANALYSIS.clear()
        try:
            while True:
                cls._BG_TASK_QUEUE.get_nowait()
                cls._BG_TASK_QUEUE.task_done()
        except Empty:
            pass
        cls.clear_pending_ctx_writes_queue()

    @classmethod
    def clear_pending_ctx_writes_queue(cls) -> None:
        try:
            while True:
                cls._PENDING_CTX_WRITES.get_nowait()
                cls._PENDING_CTX_WRITES.task_done()
        except Empty:
            pass

    @classmethod
    def flush_pending_ctx_writes(cls, task_registry: dict) -> None:
        """Apply detected-tempo results to bpy PropertyGroup objects.

        Must be called from the main thread.
        """
        utils.assert_main_thread("SourceSpecsTask.flush_pending_ctx_writes")
        try:
            while True:
                payload = cls._PENDING_CTX_WRITES.get_nowait()
                if len(payload) >= 4:
                    target, bpm_val, beats_per_bar, generation_id = payload[:4]
                else:
                    target, bpm_val, beats_per_bar = payload
                    generation_id = None
                task = task_registry.get(target)
                ctx = getattr(task, "source_context", None) if task else None
                if ctx is None:
                    continue
                task_generation = int(getattr(task, "undo_generation", 0) or 0)
                engine_generation = int(
                    getattr(getattr(task, "engine_ref", None), "undo_generation", task_generation) or task_generation
                )
                if generation_id is not None and int(generation_id or 0) != engine_generation:
                    continue
                if task_generation != engine_generation:
                    continue
                if cls._user_overrode_tempo(ctx):
                    continue
                with utils.logged_errors("audio_tasks_registry.SourceSpecsTask.flush_pending_ctx_writes#2"):
                    ctx[task._auto_bpm_key] = float(bpm_val)
                    ctx[task._auto_meter_key] = int(max(1, beats_per_bar))
                    ctx.bpm = float(bpm_val)
                    ctx.beats_per_bar = int(max(1, beats_per_bar))
                    ctx["ace_bpm_user_override"] = False
                    ctx["ace_meter_user_override"] = False
        except Empty:
            pass
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.SourceSpecsTask.flush_pending_ctx_writes", exc)

    @classmethod
    def _queue_background_analysis(cls, target: Tuple[str, float]) -> bool:
        cls._ensure_background_worker()
        with cls._GLOBAL_CACHE_LOCK:
            if target in cls._BG_QUEUED:
                return True
            cls._BG_PENDING.add(target)
        try:
            cls._BG_TASK_QUEUE.put_nowait(target)
            with cls._GLOBAL_CACHE_LOCK:
                cls._BG_QUEUED.add(target)
            return True
        except Full:
            # Queue is full. Keep target in _BG_PENDING so callers avoid flooding
            # duplicate enqueue attempts while the worker drains existing items.
            return False

    @classmethod
    def _background_worker_loop(cls):
        while not cls._BG_STOP_EVENT.is_set():
            try:
                target = cls._BG_TASK_QUEUE.get(timeout=0.25)
            except Empty:
                continue
            with cls._GLOBAL_CACHE_LOCK:
                cls._BG_QUEUED.discard(target)

            try:
                detected = cls._estimate_tempo_for_target(target)
                if not detected:
                    cls._store_global_analysis_failure(target)
                else:
                    cls._store_global_analysis_result(
                        target,
                        float(detected["bpm"]),
                        int(detected["beats_per_bar"]),
                    )
            except Exception as exc:
                utils.note_swallowed("audio_tasks_registry.SourceSpecsTask._background_worker_loop", exc)
                cls._store_global_analysis_failure(target)
            finally:
                with cls._GLOBAL_CACHE_LOCK:
                    cls._BG_PENDING.discard(target)
                    cls._BG_QUEUED.discard(target)
                with utils.logged_errors("audio_tasks_registry.SourceSpecsTask._background_worker_loop#2"):
                    cls._BG_TASK_QUEUE.task_done()

    @staticmethod
    def _flux_peak_indices(flux_arr: np.ndarray) -> np.ndarray:
        flux = np.asarray(flux_arr, dtype=np.float32).reshape(-1)
        if flux.size < 4:
            return np.empty(0, dtype=np.int32)
        mean = float(np.mean(flux))
        std = float(np.std(flux))
        thresh = mean + (0.5 * std)
        mids = flux[1:-1]
        if mids.size <= 0:
            return np.empty(0, dtype=np.int32)
        peaks_mask = (mids > flux[:-2]) & (mids > flux[2:]) & (mids > thresh)
        return (np.nonzero(peaks_mask)[0] + 1).astype(np.int32, copy=False)

    @classmethod
    def _estimate_beats_per_bar_from_flux(
        cls,
        flux_arr: np.ndarray,
        peaks: np.ndarray,
        beat_period_blocks: float,
    ) -> int:
        flux = np.asarray(flux_arr, dtype=np.float32).reshape(-1)
        peak_idx = np.asarray(peaks, dtype=np.int32).reshape(-1)
        if flux.size < 8 or peak_idx.size < 4 or beat_period_blocks <= 1e-6:
            return 4

        strengths = flux[peak_idx].astype(np.float64, copy=False)
        mean_strength = float(np.mean(strengths) or 0.0)
        if mean_strength <= 1e-9:
            return 4

        beat_numbers = np.rint(
            (peak_idx.astype(np.float64) - float(peak_idx[0])) / max(float(beat_period_blocks), 1e-6)
        ).astype(np.int32, copy=False)
        centered_flux = flux.astype(np.float64) - float(np.mean(flux))
        best_candidate = 4
        best_score = -1.0e12
        max_candidate = min(8, max(2, int(peak_idx.size // 2)))

        for candidate in range(2, max_candidate + 1):
            sums = np.zeros(candidate, dtype=np.float64)
            counts = np.zeros(candidate, dtype=np.float64)
            for beat_num, strength in zip(beat_numbers, strengths):
                idx = int(beat_num % candidate)
                sums[idx] += float(strength)
                counts[idx] += 1.0
            if np.count_nonzero(counts) < candidate:
                continue

            means = np.divide(sums, np.maximum(counts, 1.0))
            normalized = means / mean_strength
            variance_score = float(np.var(normalized))
            downbeat_score = float(normalized[0] - np.mean(normalized[1:])) if candidate > 1 else 0.0

            periodicity_score = 0.0
            lag = int(round(float(beat_period_blocks) * float(candidate)))
            if 0 < lag < centered_flux.size:
                lhs = centered_flux[:-lag]
                rhs = centered_flux[lag:]
                denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
                if denom > 1e-9:
                    periodicity_score = float(np.dot(lhs, rhs) / denom)

            score = variance_score + max(downbeat_score, 0.0) + (0.5 * max(periodicity_score, 0.0))
            score -= 0.05 * float(candidate - 2)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        return int(max(best_candidate, 1))

    @classmethod
    def _estimate_tempo_from_flux(
        cls,
        flux_vals: list[float] | np.ndarray,
        block_size: int,
        sample_rate: float,
    ) -> Optional[Dict[str, Any]]:
        flux_arr = np.asarray(flux_vals, dtype=np.float32).reshape(-1)
        if flux_arr.size < 4:
            return None

        peaks = cls._flux_peak_indices(flux_arr)
        if peaks.size < 2:
            return None

        intervals = np.diff(peaks).astype(np.float64)
        intervals = intervals[intervals > 0.5]
        if intervals.size == 0:
            return None

        beat_period_blocks = float(np.median(intervals))
        seconds_per_beat = beat_period_blocks * (float(block_size) / max(float(sample_rate), 1.0))
        if seconds_per_beat <= 1e-6:
            return None

        bpm_est = 60.0 / seconds_per_beat
        if not np.isfinite(bpm_est):
            return None

        beats_per_bar = cls._estimate_beats_per_bar_from_flux(flux_arr, peaks, beat_period_blocks)
        return {
            "bpm": float(np.clip(bpm_est, 20.0, 300.0)),
            "beats_per_bar": int(max(1, beats_per_bar)),
        }

    @classmethod
    def _estimate_tempo_for_target(cls, target: Tuple[str, float]) -> Optional[Dict[str, Any]]:
        filepath, _mtime = target
        if not filepath or not os.path.isfile(filepath):
            return None

        rec = AudioReception("FILE", filepath=filepath)
        with _get_file_open_lock(filepath):
            opened = rec.open()
        if not opened:
            return None

        try:
            sr = int(getattr(rec, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
            block = _resolve_block_size_default(sr, 24.0)
            block = int(max(MIN_BLOCK_SIZE, min(block, MAX_BLOCK_SIZE)))
            max_samples = int(max(float(sr), 1.0) * cls._LONG_TERM_MAX_SECONDS)

            flux_vals: list[float] = []
            prev_mag: Optional[np.ndarray] = None
            total_samples = 0

            while total_samples < max_samples and not cls._BG_STOP_EVENT.is_set():
                payload = rec.read_block(block)
                if payload is None:
                    break
                mono, _stereo, sr_val = payload
                mono_arr = np.asarray(mono, dtype=np.float32).reshape(-1)
                if mono_arr.size <= 0:
                    break
                sr = int(max(int(sr_val or sr), 1))
                total_samples += int(mono_arr.size)

                mag = np.abs(np.fft.rfft(mono_arr, n=block)).astype(np.float32, copy=False)
                if prev_mag is not None and prev_mag.shape == mag.shape:
                    diff = mag - prev_mag
                    flux_vals.append(float(np.sum(np.maximum(diff, 0.0), dtype=np.float64)))
                prev_mag = mag

            return cls._estimate_tempo_from_flux(flux_vals, block, sr)
        finally:
            with utils.logged_errors("audio_tasks_registry.SourceSpecsTask._estimate_tempo_for_target"):
                rec.close()

    def _apply_detected_tempo(
        self,
        ctx: Optional[Any],
        bpm_val: float,
        beats_per_bar: int,
        *,
        respect_override: bool = True,
    ):
        # Update the instance cache (task-local state).
        self._long_term_cache["bpm"] = float(bpm_val)
        self._long_term_cache["beats_per_bar"] = int(max(1, beats_per_bar))
        self._long_term_done = True
        # Do not write bpy properties here: this may be called from background work.
        # Queue the write for main-thread flush instead.
        if ctx and (not respect_override or not SourceSpecsTask._user_overrode_tempo(ctx)):
            target = self._long_term_target
            if target is not None:
                with utils.logged_errors("audio_tasks_registry.SourceSpecsTask._apply_detected_tempo"):
                    SourceSpecsTask._PENDING_CTX_WRITES.put_nowait(
                        (
                            target,
                            float(bpm_val),
                            int(max(1, beats_per_bar)),
                            int(getattr(self, "undo_generation", 0) or 0),
                        )
                    )

    def _start_long_term_analysis(self, target: Tuple[str, float]) -> bool:
        filepath, _ = target
        rec = AudioReception("FILE", filepath=filepath)
        with _get_file_open_lock(filepath):
            opened = rec.open()
        if not opened:
            return False
        sr = int(getattr(rec, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
        block = _resolve_block_size_default(sr, getattr(self, "scene_fps", 24.0))
        block = int(max(MIN_BLOCK_SIZE, min(block, MAX_BLOCK_SIZE)))
        try:
            analyzer = AudioFluxAnalyzer(sr, block)
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.SourceSpecsTask._start_long_term_analysis", exc)
            with utils.logged_errors("audio_tasks_registry.SourceSpecsTask._start_long_term_analysis#2"):
                rec.close()
            return False

        self._analysis_rec = rec
        self._analysis_analyzer = analyzer
        self._analysis_block = block
        self._analysis_sr = sr
        self._analysis_flux_vals = []
        self._analysis_total_samples = 0
        self._analysis_frame_idx = 0
        return True

    def _finalize_long_term_analysis(self, ctx: Optional[Any]):
        flux_vals = list(self._analysis_flux_vals)
        sr = max(float(self._analysis_sr), 1.0)
        block = max(int(self._analysis_block), 1)
        detected = self._estimate_tempo_from_flux(flux_vals, block, sr)
        if detected:
            target = self._long_term_target
            if target is not None:
                SourceSpecsTask._store_global_analysis_result(
                    target,
                    float(detected["bpm"]),
                    int(detected["beats_per_bar"]),
                )
            self._apply_detected_tempo(
                ctx,
                float(detected["bpm"]),
                int(detected["beats_per_bar"]),
            )
        else:
            self._long_term_done = True
        self._reset_long_term_analysis_state(clear_target=False)

    def _step_long_term_analysis(self, ctx: Optional[Any]):
        rec = self._analysis_rec
        analyzer = self._analysis_analyzer
        if rec is None or analyzer is None:
            self._long_term_done = True
            return

        max_samples = int(max(float(self._analysis_sr), 1.0) * self._LONG_TERM_MAX_SECONDS)
        reached_end = False
        for _ in range(max(int(self._LONG_TERM_BLOCKS_PER_TICK), 1)):
            if self._analysis_total_samples >= max_samples:
                reached_end = True
                break
            chunk = rec.read_block(self._analysis_block)
            if chunk is None:
                reached_end = True
                break
            mono, stereo, sr_val = chunk
            self._analysis_total_samples += int(np.asarray(mono).shape[-1])
            temp_chunk = AudioChunk(
                mono,
                stereo,
                sr_val,
                frame_index=self._analysis_frame_idx,
                lineage=("long_term",),
            )
            flux = analyzer.compute_feature("flux", temp_chunk)
            if flux is not None:
                self._analysis_flux_vals.append(float(flux))
            self._analysis_frame_idx += 1

        if reached_end or self._analysis_total_samples >= max_samples:
            self._finalize_long_term_analysis(ctx)

    def _maybe_run_long_term_analysis(self, ctx: Optional[Any]):
        target = self.resolve_analysis_target(ctx)
        if target is None:
            self._reset_long_term_analysis_state(clear_target=True)
            return
        if self._user_overrode_tempo(ctx):
            self._long_term_done = True
            self._reset_long_term_analysis_state(clear_target=False)
            return

        if self._long_term_target != target:
            self._reset_long_term_analysis_state(clear_target=False)
            self._long_term_target = target
            self._long_term_cache.clear()
            self._long_term_done = False

        cached = SourceSpecsTask.get_cached_analysis(target)
        if cached:
            bpm_cached = cached.get("bpm", None)
            beats_per_bar_cached = cached.get("beats_per_bar", None)
            if bpm_cached is not None:
                self._apply_detected_tempo(
                    ctx,
                    float(bpm_cached),
                    int(beats_per_bar_cached or 4),
                )
            else:
                self._long_term_done = True
            return

        if self._long_term_done:
            return

        now = time.time()
        if now < self._analysis_next_tick_ts:
            return
        self._analysis_next_tick_ts = now + self._LONG_TERM_TICK_INTERVAL_S

        # Run long-term tempo estimation outside the realtime DSP frame path.
        SourceSpecsTask._queue_background_analysis(target)


def _ensure_audio_chunk(data: Any, block_size: int, sample_rate: int = SAMPLE_RATE) -> AudioChunk:
    if isinstance(data, AudioChunk):
        return data

    if data is None:
        mono = np.zeros(block_size, dtype=np.float32)
    else:
        mono = np.asarray(data, dtype=np.float32).reshape(-1)
        if mono.shape[-1] < block_size:
            padded = np.zeros(block_size, dtype=np.float32)
            padded[:mono.shape[-1]] = mono
            mono = padded
        elif mono.shape[-1] > block_size:
            mono = mono[:block_size]

    return AudioChunk(mono, None, sample_rate, lineage=(), latency_ms=0.0)


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


# [LAYER-2-ONLY: DO NOT CALL FROM PREVIEW PATH]
# This function performs iFFT reconstruction and must never be reached from
# PreviewSinkTask.process() or any Layer 3 (time-domain preview) code path.
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
        with utils.logged_errors("audio_tasks_registry._resolve_chunk_fft_size"):
            return max(int(fft_len), 1)
    return max(int(fallback_size), 1)


def _resolve_effective_bin_indices(chunk: AudioChunk, fft_size: int) -> Optional[np.ndarray]:
    """
    Resolve the effective FFT-bin mask represented by a chunk:
    intersection of explicit `band_indices` and `band_hz_bounds` when both exist.
    Returns:
    - None: no mask constraints
    - ndarray (possibly empty): explicit constrained bin set
    """
    idx = None
    raw_idx = getattr(chunk, "band_indices", None)
    if raw_idx is not None:
        try:
            arr = np.asarray(raw_idx, dtype=np.int64).ravel()
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry._resolve_effective_bin_indices", exc)
            arr = np.zeros(0, dtype=np.int64)
        src_fft_size = getattr(chunk, "band_index_fft_size", None)
        if src_fft_size is None:
            src_fft_size = max(int(getattr(chunk, "mono", np.zeros(1, dtype=np.float32)).shape[-1] or 1), 1)
        try:
            src_fft_size = max(int(src_fft_size), 1)
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry._resolve_effective_bin_indices#2", exc)
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
            utils.note_swallowed("audio_tasks_registry._resolve_effective_bin_indices#3", exc)
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


class FrequencySplitterTask(ACEAudioTask):
    def __init__(self, node_id, params, block_size):
        super().__init__(node_id, params, block_size)
        self._bounds_cache_key: Optional[Tuple[Tuple[float, ...], int, float, float]] = None
        self._cached_bounds: Optional[List[Tuple[float, float]]] = None

    @staticmethod
    def _resolve_band_limits(sample_rate: int, parent_bounds: Optional[Tuple[float, float]]) -> Tuple[float, float]:
        nyquist = max(float(sample_rate) * 0.5, 1.0)
        audible_low = 20.0
        audible_high = min(20000.0, nyquist)

        if parent_bounds is None:
            low = audible_low
            high = audible_high
        else:
            try:
                parent_low = float(parent_bounds[0])
                parent_high = float(parent_bounds[1])
            except Exception as exc:
                utils.note_swallowed("audio_tasks_registry.FrequencySplitterTask._resolve_band_limits", exc)
                parent_low, parent_high = audible_low, audible_high
            low = max(audible_low, parent_low)
            high = min(audible_high, parent_high)

        if high <= low:
            high = low + 1e-3
        return low, high

    @classmethod
    def _map_split_values(
        cls,
        split_values: List[float],
        sample_rate: int,
        parent_bounds: Optional[Tuple[float, float]],
    ) -> List[float]:
        if not split_values:
            return []

        low, high = cls._resolve_band_limits(sample_rate, parent_bounds)
        vals = np.asarray(split_values, dtype=np.float64).ravel()
        if vals.size == 0:
            return []

        if parent_bounds is not None:
            if np.all((vals >= 0.0) & (vals <= 1.0)):
                # Explicit normalized mode for nested splitters.
                mapped = low + vals * (high - low)
            else:
                inside_ratio = float(np.mean((vals > low) & (vals < high)))
                if inside_ratio >= 0.6:
                    # Already in parent-band Hz space.
                    mapped = vals
                else:
                    # Remap from global audible space so nested splitters subdivide
                    # their parent band by default (recursive behavior).
                    g_low = 20.0
                    g_high = min(20000.0, max(float(sample_rate) * 0.5, 21.0))
                    denom = max(g_high - g_low, 1e-6)
                    t = (np.clip(vals, g_low, g_high) - g_low) / denom
                    mapped = low + t * (high - low)
        else:
            mapped = vals

        eps = 1e-3
        clipped = np.clip(mapped, low + eps, high - eps)
        clipped = np.unique(clipped)
        if clipped.size == 0:
            return []
        return [float(v) for v in clipped.tolist()]

    def process(self, inputs):
        chunk = _ensure_audio_chunk(inputs.get(0), self.block_size)
        audio = chunk.mono
        stereo = getattr(chunk, "stereo", None)
        sample_rate = getattr(chunk, "sample_rate", SAMPLE_RATE)
        shared_spec_token = getattr(chunk, "spectrum_token", None)
        analysis_ops = _normalized_analysis_ops(chunk)
        band_index_fft_size = getattr(chunk, "band_index_fft_size", None)
        latency_ms = float(getattr(chunk, "latency_ms", 0.0) or 0.0)
        
        split_values = []
        i = 1
        while True:
            val = inputs.get(i)
            if val is None: val = self.parameters.get(f"in_{i}")
            if val is None: break
            split_values.append(float(val))
            i += 1

        parent_bounds = getattr(chunk, "band_hz_bounds", None)
        low_band, high_band = self._resolve_band_limits(sample_rate, parent_bounds)
        split_freqs = self._map_split_values(split_values, sample_rate, parent_bounds)

        if not split_freqs:
            return (
                _spawn_audio_chunk(
                    chunk,
                    mono=audio,
                    stereo=stereo,
                    sample_rate=sample_rate,
                    band_indices=chunk.band_indices,
                    band_index_fft_size=band_index_fft_size,
                    band_hz_bounds=(low_band, high_band),
                    lineage=tuple(getattr(chunk, "lineage", ())) + (f"{self.node_id}:0",),
                    spectrum_token=shared_spec_token,
                    analysis_ops=analysis_ops,
                    latency_ms=latency_ms,
                ),
            )

        split_freqs.sort()
        cache_key = (
            tuple(round(float(v), 6) for v in split_freqs),
            int(sample_rate),
            round(float(low_band), 6),
            round(float(high_band), 6),
        )
        if self._bounds_cache_key != cache_key:
            boundaries = [low_band] + split_freqs + [high_band]
            bounds: List[Tuple[float, float]] = []
            for k in range(len(boundaries) - 1):
                low_cut = float(boundaries[k])
                high_cut = float(boundaries[k + 1])
                bounds.append((low_cut, high_cut))
            self._bounds_cache_key = cache_key
            self._cached_bounds = bounds

        bands = []
        if self._cached_bounds is not None:
            for out_idx, bounds in enumerate(self._cached_bounds):
                low_cut, high_cut = bounds
                if high_cut <= low_cut:
                    high_cut = low_cut + 1e-6
                bands.append(
                    _spawn_audio_chunk(
                        chunk,
                        mono=audio,
                        stereo=stereo,
                        sample_rate=sample_rate,
                        band_indices=chunk.band_indices,
                        band_index_fft_size=band_index_fft_size,
                        band_hz_bounds=(low_cut, high_cut),
                        lineage=tuple(getattr(chunk, "lineage", ())) + (f"{self.node_id}:{out_idx}",),
                        spectrum_token=shared_spec_token,
                        analysis_ops=analysis_ops,
                        latency_ms=latency_ms,
                    )
                )
        else:
            bands.append(chunk)

        return tuple(bands)


class MidsideSplitterTask(ACEAudioTask):
    def process(self, inputs):
        chunk = _ensure_audio_chunk(inputs.get(0), self.block_size)
        sr = getattr(chunk, "sample_rate", SAMPLE_RATE)
        shared_spec_token = getattr(chunk, "spectrum_token", None)
        band_index_fft_size = getattr(chunk, "band_index_fft_size", None)
        latency_ms = float(getattr(chunk, "latency_ms", 0.0) or 0.0)
        mid_ops = _append_analysis_op(chunk, "midside", "mid")
        side_ops = _append_analysis_op(chunk, "midside", "side")
        return (
            _spawn_audio_chunk(
                chunk,
                sample_rate=sr,
                band_index_fft_size=band_index_fft_size,
                band_hz_bounds=getattr(chunk, "band_hz_bounds", None),
                lineage=tuple(getattr(chunk, "lineage", ())) + (f"{self.node_id}:0",),
                spectrum_token=shared_spec_token,
                analysis_ops=mid_ops,
                latency_ms=latency_ms,
            ),
            _spawn_audio_chunk(
                chunk,
                sample_rate=sr,
                band_index_fft_size=band_index_fft_size,
                band_hz_bounds=getattr(chunk, "band_hz_bounds", None),
                lineage=tuple(getattr(chunk, "lineage", ())) + (f"{self.node_id}:1",),
                spectrum_token=shared_spec_token,
                analysis_ops=side_ops,
                latency_ms=latency_ms,
            ),
        )


class AttackReleaseSplitterTask(ACEAudioTask):
    def process(self, inputs):
        chunk = _ensure_audio_chunk(inputs.get(0), self.block_size)
        shared_spec_token = getattr(chunk, "spectrum_token", None)
        band_index_fft_size = getattr(chunk, "band_index_fft_size", None)
        latency_ms = float(getattr(chunk, "latency_ms", 0.0) or 0.0)
        attack_ops = _append_analysis_op(chunk, "attackrelease", "attack")
        release_ops = _append_analysis_op(chunk, "attackrelease", "release")

        return (
            _spawn_audio_chunk(
                chunk,
                sample_rate=chunk.sample_rate,
                band_index_fft_size=band_index_fft_size,
                band_hz_bounds=getattr(chunk, "band_hz_bounds", None),
                lineage=tuple(getattr(chunk, "lineage", ())) + (f"{self.node_id}:0",),
                spectrum_token=shared_spec_token,
                analysis_ops=attack_ops,
                latency_ms=latency_ms,
            ),
            _spawn_audio_chunk(
                chunk,
                sample_rate=chunk.sample_rate,
                band_index_fft_size=band_index_fft_size,
                band_hz_bounds=getattr(chunk, "band_hz_bounds", None),
                lineage=tuple(getattr(chunk, "lineage", ())) + (f"{self.node_id}:1",),
                spectrum_token=shared_spec_token,
                analysis_ops=release_ops,
                latency_ms=latency_ms,
            ),
        )


class ChromagramTask(ACEAudioTask):
    def __init__(self, node_id, params, block_size):
        super().__init__(node_id, params, block_size)
        self._mask_cache_key: Optional[Tuple[int, int]] = None
        self._mask_cache: Optional[np.ndarray] = None

    def _build_masks(self, fft_size: int, sample_rate: int):
        freq_bins = np.fft.rfftfreq(max(int(fft_size), 1), d=1.0 / sample_rate)
        masks = np.zeros((12, len(freq_bins)), dtype=bool)
        for idx, f in enumerate(freq_bins):
            if f <= 0.0:
                continue
            midi = int(round(69 + 12 * math.log2(f / 440.0)))
            chroma = midi % 12
            masks[chroma, idx] = True
        self._mask_cache_key = (max(int(fft_size), 1), sample_rate)
        self._mask_cache = masks

    @staticmethod
    def _fallback_indices_for_chroma(
        chroma_idx: int,
        domain_indices: np.ndarray,
        fft_size: int,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Coarse FFT grids can miss a chroma class entirely in narrow parent bands.
        When that happens, pick the closest bins in the available domain so the
        branch remains usable instead of collapsing to silence.
        """
        dom = np.asarray(domain_indices, dtype=np.int64).ravel()
        if dom.size == 0:
            return np.zeros(0, dtype=np.int64)

        freqs = np.fft.rfftfreq(max(int(fft_size), 1), d=1.0 / max(float(sample_rate), 1.0))
        dom = dom[(dom >= 0) & (dom < freqs.shape[0])]
        if dom.size == 0:
            return np.zeros(0, dtype=np.int64)

        dom_freqs = freqs[dom]
        valid = dom_freqs > 0.0
        if not np.any(valid):
            return np.zeros(0, dtype=np.int64)
        dom = dom[valid]
        dom_freqs = dom_freqs[valid]

        midi = 69.0 + (12.0 * np.log2(dom_freqs / 440.0))
        chroma = np.mod(np.round(midi).astype(np.int64), 12)
        # Circular semitone distance in chroma space.
        delta = np.abs(((chroma - int(chroma_idx) + 6) % 12) - 6)
        min_delta = int(np.min(delta))
        nearest = dom[delta == min_delta]
        if nearest.size == 0:
            return np.zeros(0, dtype=np.int64)
        return np.unique(nearest.astype(np.int64, copy=False))

    def process(self, inputs):
        chunk = _ensure_audio_chunk(inputs.get(0), self.block_size)
        sr = getattr(chunk, "sample_rate", SAMPLE_RATE)
        shared_spec_token = getattr(chunk, "spectrum_token", None)
        analysis_ops = _normalized_analysis_ops(chunk)
        base_latency_ms = float(getattr(chunk, "latency_ms", 0.0) or 0.0)
        fft_size = _resolve_chunk_fft_size(
            chunk,
            fallback_size=max(int(getattr(chunk, "mono", np.zeros(1, dtype=np.float32)).shape[-1] or 0), 1),
        )
        # Chroma projection is spectral-domain constrained; account for an
        # approximate half-window analysis lag in branch latency tracking.
        latency_ms = base_latency_ms + (0.5 * _block_duration_ms(sr, fft_size))
        upstream_idx = _resolve_effective_bin_indices(chunk, fft_size)
        if self._mask_cache_key != (fft_size, sr) or self._mask_cache is None:
            self._build_masks(fft_size, sr)

        outputs = []
        for chroma_idx in range(12):
            mask = self._mask_cache[chroma_idx] if self._mask_cache is not None else None
            if mask is None:
                band_indices = None
            else:
                chroma_indices = np.nonzero(mask)[0].astype(np.int64, copy=False)
                if upstream_idx is None:
                    band_indices = chroma_indices
                else:
                    band_indices = np.intersect1d(chroma_indices, upstream_idx, assume_unique=False).astype(np.int64, copy=False)
                if band_indices.size == 0:
                    if upstream_idx is None:
                        domain = np.arange(mask.shape[0], dtype=np.int64)
                    else:
                        domain = np.asarray(upstream_idx, dtype=np.int64).ravel()
                    fallback = self._fallback_indices_for_chroma(chroma_idx, domain, fft_size, sr)
                    if fallback.size > 0:
                        band_indices = fallback
                    else:
                        # Keep bounds-based processing alive downstream instead of
                        # forcing an explicit empty sparse mask.
                        band_indices = None

            outputs.append(
                _spawn_audio_chunk(
                    chunk,
                    sample_rate=sr,
                    band_indices=band_indices,
                    band_index_fft_size=fft_size,
                    band_hz_bounds=getattr(chunk, "band_hz_bounds", None),
                    lineage=tuple(getattr(chunk, "lineage", ())) + (f"{self.node_id}:{chroma_idx}",),
                    spectrum_token=shared_spec_token,
                    analysis_ops=analysis_ops,
                    latency_ms=latency_ms,
                )
            )

        return tuple(outputs)


class AutoComponentSplitterTask(ACEAudioTask):
    def __init__(self, node_id, params, block_size):
        super().__init__(node_id, params, block_size)
        self._prev_mag: Optional[np.ndarray] = None
        self._last_sr: Optional[int] = None

    def process(self, inputs):
        chunk = _ensure_audio_chunk(inputs.get(0), self.block_size)
        sr = getattr(chunk, "sample_rate", SAMPLE_RATE)
        shared_spec_token = getattr(chunk, "spectrum_token", None)
        analysis_ops = _normalized_analysis_ops(chunk)
        base_latency_ms = float(getattr(chunk, "latency_ms", 0.0) or 0.0)
        fft_size = _resolve_chunk_fft_size(
            chunk,
            fallback_size=max(int(getattr(chunk, "mono", np.zeros(1, dtype=np.float32)).shape[-1] or 0), 1),
        )
        # Auto-component separation uses inter-frame spectral deltas and keeps
        # one extra frame of historical context.
        latency_ms = base_latency_ms + _block_duration_ms(sr, fft_size)
        upstream_idx = _resolve_effective_bin_indices(chunk, fft_size)

        # Build a simple per-bin transient mask instead of performing HPSS:
        # - bins with higher flux than the median+std are tagged percussive
        # - quieter bins go to residual
        # - everything else is considered harmonic
        analysis_mono, _ = _analysis_signal_for_chunk(chunk)
        mag = np.abs(np.fft.rfft(analysis_mono, n=fft_size))
        if mag.size == 0:
            return (chunk, chunk, chunk)

        if self._prev_mag is None or self._prev_mag.shape != mag.shape or self._last_sr != sr:
            self._prev_mag = mag.copy()
            self._last_sr = sr

        diff = np.abs(mag - self._prev_mag)
        self._prev_mag = mag.copy()

        if upstream_idx is None:
            work_idx = np.arange(mag.shape[0], dtype=np.int64)
        else:
            work_idx = upstream_idx.astype(np.int64, copy=False)

        if work_idx.size == 0:
            perc_idx = np.zeros(0, dtype=np.int64)
            harm_idx = np.zeros(0, dtype=np.int64)
            res_idx = np.zeros(0, dtype=np.int64)
        else:
            diff_work = diff[work_idx]
            mag_work = mag[work_idx]
            flux_thresh = np.median(diff_work) + 0.5 * np.std(diff_work)
            energy_thresh = max(np.median(mag_work) * 0.25, 1e-8)

            perc_mask = diff_work > flux_thresh
            harm_mask = diff_work <= flux_thresh
            res_mask = mag_work <= energy_thresh

            perc_idx = work_idx[perc_mask]
            harm_idx = work_idx[harm_mask]
            res_idx = work_idx[res_mask]

            if perc_idx.size == 0:
                perc_idx = np.array([int(work_idx[int(np.argmax(diff_work))])], dtype=np.int64)
            if harm_idx.size == 0:
                harm_idx = work_idx[mag_work > energy_thresh]
            if res_idx.size == 0:
                res_idx = np.array([], dtype=np.int64)

        def _masked(indices: np.ndarray, out_idx: int):
            valid_idx = np.asarray(indices, dtype=np.int64).ravel() if indices is not None else None
            return _spawn_audio_chunk(
                chunk,
                sample_rate=sr,
                band_indices=valid_idx,
                band_index_fft_size=fft_size,
                band_hz_bounds=getattr(chunk, "band_hz_bounds", None),
                lineage=tuple(getattr(chunk, "lineage", ())) + (f"{self.node_id}:{out_idx}",),
                spectrum_token=shared_spec_token,
                analysis_ops=analysis_ops,
                latency_ms=latency_ms,
            )

        return (_masked(harm_idx, 0), _masked(perc_idx, 1), _masked(res_idx, 2))


class RMSTask(ACEAudioTask):
    """Compute Root-Mean-Square energy of the audio chunk.

    Output range: [0.0, 1.0] (float).
    Uses a fast Parseval-theorem frequency-domain approach when a cached
    spectrum is available; falls back to a time-domain RMS on silence blocks.
    """

    def process(self, inputs):
        chunk = _ensure_audio_chunk(inputs.get(0), self.block_size)
        if not hasattr(chunk, "mono") or chunk.mono.size == 0:
            return 0.0
        analysis_mono, _ = _analysis_signals_normalized(chunk)
        if analysis_mono.size == 0:
            return 0.0

        # Fast path for unmasked streams: pure time-domain RMS.
        mask_indices = getattr(chunk, "band_indices", None)
        band_bounds = getattr(chunk, "band_hz_bounds", None)
        if mask_indices is None and band_bounds is None:
            return _safe_rms(analysis_mono)

        # Masked RMS should reuse the source analyzer's shared spectrum state.
        if isinstance(chunk, AudioChunk) and chunk.provider:
            val = chunk.get_feature("rms", lambda: chunk.provider.compute_feature("rms", chunk))
            if val is not None:
                return float(val)

        # Masked RMS: estimate energy from selected FFT bins (Parseval).
        mono_fft, _, _ = _get_rfft(chunk, self.block_size)
        idx = _resolve_effective_bin_indices(chunk, self.block_size)
        if idx is None or idx.size == 0:
            return 0.0

        n = max(int(self.block_size), 1)
        power = np.abs(mono_fft[idx]) ** 2
        weights = np.full(power.shape, 2.0, dtype=np.float64)
        weights[idx == 0] = 1.0
        if (n % 2) == 0:
            weights[idx == (n // 2)] = 1.0
        mean_square = float(np.sum(power * weights) / (n * n))
        return float(np.sqrt(max(mean_square, 0.0)))


class LUFSTask(ACEAudioTask):
    """Estimate integrated short-window loudness using K-weighting (LUFS).

    Applies the ITU-R BS.1770-style K-weighting filter chain:
      1. +4 dB high-shelf pre-filter at 1500 Hz (Q = 1/sqrt(2))
      2. RLB high-pass at 38 Hz (Q = 0.5)

    The loudness output is recomputed every block from the mean-square of the
    last ``Window (s)`` seconds of weighted signal:
      ``LUFS = -0.691 + 10 * log10(mean_square)``

    Output (normalized):
      - socket 0: loudness in [0, 1], where 0 = silence (-inf LUFS), 1 = 0 LUFS
    """

    _DEFAULT_WINDOW_S = 0.4
    _MIN_WINDOW_S = 0.05
    _MAX_WINDOW_S = 3.0

    def __init__(self, node_id, params, block_size):
        super().__init__(node_id, params, block_size)
        self._sample_rate: int = 0
        self._prefilter_coeffs: Tuple[float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0, 0.0)
        self._rlb_coeffs: Tuple[float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0, 0.0)
        self._prefilter_state: List[float] = [0.0, 0.0]
        self._rlb_state: List[float] = [0.0, 0.0]
        self._weighted_energy: np.ndarray = np.zeros(0, dtype=np.float64)
        self._last_lufs: float = 0.0
        self._last_frame_request: Optional[int] = None

    def _reset_temporal_state(self):
        self._weighted_energy = np.zeros(0, dtype=np.float64)
        self._last_lufs = 0.0

    def _reset_filter_state(self):
        self._prefilter_state[0] = 0.0
        self._prefilter_state[1] = 0.0
        self._rlb_state[0] = 0.0
        self._rlb_state[1] = 0.0

    @staticmethod
    def _coerce_scalar(value: Any, fallback: float) -> float:
        if isinstance(value, AudioChunk):
            return float(fallback)
        if isinstance(value, np.ndarray):
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            return float(arr[0]) if arr.size > 0 else float(fallback)
        if isinstance(value, (list, tuple)):
            if not value:
                return float(fallback)
            try:
                return float(value[0])
            except Exception as exc:
                utils.note_swallowed("audio_tasks_registry.LUFSTask._coerce_scalar", exc)
                return float(fallback)
        try:
            return float(value)
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.LUFSTask._coerce_scalar#2", exc)
            return float(fallback)

    def _resolve_seconds(
        self,
        inputs: Dict[int, Any],
        input_idx: int,
        param_key: str,
        fallback: float,
        min_value: float,
        max_value: float,
    ) -> float:
        raw = inputs.get(int(input_idx))
        if raw is None:
            raw = self.parameters.get(f"in_{int(input_idx)}", self.parameters.get(param_key, fallback))
        val = self._coerce_scalar(raw, fallback)
        if not np.isfinite(val):
            val = float(fallback)
        return float(np.clip(val, float(min_value), float(max_value)))

    @staticmethod
    def _design_high_shelf(sample_rate: int, freq_hz: float = 1500.0, gain_db: float = 4.0, q: float = 0.70710678118):
        sr = max(float(sample_rate), 1.0)
        nyquist = 0.5 * sr
        f0 = float(np.clip(freq_hz, 10.0, max(nyquist * 0.95, 10.0)))
        q_safe = max(float(q), 1e-6)
        A = 10.0 ** (float(gain_db) / 40.0)
        w0 = 2.0 * math.pi * f0 / sr
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * q_safe)
        sqrt_A = math.sqrt(A)

        b0 = A * ((A + 1.0) + ((A - 1.0) * cos_w0) + (2.0 * sqrt_A * alpha))
        b1 = -2.0 * A * ((A - 1.0) + ((A + 1.0) * cos_w0))
        b2 = A * ((A + 1.0) + ((A - 1.0) * cos_w0) - (2.0 * sqrt_A * alpha))
        a0 = (A + 1.0) - ((A - 1.0) * cos_w0) + (2.0 * sqrt_A * alpha)
        a1 = 2.0 * ((A - 1.0) - ((A + 1.0) * cos_w0))
        a2 = (A + 1.0) - ((A - 1.0) * cos_w0) - (2.0 * sqrt_A * alpha)
        if abs(a0) <= 1e-12:
            return (1.0, 0.0, 0.0, 0.0, 0.0)
        return (
            float(b0 / a0),
            float(b1 / a0),
            float(b2 / a0),
            float(a1 / a0),
            float(a2 / a0),
        )

    @staticmethod
    def _design_high_pass(sample_rate: int, freq_hz: float = 38.0, q: float = 0.5):
        sr = max(float(sample_rate), 1.0)
        nyquist = 0.5 * sr
        f0 = float(np.clip(freq_hz, 5.0, max(nyquist * 0.95, 5.0)))
        q_safe = max(float(q), 1e-6)
        w0 = 2.0 * math.pi * f0 / sr
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * q_safe)

        b0 = 0.5 * (1.0 + cos_w0)
        b1 = -(1.0 + cos_w0)
        b2 = 0.5 * (1.0 + cos_w0)
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
        if abs(a0) <= 1e-12:
            return (1.0, 0.0, 0.0, 0.0, 0.0)
        return (
            float(b0 / a0),
            float(b1 / a0),
            float(b2 / a0),
            float(a1 / a0),
            float(a2 / a0),
        )

    @staticmethod
    def _apply_biquad(
        signal: np.ndarray,
        coeffs: Tuple[float, float, float, float, float],
        state: List[float],
    ) -> np.ndarray:
        src = np.asarray(signal, dtype=np.float32).reshape(-1)
        if src.size == 0:
            return src
        b0, b1, b2, a1, a2 = coeffs
        z1 = float(state[0]) if len(state) > 0 else 0.0
        z2 = float(state[1]) if len(state) > 1 else 0.0

        if NUMBA_AVAILABLE:
            out, z1_new, z2_new = _biquad_filter_block(src, b0, b1, b2, a1, a2, z1, z2)
            state[0] = float(z1_new)
            state[1] = float(z2_new)
            return out.astype(np.float32, copy=False)

        if SCIPY_AVAILABLE and callable(scipy_lfilter):
            with utils.logged_errors("audio_tasks_registry.LUFSTask._apply_biquad"):
                out_64, zf = scipy_lfilter(
                    np.array([b0, b1, b2], dtype=np.float64),
                    np.array([1.0, a1, a2], dtype=np.float64),
                    src.astype(np.float64, copy=False),
                    zi=np.array([z1, z2], dtype=np.float64),
                )
                state[0] = float(zf[0])
                state[1] = float(zf[1])
                return out_64.astype(np.float32, copy=False)

        out = np.empty_like(src, dtype=np.float32)
        s1 = z1
        s2 = z2
        for i in range(src.shape[0]):
            x = float(src[i])
            y = (b0 * x) + s1
            s1 = (b1 * x) - (a1 * y) + s2
            s2 = (b2 * x) - (a2 * y)
            out[i] = y
        state[0] = float(s1)
        state[1] = float(s2)
        return out

    def _ensure_filter_coeffs(self, sample_rate: int):
        sr = max(int(sample_rate), 1)
        if int(self._sample_rate) == sr:
            return
        self._sample_rate = sr
        self._prefilter_coeffs = self._design_high_shelf(sr)
        self._rlb_coeffs = self._design_high_pass(sr)
        self._reset_filter_state()
        self._weighted_energy = np.zeros(0, dtype=np.float64)

    @staticmethod
    def _db_to_unit(db_value: float) -> float:
        """Map a dB-like value to [0, 1] with 0 dB -> 1 and -inf dB -> 0."""
        try:
            db = float(db_value)
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.LUFSTask._db_to_unit", exc)
            return 0.0
        if not np.isfinite(db):
            return 0.0
        if db <= -120.0:
            return 0.0
        if db >= 0.0:
            return 1.0
        return float(np.clip(10.0 ** (db / 20.0), 0.0, 1.0))

    def process(self, inputs):
        chunk = _ensure_audio_chunk(inputs.get(0), self.block_size)
        frame_idx = utils.as_int(getattr(self, "_current_frame_request", None), -1)
        last_frame = self._last_frame_request
        if last_frame is not None and frame_idx >= 0 and frame_idx < int(last_frame):
            self._reset_temporal_state()
            self._reset_filter_state()

        sample_rate = max(int(getattr(chunk, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE), 1)
        self._ensure_filter_coeffs(sample_rate)

        window_s = self._resolve_seconds(
            inputs,
            input_idx=1,
            param_key="window_size",
            fallback=self._DEFAULT_WINDOW_S,
            min_value=self._MIN_WINDOW_S,
            max_value=self._MAX_WINDOW_S,
        )
        window_samples = max(int(round(window_s * float(sample_rate))), 1)

        analysis_mono = _masked_mono_signal(chunk)
        if analysis_mono.size <= 0:
            self._last_lufs = 0.0
            if frame_idx >= 0:
                self._last_frame_request = frame_idx
            return float(self._last_lufs)

        weighted = self._apply_biquad(analysis_mono, self._prefilter_coeffs, self._prefilter_state)
        weighted = self._apply_biquad(weighted, self._rlb_coeffs, self._rlb_state)
        energy = np.square(weighted.astype(np.float64, copy=False))

        if self._weighted_energy.size <= 0:
            self._weighted_energy = energy[-window_samples:]
        else:
            self._weighted_energy = np.concatenate((self._weighted_energy, energy), axis=0)
            if self._weighted_energy.size > window_samples:
                self._weighted_energy = self._weighted_energy[-window_samples:]

        mean_square = float(np.mean(self._weighted_energy, dtype=np.float64)) if self._weighted_energy.size > 0 else 0.0
        if (not np.isfinite(mean_square)) or mean_square <= 1e-12:
            lufs = float("-inf")
        else:
            lufs = -0.691 + (10.0 * math.log10(mean_square))
            if not np.isfinite(lufs):
                lufs = float("-inf")
        self._last_lufs = self._db_to_unit(lufs)

        if frame_idx >= 0:
            self._last_frame_request = frame_idx
        return float(self._last_lufs)


class PeakTask(LUFSTask):
    """Estimate K-weighted peak level.

    Output (normalized):
      - socket 0: weighted peak in [0, 1], where 0 = -inf dB, 1 = 0 dB
    """

    def __init__(self, node_id, params, block_size):
        super().__init__(node_id, params, block_size)
        self._last_peak: float = 0.0

    def _reset_temporal_state(self):
        self._last_peak = 0.0

    def process(self, inputs):
        chunk = _ensure_audio_chunk(inputs.get(0), self.block_size)
        frame_idx = utils.as_int(getattr(self, "_current_frame_request", None), -1)
        last_frame = self._last_frame_request
        if last_frame is not None and frame_idx >= 0 and frame_idx < int(last_frame):
            self._reset_temporal_state()
            self._reset_filter_state()

        sample_rate = max(int(getattr(chunk, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE), 1)
        self._ensure_filter_coeffs(sample_rate)

        analysis_mono = _masked_mono_signal(chunk)
        if analysis_mono.size <= 0:
            self._last_peak = 0.0
            if frame_idx >= 0:
                self._last_frame_request = frame_idx
            return float(self._last_peak)

        weighted = self._apply_biquad(analysis_mono, self._prefilter_coeffs, self._prefilter_state)
        weighted = self._apply_biquad(weighted, self._rlb_coeffs, self._rlb_state)
        peak = float(np.max(np.abs(weighted))) if weighted.size > 0 else 0.0
        if peak <= 1e-12 or (not np.isfinite(peak)):
            peak_db = float("-inf")
        else:
            peak_db = float(20.0 * math.log10(max(float(peak), 1e-12)))
        self._last_peak = self._db_to_unit(peak_db)

        if frame_idx >= 0:
            self._last_frame_request = frame_idx
        return float(self._last_peak)


class PitchTask(ACEAudioTask):
    """Estimate the fundamental frequency (pitch) of the audio chunk.

    Output: fundamental frequency in Hz (float).  Returns 0.0 when the signal
    is silent or no reliable pitch is detectable.  Prefers the audioflux
    pitch estimator when available; falls back to a time-domain autocorrelation
    estimate.
    """

    def process(self, inputs):
        chunk = _ensure_audio_chunk(inputs.get(0), self.block_size)
        val = None
        if isinstance(chunk, AudioChunk) and chunk.provider:
            val = chunk.get_feature("pitch", lambda: chunk.provider.compute_feature("pitch", chunk))
        if val is not None:
            with utils.logged_errors("audio_tasks_registry.PitchTask.process"):
                resolved = float(val)
                if np.isfinite(resolved):
                    return resolved

        analysis_mono = _masked_mono_signal(chunk)
        return _autocorr_pitch_estimate(analysis_mono, int(getattr(chunk, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE))


class SpectralCentroidTask(ACEAudioTask):
    """Compute the spectral centroid (brightness) of the audio chunk.

    Output: centroid frequency in Hz (float).  The centroid is the
    amplitude-weighted mean frequency of the power spectrum — perceptually
    related to perceived brightness.  Prefers the audioflux centroid estimate;
    falls back to a direct FFT magnitude centroid calculation.
    """

    def process(self, inputs):
        chunk = _ensure_audio_chunk(inputs.get(0), self.block_size)
        val = None
        if isinstance(chunk, AudioChunk) and chunk.provider:
            val = chunk.get_feature("centroid", lambda: chunk.provider.compute_feature("centroid", chunk))
        if val is not None:
            with utils.logged_errors("audio_tasks_registry.SpectralCentroidTask.process"):
                resolved = float(val)
                if np.isfinite(resolved):
                    return resolved

        mag, freqs = _masked_spectrum(chunk)
        denom = float(np.sum(mag, dtype=np.float64))
        if denom <= 1e-9:
            return 0.0
        centroid = float(np.sum(mag.astype(np.float64) * freqs.astype(np.float64), dtype=np.float64) / denom)
        return float(centroid if np.isfinite(centroid) else 0.0)


class SpectralFluxTask(ACEAudioTask):
    """Measure frame-to-frame change in the power spectrum (spectral flux).

    Output range: [0.0, 10.0] (float, clipped).  A high value indicates rapid
    timbral change — useful for transient detection, beat tracking, and gate
    triggers.  Computed as the normalised half-wave-rectified spectral
    difference relative to the previous frame's magnitude spectrum.
    """

    def __init__(self, node_id, params, block_size):
        super().__init__(node_id, params, block_size)
        self._prev_mag: Optional[np.ndarray] = None
        self._prev_sr: int = 0

    def _fallback_flux(self, chunk: AudioChunk) -> float:
        mag, _ = _masked_spectrum(chunk)
        sr = int(getattr(chunk, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
        if mag.size <= 0:
            self._prev_mag = None
            self._prev_sr = sr
            return 0.0
        if (
            self._prev_mag is None
            or self._prev_mag.shape != mag.shape
            or int(self._prev_sr) != int(sr)
        ):
            self._prev_mag = mag.copy()
            self._prev_sr = int(sr)
            return 0.0
        prev = self._prev_mag.astype(np.float64, copy=False)
        curr = mag.astype(np.float64, copy=False)
        diff = curr - prev
        flux = float(np.sum(np.maximum(diff, 0.0), dtype=np.float64))
        denom = float(np.sum(prev, dtype=np.float64)) + 1e-9
        self._prev_mag = mag.copy()
        self._prev_sr = int(sr)
        val = flux / denom
        if not np.isfinite(val):
            return 0.0
        return float(np.clip(val, 0.0, 10.0))

    def process(self, inputs):
        chunk = _ensure_audio_chunk(inputs.get(0), self.block_size)
        val = None
        if isinstance(chunk, AudioChunk) and chunk.provider:
            val = chunk.get_feature("flux", lambda: chunk.provider.compute_feature("flux", chunk))
        if val is not None:
            with utils.logged_errors("audio_tasks_registry.SpectralFluxTask.process"):
                resolved = float(val)
                if np.isfinite(resolved):
                    return resolved

        return self._fallback_flux(chunk)


class OnsetTask(ACEAudioTask):
    """Detect audio onsets (transient events) in the audio chunk.

    Output range: [0.0, 1.0] (float).  Values near 1.0 indicate a strong onset
    (note attack, drum hit, etc.) within the current frame.  Uses an adaptive
    spectral flux envelope detector: the raw flux is smoothed with an
    attack/release envelope and thresholded so intermittent transients produce
    clean pulses rather than sustained high values.
    """

    def __init__(self, node_id, params, block_size):
        super().__init__(node_id, params, block_size)
        self._prev_mag: Optional[np.ndarray] = None
        self._prev_sr: int = 0
        self._flux_env: float = 0.0

    def _fallback_onset(self, chunk: AudioChunk) -> float:
        mag, _ = _masked_spectrum(chunk)
        sr = int(getattr(chunk, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
        if mag.size <= 0:
            self._prev_mag = None
            self._prev_sr = sr
            self._flux_env = 0.0
            return 0.0

        if (
            self._prev_mag is None
            or self._prev_mag.shape != mag.shape
            or int(self._prev_sr) != int(sr)
        ):
            self._prev_mag = mag.copy()
            self._prev_sr = int(sr)
            self._flux_env = 0.0
            return 0.0

        prev = self._prev_mag.astype(np.float64, copy=False)
        curr = mag.astype(np.float64, copy=False)
        flux = float(np.sum(np.maximum(curr - prev, 0.0), dtype=np.float64))
        self._prev_mag = mag.copy()
        self._prev_sr = int(sr)

        self._flux_env = (0.92 * float(self._flux_env)) + (0.08 * flux)
        env = max(float(self._flux_env), 1e-6)
        onset = (flux - (0.6 * env)) / (1.4 * env)
        if not np.isfinite(onset):
            return 0.0
        return float(np.clip(onset, 0.0, 1.0))

    def process(self, inputs):
        chunk = _ensure_audio_chunk(inputs.get(0), self.block_size)
        val = None
        if isinstance(chunk, AudioChunk) and chunk.provider:
            val = chunk.get_feature("onset", lambda: chunk.provider.compute_feature("onset", chunk))
        if val is not None:
            with utils.logged_errors("audio_tasks_registry.OnsetTask.process"):
                resolved = float(val)
                if np.isfinite(resolved):
                    return resolved

        return self._fallback_onset(chunk)


class ZeroCrossTask(ACEAudioTask):
    """Measure the Zero-Crossing Rate (ZCR) of the audio chunk.

    Output range: [0.0, 1.0] (float, normalised to Nyquist).  High ZCR values
    indicate noisy or high-frequency content; low values correlate with tonal
    or bass-heavy material.  Prefers the audioflux ZCR implementation; falls
    back to a direct sample-sign-change count.
    """

    def process(self, inputs):
        chunk = _ensure_audio_chunk(inputs.get(0), self.block_size)
        val = None
        if isinstance(chunk, AudioChunk) and chunk.provider:
            val = chunk.get_feature("zcr", lambda: chunk.provider.compute_feature("zcr", chunk))
        if val is not None:
            with utils.logged_errors("audio_tasks_registry.ZeroCrossTask.process"):
                resolved = float(val)
                if np.isfinite(resolved):
                    return resolved
        analysis_mono, _ = _analysis_signals_normalized(chunk)
        return _zero_crossing_rate(analysis_mono)


class StereoWidthTask(ACEAudioTask):
    """Measure the stereo width of the audio chunk.

    Output range: [0.0, 1.0] (float).
    0.0 = perfectly mono (side energy is zero).
    1.0 = maximally wide (all energy in the side channel, no mid).
    Computed as ``side_energy / (mid_energy + side_energy)`` to give a
    naturally bounded result without the overrun of a raw side/mid ratio.
    """

    def process(self, inputs):
        chunk = _ensure_audio_chunk(inputs.get(0), self.block_size)
        _, stereo = _analysis_signals_normalized(chunk)
        if stereo is None or stereo.shape[0] < 2:
            return 0.0
        left = stereo[0]
        right = stereo[1]
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)
        mid_energy = float(np.mean(mid * mid))
        side_energy = float(np.mean(side * side))
        total_energy = mid_energy + side_energy
        # Normalize to [0, 1]: 0 = perfectly mono, 1 = pure side (maximum stereo width).
        # side / (mid + side) is always in [0, 1] by construction and matches the
        # node description "0 = mono, 1 = very wide" without the [0, 10] overrun.
        if total_energy < 1e-20:
            return 0.0
        return float(np.clip(side_energy / total_energy, 0.0, 1.0))


class StereoBalanceTask(ACEAudioTask):
    """Measure the left/right energy balance of the audio chunk.

    Output range: [-1.0, 1.0] (float).
    -1.0 = all energy on the left channel.
    0.0 = equal energy on both channels (centre-panned).
    +1.0 = all energy on the right channel.
    """

    def process(self, inputs):
        chunk = _ensure_audio_chunk(inputs.get(0), self.block_size)
        _, stereo = _analysis_signals_normalized(chunk)
        if stereo is None or stereo.shape[0] < 2:
            return 0.0
        left = stereo[0]
        right = stereo[1]
        l_rms = _safe_rms(left)
        r_rms = _safe_rms(right)
        denom = l_rms + r_rms + 1e-9
        return float(np.clip((r_rms - l_rms) / denom, -1.0, 1.0))


class StereoVectorTask(ACEAudioTask):
    """Time-domain vector summary of stereo energy and balance.

    Output: 3-component FLOAT_VECTOR ``(left_rms, right_rms, balance)``.
    ``left_rms`` and ``right_rms`` are in [0.0, 1.0].
    ``balance`` mirrors ``StereoBalanceTask``: -1.0 = full left, +1.0 = full right.
    Falls back to a mono-derived vector (equal L/R, centre balance) when the
    input has no stereo data.
    """
    def process(self, inputs):
        chunk = _ensure_audio_chunk(inputs.get(0), self.block_size)
        analysis_mono, stereo = _analysis_signals_normalized(chunk)
        if stereo is not None and stereo.shape[0] >= 2:
            left = stereo[0]
            right = stereo[1]
        else:
            left = analysis_mono
            right = analysis_mono

        l_rms = _safe_rms(left)
        r_rms = _safe_rms(right)
        denom = l_rms + r_rms + 1e-9
        balance = float(np.clip((r_rms - l_rms) / denom, -1.0, 1.0))
        return (l_rms, r_rms, balance)


def _run_biquad(
    signal: np.ndarray,
    coeffs: Tuple[float, float, float, float, float],
    state: List[float],
    use_numba: bool = True,
) -> np.ndarray:
    """Apply a single Direct-Form-II biquad section to *signal* in-place on the state.

    Three-tier implementation:

    * **Tier 1** — Numba JIT kernel: zero Python overhead, preferred when available.
    * **Tier 2** — ``scipy.signal.lfilter``: vectorized C, ~100× faster than the
      Python loop.  Used when Numba is absent.  Reports a WARNING on unexpected
      failure so the user is not left debugging silent audio dropouts.
    * **Tier 3** — Pure-Python sample loop: last resort.  Emits a one-time WARNING
      because at 512 samples/block this path takes ~10 ms on the preview thread and
      will cause audible dropouts.

    Args:
        signal:    1-D float32 input samples (length N).
        coeffs:    Normalised biquad coefficients (b0, b1, b2, a1, a2) — a0 already
                   divided out.
        state:     Two-element list [z1, z2] carrying the filter state between calls.
                   Updated in-place.
        use_numba: Allow Tier-1 (Numba JIT).  Pass ``False`` to force Tier-2/3.

    Returns:
        Filtered float32 array of the same length as *signal*.
    """
    global _BIQUAD_PYTHON_FALLBACK_WARNED
    b0, b1, b2, a1, a2 = coeffs
    z1, z2 = float(state[0]), float(state[1])
    src = signal.astype(np.float32, copy=False)

    # ── Tier 1: Numba JIT ────────────────────────────────────────────────────
    if use_numba and NUMBA_AVAILABLE:
        try:
            out, z1_new, z2_new = _biquad_filter_block(
                src,
                float(b0), float(b1), float(b2),
                float(a1), float(a2),
                float(z1), float(z2),
            )
            state[0] = float(z1_new)
            state[1] = float(z2_new)
            return out
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry._run_biquad", exc)
            pass  # Fall through to Tier 2.

    # ── Tier 2: scipy.signal.lfilter ─────────────────────────────────────────
    # Converts Direct Form I coefficients to the [b0,b1,b2] / [1,a1,a2] vectors
    # scipy expects, and passes the DF-II state as initial conditions.
    if SCIPY_AVAILABLE:
        try:
            b_vec = np.array([b0, b1, b2], dtype=np.float64)
            a_vec = np.array([1.0, a1, a2], dtype=np.float64)
            zi = np.array([z1, z2], dtype=np.float64)  # zi shape must be (2,)
            out_64, zf = scipy_lfilter(b_vec, a_vec, src.astype(np.float64), zi=zi)
            state[0] = float(zf[0])
            state[1] = float(zf[1])
            return out_64.astype(np.float32, copy=False)
        except Exception as _exc:
            utils.log(
                f"[ACE-Preview] scipy biquad Tier-2 failed (falling to Python loop): {_exc!r}",
                "WARNING",
            )

    # ── Tier 3: Pure-Python sample loop ──────────────────────────────────────
    # Last resort.  At 512 samples/block this path takes ~10 ms on the preview
    # thread and will cause audible dropouts.  Emit a one-time WARNING.
    if not _BIQUAD_PYTHON_FALLBACK_WARNED:
        _BIQUAD_PYTHON_FALLBACK_WARNED = True
        utils.log(
            "ACE Layer-3 preview: Numba and scipy.signal are both unavailable. "
            "Using a slow Python biquad fallback that may cause audio dropouts. "
            "Install Numba or scipy to restore full preview performance.",
            "WARNING",
        )

    out = np.empty_like(src, dtype=np.float32)
    for i in range(src.shape[0]):
        x = float(src[i])
        y = (b0 * x) + z1
        z1 = (b1 * x) - (a1 * y) + z2
        z2 = (b2 * x) - (a2 * y)
        out[i] = y
    state[0] = z1
    state[1] = z2
    return out


class PreviewSinkTask(ACEAudioTask):
    """Terminal monitor sink for branch-audio preview.

    Implements strict Layer-3 time-domain filtering: no iFFT/iBFT is performed.
    Frequency-band selection is translated to equivalent time-domain filter chains:

    * **Contiguous bounds** (``band_hz_bounds``): cascaded 2×2nd-order Butterworth
      highpass + lowpass sections (4th-order equivalent slope).
    * **Sparse/non-contiguous masks** (``band_indices``): the mask is converted to
      contiguous Hz ranges which are individually band-limited and summed.
    * **No mask**: raw signal is passed through unchanged.

    Numba JIT acceleration is used whenever available (see ``_numba_turbo_enabled``).
    """

    _MAX_SPARSE_SECTIONS = 16

    def __init__(self, node_id, params, block_size):
        super().__init__(node_id, params, block_size)
        self._iir_key: Optional[Tuple[int, float, float, int]] = None
        self._hp_coeffs: List[Tuple[float, float, float, float, float]] = []
        self._lp_coeffs: List[Tuple[float, float, float, float, float]] = []
        self._iir_state: Dict[str, List[float]] = {}
        self._sparse_iir_key: Optional[Tuple[int, int, Tuple[Tuple[float, float], ...]]] = None
        # OrderedDict enables O(1) LRU eviction: move_to_end() on hit, pop FIFO on overflow.
        self._section_coeff_cache: OrderedDict[Tuple[int, float, float], Tuple[List[Any], List[Any]]] = OrderedDict()

    def _numba_turbo_enabled(self) -> bool:
        # Runtime policy: force Numba acceleration for preview filtering whenever
        # dependency is available. UI can still expose install guidance.
        return bool(NUMBA_AVAILABLE)

    @staticmethod
    def _resolve_bounds(chunk: AudioChunk) -> Optional[Tuple[float, float]]:
        bounds = getattr(chunk, "band_hz_bounds", None)
        if bounds is None:
            return None
        try:
            low_hz = float(bounds[0])
            high_hz = float(bounds[1])
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.PreviewSinkTask._resolve_bounds", exc)
            return None
        if high_hz <= low_hz:
            return None
        return (low_hz, high_hz)

    @staticmethod
    def _resolve_sparse_mask(chunk: AudioChunk, signal_length: Optional[int] = None) -> Optional[np.ndarray]:
        idx = getattr(chunk, "band_indices", None)
        if idx is None:
            return None
        arr = np.asarray(idx, dtype=np.int64).ravel()

        n = int(signal_length if signal_length is not None else (getattr(chunk, "mono", np.zeros(0, dtype=np.float32)).shape[-1] or 0))
        if n <= 0:
            return None
        fft_bins = (n // 2) + 1
        if arr.size == 0:
            return None
        src_fft_size = getattr(chunk, "band_index_fft_size", None)
        if src_fft_size is None:
            src_fft_size = max(int(n), 1)
        try:
            src_fft_size = max(int(src_fft_size), 1)
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.PreviewSinkTask._resolve_sparse_mask", exc)
            src_fft_size = max(int(n), 1)
        arr = _remap_indices_fft_domain(
            arr,
            src_fft_size=src_fft_size,
            dst_fft_size=max(int(n), 1),
            sample_rate=int(max(getattr(chunk, "sample_rate", SAMPLE_RATE), 1)),
        )
        arr = arr[(arr >= 0) & (arr < fft_bins)]
        if arr.size == 0:
            return None
        mask = np.zeros(fft_bins, dtype=bool)
        mask[np.unique(arr)] = True
        return mask

    @staticmethod
    def _coerce_bounds(bounds: Optional[Tuple[float, float]], sample_rate: float) -> Optional[Tuple[float, float]]:
        if bounds is None:
            return None
        try:
            low_hz = float(bounds[0])
            high_hz = float(bounds[1])
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.PreviewSinkTask._coerce_bounds", exc)
            return None
        nyquist = 0.5 * max(float(sample_rate), 1.0)
        low = max(0.0, min(low_hz, nyquist - 2.0))
        high = max(low + 1.0, min(high_hz, nyquist - 1.0))
        if high <= low:
            return None
        return (low, high)

    @staticmethod
    def _biquad_coeffs(kind: str, sample_rate: float, cutoff_hz: float, q: float = 0.7071) -> Tuple[float, float, float, float, float]:
        sr = max(float(sample_rate), 1.0)
        nyquist = 0.5 * sr
        cutoff = min(max(float(cutoff_hz), 5.0), nyquist * 0.99)
        w0 = 2.0 * math.pi * cutoff / sr
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / max(2.0 * q, 1e-6)

        if kind == "highpass":
            b0 = (1.0 + cos_w0) * 0.5
            b1 = -(1.0 + cos_w0)
            b2 = (1.0 + cos_w0) * 0.5
        else:
            b0 = (1.0 - cos_w0) * 0.5
            b1 = 1.0 - cos_w0
            b2 = (1.0 - cos_w0) * 0.5
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        inv_a0 = 1.0 / max(a0, 1e-9)
        return (
            b0 * inv_a0,
            b1 * inv_a0,
            b2 * inv_a0,
            a1 * inv_a0,
            a2 * inv_a0,
        )

    def _ensure_iir_setup(self, sample_rate: float, low_hz: float, high_hz: float, channels: int):
        key = (int(sample_rate), round(float(low_hz), 3), round(float(high_hz), 3), int(channels))
        if key == self._iir_key:
            return
        self._iir_key = key
        self._iir_state.clear()

        nyquist = 0.5 * float(sample_rate)
        self._hp_coeffs = []
        self._lp_coeffs = []
        if low_hz > 20.0 and low_hz < nyquist * 0.995:
            # Cascading two 2nd-order sections gives a steeper, cleaner 4th-order slope.
            stage = self._biquad_coeffs("highpass", sample_rate, low_hz, q=0.70710678)
            self._hp_coeffs = [stage, stage]
        if high_hz > 1.0 and high_hz < nyquist * 0.995:
            stage = self._biquad_coeffs("lowpass", sample_rate, high_hz, q=0.70710678)
            self._lp_coeffs = [stage, stage]

    def _iir_filter_channel(self, signal: np.ndarray, channel_key: str, use_numba: bool = True) -> np.ndarray:
        out = signal.astype(np.float32, copy=False)
        for stage_idx, coeffs in enumerate(self._hp_coeffs):
            hp_state = self._iir_state.setdefault(f"{channel_key}:hp{stage_idx}", [0.0, 0.0])
            out = _run_biquad(out, coeffs, hp_state, use_numba=use_numba)
        for stage_idx, coeffs in enumerate(self._lp_coeffs):
            lp_state = self._iir_state.setdefault(f"{channel_key}:lp{stage_idx}", [0.0, 0.0])
            out = _run_biquad(out, coeffs, lp_state, use_numba=use_numba)
        return out

    def _section_coeffs(
        self,
        sample_rate: float,
        low_hz: float,
        high_hz: float,
    ) -> Tuple[List[Tuple[float, float, float, float, float]], List[Tuple[float, float, float, float, float]]]:
        key = (int(sample_rate), round(float(low_hz), 3), round(float(high_hz), 3))
        cached = self._section_coeff_cache.get(key)
        if cached is not None:
            # LRU: move accessed entry to the end so it is evicted last.
            self._section_coeff_cache.move_to_end(key)
            return cached
        nyquist = 0.5 * float(sample_rate)
        hp_coeffs: List[Tuple[float, float, float, float, float]] = []
        lp_coeffs: List[Tuple[float, float, float, float, float]] = []
        if low_hz > 20.0 and low_hz < nyquist * 0.995:
            stage = self._biquad_coeffs("highpass", sample_rate, low_hz, q=0.70710678)
            hp_coeffs = [stage, stage]
        if high_hz > 1.0 and high_hz < nyquist * 0.995:
            stage = self._biquad_coeffs("lowpass", sample_rate, high_hz, q=0.70710678)
            lp_coeffs = [stage, stage]
        self._section_coeff_cache[key] = (hp_coeffs, lp_coeffs)
        if len(self._section_coeff_cache) > 64:
            # LRU eviction: the first item is the least recently used.
            self._section_coeff_cache.popitem(last=False)
        return self._section_coeff_cache[key]

    def _filter_section_channel(
        self,
        signal: np.ndarray,
        sample_rate: float,
        low_hz: float,
        high_hz: float,
        section_key: str,
        use_numba: bool,
    ) -> np.ndarray:
        out = signal.astype(np.float32, copy=False)
        hp_coeffs, lp_coeffs = self._section_coeffs(sample_rate, low_hz, high_hz)
        for stage_idx, coeffs in enumerate(hp_coeffs):
            state = self._iir_state.setdefault(f"{section_key}:hp{stage_idx}", [0.0, 0.0])
            out = _run_biquad(out, coeffs, state, use_numba=use_numba)
        for stage_idx, coeffs in enumerate(lp_coeffs):
            state = self._iir_state.setdefault(f"{section_key}:lp{stage_idx}", [0.0, 0.0])
            out = _run_biquad(out, coeffs, state, use_numba=use_numba)
        return out

    @staticmethod
    def _sections_coverage_ratio(sections: List[Tuple[float, float]], sample_rate: float) -> float:
        nyquist = 0.5 * max(float(sample_rate), 1.0)
        total_band = max(min(20000.0, nyquist) - 20.0, 20.0)
        covered = float(sum(max(float(high) - float(low), 0.0) for low, high in sections))
        if covered <= 0.0:
            return 0.0
        return float(min(max(covered / total_band, 0.0), 1.0))

    @classmethod
    def _mask_to_sections(
        cls,
        mask: np.ndarray,
        signal_length: int,
        sample_rate: float,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> List[Tuple[float, float]]:
        if signal_length <= 0:
            return []
        mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
        if mask_arr.size <= 0:
            return []
        idx = np.flatnonzero(mask_arr)
        if idx.size <= 0:
            return []
        nyquist = 0.5 * max(float(sample_rate), 1.0)
        bin_hz = float(max(float(sample_rate), 1.0) / float(max(signal_length, 1)))
        lo_bound = 0.0
        hi_bound = nyquist
        if bounds is not None:
            lo_bound = float(max(bounds[0], 0.0))
            hi_bound = float(min(bounds[1], nyquist))
            if hi_bound <= lo_bound:
                return []

        ranges: List[Tuple[int, int]] = []
        run_start = int(idx[0])
        run_end = int(idx[0])
        for value in idx[1:]:
            v = int(value)
            if v == (run_end + 1):
                run_end = v
                continue
            ranges.append((run_start, run_end))
            run_start = v
            run_end = v
        ranges.append((run_start, run_end))

        sections: List[Tuple[float, float]] = []
        for start_idx, end_idx in ranges:
            low_hz = max((float(start_idx) - 0.5) * bin_hz, 0.0)
            high_hz = min((float(end_idx) + 0.5) * bin_hz, nyquist)
            low_hz = max(low_hz, lo_bound)
            high_hz = min(high_hz, hi_bound)
            if high_hz <= low_hz:
                continue
            sections.append((low_hz, high_hz))

        if len(sections) > cls._MAX_SPARSE_SECTIONS:
            merged: List[Tuple[float, float]] = []
            partitions = np.array_split(np.arange(len(sections), dtype=np.int64), cls._MAX_SPARSE_SECTIONS)
            for part in partitions:
                if part.size <= 0:
                    continue
                lows = [sections[int(i)][0] for i in part]
                highs = [sections[int(i)][1] for i in part]
                low_hz = float(min(lows))
                high_hz = float(max(highs))
                if high_hz > low_hz:
                    merged.append((low_hz, high_hz))
            sections = merged
        return sections

    def _ensure_sparse_iir_layout(self, sample_rate: float, channels: int, sections: List[Tuple[float, float]]):
        key = (
            int(sample_rate),
            int(max(channels, 1)),
            tuple((round(float(low), 2), round(float(high), 2)) for low, high in sections),
        )
        if self._sparse_iir_key == key:
            return
        self._sparse_iir_key = key
        self._iir_state.clear()
        # Clear the coefficient cache on layout change: stale entries for the old
        # section layout (e.g. a previous chromagram frame) waste memory and can
        # pollute cache hits if the same (sr, low, high) key reappears later with
        # different surrounding context.
        self._section_coeff_cache.clear()

    @staticmethod
    def _apply_preview_gain(
        mono: np.ndarray,
        stereo: Optional[np.ndarray],
        coverage_ratio: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        frac = float(min(max(coverage_ratio, 1e-6), 1.0))
        # Keep branch previews audible without aggressive clipping distortion.
        gain = min(1.6, max(1.0, 1.0 / max(math.sqrt(frac), 0.65)))
        mono_scaled = (mono.astype(np.float32, copy=False) * gain).astype(np.float32, copy=False)
        stereo_scaled: Optional[np.ndarray] = None
        if stereo is not None:
            _s = stereo.astype(np.float32, copy=False)
            if _s.ndim == 2:
                # Normalise to channels-first layout (shape: [channels, samples]).
                # Incoming stereo from Stage-B workers is always (2, N); guard
                # against any caller that passes (N, 2) by transposing when the
                # second axis is the shorter one.
                if _s.shape[0] > _s.shape[1] and _s.shape[1] in (1, 2):
                    _s = _s.T
                stereo_scaled = (_s * gain).astype(np.float32, copy=False)
            # Non-2D stereo is silently dropped; caller is responsible for shape.

        if mono_scaled.size == 0:
            return mono_scaled, stereo_scaled

        # Per-sample look-ahead limiter using instant-attack vectorized numpy.
        # Replaces the previous single-block peak divider (0.98/peak) which
        # produced audible level-pumping: a single loud transient made the
        # entire block quiet even when most samples were at a normal level.
        ceiling = np.float32(0.98)
        abs_mono = np.abs(mono_scaled)
        if stereo_scaled is not None and stereo_scaled.ndim == 2 and stereo_scaled.size > 0:
            abs_envelope = np.maximum(abs_mono, np.max(np.abs(stereo_scaled), axis=0))
        else:
            abs_envelope = abs_mono

        # desired_gain[i] = min(1.0, ceiling / |signal[i]|)
        desired = np.where(
            abs_envelope > ceiling,
            ceiling / np.maximum(abs_envelope, np.float32(1e-9)),
            np.float32(1.0),
        ).astype(np.float32, copy=False)

        # Instant-attack via running minimum: gain is only allowed to fall,
        # never rise within a block.  This prevents inter-sample pumping while
        # keeping quiet regions at full gain.
        gain_arr = np.minimum.accumulate(desired)

        mono_out = (mono_scaled * gain_arr).astype(np.float32, copy=False)
        stereo_out: Optional[np.ndarray] = None
        if stereo_scaled is not None:
            if stereo_scaled.ndim == 2:
                stereo_out = (stereo_scaled * gain_arr[np.newaxis, :]).astype(np.float32, copy=False)
            else:
                stereo_out = (stereo_scaled * gain_arr).astype(np.float32, copy=False)
        return mono_out, stereo_out

    def process(self, inputs):
        # Mark this thread as Layer-3 preview so the Layer-2 guard can detect
        # and report a guardrail violation if accidentally reached from here.
        _LAYER3_THREAD_LOCAL.in_preview = True
        try:
            return self._process_impl(inputs)
        finally:
            _LAYER3_THREAD_LOCAL.in_preview = False

    def _process_impl(self, inputs):
        chunk = _ensure_audio_chunk(inputs.get(0), self.block_size)
        # Fetch the raw signal WITHOUT applying analysis_ops so that biquad
        # filtering runs on the unmodified audio.  analysis_ops (midside matrix,
        # attackrelease gate) are applied AFTER the filter chain to preserve the
        # correct order: band-limit first, then gate/matrix.
        mono_base, stereo_base = _signal_for_chunk(
            chunk, prefer_analysis_track=False, apply_analysis_ops=False
        )
        analysis_ops = _normalized_analysis_ops(chunk)
        sample_rate = float(max(getattr(chunk, "sample_rate", SAMPLE_RATE), 1))
        nyquist = 0.5 * sample_rate
        stereo_in = stereo_base
        channels = int(stereo_in.shape[0]) if isinstance(stereo_in, np.ndarray) and stereo_in.ndim == 2 else 1
        mono = mono_base
        stereo_out = None
        coverage = 1.0

        signal_length = int(mono.shape[-1] or 0)
        bounds = self._coerce_bounds(self._resolve_bounds(chunk), sample_rate)
        sparse_mask = self._resolve_sparse_mask(chunk, signal_length=signal_length)
        if bounds is None and sparse_mask is None:
            if analysis_ops:
                # No frequency filtering needed; apply ops directly and return.
                mono, stereo_out = _preview_signal_for_chunk(chunk)
                return AudioChunk(
                    mono,
                    stereo_out,
                    getattr(chunk, "sample_rate", SAMPLE_RATE),
                    provider=chunk.provider,
                    frame_index=chunk.frame_index,
                    band_indices=getattr(chunk, "band_indices", None),
                    band_index_fft_size=getattr(chunk, "band_index_fft_size", None),
                    band_hz_bounds=getattr(chunk, "band_hz_bounds", None),
                    lineage=tuple(getattr(chunk, "lineage", ())) + (f"{self.node_id}:preview",),
                    spectrum_token=getattr(chunk, "spectrum_token", None),
                    latency_ms=float(getattr(chunk, "latency_ms", 0.0) or 0.0),
                )
            return chunk

        if bounds is not None and sparse_mask is None:
            low_hz, high_hz = bounds
            if (not analysis_ops) and low_hz <= 20.5 and high_hz >= (nyquist * 0.985):
                return chunk
            self._ensure_iir_setup(sample_rate, low_hz, high_hz, channels)
            use_numba = self._numba_turbo_enabled()
            if isinstance(stereo_in, np.ndarray) and stereo_in.ndim == 2 and stereo_in.shape[1] == mono_base.shape[-1]:
                stereo_out = np.empty_like(stereo_in, dtype=np.float32)
                for c in range(int(stereo_in.shape[0])):
                    stereo_out[c] = self._iir_filter_channel(stereo_in[c], f"ch{c}", use_numba=use_numba)
                mono = np.mean(stereo_out, axis=0).astype(np.float32, copy=False)
            else:
                mono = self._iir_filter_channel(mono_base, "mono", use_numba=use_numba)
            coverage = self._sections_coverage_ratio([(low_hz, high_hz)], sample_rate)
        else:
            sections: List[Tuple[float, float]] = []
            if sparse_mask is not None:
                sections = self._mask_to_sections(
                    sparse_mask,
                    signal_length=max(signal_length, 1),
                    sample_rate=sample_rate,
                    bounds=bounds,
                )
            if not sections:
                mono = np.zeros_like(mono_base, dtype=np.float32)
                if isinstance(stereo_in, np.ndarray) and stereo_in.ndim == 2 and stereo_in.shape[1] == mono_base.shape[-1]:
                    stereo_out = np.zeros_like(stereo_in, dtype=np.float32)
                else:
                    stereo_out = None
                coverage = 0.0
            else:
                self._ensure_sparse_iir_layout(sample_rate, channels, sections)
                use_numba = self._numba_turbo_enabled()
                if isinstance(stereo_in, np.ndarray) and stereo_in.ndim == 2 and stereo_in.shape[1] == mono_base.shape[-1]:
                    stereo_out = np.zeros_like(stereo_in, dtype=np.float32)
                    for c in range(int(stereo_in.shape[0])):
                        accum = np.zeros(stereo_in.shape[1], dtype=np.float32)
                        for sec_idx, (low_hz, high_hz) in enumerate(sections):
                            band = self._filter_section_channel(
                                stereo_in[c],
                                sample_rate,
                                low_hz,
                                high_hz,
                                section_key=f"ch{c}:sec{sec_idx}",
                                use_numba=use_numba,
                            )
                            accum += band
                        stereo_out[c] = accum
                    mono = np.mean(stereo_out, axis=0).astype(np.float32, copy=False)
                else:
                    accum = np.zeros_like(mono_base, dtype=np.float32)
                    for sec_idx, (low_hz, high_hz) in enumerate(sections):
                        band = self._filter_section_channel(
                            mono_base,
                            sample_rate,
                            low_hz,
                            high_hz,
                            section_key=f"mono:sec{sec_idx}",
                            use_numba=use_numba,
                        )
                        accum += band
                    mono = accum
                coverage = self._sections_coverage_ratio(sections, sample_rate)

        mono, stereo_out = self._apply_preview_gain(mono, stereo_out, coverage)

        # Apply analysis_ops (midside matrix / attackrelease gate) to the
        # already-filtered signal so the processing order mirrors the abstract
        # graph: frequency-band first, then temporal/spatial processing.
        if analysis_ops:
            _filtered = AudioChunk(
                mono,
                stereo_out,
                int(getattr(chunk, "sample_rate", SAMPLE_RATE)),
                provider=chunk.provider,
                frame_index=chunk.frame_index,
                analysis_ops=analysis_ops,
            )
            mono, stereo_out = _signal_for_chunk(
                _filtered, prefer_analysis_track=False, apply_analysis_ops=True
            )

        return AudioChunk(
            mono,
            stereo_out,
            getattr(chunk, "sample_rate", SAMPLE_RATE),
            provider=chunk.provider,
            frame_index=chunk.frame_index,
            band_indices=getattr(chunk, "band_indices", None),
            band_index_fft_size=getattr(chunk, "band_index_fft_size", None),
            band_hz_bounds=getattr(chunk, "band_hz_bounds", None),
            lineage=tuple(getattr(chunk, "lineage", ())) + (f"{self.node_id}:preview",),
            spectrum_token=getattr(chunk, "spectrum_token", None),
            analysis_ops=(),
            latency_ms=float(getattr(chunk, "latency_ms", 0.0) or 0.0),
        )

    def is_enabled(self) -> bool:
        return bool(self.parameters.get("preview_enabled", True))


class RollingAvgTask(ACEAudioTask):
    """Computes a rolling (moving) average of a scalar attribute stream.

    Receives a float ``Value`` from a connected stream operator and maintains a
    fixed-size frame buffer whose capacity is derived from ``Number of Frames``.
    Each call to ``process()`` appends the current value and returns the mean of
    the buffer.  The buffer is resized automatically when the frame count
    changes, preserving as many recent samples as possible.

    This is a scalar post-processing node: it does not touch the time-domain
    audio signal and is therefore transparent to the Layer-3 preview chain.
    """

    def __init__(self, node_id: str, parameters: Dict[str, Any], block_size: int):
        super().__init__(node_id, parameters, block_size)
        self._buffer: deque = deque(maxlen=1)
        self._buffer_maxlen: int = 1

    def process(self, inputs):
        # ── Resolve current value ─────────────────────────────────────────────
        raw_value = inputs.get(0)
        if raw_value is None:
            raw_value = self.parameters.get("in_0", 0.0)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = 0.0

        # ── Resolve frame count ───────────────────────────────────────────────
        raw_frame_count = inputs.get(1)
        if raw_frame_count is None:
            raw_frame_count = self.parameters.get("in_1", 1)
        try:
            frame_count = max(1, int(round(float(raw_frame_count))))
        except (TypeError, ValueError):
            frame_count = 1

        # ── Resize buffer when capacity changes ───────────────────────────────
        new_maxlen = max(1, int(frame_count))
        if new_maxlen != self._buffer_maxlen:
            # Preserve as many recent samples as fit in the new capacity.
            existing = list(self._buffer)
            self._buffer = deque(existing[-new_maxlen:], maxlen=new_maxlen)
            self._buffer_maxlen = new_maxlen

        self._buffer.append(value)
        return float(np.mean(self._buffer))


class StreamAccumulatorTask(ACEAudioTask):
    """Running sum utility for scalar or vector stream values.

    Maintains accumulator state in the task instance so values persist across
    frames in Stage-B worker execution.  Resets automatically on timeline
    rewind / first frame and supports an optional manual reset trigger through
    ``parameters`` keys (``manual_reset``, ``reset_trigger``, ``reset``).
    """

    def __init__(self, node_id: str, parameters: Dict[str, Any], block_size: int):
        super().__init__(node_id, parameters, block_size)
        self._accumulator: Optional[np.ndarray] = None
        self._last_frame_request: Optional[int] = None
        self._manual_reset_latched: bool = False

    def export_state(self) -> Dict[str, Any]:
        """Return a serializable snapshot of the current accumulator state."""
        if self._accumulator is None:
            return {}
        try:
            acc = np.asarray(self._accumulator, dtype=np.float64).reshape(-1)
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.StreamAccumulatorTask.export_state", exc)
            return {}
        if acc.size <= 0:
            return {}
        return {
            "accumulator": acc.copy(),
            "last_frame_request": (
                int(self._last_frame_request) if self._last_frame_request is not None else None
            ),
            "manual_reset_latched": bool(self._manual_reset_latched),
        }

    def import_state(self, state: Dict[str, Any]) -> bool:
        """Restore accumulator state from ``export_state()`` data."""
        if not isinstance(state, dict):
            return False
        raw_acc = state.get("accumulator")
        if raw_acc is None:
            return False
        acc = self._coerce_input_array(raw_acc)
        if int(acc.size) <= 0:
            return False
        self._accumulator = np.asarray(acc, dtype=np.float64).copy()
        raw_last = state.get("last_frame_request")
        try:
            self._last_frame_request = int(raw_last) if raw_last is not None else None
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.StreamAccumulatorTask.import_state", exc)
            self._last_frame_request = None
        self._manual_reset_latched = bool(state.get("manual_reset_latched", False))
        return True

    @staticmethod
    def _coerce_input_array(raw_value: Any) -> np.ndarray:
        if raw_value is None:
            return np.zeros(1, dtype=np.float64)
        if isinstance(raw_value, (int, float, np.integer, np.floating)):
            resolved = float(raw_value)
            if not np.isfinite(resolved):
                resolved = 0.0
            return np.asarray([resolved], dtype=np.float64)
        try:
            arr = np.asarray(raw_value, dtype=np.float64).reshape(-1)
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.StreamAccumulatorTask._coerce_input_array", exc)
            return np.zeros(1, dtype=np.float64)
        if arr.size <= 0:
            return np.zeros(1, dtype=np.float64)
        arr = _finite_copy(arr, dtype=None)
        return arr.astype(np.float64, copy=False)

    @staticmethod
    def _coerce_reset_flag(raw_value: Any) -> bool:
        if isinstance(raw_value, bool):
            return bool(raw_value)
        if isinstance(raw_value, (int, float, np.integer, np.floating)):
            try:
                return bool(np.isfinite(float(raw_value)) and abs(float(raw_value)) > 1e-9)
            except Exception as exc:
                utils.note_swallowed("audio_tasks_registry.StreamAccumulatorTask._coerce_reset_flag", exc)
                return False
        if isinstance(raw_value, str):
            token = raw_value.strip().lower()
            if token in {"1", "true", "yes", "on"}:
                return True
            if token in {"0", "false", "no", "off", ""}:
                return False
        return False

    def _manual_reset_requested(self, inputs: Dict[int, Any]) -> bool:
        raw_reset = inputs.get(1)
        if raw_reset is None:
            for key in ("manual_reset", "reset_trigger", "reset"):
                if key in self.parameters:
                    raw_reset = self.parameters.get(key)
                    break
        flag = self._coerce_reset_flag(raw_reset)
        triggered = bool(flag and not self._manual_reset_latched)
        self._manual_reset_latched = bool(flag)
        return triggered

    def _should_reset_for_frame(self, frame_idx: int, manual_reset: bool) -> bool:
        if manual_reset:
            return True
        if frame_idx < 0:
            return False
        frame_start = int(getattr(self, "scene_frame_start", 1) or 1)
        if frame_idx <= frame_start:
            return True
        last_frame = self._last_frame_request
        if last_frame is not None and frame_idx < int(last_frame):
            return True
        return False

    def process(self, inputs):
        raw_value = inputs.get(0)
        if raw_value is None:
            raw_value = self.parameters.get("in_0", 0.0)
        value_arr = self._coerce_input_array(raw_value)

        try:
            frame_idx = utils.as_int(getattr(self, "_current_frame_request", None), -1)
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.StreamAccumulatorTask.process", exc)
            frame_idx = -1
        manual_reset = self._manual_reset_requested(inputs)
        should_reset = self._should_reset_for_frame(frame_idx, manual_reset)

        if self._accumulator is None or tuple(self._accumulator.shape) != tuple(value_arr.shape):
            self._accumulator = np.zeros_like(value_arr, dtype=np.float64)
            should_reset = False
        elif should_reset:
            self._accumulator.fill(0.0)

        np.add(self._accumulator, value_arr, out=self._accumulator, casting="unsafe")
        if frame_idx >= 0:
            self._last_frame_request = int(frame_idx)

        if int(self._accumulator.size) == 1:
            return float(self._accumulator[0])
        return self._accumulator.astype(np.float32, copy=False)


class AudioMathTask(ACEAudioTask):
    """Element-wise arithmetic over multiple audio streams.

    Combines 1–8 mono/stereo ``AudioChunk`` inputs using a single configurable
    operation (``add``, ``subtract``, ``multiply``).  Streams are resampled
    to the length of the first (reference) input before the operation is
    applied so mismatched block sizes are handled gracefully.

    Acts as a transparent pass-through when only one input is present.
    """
    @staticmethod
    def _signal_for_math(
        mono_sig: np.ndarray,
        stereo_sig: Optional[np.ndarray],
        channels: int,
        target_len: int,
        resample: bool = False,
    ) -> np.ndarray:
        mono_arr_raw = np.asarray(mono_sig, dtype=np.float32).reshape(-1)
        if resample:
            mono_arr = AudioMathTask._resample_1d(mono_arr_raw, target_len)
        else:
            mono_arr = _aligned_signal_length(mono_arr_raw, target_len)
        stereo_arr = None
        if isinstance(stereo_sig, np.ndarray):
            raw = np.asarray(stereo_sig, dtype=np.float32)
            if raw.ndim == 2:
                if raw.shape[1] == mono_arr.shape[0]:
                    stereo_arr = raw.astype(np.float32, copy=False)
                elif raw.shape[0] == mono_arr.shape[0]:
                    stereo_arr = raw.T.astype(np.float32, copy=False)
                elif raw.shape[1] > 0 and raw.shape[0] > 0:
                    stereo_arr = raw.astype(np.float32, copy=False)

        if stereo_arr is None:
            stereo_arr = mono_arr[np.newaxis, :]
        else:
            if resample:
                stereo_arr = AudioMathTask._resample_nd(stereo_arr, target_len)
            else:
                stereo_arr = _aligned_signal_length(stereo_arr, target_len)

        if stereo_arr.shape[0] < channels:
            if stereo_arr.shape[0] == 1:
                stereo_arr = np.repeat(stereo_arr, channels, axis=0)
            else:
                pad = np.repeat(stereo_arr[-1:, :], channels - stereo_arr.shape[0], axis=0)
                stereo_arr = np.concatenate((stereo_arr, pad), axis=0)
        elif stereo_arr.shape[0] > channels:
            stereo_arr = stereo_arr[:channels]

        return stereo_arr.astype(np.float32, copy=False)

    @staticmethod
    def _resample_1d(signal: np.ndarray, target_len: int) -> np.ndarray:
        src = np.asarray(signal, dtype=np.float32).reshape(-1)
        n = int(src.shape[0] or 0)
        t_len = max(int(target_len), 1)
        if n == t_len:
            return src.astype(np.float32, copy=False)
        if n <= 0:
            return np.zeros(t_len, dtype=np.float32)
        if n == 1:
            return np.full(t_len, float(src[0]), dtype=np.float32)

        x_old = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, t_len, endpoint=False, dtype=np.float64)
        out = np.interp(x_new, x_old, src.astype(np.float64, copy=False))
        return out.astype(np.float32, copy=False)

    @staticmethod
    def _resample_nd(signal: np.ndarray, target_len: int) -> np.ndarray:
        arr = np.asarray(signal, dtype=np.float32)
        if arr.ndim != 2:
            flat = AudioMathTask._resample_1d(arr.reshape(-1), target_len)
            return flat[np.newaxis, :]
        rows = [AudioMathTask._resample_1d(arr[idx], target_len) for idx in range(int(arr.shape[0]))]
        return np.stack(rows, axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _merge_band_indices(chunks: List[AudioChunk], fft_size: int) -> Optional[np.ndarray]:
        merged: Optional[np.ndarray] = None
        for chunk in chunks:
            idx = _resolve_effective_bin_indices(chunk, fft_size)
            if idx is None:
                return None
            if merged is None:
                merged = idx.astype(np.int64, copy=False)
            else:
                merged = np.union1d(merged, idx).astype(np.int64, copy=False)
        return merged

    @staticmethod
    def _merge_band_hz_bounds(chunks: List[AudioChunk]) -> Optional[Tuple[float, float]]:
        merged_low: Optional[float] = None
        merged_high: Optional[float] = None

        for chunk in chunks:
            bounds = getattr(chunk, "band_hz_bounds", None)
            if bounds is None:
                return None
            try:
                low = float(bounds[0])
                high = float(bounds[1])
            except Exception as exc:
                utils.note_swallowed("audio_tasks_registry.AudioMathTask._merge_band_hz_bounds", exc)
                return None
            if high <= low:
                return None
            if merged_low is None or low < merged_low:
                merged_low = low
            if merged_high is None or high > merged_high:
                merged_high = high

        if merged_low is None or merged_high is None or merged_high <= merged_low:
            return None
        return (merged_low, merged_high)

    def process(self, inputs):
        ordered = [inputs[idx] for idx in sorted(inputs.keys()) if idx in inputs]
        if not ordered:
            return self.silence_buffer

        chunks = [_ensure_audio_chunk(val, self.block_size) for val in ordered]
        analysis_pairs = [_analysis_signals_normalized(c, prefer_analysis_track=True) for c in chunks]
        audio_pairs = [_analysis_signals_normalized(c, prefer_analysis_track=False) for c in chunks]
        sample_rates = [max(int(getattr(c, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE), 1) for c in chunks]
        unique_sample_rates = set(sample_rates)
        resample_inputs = len(unique_sample_rates) > 1
        target_sample_rate = max(unique_sample_rates) if unique_sample_rates else int(SAMPLE_RATE)
        latency_ms = float(max((float(getattr(c, "latency_ms", 0.0) or 0.0) for c in chunks), default=0.0))
        op = str(self.parameters.get("op", "add")).lower().strip()

        def _mix_signal_pairs(signal_pairs):
            if resample_inputs:
                candidate_lengths = []
                for idx, pair in enumerate(signal_pairs):
                    src_len = int(np.asarray(pair[0]).shape[-1] or 0)
                    src_sr = max(int(sample_rates[idx]), 1)
                    mapped = int(round((float(src_len) / float(src_sr)) * float(target_sample_rate)))
                    candidate_lengths.append(max(mapped, 1))
                target_len = max(candidate_lengths, default=0)
                if target_len <= 0:
                    target_len = max(int(self.block_size), 1)
                mono_inputs = [self._resample_1d(pair[0], target_len) for pair in signal_pairs]
            else:
                target_len = max((int(np.asarray(pair[0]).shape[-1] or 0) for pair in signal_pairs), default=0)
                if target_len <= 0:
                    target_len = max(int(self.block_size), 1)
                mono_inputs = [_aligned_signal_length(pair[0], target_len) for pair in signal_pairs]

            mono_stack = np.stack(mono_inputs, axis=0)
            if op == "subtract":
                mono_out = mono_stack[0].astype(np.float32, copy=True)
                if mono_stack.shape[0] > 1:
                    mono_out -= np.sum(mono_stack[1:], axis=0, dtype=np.float32)
            elif op == "multiply":
                mono_out = np.prod(mono_stack, axis=0, dtype=np.float32)
            else:
                mono_out = np.sum(mono_stack, axis=0, dtype=np.float32)

            stereo_out = None
            stereo_inputs = [pair[1] for pair in signal_pairs]
            has_stereo = any(
                isinstance(sig, np.ndarray) and sig.ndim == 2 and sig.shape[0] >= 2 and sig.shape[1] > 0
                for sig in stereo_inputs
            )
            if has_stereo:
                channel_count = max(
                    int(sig.shape[0])
                    for sig in stereo_inputs
                    if isinstance(sig, np.ndarray) and sig.ndim == 2 and sig.shape[1] > 0
                )
                channel_count = max(channel_count, 2)
                stereo_stack = np.stack(
                    [
                        self._signal_for_math(
                            mono_inputs[idx],
                            stereo_inputs[idx],
                            channel_count,
                            target_len,
                            resample=resample_inputs,
                        )
                        for idx in range(len(chunks))
                    ],
                    axis=0,
                )
                if op == "subtract":
                    stereo_out = stereo_stack[0].astype(np.float32, copy=True)
                    if stereo_stack.shape[0] > 1:
                        stereo_out -= np.sum(stereo_stack[1:], axis=0, dtype=np.float32)
                elif op == "multiply":
                    stereo_out = np.prod(stereo_stack, axis=0, dtype=np.float32)
                else:
                    stereo_out = np.sum(stereo_stack, axis=0, dtype=np.float32)
                mono_out = np.mean(stereo_out, axis=0).astype(np.float32, copy=False)
            return mono_out.astype(np.float32, copy=False), stereo_out, int(target_len)

        mono, stereo, target_len = _mix_signal_pairs(audio_pairs)
        analysis_mono, analysis_stereo, analysis_len = _mix_signal_pairs(analysis_pairs)

        sample_rate = int(target_sample_rate)
        merged_fft_size: Optional[int]
        merged_band_indices: Optional[np.ndarray]
        merged_band_bounds: Optional[Tuple[float, float]]
        if resample_inputs:
            # Cross-rate mixes do not preserve a single authoritative spectral mask domain.
            merged_fft_size = None
            merged_band_indices = None
            merged_band_bounds = None
        else:
            fft_size_candidates = [
                _resolve_chunk_fft_size(
                    c,
                    fallback_size=max(int(getattr(c, "mono", np.zeros(1, dtype=np.float32)).shape[-1] or 1), 1),
                )
                for c in chunks
            ]
            merged_fft_size = max([max(int(analysis_len or 1), 1)] + fft_size_candidates)
            merged_band_indices = self._merge_band_indices(chunks, merged_fft_size)
            merged_band_bounds = self._merge_band_hz_bounds(chunks)

        src = chunks[0]
        output_provider = getattr(src, "provider", None) if len(chunks) == 1 else None
        return AudioChunk(
            mono.astype(np.float32, copy=False),
            stereo,
            sample_rate,
            provider=output_provider,
            frame_index=src.frame_index,
            analysis_mono=analysis_mono,
            analysis_stereo=analysis_stereo,
            frontend_config=None,
            band_indices=merged_band_indices,
            band_index_fft_size=(merged_fft_size if (merged_fft_size is not None and merged_band_indices is not None) else None),
            band_hz_bounds=merged_band_bounds,
            lineage=tuple(getattr(src, "lineage", ())) + (f"{self.node_id}:0",),
            analysis_ops=(),
            latency_ms=latency_ms,
        )


class ACEAudioGraph:
    """Complete DSP processing chain for a single audio source.

    Each graph owns a dictionary of ``ACEAudioTask`` nodes keyed by their
    node ID, plus the serialized parallel IR used to dispatch Stage-B worker
    jobs.  Graphs are compiled from the abstract tree IR by
    ``ACEAudioEngine.compile()`` and evaluated each frame in
    ``ACEAudioEngine.execute_next_frame()``.

    Filter state, block-size buffers, and parallel bookkeeping are isolated
    per-graph so that multiple simultaneous sources never share mutable DSP
    state.
    """
    def __init__(self, source_context: Any, proxy_name: str):
        self.source_context = source_context
        self.proxy_name = proxy_name
        self.tasks: Dict[str, ACEAudioTask] = {}
        self.output_attr_keys: List[str] = []
        self.sink_bindings_by_attr: Dict[str, List["ACESinkBinding"]] = {}
        self.parallel_graph_uid: str = ""
        self.parallel_ir: Dict[str, Any] = {}
        self.parallel_ir_digest: str = ""
        self.parallel_offload_attr_keys: List[str] = []
        self.parallel_output_shards: List[List[str]] = []
        self.parallel_local_attr_keys: List[str] = []
        self.manual_preview_task_ids: List[str] = []
        self.auto_preview_task_ids: List[str] = []
        self.auto_preview_passthrough_specs: List[Tuple[str, int]] = []
        self.preview_task_ids: List[str] = []
        self.source_task_ids: List[str] = []
        self.live_source_task_ids: List[str] = []
        self.file_source_task_ids: List[str] = []
        self.block_size: int = 0
        self.source_uid: str = ""
        self.scheduler_mode: str = "legacy"
        self.timeslice_mode: str = "fixed"
        self.execution_order: List[str] = []
        self.eval_plan_cache: Dict[Tuple[str, ...], List[str]] = {}
        self.output_slot_specs: List[Tuple[str, str, int]] = []
        self.output_slot_specs_by_subset: Dict[Tuple[str, ...], List[Tuple[str, str, int]]] = {}
        self.proxy_lookup_fn: Optional[Callable[[str], Any]] = None

    def get_proxy_object(self):
        resolver = getattr(self, "proxy_lookup_fn", None)
        if callable(resolver):
            try:
                return resolver(str(self.proxy_name or ""))
            except Exception as exc:
                utils.note_swallowed("audio_tasks_registry.ACEAudioGraph.get_proxy_object", exc)
                return None
        return None


@dataclass(frozen=True)
class ACESinkBinding:
    """Concrete main-thread write target resolved at compile time."""

    sink_kind: str
    target_object_name: str
    target_data_name: str
    attr_type: str = "FLOAT"
