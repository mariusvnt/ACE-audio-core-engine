"""Audio chunk passed between tasks, and its signal helpers."""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from ... import utils
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .source import SourceTask


SAMPLE_RATE = 44100


MIN_BLOCK_SIZE = 512


MAX_BLOCK_SIZE = 4096


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


_KEEP = object()


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
