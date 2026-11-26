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


# Samples before a block that a source hands out with it, for analysis windows longer than the block.
HISTORY_SAMPLES = 4096


def _finite_copy(arr, dtype="float32"):
    out = np.array(arr, dtype=dtype, copy=True)
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = 0.0
    return out


def _make_silence_chunk(block_size: int, sample_rate: int = SAMPLE_RATE) -> "AudioChunk":
    silence = np.zeros(max(int(block_size), 1), dtype=np.float32)
    return AudioChunk(silence, None, int(max(sample_rate, 1)), latency_ms=0.0)


_KEEP = object()


class AudioChunk:
    """Lightweight wrapper around one audio block passed between DSP task nodes."""
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
        "history_mono",
        "history_stereo",
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
                 latency_ms: float = 0.0,
                 history_mono: Optional[np.ndarray] = None,
                 history_stereo: Optional[np.ndarray] = None):
        # The audio just before this block (zeros before the start of a file), in the domain of
        # analysis_mono before frontend processing. None when nothing is known about it.
        self.history_mono = history_mono
        self.history_stereo = history_stereo
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
                utils.note_swallowed("chunk.AudioChunk.__init__", exc)
                self.band_index_fft_size = None
        elif band_indices is not None:
            # Default behavior: assume indices target the chunk FFT domain.
            self.band_index_fft_size = max(int(self.mono.shape[-1] or 1), 1)
        else:
            self.band_index_fft_size = None
        self.band_hz_bounds = band_hz_bounds
        self.lineage = lineage or ()
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
        """Return a cached feature, optionally computing it on the first call."""
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
    latency_ms: Any = _KEEP,
    history_mono: Any = _KEEP,
    history_stereo: Any = _KEEP,
) -> AudioChunk:
    """Create a new AudioChunk by overriding selected fields from *source*."""
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
        latency_ms=(source.latency_ms if latency_ms is _KEEP else latency_ms),
        history_mono=(source.history_mono if history_mono is _KEEP else history_mono),
        history_stereo=(source.history_stereo if history_stereo is _KEEP else history_stereo),
    )


def _signal_for_chunk(chunk: AudioChunk, *, prefer_analysis_track: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """The (mono, stereo) arrays of a chunk: its analysis track when it has one and that is preferred."""
    analysis_mono = getattr(chunk, "analysis_mono", None)
    analysis_stereo = getattr(chunk, "analysis_stereo", None)
    mono_base = analysis_mono if prefer_analysis_track and isinstance(analysis_mono, np.ndarray) else getattr(chunk, "mono", None)
    stereo_base = analysis_stereo if prefer_analysis_track and isinstance(analysis_stereo, np.ndarray) else getattr(chunk, "stereo", None)
    mono = mono_base if isinstance(mono_base, np.ndarray) else np.zeros(0, dtype=np.float32)
    return np.asarray(mono, dtype=np.float32).reshape(-1), stereo_base


def _analysis_signal_for_chunk(chunk: AudioChunk) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    return _signal_for_chunk(chunk, prefer_analysis_track=True)


def _preview_signal_for_chunk(chunk: AudioChunk) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    return _signal_for_chunk(chunk, prefer_analysis_track=False)


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
