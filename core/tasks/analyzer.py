"""Pooled AudioFlux analyzers and feature extraction."""

import math
import os
import hashlib
import time
import zlib
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple
import numpy as np
try:
    import audioflux  # type: ignore
    from audioflux import type as af_type  # type: ignore
    AUDIOFLUX_AVAILABLE = True
except ImportError:
    AUDIOFLUX_AVAILABLE = False
from ... import utils
from .chunk import AudioChunk, SAMPLE_RATE, _analysis_signal_for_chunk, _normalized_analysis_ops
from .spectral import _analysis_signals_normalized, _remap_indices_fft_domain, _safe_rms, _zero_crossing_rate


class AudioFluxAnalyzer:
    """Pool-managed wrapper around long-lived audioflux BFT / Spectral / Pitch objects."""
    _POOL: Dict[Tuple[int, int], list['AudioFluxAnalyzer']] = {}
    _MAX_POOL_PER_KEY = 4
    _MAX_POOL_KEYS = 16
    _MAX_PREV_MASK_ENTRIES = 128
    # Spectra shared by all analyzers: one transform per chunk and frame for every splitter band.
    # Worker processes are single-threaded, so no lock.
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
                utils.note_swallowed("analyzer.AudioFluxAnalyzer.__init__", exc)
                self.bft = None
                self.spectral = None
                self.pitch = None
                self.pitch_buffer = None

    @classmethod
    def borrow(cls, sample_rate: int, block_size: int) -> 'AudioFluxAnalyzer':
        """Retrieve an analyzer from the pool or create a new one."""
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
        try:
            raw = int(
                os.environ.get(
                    "ACE_GLOBAL_SPECTRUM_CACHE_MAX",
                    cls._GLOBAL_SPECTRUM_CACHE_MAX_DEFAULT,
                )
                or cls._GLOBAL_SPECTRUM_CACHE_MAX_DEFAULT
            )
        except Exception as exc:
            utils.note_swallowed("analyzer.AudioFluxAnalyzer._global_spectrum_cache_limit", exc)
            raw = int(cls._GLOBAL_SPECTRUM_CACHE_MAX_DEFAULT)
        return int(min(max(raw, 16), 1024))

    def reset_state(self):
        """Clear all per-frame caches and flux-history state."""
        self.prev_masked_mag.clear()
        self.cache_frame = -1
        self.cache.clear()

    def _prepare_frame_cache(self, frame_index: int):
        if self.cache_frame != frame_index:
            self.cache_frame = frame_index
            self.cache.clear()

    @staticmethod
    def _chunk_token(chunk: AudioChunk):
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
        """Resolve the effective FFT-bin mask for this analyzer's FFT domain."""
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
                    utils.note_swallowed("analyzer.AudioFluxAnalyzer._resolve_chunk_mask_indices", exc)
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
            utils.note_swallowed("analyzer.AudioFluxAnalyzer._resolve_chunk_mask_indices#2", exc)
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

        data, _ = _analysis_signal_for_chunk(chunk)
        # Guard against empty chunks (e.g. from _make_silence_chunk on broken sources)
        # to prevent null/invalid spectrum entries from entering the cache.
        if data is None or data.size == 0:
            return None

        # The chunk token and frame stay the same when the file, sample rate or
        # frame rate changes, so the key also holds what the spectrum is made
        # of: the samples and this analyzer's FFT. Otherwise another setup's
        # spectrum comes back, and with another FFT size the band masks index
        # past its end.
        samples = np.ascontiguousarray(data, dtype=np.float32)
        global_key = (
            chunk_token, "full_spectrum", int(chunk.frame_index),
            int(self.sample_rate), int(self.fft_length), int(samples.size), zlib.crc32(samples),
        )
        global_mag = AudioFluxAnalyzer._GLOBAL_SPECTRUM_CACHE.get(global_key)
        if global_mag is not None:
            AudioFluxAnalyzer._GLOBAL_SPECTRUM_CACHE.move_to_end(global_key)
            self.cache[spec_key] = global_mag
            return global_mag
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
        with utils.logged_errors("analyzer.AudioFluxAnalyzer._ensure_spectrum"):
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
                utils.note_swallowed("analyzer.AudioFluxAnalyzer._ensure_spectrum#2", exc)
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

        # Positive flux, sum(max(cur - prev, 0)) / sum(prev + eps), computed here because
        # audioflux's flux() depends on a time length set elsewhere.
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
        """Compute (and cache) a feature for the given frame."""
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
            # Onset comes from OnsetTask's own envelope detector.

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
