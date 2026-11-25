"""Preview sink task: time-domain monitor audio."""

import math
from collections import OrderedDict
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from .filters import NUMBA_AVAILABLE
from ... import utils
from .base import ACEAudioTask, _LAYER3_THREAD_LOCAL
from .chunk import AudioChunk, SAMPLE_RATE, _ensure_audio_chunk, _normalized_analysis_ops, _preview_signal_for_chunk, _signal_for_chunk
from .filters import _run_biquad
from .spectral import _remap_indices_fft_domain


class PreviewSinkTask(ACEAudioTask):
    """Terminal monitor sink for branch-audio preview."""

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
            utils.note_swallowed("preview.PreviewSinkTask._resolve_bounds", exc)
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
            utils.note_swallowed("preview.PreviewSinkTask._resolve_sparse_mask", exc)
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
            utils.note_swallowed("preview.PreviewSinkTask._coerce_bounds", exc)
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
        # Coefficients depend on the section layout; drop them when it changes.
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
                # Channels-first; (N, 2) input is transposed.
                if _s.shape[0] > _s.shape[1] and _s.shape[1] in (1, 2):
                    _s = _s.T
                stereo_scaled = (_s * gain).astype(np.float32, copy=False)
            # Non-2D stereo is silently dropped; caller is responsible for shape.

        if mono_scaled.size == 0:
            return mono_scaled, stereo_scaled

        # Per-sample limiter with instant attack: one transient does not quiet the whole block.
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

        # Running minimum: gain only falls within a block.
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
        # Filter the raw signal; analysis ops apply after the filters (band first, then gate or matrix).
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
