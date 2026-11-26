"""Delta tasks: splitters and masks that constrain downstream analysis."""

import math
from typing import List, Optional, Tuple
import numpy as np
from ... import utils
from .base import ACEAudioTask
from .block_size import _block_duration_ms
from .chunk import SAMPLE_RATE, _analysis_signal_for_chunk, _append_analysis_op, _ensure_audio_chunk, _normalized_analysis_ops, _spawn_audio_chunk
from .spectral import _resolve_chunk_fft_size, _resolve_effective_bin_indices


# Split frequencies closer than this coincide: the band between them is empty.
_EMPTY_BAND_HZ = 1.0


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
                utils.note_swallowed("delta.FrequencySplitterTask._resolve_band_limits", exc)
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

        # One value per cut, duplicates included: every output keeps its band.
        return [float(v) for v in np.sort(np.clip(mapped, low, high)).tolist()]

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
        unknown = set(self.parameters.get("unknown_inputs", ()) or ())
        # A linked split frequency is what travels through its link: computed on the main side
        # (audio maths included), it wins over the raw value of an upstream operator.
        driven = set(self.parameters.get("driven_inputs", ()) or ())
        while True:
            val = None if i in driven else inputs.get(i)
            if val is None: val = self.parameters.get(f"in_{i}")
            if val is None and i not in unknown and i not in driven: break
            split_values.append(None if val is None else float(val))
            i += 1
        # A linked split frequency nobody can compute joins the next one up: its band is empty.
        above = 20000.0
        for index in range(len(split_values) - 1, -1, -1):
            if split_values[index] is None:
                split_values[index] = above
            above = split_values[index]

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
                lineage = tuple(getattr(chunk, "lineage", ())) + (f"{self.node_id}:{out_idx}",)
                if high_cut - low_cut < _EMPTY_BAND_HZ:
                    # Silence of its own: zero-width bounds read as "no limit" downstream, and the
                    # source's shared spectrum would put the whole signal back.
                    bands.append(
                        _spawn_audio_chunk(
                            chunk,
                            mono=np.zeros_like(audio),
                            stereo=(None if stereo is None else np.zeros_like(stereo)),
                            analysis_mono=None,
                            analysis_stereo=None,
                            history_mono=None,
                            history_stereo=None,
                            provider=None,
                            band_hz_bounds=(low_cut, low_cut),
                            lineage=lineage,
                            spectrum_token=None,
                            analysis_ops=(),
                            latency_ms=latency_ms,
                        )
                    )
                    continue
                bands.append(
                    _spawn_audio_chunk(
                        chunk,
                        mono=audio,
                        stereo=stereo,
                        sample_rate=sample_rate,
                        band_indices=chunk.band_indices,
                        band_index_fft_size=band_index_fft_size,
                        band_hz_bounds=(low_cut, high_cut),
                        lineage=lineage,
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
        """Coarse FFT grids can miss a chroma class entirely in narrow parent bands."""
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

        # Per-bin mask instead of HPSS: flux above median + std is percussive, quieter bins are residual, the rest harmonic.
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
