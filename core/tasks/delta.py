"""Delta tasks: splitters and masks that constrain downstream analysis."""

from typing import List, Optional, Tuple
import numpy as np
from ... import utils
from .base import ACEAudioTask
from .chunk import SAMPLE_RATE, _ensure_audio_chunk, _spawn_audio_chunk


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
                            history_mono=None,
                            history_stereo=None,
                            provider=None,
                            band_hz_bounds=(low_cut, low_cut),
                            lineage=lineage,
                            spectrum_token=None,
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
                        latency_ms=latency_ms,
                    )
                )
        else:
            bands.append(chunk)

        return tuple(bands)
