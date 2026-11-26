"""Stream tasks: per-frame numeric features."""

import math
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from .filters import NUMBA_AVAILABLE, SCIPY_AVAILABLE, scipy_lfilter
from ... import utils
from ...vars import LEGACY_STREAM_TYPES, LOUDNESS_FLOOR_DB, resolve_node_mode
from .base import ACEAudioTask
from .chunk import AudioChunk, SAMPLE_RATE, _ensure_audio_chunk
from .filters import _biquad_filter_block
from .spectral import _analysis_signals_normalized, _band_rms, _chunk_pitch, _masked_mono_signal, _masked_spectrum, _safe_rms, _zero_crossing_rate


def _loudness_db_to_unit(db_value: float) -> float:
    """Map a level in dB to 0..1, linear in dB: 0 at LOUDNESS_FLOOR_DB and below, 1 at 0 dB and above."""
    db = utils.as_float(db_value, float("-inf"))
    if not np.isfinite(db) or db <= LOUDNESS_FLOOR_DB:
        return 0.0
    if db >= 0.0:
        return 1.0
    return float((db - LOUDNESS_FLOOR_DB) / (0.0 - LOUDNESS_FLOOR_DB))


class RMSTask(ACEAudioTask):
    """Compute Root-Mean-Square energy of the audio chunk."""

    def process(self, inputs):
        chunk = _ensure_audio_chunk(inputs.get(0), self.block_size)
        if not hasattr(chunk, "mono") or chunk.mono.size == 0:
            return 0.0
        analysis_mono, _ = _analysis_signals_normalized(chunk)
        if analysis_mono.size == 0:
            return 0.0

        # Unmasked streams: pure time-domain RMS. A band is measured in the spectrum.
        if getattr(chunk, "band_indices", None) is None and getattr(chunk, "band_hz_bounds", None) is None:
            return _safe_rms(analysis_mono)
        return chunk.get_feature("rms", lambda: _band_rms(chunk))


class RMSLoudnessTask(RMSTask):
    """RMS level on the Loudness scale: dB of the frame RMS, mapped between -70 dB and 0 dB."""

    def process(self, inputs):
        rms = utils.as_float(super().process(inputs), 0.0)
        db = 20.0 * math.log10(rms) if rms > 1e-12 else float("-inf")
        return _loudness_db_to_unit(db)


class LUFSTask(ACEAudioTask):
    """Estimate integrated short-window loudness using K-weighting (LUFS)."""

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
                utils.note_swallowed("stream.LUFSTask._coerce_scalar", exc)
                return float(fallback)
        try:
            return float(value)
        except Exception as exc:
            utils.note_swallowed("stream.LUFSTask._coerce_scalar#2", exc)
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
            with utils.logged_errors("stream.LUFSTask._apply_biquad"):
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
        return _loudness_db_to_unit(db_value)

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
    """Estimate K-weighted peak level."""

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
    """Estimate the fundamental frequency (pitch) of the audio chunk."""

    def process(self, inputs):
        chunk = _ensure_audio_chunk(inputs.get(0), self.block_size)
        return chunk.get_feature("pitch", lambda: _chunk_pitch(chunk))


class SpectralCentroidTask(ACEAudioTask):
    """Compute the spectral centroid (brightness) of the audio chunk."""

    def process(self, inputs):
        chunk = _ensure_audio_chunk(inputs.get(0), self.block_size)
        val = None
        if isinstance(chunk, AudioChunk) and chunk.provider:
            val = chunk.get_feature("centroid", lambda: chunk.provider.compute_feature("centroid", chunk))
        if val is not None:
            with utils.logged_errors("stream.SpectralCentroidTask.process"):
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
    """Measure frame-to-frame change in the power spectrum (spectral flux)."""

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
            with utils.logged_errors("stream.SpectralFluxTask.process"):
                resolved = float(val)
                if np.isfinite(resolved):
                    return resolved

        return self._fallback_flux(chunk)


class OnsetTask(ACEAudioTask):
    """Detect audio onsets (transient events) in the audio chunk."""

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
            with utils.logged_errors("stream.OnsetTask.process"):
                resolved = float(val)
                if np.isfinite(resolved):
                    return resolved

        return self._fallback_onset(chunk)


class ZeroCrossTask(ACEAudioTask):
    """Measure the Zero-Crossing Rate (ZCR) of the audio chunk."""

    def process(self, inputs):
        chunk = _ensure_audio_chunk(inputs.get(0), self.block_size)
        val = None
        if isinstance(chunk, AudioChunk) and chunk.provider:
            val = chunk.get_feature("zcr", lambda: chunk.provider.compute_feature("zcr", chunk))
        if val is not None:
            with utils.logged_errors("stream.ZeroCrossTask.process"):
                resolved = float(val)
                if np.isfinite(resolved):
                    return resolved
        analysis_mono, _ = _analysis_signals_normalized(chunk)
        return _zero_crossing_rate(analysis_mono)


class StereoWidthTask(ACEAudioTask):
    """Measure the stereo width of the audio chunk."""

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
        # side / (mid + side): 0 = mono, 1 = pure side.
        if total_energy < 1e-20:
            return 0.0
        return float(np.clip(side_energy / total_energy, 0.0, 1.0))


class StereoBalanceTask(ACEAudioTask):
    """Measure the left/right energy balance of the audio chunk."""

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
    """Time-domain vector summary of stereo energy and balance."""
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



# Stream node type -> {mode: task class}. Nodes without modes use the "" key.
_STREAM_TASKS: Dict[str, Dict[str, type]] = {
    "loudness": {"rms": RMSLoudnessTask, "lufs": LUFSTask, "peak": PeakTask},
    "stereo": {"width": StereoWidthTask, "balance": StereoBalanceTask},
    "spectral": {"centroid": SpectralCentroidTask, "flux": SpectralFluxTask, "zcr": ZeroCrossTask},
    "stereo_vector": {"": StereoVectorTask},
    "pitch": {"": PitchTask},
    "onset": {"": OnsetTask},
}


def create_stream_task(node_id: str, node_type: str, params: Dict[str, Any], block_size: int) -> Optional[ACEAudioTask]:
    """Task for a stream node, chosen by its type and mode; old types resolve to their mode."""
    typ = str(node_type or "")
    shell_type = LEGACY_STREAM_TYPES.get(typ, (typ, ""))[0]
    by_mode = _STREAM_TASKS.get(shell_type)
    if not by_mode:
        return None
    mode = resolve_node_mode("stream", typ, (params or {}).get("mode"))
    task_cls = by_mode.get(mode) or by_mode.get("") or next(iter(by_mode.values()))
    return task_cls(node_id, params, block_size)
