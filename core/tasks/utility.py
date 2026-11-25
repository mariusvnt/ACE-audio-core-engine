"""Utility tasks: rolling average, accumulator and audio math."""

from collections import deque
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from ... import utils
from .base import ACEAudioTask
from .chunk import AudioChunk, SAMPLE_RATE, _ensure_audio_chunk, _finite_copy
from .spectral import _aligned_signal_length, _analysis_signals_normalized, _resolve_chunk_fft_size, _resolve_effective_bin_indices


class RollingAvgTask(ACEAudioTask):
    """Computes a rolling (moving) average of a scalar attribute stream."""

    def __init__(self, node_id: str, parameters: Dict[str, Any], block_size: int):
        super().__init__(node_id, parameters, block_size)
        self._buffer: deque = deque(maxlen=1)
        self._buffer_maxlen: int = 1

    def process(self, inputs):
        raw_value = inputs.get(0)
        if raw_value is None:
            raw_value = self.parameters.get("in_0", 0.0)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = 0.0

        raw_frame_count = inputs.get(1)
        if raw_frame_count is None:
            raw_frame_count = self.parameters.get("in_1", 1)
        try:
            frame_count = max(1, int(round(float(raw_frame_count))))
        except (TypeError, ValueError):
            frame_count = 1

        new_maxlen = max(1, int(frame_count))
        if new_maxlen != self._buffer_maxlen:
            # Preserve as many recent samples as fit in the new capacity.
            existing = list(self._buffer)
            self._buffer = deque(existing[-new_maxlen:], maxlen=new_maxlen)
            self._buffer_maxlen = new_maxlen

        self._buffer.append(value)
        return float(np.mean(self._buffer))


class StreamAccumulatorTask(ACEAudioTask):
    """Running sum utility for scalar or vector stream values."""

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
            utils.note_swallowed("utility.StreamAccumulatorTask.export_state", exc)
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
            utils.note_swallowed("utility.StreamAccumulatorTask.import_state", exc)
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
            utils.note_swallowed("utility.StreamAccumulatorTask._coerce_input_array", exc)
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
                utils.note_swallowed("utility.StreamAccumulatorTask._coerce_reset_flag", exc)
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
            utils.note_swallowed("utility.StreamAccumulatorTask.process", exc)
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
    """Element-wise arithmetic over multiple audio streams."""
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
                utils.note_swallowed("utility.AudioMathTask._merge_band_hz_bounds", exc)
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
