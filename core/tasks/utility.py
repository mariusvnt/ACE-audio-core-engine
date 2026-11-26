"""Utility tasks: rolling average and accumulator."""

from collections import deque
from typing import Dict, Any, Optional
import numpy as np
from ... import utils
from .base import ACEAudioTask
from .chunk import _finite_copy


class RollingAvgTask(ACEAudioTask):
    """Computes a rolling (moving) average of a scalar attribute stream."""

    _DEFAULT_WINDOW_S = 0.25

    def __init__(self, node_id: str, parameters: Dict[str, Any], block_size: int):
        super().__init__(node_id, parameters, block_size)
        self._samples: deque = deque()  # (frame, value)
        self._current_frame_request: Optional[int] = None

    def process(self, inputs):
        raw_value = inputs.get(0)
        if raw_value is None:
            raw_value = self.parameters.get("in_0", 0.0)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = 0.0

        raw_window = inputs.get(1)
        if raw_window is None:
            raw_window = self.parameters.get("in_1", self._DEFAULT_WINDOW_S)
        try:
            window_s = max(float(raw_window), 0.0)
        except (TypeError, ValueError):
            window_s = self._DEFAULT_WINDOW_S
        fps = max(float(getattr(self.engine_ref, "_current_scene_fps", 0.0) or self.scene_fps or 24.0), 1.0)
        last = self._samples[-1][0] if self._samples else None
        frame = self._current_frame_request
        if frame is None:
            frame = 0 if last is None else last + 1
        if last is not None and frame <= last:
            if frame < last:
                self._samples.clear()  # the timeline was rewound
            else:
                self._samples.pop()  # the same frame again
        self._samples.append((frame, value))
        # Samples younger than the window; the current one always counts.
        while len(self._samples) > 1 and (frame - self._samples[0][0]) / fps >= window_s - 1e-6:
            self._samples.popleft()
        return float(np.mean([sample for _frame, sample in self._samples]))


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
