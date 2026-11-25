"""Source-side preprocessing: filter workspaces and the high/low-pass IIR."""

import math
from typing import Dict, Any, Optional, Tuple
import numpy as np
from .filters import NUMBA_AVAILABLE
from ... import utils
from .block_size import _ADAPTIVE_FILE_BLOCK_BUCKETS, _ADAPTIVE_LIVE_BLOCK_BUCKETS
from .chunk import MAX_BLOCK_SIZE, MIN_BLOCK_SIZE
from .filters import _iir1_filter_block


class SourceFrontendStage:
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
        resolver = getattr(getattr(engine, "parallel", None), "source_task_requires_local_frontend", None)
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
