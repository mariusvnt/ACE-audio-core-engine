"""Source preprocessing: high/low-pass filters and attack/release smoothing."""

import math
from typing import Dict, Optional, Tuple
import numpy as np
from .filters import NUMBA_AVAILABLE
from ... import utils
from .chunk import _finite_copy
from .filters import _iir1_filter_block, _signed_smooth_block


class SourceFrontendProcessor:
    """Stateful source-side preprocessing for analysis-domain audio."""

    def __init__(self):
        self._iir_states: Dict[tuple[str, str], Dict[str, float]] = {}
        self._env_states: Dict[str, float] = {}
        self._config_sig: Optional[Tuple[int, Optional[Tuple[float, float, float, float]], int]] = None
        self._coeff_cache = {
            "hpf": None,
            "lpf": None,
            "attack_c": 0.0,
            "release_c": 0.0,
        }

    def reset(self):
        """Clear all filter states and force coefficient recomputation on the next call."""
        self._iir_states.clear()
        self._env_states.clear()
        self._config_sig = None
        self._coeff_cache = {
            "hpf": None,
            "lpf": None,
            "attack_c": 0.0,
            "release_c": 0.0,
        }

    @staticmethod
    def _butterworth_coeffs(cutoff_hz: float, sample_rate: float, mode: str):
        """Compute 1st-order Butterworth IIR coefficients (b0, b1, a1)."""
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

    def _ensure_config(
        self,
        sample_rate: int,
        frontend_config: Optional[Tuple[float, float, float, float]],
        channel_count: int,
    ):
        """Recompute filter coefficients when the configuration has changed."""
        sig = (
            int(max(sample_rate, 1)),
            tuple(frontend_config) if frontend_config is not None else None,
            int(max(channel_count, 1)),
        )
        if sig == self._config_sig:
            return

        self._config_sig = sig
        self._iir_states.clear()
        self._env_states.clear()

        freq_min = 0.0
        freq_max = 0.0
        attack_ms = 0.0
        release_ms = 0.0
        if frontend_config is not None and len(frontend_config) >= 4:
            try:
                freq_min = float(frontend_config[0] or 0.0)
                freq_max = float(frontend_config[1] or 0.0)
                attack_ms = float(frontend_config[2] or 0.0)
                release_ms = float(frontend_config[3] or 0.0)
            except Exception as exc:
                utils.note_swallowed("frontend.SourceFrontendProcessor._ensure_config", exc)
                freq_min = 0.0
                freq_max = 0.0
                attack_ms = 0.0
                release_ms = 0.0

        sr = float(max(sample_rate, 1))
        dt = 1.0 / sr
        self._coeff_cache["hpf"] = self._butterworth_coeffs(freq_min, sr, "high") if freq_min > 0.0 else None
        self._coeff_cache["lpf"] = self._butterworth_coeffs(freq_max, sr, "low") if (freq_max > 0.0 and freq_max < (sr * 0.5)) else None
        attack_tc = max(attack_ms, 0.0) / 1000.0
        release_tc = max(release_ms, 0.0) / 1000.0
        self._coeff_cache["attack_c"] = math.exp(-dt / max(attack_tc, 1e-6)) if attack_tc > 0.0 else 0.0
        self._coeff_cache["release_c"] = math.exp(-dt / max(release_tc, 1e-6)) if release_tc > 0.0 else 0.0

    def _run_iir(self, data: np.ndarray, coeffs: Optional[Tuple[float, float, float]], state_key: tuple[str, str]) -> np.ndarray:
        src = np.asarray(data, dtype=np.float32).reshape(-1).astype(np.float32, copy=False)
        if coeffs is None or src.size == 0:
            return src

        state = self._iir_states.setdefault(state_key, {"x1": 0.0, "y1": 0.0})
        b0, b1, a1 = coeffs
        x_prev = float(state.get("x1", 0.0))
        y_prev = float(state.get("y1", 0.0))

        if NUMBA_AVAILABLE:
            with utils.logged_errors("frontend.SourceFrontendProcessor._run_iir"):
                out, x_last, y_last = _iir1_filter_block(
                    src,
                    float(b0),
                    float(b1),
                    float(a1),
                    float(x_prev),
                    float(y_prev),
                )
                state["x1"] = float(x_last)
                state["y1"] = float(y_last)
                return out.astype(np.float32, copy=False)

        out = np.empty_like(src, dtype=np.float32)
        x1 = float(x_prev)
        y1 = float(y_prev)
        for idx in range(src.shape[0]):
            x = float(src[idx])
            y = (float(b0) * x) + (float(b1) * x1) - (float(a1) * y1)
            out[idx] = y
            x1 = x
            y1 = y
        state["x1"] = float(x1)
        state["y1"] = float(y1)
        return out

    def _run_envelope(self, data: np.ndarray, attack_c: float, release_c: float, state_key: str) -> np.ndarray:
        src = np.asarray(data, dtype=np.float32).reshape(-1).astype(np.float32, copy=False)
        if src.size == 0:
            return src
        if attack_c <= 0.0 and release_c <= 0.0:
            return src

        env_state = float(self._env_states.get(state_key, 0.0) or 0.0)
        if NUMBA_AVAILABLE:
            with utils.logged_errors("frontend.SourceFrontendProcessor._run_envelope"):
                env_out, env_last = _signed_smooth_block(src, float(attack_c), float(release_c), float(env_state))
                self._env_states[state_key] = float(env_last)
                return env_out.astype(np.float32, copy=False)

        out = np.empty_like(src, dtype=np.float32)
        env = float(env_state)
        for idx in range(src.shape[0]):
            x = float(src[idx])
            coeff = float(attack_c) if abs(x) > abs(env) else float(release_c)
            env = (coeff * env) + ((1.0 - coeff) * x)
            out[idx] = env
        self._env_states[state_key] = float(env)
        return out

    def _process_channel(self, signal: np.ndarray, channel_key: str) -> np.ndarray:
        data = np.asarray(signal, dtype=np.float32).reshape(-1).astype(np.float32, copy=False)
        data = self._run_iir(data, self._coeff_cache.get("hpf"), (channel_key, "hpf"))
        data = self._run_iir(data, self._coeff_cache.get("lpf"), (channel_key, "lpf"))
        data = self._run_envelope(
            data,
            float(self._coeff_cache.get("attack_c", 0.0) or 0.0),
            float(self._coeff_cache.get("release_c", 0.0) or 0.0),
            channel_key,
        )
        return _finite_copy(data)

    def process(
        self,
        mono: np.ndarray,
        stereo: Optional[np.ndarray],
        sample_rate: int,
        frontend_config: Optional[Tuple[float, float, float, float]],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply the full frontend processing chain and return (analysis_mono, analysis_stereo)."""
        mono_arr = np.asarray(mono, dtype=np.float32).reshape(-1).astype(np.float32, copy=False)
        stereo_arr: Optional[np.ndarray] = None
        if isinstance(stereo, np.ndarray):
            arr = np.asarray(stereo, dtype=np.float32)
            if arr.ndim == 2:
                if arr.shape[1] == mono_arr.shape[0]:
                    stereo_arr = arr.astype(np.float32, copy=False)
                elif arr.shape[0] == mono_arr.shape[0]:
                    stereo_arr = arr.T.astype(np.float32, copy=False)

        if frontend_config is None:
            return mono_arr, stereo_arr

        channel_count = int(stereo_arr.shape[0]) if isinstance(stereo_arr, np.ndarray) and stereo_arr.ndim == 2 else 1
        self._ensure_config(sample_rate=sample_rate, frontend_config=frontend_config, channel_count=channel_count)

        if stereo_arr is not None and stereo_arr.shape[1] == mono_arr.shape[0]:
            processed_channels = []
            for ch in range(int(stereo_arr.shape[0])):
                processed_channels.append(self._process_channel(stereo_arr[ch], f"ch{int(ch)}"))
            analysis_stereo = np.stack(processed_channels, axis=0).astype(np.float32, copy=False)
            analysis_mono = np.mean(analysis_stereo, axis=0).astype(np.float32, copy=False)
            return analysis_mono, analysis_stereo

        analysis_mono = self._process_channel(mono_arr, "mono")
        return analysis_mono, None
