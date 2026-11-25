"""Source task: file and live audio blocks."""

import math
import threading
import time
from collections import deque
from typing import Dict, Any, Optional, Tuple
import numpy as np
from .filters import NUMBA_AVAILABLE
from ..audio_influx import AudioReception, build_live_flow_identifier
from ... import utils
from .analyzer import AudioFluxAnalyzer
from .base import ACEAudioTask
from .block_size import _ADAPTIVE_FILE_BLOCK_BUCKETS, _ADAPTIVE_LIVE_BLOCK_BUCKETS, _resolve_block_size_default
from .chunk import AudioChunk, MAX_BLOCK_SIZE, MIN_BLOCK_SIZE, SAMPLE_RATE, _make_silence_chunk
from .filters import _iir1_filter_block
from .frontend import SourceFrontendProcessor


# Per-filepath open locks: prevents simultaneous AudioReception.open() calls on the
# same file from different threads (SourceSpecsTask BPM thread + SourceTask prefetch),
# which on Windows can hit exclusive-lock failures.
_FILE_OPEN_LOCKS: Dict[str, threading.Lock] = {}


_FILE_OPEN_LOCKS_MUTEX = threading.Lock()


_FILE_OPEN_LOCKS_MAX = 128  # Max dict entries before evicting idle (unlocked) entries.


def _get_file_open_lock(filepath: str) -> threading.Lock:
    """Return (creating if needed) a per-filepath Lock to serialize open() calls.

    When the dict exceeds _FILE_OPEN_LOCKS_MAX entries, idle (unlocked) locks are
    evicted to prevent unbounded growth in long sessions with many different files.
    """
    with _FILE_OPEN_LOCKS_MUTEX:
        lock = _FILE_OPEN_LOCKS.get(filepath)
        if lock is None:
            lock = threading.Lock()
            _FILE_OPEN_LOCKS[filepath] = lock
        if len(_FILE_OPEN_LOCKS) > _FILE_OPEN_LOCKS_MAX:
            # Evict only locks not currently held to avoid disrupting active I/O.
            stale = [k for k, v in _FILE_OPEN_LOCKS.items() if k != filepath and not v.locked()]
            for k in stale[: max(len(stale) // 2, 1)]:
                _FILE_OPEN_LOCKS.pop(k, None)
        return lock


def _clear_file_open_locks():
    """Remove all idle filepath locks. Called on engine shutdown to reclaim memory."""
    with _FILE_OPEN_LOCKS_MUTEX:
        idle = [k for k, v in _FILE_OPEN_LOCKS.items() if not v.locked()]
        for k in idle:
            _FILE_OPEN_LOCKS.pop(k, None)


class SourceTask(ACEAudioTask):
    """Audio source bridge task — supplies raw audio chunks to downstream nodes.

    Holds a reference to an ``AudioReception`` instance that owns the actual
    I/O (file prefetch or live ring buffer).  Each ``process()`` call returns
    the current audio chunk together with lightweight metadata (sample rate,
    latency estimate) for downstream tasks.
    """
    def __init__(self, node_id, params, block_size):
        super().__init__(node_id, params, block_size)
        self.read_cursor = 0
        self.reader_path: Optional[str] = None
        self.reader_channels = 1
        self.sample_rate = SAMPLE_RATE
        self.analysis: Optional[AudioFluxAnalyzer] = None
        self._frontend_processor = SourceFrontendProcessor()
        self.initialized = False
        self._env_state = 0.0
        self.reception: Optional[AudioReception] = None
        self._last_seek_sample = -1
        self._hpf_state = {"x1": 0.0, "y1": 0.0}
        self._lpf_state = {"x1": 0.0, "y1": 0.0}
        self._frontend_buffer: Optional[np.ndarray] = None
        self._env_buffer: Optional[np.ndarray] = None
        self._iir_prev_input: Optional[np.ndarray] = None
        self._iir_u_buffer: Optional[np.ndarray] = None
        self._iir_pow_buffer: Optional[np.ndarray] = None
        self._iir_accum_buffer: Optional[np.ndarray] = None
        self._frontend_index_buffer: Optional[np.ndarray] = None
        self._workspace_pool: Dict[int, Dict[str, np.ndarray]] = {}
        self._last_live_retry_ts: float = 0.0
        self.engine_settings = None
        self._live_block_buffer: deque[Tuple[np.ndarray, Optional[np.ndarray], int, int, int]] = deque(maxlen=16)
        self._last_frame_request: Optional[int] = None
        self._live_clock_anchor_frame: Optional[int] = None
        self._live_clock_anchor_sample: Optional[float] = None
        self._live_clock_last_target_sample: Optional[int] = None
        self._filter_cache = {
            "sr": None,
            "freq_min": None,
            "freq_max": None,
            "attack_ms": None,
            "release_ms": None,
            "hpf_coeffs": None,
            "lpf_coeffs": None,
            "attack_c": 0.0,
            "release_c": 0.0,
        }
        self.live_ready: bool = True
        self._last_config_signature: Optional[Tuple[Any, ...]] = None
        self.source_ready: bool = True
        self.last_error: str = ""
        # Pending bpy PropertyGroup writes deferred from non-main threads.
        self._pending_ctx_writes: dict[str, Any] = {}

        # --- Prefetch double-buffer (FILE sources only) ---
        # A daemon thread pre-reads the next sequential block while the main
        # thread processes the current one, eliminating file I/O from the hot path.
        self._prefetch_enabled: bool = False
        self._prefetch_chunk: Optional[Any] = None
        self._prefetch_for_frame: Optional[int] = None
        self._prefetch_target_frame: Optional[int] = None
        self._prefetch_ready: threading.Event = threading.Event()
        self._prefetch_trigger: threading.Event = threading.Event()
        self._prefetch_stop: threading.Event = threading.Event()
        self._prefetch_thread: Optional[threading.Thread] = None

    def _start_prefetch_thread(self):
        """Start the background pre-reader thread (FILE sources only)."""
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            return
        self._prefetch_stop.clear()
        self._prefetch_ready.clear()
        self._prefetch_enabled = True
        t = threading.Thread(
            target=self._prefetch_loop,
            daemon=True,
            name=f"ACEPrefetch-{str(self.node_id)[:20]}",
        )
        self._prefetch_thread = t
        t.start()

    def _stop_prefetch_thread(self):
        """Stop the background pre-reader thread and drain its state."""
        self._prefetch_enabled = False
        self._prefetch_stop.set()
        self._prefetch_trigger.set()   # Unblock a waiting thread
        t = self._prefetch_thread
        self._prefetch_thread = None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=1.0)
        self._prefetch_ready.clear()
        self._prefetch_chunk = None
        self._prefetch_for_frame = None

    def _prefetch_loop(self):
        """
        Worker body: waits for a trigger, reads one block via _read_next_block_sync,
        stores it, and signals ready.  Only active for FILE sources.
        """
        while not self._prefetch_stop.is_set():
            self._prefetch_trigger.wait(timeout=0.5)
            if self._prefetch_stop.is_set():
                break
            self._prefetch_trigger.clear()
            target_frame = self._prefetch_target_frame
            if target_frame is None:
                continue
            # Temporarily set the frame request so _read_next_block_sync uses
            # the correct timestamp for cursor alignment.
            old_request = getattr(self, "_current_frame_request", None)
            self._current_frame_request = target_frame
            try:
                chunk = self._read_next_block_sync(track_perf_stage=False)
            except Exception as exc:
                utils.note_swallowed("audio_tasks_registry.SourceTask._prefetch_loop", exc)
                chunk = None
            finally:
                # Only restore the previous value if the main thread hasn't
                # written a newer request in the meantime; avoids clobbering it.
                if self._current_frame_request == target_frame:
                    self._current_frame_request = old_request
            self._prefetch_chunk = chunk
            self._prefetch_for_frame = target_frame
            self._prefetch_ready.set()

    def process(self, inputs):
        self._flush_pending_ctx_writes()   # drain RNA writes from prefetch thread
        chunk = self._read_next_block()
        if chunk:
            return chunk
        if not self.source_ready:
            # Return a zero-filled chunk rather than None so Stage-B workers always
            # receive a well-formed input.  The UI reads ace_source_error for detail.
            return _make_silence_chunk(self.block_size, self.sample_rate)
        return self.silence_buffer

    def _flush_pending_ctx_writes(self) -> None:
        """Apply deferred PropertyGroup writes. Call only from the main thread."""
        if not self._pending_ctx_writes:
            return
        engine = getattr(self, "engine_ref", None)
        engine_generation = int(getattr(engine, "undo_generation", getattr(self, "undo_generation", 0)) or 0)
        task_generation = int(getattr(self, "undo_generation", engine_generation) or engine_generation)
        if task_generation != engine_generation:
            self._pending_ctx_writes.clear()
            return
        ctx = self.source_context
        if ctx is None:
            self._pending_ctx_writes.clear()
            return
        for attr, val in list(self._pending_ctx_writes.items()):
            with utils.logged_errors("audio_tasks_registry.SourceTask._flush_pending_ctx_writes"):
                setattr(ctx, attr, val)
        self._pending_ctx_writes.clear()

    def _timeslice_mode_hint(self) -> str:
        settings = getattr(self, "engine_settings", None)
        try:
            mode = str(getattr(settings, "ace_timeslice_mode", "fixed") or "fixed").strip().lower()
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.SourceTask._timeslice_mode_hint", exc)
            mode = "fixed"
        if mode not in {"fixed", "adaptive"}:
            mode = "fixed"
        return mode

    def _config_signature(self) -> Tuple[Any, ...]:
        ctx = self.source_context
        source_type = str(getattr(ctx, "source_type", "") or "").strip().upper() if ctx else ""
        filepath = str(getattr(ctx, "filepath", "") or "") if ctx else ""
        flow_id = build_live_flow_identifier(ctx) if ctx else ""
        live_latency_ms = int(getattr(ctx, "live_latency_ms", 10) or 10) if ctx else 10
        requested_sr = int(getattr(ctx, "live_udp_sample_rate", 0) or 0) if ctx else 0
        live_transport = str(getattr(ctx, "live_transport", "") or "").strip().upper() if ctx else ""
        live_device = str(getattr(ctx, "live_device_selector", "") or "") if ctx else ""
        live_udp_port = int(getattr(ctx, "live_udp_port", 0) or 0) if ctx else 0
        live_udp_channels = int(getattr(ctx, "live_udp_channels", 0) or 0) if ctx else 0
        return (
            source_type,
            filepath,
            flow_id,
            live_latency_ms,
            requested_sr,
            live_transport,
            live_device,
            live_udp_port,
            live_udp_channels,
        )

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

    def _set_block_size_from_engine(self, new_size: int):
        size = max(int(new_size), 1)
        if size == int(self.block_size):
            self._adopt_workspace_for_size(size)
            return
        if self.analysis:
            AudioFluxAnalyzer.release(self.analysis)
        self.block_size = size
        self._adopt_workspace_for_size(size)
        self._frontend_processor.reset()
        if self.source_ready:
            self.analysis = AudioFluxAnalyzer.borrow(self.sample_rate, size)
        else:
            self.analysis = None
        self._reset_sample_clocks()

    def _reopen_for_config_change(self):
        if self.reception is not None:
            with utils.logged_errors("audio_tasks_registry.SourceTask._reopen_for_config_change"):
                self.reception.close()
        self.reception = None
        if self.analysis is not None:
            with utils.logged_errors("audio_tasks_registry.SourceTask._reopen_for_config_change#2"):
                AudioFluxAnalyzer.release(self.analysis)
        self.analysis = None
        self._reset_sample_clocks()
        self.cached_output = None
        self.last_frame_processed = -1
        self.initialized = False
        self._frontend_processor.reset()
        self.source_ready = True
        self.live_ready = True
        self.last_error = ""

    def _load_source(self):
        """Load source audio into memory and prepare analyzers."""
        self.initialized = True
        ctx = self.source_context

        source_type = getattr(ctx, "source_type", "") if ctx else ""
        filepath = getattr(ctx, "filepath", "") if ctx else ""
        flow_id = build_live_flow_identifier(ctx) if ctx else ""
        if ctx is not None:
            self._pending_ctx_writes["flow_identifier"] = flow_id
        live_latency_ms = int(getattr(ctx, "live_latency_ms", 10) or 10) if ctx else 10
        target_sample_rate = None
        engine = getattr(self, "engine_ref", None)
        target_sr_resolver = getattr(getattr(engine, "compiler", None), "_source_task_target_sample_rate", None)
        if callable(target_sr_resolver):
            try:
                target_sample_rate = target_sr_resolver(self)
            except Exception as exc:
                utils.note_swallowed("audio_tasks_registry.SourceTask._load_source", exc)
                target_sample_rate = None

        self.reader_path = filepath
        self.reception = AudioReception(source_type, filepath, flow_id, live_latency_ms=live_latency_ms)
        opened = self.reception.open(
            block_size=self.block_size,
            target_sample_rate=target_sample_rate,
        )
        self._live_block_buffer.clear()
        if opened:
            self.reader_channels = self.reception.channels
            self.sample_rate = self.reception.sample_rate
            self.source_ready = True
            self.last_error = ""
            if ctx and getattr(ctx, "source_type", "") == "FILE":
                self._pending_ctx_writes["ace_source_error"] = ""
            if ctx and getattr(ctx, "source_type", "") == "LIVE":
                self._pending_ctx_writes["live_status"] = ""
            self.live_ready = True
        else:
            self.sample_rate = SAMPLE_RATE
            self.source_ready = False
            rec = self.reception
            self.last_error = (
                rec.format_error_summary()
                if rec is not None and hasattr(rec, "format_error_summary")
                else (getattr(rec, "last_error", "") if rec is not None else "Source open failed")
            ) or "Source open failed"
            if ctx and getattr(ctx, "source_type", "") == "LIVE":
                self._pending_ctx_writes["live_status"] = (
                    getattr(self.reception, "last_error", "") if self.reception else "Live input unavailable"
                )
            self.reception = None
            if ctx and getattr(ctx, "source_type", "") == "LIVE":
                self.live_ready = False
            if ctx and getattr(ctx, "source_type", "") == "FILE":
                self._pending_ctx_writes["ace_source_error"] = self.last_error

        # Recompute block size now that the real sample rate is known
        resolve_fn = self.resolve_block_size_fn or _resolve_block_size_default
        try:
            delta_frames = float(getattr(getattr(self, "engine_ref", None), "_current_delta_frames", 1.0) or 1.0)
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.SourceTask._load_source#2", exc)
            delta_frames = 1.0
        timeslice_mode = self._timeslice_mode_hint()
        try:
            new_block_size = int(
                resolve_fn(
                    self.sample_rate,
                    self.scene_fps,
                    source_type=source_type,
                    timeslice_mode=timeslice_mode,
                    delta_frames=delta_frames,
                )
            )
        except TypeError:
            new_block_size = int(resolve_fn(self.sample_rate, self.scene_fps))
        self._apply_block_size(new_block_size)
        self._reset_sample_clocks()

        if ctx:
            with utils.logged_errors("audio_tasks_registry.SourceTask._load_source#3"):
                duration_val = float(getattr(self.reception, "duration", 0.0) or 0.0)
                native_sample_rate = int(getattr(self.reception, "native_sample_rate", 0) or 0)
                if native_sample_rate <= 0:
                    native_sample_rate = int(self.sample_rate)
                self._pending_ctx_writes["sample_rate"] = int(self.sample_rate)
                self._pending_ctx_writes["native_sample_rate"] = int(native_sample_rate)
                self._pending_ctx_writes["duration"] = duration_val
                if hasattr(ctx, "live_status") and opened:
                    self._pending_ctx_writes["live_status"] = ""

        if not self.source_ready:
            self.analysis = None
            self._last_config_signature = self._config_signature()
            self._last_live_retry_ts = time.time()
            return

        self.analysis = AudioFluxAnalyzer.borrow(self.sample_rate, self.block_size)
        self._last_config_signature = self._config_signature()
        self._hpf_state = {"x1": 0.0, "y1": 0.0}
        self._lpf_state = {"x1": 0.0, "y1": 0.0}
        self._env_state = 0.0
        self._frontend_processor.reset()
        self._adopt_workspace_for_size(self.block_size)
        self._ensure_frontend_workspace(self.block_size)
        self._last_live_retry_ts = time.time()

    def _apply_block_size(self, new_size: int):
        """Update local buffers and ask the engine to retarget the graph size."""
        if new_size <= 0 or new_size == self.block_size:
            return
        self._set_block_size_from_engine(int(new_size))
        engine = getattr(self, "engine_ref", None)
        if engine and hasattr(getattr(engine, "frame", None), "adjust_graph_block_size_for_task"):
            engine.frame.adjust_graph_block_size_for_task(self, new_size)

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

    def _reset_sample_clocks(self):
        self.read_cursor = 0
        self._last_seek_sample = -1
        self._last_frame_request = None
        self._live_clock_anchor_frame = None
        self._live_clock_anchor_sample = None
        self._live_clock_last_target_sample = None
        self._live_block_buffer.clear()

    def _requested_frame(self) -> int:
        try:
            return int(getattr(self, "_current_frame_request", 0) or 0)
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.SourceTask._requested_frame", exc)
            return 0

    def _samples_per_frame(self, sample_rate: int) -> float:
        sr = max(float(sample_rate), 1.0)
        fps = float(self.scene_fps) if float(self.scene_fps) > 0.0 else 24.0
        return sr / max(fps, 1e-6)

    def _file_sample_clock_target(self, frame_index: int) -> int:
        frame_start = int(getattr(self, "scene_frame_start", 1) or 1)
        frame_offset = max(int(frame_index) - frame_start, 0)
        target = float(frame_offset) * self._samples_per_frame(self.sample_rate)
        return max(int(round(target)), 0)

    def _live_sample_clock_target(self, frame_index: int, latest_capture_sample: int) -> int:
        sr = max(int(self.sample_rate), 1)
        spf = self._samples_per_frame(sr)
        live_latency_ms = float(getattr(self.source_context, "live_latency_ms", 0.0) or 0.0) if self.source_context else 0.0
        latency_samples = (live_latency_ms / 1000.0) * float(sr)
        desired_target = float(latest_capture_sample) - latency_samples

        if self._live_clock_anchor_frame is None or self._live_clock_anchor_sample is None:
            self._live_clock_anchor_frame = int(frame_index)
            self._live_clock_anchor_sample = float(desired_target)
        else:
            elapsed = float(int(frame_index) - int(self._live_clock_anchor_frame))
            predicted = float(self._live_clock_anchor_sample) + (elapsed * spf)
            error = desired_target - predicted

            # Hard realign after large jumps/stalls to avoid stale frame-to-sample mapping.
            if abs(error) > float(self.block_size * 8):
                self._live_clock_anchor_frame = int(frame_index)
                self._live_clock_anchor_sample = float(desired_target)
            else:
                # Soft phase correction keeps A/V alignment stable without visible jitter.
                max_correction = float(self.block_size) * 0.5
                correction = max(-max_correction, min(max_correction, error * 0.2))
                self._live_clock_anchor_sample = float(self._live_clock_anchor_sample) + correction

        elapsed = float(int(frame_index) - int(self._live_clock_anchor_frame or frame_index))
        target = float(self._live_clock_anchor_sample or 0.0) + (elapsed * spf)
        target_i = int(round(target))
        if self._live_clock_last_target_sample is not None:
            target_i = max(target_i, int(self._live_clock_last_target_sample))
        self._live_clock_last_target_sample = int(target_i)
        return max(target_i, 0)

    def _push_live_block(
        self,
        mono: np.ndarray,
        stereo: Optional[np.ndarray],
        sample_rate: int,
        capture_end_sample: int,
        capture_span_samples: int,
    ):
        cap_end = max(int(capture_end_sample), 0)
        cap_span = max(int(capture_span_samples), 1)
        mono_arr = np.asarray(mono, dtype=np.float32).reshape(-1)
        stereo_arr = None
        if isinstance(stereo, np.ndarray):
            stereo_arr = np.asarray(stereo, dtype=np.float32)
        entry = (
            mono_arr,
            stereo_arr,
            int(sample_rate),
            cap_end,
            cap_span,
        )
        if self._live_block_buffer and int(self._live_block_buffer[-1][3]) == cap_end:
            self._live_block_buffer[-1] = entry
            return
        self._live_block_buffer.append(entry)

    def _select_live_block_for_target(self, target_sample: int) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], int]]:
        if not self._live_block_buffer:
            return None
        best_entry = self._live_block_buffer[-1]
        best_distance = float("inf")
        target = int(target_sample)
        for entry in reversed(self._live_block_buffer):
            _, _, _, cap_end, cap_span = entry
            block_end = int(cap_end)
            block_start = max(block_end - int(cap_span), 0)
            if target < block_start:
                distance = float(block_start - target)
            elif target > block_end:
                distance = float(target - block_end)
            else:
                distance = 0.0
            if distance < best_distance:
                best_distance = distance
                best_entry = entry
                if distance <= 0.0:
                    break
        mono, stereo, sr, _end, _span = best_entry
        return mono, stereo, int(sr)

    def _sync_reader_cursor(self, desired_sample: int, force: bool = False):
        if self.reception is None or self.reception.reader is None:
            return
        if desired_sample < 0:
            return
        try:
            current_cursor = int(getattr(self.reception, "_cursor_sample", self.read_cursor) or self.read_cursor)
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.SourceTask._sync_reader_cursor", exc)
            current_cursor = int(self.read_cursor)
        self.read_cursor = current_cursor
        delta = int(desired_sample) - int(current_cursor)
        if delta == 0:
            return
        if not force:
            tolerance = max(8, int(self.block_size) // 4)
            if abs(delta) <= tolerance:
                return
            if delta >= 0 and delta <= self.block_size * 2:
                return
        try:
            if self.reception.seek(int(desired_sample)):
                self.read_cursor = int(desired_sample)
                self._last_seek_sample = int(desired_sample)
        except Exception as e:
            utils.log(f"Source cursor sync failed: {e}", "ERROR")

    def _read_next_block(self) -> Optional["AudioChunk"]:
        """
        Main-thread entry point for reading the next audio block.

        For FILE sources with an active prefetch thread: returns the pre-read
        chunk immediately (zero I/O) and triggers the next speculative read.
        Falls back to _read_next_block_sync on scrub/seek or first call.
        For LIVE sources: delegates directly to _read_next_block_sync.
        """
        source_type = str(
            getattr(getattr(self, "source_context", None), "source_type", "") or ""
        ).upper()
        frame_request = getattr(self, "_current_frame_request", None)

        if self._prefetch_enabled and source_type == "FILE" and frame_request is not None:
            if self._prefetch_ready.is_set() and self._prefetch_for_frame == frame_request:
                # Cache hit: take pre-read chunk, clear slot, trigger next read.
                self._prefetch_ready.clear()
                chunk = self._prefetch_chunk
                self._prefetch_chunk = None
                self._prefetch_target_frame = frame_request + 1
                self._prefetch_trigger.set()
                return chunk
            else:
                # Cache miss (scrub or first frame): stop prefetch, do sync read,
                # then restart prefetch for next sequential frame.
                self._stop_prefetch_thread()
                chunk = self._read_next_block_sync()
                if chunk is not None:
                    self._prefetch_target_frame = (
                        (frame_request + 1) if frame_request is not None else None
                    )
                    self._start_prefetch_thread()
                    self._prefetch_trigger.set()
                return chunk

        return self._read_next_block_sync()

    def _read_next_block_sync(self, track_perf_stage: bool = True) -> Optional["AudioChunk"]:
        """Synchronous block read.  May perform file I/O or queue dequeue."""
        t0 = time.perf_counter()
        try:
            current_signature = self._config_signature()
            if self.initialized and self._last_config_signature is not None and current_signature != self._last_config_signature:
                # Source IO params changed at runtime: reopen cleanly so readers/analyzers follow.
                self._reopen_for_config_change()
            if not self.initialized:
                self._load_source()
            if self._last_config_signature is None:
                self._last_config_signature = current_signature

            source_type = str(getattr(self.source_context, "source_type", "") or "").upper() if self.source_context else ""
            if source_type != "LIVE" and not self.source_ready:
                return None

            # Arm prefetch thread on the first successful FILE source open.
            if source_type == "FILE" and not self._prefetch_enabled:
                self._start_prefetch_thread()

            resolve_fn = self.resolve_block_size_fn or _resolve_block_size_default
            try:
                delta_frames = float(getattr(getattr(self, "engine_ref", None), "_current_delta_frames", 1.0) or 1.0)
            except Exception as exc:
                utils.note_swallowed("audio_tasks_registry.SourceTask._read_next_block_sync", exc)
                delta_frames = 1.0
            timeslice_mode = self._timeslice_mode_hint()
            try:
                target_block = int(
                    resolve_fn(
                        self.sample_rate,
                        self.scene_fps,
                        source_type=source_type,
                        timeslice_mode=timeslice_mode,
                        delta_frames=delta_frames,
                    )
                )
            except TypeError:
                try:
                    target_block = int(resolve_fn(self.sample_rate, self.scene_fps, source_type=source_type))
                except TypeError:
                    target_block = int(resolve_fn(self.sample_rate, self.scene_fps))
            target_block = max(MIN_BLOCK_SIZE, min(target_block, MAX_BLOCK_SIZE))
            if target_block != self.block_size:
                self._apply_block_size(target_block)

            if source_type == "LIVE":
                if (self.reception is None or getattr(self.reception, "kind", "") == "live_stub"):
                    now = time.time()
                    if hasattr(self.source_context, "live_status"):
                        message = (
                            (getattr(self.reception, "last_error", "") if self.reception else "")
                            or str(getattr(self, "last_error", "") or "")
                            or str(getattr(self.source_context, "live_status", "") or "")
                            or "Live input unavailable"
                        )
                        self.source_context.live_status = message or "Live input unavailable"
                    if (now - self._last_live_retry_ts) > 2.0:
                        self._load_source()
                    self.live_ready = False
                    return None
                elif hasattr(self.source_context, "live_status") and getattr(self.source_context, "live_status", ""):
                    # Stream recovered, clear any stale warning
                    self.source_context.live_status = ""
                    self.live_ready = True

            mono = None
            stereo = None
            frame_request = self._requested_frame()

            if source_type == "LIVE" and self._last_frame_request is not None:
                frame_jump = int(frame_request) - int(self._last_frame_request)
                if frame_jump <= -1 or frame_jump > 3:
                    self._live_clock_anchor_frame = None
                    self._live_clock_anchor_sample = None
                    self._live_clock_last_target_sample = None

            if self.reception:
                if source_type == "LIVE":
                    block = self.reception.read_block(self.block_size)
                    if block is not None:
                        mono_live, stereo_live, sr_live = block
                        self.sample_rate = int(sr_live)
                        self.live_ready = True
                        capture_end = int(getattr(self.reception, "_live_last_capture_sample", 0) or 0)
                        capture_span = int(
                            getattr(self.reception, "_live_last_capture_block_samples", mono_live.shape[-1]) or mono_live.shape[-1]
                        )
                        self._push_live_block(mono_live, stereo_live, sr_live, capture_end, capture_span)
                    elif getattr(self.reception, "_eof", False):
                        return None

                    if self._live_block_buffer:
                        latest_capture = int(self._live_block_buffer[-1][3])
                        live_target = self._live_sample_clock_target(frame_request, latest_capture)
                        selected = self._select_live_block_for_target(live_target)
                        if selected is not None:
                            mono, stereo, sr = selected
                            self.sample_rate = int(sr)
                            self.read_cursor = int(live_target)
                            self.live_ready = True
                        # If selected is None, retain current live_ready so the engine
                        # keeps calling process() and the buffer can fill on the next frame.
                    # If buffer is empty (stream just opened, first callback not yet fired),
                    # retain live_ready = True so the engine does not deadlock by skipping
                    # process() indefinitely before any audio arrives.
                else:
                    desired_sample = self._file_sample_clock_target(frame_request)
                    file_force_seek = False
                    if self._last_frame_request is None:
                        file_force_seek = True
                    else:
                        frame_jump = int(frame_request) - int(self._last_frame_request)
                        # Hard-seek only on scrubs/backward jumps or large frame skips.
                        if frame_jump <= 0 or frame_jump > 3:
                            file_force_seek = True
                    self._sync_reader_cursor(desired_sample, force=file_force_seek)
                    block = self.reception.read_block(self.block_size)
                    if block is not None:
                        mono, stereo, sr = block
                        self.sample_rate = int(sr)
                        try:
                            self.read_cursor = int(
                                getattr(self.reception, "_cursor_sample", desired_sample + int(mono.shape[-1]))
                                or (desired_sample + int(mono.shape[-1]))
                            )
                        except Exception as exc:
                            utils.note_swallowed("audio_tasks_registry.SourceTask._read_next_block_sync#2", exc)
                            self.read_cursor = desired_sample + int(mono.shape[-1])
                    elif getattr(self.reception, "_eof", False):
                        return None

            self._last_frame_request = int(frame_request)

            live_kind = str(getattr(self.reception, "kind", "")) if self.reception else ""
            if source_type == "LIVE" and self.reception and live_kind.endswith("sounddevice"):
                # Skip extra processing when the live queue has not delivered a block yet.
                if mono is None:
                    return None

            if mono is None:
                mono = np.copy(self.silence_buffer)

            mono_raw = np.asarray(mono, dtype=np.float32).reshape(-1).astype(np.float32, copy=False)
            stereo_raw = self._coerce_stereo_block(stereo, mono_raw.shape[0])
            analysis_mono = None
            analysis_stereo = None
            frontend_config = self._frontend_runtime_config()
            if self._should_materialize_local_frontend():
                analysis_mono, analysis_stereo, frontend_config = self._analysis_frontend_tracks(
                    mono_raw,
                    stereo_raw,
                )
            live_latency_ms = float(getattr(self.source_context, "live_latency_ms", 0.0) or 0.0) if source_type == "LIVE" else 0.0
            sr = max(float(self.sample_rate), 1.0)
            block_latency_ms = (float(self.block_size) / sr) * 1000.0
            total_latency_ms = float(max(0.0, live_latency_ms + block_latency_ms))
            return AudioChunk(
                mono_raw,
                stereo_raw,
                self.sample_rate,
                provider=self,
                analysis_mono=analysis_mono,
                analysis_stereo=analysis_stereo,
                frontend_config=frontend_config,
                lineage=(self.node_id,),
                latency_ms=total_latency_ms,
            )
        finally:
            engine = getattr(self, "engine_ref", None)
            elapsed = time.perf_counter() - t0
            if bool(track_perf_stage):
                perf_add = getattr(getattr(engine, "diagnostics", None), "_perf_add", None)
                if callable(perf_add):
                    with utils.logged_errors("audio_tasks_registry.SourceTask._read_next_block_sync#3"):
                        perf_add("source_read", elapsed)
            else:
                prefetch_add = getattr(getattr(engine, "diagnostics", None), "_record_source_prefetch", None)
                if callable(prefetch_add):
                    with utils.logged_errors("audio_tasks_registry.SourceTask._read_next_block_sync#4"):
                        prefetch_add(elapsed)

    def compute_feature(self, name: str, chunk: AudioChunk):
        if not self.analysis:
            return None
        return self.analysis.compute_feature(name, chunk)

    def close(self):
        # Stop prefetch thread before closing reception so it cannot call
        # reception.read_block() after the underlying resource is freed.
        self._stop_prefetch_thread()
        if self.reception:
            with utils.logged_errors("audio_tasks_registry.SourceTask.close"):
                self.reception.close()
            self.reception = None
        if self.analysis:
            AudioFluxAnalyzer.release(self.analysis)
            self.analysis = None
        self._last_config_signature = None
        self._reset_sample_clocks()
        self.live_ready = True
        self.source_ready = True
        self.last_error = ""
        self._frontend_processor.reset()
        self._workspace_pool.clear()
        self.initialized = False
