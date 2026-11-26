"""Source task: opens the file or live input and serves one block per frame."""

import threading
import time
from collections import deque
from typing import Dict, Any, Optional, Tuple
import numpy as np
from ..input.reception import AudioReception, build_live_flow_identifier
from ... import utils
from .base import ACEAudioTask
from .block_size import _resolve_block_size_default
from .chunk import SAMPLE_RATE, _make_silence_chunk
from .frontend import SourceFrontendProcessor
from .source_clock import SourceClock
from .source_frontend import SourceFrontendStage


# One open per file at a time: the tempo thread and the prefetcher may open the same file (exclusive on Windows).
_FILE_OPEN_LOCKS: Dict[str, threading.Lock] = {}
_FILE_OPEN_LOCKS_MUTEX = threading.Lock()
_FILE_OPEN_LOCKS_MAX = 128  # Max dict entries before evicting idle (unlocked) entries.
def _get_file_open_lock(filepath: str) -> threading.Lock:
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
    with _FILE_OPEN_LOCKS_MUTEX:
        idle = [k for k, v in _FILE_OPEN_LOCKS.items() if not v.locked()]
        for k in idle:
            _FILE_OPEN_LOCKS.pop(k, None)
class SourceTask(SourceFrontendStage, SourceClock, ACEAudioTask):
    """Audio source bridge task — supplies raw audio chunks to downstream nodes."""
    def __init__(self, node_id, params, block_size):
        super().__init__(node_id, params, block_size)
        self.read_cursor = 0
        self.reader_path: Optional[str] = None
        self.reader_channels = 1
        self.sample_rate = SAMPLE_RATE
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
        self._live_history_block: Optional[np.ndarray] = None
        self._live_history_tail: Optional[Tuple[np.ndarray, Optional[np.ndarray]]] = None
        self._live_history_before: Optional[Tuple[np.ndarray, Optional[np.ndarray]]] = None
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
        """Worker body: waits for a trigger, reads one block via _read_next_block_sync, stores it, and signals ready."""
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
                utils.note_swallowed("source.SourceTask._prefetch_loop", exc)
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
        """Apply deferred PropertyGroup writes."""
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
            with utils.logged_errors("source.SourceTask._flush_pending_ctx_writes"):
                setattr(ctx, attr, val)
        self._pending_ctx_writes.clear()

    def _timeslice_mode_hint(self) -> str:
        settings = getattr(self, "engine_settings", None)
        try:
            mode = str(getattr(settings, "ace_timeslice_mode", "fixed") or "fixed").strip().lower()
        except Exception as exc:
            utils.note_swallowed("source.SourceTask._timeslice_mode_hint", exc)
            mode = "fixed"
        if mode not in {"fixed", "adaptive"}:
            mode = "fixed"
        return mode

    def _config_signature(self) -> Tuple[Any, ...]:
        ctx = self.source_context
        source_type = str(getattr(ctx, "source_type", "") or "").strip().upper() if ctx else ""
        filepath = str(getattr(ctx, "filepath", "") or "") if ctx else ""
        flow_id = build_live_flow_identifier(ctx) if ctx else ""
        live_latency_ms = int(getattr(ctx, "live_latency_ms", 5) or 5) if ctx else 5
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

    def _set_block_size_from_engine(self, new_size: int):
        size = max(int(new_size), 1)
        if size == int(self.block_size):
            self._adopt_workspace_for_size(size)
            return
        self.block_size = size
        self._adopt_workspace_for_size(size)
        self._frontend_processor.reset()
        self._reset_sample_clocks()

    def _reopen_for_config_change(self):
        if self.reception is not None:
            with utils.logged_errors("source.SourceTask._reopen_for_config_change"):
                self.reception.close()
        self.reception = None
        self._reset_sample_clocks()
        self.cached_output = None
        self.last_frame_processed = -1
        self.initialized = False
        self._frontend_processor.reset()
        self.source_ready = True
        self.live_ready = True
        self.last_error = ""

    def _load_source(self):
        """Load source audio into memory."""
        self.initialized = True
        ctx = self.source_context

        source_type = getattr(ctx, "source_type", "") if ctx else ""
        filepath = getattr(ctx, "filepath", "") if ctx else ""
        flow_id = build_live_flow_identifier(ctx) if ctx else ""
        if ctx is not None:
            self._pending_ctx_writes["flow_identifier"] = flow_id
        live_latency_ms = int(getattr(ctx, "live_latency_ms", 5) or 5) if ctx else 5
        target_sample_rate = None
        engine = getattr(self, "engine_ref", None)
        target_sr_resolver = getattr(getattr(engine, "compiler", None), "_source_task_target_sample_rate", None)
        if callable(target_sr_resolver):
            try:
                target_sample_rate = target_sr_resolver(self)
            except Exception as exc:
                utils.note_swallowed("source.SourceTask._load_source", exc)
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
            utils.note_swallowed("source.SourceTask._load_source#2", exc)
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
            with utils.logged_errors("source.SourceTask._load_source#3"):
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
            self._last_config_signature = self._config_signature()
            self._last_live_retry_ts = time.time()
            return

        self._last_config_signature = self._config_signature()
        self._hpf_state = {"x1": 0.0, "y1": 0.0}
        self._lpf_state = {"x1": 0.0, "y1": 0.0}
        self._env_state = 0.0
        self._frontend_processor.reset()
        self._adopt_workspace_for_size(self.block_size)
        self._ensure_frontend_workspace(self.block_size)
        self._last_live_retry_ts = time.time()

    def _apply_block_size(self, new_size: int):
        if new_size <= 0 or new_size == self.block_size:
            return
        self._set_block_size_from_engine(int(new_size))
        engine = getattr(self, "engine_ref", None)
        if engine and hasattr(getattr(engine, "frame", None), "adjust_graph_block_size_for_task"):
            engine.frame.adjust_graph_block_size_for_task(self, new_size)

    def close(self):
        # Stop prefetch thread before closing reception so it cannot call
        # reception.read_block() after the underlying resource is freed.
        self._stop_prefetch_thread()
        if self.reception:
            with utils.logged_errors("source.SourceTask.close"):
                self.reception.close()
            self.reception = None
        self._last_config_signature = None
        self._reset_sample_clocks()
        self.live_ready = True
        self.source_ready = True
        self.last_error = ""
        self._frontend_processor.reset()
        self._workspace_pool.clear()
        self.initialized = False
