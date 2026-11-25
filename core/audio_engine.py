"""
Core orchestration layer for the AudioCore Engine Blender add-on.

Responsibilities:
  - ``ACEAudioEngine``: singleton-per-scene orchestrator that owns the
    compile/execute lifecycle, the parallel DSP worker pool, all
    ``ACEAudioGraph`` instances, and the shared preview monitor.
  - Stage-A (main thread): discover all managed scene frontends, parse them,
    merge them into one ``ACEAbstractTree``, and compile one runtime graph per
    source.
  - Stage-B (subprocess workers): bpy-free evaluation delegated via
    ``evaluate_parallel_job`` from ``parallel_dsp``.
  - Stage-C (main thread): ``ACEProxyManager`` commits Stage-B results to
    Blender proxy attributes and object custom properties.
  - Layer-3 preview: ``_ACEPreviewAudioMonitor`` / ``_ACEPreviewFeeder``
    drive a real-time sounddevice output stream fed from the DSP pipeline.
"""
import json
import hashlib
import math
import os
import queue
import tempfile
import threading
import time
import multiprocessing
import concurrent.futures
from dataclasses import dataclass
from collections import deque, OrderedDict
from typing import Dict, List, Any, Optional

import bpy  # type: ignore
import numpy as np
try:
    from numba import njit  # type: ignore
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False
try:
    from bpy.app.handlers import persistent  # type: ignore
except Exception:  # pragma: no cover - Blender provides this at runtime
    def persistent(fn):
        return fn

from .abstract_tree import ACEAbstractTree, ACEAbstractNode, ACEFrontendRef, ACEConsumerIntent
from .audio_influx import AudioReception, build_live_flow_identifier
from ..nodes.dummy_tree import ACEDummyTreeReader
from .id_management import ACEIdManager
from ..properties import ACEAudioSource
from .proxy_management import ACEProxyManager
from .parallel_dsp import evaluate_parallel_job
from .audio_tasks_registry import (
    SAMPLE_RATE,
    AudioChunk,
    resolve_context_block_size,
    resolve_preview_block_size,
    ACEAudioGraph,
    ACEAudioTask,
    SourceTask,
    SourceSpecsTask,
    FrequencySplitterTask,
    MidsideSplitterTask,
    AttackReleaseSplitterTask,
    ChromagramTask,
    AutoComponentSplitterTask,
    RMSTask,
    LUFSTask,
    PeakTask,
    PitchTask,
    SpectralCentroidTask,
    SpectralFluxTask,
    OnsetTask,
    ZeroCrossTask,
    StereoWidthTask,
    StereoBalanceTask,
    StereoVectorTask,
    PreviewSinkTask,
    RollingAvgTask,
    StreamAccumulatorTask,
    AudioMathTask,
    ACESinkBinding,
    _clear_file_open_locks,
    reset_biquad_fallback_warned,
)
from .. import utils

try:
    from concurrent.futures.process import BrokenProcessPool
except Exception:  # pragma: no cover - platform/runtime dependent
    BrokenProcessPool = RuntimeError

try:
    from multiprocessing import shared_memory as mp_shared_memory
    _SHARED_MEMORY_AVAILABLE = True
except Exception:  # pragma: no cover - runtime/platform dependent
    mp_shared_memory = None
    _SHARED_MEMORY_AVAILABLE = False


FRONTEND_KIND_GEOMETRY = "GEOMETRY_NODES"
FRONTEND_KIND_SHADER = "SHADER_NODES"


@dataclass
class _ACEFrontendDiscovery:
    tree: Any
    frontend_ref: ACEFrontendRef
    consumer_object_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ACECompiledFrontendSnapshot:
    uid: str
    kind: str
    tree_name: str
    tree_bl_idname: str
    owner_keys: tuple[str, ...] = ()
    consumer_object_names: tuple[str, ...] = ()
    material_name: str = ""
    tree_ptr: int = 0


if _NUMBA_AVAILABLE:
    @njit(cache=True)  # pragma: no cover - compiled at runtime
    def _preview_limiter_block_numba(
        frames: np.ndarray,
        ceiling: float,
        release_step: float,
        gain: float,
    ):
        out = np.empty_like(frames)
        g = gain
        channels = int(frames.shape[1]) if frames.ndim == 2 else 1
        for i in range(frames.shape[0]):
            frame_peak = 0.0
            if channels <= 1:
                frame_peak = abs(float(frames[i, 0])) if frames.ndim == 2 else abs(float(frames[i]))
            else:
                for c in range(channels):
                    val = abs(float(frames[i, c]))
                    if val > frame_peak:
                        frame_peak = val
            desired = 1.0 if frame_peak <= 1e-9 else min(1.0, ceiling / frame_peak)
            if desired < g:
                g = desired
            else:
                g += (desired - g) * release_step
            if frames.ndim == 2:
                for c in range(channels):
                    out[i, c] = frames[i, c] * g
            else:
                out[i] = frames[i] * g
        return out, g


# ==============================================================================
# PREVIEW AUDIO MONITOR
# ==============================================================================


class _ACEPreviewFeeder:
    """
    Autonomous rate-converting audio feeder for the preview monitor ring buffer.

    The sounddevice hardware callback drains the ring at a fixed hardware rate
    (~11.6 ms per 512-sample block at 44100 Hz), but Blender's frame handler
    pushes audio at an irregular, frame-rate-coupled rate (~33 ms per frame at
    30 fps). With only ~446 samples in the ring at minimum fill (one analysis
    block minus two callbacks), any Blender jitter ≥ 2 ms causes an underrun
    and produces an audible silence gap / crackle.

    This class absorbs that jitter by:
      1. Accumulating incoming audio in an internal linear buffer via push_frames().
      2. Running a daemon thread that wakes on demand when the sounddevice
         callback reports low ring fill, then refilling the ring up to a target
         level without depending on time.sleep().
      3. Applying a soft drift-correction trim: when the analysis block size
         slightly exceeds the real-time frame size (typical due to integer
         rounding), old audio is silently trimmed from the internal buffer to
         prevent A/V delay from accumulating over long playback sessions.

    Thread safety: push_frames() is called from the prepare thread; the feed
    loop runs in its own daemon thread. All shared state is protected by
    self._lock / self._work_event.
    """

    _TARGET_FILL: float = 0.35   # Desired ring fill fraction [0..1]
    _RT_FRAMES_CAP_MULTIPLIER: int = 4   # Internal buffer cap in real-time frame multiples
    _PREFILL_RT_FRAMES: int = 2
    _MIN_PUSH_FRAMES: int = 64

    def __init__(self):
        """Initialise the feeder with empty buffers and a stopped thread."""
        # Internal audio store: deque of (n, 2) float32 blocks + total frame count.
        # Using a deque avoids the per-push np.concatenate allocation of the old
        # single-ndarray approach; blocks are appended without copying and consumed
        # from the front in _loop via _pop_frames_locked().
        self._deque: deque = deque()
        self._deque_frames: int = 0
        self._lock = threading.Lock()
        self._work_event = threading.Event()
        self._stop_flag: bool = True
        self._thread: Optional[threading.Thread] = None
        self._sample_rate: int = 44100
        self._fps: float = 30.0
        self._drain_block: int = 512
        self._ring_append_fn = None   # callable(np.ndarray) — thread-safe ring append
        self._ring_pressure_fn = None  # callable() → float [0..1]
        self._prefilled: bool = False

    # ------------------------------------------------------------------
    # Configuration & lifecycle
    # ------------------------------------------------------------------

    def configure(
        self,
        sample_rate: int,
        fps: float,
        drain_block: int,
        ring_append_fn,
        ring_pressure_fn,
    ):
        """(Re-)configure the feeder. Flushes the internal buffer on change."""
        with self._lock:
            self._deque.clear()
            self._deque_frames = 0
            self._sample_rate = max(int(sample_rate), 1)
            self._fps = max(float(fps), 1.0)
            self._drain_block = max(int(drain_block), 64)
            self._ring_append_fn = ring_append_fn
            self._ring_pressure_fn = ring_pressure_fn
            self._prefilled = False
        self._work_event.clear()

    def update_fps(self, fps: float):
        """Update the playback frame rate used by the drift-correction cap."""
        with self._lock:
            self._fps = max(float(fps), 1.0)

    def push_frames(self, frames: np.ndarray):
        """Accept stereo float32 audio from the prepare thread.

        Appends *frames* as a new block in the internal deque without copying or
        allocating a merged array.  The soft drift-correction cap evicts the oldest
        block(s) when the total frame count exceeds the real-time window ceiling.
        """
        arr = np.asarray(frames, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
            return
        with self._lock:
            self._deque.append(arr)
            self._deque_frames += int(arr.shape[0])
            # Soft drift-correction cap: drop oldest blocks when total stored audio
            # exceeds N real-time frames to prevent A/V sync drift.
            rt = max(int(round(self._sample_rate / max(self._fps, 1.0))), 1)
            cap = max(rt * self._RT_FRAMES_CAP_MULTIPLIER, self._drain_block * 8)
            while self._deque_frames > cap and self._deque:
                dropped = self._deque.popleft()
                self._deque_frames -= int(dropped.shape[0])
            if self._deque_frames < 0:
                self._deque_frames = 0
        self._work_event.set()

    def flush(self):
        """Discard all buffered audio (call on seek / scrub)."""
        with self._lock:
            self._deque.clear()
            self._deque_frames = 0
            self._prefilled = False
        self._work_event.clear()

    def buffered_frame_count(self) -> int:
        """Return the total number of stereo frames currently held in the buffer."""
        with self._lock:
            return int(self._deque_frames)

    def notify_ring_low(self):
        """Wake the feeder when the hardware callback drained the ring."""
        self._work_event.set()

    def start(self):
        """Start the feeder thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            with self._lock:
                self._stop_flag = False
            return
        with self._lock:
            self._stop_flag = False
        t = threading.Thread(
            target=self._loop, daemon=True, name="ACEPreviewFeeder"
        )
        self._thread = t
        t.start()

    def stop(self):
        """Signal the feeder to stop and wait for exit (≤ 1 s)."""
        with self._lock:
            self._stop_flag = True
        self._work_event.set()   # Unblock the waiting thread
        t = self._thread
        self._thread = None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=1.0)

    # ------------------------------------------------------------------
    # Feed loop
    # ------------------------------------------------------------------

    def _pop_frames_locked(self, n: int) -> np.ndarray:
        """Pop exactly *n* stereo frames from the front of the deque.

        Must be called while holding self._lock.  Handles partial head-block
        reads without copying the entire buffer — only the head slice is
        materialised when a block spans the take boundary.  For the common case
        where *n* == block size the head block is returned directly (zero copy).
        """
        out: list = []
        remaining = n
        while remaining > 0 and self._deque:
            head = self._deque[0]
            head_n = int(head.shape[0])
            if head_n <= remaining:
                self._deque.popleft()
                self._deque_frames -= head_n
                out.append(head)
                remaining -= head_n
            else:
                out.append(head[:remaining])
                self._deque[0] = head[remaining:]
                self._deque_frames -= remaining
                remaining = 0
        if self._deque_frames < 0:
            self._deque_frames = 0
        if not out:
            return np.zeros((0, 2), dtype=np.float32)
        if len(out) == 1:
            return out[0]
        return np.concatenate(out, axis=0)

    def _loop(self):
        while True:
            if not self._work_event.wait(timeout=0.25):
                with self._lock:
                    if self._stop_flag:
                        break
                continue
            self._work_event.clear()

            while True:
                with self._lock:
                    if self._stop_flag:
                        return
                    sr = max(int(self._sample_rate), 1)
                    fps = max(float(self._fps), 1.0)
                    drain_block = max(int(self._drain_block), 1)
                    ring_append_fn = self._ring_append_fn
                    ring_pressure_fn = self._ring_pressure_fn
                    prefilled = bool(self._prefilled)

                if ring_append_fn is None:
                    break

                # ring_pressure_fn reads _ACEPreviewAudioMonitor._ring_frames without
                # holding the monitor's lock.  On CPython the GIL makes integer reads
                # atomic, so the worst case is a momentarily stale fill estimate that
                # causes at most one extra block pushed into the ring — acceptable.
                # Adding the monitor lock here would create a cross-lock dependency and
                # is not warranted for a soft pressure signal.
                pressure = (
                    float(ring_pressure_fn())
                    if ring_pressure_fn is not None
                    else 0.0
                )
                pressure = min(max(pressure, 0.0), 1.0)
                if prefilled and pressure >= self._TARGET_FILL:
                    break

                rt_frames = max(int(round(float(sr) / fps)), 1)
                prefill_frames = max(rt_frames * self._PREFILL_RT_FRAMES, drain_block * 2)
                min_push = max(min(drain_block, self._MIN_PUSH_FRAMES), 1)
                block = None
                partial = False

                with self._lock:
                    if self._stop_flag:
                        return
                    available = int(self._deque_frames)
                    if not self._prefilled:
                        if available < prefill_frames:
                            break
                        self._prefilled = True
                        available = int(self._deque_frames)
                    if available >= drain_block:
                        take_n = drain_block
                    elif available >= min_push or (available > 0 and pressure <= 0.0):
                        take_n = available
                        partial = True
                    else:
                        break
                    block = self._pop_frames_locked(take_n)

                try:
                    ring_append_fn(block)
                except Exception as _ring_exc:
                    # Log the first ring-append failure only to avoid flooding the
                    # console in the audio-callback hot path.  Subsequent failures are
                    # silent because the feeder continues running and will retry on the
                    # next loop iteration.
                    if not getattr(self, "_ring_append_error_logged", False):
                        self._ring_append_error_logged = True
                        try:
                            utils.log(
                                f"[ACE-Feeder] ring_append_fn failed (audio may drop): {_ring_exc!r}",
                                "WARNING",
                            )
                        except Exception:
                            pass

                if partial:
                    break


class _ACEPreviewAudioMonitor:
    """sounddevice-backed real-time audio monitor for branch preview output.

    Manages a hardware output stream at a fixed block size and feeds it from a
    small ring buffer.  Audio is prepared (limiter, resampling, stereo layout)
    on a dedicated ``ACEPreviewMonitor`` daemon thread and then handed to an
    ``_ACEPreviewFeeder`` instance that paces delivery to the ring at the
    hardware drain rate.

    Lifecycle:
      - ``push_chunk()`` enqueues an ``AudioChunk`` from the main thread.
      - The prepare thread calls ``_prepare_preview_payload()`` which runs the
        safety limiter and routes the result through the feeder.
      - The sounddevice callback drains the ring in real time.
      - ``close()`` stops the prepare thread and the hardware stream.

    Thread safety: all ring and statistic fields are protected by ``self._lock``.
    The prepare thread is coordinated via ``self._prepare_condition``.
    """

    def __init__(self):
        self._backend = None
        self._stream = None
        # Keep stream output block-size fixed and feed it from a small ring buffer.
        self._ring = deque()
        self._ring_frames = 0
        self._lock = threading.Lock()
        self._prepare_condition = threading.Condition()
        self._prepare_queue = deque()
        self._prepare_queue_limit = 8   # Increased: feeder absorbs backpressure
        self._prepare_thread: Optional[threading.Thread] = None
        self._prepare_stop = False
        self._sample_rate = 0
        self._block_size = 1024
        self._max_ring_frames = self._block_size * 8
        # Autonomous feeder: decouples push rate from hardware drain rate.
        self._feeder = _ACEPreviewFeeder()
        self._scene_fps: float = 30.0
        self._unavailable_logged = False
        self._limiter_ceiling_db = -1.0
        self._limiter_ceiling_lin = float(10.0 ** (self._limiter_ceiling_db / 20.0))
        self._limiter_release_ms = 120.0
        self._limiter_gain = 1.0
        self._preview_push_count = 0
        self._preview_push_frames = 0
        self._preview_queue_drops = 0
        self._preview_underruns = 0
        self._preview_underrun_frames = 0
        self._preview_callback_ts = deque(maxlen=240)
        self._preview_latency_ms = deque(maxlen=240)
        self._preview_prepare_drops = 0
        self._preview_prepare_drop_frames = 0
        self._last_callback_ts = 0.0
        self._last_push_ts = 0.0
        self._preview_consecutive_errors: int = 0
        self._preview_error_threshold: int = 5
        self._analysis_block_size_hint: int = 0

    @staticmethod
    def _diag_percentile(values, q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(v) for v in values)
        idx = int(round((float(q) / 100.0) * max(len(ordered) - 1, 0)))
        return float(ordered[idx])

    def _resolve_backend(self):
        if self._backend is not None:
            return self._backend
        try:
            import sounddevice as sd  # type: ignore
            self._backend = sd
        except Exception:
            self._backend = False
            if not self._unavailable_logged:
                utils.log("Preview monitor unavailable: install 'sounddevice' in add-on preferences.", "WARNING")
                self._unavailable_logged = True
        return self._backend

    @staticmethod
    def _chunk_to_stereo_frames(chunk):
        mono = np.asarray(getattr(chunk, "mono", None), dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return None

        stereo = getattr(chunk, "stereo", None)
        if stereo is None:
            data = np.column_stack((mono, mono)).astype(np.float32, copy=False)
            return data

        arr = np.asarray(stereo, dtype=np.float32)
        if arr.ndim != 2:
            data = np.column_stack((mono, mono)).astype(np.float32, copy=False)
            return data

        if arr.shape[0] >= 2 and arr.shape[1] == mono.size:
            left = arr[0]
            right = arr[1]
        elif arr.shape[1] >= 2 and arr.shape[0] == mono.size:
            left = arr[:, 0]
            right = arr[:, 1]
        elif arr.shape[0] >= 1 and arr.shape[1] == mono.size:
            left = arr[0]
            right = arr[0]
        else:
            left = mono
            right = mono

        data = np.column_stack((left, right)).astype(np.float32, copy=False)
        return data

    @staticmethod
    def _preferred_block_size(sample_rate: int) -> int:
        sr = max(int(sample_rate), 1)
        if sr > 48_000:
            return 1024
        return 512

    def hint_block_size(self, block_size: int) -> None:
        """Inform the monitor of the current analysis block size.

        When non-zero, this value is used instead of _preferred_block_size() as
        the hardware stream block size, aligning the hardware drain cadence with
        the DSP chunk cadence and reducing preview latency by up to 2×.
        """
        self._analysis_block_size_hint = max(int(block_size), 0)

    @staticmethod
    def _chunk_frame_count(chunk) -> int:
        try:
            mono = np.asarray(getattr(chunk, "mono", None), dtype=np.float32).reshape(-1)
            return int(mono.shape[0] or 0)
        except Exception:
            return 0

    def _drop_prepare_payload_locked(self, payload):
        if payload is None:
            return
        self._preview_prepare_drops += 1
        self._preview_prepare_drop_frames += max(self._chunk_frame_count(payload.get("chunk")), 0)

    def _enqueue_prepare_payload_locked(self, payload):
        while len(self._prepare_queue) >= int(max(self._prepare_queue_limit, 1)):
            try:
                dropped = self._prepare_queue.popleft()
            except IndexError:
                break
            self._drop_prepare_payload_locked(dropped)
        self._prepare_queue.append(payload)

    def _ensure_prepare_worker(self) -> bool:
        with self._prepare_condition:
            worker = self._prepare_thread
            if worker is not None and worker.is_alive():
                return True
            self._prepare_stop = False
            worker = threading.Thread(
                target=self._prepare_loop,
                name="ACEPreviewMonitor",
                daemon=True,
            )
            self._prepare_thread = worker
            worker.start()
            return True

    def _prepare_loop(self):
        while True:
            payload = None
            with self._prepare_condition:
                while not self._prepare_queue and not self._prepare_stop:
                    self._prepare_condition.wait(timeout=0.25)
                if self._prepare_stop and not self._prepare_queue:
                    break
                if self._prepare_queue:
                    payload = self._prepare_queue.popleft()
            if payload is None:
                continue
            try:
                self._prepare_preview_payload(payload)
                self._preview_consecutive_errors = 0
            except Exception as _exc:
                self._preview_consecutive_errors += 1
                try:
                    import traceback as _tb
                    utils.log(
                        f"[ACE-Preview] _prepare_preview_payload error "
                        f"(#{self._preview_consecutive_errors}): {_exc!r}\n{_tb.format_exc()}",
                        "WARNING",
                    )
                except Exception:
                    pass
                continue

    def _append_ring_block_locked(self, frames: np.ndarray):
        arr = np.asarray(frames, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return
        total = int(arr.shape[0])
        if total <= 0:
            return
        cap = max(int(self._max_ring_frames), 1)
        if total > cap:
            arr = arr[-cap:, :]
            total = int(arr.shape[0])
        self._ring.append(arr)
        self._ring_frames += total
        # Drop oldest blocks first and keep only fresh audio to prevent A/V drift.
        while self._ring and self._ring_frames > cap:
            drop = self._ring.popleft()
            drop_n = int(getattr(drop, "shape", [0])[0] or 0)
            self._ring_frames -= drop_n
            self._preview_queue_drops += max(drop_n, 0)
        if self._ring_frames < 0:
            self._ring_frames = 0

    def _callback(self, outdata, frames, _time, _status):
        out = np.zeros((frames, 2), dtype=np.float32)
        now_ts = time.perf_counter()
        if self._last_callback_ts > 0.0:
            self._preview_callback_ts.append(max(now_ts - float(self._last_callback_ts), 0.0))
        self._last_callback_ts = now_ts
        write_idx = 0
        ring_frames_after = 0
        refill_threshold = 0
        with self._lock:
            while write_idx < frames and self._ring:
                block = self._ring[0]
                block_n = int(block.shape[0] or 0)
                if block_n <= 0:
                    self._ring.popleft()
                    continue
                take_n = min(frames - write_idx, block_n)
                out[write_idx: write_idx + take_n, :] = block[:take_n, :]
                write_idx += take_n
                self._ring_frames -= take_n
                if take_n >= block_n:
                    self._ring.popleft()
                else:
                    self._ring[0] = block[take_n:, :]
            if self._ring_frames < 0:
                self._ring_frames = 0
            if write_idx < int(frames):
                self._preview_underruns += 1
                self._preview_underrun_frames += int(frames - write_idx)
            ring_frames_after = int(self._ring_frames)
            refill_threshold = int(self._max_ring_frames * _ACEPreviewFeeder._TARGET_FILL)

        outdata[:, :] = out
        if ring_frames_after < refill_threshold:
            try:
                self._feeder.notify_ring_low()
            except Exception:
                pass

    def _ensure_stream(self, sample_rate: int):
        sd = self._resolve_backend()
        if not sd:
            return False

        sample_rate = max(int(sample_rate), 1)
        # Prefer the analysis block size (set by hint_block_size) so the hardware
        # drain cadence matches the DSP chunk cadence, halving preview latency.
        if self._analysis_block_size_hint >= 64:
            block_size = max(int(self._analysis_block_size_hint), 64)
        else:
            block_size = max(int(self._preferred_block_size(sample_rate)), 64)
        if self._stream and self._sample_rate == sample_rate and self._block_size == block_size:
            return True

        self._close_stream_only()
        try:
            self._stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=2,
                blocksize=block_size,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
            self._sample_rate = sample_rate
            self._block_size = block_size
            # Ring buffer: 16 hardware blocks gives ~186 ms at 512/44100 Hz —
            # enough headroom for the feeder's adaptive fill target (~35%).
            self._max_ring_frames = max(self._block_size * 16, sample_rate // 4)
            # Configure and start the autonomous feeder.
            self._feeder.configure(
                sample_rate=sample_rate,
                fps=self._scene_fps,
                drain_block=block_size,
                ring_append_fn=self._feeder_ring_append,
                ring_pressure_fn=lambda: float(self._ring_frames)
                / float(max(self._max_ring_frames, 1)),
            )
            self._feeder.start()
            return True
        except Exception as e:
            utils.log(f"Preview monitor stream init failed: {e}", "WARNING")
            self._stream = None
            self._sample_rate = 0
            self._block_size = block_size  # already resolved above (hint or heuristic)
            self._max_ring_frames = max(self._block_size * 16, sample_rate // 4)
            return False

    def _apply_safety_limiter(self, frames: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Protect preview output from clipping spikes.
        Instant attack, timed release, and final hard-cap at -1 dBFS.
        """
        arr = np.asarray(frames, dtype=np.float32)
        if arr.size == 0:
            return arr

        ceiling = float(max(self._limiter_ceiling_lin, 1e-6))
        sr = max(int(sample_rate), 1)
        release_tc = max(float(self._limiter_release_ms), 1.0) / 1000.0
        release_step = 1.0 - math.exp(-1.0 / (float(sr) * release_tc))

        gain = float(min(max(self._limiter_gain, 0.0), 1.0))
        if _NUMBA_AVAILABLE:
            try:
                out, gain = _preview_limiter_block_numba(arr, ceiling, release_step, gain)
                np.clip(out, -ceiling, ceiling, out=out)
                self._limiter_gain = float(min(max(gain, 0.0), 1.0))
                return out.astype(np.float32, copy=False)
            except Exception:
                pass

        # Vectorized fallback: instant-attack via np.minimum.accumulate,
        # release recovery via np.maximum.accumulate on the per-frame increments.
        n_frames = int(arr.shape[0])
        if arr.ndim >= 2:
            peaks = np.max(np.abs(arr.reshape(n_frames, -1)), axis=1).astype(np.float64)
        else:
            peaks = np.abs(arr).astype(np.float64)
        desired = np.where(
            peaks > 1e-9,
            np.minimum(1.0, float(ceiling) / np.where(peaks > 1e-9, peaks, 1.0)),
            1.0,
        )
        # Instant attack: running minimum across the block including the carried-over gain.
        all_desired = np.empty(n_frames + 1, dtype=np.float64)
        all_desired[0] = float(gain)
        all_desired[1:] = desired
        attack_gains = np.minimum.accumulate(all_desired)  # shape (n_frames+1,)
        # Release: per-frame recovery increments toward desired, accumulated via maximum.
        # np.maximum.accumulate gives the running maximum of the release potential,
        # which bounds how much the gain can recover within this block.
        release_increments = np.maximum(0.0, desired - attack_gains[1:]) * float(release_step)
        release_ceiling = np.maximum.accumulate(
            np.concatenate(([0.0], release_increments))
        )[1:]
        final_gains = np.minimum(desired, attack_gains[1:] + release_ceiling).astype(np.float32)
        gain = float(np.clip(final_gains[-1], 0.0, 1.0))
        if arr.ndim >= 2:
            out = (arr * final_gains.reshape(-1, *([1] * (arr.ndim - 1)))).astype(np.float32, copy=False)
        else:
            out = (arr * final_gains).astype(np.float32, copy=False)
        np.clip(out, -ceiling, ceiling, out=out)
        self._limiter_gain = gain
        return out

    def _feeder_ring_append(self, block: np.ndarray):
        """Thread-safe ring append called by the feeder daemon thread."""
        with self._lock:
            self._append_ring_block_locked(block)

    def update_scene_fps(self, fps: float):
        """Update the playback fps used for feeder drift-correction."""
        self._scene_fps = max(float(fps), 1.0)
        self._feeder.update_fps(self._scene_fps)

    def _prepare_preview_payload(self, payload: Dict[str, Any]):
        """Apply the safety limiter to a queued payload and route it through the feeder.

        Called from the prepare thread.  Extracts stereo frames from the chunk,
        ensures the hardware stream exists at the correct sample rate, runs the
        limiter, then hands the result to the autonomous feeder rather than writing
        directly to the ring so that hardware drain pacing is preserved.
        """
        chunk = payload.get("chunk")
        sample_rate = int(payload.get("sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
        frames = self._chunk_to_stereo_frames(chunk)
        if frames is None:
            return
        if not self._ensure_stream(sample_rate):
            return
        frames = self._apply_safety_limiter(frames, sample_rate)
        arr = np.asarray(frames, dtype=np.float32)
        # Update push statistics under lock, but route audio through the feeder
        # (not directly to the ring) so the daemon thread paces delivery at the
        # hardware block rate, preventing underruns caused by Blender timer jitter.
        with self._lock:
            self._preview_push_count += 1
            self._preview_push_frames += int(arr.shape[0] or 0)
            self._last_push_ts = time.perf_counter()
            latency_ms = float(getattr(chunk, "latency_ms", 0.0) or 0.0)
            if latency_ms > 0.0:
                self._preview_latency_ms.append(latency_ms)
        self._feeder.push_frames(arr)

    def push_chunk(self, chunk) -> bool:
        """Enqueue an AudioChunk for asynchronous prepare and playback.

        Called from the main thread.  Returns True if the chunk was accepted
        into the prepare queue; False if the backend is unavailable, the
        sample rate is invalid, or the worker is stopped.
        """
        sample_rate = int(getattr(chunk, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
        if sample_rate <= 0:
            return False
        if not self._resolve_backend():
            return False
        if not self._ensure_prepare_worker():
            return False
        payload = {"chunk": chunk, "sample_rate": sample_rate}
        with self._prepare_condition:
            if self._prepare_stop:
                return False
            self._enqueue_prepare_payload_locked(payload)
            self._prepare_condition.notify()
        return True

    def diagnostics_snapshot(self, reset: bool = False) -> Dict[str, float]:
        """Return accumulated playback diagnostics as a flat float dict.

        Args:
            reset: When True, clear all counters and history after sampling.

        Returns:
            Dict with keys: preview_pushes, preview_pushed_frames,
            preview_underruns, preview_underrun_frames, preview_queue_drops,
            preview_queue_pressure, preview_callback_p95_ms,
            preview_latency_p95_ms, preview_prepare_drops,
            preview_prepare_drop_frames, preview_prepare_queue_pressure.
        """
        with self._prepare_condition:
            prepare_depth = int(len(self._prepare_queue))
            prepare_capacity = max(int(self._prepare_queue_limit), 1)
        with self._lock:
            callback_ms = [float(v) * 1000.0 for v in list(self._preview_callback_ts)]
            latency_ms = list(self._preview_latency_ms)
            queue_capacity = max(int(self._max_ring_frames), 1)
            queue_frames = int(self._ring_frames)
            snapshot = {
                "preview_pushes": float(self._preview_push_count),
                "preview_pushed_frames": float(self._preview_push_frames),
                "preview_underruns": float(self._preview_underruns),
                "preview_underrun_frames": float(self._preview_underrun_frames),
                "preview_queue_drops": float(self._preview_queue_drops),
                "preview_queue_pressure": float(min(max(queue_frames / float(queue_capacity), 0.0), 1.0)),
                "preview_callback_p95_ms": float(self._diag_percentile(callback_ms, 95.0)),
                "preview_latency_p95_ms": float(self._diag_percentile(latency_ms, 95.0)),
                "preview_prepare_drops": float(self._preview_prepare_drops),
                "preview_prepare_drop_frames": float(self._preview_prepare_drop_frames),
                "preview_prepare_queue_pressure": float(
                    min(max(prepare_depth / float(prepare_capacity), 0.0), 1.0)
                ),
            }
            if reset:
                self._preview_push_count = 0
                self._preview_push_frames = 0
                self._preview_queue_drops = 0
                self._preview_underruns = 0
                self._preview_underrun_frames = 0
                self._preview_callback_ts.clear()
                self._preview_latency_ms.clear()
                self._preview_prepare_drops = 0
                self._preview_prepare_drop_frames = 0
            return snapshot

    def current_snapshot(self) -> Dict[str, float]:
        """Return a real-time latency snapshot for the N-panel live display.

        Unlike ``diagnostics_snapshot``, this method computes instantaneous
        values (current buffer backlog, latest callback interval) rather than
        accumulating history, and does not support a reset flag.

        Returns:
            Dict with keys: preview_branch_latency_ms, preview_output_buffer_ms,
            preview_callback_ms.  All values are 0.0 when preview is inactive.
        """
        now_ts = time.perf_counter()
        feeder_frames = int(self._feeder.buffered_frame_count())
        with self._prepare_condition:
            prepare_payloads = tuple(self._prepare_queue)
        prepare_frames = sum(
            max(self._chunk_frame_count(payload.get("chunk")), 0)
            for payload in prepare_payloads
            if payload is not None
        )
        with self._lock:
            sample_rate = int(self._sample_rate or 0)
            ring_frames = int(self._ring_frames or 0)
            latest_latency_ms = (
                float(self._preview_latency_ms[-1])
                if self._preview_latency_ms
                else 0.0
            )
            latest_callback_ms = (
                float(self._preview_callback_ts[-1] * 1000.0)
                if self._preview_callback_ts
                else 0.0
            )
            last_push_ts = float(self._last_push_ts or 0.0)

        total_buffered_frames = max(ring_frames, 0) + max(feeder_frames, 0) + max(prepare_frames, 0)
        preview_is_active = bool(
            total_buffered_frames > 0
            or (last_push_ts > 0.0 and (now_ts - last_push_ts) <= 1.0)
        )
        output_buffer_ms = 0.0
        if preview_is_active and sample_rate > 0:
            # The monitor intentionally keeps a healthy amount of headroom in its
            # ring/feeder buffers to prevent crackles from Blender timer jitter.
            # Report only excess backlog above that steady-state headroom.
            scene_fps = max(float(self._scene_fps), 1.0)
            rt_frames = max(int(round(float(sample_rate) / scene_fps)), 1)
            healthy_ring_frames = int(
                max(float(self._max_ring_frames) * float(_ACEPreviewFeeder._TARGET_FILL), 0.0)
            )
            healthy_feeder_frames = max(
                rt_frames * int(max(_ACEPreviewFeeder._PREFILL_RT_FRAMES, 1)),
                int(self._block_size) * 2,
            )
            healthy_buffer_frames = max(healthy_ring_frames + healthy_feeder_frames, 0)
            excess_buffer_frames = max(total_buffered_frames - healthy_buffer_frames, 0)
            output_buffer_ms = float(excess_buffer_frames / float(sample_rate) * 1000.0)
        return {
            "preview_branch_latency_ms": float(latest_latency_ms if preview_is_active else 0.0),
            "preview_output_buffer_ms": float(output_buffer_ms),
            "preview_callback_ms": float(latest_callback_ms if preview_is_active else 0.0),
        }

    def clear(self):
        """Discard all queued and buffered audio (called on seek or scrub)."""
        with self._prepare_condition:
            self._prepare_queue.clear()
        self._feeder.flush()
        with self._lock:
            self._ring.clear()
            self._ring_frames = 0
            self._last_callback_ts = 0.0
            self._last_push_ts = 0.0

    def _close_stream_only(self):
        """Stop the feeder and hardware stream without touching the prepare thread.

        Stops the autonomous feeder first so it cannot append frames to the ring
        concurrently with teardown, then stops and closes the sounddevice stream.
        Resets stream-related state but leaves the prepare thread running.
        """
        # Stop the feeder before clearing the ring so it cannot push new frames
        # concurrently with the ring teardown.
        self._feeder.stop()
        with self._lock:
            self._ring.clear()
            self._ring_frames = 0
            self._last_callback_ts = 0.0
            self._last_push_ts = 0.0
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
        self._stream = None
        self._sample_rate = 0
        self._block_size = max(int(self._preferred_block_size(SAMPLE_RATE)), 64)
        self._max_ring_frames = max(self._block_size * 8, self._block_size * 2)
        self._limiter_gain = 1.0

    def close(self):
        """Fully shut down the monitor: stop the prepare thread and hardware stream."""
        worker = None
        with self._prepare_condition:
            self._prepare_queue.clear()
            self._prepare_stop = True
            self._prepare_condition.notify_all()
            worker = self._prepare_thread
            self._prepare_thread = None
        if worker is not None and worker is not threading.current_thread():
            try:
                worker.join(timeout=0.5)
            except Exception:
                pass
        self._close_stream_only()


# ==============================================================================
# ENGINE ORCHESTRATOR
# ==============================================================================

class ACEAudioEngine:
    """Central orchestrator for the AudioCore Engine DSP pipeline.

    One instance exists per Blender scene (keyed by ``scene.as_pointer()``).
    It owns the compile/execute lifecycle, the parallel worker pool, all
    ``ACEAudioGraph`` instances, and the shared ``_ACEPreviewAudioMonitor``.

    Architecture overview:
      - Stage A (main thread): ``compile_scene_frontends()`` discovers all
        managed Geometry/Shader frontends in the scene, merges them into one
        ``ACEAbstractTree``, builds per-source ``ACEAudioGraph`` objects, and
        prepares a serialisable parallel IR for each graph.
      - Stage B (worker processes): numeric DSP (RMS, pitch, centroid, etc.) is
        offloaded via ``evaluate_parallel_job()`` in a
        ``ProcessPoolExecutor``; results flow back through
        ``_parallel_poll_futures()`` into ``_parallel_last_valid``.
      - Stage C (main thread): ``execute_next_frame()`` merges Stage-B values
        with any locally-evaluated outputs and writes them into Blender geometry
        node proxy attributes via ``ACEProxyManager``.

    Instance management:
      ``ensure_instance(scene)`` returns or creates the engine for *scene*.
      ``shutdown_all_instances()`` shuts everything down cleanly on addon
      unregister or file load.

    Parallel pool constants:
      - ``_PARALLEL_MAX_INFLIGHT_PER_SOURCE``: max 2 concurrent jobs per source.
      - ``_PARALLEL_STALE_REUSE_MS``: last-valid values reused within this window.
      - ``_PARALLEL_RESTART_LIMIT``: maximum pool restart attempts per session.
      - ``_PARALLEL_RESTART_BACKOFF_S``: per-attempt exponential backoff delays.
    """

    _instances: Dict[int, "ACEAudioEngine"] = {}
    _preview_monitor = _ACEPreviewAudioMonitor()
    # Attribute schema is synced via a dirty flag set on compile, not by frame polling.
    # The fallback interval applies only when the dirty set is missing (defensive).
    _ATTR_RESYNC_INTERVAL_FRAMES = 120
    _META_CACHE_MAX_ENTRIES = 256
    _RUNTIME_LOG_MAX_ENTRIES = 512
    _BLOCK_SIZE_HYSTERESIS_FRAMES = 2
    _BLOCK_SIZE_IMMEDIATE_DELTA = 1024
    _PERF_STAGES = ("source_read", "dsp", "preview", "proxy_write", "ui_sync")
    _PERF_HISTORY_FRAMES = 360
    _DSP_NODE_PROFILE_SAMPLES = 1024
    _BLENDER_LATENCY_EVENT_TTL_S = 2.0
    _CURRENT_LATENCY_SAMPLE_FRESHNESS_FRAMES = 2
    _PARALLEL_OFFLOAD_STREAM_TYPES = frozenset(
        {
            "rms",
            "lufs",
            "peak",
            "pitch",
            "spectral_centroid",
            "spectral_flux",
            "onset",
            "zero_crossing_rate",
            "stereo_width",
            "stereo_balance",
            "stereo_vector",
        }
    )
    _PARALLEL_MAX_INFLIGHT_PER_SOURCE = 2
    _PARALLEL_ATTRS_PER_JOB = 8
    _PARALLEL_WORKER_HEADROOM_CORES = 1
    # Hard realtime stale-job budget: jobs older than this are dropped instead
    # of being allowed to resolve late and contaminate frame determinism.
    _PARALLEL_STALE_JOB_MAX_MS = 66
    _PARALLEL_RESTART_LIMIT = 3  # Allow up to 3 restarts before disabling; exponential backoff between attempts.
    # Backoff delays (seconds) indexed by restart attempt number (0-based).
    # Attempt 0 → 0.5 s, attempt 1 → 1.0 s, attempt 2 → 2.0 s.
    _PARALLEL_RESTART_BACKOFF_S: tuple = (0.5, 1.0, 2.0)
    _PARALLEL_STALE_FRAME_GRACE = 2
    # Time-based stale-reuse window (FPS-invariant). At 24 fps this is 1 frame;
    # at 30 fps it is 1 frame; at 60 fps it is 3 frames — reduced from 66 ms (4 frames
    # at 60 fps) to tighten the stale-result contract closer to the ≤2-frame spec.
    _PARALLEL_STALE_REUSE_MS = 50
    _PARALLEL_CANCEL_STALE_FRAME_DELTA = 4
    _PREVIEW_MIN_INTERVAL_S = 0.003
    _FILE_WORKING_SAMPLE_RATE_CAP = 48_000

    def __init__(self, scene_key: int):
        """Initialise a new engine instance for a Blender scene.

        Args:
            scene_key: Integer scene pointer (``scene.as_pointer()``).  Used
                as the lookup key in ``ACEAudioEngine._instances``.
        """
        self.scene_key = int(scene_key)
        self.scene_name: str = ""
        self.graphs: Dict[str, ACEAudioGraph] = {}

        self.current_frame = 0
        self.last_scene_frame = -1
        self.writer: Optional[ACEProxyManager] = None
        self.abstract_tree_ref: Optional[ACEAbstractTree] = None
        # Legacy compatibility slots: keep these unset so the engine never holds
        # long-lived RNA object references across undo/redo.
        self._pending_tree: Optional[bpy.types.NodeTree] = None
        self._pending_scene_rebuild: bool = False
        self._last_request_time: float = 0.0
        self._last_compile_time: float = 0.0
        self.effective_sample_rate: int = SAMPLE_RATE
        self._dirty_trees: set[bpy.types.NodeTree] = set()
        self._dirty_tree_ptrs: set[int] = set()
        self._dirty_tree_bl_idnames: Dict[int, str] = {}
        self._dirty_frontend_uids: set[str] = set()
        self._compiled_tree_ref: Optional[bpy.types.NodeTree] = None
        self._compiled_frontend_refs: Dict[str, Any] = {}
        self._compiled_frontend_snapshots: Dict[str, _ACECompiledFrontendSnapshot] = {}
        self._compiled_frontend_ptrs: Dict[str, int] = {}
        self._compiled_frontend_uids: tuple[str, ...] = ()
        self._last_frontend_membership_signature: tuple[Any, ...] = ()
        self._last_dirty_time: float = 0.0
        self._is_compiling: bool = False
        self._meta_cache: Dict[str, Dict[str, Any]] = {}
        self._meta_cache_ttl = 30.0  # seconds
        self._last_tree_signature: Optional[Any] = None
        self._pending_tree_ptr: int = 0
        self._pending_tree_bl_idname: str = ""
        self._compiled_tree_ptr: int = 0
        self._compiled_tree_frontend_uid: str = ""
        self._compiled_tree_bl_idname: str = ""
        self._last_attr_sync_frame: Dict[str, int] = {}
        self._block_size_state: Dict[str, Dict[str, int]] = {}
        self._bootstrap_retry_at: float = 0.0
        self._bootstrap_fail_count: int = 0
        self._runtime_log_last: Dict[str, float] = {}
        self._bake_cache: Dict[str, Dict[str, Any]] = {}
        self._bake_jobs: Dict[str, Dict[str, Any]] = {}
        self._bake_result_queue: "queue.Queue[tuple]" = queue.Queue()
        self._source_specs_task_registry: Dict[tuple[str, float], SourceSpecsTask] = {}
        self._current_delta_frames: float = 1.0
        self._perf_frame_accumulator: Dict[str, float] = {}
        self._perf_history: Dict[str, deque[float]] = {
            stage: deque(maxlen=self._PERF_HISTORY_FRAMES) for stage in self._PERF_STAGES
        }
        self._perf_last_report: Dict[str, Dict[str, float]] = {}
        self._source_prefetch_history: deque[float] = deque(maxlen=self._PERF_HISTORY_FRAMES)
        self._source_prefetch_lock = threading.Lock()
        self._geometry_eval_history: deque[float] = deque(maxlen=self._PERF_HISTORY_FRAMES)
        self._render_history: deque[float] = deque(maxlen=self._PERF_HISTORY_FRAMES)
        self._geometry_eval_last_frame: int = -10_000_000
        self._pending_geometry_events: list[Dict[str, Any]] = []
        self._render_start_ts: float = 0.0
        self._dsp_node_profile: Dict[str, deque[float]] = {}
        self._parallel_generation_id: int = 0
        self._parallel_job_seq: int = 0
        self._parallel_executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self._parallel_inflight: Dict[str, deque] = {}
        self._parallel_future_meta: Dict[Any, Dict[str, Any]] = {}
        self._parallel_results: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._parallel_last_valid: Dict[str, Dict[str, Any]] = {}
        self._parallel_last_valid_frame: Dict[str, int] = {}
        self._parallel_failures: Dict[str, int] = {}
        self._parallel_disabled_sources: set[str] = set()
        self._parallel_graph_ir_required: set[str] = set()
        self._parallel_restart_count: int = 0
        self._parallel_restart_backoff_until: float = 0.0
        self._parallel_pool_failed: bool = False
        self._parallel_payload_slots: Dict[str, Dict[str, Any]] = {}
        self._parallel_payload_slot_refs: Dict[str, int] = {}
        self._parallel_payload_slot_counters: Dict[str, int] = {}
        self._parallel_payload_free_slots: Dict[str, deque[str]] = {}
        self._parallel_frame_payload_slots: Dict[tuple[str, int, str], str] = {}
        self._parallel_queue_pressure_drops: int = 0
        self._parallel_job_timeouts: int = 0
        self._parallel_job_exceptions: int = 0
        self._parallel_stale_job_drops: int = 0
        self._parallel_stale_result_drops: int = 0
        self._parallel_result_accepts: int = 0
        self._preview_last_chunk: Optional[AudioChunk] = None
        self._preview_last_eval_frame: int = -1
        self._preview_next_eval_ts: float = 0.0
        self._preview_worker_stop: bool = False
        self._preview_worker_cond: threading.Condition = threading.Condition()
        self._preview_worker_queue: list = []
        self._preview_worker_thread: Optional[threading.Thread] = None
        # Async metadata probe: background thread writes here; main thread flushes to bpy props.
        self._probe_pending_updates: Dict[str, Dict[str, Any]] = {}
        self._probe_thread_lock = threading.Lock()
        self._probe_in_flight: set[str] = set()
        # Tracks scene FPS so time-based constants can be converted to frame counts.
        self._current_scene_fps: float = 24.0
        # Parallel submit tracking for pause-drain SHM reclaim.
        self._parallel_last_submit_frame: int = -10_000_000
        # Dirty-flag set for attribute schema resync: populated on compile,
        # cleared per-graph after ensure_attributes_exist() runs.
        self._attr_schema_dirty: set[str] = set()
        self.undo_generation: int = 0
        self.undo_recovery_until: float = 0.0

    @staticmethod
    def _scene_key(scene) -> int:
        """Return ``scene.as_pointer()`` as an int, or 0 if unavailable."""
        if scene is not None and hasattr(scene, "as_pointer"):
            try:
                return int(scene.as_pointer())
            except Exception:
                pass
        return 0

    @staticmethod
    def _scene_name(scene) -> str:
        """Return the scene name string, preferring ``name_full`` over ``name``."""
        if scene is None:
            return ""
        try:
            return str(getattr(scene, "name_full", "") or getattr(scene, "name", "") or "")
        except Exception:
            return ""

    def _clear_blender_runtime_refs(self) -> None:
        self.abstract_tree_ref = None
        self._compiled_tree_ref = None
        self._compiled_tree_ptr = 0
        self._compiled_tree_frontend_uid = ""
        self._compiled_tree_bl_idname = ""
        self._compiled_frontend_refs.clear()
        self._compiled_frontend_snapshots.clear()
        self._compiled_frontend_ptrs.clear()
        self._compiled_frontend_uids = ()
        self._dirty_trees.clear()
        self._dirty_tree_ptrs.clear()
        self._dirty_tree_bl_idnames.clear()
        self._dirty_frontend_uids.clear()
        self._pending_tree = None
        self._pending_tree_ptr = 0
        self._pending_tree_bl_idname = ""
        self._pending_scene_rebuild = False
        self._last_frontend_membership_signature = ()

    def _clear_probe_runtime_state(self) -> None:
        with self._probe_thread_lock:
            self._probe_pending_updates.clear()
            self._probe_in_flight.clear()

    def _undo_recovery_active(self) -> bool:
        recovery_until = float(getattr(self, "undo_recovery_until", 0.0) or 0.0)
        return recovery_until > 0.0 and time.monotonic() < recovery_until

    @classmethod
    def _undo_recovery_active_for_scene(cls, scene=None) -> bool:
        engine = cls.get_instance(scene)
        if engine is None:
            return False
        try:
            return bool(engine._undo_recovery_active())
        except Exception:
            return False

    @classmethod
    def _structural_tree_writes_allowed(
        cls,
        scene=None,
        tree=None,
        frontend_kind: str = "",
        tree_bl_idname: str = "",
    ) -> bool:
        if cls._undo_recovery_active_for_scene(scene):
            return False
        resolved_kind = str(frontend_kind or "")
        resolved_tree_bl_idname = str(tree_bl_idname or getattr(tree, "bl_idname", "") or "")
        if resolved_kind == FRONTEND_KIND_SHADER or resolved_tree_bl_idname == "ShaderNodeTree":
            return not cls._unsafe_for_shader_tree_mutations(scene)
        return True

    @classmethod
    def _instance_for_scene_name(cls, scene_name: str):
        """Find an existing instance whose ``scene_name`` matches the given string."""
        token = str(scene_name or "").strip()
        if not token:
            return None
        for inst in cls._instances.values():
            if str(getattr(inst, "scene_name", "") or "") == token:
                return inst
        return None

    @staticmethod
    def _node_tree_pointer(tree) -> int:
        """Return a stable integer identity for a node tree (pointer or id())."""
        if tree is None:
            return 0
        if hasattr(tree, "as_pointer"):
            try:
                return int(tree.as_pointer())
            except Exception:
                pass
        return id(tree)

    @staticmethod
    def _prune_timestamp_map(mapping: Dict[str, float], max_entries: int):
        """Evict the oldest entries from a ``{key: timestamp}`` dict in-place."""
        if max_entries <= 0 or len(mapping) <= max_entries:
            return
        overflow = len(mapping) - max_entries
        if overflow <= 0:
            return
        for key, _ts in sorted(mapping.items(), key=lambda kv: float(kv[1]))[:overflow]:
            mapping.pop(key, None)

    @classmethod
    def get_instance(cls, scene=None):
        """Return the existing engine for *scene*, or None if none exists.

        Falls back to a name-based lookup, then to the active context scene,
        then to the first registered instance.  Returns None only when no
        instance has been created yet.
        """
        key = cls._scene_key(scene)
        inst = cls._instances.get(key)
        if inst:
            return inst
        if scene is not None:
            by_name = cls._instance_for_scene_name(cls._scene_name(scene))
            if by_name is not None:
                return by_name
        if scene is None:
            active_scene = getattr(getattr(bpy, "context", None), "scene", None)
            active_key = cls._scene_key(active_scene)
            inst = cls._instances.get(active_key)
            if inst:
                return inst
            if cls._instances:
                return next(iter(cls._instances.values()))
        return None

    @classmethod
    def ensure_instance(cls, scene=None):
        """Return the engine for *scene*, creating one if it does not yet exist."""
        resolved_scene = scene if scene is not None else getattr(getattr(bpy, "context", None), "scene", None)
        key = cls._scene_key(resolved_scene)
        inst = cls._instances.get(key)
        if inst is None:
            inst = cls._instance_for_scene_name(cls._scene_name(resolved_scene))
        if inst is None:
            inst = cls(key)
            cls._instances[key] = inst
        if resolved_scene is not None and not getattr(inst, "scene_name", ""):
            inst.scene_name = cls._scene_name(resolved_scene)
        return inst

    @classmethod
    def iter_instances(cls):
        """Return a snapshot list of all active engine instances."""
        return list(cls._instances.values())

    @classmethod
    def shutdown_all_instances(cls):
        """Shut down every engine instance, the preview monitor, and background workers.

        Called from ``unregister_handlers()`` and the ``load_post`` handler.
        Also resets the one-time biquad-fallback warning so it fires again on
        the next session if Numba and scipy are still absent.
        """
        for inst in list(cls._instances.values()):
            try:
                inst._shutdown_graphs(wait_parallel_pool=True, clear_bake_cache=True)
            except Exception:
                pass
        cls._instances.clear()
        try:
            cls._preview_monitor.close()
        except Exception:
            pass
        try:
            SourceSpecsTask.shutdown_background_worker()
        except Exception:
            pass
        try:
            _clear_file_open_locks()
        except Exception:
            pass
        try:
            reset_biquad_fallback_warned()
        except Exception:
            pass

    @classmethod
    def preview_monitor_available(cls) -> bool:
        """Return True if the sounddevice backend loaded successfully."""
        try:
            return bool(cls._preview_monitor._resolve_backend())
        except Exception:
            return False

    @classmethod
    def preview_monitor_consecutive_errors(cls) -> int:
        """Return the current consecutive DSP error count from the prepare worker."""
        try:
            return int(cls._preview_monitor._preview_consecutive_errors)
        except Exception:
            return 0

    @staticmethod
    def _engine_settings(scene_or_engine_settings=None):
        if scene_or_engine_settings is None:
            scene = getattr(getattr(bpy, "context", None), "scene", None)
            return getattr(getattr(scene, "audiocore", None), "engine_settings", None)
        # Accept direct engine settings object
        if hasattr(scene_or_engine_settings, "ace_scheduler_mode"):
            return scene_or_engine_settings
        scene = scene_or_engine_settings
        return getattr(getattr(scene, "audiocore", None), "engine_settings", None)

    @classmethod
    def _scheduler_mode(cls, scene_or_engine_settings=None) -> str:
        settings = cls._engine_settings(scene_or_engine_settings)
        try:
            val = str(getattr(settings, "ace_scheduler_mode", "demand") or "demand").strip().lower()
            return val if val in {"demand", "legacy"} else "demand"
        except Exception:
            return "demand"

    @classmethod
    def _timeslice_mode(cls, scene_or_engine_settings=None) -> str:
        settings = cls._engine_settings(scene_or_engine_settings)
        try:
            val = str(getattr(settings, "ace_timeslice_mode", "fixed") or "fixed").strip().lower()
            return val if val in {"fixed", "adaptive"} else "fixed"
        except Exception:
            return "fixed"

    @classmethod
    def _parallel_mode(cls, scene_or_engine_settings=None) -> str:
        settings = cls._engine_settings(scene_or_engine_settings)
        try:
            val = str(getattr(settings, "ace_parallel_mode", "processes_dsp") or "processes_dsp").strip().lower()
            return val if val in {"processes_dsp", "threads_io", "off"} else "processes_dsp"
        except Exception:
            return "processes_dsp"

    @classmethod
    def _preview_block_size(cls, scene_or_engine_settings=None) -> int:
        settings = cls._engine_settings(scene_or_engine_settings)
        raw = 256
        try:
            raw = int(getattr(settings, "ace_preview_block_size", 256) or 256)
        except Exception:
            raw = 256
        return int(resolve_preview_block_size(raw))

    @classmethod
    def _perf_enabled(cls, scene_or_engine_settings=None) -> bool:
        settings = cls._engine_settings(scene_or_engine_settings)
        try:
            return bool(getattr(settings, "ace_perf_enabled", False))
        except Exception:
            return False

    def _perf_add(self, stage: str, duration_s: float):
        if stage not in self._PERF_STAGES:
            return
        if not self._perf_enabled():
            return
        dur = max(float(duration_s), 0.0)
        self._perf_frame_accumulator[stage] = float(self._perf_frame_accumulator.get(stage, 0.0)) + dur

    def _perf_finalize_frame(self):
        if not self._perf_enabled():
            self._perf_frame_accumulator.clear()
            return
        for stage in self._PERF_STAGES:
            val = float(self._perf_frame_accumulator.get(stage, 0.0))
            self._perf_history[stage].append(val)
        self._perf_frame_accumulator.clear()

    def get_perf_snapshot(self, reset: bool = False) -> Dict[str, Dict[str, float]]:
        report: Dict[str, Dict[str, float]] = {}
        for stage in self._PERF_STAGES:
            series = list(self._perf_history.get(stage, ()))
            if not series:
                report[stage] = {"count": 0.0, "mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
                continue
            sorted_series = sorted(series)
            idx = int(round(0.95 * max(len(sorted_series) - 1, 0)))
            mean_ms = (sum(series) / max(len(series), 1)) * 1000.0
            p95_ms = float(sorted_series[idx]) * 1000.0
            max_ms = float(max(series)) * 1000.0
            report[stage] = {
                "count": float(len(series)),
                "mean_ms": float(mean_ms),
                "p95_ms": float(p95_ms),
                "max_ms": float(max_ms),
            }
            if reset:
                self._perf_history[stage].clear()
        self._perf_last_report = report
        return report

    def get_perf_current(self) -> Dict[str, float]:
        report: Dict[str, float] = {}
        for stage in self._PERF_STAGES:
            series = self._perf_history.get(stage)
            if not series:
                report[stage] = 0.0
                continue
            report[stage] = float(float(series[-1]) * 1000.0)
        return report

    def _dsp_profile_add(self, task: Any, duration_s: float):
        dur = max(float(duration_s), 0.0)
        key = str(getattr(type(task), "__name__", "UnknownTask") or "UnknownTask")
        series = self._dsp_node_profile.get(key)
        if series is None:
            series = deque(maxlen=self._DSP_NODE_PROFILE_SAMPLES)
            self._dsp_node_profile[key] = series
        series.append(dur)

    def _record_source_prefetch(self, duration_s: float):
        dur = max(float(duration_s), 0.0)
        with self._source_prefetch_lock:
            self._source_prefetch_history.append(dur)

    @staticmethod
    def _id_metric_token(datablock) -> str:
        if datablock is None:
            return ""
        try:
            name = str(getattr(datablock, "name_full", "") or getattr(datablock, "name", "") or "")
        except Exception:
            name = ""
        if not name:
            return ""
        try:
            type_name = str(getattr(getattr(datablock, "bl_rna", None), "identifier", "") or type(datablock).__name__)
        except Exception:
            type_name = type(datablock).__name__
        return f"{type_name}:{name}"

    @classmethod
    def _extend_metric_tokens_from_id(cls, tokens: set[str], datablock):
        token = cls._id_metric_token(datablock)
        if token:
            tokens.add(token)
        if isinstance(datablock, bpy.types.Object):
            data_token = cls._id_metric_token(getattr(datablock, "data", None))
            if data_token:
                tokens.add(data_token)

    @classmethod
    def _resolve_geometry_metric_tokens(cls, scene, proxy_names: list[str] | set[str] | tuple[str, ...] | None = None) -> set[str]:
        tokens: set[str] = set()
        ace = getattr(scene, "audiocore", None)
        target_obj = getattr(ace, "gn_target_object", None) if ace else None
        if target_obj is not None:
            cls._extend_metric_tokens_from_id(tokens, target_obj)
        sonar_obj = bpy.data.objects.get(ACEProxyManager.SONAR_OBJECT_NAME)
        if sonar_obj is not None:
            cls._extend_metric_tokens_from_id(tokens, sonar_obj)
        if not tokens:
            for proxy_name in list(proxy_names or ()):
                proxy_obj = bpy.data.objects.get(str(proxy_name or ""))
                if proxy_obj is not None:
                    cls._extend_metric_tokens_from_id(tokens, proxy_obj)
        return tokens

    def _prune_pending_geometry_events(self, now_ts: float | None = None):
        ts_now = float(now_ts if now_ts is not None else time.perf_counter())
        ttl_s = float(max(self._BLENDER_LATENCY_EVENT_TTL_S, 0.1))
        self._pending_geometry_events = [
            event
            for event in self._pending_geometry_events
            if (ts_now - float(event.get("start_ts", 0.0) or 0.0)) <= ttl_s
        ]

    def _mark_geometry_eval_pending(self, scene, proxy_names: list[str] | set[str] | tuple[str, ...] | None = None):
        if not self._perf_enabled(scene):
            return
        target_tokens = self._resolve_geometry_metric_tokens(scene, proxy_names=proxy_names)
        if not target_tokens:
            return
        now_ts = time.perf_counter()
        self._prune_pending_geometry_events(now_ts)
        self._pending_geometry_events.append(
            {
                "scene_key": int(self._scene_key(scene)),
                "frame": int(self.current_frame),
                "start_ts": float(now_ts),
                "target_tokens": set(target_tokens),
            }
        )

    def _consume_geometry_eval_updates(self, scene, depsgraph):
        if not self._pending_geometry_events:
            return
        now_ts = time.perf_counter()
        self._prune_pending_geometry_events(now_ts)
        if not self._pending_geometry_events:
            return
        scene_key = int(self._scene_key(scene))
        updated_tokens: set[str] = set()
        for update in getattr(depsgraph, "updates", []) or []:
            self._extend_metric_tokens_from_id(updated_tokens, getattr(update, "id", None))
        if not updated_tokens:
            return
        matched_idx = -1
        for idx, event in enumerate(self._pending_geometry_events):
            if int(event.get("scene_key", 0) or 0) != scene_key:
                continue
            target_tokens = set(event.get("target_tokens", set()) or set())
            if target_tokens and target_tokens.intersection(updated_tokens):
                matched_idx = idx
                break
        if matched_idx < 0:
            return
        event = self._pending_geometry_events.pop(matched_idx)
        start_ts = float(event.get("start_ts", 0.0) or 0.0)
        if start_ts > 0.0:
            self._geometry_eval_history.append(max(now_ts - start_ts, 0.0))
            self._geometry_eval_last_frame = int(self.current_frame)

    def _mark_render_start(self, scene=None):
        if not self._perf_enabled(scene):
            self._render_start_ts = 0.0
            return
        self._render_start_ts = time.perf_counter()

    def _mark_render_complete(self, scene=None, cancelled: bool = False):
        start_ts = float(self._render_start_ts or 0.0)
        self._render_start_ts = 0.0
        if cancelled or start_ts <= 0.0 or not self._perf_enabled(scene):
            return
        self._render_history.append(max(time.perf_counter() - start_ts, 0.0))

    def get_dsp_node_profile_snapshot(self, reset: bool = False) -> Dict[str, Dict[str, float]]:
        report: Dict[str, Dict[str, float]] = {}
        for key, series in sorted(self._dsp_node_profile.items()):
            values = list(series)
            if not values:
                continue
            ordered = sorted(values)
            idx = int(round(0.95 * max(len(ordered) - 1, 0)))
            report[key] = {
                "count": float(len(values)),
                "mean_ms": float((sum(values) / max(len(values), 1)) * 1000.0),
                "p95_ms": float(ordered[idx] * 1000.0),
                "max_ms": float(max(values) * 1000.0),
            }
            if reset:
                series.clear()
        return report

    def get_runtime_diagnostics(self, reset: bool = False) -> Dict[str, float]:
        preview_diag = dict(self._preview_monitor.diagnostics_snapshot(reset=reset))
        worker_target = self._parallel_worker_count(None)
        executor = self._parallel_executor
        active_workers = int(getattr(executor, "_max_workers", 0) or 0) if executor is not None else 0
        geometry_values = list(self._geometry_eval_history)
        render_values = list(self._render_history)
        with self._source_prefetch_lock:
            prefetch_values = list(self._source_prefetch_history)
            if reset:
                self._source_prefetch_history.clear()
        if prefetch_values:
            ordered = sorted(float(v) for v in prefetch_values)
            idx = int(round(0.95 * max(len(ordered) - 1, 0)))
            source_prefetch_mean_ms = float((sum(ordered) / max(len(ordered), 1)) * 1000.0)
            source_prefetch_p95_ms = float(ordered[idx] * 1000.0)
        else:
            source_prefetch_mean_ms = 0.0
            source_prefetch_p95_ms = 0.0
        if geometry_values:
            geometry_ordered = sorted(float(v) for v in geometry_values)
            geometry_idx = int(round(0.95 * max(len(geometry_ordered) - 1, 0)))
            geometry_mean_ms = float((sum(geometry_ordered) / max(len(geometry_ordered), 1)) * 1000.0)
            geometry_p95_ms = float(geometry_ordered[geometry_idx] * 1000.0)
        else:
            geometry_mean_ms = 0.0
            geometry_p95_ms = 0.0
        if render_values:
            render_ordered = sorted(float(v) for v in render_values)
            render_idx = int(round(0.95 * max(len(render_ordered) - 1, 0)))
            render_mean_ms = float((sum(render_ordered) / max(len(render_ordered), 1)) * 1000.0)
            render_p95_ms = float(render_ordered[render_idx] * 1000.0)
        else:
            render_mean_ms = 0.0
            render_p95_ms = 0.0
        report = {
            "parallel_queue_pressure_drops": float(self._parallel_queue_pressure_drops),
            "parallel_job_timeouts": float(self._parallel_job_timeouts),
            "parallel_job_exceptions": float(self._parallel_job_exceptions),
            "parallel_pool_restarts": float(self._parallel_restart_count),
            "parallel_stale_job_drops": float(self._parallel_stale_job_drops),
            "parallel_stale_result_drops": float(self._parallel_stale_result_drops),
            "parallel_result_accepts": float(self._parallel_result_accepts),
            "parallel_disabled_sources": float(len(self._parallel_disabled_sources)),
            "parallel_worker_target": float(worker_target),
            "parallel_worker_active": float(active_workers),
            "parallel_inflight_limit": float(self._parallel_inflight_limit(None)),
            "parallel_payload_slots": float(len(self._parallel_payload_slots)),
            "source_prefetch_mean_ms": float(source_prefetch_mean_ms),
            "source_prefetch_p95_ms": float(source_prefetch_p95_ms),
            "blender_geometry_eval_mean_ms": float(geometry_mean_ms),
            "blender_geometry_eval_p95_ms": float(geometry_p95_ms),
            "blender_render_mean_ms": float(render_mean_ms),
            "blender_render_p95_ms": float(render_p95_ms),
        }
        report.update(preview_diag)
        if reset:
            self._parallel_queue_pressure_drops = 0
            self._parallel_job_timeouts = 0
            self._parallel_job_exceptions = 0
            self._parallel_stale_job_drops = 0
            self._parallel_stale_result_drops = 0
            self._parallel_result_accepts = 0
            self._geometry_eval_history.clear()
            self._render_history.clear()
        return report

    def get_runtime_diagnostics_current(self) -> Dict[str, float]:
        now_ts = time.perf_counter()
        preview_diag = dict(self._preview_monitor.current_snapshot())
        with self._source_prefetch_lock:
            source_prefetch_ms = float(self._source_prefetch_history[-1] * 1000.0) if self._source_prefetch_history else 0.0
        geometry_ms = 0.0
        if self._pending_geometry_events:
            latest_pending_ts = max(
                float(event.get("start_ts", 0.0) or 0.0)
                for event in self._pending_geometry_events
            )
            if latest_pending_ts > 0.0:
                geometry_ms = float(max(now_ts - latest_pending_ts, 0.0) * 1000.0)
        elif self._geometry_eval_history:
            current_frame = int(self.current_frame)
            last_frame = int(self._geometry_eval_last_frame)
            if (current_frame - last_frame) <= int(max(self._CURRENT_LATENCY_SAMPLE_FRESHNESS_FRAMES, 0)):
                geometry_ms = float(self._geometry_eval_history[-1] * 1000.0)
        render_ms = 0.0
        if self._is_rendering_job() and float(self._render_start_ts or 0.0) > 0.0:
            render_ms = float(max(now_ts - float(self._render_start_ts), 0.0) * 1000.0)
        return {
            "source_prefetch_ms": float(source_prefetch_ms),
            "preview_branch_latency_ms": float(preview_diag.get("preview_branch_latency_ms", 0.0) or 0.0),
            "preview_output_buffer_ms": float(preview_diag.get("preview_output_buffer_ms", 0.0) or 0.0),
            "preview_callback_ms": float(preview_diag.get("preview_callback_ms", 0.0) or 0.0),
            "blender_geometry_eval_ms": float(geometry_ms),
            "blender_render_ms": float(render_ms),
        }

    @staticmethod
    def _parallel_shard_attr_keys(
        attr_keys: List[str],
        source_groups: Optional[Dict[str, str]] = None,
    ) -> List[List[str]]:
        """Partition attr_keys into per-worker shards.

        When source_groups is provided (a dict mapping each attr_key to its
        primary source node ID), keys are bucketed by source so that each
        source's DSP chain runs in a dedicated worker job.  This allows
        multi-source scenes to exploit all available worker processes in
        parallel rather than serialising through a single job.

        Single-source graphs (or graphs where source grouping is unknown)
        fall back to the legacy single-shard path to avoid extra scheduling
        overhead.
        """
        keys = [str(v) for v in list(attr_keys or []) if str(v)]
        if not keys:
            return []
        if not source_groups or len(set(source_groups.values())) <= 1:
            # Single-source (or unknown) graph: keep legacy single-shard behavior.
            return [keys]
        # Shard by source ancestry so each source group runs in a separate worker job.
        buckets: Dict[str, List[str]] = {}
        ungrouped: List[str] = []
        for k in keys:
            group = source_groups.get(k)
            if group:
                buckets.setdefault(str(group), []).append(k)
            else:
                ungrouped.append(k)
        shards = [v for v in buckets.values() if v]
        if ungrouped:
            shards.append(ungrouped)
        return shards if shards else [keys]

    @classmethod
    def _file_working_sample_rate_cap(cls, scene_or_engine_settings=None) -> int:
        settings = cls._engine_settings(scene_or_engine_settings)
        default_cap = int(cls._FILE_WORKING_SAMPLE_RATE_CAP)
        try:
            raw = str(
                getattr(settings, "ace_file_working_sample_rate", str(default_cap)) or str(default_cap)
            ).strip().lower()
        except Exception:
            raw = str(default_cap)
        if raw in {"native", "off", "0"}:
            return 0
        try:
            return max(int(raw), 0)
        except Exception:
            return default_cap

    @classmethod
    def _cap_source_sample_rate(
        cls,
        source_type: str,
        sample_rate: int,
        scene_or_engine_settings=None,
    ) -> int:
        sr = max(int(sample_rate or SAMPLE_RATE), 1)
        if str(source_type or "").strip().upper() == "FILE":
            cap_sr = cls._file_working_sample_rate_cap(scene_or_engine_settings)
            if cap_sr > 0:
                sr = min(sr, int(cap_sr))
        return sr

    @staticmethod
    def _source_original_sample_rate(source) -> int:
        source_type = str(getattr(source, "source_type", "") or "").strip().upper()
        if source_type == "FILE":
            try:
                sr = int(getattr(source, "native_sample_rate", 0) or 0)
            except Exception:
                sr = 0
            if sr > 0:
                return sr
        try:
            return int(getattr(source, "sample_rate", 0) or 0)
        except Exception:
            return 0

    def _source_task_target_sample_rate(self, task: Optional[SourceTask]) -> Optional[int]:
        if not isinstance(task, SourceTask):
            return None
        ctx = getattr(task, "source_context", None)
        source_type = str(getattr(ctx, "source_type", "") or "").strip().upper()
        if source_type != "FILE":
            return None
        cap_sr = self._file_working_sample_rate_cap(getattr(task, "engine_settings", None))
        return int(cap_sr) if cap_sr > 0 else None

    def _source_task_requires_local_frontend(self, task: Optional[SourceTask]) -> bool:
        if not isinstance(task, SourceTask):
            return True
        target_graph = None
        for graph in self.graphs.values():
            node_id = str(getattr(task, "node_id", "") or "")
            if graph.tasks.get(node_id) is task:
                target_graph = graph
                break
        if target_graph is None:
            return True

        scene = getattr(getattr(bpy, "context", None), "scene", None)
        parallel_mode = self._parallel_mode(scene)
        parallel_offload_graph = (
            parallel_mode == "processes_dsp"
            and bool(getattr(target_graph, "parallel_ir", {}))
            and bool(getattr(target_graph, "parallel_offload_attr_keys", []))
        )
        if not parallel_offload_graph:
            return True
        local_keys = list(getattr(target_graph, "parallel_local_attr_keys", []) or [])
        return bool(local_keys)

    @classmethod
    def _is_parallel_offload_attr_candidate(cls, node: Optional[ACEAbstractNode]) -> bool:
        """
        Runtime offload policy:
        - offload all numeric Layer 2 outputs so the full upstream chain,
          including source preprocessing, can be evaluated in worker jobs.
        """
        if node is None:
            return False
        category = str(getattr(node, "category", "") or "")
        node_type = str(getattr(node, "type", "") or "")
        if category == "utilities" and node_type == "src_specs":
            return True
        if category == "utilities" and node_type == "stream_accumulator":
            # Stateful running-sum nodes must remain local. Their output should
            # never jump when the engine temporarily falls back between worker
            # and main-thread execution paths.
            return False
        if category != "stream":
            return False
        if node_type in cls._PARALLEL_OFFLOAD_STREAM_TYPES:
            return True
        # Future stream operators should default to worker offload.
        return True

    def _rebuild_graph_output_slot_specs(self, graph: ACEAudioGraph):
        specs: list[tuple[str, str, int]] = []
        for attr_key in list(getattr(graph, "output_attr_keys", []) or []):
            node_id, _, sock_idx_str = str(attr_key).partition(":")
            if node_id not in graph.tasks:
                continue
            try:
                socket_idx = int(sock_idx_str) if sock_idx_str else 0
            except Exception:
                socket_idx = 0
            specs.append((str(attr_key), str(node_id), int(socket_idx)))
        graph.output_slot_specs = specs
        graph.output_slot_specs_by_subset = {}

    def _graph_output_slot_specs(
        self,
        graph: ACEAudioGraph,
        attr_keys: Optional[List[str]] = None,
    ) -> List[tuple[str, str, int]]:
        if attr_keys is None:
            specs = list(getattr(graph, "output_slot_specs", []) or [])
            if specs:
                return specs
            self._rebuild_graph_output_slot_specs(graph)
            return list(getattr(graph, "output_slot_specs", []) or [])

        keys = [str(v) for v in list(attr_keys or []) if str(v)]
        if not keys:
            return []
        key_tuple = tuple(keys)
        subset_cache = getattr(graph, "output_slot_specs_by_subset", None)
        if not isinstance(subset_cache, OrderedDict):
            subset_cache = OrderedDict()
            graph.output_slot_specs_by_subset = subset_cache
        cached = subset_cache.get(key_tuple)
        if isinstance(cached, list):
            # Move to end (most-recently-used) for true LRU eviction ordering.
            subset_cache.move_to_end(key_tuple)
            return cached

        full_specs = list(getattr(graph, "output_slot_specs", []) or [])
        if not full_specs:
            self._rebuild_graph_output_slot_specs(graph)
            full_specs = list(getattr(graph, "output_slot_specs", []) or [])
        full_map = {attr_key: (node_id, socket_idx) for attr_key, node_id, socket_idx in full_specs}

        subset: list[tuple[str, str, int]] = []
        for attr_key in keys:
            rec = full_map.get(attr_key)
            if rec is not None:
                subset.append((attr_key, str(rec[0]), int(rec[1])))
                continue
            node_id, _, sock_idx_str = str(attr_key).partition(":")
            if node_id not in graph.tasks:
                continue
            try:
                socket_idx = int(sock_idx_str) if sock_idx_str else 0
            except Exception:
                socket_idx = 0
            subset.append((str(attr_key), str(node_id), int(socket_idx)))
        subset_cache[key_tuple] = subset
        # Evict the least-recently-used entry when the cache exceeds the limit.
        if len(subset_cache) > 32:
            subset_cache.popitem(last=False)
        return subset

    @staticmethod
    def _sanitize_ir_value(value: Any):
        if value is None:
            return None
        if isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [ACEAudioEngine._sanitize_ir_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): ACEAudioEngine._sanitize_ir_value(v) for k, v in value.items()}
        return str(value)

    @staticmethod
    def _parallel_graph_uid(graph: ACEAudioGraph) -> str:
        key = (
            str(getattr(graph, "source_uid", "") or "")
            or str(getattr(getattr(graph, "source_context", None), "stable_id", "") or "")
            or str(getattr(graph, "proxy_name", "") or "")
        )
        return key if key else f"graph:{id(graph)}"

    def _parallel_worker_count(self, scene) -> int:
        """Return the target process-pool worker count for the current machine.

        Respects the ``ACE_PARALLEL_WORKERS`` environment variable override.
        Otherwise uses ``os.cpu_count() - _PARALLEL_WORKER_HEADROOM_CORES`` so
        one core stays free for Blender's main thread and render pipeline.
        """
        try:
            override = int(os.environ.get("ACE_PARALLEL_WORKERS", "0") or 0)
        except Exception:
            override = 0
        cpu_count = max(int(os.cpu_count() or 1), 1)
        if override > 0:
            return max(1, min(int(override), cpu_count))
        reserve = int(self._PARALLEL_WORKER_HEADROOM_CORES) if cpu_count > 1 else 0
        return max(1, cpu_count - max(reserve, 0))

    def _parallel_inflight_limit(self, scene=None) -> int:
        # Always honour the spec constant regardless of worker count.
        # The previous implementation returned max(..., 3) for ≥4 workers,
        # silently exceeding _PARALLEL_MAX_INFLIGHT_PER_SOURCE = 2 and causing
        # SHM slot exhaustion under high-core-count machines.
        return max(min(int(self._PARALLEL_MAX_INFLIGHT_PER_SOURCE), 2), 1)

    def _parallel_clear_runtime_state(self, keep_last_valid: bool = True):
        for meta in list(self._parallel_future_meta.values()):
            self._parallel_cleanup_ipc_handles(meta)
        self._parallel_inflight.clear()
        self._parallel_future_meta.clear()
        self._parallel_results.clear()
        self._parallel_failures.clear()
        self._parallel_disabled_sources.clear()
        self._parallel_graph_ir_required.clear()
        self._parallel_frame_payload_slots.clear()
        self._parallel_payload_slot_refs.clear()
        self._parallel_payload_slot_counters.clear()
        self._parallel_release_payload_slots()
        if not keep_last_valid:
            self._parallel_last_valid.clear()
            self._parallel_last_valid_frame.clear()

    def _mark_parallel_ir_required_for_all_graphs(self):
        self._parallel_graph_ir_required.clear()
        for graph in self.graphs.values():
            graph_key = str(getattr(graph, "parallel_graph_uid", "") or self._parallel_graph_uid(graph))
            if not graph_key:
                continue
            if bool(getattr(graph, "parallel_ir", {})) and bool(getattr(graph, "parallel_offload_attr_keys", [])):
                self._parallel_graph_ir_required.add(graph_key)

    def _shutdown_parallel_pool(self, wait: bool = False):
        executor = self._parallel_executor
        self._parallel_executor = None
        wait_effective = bool(wait)
        if not wait_effective:
            # Realtime paths must stay non-blocking. Best-effort cancel pending jobs.
            for future_obj in list(self._parallel_future_meta.keys()):
                try:
                    future_obj.cancel()
                except Exception:
                    pass
            # Terminate worker processes immediately so they release SHM file
            # descriptors and CPU. Without this, already-running workers outlive
            # the executor shutdown and hold /dev/shm blocks until completion.
            try:
                procs = dict(getattr(executor, "_processes", {}) or {})
                for proc in procs.values():
                    try:
                        proc.terminate()
                    except Exception:
                        pass
            except Exception:
                pass
        if executor is not None:
            try:
                executor.shutdown(wait=wait_effective, cancel_futures=True)
            except Exception:
                pass
        self._parallel_clear_runtime_state(keep_last_valid=False)

    def _parallel_reset_generation(self):
        self._parallel_generation_id = int(self._parallel_generation_id) + 1
        self._parallel_job_seq = 0
        # NOTE: _parallel_restart_count intentionally NOT reset here.
        # Resetting it on every compile() would bypass _PARALLEL_RESTART_LIMIT
        # under auto-compile mode (frequent recompiles = unlimited pool restarts).
        # The count is only cleared in __init__ (engine session boundary).
        self._parallel_pool_failed = False
        self._parallel_clear_runtime_state(keep_last_valid=False)

    def _ensure_parallel_pool(self, scene) -> bool:
        """Ensure the ProcessPoolExecutor is running; create it if needed.

        Returns True when the pool is ready for job submission.  Returns False
        when parallel mode is disabled, the pool has permanently failed, or a
        new spawn attempt raises an exception (in which case
        ``_parallel_pool_failed`` is set to prevent repeated failures).
        """
        if self._parallel_mode(scene) != "processes_dsp":
            self._shutdown_parallel_pool(wait=False)
            return False
        if self._parallel_pool_failed:
            return False
        if self._parallel_executor is not None:
            return True
        worker_count = self._parallel_worker_count(scene)
        try:
            spawn_ctx = multiprocessing.get_context("spawn")
            self._parallel_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=max(int(worker_count), 1),
                mp_context=spawn_ctx,
            )
            # New pools start with empty worker caches and need a first IR push.
            self._mark_parallel_ir_required_for_all_graphs()
            return True
        except Exception as e:
            self._parallel_executor = None
            self._parallel_pool_failed = True
            self._log_runtime(
                "parallel_init",
                f"Processes-DSP unavailable, reusing last-valid offload values: {e}",
                "WARNING",
                min_interval=0.0,
            )
            return False

    def _restart_parallel_pool(self, scene, reason: str, source_key: str = "") -> bool:
        if self._parallel_restart_count >= self._PARALLEL_RESTART_LIMIT:
            if source_key:
                self._parallel_disabled_sources.add(source_key)
            return False
        # Enforce exponential backoff between restart attempts to absorb transient OS
        # errors (OOM spikes, process spawn latency) without burning the restart budget.
        now = time.monotonic()
        if now < self._parallel_restart_backoff_until:
            # Still in the backoff window — defer this attempt; reuse last-valid values.
            return False
        self._parallel_restart_count += 1
        backoff_idx = min(self._parallel_restart_count - 1, len(self._PARALLEL_RESTART_BACKOFF_S) - 1)
        delay_s = float(self._PARALLEL_RESTART_BACKOFF_S[backoff_idx])
        self._parallel_restart_backoff_until = now + delay_s
        self._log_runtime(
            "parallel_restart",
            f"Restarting Processes-DSP pool ({reason}), "
            f"attempt {self._parallel_restart_count}/{self._PARALLEL_RESTART_LIMIT}, "
            f"backoff {delay_s:.1f}s.",
            "WARNING",
            min_interval=0.0,
        )
        self._shutdown_parallel_pool(wait=False)
        self._parallel_pool_failed = False
        ok = self._ensure_parallel_pool(scene)
        if not ok and source_key:
            self._parallel_disabled_sources.add(source_key)
        return ok

    def _prepare_parallel_graph_ir(self, graph: ACEAudioGraph, abstract_tree: ACEAbstractTree):
        graph.parallel_graph_uid = self._parallel_graph_uid(graph)
        graph.parallel_ir = {}
        graph.parallel_ir_digest = ""
        graph.parallel_offload_attr_keys = []
        graph.parallel_output_shards = []
        graph.parallel_local_attr_keys = list(getattr(graph, "output_attr_keys", []) or [])
        self._parallel_graph_ir_required.discard(str(getattr(graph, "parallel_graph_uid", "") or ""))

        if abstract_tree is None or self.writer is None:
            return

        offload_attr_keys: list[str] = []
        offload_node_ids: set[str] = set()
        for attr_key in list(getattr(graph, "output_attr_keys", []) or []):
            node_id, _, sock_idx_str = attr_key.partition(":")
            node = abstract_tree.nodes.get(node_id)
            if not node:
                continue
            if not self._is_parallel_offload_attr_candidate(node):
                continue
            try:
                _ = int(sock_idx_str) if sock_idx_str else 0
            except Exception:
                continue
            offload_attr_keys.append(attr_key)
            offload_node_ids.add(node_id)

        if not offload_attr_keys:
            return

        reverse_adj: Dict[str, set[str]] = {}
        for link in getattr(abstract_tree, "links", []) or []:
            reverse_adj.setdefault(str(link.target_node_id), set()).add(str(link.source_node_id))

        required_node_ids: set[str] = set()
        stack = list(offload_node_ids)
        graph_task_ids = set(getattr(graph, "tasks", {}).keys())
        while stack:
            node_id = str(stack.pop())
            if node_id in required_node_ids:
                continue
            required_node_ids.add(node_id)
            for src_id in reverse_adj.get(node_id, set()):
                if src_id in graph_task_ids and src_id not in required_node_ids:
                    stack.append(src_id)

        nodes_ir: Dict[str, Dict[str, Any]] = {}
        source_node_ids: list[str] = []
        source_ir: Dict[str, Dict[str, Any]] = {}
        for node_id in sorted(required_node_ids):
            node = abstract_tree.nodes.get(node_id)
            if node is None:
                continue
            params = {
                str(k): self._sanitize_ir_value(v)
                for k, v in dict(getattr(node, "parameters", {}) or {}).items()
            }
            # The worker's SourceSpecs replica needs BPM/FPS for beat phase; each
            # graph takes them from its own source.
            if node.category == "utilities" and node.type == "src_specs":
                _specs_ctx = getattr(graph, "source_context", None)
                params["scene_bpm"] = float(getattr(_specs_ctx, "bpm", 120.0) or 120.0)
                params["scene_fps"] = float(getattr(self, "_current_scene_fps", 24.0) or 24.0)
                params["scene_beats_per_bar"] = float(getattr(_specs_ctx, "beats_per_bar", 4.0) or 4.0)
            nodes_ir[node_id] = {
                "category": str(getattr(node, "category", "") or ""),
                "type": str(getattr(node, "type", "") or ""),
                "parameters": params,
            }
            if node.category == "source_mngmt" and node.type == "src":
                source_node_ids.append(node_id)
                src_task = graph.tasks.get(node_id)
                src_ctx = getattr(src_task, "source_context", None) if isinstance(src_task, SourceTask) else None
                sr_hint = int(
                    getattr(src_task, "sample_rate", 0) if isinstance(src_task, SourceTask) else 0
                    or getattr(src_ctx, "sample_rate", 0)
                    or SAMPLE_RATE
                )
                source_ir[node_id] = {
                    "sample_rate": int(max(sr_hint, 1)),
                    "lineage": [node_id],
                    "spectrum_token": [node_id],
                }

        links_ir: list[tuple[str, int, str, int]] = []
        for link in getattr(abstract_tree, "links", []) or []:
            src_id = str(getattr(link, "source_node_id", "") or "")
            dst_id = str(getattr(link, "target_node_id", "") or "")
            if src_id not in nodes_ir or dst_id not in nodes_ir:
                continue
            links_ir.append(
                (
                    src_id,
                    int(getattr(link, "source_socket_idx", 0) or 0),
                    dst_id,
                    int(getattr(link, "target_socket_idx", 0) or 0),
                )
            )

        output_specs: Dict[str, Dict[str, Any]] = {}
        for attr_key in offload_attr_keys:
            node_id, _, sock_idx_str = attr_key.partition(":")
            if node_id not in nodes_ir:
                continue
            info = self.writer.output_defs.get(attr_key, {}) if self.writer else {}
            output_specs[attr_key] = {
                "node_id": node_id,
                "socket_idx": int(sock_idx_str) if sock_idx_str else 0,
                "attr_type": str(info.get("type", "FLOAT") or "FLOAT"),
            }

        if not output_specs:
            return

        graph.parallel_ir = {
            "nodes": nodes_ir,
            "links": links_ir,
            "source_node_ids": sorted(set(source_node_ids)),
            "source_ir": source_ir,
            "output_specs": output_specs,
        }
        graph.parallel_offload_attr_keys = sorted(output_specs.keys())
        # Build per-source grouping: trace each attr_key's node back to its
        # primary source ancestor so multi-source graphs can be sharded across
        # multiple worker processes (one shard per source lineage).
        attr_key_source_group: Dict[str, str] = {}
        _src_id_set = set(source_node_ids)
        for attr_key in graph.parallel_offload_attr_keys:
            nid, _, _ = attr_key.partition(":")
            visited_s: set[str] = set()
            stack_s = [nid]
            found_src = ""
            while stack_s and not found_src:
                cur = stack_s.pop()
                if cur in visited_s:
                    continue
                visited_s.add(cur)
                if cur in _src_id_set:
                    found_src = cur
                    break
                for parent in reverse_adj.get(cur, set()):
                    if parent not in visited_s:
                        stack_s.append(parent)
            if found_src:
                attr_key_source_group[attr_key] = found_src
        graph.parallel_output_shard_groups = attr_key_source_group
        graph.parallel_output_shards = self._parallel_shard_attr_keys(
            graph.parallel_offload_attr_keys,
            source_groups=attr_key_source_group if len(_src_id_set) > 1 else None,
        )
        offload_set = set(graph.parallel_offload_attr_keys)
        graph.parallel_local_attr_keys = [
            key for key in list(getattr(graph, "output_attr_keys", []) or [])
            if key not in offload_set
        ]
        try:
            digest_payload = json.dumps(
                graph.parallel_ir,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            graph.parallel_ir_digest = hashlib.blake2b(digest_payload, digest_size=8).hexdigest()
        except Exception:
            graph.parallel_ir_digest = hashlib.blake2b(
                str(graph.parallel_ir).encode("utf-8", "ignore"),
                digest_size=8,
            ).hexdigest()
        if graph.parallel_graph_uid:
            self._parallel_graph_ir_required.add(str(graph.parallel_graph_uid))

    def _refresh_specs_ir_bpm(self) -> None:
        """Patch scene_bpm / scene_beats_per_bar / scene_fps in each graph's
        parallel IR so _WorkerSourceSpecsPassthroughTask always sees the current
        per-source BPM without requiring a full graph recompile.

        Called each frame from execute_next_frame() after _current_scene_fps is
        updated but before _maybe_compile_dirty() so any compile triggered this
        frame starts with correct values already in place.
        """
        fps = float(getattr(self, "_current_scene_fps", 24.0) or 24.0)
        for graph in self.graphs.values():
            parallel_ir = getattr(graph, "parallel_ir", None)
            if not parallel_ir:
                continue
            src_ctx = getattr(graph, "source_context", None)
            if src_ctx is None:
                continue
            bpm = float(getattr(src_ctx, "bpm", 120.0) or 120.0)
            bpb = float(getattr(src_ctx, "beats_per_bar", 4.0) or 4.0)
            changed = False
            nodes_ir = parallel_ir.get("nodes", {})
            for node_info in nodes_ir.values():
                if (
                    node_info.get("category") != "utilities"
                    or node_info.get("type") != "src_specs"
                ):
                    continue
                params = node_info.get("parameters")
                if not isinstance(params, dict):
                    continue
                old_bpm = float(params.get("scene_bpm", 120.0) or 120.0)
                old_bpb = float(params.get("scene_beats_per_bar", 4.0) or 4.0)
                old_fps = float(params.get("scene_fps", 24.0) or 24.0)
                if abs(old_bpm - bpm) > 1e-6 or abs(old_bpb - bpb) > 1e-6 or abs(old_fps - fps) > 1e-6:
                    params["scene_bpm"] = bpm
                    params["scene_fps"] = fps
                    params["scene_beats_per_bar"] = bpb
                    changed = True
            if changed:
                graph_key = str(getattr(graph, "parallel_graph_uid", "") or "")
                if graph_key:
                    self._parallel_graph_ir_required.add(graph_key)

    def _build_parallel_source_payload(self, chunk, block_size: int, sample_rate: int) -> Dict[str, Any]:
        """Serialise an AudioChunk into a picklable worker payload dict.

        Pads or trims the mono signal to exactly *block_size* samples.  Stereo
        is normalised to channels-first layout (shape ``(C, N)``).  Optional
        fields (``frontend_config``, ``band_indices``, ``band_hz_bounds``) are
        included only when present on the chunk.

        Args:
            chunk: Source AudioChunk to serialise.
            block_size: Expected number of mono samples in the worker.
            sample_rate: Sample rate to record in the payload.

        Returns:
            Dict suitable for passing to ``evaluate_parallel_job``.
        """
        bs = max(int(block_size), 1)
        sr = max(int(sample_rate), 1)
        mono_raw = getattr(chunk, "mono", chunk)
        mono = np.asarray(mono_raw, dtype=np.float32).reshape(-1)
        if mono.shape[0] < bs:
            padded = np.zeros(bs, dtype=np.float32)
            if mono.shape[0] > 0:
                padded[: mono.shape[0]] = mono
            mono = padded
        elif mono.shape[0] > bs:
            mono = mono[:bs]

        stereo = getattr(chunk, "stereo", None)
        stereo_payload = None
        if isinstance(stereo, np.ndarray):
            arr = np.asarray(stereo, dtype=np.float32)
            if arr.ndim == 2:
                if arr.shape[1] == mono.shape[0]:
                    stereo_payload = arr.astype(np.float32, copy=False)
                elif arr.shape[0] == mono.shape[0]:
                    stereo_payload = arr.T.astype(np.float32, copy=False)

        payload: Dict[str, Any] = {
            "mono": mono.astype(np.float32, copy=False),
            "stereo": stereo_payload,
            "sample_rate": sr,
            "latency_ms": float(getattr(chunk, "latency_ms", 0.0) or 0.0),
        }
        frontend_config = getattr(chunk, "frontend_config", None)
        if isinstance(frontend_config, (list, tuple)) and len(frontend_config) >= 4:
            try:
                payload["frontend_config"] = (
                    float(frontend_config[0] or 0.0),
                    float(frontend_config[1] or 0.0),
                    float(frontend_config[2] or 0.0),
                    float(frontend_config[3] or 0.0),
                )
            except Exception:
                pass
        band_indices = getattr(chunk, "band_indices", None)
        if band_indices is not None:
            try:
                band_arr = np.asarray(band_indices, dtype=np.int64).reshape(-1)
                payload["band_indices"] = band_arr
                payload["band_index_fft_size"] = int(
                    max(int(getattr(chunk, "band_index_fft_size", 0) or 0), 1)
                )
            except Exception:
                pass
        band_hz_bounds = getattr(chunk, "band_hz_bounds", None)
        if isinstance(band_hz_bounds, (list, tuple)) and len(band_hz_bounds) >= 2:
            try:
                low_hz = float(band_hz_bounds[0])
                high_hz = float(band_hz_bounds[1])
                if high_hz > low_hz:
                    payload["band_hz_bounds"] = (low_hz, high_hz)
            except Exception:
                pass
        return payload

    def _parallel_drop_future(self, future_obj):
        meta = self._parallel_future_meta.pop(future_obj, None)
        if not meta:
            return
        self._parallel_cleanup_ipc_handles(meta)
        queue_key = str(meta.get("queue_key", "") or meta.get("graph_key", "") or "")
        pending = self._parallel_inflight.get(queue_key)
        if pending:
            try:
                pending.remove(future_obj)
            except ValueError:
                pass
            if not pending:
                self._parallel_inflight.pop(queue_key, None)

    def _parallel_poll_futures(self, scene, current_frame: int):
        if not self._parallel_future_meta:
            return
        now = time.time()
        restart_source = ""
        restart_reason = ""

        for future_obj in list(self._parallel_future_meta.keys()):
            meta = dict(self._parallel_future_meta.get(future_obj, {}) or {})
            graph_key = str(meta.get("graph_key", "") or "")
            owner_graph_key = str(meta.get("owner_graph_key", "") or graph_key)
            submit_ts = float(meta.get("submit_ts", now) or now)
            undo_generation = int(meta.get("undo_generation", getattr(self, "undo_generation", 0)) or 0)
            if undo_generation != int(getattr(self, "undo_generation", 0) or 0):
                self._parallel_drop_future(future_obj)
                continue
            stale_budget_s = max(float(self._PARALLEL_STALE_JOB_MAX_MS) / 1000.0, 0.001)
            age_s = max(float(now - submit_ts), 0.0)

            if (not future_obj.done()) and age_s > stale_budget_s:
                # Hard-realtime contract: never keep over-budget jobs alive in
                # frame tracking. Drop immediately, regardless of cancel result.
                if not bool(meta.get("stale_job", False)):
                    meta["stale_job"] = True
                    self._parallel_future_meta[future_obj] = dict(meta)
                    self._parallel_stale_job_drops += 1
                try:
                    future_obj.cancel()
                except Exception:
                    pass
                self._parallel_drop_future(future_obj)
                continue

            if future_obj.done():
                if bool(meta.get("stale_job", False)):
                    self._parallel_drop_future(future_obj)
                    continue
                try:
                    payload = future_obj.result(timeout=0.0)
                except Exception as e:
                    self._parallel_job_exceptions += 1
                    self._parallel_drop_future(future_obj)
                    fail_key = owner_graph_key or "unknown"
                    fail_count = int(self._parallel_failures.get(fail_key, 0)) + 1
                    self._parallel_failures[fail_key] = fail_count
                    if isinstance(e, BrokenProcessPool):
                        restart_source = fail_key
                        restart_reason = f"worker crash: {e}"
                        break
                    if fail_count <= 1 and self._parallel_restart_count < self._PARALLEL_RESTART_LIMIT:
                        restart_source = fail_key
                        restart_reason = f"worker exception: {e}"
                        break
                    if fail_key:
                        self._parallel_disabled_sources.add(fail_key)
                    self._log_runtime(
                        f"parallel_job_error:{fail_key}",
                        f"Parallel job failed for '{fail_key}', reusing last-valid offload values: {e}",
                        "WARNING",
                        min_interval=0.0,
                    )
                    continue

                self._parallel_drop_future(future_obj)
                if not isinstance(payload, dict):
                    continue
                generation_id = utils.as_int(payload.get("generation_id"), -1)
                if generation_id != int(self._parallel_generation_id):
                    continue
                res_key = str(payload.get("graph_uid", "") or graph_key)
                if bool(payload.get("needs_graph_ir", False)):
                    if res_key:
                        self._parallel_graph_ir_required.add(res_key)
                    continue
                worker_elapsed_s = float(payload.get("elapsed_s", 0.0) or 0.0)
                if worker_elapsed_s > 0.0:
                    try:
                        self._perf_add("dsp", worker_elapsed_s)
                    except Exception:
                        pass
                frame_id = utils.as_int(payload.get("frame_id"), -1)
                if frame_id < (int(current_frame) - int(self._PARALLEL_STALE_FRAME_GRACE)):
                    self._parallel_stale_result_drops += 1
                    continue
                values = dict(payload.get("values", {}) or {})
                if not res_key:
                    continue
                self._parallel_graph_ir_required.discard(res_key)
                if values:
                    self._parallel_result_accepts += 1
                    merged_last = dict(self._parallel_last_valid.get(res_key, {}) or {})
                    merged_last.update(values)
                    self._parallel_last_valid[res_key] = merged_last
                    self._parallel_last_valid_frame[res_key] = int(frame_id)
                frame_map = self._parallel_results.setdefault(res_key, {})
                merged_frame = dict(frame_map.get(frame_id, {}) or {})
                merged_frame.update(values)
                frame_map[frame_id] = merged_frame
                if len(frame_map) > 4:
                    for old_frame in sorted(frame_map.keys())[:-4]:
                        frame_map.pop(old_frame, None)
                self._parallel_failures.pop(res_key, None)
                continue

        if restart_reason:
            self._restart_parallel_pool(scene, restart_reason, source_key=restart_source)

        # Pause-drain: when Blender is paused (no new submits for several frames),
        # drain all completed futures so SHM ref-counts reach zero and blocks are freed.
        _PAUSE_DRAIN_FRAMES = 3
        last_submit = int(self._parallel_last_submit_frame)
        if (int(current_frame) - last_submit) > _PAUSE_DRAIN_FRAMES and self._parallel_future_meta:
            for future_obj in list(self._parallel_future_meta.keys()):
                if future_obj.done():
                    self._parallel_drop_future(future_obj)

        if self._parallel_results:
            threshold = int(current_frame) - 4
            for graph_key, frame_map in list(self._parallel_results.items()):
                for old_frame in [fid for fid in frame_map.keys() if int(fid) < threshold]:
                    frame_map.pop(old_frame, None)
                if not frame_map:
                    self._parallel_results.pop(graph_key, None)

    def _parallel_submit_graph_job(
        self,
        scene,
        graph: ACEAudioGraph,
        graph_key: str,
        queue_key: str,
        owner_graph_key: str,
        frame_index: int,
        source_chunks: Dict[str, Any],
        output_keys: Optional[List[str]] = None,
        include_graph_ir: Optional[bool] = None,
    ) -> bool:
        if owner_graph_key in self._parallel_disabled_sources:
            return False
        if not getattr(graph, "parallel_ir", {}):
            return False
        requested_output_keys = [str(v) for v in list(output_keys or []) if str(v)]
        if not requested_output_keys:
            requested_output_keys = list(getattr(graph, "parallel_offload_attr_keys", []) or [])
        if not requested_output_keys:
            return False
        if not self._ensure_parallel_pool(scene):
            return False
        if self._parallel_executor is None:
            return False

        queue_key = str(queue_key or graph_key)
        inflight_limit = self._parallel_inflight_limit(scene)
        pending = self._parallel_inflight.get(queue_key)
        if pending is None or int(getattr(pending, "maxlen", 0) or 0) != int(inflight_limit):
            pending = deque(list(pending or []), maxlen=max(int(inflight_limit), 1))
            self._parallel_inflight[queue_key] = pending
        for fut in list(pending):
            if fut.done():
                self._parallel_drop_future(fut)
        pending = self._parallel_inflight.get(queue_key)
        if pending is None or int(getattr(pending, "maxlen", 0) or 0) != int(inflight_limit):
            pending = deque(list(pending or []), maxlen=max(int(inflight_limit), 1))
            self._parallel_inflight[queue_key] = pending

        if len(pending) >= int(inflight_limit):
            oldest = pending[0]
            oldest_meta = dict(self._parallel_future_meta.get(oldest, {}) or {})
            try:
                oldest_frame = int(oldest_meta.get("frame_id", frame_index) or frame_index)
            except Exception:
                oldest_frame = int(frame_index)
            frame_delta = int(frame_index) - int(oldest_frame)
            if frame_delta <= int(self._PARALLEL_CANCEL_STALE_FRAME_DELTA):
                self._parallel_queue_pressure_drops += 1
                return False
            if oldest.cancel():
                self._parallel_drop_future(oldest)
            else:
                self._parallel_queue_pressure_drops += 1
                return False

        chunks_by_id = {
            str(src_id): chunk
            for src_id, chunk in dict(source_chunks or {}).items()
            if hasattr(chunk, "mono") and hasattr(chunk, "sample_rate")
        }
        if not chunks_by_id:
            return False

        block_size = int(getattr(graph, "block_size", 0) or 0)
        if block_size <= 0:
            block_size = max(
                int(getattr(getattr(chunk, "mono", np.zeros(1, dtype=np.float32)), "shape", [1])[-1] or 1)
                for chunk in chunks_by_id.values()
            )
        block_size = int(max(block_size, 1))
        sample_rate = max(
            int(getattr(chunk, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
            for chunk in chunks_by_id.values()
        )
        include_graph_ir_flag = bool(
            (graph_key in self._parallel_graph_ir_required)
            if include_graph_ir is None
            else include_graph_ir
        )
        source_payloads: Dict[str, Dict[str, Any]] = {}
        for src_id, chunk in chunks_by_id.items():
            src_sr = int(getattr(chunk, "sample_rate", sample_rate) or sample_rate)
            source_payloads[src_id] = self._build_parallel_source_payload(
                chunk,
                block_size=block_size,
                sample_rate=int(max(src_sr, 1)),
            )
        self._parallel_job_seq = int(self._parallel_job_seq) + 1
        job_key = (
            f"g{int(self._parallel_generation_id)}:"
            f"q{str(queue_key)}:"
            f"f{int(frame_index)}:"
            f"j{int(self._parallel_job_seq)}"
        )
        source_payload_ipc, payload_slot_keys = self._parallel_payload_to_ipc(
            source_payloads,
            graph_key=graph_key,
            frame_index=frame_index,
        )
        payload = {
            "generation_id": int(self._parallel_generation_id),
            "graph_uid": str(getattr(graph, "parallel_graph_uid", "") or owner_graph_key),
            "frame_id": int(frame_index),
            "block_size": int(block_size),
            "sample_rate": int(max(sample_rate, 1)),
            "graph_ir_digest": str(getattr(graph, "parallel_ir_digest", "") or ""),
            "output_keys": list(requested_output_keys),
            "job_key": str(job_key),
        }
        payload.update(source_payload_ipc)
        if include_graph_ir_flag:
            payload["graph_ir"] = dict(getattr(graph, "parallel_ir", {}) or {})

        try:
            future_obj = self._parallel_executor.submit(evaluate_parallel_job, payload)
        except Exception as e:
            for slot_key in payload_slot_keys:
                self._parallel_release_payload_slot(slot_key)
            fail_count = int(self._parallel_failures.get(owner_graph_key, 0)) + 1
            self._parallel_failures[owner_graph_key] = fail_count
            if fail_count <= 1 and self._parallel_restart_count < self._PARALLEL_RESTART_LIMIT:
                self._restart_parallel_pool(scene, f"submit failure: {e}", source_key=owner_graph_key)
            else:
                self._parallel_disabled_sources.add(owner_graph_key)
            self._log_runtime(
                f"parallel_submit:{owner_graph_key}",
                f"Parallel submit failed for '{owner_graph_key}', reusing last-valid offload values: {e}",
                "WARNING",
                min_interval=0.0,
            )
            return False

        pending.append(future_obj)
        self._parallel_future_meta[future_obj] = {
            "graph_key": graph_key,
            "queue_key": queue_key,
            "owner_graph_key": owner_graph_key,
            "frame_id": int(frame_index),
            "generation_id": int(self._parallel_generation_id),
            "undo_generation": int(getattr(self, "undo_generation", 0) or 0),
            "submit_ts": time.time(),
            "payload_slot_keys": list(payload_slot_keys),
        }
        self._parallel_last_submit_frame = int(frame_index)
        return True

    def _parallel_values_for_graph_frame(self, graph: ACEAudioGraph, graph_key: str, frame_index: int) -> Dict[str, Any]:
        """Retrieve the best available parallel worker results for *frame_index*.

        Priority: exact frame match > stale-reuse window > zero values.
        The stale window is derived from ``_PARALLEL_STALE_REUSE_MS`` converted
        to frames at the current scene FPS so behaviour is FPS-invariant.

        Returns:
            Dict mapping each offload attr_key to its normalised output value.
        """
        offload_keys = list(getattr(graph, "parallel_offload_attr_keys", []) or [])
        if not offload_keys:
            return {}
        frame_map = self._parallel_results.get(graph_key, {}) or {}
        frame_values = dict(frame_map.pop(int(frame_index), {}) or {})
        if isinstance(frame_map, dict) and not frame_map:
            self._parallel_results.pop(graph_key, None)
        if frame_values:
            merged_last = dict(self._parallel_last_valid.get(graph_key, {}) or {})
            merged_last.update(frame_values)
            self._parallel_last_valid[graph_key] = merged_last
            self._parallel_last_valid_frame[graph_key] = int(frame_index)
            base = merged_last
        else:
            last_frame = int(self._parallel_last_valid_frame.get(graph_key, -10_000_000) or -10_000_000)
            fps_now = max(float(self._current_scene_fps), 1.0)
            stale_window_frames = max(1, int(self._PARALLEL_STALE_REUSE_MS * fps_now / 1000.0 + 0.5))
            if (int(frame_index) - last_frame) <= stale_window_frames:
                base = dict(self._parallel_last_valid.get(graph_key, {}) or {})
            else:
                base = {}
        resolved: Dict[str, Any] = {}
        for attr_key in offload_keys:
            if attr_key in base:
                resolved[attr_key] = self._normalize_output_value(attr_key, base[attr_key])
            else:
                resolved[attr_key] = self._zero_value_for_attr(attr_key)
        return resolved

    def _parallel_has_usable_values_for_graph_frame(self, graph_key: str, frame_index: int) -> bool:
        """Return True when Stage-B has usable data for *graph_key* at *frame_index*.

        "Usable" means either fresh values for the current frame or non-empty
        last-valid values still inside the stale reuse window.
        """
        key = str(graph_key or "")
        frame_map = dict(self._parallel_results.get(key, {}) or {})
        if dict(frame_map.get(int(frame_index), {}) or {}):
            return True

        last_vals = dict(self._parallel_last_valid.get(key, {}) or {})
        if not last_vals:
            return False

        last_frame = int(self._parallel_last_valid_frame.get(key, -10_000_000) or -10_000_000)
        fps_now = max(float(self._current_scene_fps), 1.0)
        stale_window_frames = max(1, int(self._PARALLEL_STALE_REUSE_MS * fps_now / 1000.0 + 0.5))
        return (int(frame_index) - last_frame) <= stale_window_frames

    def compile(self, abstract_tree: ACEAbstractTree, blender_scene):
        """Build one ACEAudioGraph per audio source from the abstract tree.

        Shuts down any existing graphs, resets the parallel generation counter,
        resolves sample rates and block sizes for each source, creates all task
        objects (SourceTask, stream tasks, PreviewSinkTask, etc.), wires their
        inputs according to the abstract tree links, prepares the parallel IR for
        worker offload, and ensures Blender proxy attributes exist.

        Should only be called from the main thread.  For tree-change-triggered
        recompiles, prefer ``compile_from_blender_tree()`` which also parses the
        GN node tree and syncs attribute IDs.

        Args:
            abstract_tree: Parsed representation of the Geometry Nodes tree.
            blender_scene: The active ``bpy.types.Scene`` object.
        """
        utils.assert_main_thread("compile")
        self.scene_key = self._scene_key(blender_scene)
        self.scene_name = self._scene_name(blender_scene)
        self._pending_tree = None
        self._pending_tree_ptr = 0
        self._pending_tree_bl_idname = ""
        self._pending_scene_rebuild = False
        # Preserve stateful utility-node runtime across graph recompiles.
        stream_accumulator_state: Dict[tuple[str, str], Dict[str, Any]] = {}
        for graph_key, graph in dict(getattr(self, "graphs", {}) or {}).items():
            source_key = str(
                getattr(graph, "source_uid", "")
                or graph_key
                or getattr(graph, "proxy_name", "")
                or ""
            )
            if not source_key:
                continue
            for node_id, task in dict(getattr(graph, "tasks", {}) or {}).items():
                if not isinstance(task, StreamAccumulatorTask):
                    continue
                export_fn = getattr(task, "export_state", None)
                if not callable(export_fn):
                    continue
                try:
                    state = dict(export_fn() or {})
                except Exception:
                    state = {}
                if state:
                    stream_accumulator_state[(source_key, str(node_id))] = state
        self._shutdown_graphs()
        self._parallel_reset_generation()
        self.graphs.clear()
        self._last_attr_sync_frame.clear()
        self._attr_schema_dirty.clear()
        self._block_size_state.clear()
        self.abstract_tree_ref = abstract_tree

        self.writer = ACEProxyManager(abstract_tree)
        sink_writes_allowed = not self._unsafe_for_handler_writes(blender_scene)
        tree_signature = self._compute_tree_signature(abstract_tree)
        self._last_tree_signature = tree_signature
        if not hasattr(blender_scene, "audiocore"):
            return

        fps = 24.0
        if hasattr(blender_scene, "render"):
            raw_fps = blender_scene.render.fps
            base = blender_scene.render.fps_base
            if base > 0:
                fps = raw_fps / base
        engine_settings = getattr(getattr(blender_scene, "audiocore", None), "engine_settings", None)
        scheduler_mode = self._scheduler_mode(engine_settings)
        timeslice_mode = self._timeslice_mode(engine_settings)
        preview_block_size = self._preview_block_size(engine_settings)
        sources = list(getattr(getattr(blender_scene, "audiocore", None), "audio_sources", []) or [])
        single_source_mode = len(sources) <= 1
        self._prune_bake_cache(sources, tree_signature)

        self.effective_sample_rate = self._collect_sample_rate_floor(
            sources,
            engine_settings,
            fps,
            allow_io=False,
        )
        fallback_sr = self.effective_sample_rate

        existing_ids = [getattr(s, "stable_id", "") for s in getattr(blender_scene.audiocore, "audio_sources", []) if getattr(s, "stable_id", "")]
        source_runtime: list[dict[str, Any]] = []
        source_context_lookup: Dict[str, ACEAudioSource] = {}

        for source in sources:
            # Ensure a proxy exists and resolve the stable proxy name for filtering
            try:
                proxy_obj = ACEProxyManager.ensure_proxy_for_source(source, scene=blender_scene)
            except Exception:
                proxy_obj = None
            src_uid = ACEIdManager.ensure_source_uid(source, existing_ids=existing_ids)
            if src_uid and src_uid not in existing_ids:
                existing_ids.append(src_uid)
            src_proxy_name = getattr(source, "proxy_object_name", "") or (proxy_obj.name if proxy_obj else "")
            if not src_proxy_name:
                src_proxy_name = ACEIdManager.generate_source_proxy_name(
                    getattr(source, "name", "") or src_uid or "Source",
                    stable_seed=src_uid or getattr(source, "name", "") or "source",
                )
                try:
                    source.proxy_object_name = src_proxy_name
                except Exception:
                    pass

            if src_uid:
                source_context_lookup[src_uid] = source
            if src_proxy_name:
                source_context_lookup[src_proxy_name] = source

            self._probe_source_metadata(source, allow_io=False)
            source_sr = self._resolve_source_sample_rate(source, fallback_sr, engine_settings)
            source_type = str(getattr(source, "source_type", "") or "").upper()
            block_size = self._resolve_block_size(
                source_sr,
                fps,
                source_type=source_type,
                timeslice_mode=timeslice_mode,
                delta_frames=1.0,
            )
            source_runtime.append({
                "source": source,
                "src_uid": src_uid,
                "src_proxy_name": src_proxy_name,
                "source_sr": source_sr,
                "source_type": source_type,
                "block_size": block_size,
            })

        manual_preview_winner_id, manual_preview_candidates = self._resolve_global_manual_preview_winner(
            abstract_tree
        )
        has_manual_preview = bool(manual_preview_winner_id)
        if len(manual_preview_candidates) > 1 and manual_preview_winner_id:
            self._log_runtime(
                "preview_conflict_resolution",
                (
                    "Multiple Audio Preview nodes are present. "
                    f"Using the most recent preview sink '{manual_preview_winner_id}' and muting the others."
                ),
                "WARNING",
                min_interval=2.0,
            )

        for runtime in source_runtime:
            source = runtime["source"]
            src_uid = runtime["src_uid"]
            src_proxy_name = runtime["src_proxy_name"]
            source_sr = runtime["source_sr"]
            source_type = runtime["source_type"]
            block_size = runtime["block_size"]
            required_node_ids = self._compute_required_node_ids_for_source(
                abstract_tree,
                src_uid,
                src_proxy_name,
                single_source_mode,
                scheduler_mode=scheduler_mode,
            )
            graph = ACEAudioGraph(source, src_proxy_name)
            graph.source_uid = src_uid
            graph.block_size = block_size
            graph.scheduler_mode = scheduler_mode
            graph.timeslice_mode = timeslice_mode
            graph.proxy_lookup_fn = self._lookup_proxy_object
            for node_id in sorted(required_node_ids):
                node = abstract_tree.nodes.get(node_id)
                if node is None:
                    continue
                task = self._create_task_for_node(node, block_size)
                if task:
                    graph.tasks[node_id] = task
                    if isinstance(task, StreamAccumulatorTask):
                        restore_source_keys = [
                            str(src_uid or ""),
                            str(src_proxy_name or ""),
                        ]
                        for restore_source_key in restore_source_keys:
                            if not restore_source_key:
                                continue
                            state = stream_accumulator_state.get((restore_source_key, str(node_id)))
                            if not state:
                                continue
                            import_fn = getattr(task, "import_state", None)
                            if callable(import_fn):
                                try:
                                    import_fn(state)
                                except Exception:
                                    pass
                            break
                    task.engine_ref = self
                    task.undo_generation = int(getattr(self, "undo_generation", 0) or 0)
                    task.resolve_block_size_fn = (
                        lambda sr=None, fps_val=fps, src_sr=source_sr, src_type=source_type, g=graph, eng=self:
                        eng._resolve_block_size(
                            sr or src_sr,
                            fps_val,
                            source_type=src_type,
                            timeslice_mode=getattr(g, "timeslice_mode", eng._timeslice_mode()),
                            delta_frames=float(getattr(eng, "_current_delta_frames", 1.0) or 1.0),
                        )
                    )
                    if isinstance(task, (SourceTask, SourceSpecsTask)):
                        node_ctx = self._resolve_source_context_for_node(
                            node,
                            source_context_lookup,
                            default_source=source,
                        )
                        task.source_context = node_ctx
                    task.scene_fps = fps
                    task.scene_frame_start = int(getattr(blender_scene, "frame_start", 1) or 1)
                    task.engine_settings = engine_settings

            for link in abstract_tree.links:
                if link.source_node_id not in required_node_ids or link.target_node_id not in required_node_ids:
                    continue
                source_task = graph.tasks.get(link.source_node_id)
                target_task = graph.tasks.get(link.target_node_id)

                if source_task and target_task:
                    target_task.connect_input(
                        input_idx=link.target_socket_idx,
                        target_task=source_task,
                        target_output_idx=link.source_socket_idx
                    )
            self._rebuild_graph_execution_order(graph)

            filtered_keys = []
            for k in self.writer.output_defs.keys():
                info = self.writer.output_defs.get(k)
                node = abstract_tree.nodes.get(info.get("node_id")) if info else None
                if not node:
                    continue
                if node.id not in graph.tasks:
                    continue
                if not self._is_output_node(node):
                    continue
                if not self._node_matches_source(node, src_uid, src_proxy_name, single_source_mode):
                    continue
                filtered_keys.append(k)
            graph.output_attr_keys = filtered_keys
            graph.sink_bindings_by_attr = self._resolve_sink_bindings_for_graph(
                graph,
                abstract_tree,
                source_uid=src_uid,
                proxy_name=src_proxy_name,
                single_source_mode=single_source_mode,
            )
            self._rebuild_graph_output_slot_specs(graph)
            graph.manual_preview_task_ids = []
            if has_manual_preview and manual_preview_winner_id in graph.tasks:
                graph.manual_preview_task_ids = [str(manual_preview_winner_id)]

            auto_preview_task_ids: list[str] = []
            auto_preview_passthrough_specs: list[tuple[str, int]] = []
            if not has_manual_preview:
                auto_preview_specs = self._group_output_preview_specs_for_source(
                    abstract_tree,
                    src_uid,
                    src_proxy_name,
                    single_source_mode,
                )
                for upstream_node_id, upstream_sock_idx in auto_preview_specs:
                    upstream_task = graph.tasks.get(upstream_node_id)
                    if upstream_task is None:
                        continue
                    upstream_node = abstract_tree.nodes.get(upstream_node_id)
                    if (
                        isinstance(upstream_task, SourceTask)
                        and upstream_node is not None
                        and str(getattr(upstream_node, "category", "")) == "source_mngmt"
                        and str(getattr(upstream_node, "type", "")) == "src"
                    ):
                        auto_preview_passthrough_specs.append((str(upstream_node_id), int(upstream_sock_idx)))
                        continue
                    auto_task_id = f"__ace_auto_preview__{upstream_node_id}_{int(upstream_sock_idx)}"
                    if auto_task_id in graph.tasks:
                        continue
                    preview_task = PreviewSinkTask(auto_task_id, {"preview_enabled": True}, int(preview_block_size))
                    preview_task.engine_ref = self
                    preview_task.resolve_block_size_fn = (
                        lambda sr=None, fps_val=fps, src_sr=source_sr, src_type=source_type, g=graph, eng=self:
                        eng._resolve_block_size(
                            sr or src_sr,
                            fps_val,
                            source_type=src_type,
                            timeslice_mode=getattr(g, "timeslice_mode", eng._timeslice_mode()),
                            delta_frames=float(getattr(eng, "_current_delta_frames", 1.0) or 1.0),
                        )
                    )
                    preview_task.scene_fps = fps
                    preview_task.scene_frame_start = int(getattr(blender_scene, "frame_start", 1) or 1)
                    preview_task.engine_settings = engine_settings
                    preview_task.connect_input(
                        input_idx=0,
                        target_task=upstream_task,
                        target_output_idx=int(upstream_sock_idx),
                    )
                    graph.tasks[auto_task_id] = preview_task
                    auto_preview_task_ids.append(auto_task_id)

            graph.auto_preview_task_ids = sorted(auto_preview_task_ids)
            graph.auto_preview_passthrough_specs = sorted(set(auto_preview_passthrough_specs))
            # A single manual Audio Preview node wins globally; otherwise use per-graph auto taps.
            if has_manual_preview:
                graph.preview_task_ids = list(graph.manual_preview_task_ids)
            else:
                graph.preview_task_ids = list(graph.auto_preview_task_ids)
            source_task_ids: list[str] = []
            live_source_task_ids: list[str] = []
            file_source_task_ids: list[str] = []
            for node_id, task in graph.tasks.items():
                if not isinstance(task, SourceTask):
                    continue
                source_task_ids.append(node_id)
                source_kind = str(getattr(getattr(task, "source_context", None), "source_type", "") or "").upper()
                if source_kind == "LIVE":
                    live_source_task_ids.append(node_id)
                else:
                    file_source_task_ids.append(node_id)
            graph.source_task_ids = source_task_ids
            graph.live_source_task_ids = live_source_task_ids
            graph.file_source_task_ids = file_source_task_ids
            # _prepare_parallel_graph_ir() reads BPM/beats-per-bar from this source
            # context, so it must be set before the IR is built.
            graph.source_context = source
            self._prepare_parallel_graph_ir(graph, abstract_tree)
            self.graphs[src_uid] = graph
            self._attr_schema_dirty.add(src_uid)  # mark for first-frame schema sync

            if self.writer is not None and sink_writes_allowed:
                try:
                    self.writer.ensure_sink_layout(graph, active_keys=set(graph.output_attr_keys))
                    self._last_attr_sync_frame[src_uid] = int(self.current_frame)
                except Exception:
                    pass
            elif self.writer is not None and self._is_rendering_job():
                self._log_runtime(
                    "render_lock_interface_compile_stage_c",
                    "Skipping ACE sink layout writes during render. Enable Render > Lock Interface for safe handler updates.",
                    "WARNING",
                    min_interval=5.0,
                )

        # Prune stale ACE_* custom properties from consumer objects.
        # After all graphs' sink bindings are resolved, build the active set
        # and remove any ACE_ props that are no longer written by the engine.
        if self.writer is not None and sink_writes_allowed:
            try:
                active_props_by_object: Dict[str, Set[str]] = {}
                for g in self.graphs.values():
                    for bindings in getattr(g, "sink_bindings_by_attr", {}).values():
                        for binding in (bindings or []):
                            if str(getattr(binding, "sink_kind", "") or "") != "OBJECT_PROPERTY":
                                continue
                            obj_name = str(getattr(binding, "target_object_name", "") or "")
                            prop_name = str(getattr(binding, "target_data_name", "") or "")
                            if obj_name and prop_name:
                                active_props_by_object.setdefault(obj_name, set()).add(prop_name)
                self.writer.prune_consumer_objects(active_props_by_object)
            except Exception:
                pass

        try:
            ACEProxyManager.ensure_sonar_setup(blender_scene, getattr(blender_scene.audiocore, "audio_sources", []))
        except Exception:
            pass
        self._last_compile_time = time.time()

    def _log_runtime(self, key: str, message: str, level: str = "ERROR", min_interval: float = 2.0):
        """Emit a log message at most once per *min_interval* seconds per *key*.

        Prevents log spam from hot paths (frame handlers, parallel pollers) where
        the same error condition can recur every frame.
        """
        now = time.time()
        last = self._runtime_log_last.get(key, 0.0)
        if (now - last) < min_interval:
            return
        self._runtime_log_last[key] = now
        if len(self._runtime_log_last) > self._RUNTIME_LOG_MAX_ENTRIES:
            self._prune_timestamp_map(self._runtime_log_last, self._RUNTIME_LOG_MAX_ENTRIES)
        utils.log(message, level)

    def _probe_source_metadata(self, source: ACEAudioSource, allow_io: bool = True):
        """Populate ``source.sample_rate``, ``native_sample_rate``, and ``duration``.

        Uses the in-memory meta cache when available and not stale.  On a cache
        miss, dispatches ``_dispatch_async_probe()`` so the main thread is never
        blocked.  Results are applied on the next frame via ``_flush_probe_updates()``.

        Args:
            source: The ACEAudioSource property group to probe.
            allow_io: Kept for interface compatibility; blocking I/O is no longer
                performed directly here regardless of this flag.
        """
        if getattr(source, "source_type", "") != "FILE":
            return
        fp = getattr(source, "filepath", "")
        if not fp:
            return
        import os, time
        try:
            mtime = os.path.getmtime(fp)
        except OSError:
            mtime = None
        current_duration = float(getattr(source, "duration", 0.0) or 0.0)
        current_sample_rate = int(getattr(source, "sample_rate", 0) or 0)
        should_refresh_working_rate = current_duration <= 0.0 or current_sample_rate <= 0
        cache_key = f"FILE:{fp}"
        cached = self._meta_cache.get(cache_key)
        if cached and cached.get("mtime") == mtime and (time.time() - cached.get("ts", 0.0) <= self._meta_cache_ttl):
            try:
                cached_sample_rate = int(cached.get("sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
                if should_refresh_working_rate:
                    source.sample_rate = cached_sample_rate
                source.native_sample_rate = int(cached.get("native_sample_rate", cached_sample_rate) or cached_sample_rate)
                source.duration = cached.get("duration", 0.0)
            except Exception:
                pass
            return
        # Cache miss: dispatch background probe to avoid blocking the main thread.
        # Results are flushed into bpy props by _flush_probe_updates() on the next frame.
        self._dispatch_async_probe(fp)

    def _dispatch_async_probe(self, fp: str):
        """Schedule a background file-metadata probe. Thread-safe; no-op if already in flight."""
        if not fp:
            return
        undo_generation = int(getattr(self, "undo_generation", 0) or 0)
        with self._probe_thread_lock:
            if fp in self._probe_in_flight or fp in self._probe_pending_updates:
                return
            self._probe_in_flight.add(fp)
        t = threading.Thread(
            target=self._run_probe_in_bg,
            args=(fp, undo_generation),
            name="ACEProbeThread",
            daemon=True,
        )
        t.start()

    def _run_probe_in_bg(self, fp: str, undo_generation: int):
        """Worker: open the file, read metadata, write to _probe_pending_updates (no bpy access)."""
        try:
            import os as _os, time as _time
            try:
                mtime = _os.path.getmtime(fp)
            except OSError:
                mtime = None
            rec = AudioReception("FILE", filepath=fp)
            if rec.open():
                try:
                    native_sr = int(
                        getattr(rec, "native_sample_rate", 0)
                        or getattr(rec, "sample_rate", SAMPLE_RATE)
                        or SAMPLE_RATE
                    )
                    duration = float(getattr(rec, "duration", 0.0) or 0.0)
                finally:
                    try:
                        rec.close()
                    except Exception:
                        pass
                with self._probe_thread_lock:
                    if int(undo_generation) != int(getattr(self, "undo_generation", 0) or 0):
                        return
                    self._probe_pending_updates[fp] = {
                        "sample_rate": native_sr,
                        "native_sample_rate": native_sr,
                        "duration": duration,
                        "mtime": mtime,
                        "ts": _time.time(),
                        "undo_generation": int(undo_generation),
                    }
        except Exception:
            pass
        finally:
            with self._probe_thread_lock:
                self._probe_in_flight.discard(fp)

    def _flush_probe_updates(self, sources, scene=None, preferred_tree_ptr: int = 0) -> bool:
        """
        Flush background-probe results into bpy source props and meta_cache.
        Must be called from the main thread. Returns True if any update was applied
        (callers may wish to request a recompile so the new sample-rate is honoured).
        """
        with self._probe_thread_lock:
            if not self._probe_pending_updates:
                return False
            updates = dict(self._probe_pending_updates)
            self._probe_pending_updates.clear()
        any_updated = False
        active_generation = int(getattr(self, "undo_generation", 0) or 0)
        for source in (sources or []):
            if getattr(source, "source_type", "") != "FILE":
                continue
            fp = getattr(source, "filepath", "")
            if not fp or fp not in updates:
                continue
            result = updates[fp]
            if int(result.get("undo_generation", active_generation) or active_generation) != active_generation:
                continue
            try:
                native_sr = int(result.get("native_sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
                source.native_sample_rate = native_sr
                source.duration = float(result.get("duration", 0.0) or 0.0)
                if not int(getattr(source, "sample_rate", 0) or 0):
                    source.sample_rate = native_sr
            except Exception:
                pass
            cache_key = f"FILE:{fp}"
            self._meta_cache[cache_key] = result
            if len(self._meta_cache) > self._META_CACHE_MAX_ENTRIES:
                stale_keys = sorted(
                    self._meta_cache.items(),
                    key=lambda kv: float(kv[1].get("ts", 0.0)) if isinstance(kv[1], dict) else 0.0,
                )[: max(len(self._meta_cache) - self._META_CACHE_MAX_ENTRIES, 0)]
                for stale_key, _ in stale_keys:
                    self._meta_cache.pop(stale_key, None)
            any_updated = True
        if any_updated:
            try:
                preferred_tree = self._resolve_preferred_tree_from_discoveries(
                    self.discover_scene_frontends(scene) if scene is not None else [],
                    frontend_uid=str(getattr(self, "_compiled_tree_frontend_uid", "") or ""),
                    tree_ptr=int(preferred_tree_ptr or 0),
                    tree_bl_idname=str(getattr(self, "_compiled_tree_bl_idname", "") or ""),
                )
                self.request_recompile(preferred_tree)
            except Exception:
                pass
        return any_updated

    def _apply_tempo_detection_to_source(self, source: ACEAudioSource, detected: Dict[str, Any]) -> bool:
        bpm = float(detected.get("bpm", 0.0) or 0.0)
        beats_per_bar = int(detected.get("beats_per_bar", 0) or 0)
        if bpm <= 0.0 or beats_per_bar <= 0:
            return False
        try:
            source["ace_bpm_auto"] = float(bpm)
            source["ace_meter_auto"] = int(beats_per_bar)
            source.bpm = float(bpm)
            source.beats_per_bar = int(beats_per_bar)
            source["ace_bpm_user_override"] = False
            source["ace_meter_user_override"] = False
            source["ace_tempo_detect_status"] = "ready"
            source["ace_tempo_detect_message"] = f"Detected {float(bpm):.2f} BPM / {int(beats_per_bar)} BPB"
            return True
        except Exception:
            return False

    def request_source_tempo_detection(self, source: Optional[ACEAudioSource]) -> tuple[bool, str]:
        """Queue non-blocking BPM / beats-per-bar detection for a FILE source."""
        utils.assert_main_thread("request_source_tempo_detection")
        if source is None:
            return False, "No source selected."

        target = SourceSpecsTask.resolve_analysis_target(source)
        if target is None:
            try:
                source["ace_tempo_detect_status"] = "failed"
                source["ace_tempo_detect_message"] = "Select a valid file source before detecting tempo."
            except Exception:
                pass
            return False, "Select a valid file source before detecting tempo."

        filepath, mtime = target
        try:
            source["ace_tempo_detect_status"] = "pending"
            source["ace_tempo_detect_message"] = "Analyzing source tempo in the background..."
            source["ace_tempo_detect_filepath"] = str(filepath)
            source["ace_tempo_detect_mtime"] = float(mtime)
        except Exception:
            pass

        cached = SourceSpecsTask.get_cached_analysis(target)
        if cached:
            if cached.get("bpm", None) is not None and self._apply_tempo_detection_to_source(source, cached):
                return True, "Tempo detection updated from cache."
            error_msg = str(cached.get("error", "") or "Tempo analysis failed.")
            try:
                source["ace_tempo_detect_status"] = "failed"
                source["ace_tempo_detect_message"] = error_msg
            except Exception:
                pass
            return False, error_msg

        queued = SourceSpecsTask._queue_background_analysis(target)
        if not queued:
            try:
                source["ace_tempo_detect_status"] = "failed"
                source["ace_tempo_detect_message"] = "Tempo analysis queue is full. Try again in a moment."
            except Exception:
                pass
            return False, "Tempo analysis queue is full. Try again in a moment."

        return True, "Tempo detection queued."

    def flush_source_tempo_detection_updates(self, sources) -> bool:
        """Apply finished background tempo detections to FILE source properties."""
        utils.assert_main_thread("flush_source_tempo_detection_updates")
        any_updated = False
        for source in (sources or []):
            if getattr(source, "source_type", "") != "FILE":
                continue
            try:
                status = str(source.get("ace_tempo_detect_status", "") or "")
            except Exception:
                status = ""
            if status != "pending":
                continue

            try:
                filepath = str(source.get("ace_tempo_detect_filepath", "") or "")
                mtime = float(source.get("ace_tempo_detect_mtime", 0.0) or 0.0)
            except Exception:
                filepath = ""
                mtime = 0.0
            if not filepath or mtime <= 0.0:
                continue

            current_target = SourceSpecsTask.resolve_analysis_target(source)
            if current_target is None or str(current_target[0]) != filepath or abs(float(current_target[1]) - mtime) > 1e-6:
                try:
                    source["ace_tempo_detect_status"] = "failed"
                    source["ace_tempo_detect_message"] = "Source changed. Run tempo detection again."
                except Exception:
                    pass
                any_updated = True
                continue

            cached = SourceSpecsTask.get_cached_analysis((filepath, mtime))
            if not cached:
                continue

            if cached.get("bpm", None) is None:
                try:
                    source["ace_tempo_detect_status"] = "failed"
                    source["ace_tempo_detect_message"] = str(
                        cached.get("error", "") or "Tempo analysis failed."
                    )
                except Exception:
                    pass
                any_updated = True
                continue

            if self._apply_tempo_detection_to_source(source, cached):
                any_updated = True
        return any_updated

    def _collect_sample_rate_floor(self, sources, engine_settings, fps: float, allow_io: bool = True) -> int:
        min_sr = None
        for source in sources or []:
            source_type = str(getattr(source, "source_type", "") or "").upper()
            if source_type == "FILE":
                try:
                    sr = int(getattr(source, "native_sample_rate", 0) or 0)
                except Exception:
                    sr = 0
            else:
                sr = int(getattr(source, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
            cache_key = ""
            mtime = None
            try:
                if source_type == "LIVE":
                    live_flow_id = build_live_flow_identifier(source)
                    try:
                        source.flow_identifier = live_flow_id
                    except Exception:
                        pass
                    # Compile path must not touch live devices/sockets.
                    live_transport = str(getattr(source, "live_transport", "") or "").upper()
                    udp_sr = int(getattr(source, "live_udp_sample_rate", 0) or 0)
                    if live_transport == "UDP" and udp_sr > 0:
                        sr = udp_sr
                    elif sr <= 0:
                        sr = SAMPLE_RATE
                    try:
                        source.sample_rate = int(sr)
                    except Exception:
                        pass
                    if min_sr is None or sr < min_sr:
                        min_sr = sr
                    continue

                # Fast path: trust existing metadata when available
                if source_type == "FILE" and sr > 0 and getattr(source, "duration", 0.0) > 0.0:
                    working_sr = self._cap_source_sample_rate(source_type, sr, engine_settings)
                    if min_sr is None or working_sr < min_sr:
                        min_sr = working_sr
                    continue
                if source_type == "FILE":
                    import os
                    cache_key = f"FILE:{getattr(source, 'filepath', '')}"
                    try:
                        mtime = os.path.getmtime(getattr(source, "filepath", ""))
                    except OSError:
                        mtime = None
                    cached = self._meta_cache.get(cache_key)
                    ts_ok = cached and (getattr(__import__("time"), "time")() - cached.get("ts", 0.0) <= self._meta_cache_ttl)
                    if cached and cached.get("mtime") == mtime and ts_ok:
                        sr = int(cached.get("native_sample_rate", cached.get("sample_rate", sr)) or sr)
                        try:
                            source.native_sample_rate = int(sr)
                        except Exception:
                            pass
                        if getattr(source, "duration", 0.0) == 0.0:
                            try:
                                source.duration = float(cached.get("duration", 0.0))
                            except Exception:
                                pass
                        working_sr = self._cap_source_sample_rate(source_type, sr, engine_settings)
                        if min_sr is None or working_sr < min_sr:
                            min_sr = working_sr
                        continue
                if source_type == "FILE":
                    # Dispatch a background probe instead of blocking the main thread.
                    # Results will be applied by _flush_probe_updates() on the next frame,
                    # which triggers a recompile so the correct sample-rate is picked up.
                    self._dispatch_async_probe(getattr(source, "filepath", ""))
                # Use current sr as fallback; updated on recompile after probe completes.
            except Exception:
                pass
            working_sr = self._cap_source_sample_rate(source_type, sr, engine_settings)
            if min_sr is None or working_sr < min_sr:
                min_sr = working_sr

        if min_sr is None:
            min_sr = SAMPLE_RATE
        return int(min_sr)

    @classmethod
    def _resolve_source_sample_rate(cls, source, fallback_sr: int, scene_or_engine_settings=None) -> int:
        """
        Choose the best-known sample rate for a source, preferring its metadata and
        falling back to the scene-wide minimum if unavailable.
        """
        sr = cls._source_original_sample_rate(source)
        if sr <= 0:
            sr = int(fallback_sr)
        source_type = str(getattr(source, "source_type", "") or "").strip().upper()
        return cls._cap_source_sample_rate(source_type, sr, scene_or_engine_settings)

    @staticmethod
    def _compute_tree_signature(abstract_tree: ACEAbstractTree):
        node_sig = []
        for nid, node in sorted(abstract_tree.nodes.items()):
            # repr(v) guards against NumPy scalars/arrays that would cause
            # TypeError when sorted() tries to compare values of mixed types.
            params = tuple(sorted(
                ((k, repr(v)) for k, v in node.parameters.items()),
                key=lambda x: x[0],
            ))
            inputs = tuple(sorted((s.socket_idx, s.data_type, s.name) for s in node.inputs))
            outputs = tuple(sorted((s.socket_idx, s.data_type, s.name) for s in node.outputs))
            roots = tuple(sorted(node.root_sources)) if getattr(node, "root_sources", None) else ()
            node_sig.append((
                nid,
                node.category,
                node.type,
                params,
                inputs,
                outputs,
                roots,
                str(getattr(node, "frontend_uid", "") or ""),
                str(getattr(node, "frontend_kind", "") or ""),
                str(getattr(node, "source_binding_id", "") or ""),
            ))
        links = tuple(sorted((l.source_node_id, l.source_socket_idx, l.target_node_id, l.target_socket_idx) for l in abstract_tree.links))
        output_links = tuple(
            sorted(
                (
                    str(getattr(l, "source_node_id", "")),
                    int(getattr(l, "source_socket_idx", 0)),
                    int(getattr(l, "target_socket_idx", 0)),
                    str(getattr(l, "target_socket_name", "") or ""),
                    str(getattr(l, "frontend_uid", "") or ""),
                    str(getattr(l, "frontend_kind", "") or ""),
                )
                for l in (getattr(abstract_tree, "group_output_links", []) or [])
            )
        )
        frontend_sig = tuple(
            sorted(
                (
                    str(uid),
                    str(getattr(frontend, "kind", "") or ""),
                    str(getattr(frontend, "tree_name", "") or ""),
                    str(getattr(frontend, "tree_bl_idname", "") or ""),
                    tuple(getattr(frontend, "owner_keys", ()) or ()),
                    str(getattr(frontend, "material_name", "") or ""),
                    int(getattr(frontend, "tree_ptr", 0) or 0),
                )
                for uid, frontend in (getattr(abstract_tree, "frontends", {}) or {}).items()
            )
        )
        frontends_by_uid = dict(getattr(abstract_tree, "frontends", {}) or {})
        consumer_sig = tuple(
            sorted(
                (
                    str(getattr(intent, "frontend_uid", "") or ""),
                    str(
                        getattr(
                            frontends_by_uid.get(str(getattr(intent, "frontend_uid", "") or "")),
                            "kind",
                            "",
                        )
                        or ""
                    ),
                    str(getattr(intent, "node_id", "") or ""),
                    int(getattr(intent, "socket_idx", 0) or 0),
                    str(getattr(intent, "sink_kind", "") or ""),
                    str(getattr(intent, "source_token", "") or ""),
                    tuple(
                        sorted(
                            str(v)
                            for v in (getattr(intent, "consumer_keys", ()) or ())
                            if str(v)
                        )
                    ),
                    str(getattr(intent, "output_name", "") or ""),
                    str(getattr(intent, "explicit_name", "") or ""),
                )
                for intent in (getattr(abstract_tree, "consumer_intents", []) or [])
            )
        )
        return (tuple(node_sig), links, output_links, frontend_sig, consumer_sig)

    @staticmethod
    def _build_auto_attr_name(
        proxy_name: str,
        node_id: str,
        output_name: str,
        socket_idx: int = 0,
        topology_sig: str = "",
    ) -> str:
        hashed_node_id = f"{node_id}|{topology_sig}" if topology_sig else node_id
        return ACEIdManager.generate_compact_attr_name(proxy_name, hashed_node_id, output_name, socket_idx)[:63]

    @staticmethod
    def _build_source_proxy_lookup(scene) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        ace = getattr(scene, "audiocore", None)
        for src in getattr(ace, "audio_sources", []) or []:
            sid = getattr(src, "stable_id", "")
            proxy_name = getattr(src, "proxy_object_name", "")
            if proxy_name:
                lookup[proxy_name] = proxy_name
            if sid and proxy_name:
                lookup[sid] = proxy_name
        return lookup

    @staticmethod
    def _lookup_proxy_object(proxy_name: str):
        if not proxy_name:
            return None
        try:
            return bpy.data.objects.get(str(proxy_name))
        except Exception:
            return None

    @staticmethod
    def _resolve_proxy_name_for_output_node(
        node: Optional[ACEAbstractNode],
        source_lookup: Dict[str, str],
        unique_proxies: list[str] | tuple[str, ...],
    ) -> str:
        if node is None:
            return ""
        for root in sorted(set(getattr(node, "root_sources", set()) or set())):
            proxy_name = str(source_lookup.get(str(root), "") or "")
            if proxy_name:
                return proxy_name
        if len(unique_proxies) == 1:
            return str(unique_proxies[0] or "")
        return ""

    @staticmethod
    def _tree_frontend_uid(tree) -> str:
        if tree is None:
            return ""
        try:
            return str(tree.get("ace_frontend_uid", "") or "")
        except Exception:
            return ""

    @staticmethod
    def _tree_is_internal_ace_group(tree) -> bool:
        if tree is None:
            return False
        name = str(getattr(tree, "name", "") or "")
        return bool(name.startswith(".ACE_"))

    @classmethod
    def _tree_is_ace_managed(cls, tree) -> bool:
        return bool(cls._tree_frontend_uid(tree) or cls._tree_has_ace_nodes(tree))

    @staticmethod
    def _frontend_membership_signature(frontends: list[_ACEFrontendDiscovery]) -> tuple[Any, ...]:
        return tuple(
            sorted(
                (
                    str(rec.frontend_ref.uid),
                    str(rec.frontend_ref.kind),
                    str(rec.frontend_ref.tree_name),
                    int(rec.frontend_ref.tree_ptr),
                    tuple(rec.frontend_ref.owner_keys),
                    str(rec.frontend_ref.material_name),
                    tuple(rec.consumer_object_names),
                )
                for rec in (frontends or [])
            )
        )

    @staticmethod
    def _compiled_snapshot_from_discovery(rec: _ACEFrontendDiscovery) -> _ACECompiledFrontendSnapshot:
        frontend_ref = rec.frontend_ref
        return _ACECompiledFrontendSnapshot(
            uid=str(frontend_ref.uid or ""),
            kind=str(frontend_ref.kind or ""),
            tree_name=str(frontend_ref.tree_name or ""),
            tree_bl_idname=str(frontend_ref.tree_bl_idname or ""),
            owner_keys=tuple(str(v) for v in tuple(frontend_ref.owner_keys or ()) if str(v)),
            consumer_object_names=tuple(str(v) for v in tuple(rec.consumer_object_names or ()) if str(v)),
            material_name=str(frontend_ref.material_name or ""),
            tree_ptr=int(frontend_ref.tree_ptr or 0),
        )

    @staticmethod
    def _resolve_preferred_tree_from_discoveries(
        frontends: list[_ACEFrontendDiscovery],
        frontend_uid: str = "",
        tree_ptr: int = 0,
        tree_bl_idname: str = "",
    ):
        uid_token = str(frontend_uid or "")
        ptr_token = int(tree_ptr or 0)
        type_token = str(tree_bl_idname or "")
        if uid_token:
            for rec in frontends or []:
                if str(getattr(getattr(rec, "frontend_ref", None), "uid", "") or "") == uid_token:
                    return getattr(rec, "tree", None)
        if ptr_token:
            for rec in frontends or []:
                rec_ptr = int(getattr(getattr(rec, "frontend_ref", None), "tree_ptr", 0) or 0)
                if rec_ptr == ptr_token:
                    return getattr(rec, "tree", None)
        if type_token:
            for rec in frontends or []:
                if str(getattr(getattr(rec, "frontend_ref", None), "tree_bl_idname", "") or "") == type_token:
                    return getattr(rec, "tree", None)
        return None

    @staticmethod
    def _refresh_consumer_intents_from_nodes(abstract_tree: ACEAbstractTree):
        for intent in getattr(abstract_tree, "consumer_intents", []) or []:
            node = abstract_tree.nodes.get(str(getattr(intent, "node_id", "") or ""))
            if node is None:
                continue
            socket_idx = int(getattr(intent, "socket_idx", 0) or 0)
            outputs = list(getattr(node, "outputs", []) or [])
            if 0 <= socket_idx < len(outputs):
                intent.output_name = str(getattr(outputs[socket_idx], "name", "") or "")
            params = dict(getattr(node, "parameters", {}) or {})
            explicit_name = str(params.get(f"attr_id_out_{socket_idx}", "") or "")
            intent.explicit_name = explicit_name

    @staticmethod
    def _preview_node_sort_key(
        node: Optional[ACEAbstractNode],
        abstract_tree: Optional[ACEAbstractTree],
    ) -> tuple[int, int, str, str]:
        if node is None:
            return (-1, -1, "", "")
        frontends = dict(getattr(abstract_tree, "frontends", {}) or {}) if abstract_tree is not None else {}
        frontend_uid = str(getattr(node, "frontend_uid", "") or "")
        frontend = frontends.get(frontend_uid)
        discovery_index = int(getattr(frontend, "discovery_index", 0) or 0)
        params = dict(getattr(node, "parameters", {}) or {})
        frontend_node_index = utils.as_int(params.get("frontend_node_index"), -1)
        return (
            discovery_index,
            frontend_node_index,
            frontend_uid,
            str(getattr(node, "id", "") or ""),
        )

    def _resolve_global_manual_preview_winner(self, abstract_tree: Optional[ACEAbstractTree]) -> tuple[str, list[str]]:
        candidates: list[str] = []
        if abstract_tree is None:
            return "", candidates
        for node_id, node in (getattr(abstract_tree, "nodes", {}) or {}).items():
            if str(getattr(node, "category", "") or "") != "utilities":
                continue
            if str(getattr(node, "type", "") or "") != "preview_sink":
                continue
            candidates.append(str(node_id))
        if not candidates:
            return "", []
        winner_id = max(
            candidates,
            key=lambda node_id: self._preview_node_sort_key(
                abstract_tree.nodes.get(str(node_id)),
                abstract_tree,
            ),
        )
        return str(winner_id or ""), sorted(
            candidates,
            key=lambda node_id: self._preview_node_sort_key(
                abstract_tree.nodes.get(str(node_id)),
                abstract_tree,
            ),
        )

    def discover_scene_frontends(self, scene) -> list[_ACEFrontendDiscovery]:
        """Discover managed ACE frontends used by the scene."""
        from ..nodes.dummy_tree import ensure_frontend_uid as _ensure_frontend_uid
        from ..nodes.shader_tree import ACEShaderTreeReader

        utils.assert_main_thread("discover_scene_frontends")

        if scene is None:
            return []

        by_ptr: Dict[int, Dict[str, Any]] = {}

        def _register_tree(tree, frontend_kind: str, owner_keys, material_name: str = "", consumer_object_names=()):
            if tree is None or self._tree_is_internal_ace_group(tree):
                return
            tree_type = str(getattr(tree, "bl_idname", "") or "")
            is_shader = str(frontend_kind or "") == FRONTEND_KIND_SHADER or tree_type == "ShaderNodeTree"
            if is_shader:
                managed = bool(ACEShaderTreeReader.is_managed_tree(tree) or self._tree_is_ace_managed(tree))
            else:
                managed = self._tree_is_ace_managed(tree)
            if not managed:
                return
            tree_ptr = self._node_tree_pointer(tree)
            owner_keys_tuple = tuple(sorted(str(v) for v in (owner_keys or []) if str(v)))
            consumer_keys_tuple = tuple(sorted(str(v) for v in (consumer_object_names or []) if str(v)))
            existing = by_ptr.get(tree_ptr)
            if existing is None:
                by_ptr[tree_ptr] = {
                    "tree": tree,
                    "frontend_kind": str(frontend_kind or ""),
                    "discovery_index": len(by_ptr),
                    "owner_keys": set(owner_keys_tuple),
                    "consumer_keys": set(consumer_keys_tuple),
                    "material_name": str(material_name or ""),
                }
                return
            existing["owner_keys"].update(owner_keys_tuple)
            existing["consumer_keys"].update(consumer_keys_tuple)
            if material_name and not existing.get("material_name"):
                existing["material_name"] = str(material_name)

        def _frontend_record_sort_key(rec: Dict[str, Any]) -> tuple[Any, ...]:
            tree = rec.get("tree")
            return (
                str(rec.get("frontend_kind", "") or ""),
                str(getattr(tree, "bl_idname", "") or ""),
                str(getattr(tree, "name", "") or ""),
                str(rec.get("material_name", "") or ""),
                tuple(sorted(str(v) for v in rec.get("owner_keys", set()) if str(v))),
                tuple(sorted(str(v) for v in rec.get("consumer_keys", set()) if str(v))),
                int(self._node_tree_pointer(tree)),
            )

        for obj in getattr(scene, "objects", []) or []:
            obj_name = str(getattr(obj, "name", "") or "")
            for mod in getattr(obj, "modifiers", []) or []:
                if getattr(mod, "type", None) != "NODES":
                    continue
                tree = getattr(mod, "node_group", None)
                if tree is None:
                    continue
                _register_tree(
                    tree,
                    FRONTEND_KIND_GEOMETRY,
                    owner_keys=(f"{obj_name}:{getattr(mod, 'name', '')}",),
                )
            for slot in getattr(obj, "material_slots", []) or []:
                material = getattr(slot, "material", None)
                tree = getattr(material, "node_tree", None) if material is not None else None
                if tree is None or str(getattr(tree, "bl_idname", "") or "") != "ShaderNodeTree":
                    continue
                _register_tree(
                    tree,
                    FRONTEND_KIND_SHADER,
                    owner_keys=(obj_name,),
                    material_name=str(getattr(material, "name", "") or ""),
                    consumer_object_names=(obj_name,),
                )

        discovered: list[_ACEFrontendDiscovery] = []
        existing_uids: list[str] = []
        for rec in sorted(by_ptr.values(), key=_frontend_record_sort_key):
            tree = rec.get("tree")
            frontend_kind = str(rec.get("frontend_kind", "") or "")
            frontend_uid = _ensure_frontend_uid(tree, frontend_kind, existing_uids)
            if not frontend_uid:
                continue
            if frontend_uid not in existing_uids:
                existing_uids.append(frontend_uid)
            owner_keys = tuple(sorted(str(v) for v in rec.get("owner_keys", set()) if str(v)))
            consumer_keys = tuple(sorted(str(v) for v in rec.get("consumer_keys", set()) if str(v)))
            discovered.append(
                _ACEFrontendDiscovery(
                    tree=tree,
                    frontend_ref=ACEFrontendRef(
                        uid=frontend_uid,
                        kind=frontend_kind,
                        tree_name=str(getattr(tree, "name", "") or ""),
                        tree_bl_idname=str(getattr(tree, "bl_idname", "") or ""),
                        owner_keys=owner_keys,
                        material_name=str(rec.get("material_name", "") or ""),
                        tree_ptr=int(self._node_tree_pointer(tree)),
                        discovery_index=int(rec.get("discovery_index", 0) or 0),
                    ),
                    consumer_object_names=consumer_keys,
                )
            )

        return sorted(discovered, key=lambda rec: str(rec.frontend_ref.uid or ""))

    @staticmethod
    def _is_output_node(node: ACEAbstractNode) -> bool:
        is_specs = bool(
            node
            and node.type in {'src_specs', 'util_rolling_avg', 'stream_accumulator'}
            and node.category == 'utilities'
        )
        return bool(
            node
            and (
                node.category == 'stream'
                or is_specs
                or (node.category == 'utilities' and node.type == 'preview_sink')
            )
        )

    @staticmethod
    def _node_matches_source(node: ACEAbstractNode, source_uid: str, proxy_name: str, single_source_mode: bool) -> bool:
        if single_source_mode:
            return True
        roots = {str(v) for v in set(getattr(node, "root_sources", set()) or set()) if str(v)}
        if not roots:
            return True
        source_aliases = {str(v) for v in (source_uid, proxy_name) if str(v)}
        if not source_aliases:
            return False
        return bool(roots.intersection(source_aliases))

    @staticmethod
    def _prune_ambiguous_required_node_ids(
        abstract_tree: ACEAbstractTree,
        required_node_ids: set[str],
        source_uid: str,
        proxy_name: str,
        single_source_mode: bool,
    ) -> set[str]:
        if single_source_mode:
            return set(required_node_ids)
        aliases = {str(v) for v in (source_uid, proxy_name) if str(v)}
        if not aliases:
            return set()
        reverse_adj: Dict[str, set[str]] = {}
        for link in getattr(abstract_tree, "links", []) or []:
            src_id = str(getattr(link, "source_node_id", "") or "")
            dst_id = str(getattr(link, "target_node_id", "") or "")
            if src_id and dst_id:
                reverse_adj.setdefault(dst_id, set()).add(src_id)

        ambiguous: set[str] = set()
        for node_id in set(required_node_ids):
            node = abstract_tree.nodes.get(node_id)
            if node is None:
                continue
            roots = {str(v) for v in set(getattr(node, "root_sources", set()) or set()) if str(v)}
            if not roots or roots.issubset(aliases) or not roots.intersection(aliases):
                continue
            upstream_roots: set[tuple[str, ...]] = set()
            for upstream_id in reverse_adj.get(str(node_id), set()):
                if upstream_id not in required_node_ids:
                    continue
                upstream_node = abstract_tree.nodes.get(str(upstream_id))
                if upstream_node is None:
                    continue
                uroots = sorted(
                    str(v)
                    for v in set(getattr(upstream_node, "root_sources", set()) or set())
                    if str(v)
                )
                if uroots:
                    upstream_roots.add(tuple(uroots))
            if len(upstream_roots) >= 2:
                ambiguous.add(str(node_id))

        if not ambiguous:
            return set(required_node_ids)

        pruned_adj: Dict[str, set[str]] = {}
        for link in getattr(abstract_tree, "links", []) or []:
            src_id = str(getattr(link, "source_node_id", "") or "")
            dst_id = str(getattr(link, "target_node_id", "") or "")
            if not src_id or not dst_id:
                continue
            if src_id in ambiguous or dst_id in ambiguous:
                continue
            if src_id not in required_node_ids or dst_id not in required_node_ids:
                continue
            pruned_adj.setdefault(dst_id, set()).add(src_id)

        seed_nodes = [
            node_id
            for node_id in required_node_ids
            if node_id not in ambiguous
        ]
        kept: set[str] = set()
        stack = list(seed_nodes)
        while stack:
            current = stack.pop()
            if current in kept:
                continue
            kept.add(current)
            for upstream in pruned_adj.get(current, set()):
                if upstream not in kept:
                    stack.append(upstream)
        return kept

    @staticmethod
    def _is_bundle_output_socket(node: Optional[ACEAbstractNode], socket_idx: int) -> bool:
        if node is None:
            return False
        outputs = getattr(node, "outputs", []) or []
        if socket_idx < 0 or socket_idx >= len(outputs):
            return False
        data_type = str(getattr(outputs[socket_idx], "data_type", "") or "")
        return "Bundle" in data_type

    def _group_output_preview_specs_for_source(
        self,
        abstract_tree: ACEAbstractTree,
        source_uid: str,
        proxy_name: str,
        single_source_mode: bool,
    ) -> list[tuple[str, int]]:
        specs: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for out_link in (getattr(abstract_tree, "group_output_links", []) or []):
            node_id = str(getattr(out_link, "source_node_id", "") or "")
            sock_idx = int(getattr(out_link, "source_socket_idx", 0) or 0)
            if not node_id:
                continue
            target_name = str(getattr(out_link, "target_socket_name", "") or "").strip().lower()
            target_idx = int(getattr(out_link, "target_socket_idx", 0) or 0)
            # Auto-preview only taps the intended GN Group Output "Audio" bundle.
            if target_name:
                if target_name != "audio":
                    continue
            elif target_idx != 1:
                continue
            node = abstract_tree.nodes.get(node_id)
            if node is None:
                continue
            if not self._is_bundle_output_socket(node, sock_idx):
                continue
            if not self._node_matches_source(node, source_uid, proxy_name, single_source_mode):
                continue
            key = (node_id, sock_idx)
            if key in seen:
                continue
            seen.add(key)
            specs.append(key)
        return specs

    def _compute_required_node_ids_for_source(
        self,
        abstract_tree: ACEAbstractTree,
        source_uid: str,
        proxy_name: str,
        single_source_mode: bool,
        scheduler_mode: str = "legacy",
    ) -> set[str]:
        if not abstract_tree:
            return set()

        seed_nodes: set[str] = set()
        for node in abstract_tree.nodes.values():
            if not self._is_output_node(node):
                continue
            if self._node_matches_source(node, source_uid, proxy_name, single_source_mode):
                seed_nodes.add(node.id)
        if str(scheduler_mode or "legacy").strip().lower() == "legacy":
            for node in abstract_tree.nodes.values():
                # Legacy mode keeps source nodes active each frame even when no
                # stream/spec branch requests them.
                if not (node.category == 'source_mngmt' and node.type == 'src'):
                    continue
                if self._node_matches_source(node, source_uid, proxy_name, single_source_mode):
                    seed_nodes.add(node.id)
        for node_id, _sock_idx in self._group_output_preview_specs_for_source(
            abstract_tree,
            source_uid,
            proxy_name,
            single_source_mode,
        ):
            seed_nodes.add(node_id)

        if not seed_nodes:
            return set()

        reverse_adj: Dict[str, set[str]] = {}
        for link in abstract_tree.links:
            reverse_adj.setdefault(link.target_node_id, set()).add(link.source_node_id)

        required: set[str] = set()
        stack = list(seed_nodes)
        while stack:
            current = stack.pop()
            if current in required:
                continue
            required.add(current)
            for upstream in reverse_adj.get(current, set()):
                if upstream not in required:
                    stack.append(upstream)
        return self._prune_ambiguous_required_node_ids(
            abstract_tree,
            required,
            source_uid=source_uid,
            proxy_name=proxy_name,
            single_source_mode=single_source_mode,
        )

    @staticmethod
    def _resolve_source_context_for_node(
        node: Optional[ACEAbstractNode],
        source_lookup: Dict[str, ACEAudioSource],
        default_source: Optional[ACEAudioSource] = None,
    ) -> Optional[ACEAudioSource]:
        if node is None:
            return default_source

        tokens: list[str] = []
        if getattr(node, "id", ""):
            tokens.append(str(node.id))
        for root in sorted(getattr(node, "root_sources", set()) or set()):
            tokens.append(str(root))

        for token in tokens:
            src = source_lookup.get(token)
            if src is not None:
                return src
        return default_source

    @staticmethod
    def _resolve_sink_data_name(proxy_name: str, node_id: str, output_name: str, socket_idx: int, explicit_name: str = "") -> str:
        explicit = str(explicit_name or "").strip()
        if explicit:
            if not explicit.startswith(ACEIdManager.PREFIX):
                explicit = ACEIdManager.PREFIX + ACEIdManager._clean_string(explicit)
            return explicit[:63]
        return ACEIdManager.generate_compact_attr_name(proxy_name, node_id, output_name, socket_idx)

    def _resolve_sink_bindings_for_graph(
        self,
        graph: ACEAudioGraph,
        abstract_tree: ACEAbstractTree,
        source_uid: str,
        proxy_name: str,
        single_source_mode: bool,
    ) -> Dict[str, List[ACESinkBinding]]:
        bindings_by_attr: Dict[str, List[ACESinkBinding]] = {}
        source_aliases = {str(v) for v in (source_uid, proxy_name) if str(v)}
        consumer_intents = list(getattr(abstract_tree, "consumer_intents", []) or [])
        frontends = dict(getattr(abstract_tree, "frontends", {}) or {})

        for attr_key in list(getattr(graph, "output_attr_keys", []) or []):
            info = self.writer.output_defs.get(attr_key, {}) if self.writer else {}
            node_id = str(info.get("node_id", "") or "")
            socket_idx = int(info.get("socket_idx", 0) or 0)
            node = abstract_tree.nodes.get(node_id)
            if node is None:
                bindings_by_attr[attr_key] = []
                continue
            frontend_uid = str(getattr(node, "frontend_uid", "") or info.get("frontend_uid", "") or "")
            frontend_kind = str(getattr(node, "frontend_kind", "") or "")
            output_name = str(info.get("output_name", "") or "")
            attr_type = str(info.get("type", "FLOAT") or "FLOAT")
            resolved: list[ACESinkBinding] = []

            for intent in consumer_intents:
                if str(getattr(intent, "node_id", "") or "") != node_id:
                    continue
                if int(getattr(intent, "socket_idx", 0) or 0) != socket_idx:
                    continue
                intent_frontend_uid = str(getattr(intent, "frontend_uid", "") or "")
                if intent_frontend_uid and frontend_uid and intent_frontend_uid != frontend_uid:
                    continue
                source_token = str(getattr(intent, "source_token", "") or "")
                if (not single_source_mode) and source_token and source_token not in source_aliases:
                    continue
                sink_kind = str(getattr(intent, "sink_kind", "") or "")
                explicit_name = str(getattr(intent, "explicit_name", "") or "")
                sink_name = self._resolve_sink_data_name(
                    proxy_name,
                    node_id,
                    str(getattr(intent, "output_name", "") or output_name),
                    socket_idx,
                    explicit_name=explicit_name,
                )
                if sink_kind == "PROXY_ATTRIBUTE":
                    resolved.append(
                        ACESinkBinding(
                            sink_kind="PROXY_ATTRIBUTE",
                            target_object_name=proxy_name,
                            target_data_name=sink_name,
                            attr_type=attr_type,
                        )
                    )
                elif sink_kind == "OBJECT_PROPERTY":
                    consumers = tuple(str(v) for v in (getattr(intent, "consumer_keys", ()) or ()) if str(v))
                    if not consumers and frontend_uid in frontends:
                        consumers = tuple(str(v) for v in (frontends[frontend_uid].owner_keys or ()) if str(v))
                    for object_name in consumers:
                        resolved.append(
                            ACESinkBinding(
                                sink_kind="OBJECT_PROPERTY",
                                target_object_name=str(object_name),
                                target_data_name=sink_name,
                                attr_type=attr_type,
                            )
                        )

            if not resolved:
                sink_name = self._resolve_sink_data_name(proxy_name, node_id, output_name, socket_idx)
                if frontend_kind == FRONTEND_KIND_SHADER:
                    consumer_names = ()
                    if frontend_uid in frontends:
                        consumer_names = tuple(str(v) for v in (frontends[frontend_uid].owner_keys or ()) if str(v))
                    for object_name in consumer_names:
                        resolved.append(
                            ACESinkBinding(
                                sink_kind="OBJECT_PROPERTY",
                                target_object_name=object_name,
                                target_data_name=sink_name,
                                attr_type=attr_type,
                            )
                        )
                else:
                    resolved.append(
                        ACESinkBinding(
                            sink_kind="PROXY_ATTRIBUTE",
                            target_object_name=proxy_name,
                            target_data_name=sink_name,
                            attr_type=attr_type,
                        )
                    )

            deduped: list[ACESinkBinding] = []
            seen: set[tuple[str, str, str, str]] = set()
            for binding in resolved:
                key = (
                    str(getattr(binding, "sink_kind", "") or ""),
                    str(getattr(binding, "target_object_name", "") or ""),
                    str(getattr(binding, "target_data_name", "") or ""),
                    str(getattr(binding, "attr_type", "") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(binding)
            bindings_by_attr[attr_key] = deduped
        return bindings_by_attr

    @staticmethod
    def _safe_file_mtime(filepath: str) -> Optional[float]:
        if not filepath:
            return None
        try:
            return os.path.getmtime(filepath)
        except OSError:
            return None

    @staticmethod
    def _cleanup_bake_entry(entry: Dict[str, Any] | None):
        if not isinstance(entry, dict):
            return
        npz_obj = entry.pop("_npz_cache", None)
        if npz_obj is not None:
            try:
                close_fn = getattr(npz_obj, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass
        npz_path = str(entry.get("npz_path", "") or "")
        if npz_path:
            try:
                if os.path.isfile(npz_path):
                    os.remove(npz_path)
            except Exception:
                pass

    def _prune_bake_cache(self, sources, tree_signature):
        valid_ids: set[str] = set()
        existing_ids = [getattr(s, "stable_id", "") for s in sources if getattr(s, "stable_id", "")]
        for src in sources or []:
            sid = getattr(src, "stable_id", "") or ACEIdManager.ensure_source_uid(src, existing_ids=existing_ids)
            if sid:
                valid_ids.add(sid)
                if sid not in existing_ids:
                    existing_ids.append(sid)

        for sid in list(self._bake_cache.keys()):
            entry = self._bake_cache.get(sid, {})
            if sid not in valid_ids or entry.get("tree_signature") != tree_signature:
                self._cleanup_bake_entry(entry if isinstance(entry, dict) else None)
                self._bake_cache.pop(sid, None)

    def _zero_value_for_attr(self, attr_key: str):
        info = self.writer.output_defs.get(attr_key) if self.writer else None
        attr_type = info.get("type", "FLOAT") if info else "FLOAT"
        if attr_type == "FLOAT_VECTOR":
            return (0.0, 0.0, 0.0)
        if attr_type == "FLOAT_COLOR":
            return (0.0, 0.0, 0.0, 1.0)
        return 0.0

    def _normalize_output_value(self, attr_key: str, value: Any):
        info = self.writer.output_defs.get(attr_key) if self.writer else None
        attr_type = info.get("type", "FLOAT") if info else "FLOAT"

        if attr_type == "FLOAT":
            if isinstance(value, (np.floating, float, int, np.integer)):
                return float(value)
            if isinstance(value, (np.ndarray, list, tuple)):
                arr = np.asarray(value, dtype=np.float32).ravel()
                return float(arr[0]) if arr.size > 0 else 0.0
            return 0.0

        target_len = 4 if attr_type == "FLOAT_COLOR" else 3
        if isinstance(value, (np.ndarray, list, tuple)):
            arr = np.asarray(value, dtype=np.float32).ravel()
        elif isinstance(value, (np.floating, float, int, np.integer)):
            arr = np.asarray([float(value)], dtype=np.float32)
        else:
            arr = np.zeros(0, dtype=np.float32)
        padded = np.zeros(target_len, dtype=np.float32)
        if arr.size > 0:
            padded[: min(target_len, arr.size)] = arr[:target_len]
        if attr_type == "FLOAT_COLOR" and target_len == 4 and arr.size < 4:
            padded[3] = 1.0
        return tuple(float(x) for x in padded.tolist())

    def _rebuild_graph_execution_order(self, graph: ACEAudioGraph):
        tasks = dict(getattr(graph, "tasks", {}) or {})
        if not tasks:
            graph.execution_order = []
            graph.eval_plan_cache = {}
            return

        indegree: Dict[str, int] = {node_id: 0 for node_id in tasks.keys()}
        downstream: Dict[str, set[str]] = {node_id: set() for node_id in tasks.keys()}

        for node_id, task in tasks.items():
            for conn in (getattr(task, "inputs", {}) or {}).values():
                src_task = getattr(conn, "task", None)
                src_id = str(getattr(src_task, "node_id", "") or "")
                if src_id in indegree:
                    indegree[node_id] += 1
                    downstream.setdefault(src_id, set()).add(node_id)

        queue = deque(sorted([nid for nid, deg in indegree.items() if deg == 0]))
        order: list[str] = []
        while queue:
            current = queue.popleft()
            order.append(current)
            for dst in sorted(downstream.get(current, set())):
                indegree[dst] = int(indegree.get(dst, 0)) - 1
                if indegree[dst] == 0:
                    queue.append(dst)

        if len(order) < len(tasks):
            seen = set(order)
            for node_id in sorted(tasks.keys()):
                if node_id not in seen:
                    order.append(node_id)

        graph.execution_order = order
        graph.eval_plan_cache = {}

    def _graph_eval_plan_for_targets(self, graph: ACEAudioGraph, target_node_ids: list[str]) -> list[str]:
        if not target_node_ids:
            return []
        key = tuple(sorted({str(node_id) for node_id in target_node_ids if node_id}))
        if not key:
            return []

        cache = getattr(graph, "eval_plan_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            graph.eval_plan_cache = cache
        cached = cache.get(key)
        if isinstance(cached, list):
            return cached

        required: set[str] = set(key)
        stack = list(key)
        while stack:
            node_id = stack.pop()
            task = graph.tasks.get(node_id)
            if task is None:
                continue
            for conn in (getattr(task, "inputs", {}) or {}).values():
                src_task = getattr(conn, "task", None)
                src_id = str(getattr(src_task, "node_id", "") or "")
                if not src_id or src_id not in graph.tasks or src_id in required:
                    continue
                required.add(src_id)
                stack.append(src_id)

        exec_order = list(getattr(graph, "execution_order", []) or [])
        if not exec_order:
            self._rebuild_graph_execution_order(graph)
            exec_order = list(getattr(graph, "execution_order", []) or [])

        plan = [node_id for node_id in exec_order if node_id in required]
        if len(plan) < len(required):
            seen = set(plan)
            for node_id in sorted(required):
                if node_id not in seen:
                    plan.append(node_id)

        cache[key] = plan
        if len(cache) > 32:
            cache.pop(next(iter(cache)), None)
        return plan

    def _evaluate_graph_plan_iterative(self, graph: ACEAudioGraph, frame_index: int, plan: list[str]):
        frame_idx = int(frame_index)
        for node_id in plan:
            task = graph.tasks.get(node_id)
            if task is None:
                continue
            if int(getattr(task, "last_frame_processed", -1)) == frame_idx:
                continue

            setattr(task, "_current_frame_request", frame_idx)
            input_data: Dict[int, Any] = {}
            for input_idx, conn in (getattr(task, "inputs", {}) or {}).items():
                upstream = getattr(conn, "task", None)
                if upstream is None:
                    continue
                socket_idx = int(getattr(conn, "socket_index", 0) or 0)
                if int(getattr(upstream, "last_frame_processed", -1)) == frame_idx:
                    extract_fn = getattr(upstream, "_extract_result", None)
                    if callable(extract_fn):
                        try:
                            input_data[input_idx] = extract_fn(socket_idx)
                            continue
                        except Exception:
                            pass
                try:
                    input_data[input_idx] = upstream.get_output(frame_idx, socket_idx)
                except Exception:
                    input_data[input_idx] = None

            try:
                t0 = time.perf_counter()
                task.cached_output = task.process(input_data)
                self._dsp_profile_add(task, time.perf_counter() - t0)
                tag_fn = getattr(task, "_tag_output_frame", None)
                if callable(tag_fn):
                    tag_fn(task.cached_output, frame_idx)
                task.last_frame_processed = frame_idx
            except Exception:
                continue

    def _collect_graph_results_for_frame(
        self,
        graph: ACEAudioGraph,
        frame_index: int,
        attr_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        output_slots = self._graph_output_slot_specs(graph, attr_keys=attr_keys)
        target_node_ids = sorted({node_id for _, node_id, _ in output_slots if node_id in graph.tasks})
        plan = self._graph_eval_plan_for_targets(graph, target_node_ids)
        if plan:
            t0_plan = time.perf_counter()
            self._evaluate_graph_plan_iterative(graph, frame_index, plan)
            self._perf_add("dsp", time.perf_counter() - t0_plan)
        for attr_key, node_id, socket_idx in output_slots:
            task = graph.tasks.get(node_id)
            if not task:
                continue
            if int(getattr(task, "last_frame_processed", -1)) == int(frame_index):
                extract_fn = getattr(task, "_extract_result", None)
                if callable(extract_fn):
                    try:
                        val = extract_fn(socket_idx)
                    except Exception:
                        val = task.get_output(frame_index, socket_idx)
                else:
                    val = task.get_output(frame_index, socket_idx)
            else:
                t0 = time.perf_counter()
                val = task.get_output(frame_index, socket_idx)
                self._perf_add("dsp", time.perf_counter() - t0)
            if val is None:
                continue
            results[attr_key] = self._normalize_output_value(attr_key, val)
        return results

    def _get_baked_frame_results(self, graph: ACEAudioGraph, frame_index: int) -> Optional[Dict[str, Any]]:
        source_uid = getattr(graph, "source_uid", "") or getattr(getattr(graph, "source_context", None), "stable_id", "")
        if not source_uid:
            return None
        entry = self._bake_cache.get(source_uid)
        if not entry:
            return None
        if entry.get("tree_signature") != self._last_tree_signature:
            return None
        if int(entry.get("block_size", -1)) != int(getattr(graph, "block_size", 0)):
            return None

        source_ctx = getattr(graph, "source_context", None)
        if not source_ctx or getattr(source_ctx, "source_type", "") != "FILE":
            return None
        filepath = getattr(source_ctx, "filepath", "") or ""
        if not filepath or filepath != entry.get("filepath", ""):
            return None
        if self._safe_file_mtime(filepath) != entry.get("mtime"):
            return None

        frame_start = int(entry.get("frame_start", 0))
        frame_end = int(entry.get("frame_end", -1))
        if frame_index < frame_start or frame_index > frame_end:
            return None

        frame_offset = frame_index - frame_start
        values = entry.get("values", {})
        npz_cache = entry.get("_npz_cache", None)
        npz_path = str(entry.get("npz_path", "") or "")
        npz_keys = entry.get("npz_keys", {}) if isinstance(entry.get("npz_keys", {}), dict) else {}
        if npz_cache is None and npz_path:
            if not os.path.isfile(npz_path):
                return None
            try:
                npz_cache = np.load(npz_path, allow_pickle=False)
                entry["_npz_cache"] = npz_cache
            except Exception:
                return None
        results: Dict[str, Any] = {}
        for attr_key in graph.output_attr_keys:
            series = values.get(attr_key) if isinstance(values, dict) else None
            if series is None and npz_cache is not None:
                npz_key = npz_keys.get(attr_key)
                if isinstance(npz_key, str) and npz_key in npz_cache:
                    series = npz_cache[npz_key]
            if isinstance(series, np.ndarray):
                if frame_offset >= int(series.shape[0]):
                    continue
                sample = series[frame_offset]
                if series.ndim <= 1:
                    results[attr_key] = float(sample)
                else:
                    arr = np.asarray(sample, dtype=np.float32).ravel()
                    results[attr_key] = tuple(float(x) for x in arr.tolist())
                continue
            if not isinstance(series, list) or frame_offset >= len(series):
                continue
            results[attr_key] = series[frame_offset]
        return results if results else None

    def _sync_auto_attr_ids(
        self,
        bl_tree: bpy.types.NodeTree,
        abstract_tree: ACEAbstractTree,
        scene,
    ) -> tuple[bool, set[str]]:
        changed = False
        changed_node_ids: set[str] = set()
        source_lookup = self._build_source_proxy_lookup(scene)
        unique_proxies = list(dict.fromkeys(source_lookup.values()))
        incoming_by_target: Dict[str, list[tuple[str, int, int]]] = {}
        for link in (getattr(abstract_tree, "links", []) or []):
            source_node_id = str(getattr(link, "source_node_id", "") or "")
            target_node_id = str(getattr(link, "target_node_id", "") or "")
            if not source_node_id or not target_node_id:
                continue
            incoming_by_target.setdefault(target_node_id, []).append(
                (
                    source_node_id,
                    int(getattr(link, "source_socket_idx", 0) or 0),
                    int(getattr(link, "target_socket_idx", 0) or 0),
                )
            )

        for bl_node in getattr(bl_tree, "nodes", []):
            ace_id = ""
            try:
                ace_id = bl_node.get("ace_id", "")
            except Exception:
                ace_id = ""
            if not ace_id:
                continue

            abs_node = abstract_tree.nodes.get(ace_id)
            if not abs_node:
                continue

            is_stream = abs_node.category == "stream"
            is_specs = abs_node.type == "src_specs" and abs_node.category == "utilities"
            is_stream_accumulator = abs_node.type == "stream_accumulator" and abs_node.category == "utilities"
            if not (is_stream or is_specs or is_stream_accumulator):
                continue

            proxy_name = self._resolve_proxy_name_for_output_node(abs_node, source_lookup, unique_proxies)
            if not proxy_name:
                continue

            root_sig = json.dumps(sorted(set(getattr(abs_node, "root_sources", set()) or set())))
            upstream_sig = json.dumps(sorted(incoming_by_target.get(abs_node.id, [])))
            topology_sig = f"{root_sig}|{upstream_sig}"

            desired_map: Dict[str, str] = {}
            socket_map: Dict[str, str] = {}
            if is_stream and len(abs_node.outputs) > 0:
                for out_idx, out_sock in enumerate(abs_node.outputs):
                    desired_map[str(out_idx)] = self._build_auto_attr_name(
                        proxy_name,
                        abs_node.id,
                        out_sock.name,
                        out_idx,
                        topology_sig=topology_sig,
                    )
                    socket_name = "Attribute ID" if out_idx == 0 else f"Attribute ID {int(out_idx + 1)}"
                    socket_map[socket_name] = str(out_idx)
            elif is_specs:
                specs_socket_map = (
                    ("Frames per Beat Attribute ID", 0),
                    ("Beat Phase Attribute ID", 1),
                )
                for socket_name, out_idx in specs_socket_map:
                    if out_idx >= len(abs_node.outputs):
                        continue
                    desired_map[str(out_idx)] = self._build_auto_attr_name(
                        proxy_name,
                        abs_node.id,
                        abs_node.outputs[out_idx].name,
                        out_idx,
                        topology_sig=topology_sig,
                    )
                    socket_map[socket_name] = str(out_idx)
            elif is_stream_accumulator and len(abs_node.outputs) > 0:
                desired_map["0"] = self._build_auto_attr_name(
                    proxy_name,
                    abs_node.id,
                    abs_node.outputs[0].name,
                    0,
                    topology_sig=topology_sig,
                )
                socket_map["Attribute ID"] = "0"

            if not desired_map:
                continue

            try:
                node_tree = getattr(bl_node, "node_tree", None)
                if node_tree is not None:
                    for item in getattr(node_tree.interface, "items_tree", []):
                        if (
                            getattr(item, "in_out", None) == "INPUT"
                            and getattr(item, "name", "") in socket_map
                            and hasattr(item, "hide_value")
                        ):
                            item.hide_value = False
            except Exception:
                pass
            for sock in getattr(bl_node, "inputs", []):
                if getattr(sock, "name", "") in socket_map:
                    try:
                        sock.hide_value = False
                    except Exception:
                        pass

            socket_bindings: list[tuple[Any, str, str]] = []
            for sock in getattr(bl_node, "inputs", []):
                socket_key = socket_map.get(getattr(sock, "name", ""))
                if socket_key is None:
                    continue
                current = str(getattr(sock, "default_value", "") or "")
                target = desired_map.get(socket_key, "")
                socket_bindings.append((sock, current, target))

            for sock, current, target in socket_bindings:
                if not target or current == target:
                    continue
                try:
                    sock.default_value = target
                    changed = True
                    changed_node_ids.add(str(abs_node.id))
                except Exception:
                    pass

            try:
                if str(bl_node.get("ace_auto_attr_topology_json", "")) != topology_sig:
                    bl_node["ace_auto_attr_topology_json"] = topology_sig
                    changed = True
            except Exception:
                pass
        return changed, changed_node_ids

    @staticmethod
    def _attribute_socket_to_output_index(sock_name: str) -> Optional[int]:
        name = str(sock_name or "").strip()
        if name == "Attribute ID":
            return 0
        if name.startswith("Attribute ID "):
            suffix = name[len("Attribute ID "):].strip()
            if suffix.isdigit():
                idx = int(suffix) - 1
                if idx >= 0:
                    return int(idx)
        mapping = {
            "Frames per Beat Attribute ID": 0,
            "Beat Phase Attribute ID": 1,
        }
        return mapping.get(name)

    def _refresh_abstract_attr_params_from_tree(
        self,
        abstract_tree: ACEAbstractTree,
        bl_tree: bpy.types.NodeTree,
        node_ids: set[str],
    ):
        if not node_ids:
            return
        node_id_set = {str(nid) for nid in node_ids if nid}
        if not node_id_set:
            return

        for bl_node in getattr(bl_tree, "nodes", []):
            ace_id = ""
            try:
                ace_id = str(bl_node.get("ace_id", "") or "")
            except Exception:
                ace_id = ""
            if not ace_id or ace_id not in node_id_set:
                continue

            abs_node = abstract_tree.nodes.get(ace_id)
            if abs_node is None:
                continue

            params = dict(getattr(abs_node, "parameters", {}) or {})
            for idx, sock in enumerate(getattr(bl_node, "inputs", [])):
                if bool(getattr(sock, "is_linked", False)):
                    params.pop(f"in_{idx}", None)
                else:
                    val = getattr(sock, "default_value", None)
                    if hasattr(val, "to_tuple"):
                        val = val.to_tuple()
                    elif hasattr(val, "to_list"):
                        val = val.to_list()
                    if val is not None:
                        params[f"in_{idx}"] = val
                    else:
                        params.pop(f"in_{idx}", None)

                out_idx = self._attribute_socket_to_output_index(str(getattr(sock, "name", "") or ""))
                if out_idx is None:
                    continue
                raw_val = getattr(sock, "default_value", None)
                attr_name = str(raw_val).strip() if isinstance(raw_val, str) else ""
                if attr_name:
                    params[f"attr_id_out_{out_idx}"] = attr_name
                else:
                    params.pop(f"attr_id_out_{out_idx}", None)
                params[f"attr_id_custom_out_{out_idx}"] = bool(attr_name)

            abs_node.parameters = params

    def _sync_shader_output_ids(
        self,
        bl_tree: bpy.types.NodeTree,
        abstract_tree: ACEAbstractTree,
        scene,
    ) -> tuple[bool, set[str]]:
        """Best-effort shader-side output-name sync for editable ID sockets."""
        utils.assert_main_thread("_sync_shader_output_ids")
        changed = False
        changed_node_ids: set[str] = set()
        source_lookup = self._build_source_proxy_lookup(scene)
        unique_proxies = list(dict.fromkeys(source_lookup.values()))
        shader_tree_mutations_unsafe = not self._structural_tree_writes_allowed(
            scene,
            tree=bl_tree,
            frontend_kind=FRONTEND_KIND_SHADER,
        )
        shader_factory = None
        try:
            from ..nodes.shader_tree import ACEShaderNodeFactory as _ACEShaderNodeFactory
            shader_factory = _ACEShaderNodeFactory
        except Exception:
            shader_factory = None
        if shader_factory is not None and not shader_tree_mutations_unsafe:
            try:
                if shader_factory.cleanup_tree_helper_nodes(bl_tree):
                    changed = True
            except Exception:
                pass

        incoming_by_target: Dict[str, list[tuple[str, int, int]]] = {}
        for link in (getattr(abstract_tree, "links", []) or []):
            source_node_id = str(getattr(link, "source_node_id", "") or "")
            target_node_id = str(getattr(link, "target_node_id", "") or "")
            if not source_node_id or not target_node_id:
                continue
            incoming_by_target.setdefault(target_node_id, []).append(
                (
                    source_node_id,
                    int(getattr(link, "source_socket_idx", 0) or 0),
                    int(getattr(link, "target_socket_idx", 0) or 0),
                )
            )

        for bl_node in getattr(bl_tree, "nodes", []):
            try:
                ace_id = str(bl_node.get("ace_id", "") or "")
            except Exception:
                ace_id = ""
            if not ace_id:
                continue
            abs_node = abstract_tree.nodes.get(ace_id)
            if abs_node is None:
                continue
            is_stream = abs_node.category == "stream"
            is_specs = abs_node.type == "src_specs" and abs_node.category == "utilities"
            is_rolling_avg = abs_node.type == "util_rolling_avg" and abs_node.category == "utilities"
            is_stream_accumulator = abs_node.type == "stream_accumulator" and abs_node.category == "utilities"
            if not (is_stream or is_specs or is_rolling_avg or is_stream_accumulator):
                continue

            root_sig = json.dumps(sorted(set(getattr(abs_node, "root_sources", set()) or set())))
            upstream_sig = json.dumps(sorted(incoming_by_target.get(abs_node.id, [])))
            topology_sig = f"{root_sig}|{upstream_sig}"
            proxy_name = self._resolve_proxy_name_for_output_node(abs_node, source_lookup, unique_proxies)

            desired_map: Dict[str, str] = {}
            socket_map: Dict[str, str] = {}
            if proxy_name:
                # Shader sinks write to consumer object custom properties — one
                # per node instance — so topology_sig is NOT needed for
                # disambiguation and must be omitted to keep the name stable.
                # Using topology_sig here creates a mismatch with
                # _resolve_sink_bindings_for_graph's fallback path (which also
                # omits it) and with the initial gn_menu sync.
                if is_stream and len(abs_node.outputs) > 0:
                    for out_idx, out_sock in enumerate(abs_node.outputs):
                        desired_map[str(out_idx)] = self._build_auto_attr_name(
                            proxy_name,
                            abs_node.id,
                            out_sock.name,
                            out_idx,
                            topology_sig="",
                        )
                        socket_name = "Attribute ID" if out_idx == 0 else f"Attribute ID {int(out_idx + 1)}"
                        socket_map[socket_name] = str(out_idx)
                elif is_specs:
                    specs_socket_map = (
                        ("Frames per Beat Attribute ID", 0),
                        ("Beat Phase Attribute ID", 1),
                    )
                    for socket_name, out_idx in specs_socket_map:
                        if out_idx >= len(abs_node.outputs):
                            continue
                        desired_map[str(out_idx)] = self._build_auto_attr_name(
                            proxy_name,
                            abs_node.id,
                            abs_node.outputs[out_idx].name,
                            out_idx,
                            topology_sig="",
                        )
                        socket_map[socket_name] = str(out_idx)
                elif is_rolling_avg and len(abs_node.outputs) > 0:
                    desired_map["0"] = self._build_auto_attr_name(
                        proxy_name,
                        abs_node.id,
                        abs_node.outputs[0].name,
                        0,
                        topology_sig="",
                    )
                    socket_map["Attribute ID"] = "0"
                elif is_stream_accumulator and len(abs_node.outputs) > 0:
                    desired_map["0"] = self._build_auto_attr_name(
                        proxy_name,
                        abs_node.id,
                        abs_node.outputs[0].name,
                        0,
                        topology_sig="",
                    )
                    socket_map["Attribute ID"] = "0"

            if desired_map:
                for sock in getattr(bl_node, "inputs", []):
                    socket_key = socket_map.get(str(getattr(sock, "name", "") or ""))
                    if socket_key is None:
                        continue
                    target = desired_map.get(socket_key, "")
                    current = str(getattr(sock, "default_value", "") or "")
                    if target and target != current:
                        try:
                            sock.default_value = target
                            changed = True
                            changed_node_ids.add(str(abs_node.id))
                        except Exception:
                            pass
                try:
                    if str(bl_node.get("ace_auto_attr_topology_json", "")) != topology_sig:
                        bl_node["ace_auto_attr_topology_json"] = topology_sig
                        changed = True
                except Exception:
                    pass
            # Allow sync during viewport Material Preview / Rendered view —
            # only a full render job (use_lock_interface=False) is truly unsafe.
            # Viewport renders re-sync Cycles per frame anyway, so updating the
            # ShaderNodeAttribute attribute_name or creating a missing helper
            # node during viewport render is safe and necessary.
            if shader_factory is not None and not shader_tree_mutations_unsafe:
                try:
                    _names_by_idx = {int(k): str(v) for k, v in desired_map.items() if v} or None
                    if shader_factory.sync_node_instance_group_bindings(
                        bl_node,
                        proxy_name=proxy_name,
                        scene=scene,
                        attr_names_by_idx=_names_by_idx,
                    ):
                        changed = True
                        changed_node_ids.add(str(abs_node.id))
                except Exception:
                    pass
            # Write computed attr names into abstract tree params so that
            # _emit_consumer_intents / _resolve_sink_data_name use the same
            # names that were set on the companion ShaderNodeAttribute helpers.
            if desired_map:
                params = dict(getattr(abs_node, "parameters", {}) or {})
                for out_key, attr_name in desired_map.items():
                    if attr_name:
                        params[f"attr_id_out_{out_key}"] = attr_name
                        params[f"attr_id_custom_out_{out_key}"] = True
                abs_node.parameters = params
                changed = True
                changed_node_ids.add(str(abs_node.id))
        return changed, changed_node_ids

    def _create_task_for_node(self, node: ACEAbstractNode, block_size: int) -> Optional[ACEAudioTask]:
        cat = node.category
        typ = node.type

        if cat == 'source_mngmt':
            if typ == 'src':
                return SourceTask(node.id, node.parameters, block_size)
        elif cat == 'utilities':
            if typ == 'src_specs':
                return SourceSpecsTask(node.id, node.parameters, block_size)
            if typ == 'preview_sink':
                return PreviewSinkTask(node.id, node.parameters, block_size)
            if typ == 'audio_math':
                return AudioMathTask(node.id, node.parameters, block_size)
            if typ == 'util_rolling_avg':
                return RollingAvgTask(node.id, node.parameters, block_size)
            if typ == 'stream_accumulator':
                return StreamAccumulatorTask(node.id, node.parameters, block_size)
        elif cat == 'delta':
            if typ == 'frequency_splitter':
                return FrequencySplitterTask(node.id, node.parameters, block_size)
            if typ == 'midside_splitter':
                return MidsideSplitterTask(node.id, node.parameters, block_size)
            if typ == 'attackrelease_splitter':
                return AttackReleaseSplitterTask(node.id, node.parameters, block_size)
            if typ == 'chromagram':
                return ChromagramTask(node.id, node.parameters, block_size)
            if typ == 'auto_component_splitter':
                return AutoComponentSplitterTask(node.id, node.parameters, block_size)
        elif cat == 'stream':
            if typ == 'rms':
                return RMSTask(node.id, node.parameters, block_size)
            if typ == 'lufs':
                return LUFSTask(node.id, node.parameters, block_size)
            if typ == 'peak':
                return PeakTask(node.id, node.parameters, block_size)
            if typ == 'pitch':
                return PitchTask(node.id, node.parameters, block_size)
            if typ == 'spectral_centroid':
                return SpectralCentroidTask(node.id, node.parameters, block_size)
            if typ == 'spectral_flux':
                return SpectralFluxTask(node.id, node.parameters, block_size)
            if typ == 'onset':
                return OnsetTask(node.id, node.parameters, block_size)
            if typ == 'zero_crossing_rate':
                return ZeroCrossTask(node.id, node.parameters, block_size)
            if typ == 'stereo_width':
                return StereoWidthTask(node.id, node.parameters, block_size)
            if typ == 'stereo_balance':
                return StereoBalanceTask(node.id, node.parameters, block_size)
            if typ == 'stereo_vector':
                return StereoVectorTask(node.id, node.parameters, block_size)

        return ACEAudioTask(node.id, node.parameters, block_size)

    def _parallel_cleanup_ipc_handles(self, meta: Dict[str, Any] | None):
        if not isinstance(meta, dict):
            return
        slot_keys = list(meta.pop("payload_slot_keys", []) or [])
        for slot_key in slot_keys:
            self._parallel_release_payload_slot(str(slot_key))
        meta.pop("ipc_handles", None)

    @staticmethod
    def _parallel_close_unlink_shm(shm_obj):
        if shm_obj is None:
            return
        try:
            shm_obj.close()
        except Exception:
            pass
        try:
            shm_obj.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def _parallel_destroy_payload_slot(self, slot_key: str):
        key = str(slot_key)
        self._parallel_payload_slot_refs.pop(key, None)
        prefix = key.split("#", 1)[0]
        free_list = self._parallel_payload_free_slots.get(prefix)
        if isinstance(free_list, deque):
            self._parallel_payload_free_slots[prefix] = deque(v for v in free_list if str(v) != key)
            if not self._parallel_payload_free_slots[prefix]:
                self._parallel_payload_free_slots.pop(prefix, None)
        for frame_key, mapped_slot in list(self._parallel_frame_payload_slots.items()):
            if str(mapped_slot) == key:
                self._parallel_frame_payload_slots.pop(frame_key, None)
        slot = self._parallel_payload_slots.pop(key, None)
        if not isinstance(slot, dict):
            return
        self._parallel_close_unlink_shm(slot.get("mono_shm"))
        self._parallel_close_unlink_shm(slot.get("stereo_shm"))

    def _parallel_clear_payload_slot_field(self, slot_key: str, field: str):
        entry = self._parallel_payload_slots.get(str(slot_key))
        if not isinstance(entry, dict):
            return
        shm_key = f"{field}_shm"
        bytes_key = f"{field}_bytes"
        shape_key = f"{field}_shape"
        self._parallel_close_unlink_shm(entry.pop(shm_key, None))
        entry.pop(bytes_key, None)
        entry.pop(shape_key, None)

    def _parallel_payload_slot_descriptor(self, slot_key: str, field: str) -> Optional[Dict[str, Any]]:
        entry = self._parallel_payload_slots.get(str(slot_key))
        if not isinstance(entry, dict):
            return None
        shm_obj = entry.get(f"{field}_shm")
        shape = tuple(int(v) for v in tuple(entry.get(f"{field}_shape", ()) or ()))
        if shm_obj is None or not shape:
            return None
        return {
            "name": str(shm_obj.name),
            "shape": shape,
            "dtype": "float32",
        }

    def _parallel_release_payload_slot(self, slot_key: str):
        key = str(slot_key)
        if key not in self._parallel_payload_slots:
            return
        ref_count = max(int(self._parallel_payload_slot_refs.get(key, 0) or 0) - 1, 0)
        self._parallel_payload_slot_refs[key] = int(ref_count)
        if ref_count > 0:
            return
        for frame_key, mapped_slot in list(self._parallel_frame_payload_slots.items()):
            if str(mapped_slot) == key:
                self._parallel_frame_payload_slots.pop(frame_key, None)
        prefix = key.split("#", 1)[0]
        free_list = self._parallel_payload_free_slots.setdefault(prefix, deque())
        if key not in free_list:
            free_list.append(key)

    def _parallel_release_payload_slots(self):
        if not self._parallel_payload_slots:
            return
        slot_keys = list(self._parallel_payload_slots.keys())
        for slot_key in slot_keys:
            self._parallel_destroy_payload_slot(slot_key)
        self._parallel_payload_free_slots.clear()

    def _parallel_reserve_payload_slot(
        self,
        graph_key: str,
        frame_index: int,
        src_id: str,
    ) -> tuple[str, bool]:
        graph_key = str(graph_key or "")
        src_key = str(src_id or "")
        frame_key = (graph_key, int(frame_index), src_key)
        existing = self._parallel_frame_payload_slots.get(frame_key)
        if existing and existing in self._parallel_payload_slots:
            self._parallel_payload_slot_refs[existing] = int(self._parallel_payload_slot_refs.get(existing, 0) or 0) + 1
            return str(existing), False

        slot_prefix = f"{graph_key}:{src_key}"
        free = self._parallel_payload_free_slots.get(slot_prefix)
        chosen_slot = ""
        if free:
            while free and not chosen_slot:
                candidate = str(free.popleft())
                if candidate in self._parallel_payload_slots:
                    chosen_slot = candidate
            if free is not None and not free:
                self._parallel_payload_free_slots.pop(slot_prefix, None)
        if not chosen_slot:
            slot_index = int(self._parallel_payload_slot_counters.get(slot_prefix, 0) or 0)
            chosen_slot = f"{slot_prefix}#{slot_index}"
            self._parallel_payload_slot_counters[slot_prefix] = int(slot_index) + 1
            self._parallel_payload_slots.setdefault(chosen_slot, {})
            self._parallel_payload_slot_refs[chosen_slot] = 0

        self._parallel_frame_payload_slots[frame_key] = chosen_slot
        self._parallel_payload_slot_refs[chosen_slot] = int(self._parallel_payload_slot_refs.get(chosen_slot, 0) or 0) + 1
        return chosen_slot, True

    def _parallel_ensure_payload_slot(
        self,
        slot_key: str,
        field: str,
        array: np.ndarray,
    ) -> Dict[str, Any]:
        key = str(slot_key)
        entry = self._parallel_payload_slots.setdefault(key, {})
        shm_key = f"{field}_shm"
        bytes_key = f"{field}_bytes"
        shape_key = f"{field}_shape"

        arr = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
        required_bytes = max(int(arr.nbytes), 4)
        shm_obj = entry.get(shm_key)
        current_bytes = int(entry.get(bytes_key, 0) or 0)
        if shm_obj is None or current_bytes < required_bytes:
            self._parallel_close_unlink_shm(shm_obj)
            shm_obj = mp_shared_memory.SharedMemory(create=True, size=required_bytes)
            entry[shm_key] = shm_obj
            entry[bytes_key] = int(required_bytes)
        shape = tuple(int(v) for v in arr.shape)
        entry[shape_key] = shape
        view = np.ndarray(shape, dtype=np.float32, buffer=shm_obj.buf)
        view[:] = arr
        return {
            "name": str(shm_obj.name),
            "shape": shape,
            "dtype": "float32",
        }

    def _parallel_payload_to_ipc(
        self,
        source_payloads: Dict[str, Dict[str, Any]],
        graph_key: str,
        frame_index: int,
    ) -> tuple[Dict[str, Any], list[str]]:
        payload_chunks = {
            str(src_id): dict(src_payload or {})
            for src_id, src_payload in dict(source_payloads or {}).items()
        }
        if not _SHARED_MEMORY_AVAILABLE:
            return {"source_chunks": payload_chunks}, []

        slot_keys: list[str] = []
        shm_map: Dict[str, Dict[str, Any]] = {}
        try:
            for src_id, src_payload in payload_chunks.items():
                mono = src_payload.get("mono")
                if not isinstance(mono, np.ndarray):
                    continue

                slot_key, needs_upload = self._parallel_reserve_payload_slot(
                    graph_key=str(graph_key or ""),
                    frame_index=int(frame_index),
                    src_id=str(src_id),
                )
                slot_keys.append(slot_key)
                mono_arr = np.ascontiguousarray(np.asarray(mono, dtype=np.float32))
                if needs_upload:
                    src_meta: Dict[str, Any] = {
                        "mono": self._parallel_ensure_payload_slot(slot_key, "mono", mono_arr),
                    }
                else:
                    mono_meta = self._parallel_payload_slot_descriptor(slot_key, "mono")
                    if mono_meta is None:
                        src_meta = {
                            "mono": self._parallel_ensure_payload_slot(slot_key, "mono", mono_arr),
                        }
                    else:
                        src_meta = {"mono": mono_meta}

                stereo = src_payload.get("stereo")
                if isinstance(stereo, np.ndarray):
                    stereo_arr = np.ascontiguousarray(np.asarray(stereo, dtype=np.float32))
                    if needs_upload:
                        src_meta["stereo"] = self._parallel_ensure_payload_slot(slot_key, "stereo", stereo_arr)
                    else:
                        stereo_meta = self._parallel_payload_slot_descriptor(slot_key, "stereo")
                        if stereo_meta is not None:
                            src_meta["stereo"] = stereo_meta
                        else:
                            src_meta["stereo"] = self._parallel_ensure_payload_slot(slot_key, "stereo", stereo_arr)
                elif needs_upload:
                    self._parallel_clear_payload_slot_field(slot_key, "stereo")

                src_payload.pop("mono", None)
                src_payload.pop("stereo", None)
                shm_map[src_id] = src_meta

            packed = {"source_chunks": payload_chunks}
            if shm_map:
                packed["source_chunks_shm"] = shm_map
            return packed, slot_keys
        except Exception as e:
            for slot_key in slot_keys:
                self._parallel_release_payload_slot(slot_key)
            self._log_runtime(
                "parallel_shm_ipc",
                f"Shared-memory IPC setup failed, falling back to pickle transport: {e}",
                "WARNING",
                min_interval=4.0,
            )
            return {"source_chunks": payload_chunks}, []

    def _clear_bake_cache(self):
        for source_id in list(self._bake_cache.keys()):
            entry = self._bake_cache.get(source_id)
            self._cleanup_bake_entry(entry if isinstance(entry, dict) else None)
            self._bake_cache.pop(source_id, None)

    def _shutdown_graphs(self, wait_parallel_pool: bool = False, clear_bake_cache: bool = False):
        self._stop_preview_worker()
        self._shutdown_parallel_pool(wait=wait_parallel_pool)
        for graph in self.graphs.values():
            for task in graph.tasks.values():
                close_fn = getattr(task, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        pass
        self.graphs.clear()
        self._block_size_state.clear()
        self._clear_blender_runtime_refs()
        self._pending_geometry_events.clear()
        self._clear_probe_runtime_state()
        self._last_tree_signature = None
        self._attr_schema_dirty.clear()
        self._last_attr_sync_frame.clear()
        self._source_specs_task_registry.clear()
        self.writer = None
        self._preview_last_chunk = None
        self._preview_last_eval_frame = -1
        self._preview_next_eval_ts = 0.0
        if clear_bake_cache:
            self._clear_bake_cache()
        self._bake_jobs.clear()
        try:
            while True:
                self._bake_result_queue.get_nowait()
        except queue.Empty:
            pass
        except Exception:
            pass

    def invalidate_runtime_caches(self):
        writer = self.writer
        if writer is None:
            return
        try:
            writer.clear_runtime_caches()
        except Exception:
            pass

    def begin_undo_recovery(self, scene=None, cooldown_s: float = 0.75) -> None:
        self.undo_generation = int(getattr(self, "undo_generation", 0) or 0) + 1
        self.undo_recovery_until = time.monotonic() + max(float(cooldown_s), 0.0)
        try:
            self.invalidate_runtime_caches()
        except Exception:
            pass
        try:
            SourceSpecsTask.shutdown_background_worker()
        except Exception:
            pass
        self._shutdown_graphs(wait_parallel_pool=False, clear_bake_cache=False)
        now = time.time()
        self._pending_scene_rebuild = True
        self._last_dirty_time = now
        self._first_dirty_time = now
        self._last_request_time = now
        if scene is not None:
            self.scene_key = self._scene_key(scene)
            self.scene_name = self._scene_name(scene)

    @staticmethod
    def _tree_has_ace_nodes(tree: bpy.types.NodeTree | None) -> bool:
        if not tree:
            return False
        try:
            if bool(tree.get("ace_frontend_uid", "")):
                return True
        except Exception:
            pass
        try:
            for node in getattr(tree, "nodes", []) or []:
                get_prop = getattr(node, "get", lambda _k, d=None: d)
                if bool(get_prop("ace_id")):
                    return True
                source_id = str(get_prop("ace_source_id", "") or "")
                if source_id and ACEIdManager.parse_node_id(source_id):
                    return True
                if str(getattr(node, "type", "") or "") != "GROUP":
                    continue
                group = getattr(node, "node_tree", None)
                group_name = str(getattr(group, "name", "") or "")
                norm_group_name = group_name.lstrip(".")
                if group_name.startswith(".ACE_") or norm_group_name.startswith("ACE_"):
                    return True
            return False
        except Exception:
            return False

    @staticmethod
    def _find_scene_target_tree(scene) -> Optional[bpy.types.NodeTree]:
        def _tree_from_object(obj):
            if not obj:
                return (None, None)
            ace_tree = None
            fallback = None
            for mod in getattr(obj, "modifiers", []):
                if getattr(mod, "type", None) == 'NODES' and getattr(mod, "node_group", None):
                    tree = mod.node_group
                    if fallback is None:
                        fallback = tree
                    if ACEAudioEngine._tree_has_ace_nodes(tree):
                        ace_tree = tree
                        break
            return ace_tree, fallback

        ace = getattr(scene, "audiocore", None)
        target_obj = getattr(ace, "gn_target_object", None) if ace else None
        target_ace, target_fallback = _tree_from_object(target_obj)
        if target_ace:
            utils.log(
                f"ACE: target tree resolved via 'gn_target_object' → '{target_ace.name}'",
                "DEBUG",
            )
            return target_ace

        sonar = bpy.data.objects.get(ACEProxyManager.SONAR_OBJECT_NAME)
        sonar_ace, sonar_fallback = _tree_from_object(sonar)
        if sonar_ace:
            utils.log(
                f"ACE: target tree resolved via Sonar object → '{sonar_ace.name}'",
                "DEBUG",
            )
            return sonar_ace
        if target_fallback:
            utils.log(
                f"ACE: target tree resolved via gn_target_object fallback (no ace_id nodes) → '{target_fallback.name}'",
                "DEBUG",
            )
            return target_fallback
        if sonar_fallback:
            utils.log(
                f"ACE: target tree resolved via Sonar fallback (no ace_id nodes) → '{sonar_fallback.name}'",
                "DEBUG",
            )
        return sonar_fallback

    def compile_scene_frontends(self, scene, preferred_tree=None):
        """Discover, parse, merge, and compile all ACE frontends in the scene."""
        utils.assert_main_thread("compile_scene_frontends")
        if scene is None:
            return
        if self._undo_recovery_active():
            self.request_recompile(None)
            return

        frontends = self.discover_scene_frontends(scene)
        self._last_frontend_membership_signature = self._frontend_membership_signature(frontends)
        has_shader_frontends = any(
            str(getattr(rec.frontend_ref, "kind", "") or "") == FRONTEND_KIND_SHADER
            for rec in (frontends or [])
        )
        if has_shader_frontends and not self._structural_tree_writes_allowed(
            scene,
            tree=preferred_tree,
            frontend_kind=FRONTEND_KIND_SHADER,
        ):
            target_tree = preferred_tree
            if str(getattr(target_tree, "bl_idname", "") or "") != "ShaderNodeTree":
                target_tree = next(
                    (
                        rec.tree
                        for rec in (frontends or [])
                        if str(getattr(rec.frontend_ref, "kind", "") or "") == FRONTEND_KIND_SHADER
                    ),
                    None,
                )
            self.request_recompile(target_tree)
            self._log_runtime(
                "shader_tree_preview_deferred",
                "Deferring ACE shader-tree rebuild while a live material/rendered viewport preview is active.",
                "WARNING",
                min_interval=5.0,
            )
            return
        sources = list(getattr(getattr(scene, "audiocore", None), "audio_sources", []) or [])
        # Shader driver bindings need concrete proxy object names before the
        # frontend sync stage; ensure proxies are materialized up-front.
        for source in sources:
            try:
                ACEProxyManager.ensure_proxy_for_source(source, scene=scene)
            except Exception:
                pass
        reserved_node_ids: set[str] = set()
        merged_tree = ACEAbstractTree()

        for frontend in frontends:
            tree = frontend.tree
            frontend_uid = str(frontend.frontend_ref.uid or "")
            frontend_kind = str(frontend.frontend_ref.kind or "")
            if not tree or not frontend_uid:
                continue

            if frontend_kind == FRONTEND_KIND_GEOMETRY:
                try:
                    from ..nodes.dummy_tree import ACEDummyNodeFactory
                    removed = ACEDummyNodeFactory.prune_orphaned_ace_nodes(tree, scene)
                    if removed:
                        utils.log(f"ACE: pruned {removed} orphaned source node(s) from tree '{tree.name}'", "INFO")
                except Exception:
                    pass

            try:
                ACEProxyManager.sync_managed_source_tree(tree, sources, scene=scene)
            except Exception:
                pass

            if frontend_kind == FRONTEND_KIND_SHADER:
                from ..nodes.shader_tree import ACEShaderTreeReader
                reader = ACEShaderTreeReader(
                    tree,
                    frontend_uid=frontend_uid,
                    frontend_kind=frontend_kind,
                    reserved_node_ids=reserved_node_ids,
                    consumer_keys=frontend.consumer_object_names,
                )
            else:
                reader = ACEDummyTreeReader(
                    tree,
                    frontend_uid=frontend_uid,
                    frontend_kind=frontend_kind,
                    reserved_node_ids=reserved_node_ids,
                    consumer_keys=frontend.consumer_object_names,
                )

            partial_tree = reader.build()
            changed = False
            changed_node_ids: set[str] = set()
            try:
                if frontend_kind == FRONTEND_KIND_SHADER:
                    changed, changed_node_ids = self._sync_shader_output_ids(tree, partial_tree, scene)
                else:
                    changed, changed_node_ids = self._sync_auto_attr_ids(tree, partial_tree, scene)
            except Exception:
                changed = False
                changed_node_ids = set()
            if changed:
                self._refresh_abstract_attr_params_from_tree(partial_tree, tree, changed_node_ids)
            self._refresh_consumer_intents_from_nodes(partial_tree)
            partial_tree.frontends[frontend_uid] = frontend.frontend_ref
            merged_tree.merge_from(partial_tree)

        self.compile(merged_tree, scene)
        self._compiled_frontend_refs.clear()
        self._compiled_frontend_snapshots = {
            rec.frontend_ref.uid: self._compiled_snapshot_from_discovery(rec)
            for rec in frontends
        }
        self._compiled_frontend_ptrs = {
            rec.frontend_ref.uid: int(self._node_tree_pointer(rec.tree))
            for rec in frontends
            if rec.tree is not None
        }
        self._compiled_frontend_uids = tuple(rec.frontend_ref.uid for rec in frontends)
        resolved_preferred_tree = preferred_tree or (frontends[0].tree if frontends else None)
        self._compiled_tree_ref = None
        self._compiled_tree_ptr = self._node_tree_pointer(resolved_preferred_tree) if resolved_preferred_tree else 0
        self._compiled_tree_frontend_uid = self._tree_frontend_uid(resolved_preferred_tree)
        self._compiled_tree_bl_idname = str(getattr(resolved_preferred_tree, "bl_idname", "") or "")
        self._pending_scene_rebuild = False
        self._dirty_frontend_uids.clear()

    def compile_from_blender_tree(
        self,
        bl_tree: bpy.types.NodeTree,
        scene,
    ):
        """Compatibility wrapper for scene-scoped frontend compilation.

        Public operators and older call sites still pass a specific Blender
        tree here. The actual compile path is now scene-wide:
          1. Discover all managed Geometry Nodes and Shader frontends.
          2. Parse and merge them into one ``ACEAbstractTree``.
          3. Compile one ``ACEAudioGraph`` per source.

        Must be called from the main thread (asserted via ``utils.assert_main_thread``).

        Args:
            bl_tree: Preferred frontend tree for compatibility bookkeeping.
            scene: The active ``bpy.types.Scene``.
        """
        utils.assert_main_thread("compile_from_blender_tree")
        if not scene:
            return
        self.compile_scene_frontends(scene, preferred_tree=bl_tree)
        self._compiled_tree_ref = None
        self._compiled_tree_ptr = self._node_tree_pointer(bl_tree) if bl_tree else 0
        self._compiled_tree_frontend_uid = self._tree_frontend_uid(bl_tree)
        self._compiled_tree_bl_idname = str(getattr(bl_tree, "bl_idname", "") or "")

    def adjust_graph_block_size_for_task(self, task: ACEAudioTask, new_block_size: int):
        """Update the block size for all tasks in the graph that owns *task*.

        SourceTask instances are updated via ``_set_block_size_from_engine``
        (which handles prefetch buffer resize).  All other tasks have their
        ``block_size``, ``silence_buffer``, and cached array attributes reset
        directly so stale workspace allocations are not reused.
        """
        target_graph = None
        for graph in self.graphs.values():
            if task in graph.tasks.values():
                target_graph = graph
                break
        if not target_graph or new_block_size <= 0:
            return
        current_size = getattr(target_graph, "block_size", None)
        if current_size == new_block_size:
            return
        target_graph.block_size = new_block_size
        for t in target_graph.tasks.values():
            if isinstance(t, SourceTask):
                try:
                    t._set_block_size_from_engine(new_block_size)
                except Exception:
                    t.block_size = new_block_size
                    t.silence_buffer = np.zeros(new_block_size, dtype=np.float32)
                    t._ensure_frontend_workspace(new_block_size)
                continue

            t.block_size = new_block_size
            t.silence_buffer = np.zeros(new_block_size, dtype=np.float32)
            if hasattr(t, "_mask_cache_key"):
                t._mask_cache_key = None
            if hasattr(t, "_mask_cache"):
                t._mask_cache = None
            if hasattr(t, "_cached_masks"):
                t._cached_masks = None
            if hasattr(t, "_cached_freq_bins"):
                t._cached_freq_bins = None
            if hasattr(t, "_work_fft"):
                t._work_fft = None
            if hasattr(t, "_work_fft_stereo"):
                t._work_fft_stereo = None

    @staticmethod
    def _resolve_block_size(
        sample_rate: int,
        fps: float,
        source_type: str = "FILE",
        timeslice_mode: str = "fixed",
        delta_frames: float = 1.0,
    ) -> int:
        try:
            return int(
                resolve_context_block_size(
                    sample_rate,
                    fps,
                    source_type=source_type,
                    timeslice_mode=timeslice_mode,
                    delta_frames=delta_frames,
                )
            )
        except TypeError:
            # Backward compatibility for older helper signatures.
            return int(resolve_context_block_size(sample_rate, fps, source_type=source_type))

    def _stabilize_graph_block_size(
        self,
        graph_key: str,
        current_size: int,
        proposed_size: int,
        mode: str = "fixed",
    ) -> int:
        """Apply hysteresis to block-size changes in adaptive timeslice mode.

        In ``fixed`` mode the proposed size is returned immediately.  In
        ``adaptive`` mode small changes must persist for
        ``_BLOCK_SIZE_HYSTERESIS_FRAMES`` consecutive frames before being
        accepted; large changes (≥ ``_BLOCK_SIZE_IMMEDIATE_DELTA`` samples)
        are always applied immediately to handle timeline scrubs.

        Args:
            graph_key: Stable key used to track per-graph hysteresis state.
            current_size: Currently active block size for this graph.
            proposed_size: Block size calculated from the current frame delta.
            mode: ``"fixed"`` or ``"adaptive"``.

        Returns:
            The block size to use for this frame.
        """
        curr = int(max(current_size or 0, 0))
        prop = int(max(proposed_size or 0, 0))
        timeslice_mode = str(mode or "fixed").strip().lower()

        if prop <= 0:
            return curr if curr > 0 else prop
        if timeslice_mode != "adaptive":
            self._block_size_state.pop(graph_key, None)
            return prop
        if curr <= 0 or prop == curr:
            self._block_size_state.pop(graph_key, None)
            return prop if prop > 0 else curr
        if abs(prop - curr) >= self._BLOCK_SIZE_IMMEDIATE_DELTA:
            # Large jumps usually come from timeline scrubs: adapt immediately.
            self._block_size_state.pop(graph_key, None)
            return prop

        state = self._block_size_state.get(graph_key)
        if state is None:
            self._block_size_state[graph_key] = {"candidate": prop, "count": 1}
            return curr
        if int(state.get("candidate", -1)) != prop:
            self._block_size_state[graph_key] = {"candidate": prop, "count": 1}
            return curr

        count = int(state.get("count", 1)) + 1
        if count < self._BLOCK_SIZE_HYSTERESIS_FRAMES:
            state["count"] = count
            return curr

        self._block_size_state.pop(graph_key, None)
        return prop

    @staticmethod
    def _is_rendering_job() -> bool:
        """Return True if Blender is currently executing a render job (F12 or batch).

        Uses ``bpy.app.is_job_running("RENDER")``, which covers both interactive
        F12 renders and headless ``--background`` renders.  When True,
        ``_unsafe_for_handler_writes()`` suppresses attribute/proxy writes
        unless ``scene.render.use_lock_interface`` is enabled — that flag
        serialises the render thread against the main thread and makes RNA
        mutations safe.

        This check is evaluated every frame inside ``execute_next_frame()`` to
        prevent geometry-node attribute mutations from racing against the render
        pipeline.  It returns False on any exception so that a missing or
        older Blender API never blocks normal playback.
        """
        try:
            checker = getattr(bpy.app, "is_job_running", None)
            return bool(checker("RENDER")) if callable(checker) else False
        except Exception:
            return False

    @staticmethod
    def _unsafe_for_handler_writes(scene) -> bool:
        """Avoid mesh/RNA writes from frame handlers while rendering without UI lock."""
        is_rendering = ACEAudioEngine._is_rendering_job()
        if not is_rendering:
            return False
        try:
            return not bool(getattr(getattr(scene, "render", None), "use_lock_interface", False))
        except Exception:
            return True

    @staticmethod
    def _is_shader_viewport_preview_active() -> bool:
        """Return True when a 3D viewport is compiling live shader previews."""
        try:
            wm = getattr(getattr(bpy, "context", None), "window_manager", None)
            windows = getattr(wm, "windows", []) if wm else []
            for window in windows:
                screen = getattr(window, "screen", None)
                if screen is None:
                    continue
                for area in getattr(screen, "areas", []) or []:
                    if str(getattr(area, "type", "") or "") != "VIEW_3D":
                        continue
                    spaces = getattr(area, "spaces", None)
                    active_space = getattr(spaces, "active", None) if spaces is not None else None
                    if active_space is None:
                        try:
                            space_iter = list(spaces or [])
                        except Exception:
                            space_iter = []
                    else:
                        space_iter = [active_space]
                    for space in space_iter:
                        if str(getattr(space, "type", "") or "") != "VIEW_3D":
                            continue
                        shading = getattr(space, "shading", None)
                        shading_type = str(getattr(shading, "type", "") or "")
                        if shading_type in {"RENDERED", "MATERIAL"}:
                            return True
        except Exception:
            return False
        return False

    @classmethod
    def _unsafe_for_shader_tree_mutations(cls, scene) -> bool:
        """Avoid structural ShaderNodeTree edits during renders and live preview."""
        if cls._unsafe_for_handler_writes(scene):
            return True
        return cls._is_shader_viewport_preview_active()

    @staticmethod
    def _is_animation_playing() -> bool:
        """Return True if any Blender window is currently playing back animation."""
        try:
            wm = getattr(getattr(bpy, "context", None), "window_manager", None)
            windows = getattr(wm, "windows", []) if wm else []
            for win in windows:
                screen = getattr(win, "screen", None)
                if screen and bool(getattr(screen, "is_animation_playing", False)):
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def _resample_signal_1d(signal: np.ndarray, target_len: int) -> np.ndarray:
        """Linear-interpolation resample of a 1-D float32 signal to *target_len* samples."""
        src = np.asarray(signal, dtype=np.float32).reshape(-1)
        n = int(src.shape[0] or 0)
        t_len = int(max(target_len, 1))
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

    @classmethod
    def _fit_preview_chunk_block(cls, chunk: AudioChunk, preview_block_size: int) -> AudioChunk:
        """Resample a chunk's mono and stereo signals to *preview_block_size* samples.

        Returns the original chunk unchanged if it already has the correct size.
        """
        target = int(max(preview_block_size, 1))
        mono = np.asarray(getattr(chunk, "mono", np.zeros(0, dtype=np.float32)), dtype=np.float32).reshape(-1)
        if mono.shape[0] == target:
            return chunk
        mono_resized = cls._resample_signal_1d(mono, target)

        stereo_raw = getattr(chunk, "stereo", None)
        stereo_resized = None
        if isinstance(stereo_raw, np.ndarray):
            stereo_arr = np.asarray(stereo_raw, dtype=np.float32)
            if stereo_arr.ndim == 2:
                if stereo_arr.shape[1] == mono.shape[0]:
                    stereo_resized = np.stack(
                        [cls._resample_signal_1d(stereo_arr[ch], target) for ch in range(int(stereo_arr.shape[0]))],
                        axis=0,
                    ).astype(np.float32, copy=False)
                elif stereo_arr.shape[0] == mono.shape[0]:
                    transposed = stereo_arr.T.astype(np.float32, copy=False)
                    resized_t = np.stack(
                        [cls._resample_signal_1d(transposed[ch], target) for ch in range(int(transposed.shape[0]))],
                        axis=0,
                    ).astype(np.float32, copy=False)
                    stereo_resized = resized_t

        return AudioChunk(
            mono_resized,
            stereo_resized,
            int(getattr(chunk, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE),
            provider=getattr(chunk, "provider", None),
            frame_index=utils.as_int(getattr(chunk, "frame_index", None), -1),
            band_indices=getattr(chunk, "band_indices", None),
            band_index_fft_size=getattr(chunk, "band_index_fft_size", None),
            band_hz_bounds=getattr(chunk, "band_hz_bounds", None),
            lineage=tuple(getattr(chunk, "lineage", ()) or ()),
            spectrum_token=getattr(chunk, "spectrum_token", None),
            analysis_ops=tuple(getattr(chunk, "analysis_ops", ()) or ()),
            latency_ms=float(getattr(chunk, "latency_ms", 0.0) or 0.0),
        )

    @staticmethod
    def _mix_preview_chunks(chunks: list[AudioChunk]) -> Optional[AudioChunk]:
        """Average-mix a list of stereo AudioChunks into one.

        Returns None for an empty list, the single chunk unchanged for a list of
        one, or a new AudioChunk with the equal-power average mix for multiple.
        A final peak limiter is applied to the mixed result to prevent clipping.
        """
        if not chunks:
            return None
        if len(chunks) == 1:
            return chunks[0]

        frames_list: list[np.ndarray] = []
        sample_rate = int(getattr(chunks[0], "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
        max_frames = 0
        for chunk in chunks:
            frames = _ACEPreviewAudioMonitor._chunk_to_stereo_frames(chunk)
            if frames is None:
                continue
            frames_list.append(np.asarray(frames, dtype=np.float32))
            max_frames = max(max_frames, int(frames.shape[0]))

        if not frames_list or max_frames <= 0:
            return None

        mixed = np.zeros((max_frames, 2), dtype=np.float32)
        mix_count = 0
        for frames in frames_list:
            frame_n = int(frames.shape[0] or 0)
            if frame_n <= 0:
                continue
            mixed[:frame_n, :] += frames[:frame_n, :]
            mix_count += 1

        if mix_count <= 0:
            return None
        if mix_count > 1:
            mixed *= (1.0 / float(mix_count))
        # Instant-attack vectorized limiter: match PreviewSinkTask behavior.
        ceiling = 0.98
        n_frames = int(mixed.shape[0])
        peaks = np.max(np.abs(mixed.reshape(n_frames, -1)), axis=1).astype(np.float64)
        desired = np.where(
            peaks > 1e-9,
            np.minimum(1.0, ceiling / np.where(peaks > 1e-9, peaks, 1.0)),
            1.0,
        )
        gains = np.minimum.accumulate(desired).astype(np.float32)
        mixed *= gains.reshape(-1, 1)
        np.clip(mixed, -ceiling, ceiling, out=mixed)

        stereo = mixed.T.astype(np.float32, copy=False)
        mono = (0.5 * (mixed[:, 0] + mixed[:, 1])).astype(np.float32, copy=False)
        latency_ms = max((float(getattr(chunk, "latency_ms", 0.0) or 0.0) for chunk in chunks), default=0.0)
        return AudioChunk(
            mono,
            stereo,
            sample_rate,
            lineage=("preview_mix",),
            analysis_ops=(),
            latency_ms=latency_ms,
        )

    @staticmethod
    def _chunk_is_unmasked_audio(chunk: Any) -> bool:
        """Return True if *chunk* is a plain, unprocessed audio block.

        Returns False for frequency-masked, band-limited, or analysis-ops-tagged
        chunks because these should not be played back directly through the
        preview monitor.
        """
        if not hasattr(chunk, "mono") or not hasattr(chunk, "sample_rate"):
            return False
        if getattr(chunk, "band_hz_bounds", None) is not None:
            return False
        idx = getattr(chunk, "band_indices", None)
        if idx is not None:
            try:
                if np.asarray(idx).size > 0:
                    return False
            except Exception:
                return False
        try:
            if bool(getattr(chunk, "analysis_ops", ())):
                return False
        except Exception:
            return False
        return True

    @staticmethod
    def _task_current_chunk_for_frame(task: Any, frame_index: int):
        """Return the task's cached current chunk if it was produced for *frame_index*.

        Returns None if the task has no cached chunk or the chunk belongs to a
        different frame.
        """
        try:
            chunk_frame = utils.as_int(getattr(task, "_current_chunk_frame", None), -1)
        except Exception:
            chunk_frame = -1
        if chunk_frame != int(frame_index):
            return None
        chunk = getattr(task, "_current_chunk", None)
        if hasattr(chunk, "mono") and hasattr(chunk, "sample_rate"):
            return chunk
        return None

    def _collect_preview_chunk(self, graph: ACEAudioGraph, frame_index: int, preview_block_size: int):
        """Collect and resample preview audio for *graph* at *frame_index*.

        Prefers direct SourceTask passthrough (zero DSP overhead) when no manual
        Preview Sink nodes are present.  Falls back to evaluating PreviewSinkTask
        outputs when biquad processing is needed.  Returns the mixed/resampled
        chunk, or None if no preview audio is available.
        """
        manual_ids = getattr(graph, "manual_preview_task_ids", []) or []
        auto_ids = getattr(graph, "auto_preview_task_ids", []) or []
        passthrough_specs = getattr(graph, "auto_preview_passthrough_specs", []) or []
        candidate_ids = manual_ids if manual_ids else (auto_ids or (getattr(graph, "preview_task_ids", []) or []))

        chunks: list[AudioChunk] = []
        if (not manual_ids) and passthrough_specs:
            for source_task_id, source_sock_idx in passthrough_specs:
                source_task = graph.tasks.get(str(source_task_id))
                if not isinstance(source_task, SourceTask):
                    continue
                try:
                    t0 = time.perf_counter()
                    direct_chunk = source_task.get_output(frame_index, int(source_sock_idx))
                    self._perf_add("preview", time.perf_counter() - t0)
                except Exception:
                    continue
                if direct_chunk is not None and self._chunk_is_unmasked_audio(direct_chunk):
                    chunks.append(self._fit_preview_chunk_block(direct_chunk, preview_block_size))

        for node_id in candidate_ids:
            task = graph.tasks.get(node_id)
            if not task:
                continue
            if 0 not in getattr(task, "inputs", {}):
                continue
            is_enabled_fn = getattr(task, "is_enabled", None)
            if callable(is_enabled_fn) and not is_enabled_fn():
                continue
            # Fast path: when auto-preview taps a direct source output branch with no
            # masking/analysis metadata, bypass PreviewSink reconstruction entirely.
            if isinstance(task, PreviewSinkTask):
                conn = getattr(task, "inputs", {}).get(0)
                upstream = getattr(conn, "task", None) if conn is not None else None
                if isinstance(upstream, SourceTask):
                    try:
                        t0 = time.perf_counter()
                        direct_chunk = upstream.get_output(frame_index, int(getattr(conn, "socket_index", 0) or 0))
                        self._perf_add("preview", time.perf_counter() - t0)
                    except Exception:
                        direct_chunk = None
                    if direct_chunk is not None and self._chunk_is_unmasked_audio(direct_chunk):
                        chunks.append(self._fit_preview_chunk_block(direct_chunk, preview_block_size))
                        continue
            try:
                t0 = time.perf_counter()
                chunk = task.get_output(frame_index, 0)
                self._perf_add("preview", time.perf_counter() - t0)
            except Exception:
                continue
            if hasattr(chunk, "mono") and hasattr(chunk, "sample_rate"):
                chunks.append(self._fit_preview_chunk_block(chunk, preview_block_size))
        return ACEAudioEngine._mix_preview_chunks(chunks)

    # ------------------------------------------------------------------
    # Preview DSP background worker (Fix G)
    # ------------------------------------------------------------------

    def _start_preview_worker(self):
        """Start the preview DSP thread. Idempotent: no-op if already running."""
        if self._preview_worker_thread is not None and self._preview_worker_thread.is_alive():
            return
        with self._preview_worker_cond:
            self._preview_worker_stop = False
        t = threading.Thread(
            target=self._preview_worker_loop, daemon=True, name="ACEPreviewWorker"
        )
        self._preview_worker_thread = t
        t.start()

    def _stop_preview_worker(self):
        """Signal the preview worker to stop and wait for it to exit."""
        with self._preview_worker_cond:
            self._preview_worker_stop = True
            self._preview_worker_queue.clear()
            self._preview_worker_cond.notify_all()
        t = self._preview_worker_thread
        self._preview_worker_thread = None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=1.0)

    def _preview_worker_loop(self):
        """Background thread: run PreviewSinkTask.process() and push to monitor."""
        monitor = self.__class__._preview_monitor
        while True:
            with self._preview_worker_cond:
                while not self._preview_worker_queue and not self._preview_worker_stop:
                    self._preview_worker_cond.wait(timeout=0.1)
                if self._preview_worker_stop:
                    break
                if not self._preview_worker_queue:
                    continue
                # Take the LATEST item only; drop any older submissions.
                item = self._preview_worker_queue[-1]
                self._preview_worker_queue.clear()
            task, upstream_chunk = item
            try:
                result_chunk = task.process({0: upstream_chunk})
                if result_chunk is not None and hasattr(result_chunk, "mono"):
                    monitor.push_chunk(result_chunk)
            except Exception:
                pass

    def _submit_preview_work(self, task, upstream_chunk) -> bool:
        """Enqueue a PreviewSinkTask for background DSP processing.

        Returns True if the work was accepted, False if the worker is stopped.
        Only the most-recent submission per frame is retained; older items are
        dropped to prevent queue buildup during heavy load.
        """
        with self._preview_worker_cond:
            if self._preview_worker_stop:
                return False
            self._preview_worker_queue.append((task, upstream_chunk))
            if len(self._preview_worker_queue) > 1:
                self._preview_worker_queue[:] = [self._preview_worker_queue[-1]]
            self._preview_worker_cond.notify()
        return True

    def _maybe_compile_dirty(self, scene) -> bool:
        """Recompile the dirty scene frontends if the quiet window is met.

        Requires ``_last_dirty_time`` to have been stable for 350 ms AND
        ``_last_compile_time`` to be at least 350 ms ago.  Both guards are
        bypassed when the dirty state has persisted for more than 2 s (rapid
        link-editing bursts that keep resetting the quiet timer).

        Returns:
            True if a recompile was performed, False otherwise.
        """
        if self._undo_recovery_active():
            return False
        now = time.time()
        discovered_frontends = self.discover_scene_frontends(scene)
        discovered_by_uid = {
            str(rec.frontend_ref.uid or ""): rec
            for rec in (discovered_frontends or [])
        }
        membership_sig = self._frontend_membership_signature(discovered_frontends)
        membership_changed = membership_sig != getattr(self, "_last_frontend_membership_signature", ())
        if not self._dirty_frontend_uids and not self._pending_scene_rebuild and not membership_changed:
            return False
        shader_frontend_uids = {
            str(rec.frontend_ref.uid or "")
            for rec in (discovered_frontends or [])
            if str(getattr(rec.frontend_ref, "kind", "") or "") == FRONTEND_KIND_SHADER
        }
        touches_shader_frontends = bool(
            any(
                str(self._dirty_tree_bl_idnames.get(int(tree_ptr), "") or "") == "ShaderNodeTree"
                for tree_ptr in (self._dirty_tree_ptrs or set())
            )
            or str(getattr(self, "_pending_tree_bl_idname", "") or "") == "ShaderNodeTree"
            or bool(shader_frontend_uids.intersection({str(uid) for uid in (self._dirty_frontend_uids or set())}))
            or ((self._pending_scene_rebuild or membership_changed) and bool(shader_frontend_uids))
        )
        if touches_shader_frontends and not self._structural_tree_writes_allowed(
            scene,
            frontend_kind=FRONTEND_KIND_SHADER,
            tree_bl_idname="ShaderNodeTree",
        ):
            self._log_runtime(
                "shader_tree_preview_deferred",
                "Deferring ACE shader-tree rebuild while a live material/rendered viewport preview is active.",
                "WARNING",
                min_interval=5.0,
            )
            return False
        # Require a quiet window and minimum spacing between rebuilds.
        # Exception: bypass the quiet window if dirty state has persisted for >2 s
        # (catches rapid link-editing bursts that perpetually reset the quiet timer).
        first_dirty = getattr(self, "_first_dirty_time", 0.0)
        bypass_quiet = first_dirty > 0.0 and (now - first_dirty) >= 2.0
        if not bypass_quiet and (now - self._last_dirty_time) < 0.35:
            return False
        if (now - self._last_compile_time) < 0.35:
            return False
        if self._is_compiling:
            return False
        if self._pending_scene_rebuild and not self._dirty_frontend_uids and not membership_changed:
            self._pending_scene_rebuild = False
            self._first_dirty_time = 0.0
            return False
        preferred_tree = None
        if self._dirty_frontend_uids:
            dirty_uid = next(iter(self._dirty_frontend_uids))
            preferred_tree = getattr(discovered_by_uid.get(dirty_uid), "tree", None)
        if preferred_tree is None:
            preferred_tree = self._resolve_preferred_tree_from_discoveries(
                discovered_frontends,
                tree_ptr=int(getattr(self, "_pending_tree_ptr", 0) or 0),
                tree_bl_idname=str(getattr(self, "_pending_tree_bl_idname", "") or ""),
            )
        self._pending_tree = None
        self._pending_tree_ptr = 0
        self._pending_tree_bl_idname = ""
        self._dirty_trees.clear()
        self._dirty_tree_ptrs.clear()
        self._dirty_tree_bl_idnames.clear()
        self._dirty_frontend_uids.clear()
        self._pending_scene_rebuild = False
        self._is_compiling = True
        compiled = False
        try:
            self.compile_scene_frontends(scene, preferred_tree=preferred_tree)
            compiled = True
        finally:
            self._is_compiling = False
            self._first_dirty_time = 0.0  # reset after compile
        return compiled

    def request_recompile(self, bl_tree: bpy.types.NodeTree | None):
        """Mark scene frontends as needing recompilation on the next compile poll.

        Rate-limited: multiple calls within 150 ms of each other update only
        ``_last_dirty_time`` without re-triggering the cooldown so rapid source
        property edits do not spam recompiles.

        Args:
            bl_tree: Tree to mark dirty, or None to request a scene-wide rebuild.
        """
        now = time.time()
        if self._undo_recovery_active():
            self._pending_scene_rebuild = True
            self._pending_tree = None
            self._pending_tree_ptr = 0
            self._pending_tree_bl_idname = ""
            self._last_dirty_time = now
            self._first_dirty_time = now
            return
        if bl_tree is None:
            self._pending_scene_rebuild = True
        if bl_tree:
            frontend_uid = self._tree_frontend_uid(bl_tree)
            tree_ptr = int(self._node_tree_pointer(bl_tree) or 0)
            tree_bl_idname = str(getattr(bl_tree, "bl_idname", "") or "")
            if frontend_uid:
                self._dirty_frontend_uids.add(frontend_uid)
            else:
                self._pending_scene_rebuild = True
            self._dirty_trees.add(tree_ptr)
            self._dirty_tree_ptrs.add(tree_ptr)
            self._dirty_tree_bl_idnames[tree_ptr] = tree_bl_idname
            self._pending_tree = None
            self._pending_tree_ptr = tree_ptr
            self._pending_tree_bl_idname = tree_bl_idname
        # Track when dirty state first appeared so _maybe_compile_dirty can enforce
        # a max wait (2 s) even when rapid changes keep resetting _last_dirty_time.
        if not getattr(self, "_first_dirty_time", 0.0):
            self._first_dirty_time = now
        if (now - getattr(self, "_last_request_time", 0.0)) < 0.15:
            self._last_dirty_time = now
            return
        self._last_dirty_time = now
        self._last_request_time = now

    def _refresh_source_specs_task_registry(self) -> None:
        """Rebuild the lookup used to flush SourceSpecsTask main-thread ctx writes."""
        registry: Dict[tuple[str, float], SourceSpecsTask] = {}
        for graph in (self.graphs or {}).values():
            for task in (getattr(graph, "tasks", {}) or {}).values():
                if not isinstance(task, SourceSpecsTask):
                    continue
                target = getattr(task, "_long_term_target", None)
                if target is None:
                    continue
                registry[target] = task
        self._source_specs_task_registry = registry

    def execute_next_frame(self, scene, force: bool = False):
        """Evaluate all graphs for the current scene frame and write results to proxies.

        Called every frame from ``frame_change_handler`` (and during render).
        The method:
          1. Updates FPS, BPM, and beat-signature snapshots.
          2. Patches per-source parallel IRs with any BPM changes
             (``_refresh_specs_ir_bpm``).
          3. Flushes background probe results (``_flush_probe_updates``).
          4. Triggers a recompile if the tree is dirty (``_maybe_compile_dirty``).
          5. For each graph: resolves block size, runs Stage B parallel
             submission/polling, evaluates local DSP, pushes preview audio, and
             writes numeric attribute values to the Blender proxy object.

        Args:
            scene: Active ``bpy.types.Scene``.
            force: When True, skip the same-frame deduplication guard.  Used by
                render handlers that need a guaranteed write even when the frame
                number has not changed.
        """
        utils.assert_main_thread("execute_next_frame")
        self.scene_key = self._scene_key(scene)
        if scene is not None:
            self.scene_name = self._scene_name(scene)
        if self._undo_recovery_active():
            return
        if self._unsafe_for_handler_writes(scene):
            self._log_runtime(
                "render_lock_interface_execute",
                "Skipping ACE frame writes during render. Enable Render > Lock Interface for safe handler updates.",
                "WARNING",
                min_interval=5.0,
            )
            return
        # Flush async bake progress/completions and any deferred SourceSpecs ctx writes.
        self.flush_bake_results()
        self._refresh_source_specs_task_registry()
        SourceSpecsTask.flush_pending_ctx_writes(self._source_specs_task_registry)
        if self._bake_jobs:
            # Bake worker evaluates the live graph/task set; suspend frame-runtime
            # evaluation while a bake is active to avoid cross-thread task mutation.
            self._log_runtime(
                "bake_runtime_pause",
                "ACE runtime graph evaluation paused while offline bake is active.",
                "INFO",
                min_interval=2.0,
            )
            return
        scene_frame = getattr(scene, "frame_current", None)
        if scene_frame is None:
            scene_frame = self.current_frame + 1

        prev_scene_frame = int(self.last_scene_frame)
        if (
            (not force)
            and scene_frame == prev_scene_frame
            and not self._pending_tree_ptr
            and not self._pending_scene_rebuild
            and not self._dirty_frontend_uids
        ):
            return

        try:
            delta_frames = float(int(scene_frame) - int(prev_scene_frame)) if prev_scene_frame >= 0 else 1.0
        except Exception:
            delta_frames = 1.0
        if abs(delta_frames) <= 1e-9:
            delta_frames = 1.0
        self._current_delta_frames = float(delta_frames)

        self.current_frame = int(scene_frame)
        self.last_scene_frame = int(scene_frame)

        fps = 24.0
        try:
            raw_fps = float(getattr(getattr(scene, "render", None), "fps", 24.0) or 24.0)
            fps_base = float(getattr(getattr(scene, "render", None), "fps_base", 1.0) or 1.0)
            if fps_base > 0.0:
                fps = raw_fps / fps_base
        except Exception:
            fps = 24.0
        self._current_scene_fps = max(float(fps), 1.0)

        # Keep per-source BPM / beats-per-bar in each graph's parallel IR fresh
        # so _WorkerSourceSpecsPassthroughTask always uses the current values.
        # This is especially important for LIVE sources that have no auto-BPM
        # detection: the user sets BPM manually and the worker must see the update
        # on the very next frame rather than waiting for the next full recompile.
        try:
            self._refresh_specs_ir_bpm()
        except Exception:
            pass

        # Flush background probe results before the compile so native_sample_rate
        # is populated for any sources whose probe thread has just completed.
        try:
            ace = getattr(scene, "audiocore", None)
            sources = list(getattr(ace, "audio_sources", []) or []) if ace else []
            if self._flush_probe_updates(
                sources,
                scene=scene,
                preferred_tree_ptr=int(getattr(self, "_compiled_tree_ptr", 0) or 0),
            ):
                pass  # recompile already requested inside _flush_probe_updates
            self.flush_source_tempo_detection_updates(sources)
        except Exception:
            pass

        if self._maybe_compile_dirty(scene):
            # Freshly compiled this frame — skip DSP writes so Blender's GN
            # evaluator gets one clean cycle with the new graph before we start
            # pushing attribute values into it.  Without this guard the first
            # frame after a recompile can produce partial / stale geometry output.
            return

        if not self.writer:
            return
        scheduler_mode = self._scheduler_mode(scene)
        timeslice_mode = self._timeslice_mode(scene)
        parallel_mode = self._parallel_mode(scene)
        parallel_pool_ready = False
        if parallel_mode == "processes_dsp":
            parallel_pool_ready = self._ensure_parallel_pool(scene)
        elif self._parallel_executor is not None:
            self._shutdown_parallel_pool()

        # Keep the preview feeder's drift-correction in sync with the scene fps.
        try:
            self.__class__._preview_monitor.update_scene_fps(fps)
        except Exception:
            pass

        # Seek detection: flush stale audio from all monitor buffers when the
        # user scrubs to a non-adjacent frame or scrubs backward. Without this,
        # the ring buffer can hold future-relative audio that plays at the wrong
        # timeline position.  Any backward frame (delta < 0) triggers a flush
        # because even going back one frame means buffered audio is now "ahead".
        try:
            if abs(delta_frames) > 2 or delta_frames < -0.5:
                self.__class__._preview_monitor.clear()
        except Exception:
            pass

        self._parallel_poll_futures(scene, current_frame=int(self.current_frame))
        preview_allowed = self._is_animation_playing()
        preview_block_size = int(self._preview_block_size(scene))
        preview_sample_rate = int(max(self.effective_sample_rate, SAMPLE_RATE, 1))
        _min_analysis_bs = 0
        for graph in self.graphs.values():
            source_ctx = getattr(graph, "source_context", None)
            source_sr = int(getattr(source_ctx, "sample_rate", 0) or 0)
            if source_sr > 0:
                preview_sample_rate = max(preview_sample_rate, source_sr)
            _gbs = int(getattr(graph, "block_size", 0) or 0)
            if _gbs >= 64:
                _min_analysis_bs = _gbs if _min_analysis_bs == 0 else min(_min_analysis_bs, _gbs)
        try:
            self.__class__._preview_monitor.hint_block_size(_min_analysis_bs)
        except Exception:
            pass
        preview_interval_s = max(
            float(preview_block_size) / float(max(preview_sample_rate, 1)),
            float(self._PREVIEW_MIN_INTERVAL_S),
        )
        now_ts = time.perf_counter()
        preview_eval_due = bool(
            preview_allowed
            and (
                self._preview_last_eval_frame < 0
                or now_ts >= float(self._preview_next_eval_ts)
            )
        )
        if preview_eval_due:
            self._preview_next_eval_ts = float(now_ts + preview_interval_s)
        preview_chunks: list[AudioChunk] = []
        preview_pushed = False
        geometry_proxy_names: set[str] = set()

        try:
            for _, graph in self.graphs.items():
                if parallel_mode == "processes_dsp":
                    self._parallel_poll_futures(scene, current_frame=int(self.current_frame))
                graph.scheduler_mode = scheduler_mode
                graph.timeslice_mode = timeslice_mode
                graph_key = (
                    getattr(graph, "source_uid", "")
                    or getattr(getattr(graph, "source_context", None), "stable_id", "")
                    or getattr(graph, "proxy_name", "")
                )
                if not graph_key:
                    graph_key = f"graph:{id(graph)}"

                source_task_ids = list(getattr(graph, "source_task_ids", []) or [])
                if not source_task_ids:
                    source_task_ids = [node_id for node_id, t in graph.tasks.items() if isinstance(t, SourceTask)]
                    graph.source_task_ids = source_task_ids
                source_tasks = [
                    graph.tasks.get(task_id)
                    for task_id in source_task_ids
                    if isinstance(graph.tasks.get(task_id), SourceTask)
                ]

                if source_tasks:
                    graph_source = getattr(graph, "source_context", None)
                    graph_source_uid = str(getattr(graph_source, "stable_id", "") or "")
                    size_task = next(
                        (
                            t
                            for t in source_tasks
                            if str(getattr(getattr(t, "source_context", None), "stable_id", "") or "") == graph_source_uid
                        ),
                        source_tasks[0],
                    )
                    source_ctx = graph_source or getattr(size_task, "source_context", None)
                    source_type = str(getattr(source_ctx, "source_type", "") or "").upper()
                    source_sr = int(
                        getattr(source_ctx, "sample_rate", 0)
                        or getattr(size_task, "sample_rate", 0)
                        or self.effective_sample_rate
                        or SAMPLE_RATE
                    )
                    proposed_block_size = self._resolve_block_size(
                        source_sr,
                        fps,
                        source_type=source_type,
                        timeslice_mode=timeslice_mode,
                        delta_frames=delta_frames,
                    )
                    new_block_size = self._stabilize_graph_block_size(
                        graph_key=graph_key,
                        current_size=int(getattr(graph, "block_size", 0) or 0),
                        proposed_size=int(proposed_block_size),
                        mode=timeslice_mode,
                    )
                    self.adjust_graph_block_size_for_task(size_task, new_block_size)

                # Legacy scheduler keeps source nodes hot even when disconnected.
                if scheduler_mode == "legacy":
                    for source_task in source_tasks:
                        try:
                            source_task.get_output(self.current_frame, 0)
                        except Exception:
                            continue

                proxy = graph.get_proxy_object()
                if not proxy:
                    source_proxy = getattr(getattr(graph, "source_context", None), "proxy_object_name", "")
                    if source_proxy and source_proxy != getattr(graph, "proxy_name", ""):
                        graph.proxy_name = source_proxy
                        proxy = graph.get_proxy_object()
                if not graph.output_attr_keys:
                    recovered: list[str] = []
                    if self.writer and self.abstract_tree_ref:
                        for node_id in graph.tasks.keys():
                            node = self.abstract_tree_ref.nodes.get(node_id)
                            if not node:
                                continue
                            is_output_node = (
                                node.category == 'stream'
                                or (node.type == 'src_specs' and node.category == 'utilities')
                                or (node.type == 'util_rolling_avg' and node.category == 'utilities')
                                or (node.type == 'stream_accumulator' and node.category == 'utilities')
                                or (node.category == 'utilities' and node.type == 'preview_sink')
                            )
                            if not is_output_node:
                                continue
                            for out in getattr(node, "outputs", []) or []:
                                if getattr(out, "data_type", "") in ("NodeSocketBundle", "BUNDLE", "BUNDLE_SOCKET"):
                                    continue
                                key = f"{node_id}:{int(getattr(out, 'socket_idx', 0))}"
                                if key in self.writer.output_defs:
                                    recovered.append(key)
                    if recovered:
                        graph.output_attr_keys = sorted(set(recovered))
                        graph.sink_bindings_by_attr = self._resolve_sink_bindings_for_graph(
                            graph,
                            self.abstract_tree_ref,
                            source_uid=str(getattr(graph, "source_uid", "") or ""),
                            proxy_name=str(getattr(graph, "proxy_name", "") or ""),
                            single_source_mode=bool(len(self.graphs) <= 1),
                        )
                        self._rebuild_graph_output_slot_specs(graph)
                        if self.abstract_tree_ref is not None:
                            self._prepare_parallel_graph_ir(graph, self.abstract_tree_ref)
                        try:
                            t0 = time.perf_counter()
                            touched = self.writer.ensure_sink_layout(graph, active_keys=set(graph.output_attr_keys))
                            self._perf_add("proxy_write", time.perf_counter() - t0)
                            if isinstance(touched, set):
                                geometry_proxy_names.update({str(v) for v in touched if str(v)})
                        except Exception as e:
                            self._log_runtime(f"attr_ensure_recover:{graph_key}", f"Attribute ensure after recovery failed for '{graph_key}': {e}", "ERROR", min_interval=2.0)
                        self._log_runtime(
                            f"recover_keys:{graph_key}",
                            f"Recovered {len(graph.output_attr_keys)} output keys for source '{getattr(graph, 'source_uid', '')}'.",
                            "WARNING",
                            min_interval=1.5,
                        )
                    else:
                        if graph.preview_task_ids:
                            continue
                        self._log_runtime(
                            f"missing_keys:{graph_key}",
                            f"No output keys for source '{getattr(graph, 'source_uid', '')}'. Skipping write this frame.",
                            "WARNING",
                            min_interval=2.0,
                        )
                        continue

                # Dirty-flag resync: sync immediately after compile, then fall back to
                # the interval only if the dirty set is somehow absent (defensive).
                schema_dirty = graph_key in self._attr_schema_dirty
                if not schema_dirty:
                    last_sync = self._last_attr_sync_frame.get(graph_key, -10_000_000)
                    schema_dirty = (self.current_frame - last_sync) >= self._ATTR_RESYNC_INTERVAL_FRAMES
                if schema_dirty:
                    try:
                        t0 = time.perf_counter()
                        touched = self.writer.ensure_sink_layout(graph, active_keys=set(graph.output_attr_keys))
                        self._perf_add("proxy_write", time.perf_counter() - t0)
                        if isinstance(touched, set):
                            geometry_proxy_names.update({str(v) for v in touched if str(v)})
                    except Exception as e:
                        self._log_runtime(f"attr_ensure:{graph_key}", f"Attribute ensure failed for '{graph_key}': {e}", "ERROR", min_interval=2.0)
                    self._attr_schema_dirty.discard(graph_key)
                    self._last_attr_sync_frame[graph_key] = int(self.current_frame)

                baked_results = self._get_baked_frame_results(graph, self.current_frame)
                if baked_results is not None:
                    try:
                        t0 = time.perf_counter()
                        touched = self.writer.commit_results(graph, baked_results, active_keys=set(graph.output_attr_keys))
                        self._perf_add("proxy_write", time.perf_counter() - t0)
                        if isinstance(touched, set):
                            geometry_proxy_names.update({str(v) for v in touched if str(v)})
                    except Exception as e:
                        self._log_runtime(f"attr_write_baked:{graph_key}", f"Baked attribute write failed for '{graph_key}': {e}", "ERROR", min_interval=2.0)
                    continue

                live_source_task_ids = list(getattr(graph, "live_source_task_ids", []) or [])
                if not live_source_task_ids:
                    live_source_task_ids = [
                        node_id
                        for node_id, t in graph.tasks.items()
                        if isinstance(t, SourceTask) and str(getattr(getattr(t, "source_context", None), "source_type", "") or "").upper() == "LIVE"
                    ]
                    graph.live_source_task_ids = live_source_task_ids
                if any(
                    not getattr(graph.tasks.get(task_id), "live_ready", True)
                    for task_id in live_source_task_ids
                    if isinstance(graph.tasks.get(task_id), SourceTask)
                ):
                    continue
                file_source_task_ids = list(getattr(graph, "file_source_task_ids", []) or [])
                if not file_source_task_ids:
                    file_source_task_ids = [
                        node_id
                        for node_id, t in graph.tasks.items()
                        if isinstance(t, SourceTask) and str(getattr(getattr(t, "source_context", None), "source_type", "") or "").upper() == "FILE"
                    ]
                    graph.file_source_task_ids = file_source_task_ids
                file_tasks = [
                    graph.tasks.get(task_id)
                    for task_id in file_source_task_ids
                    if isinstance(graph.tasks.get(task_id), SourceTask)
                ]
                if any(not getattr(t, "source_ready", True) for t in file_tasks):
                    msg = next((getattr(t, "last_error", "") for t in file_tasks if getattr(t, "last_error", "")), "File source unavailable")
                    self._log_runtime(f"source_unready:{graph_key}", f"Skipping source '{graph_key}': {msg}", "ERROR", min_interval=2.0)
                    continue

                parallel_graph_key = str(getattr(graph, "parallel_graph_uid", "") or graph_key)
                parallel_offload_graph = (
                    parallel_mode == "processes_dsp"
                    and bool(getattr(graph, "parallel_ir", {}))
                    and bool(getattr(graph, "parallel_offload_attr_keys", []))
                )
                parallel_submit_enabled = (
                    parallel_offload_graph
                    and parallel_pool_ready
                    and parallel_graph_key not in self._parallel_disabled_sources
                )

                if parallel_submit_enabled and source_tasks:
                    source_node_ids = list(
                        dict(getattr(graph, "parallel_ir", {}) or {}).get("source_node_ids", ())
                        or getattr(graph, "source_task_ids", ())
                    )
                    source_chunks_for_worker: Dict[str, Any] = {}
                    for src_node_id in source_node_ids:
                        source_task = graph.tasks.get(str(src_node_id))
                        if not isinstance(source_task, SourceTask):
                            continue
                        try:
                            src_chunk = source_task.get_output(self.current_frame, 0)
                        except Exception:
                            src_chunk = None
                        if hasattr(src_chunk, "mono") and hasattr(src_chunk, "sample_rate"):
                            source_chunks_for_worker[str(src_node_id)] = src_chunk
                    if source_chunks_for_worker:
                        shard_specs = list(getattr(graph, "parallel_output_shards", []) or [])
                        if not shard_specs:
                            shard_specs = self._parallel_shard_attr_keys(
                                list(getattr(graph, "parallel_offload_attr_keys", []) or [])
                            )
                            graph.parallel_output_shards = list(shard_specs)
                        if not shard_specs:
                            parallel_submit_enabled = False
                        else:
                            for shard_idx, shard_keys in enumerate(shard_specs):
                                if not shard_keys:
                                    continue
                                self._parallel_submit_graph_job(
                                    scene=scene,
                                    graph=graph,
                                    graph_key=parallel_graph_key,
                                    queue_key=f"{parallel_graph_key}:shard{int(shard_idx)}",
                                    owner_graph_key=parallel_graph_key,
                                    frame_index=int(self.current_frame),
                                    source_chunks=source_chunks_for_worker,
                                    output_keys=list(shard_keys),
                                    include_graph_ir=bool(parallel_graph_key in self._parallel_graph_ir_required),
                                )
                    else:
                        parallel_submit_enabled = False

                if parallel_offload_graph:
                    # Stage B: asynchronous numeric DSP worker path (fresh -> stale -> zero).
                    local_keys = list(getattr(graph, "parallel_local_attr_keys", []) or [])
                    has_parallel_values = self._parallel_has_usable_values_for_graph_frame(
                        parallel_graph_key,
                        int(self.current_frame),
                    )
                    if not has_parallel_values or not parallel_submit_enabled:
                        # Hard fallback: if worker values are not yet usable OR jobs
                        # cannot be submitted this frame (disabled source, pool
                        # unavailable), compute offloaded keys locally so stream
                        # outputs never stall at zero.
                        local_keys = list(
                            dict.fromkeys(
                                local_keys + list(getattr(graph, "parallel_offload_attr_keys", []) or [])
                            )
                        )
                    results = self._collect_graph_results_for_frame(graph, self.current_frame, attr_keys=local_keys)
                    if has_parallel_values and parallel_submit_enabled:
                        parallel_values = self._parallel_values_for_graph_frame(
                            graph=graph,
                            graph_key=parallel_graph_key,
                            frame_index=int(self.current_frame),
                        )
                        results.update(parallel_values)
                else:
                    results = self._collect_graph_results_for_frame(graph, self.current_frame)

                if preview_eval_due:
                    # Collect preview after the main graph/source evaluation so
                    # FILE-source prefetch has the full frame interval to fill.
                    self._start_preview_worker()  # idempotent
                    manual_ids = getattr(graph, "manual_preview_task_ids", []) or []
                    passthrough_specs = getattr(graph, "auto_preview_passthrough_specs", []) or []
                    auto_ids = getattr(graph, "auto_preview_task_ids", []) or []
                    preview_ids = manual_ids if manual_ids else (
                        auto_ids or (getattr(graph, "preview_task_ids", []) or [])
                    )
                    if (not manual_ids) and passthrough_specs:
                        # Direct passthrough: prefer the chunk already produced for
                        # this frame; only fall back to evaluation once the rest of
                        # the graph has had a chance to warm the source prefetcher.
                        for src_task_id, src_sock_idx in passthrough_specs:
                            src_task = graph.tasks.get(str(src_task_id))
                            if not isinstance(src_task, SourceTask):
                                continue
                            raw_chunk = self._task_current_chunk_for_frame(src_task, self.current_frame)
                            if raw_chunk is None:
                                try:
                                    t0 = time.perf_counter()
                                    raw_chunk = src_task.get_output(self.current_frame, int(src_sock_idx))
                                    self._perf_add("preview", time.perf_counter() - t0)
                                except Exception:
                                    raw_chunk = None
                            if raw_chunk is not None and self._chunk_is_unmasked_audio(raw_chunk):
                                preview_chunks.append(
                                    self._fit_preview_chunk_block(raw_chunk, preview_block_size)
                                )
                    else:
                        # PreviewSinkTask path: offload biquad DSP chains to the
                        # background thread and reuse already-evaluated upstream
                        # chunks whenever possible.
                        for task_id in preview_ids:
                            preview_task = graph.tasks.get(task_id)
                            if not isinstance(preview_task, PreviewSinkTask):
                                continue
                            is_enabled_fn = getattr(preview_task, "is_enabled", None)
                            if callable(is_enabled_fn) and not is_enabled_fn():
                                continue
                            conn = getattr(preview_task, "inputs", {}).get(0)
                            upstream_task = getattr(conn, "task", None) if conn is not None else None
                            if upstream_task is None:
                                continue
                            sock_idx = int(getattr(conn, "socket_index", 0) or 0)
                            upstream_chunk = self._task_current_chunk_for_frame(
                                upstream_task, self.current_frame
                            )
                            if upstream_chunk is None:
                                try:
                                    t0 = time.perf_counter()
                                    upstream_chunk = upstream_task.get_output(
                                        self.current_frame, sock_idx
                                    )
                                    self._perf_add("preview", time.perf_counter() - t0)
                                except Exception:
                                    upstream_chunk = None
                            if upstream_chunk is not None:
                                if self._submit_preview_work(preview_task, upstream_chunk):
                                    # Worker owns the push; mark preview as active so
                                    # the monitor is not cleared this frame.
                                    preview_pushed = True

                try:
                    # Stage C (main thread only): commit numeric results into resolved sinks.
                    t0 = time.perf_counter()
                    touched = self.writer.commit_results(graph, results, active_keys=set(graph.output_attr_keys))
                    self._perf_add("proxy_write", time.perf_counter() - t0)
                    if isinstance(touched, set):
                        geometry_proxy_names.update({str(v) for v in touched if str(v)})
                except Exception as e:
                    self._log_runtime(f"attr_write:{graph_key}", f"Attribute write failed for '{graph_key}': {e}", "ERROR", min_interval=2.0)

            if preview_allowed:
                t0 = time.perf_counter()
                if preview_eval_due:
                    mixed_preview = self._mix_preview_chunks(preview_chunks)
                    self._preview_last_chunk = mixed_preview
                    self._preview_last_eval_frame = int(self.current_frame)
                else:
                    mixed_preview = self._preview_last_chunk
                if mixed_preview is not None:
                    try:
                        self._preview_monitor.push_chunk(mixed_preview)
                        preview_pushed = True
                    except Exception:
                        pass
                self._perf_add("preview", time.perf_counter() - t0)
            else:
                self._preview_last_chunk = None
                self._preview_last_eval_frame = -1
                self._preview_next_eval_ts = 0.0

            if (not preview_allowed) or (not preview_pushed):
                try:
                    self._preview_monitor.clear()
                except Exception:
                    pass
        finally:
            if geometry_proxy_names:
                self._mark_geometry_eval_pending(scene, proxy_names=geometry_proxy_names)
            self._perf_finalize_frame()

    def _finalise_bake_arrays(self, src_uid: str, baked_values: Dict[str, List[Any]]) -> None:
        """Finalize completed bake payloads and commit to _bake_cache on main thread."""
        job = self._bake_jobs.get(str(src_uid))
        if not isinstance(job, dict):
            return
        source = job.get("source")
        graph = job.get("graph")
        scene = job.get("scene")
        target_tree = job.get("target_tree")
        filepath = str(job.get("filepath", "") or "")
        frame_start = int(job.get("frame_start", 0) or 0)
        frame_end = int(job.get("frame_end", frame_start) or frame_start)

        baked_arrays: Dict[str, np.ndarray] = {}
        for attr_key, series in dict(baked_values or {}).items():
            if not series:
                baked_arrays[attr_key] = np.zeros(0, dtype=np.float32)
                continue
            first = series[0]
            if isinstance(first, (tuple, list, np.ndarray)):
                arr = np.asarray(series, dtype=np.float32)
            else:
                arr = np.asarray(series, dtype=np.float32).reshape(-1)
            baked_arrays[attr_key] = arr.astype(np.float32, copy=False)

        total_bytes = int(sum(int(arr.nbytes) for arr in baked_arrays.values()))
        npz_path = ""
        npz_keys: Dict[str, str] = {}
        values_store: Dict[str, Any] = dict(baked_arrays)
        # Spill very large bake payloads to compressed disk cache to cap memory usage.
        if total_bytes >= (16 * 1024 * 1024) and baked_arrays:
            try:
                cache_dir = os.path.join(tempfile.gettempdir(), "ace_bake_cache")
                os.makedirs(cache_dir, exist_ok=True)
                sig_seed = json.dumps(
                    {
                        "source_uid": str(src_uid or ""),
                        "frame_start": int(frame_start),
                        "frame_end": int(frame_end),
                        "tree_signature": self._last_tree_signature,
                    },
                    sort_keys=True,
                    default=str,
                    separators=(",", ":"),
                ).encode("utf-8")
                sig = int.from_bytes(hashlib.blake2b(sig_seed, digest_size=4).digest(), byteorder="big")
                npz_path = os.path.join(cache_dir, f"ace_bake_{sig:08x}.npz")
                payload: Dict[str, np.ndarray] = {}
                for idx, (attr_key, arr) in enumerate(baked_arrays.items()):
                    npz_key = f"a{idx}"
                    payload[npz_key] = arr
                    npz_keys[attr_key] = npz_key
                np.savez_compressed(npz_path, **payload)
                values_store = {}
            except Exception as e:
                npz_path = ""
                npz_keys = {}
                values_store = dict(baked_arrays)
                self._log_runtime("bake_spill", f"Could not spill bake cache to disk: {e}", "WARNING", min_interval=0.0)

        self._bake_cache[str(src_uid)] = {
            "tree_signature": self._last_tree_signature,
            "block_size": int(getattr(graph, "block_size", 0)),
            "frame_start": frame_start,
            "frame_end": frame_end,
            "values": values_store,
            "npz_path": npz_path,
            "npz_keys": npz_keys,
            "bake_bytes": total_bytes,
            "filepath": filepath,
            "mtime": self._safe_file_mtime(filepath),
        }

        try:
            source["ace_bake_status"] = "done"
            source["ace_bake_error"] = ""
            source["ace_bake_progress"] = 1.0
            source["ace_bake_frame_start"] = frame_start
            source["ace_bake_frame_end"] = frame_end
            source["ace_bake_samples"] = max(frame_end - frame_start + 1, 0)
        except Exception:
            pass

        # Reset runtime tasks after offline bake so live playback state remains stable.
        try:
            if scene is not None:
                self.compile_scene_frontends(scene, preferred_tree=target_tree)
        except Exception as e:
            self._log_runtime("bake_recompile", f"Post-bake recompile failed: {e}", "ERROR", min_interval=0.0)
        finally:
            self._bake_jobs.pop(str(src_uid), None)

    def flush_bake_results(self) -> None:
        """Drain completed async bake events and write results on the main thread."""
        q = getattr(self, "_bake_result_queue", None)
        if q is None:
            return
        try:
            while True:
                event = q.get_nowait()
                if not isinstance(event, tuple) or len(event) < 3:
                    continue
                kind = str(event[0] or "")
                src_uid = str(event[1] or "")
                payload = event[2]
                if not src_uid:
                    continue
                job = self._bake_jobs.get(src_uid)
                source = job.get("source") if isinstance(job, dict) else None

                if kind == "progress":
                    if source is not None:
                        try:
                            source["ace_bake_progress"] = float(max(0.0, min(float(payload), 1.0)))
                        except Exception:
                            pass
                    continue

                if kind == "error":
                    if source is not None:
                        try:
                            source["ace_bake_status"] = "error"
                            source["ace_bake_error"] = str(payload or "Bake failed.")
                        except Exception:
                            pass
                    self._bake_jobs.pop(src_uid, None)
                    continue

                if kind == "done" and isinstance(payload, dict):
                    self._finalise_bake_arrays(src_uid, payload)
        except queue.Empty:
            pass
        except Exception:
            pass

    def bake_source_to_cache(self, source, tree=None, scene=None):
        """Precompute stream/spec outputs for a file source and cache them in memory."""
        if not source or getattr(source, "source_type", "") != "FILE":
            return False
        if scene is None:
            scene = getattr(getattr(bpy, "context", None), "scene", None)
        if scene is None:
            return False
        filepath = getattr(source, "filepath", "") or ""
        if not filepath or not os.path.isfile(filepath):
            try:
                source["ace_bake_status"] = "error"
                source["ace_bake_error"] = "Missing source file."
            except Exception:
                pass
            return False

        target_tree = tree if tree and self._tree_has_ace_nodes(tree) else None
        try:
            self.compile_scene_frontends(scene, preferred_tree=target_tree)
        except Exception as e:
            self._log_runtime("bake_compile", f"Bake compile failed: {e}", "ERROR", min_interval=0.0)
            return False

        ace = getattr(scene, "audiocore", None)
        existing_ids = [getattr(s, "stable_id", "") for s in getattr(ace, "audio_sources", []) if getattr(s, "stable_id", "")]
        src_uid = ACEIdManager.ensure_source_uid(source, existing_ids=existing_ids)
        if str(src_uid) in self._bake_jobs:
            return True
        graph = self.graphs.get(src_uid)
        if graph is None:
            src_proxy = getattr(source, "proxy_object_name", "")
            graph = next((g for g in self.graphs.values() if getattr(g, "proxy_name", "") == src_proxy), None)
        if not self.writer:
            return False
        if graph is None:
            try:
                source["ace_bake_status"] = "error"
                source["ace_bake_error"] = "Source graph unavailable."
            except Exception:
                pass
            return False
        if not graph.output_attr_keys:
            try:
                source["ace_bake_status"] = "error"
                source["ace_bake_error"] = "No stream/spec outputs to bake."
            except Exception:
                pass
            return False

        frame_start = int(getattr(scene, "frame_start", 1))
        frame_end = int(getattr(scene, "frame_end", frame_start))
        if frame_end < frame_start:
            frame_start, frame_end = frame_end, frame_start
        total_frames = max(frame_end - frame_start + 1, 1)
        output_attr_keys = list(graph.output_attr_keys)

        try:
            source["ace_bake_status"] = "baking"
            source["ace_bake_error"] = ""
            source["ace_bake_progress"] = 0.0
        except Exception:
            pass

        prev_entry = self._bake_cache.get(src_uid)
        if isinstance(prev_entry, dict):
            self._cleanup_bake_entry(prev_entry)

        self._bake_jobs[str(src_uid)] = {
            "source": source,
            "graph": graph,
            "scene": scene,
            "target_tree": target_tree,
            "filepath": filepath,
            "frame_start": int(frame_start),
            "frame_end": int(frame_end),
        }

        engine_ref = self

        def _bake_worker():
            baked_values: Dict[str, List[Any]] = {k: [] for k in output_attr_keys}
            try:
                for i, frame_idx in enumerate(range(frame_start, frame_end + 1)):
                    frame_results = engine_ref._collect_graph_results_for_frame(graph, frame_idx)
                    for attr_key in output_attr_keys:
                        baked_values[attr_key].append(
                            frame_results.get(attr_key, engine_ref._zero_value_for_attr(attr_key))
                        )
                    try:
                        engine_ref._bake_result_queue.put_nowait(
                            ("progress", str(src_uid), float(i + 1) / float(total_frames))
                        )
                    except Exception:
                        pass
            except Exception as exc:
                try:
                    engine_ref._bake_result_queue.put_nowait(("error", str(src_uid), str(exc)))
                except Exception:
                    pass
                return

            try:
                engine_ref._bake_result_queue.put_nowait(("done", str(src_uid), baked_values))
            except Exception:
                pass

        t = threading.Thread(target=_bake_worker, name=f"ACE_BakeWorker_{src_uid}", daemon=True)
        t.start()
        return True


# ==============================================================================
# BLENDER INTEGRATION (HANDLERS)
# ==============================================================================

@persistent
def _load_post_reset_engine(_dummy):
    """bpy.app.handlers.load_post: reset all engine instances after a file load."""
    ACEAudioEngine.shutdown_all_instances()
    _refresh_loaded_node_groups()
    ACEAudioEngine.ensure_instance(getattr(getattr(bpy, "context", None), "scene", None))


def _refresh_loaded_node_groups():
    """Bring the ACE node groups of a newly loaded file up to the current definitions.

    The factories skip their scan while the node registry hash is unchanged,
    but that hash only describes the groups of the data that was loaded when
    it was taken.
    """
    from ..nodes.dummy_tree import ACEDummyNodeFactory
    from ..nodes.shader_tree import ACEShaderNodeFactory

    for factory in (ACEDummyNodeFactory, ACEShaderNodeFactory):
        factory._NODES_CONTENT_HASH = ""
        try:
            factory.ensure_node_groups_exist()
        except Exception as exc:  # noqa: BLE001 boundary: a load must not fail on group refresh
            utils.log(f"ACE node group refresh after load failed ({factory.__name__}): {exc}", "ERROR")


@persistent
def _undo_redo_invalidate_engine_caches(_dummy):
    """bpy.app.handlers.undo_post / redo_post: clear caches and request recompile.

    Drops in-flight parallel futures so SHM handles cannot write into attributes
    that may have been removed or remapped by the undo/redo operation.
    """
    for engine in ACEAudioEngine.iter_instances():
        try:
            engine.begin_undo_recovery(getattr(getattr(bpy, "context", None), "scene", None))
        except Exception:
            pass


@persistent
def frame_change_handler(scene):
    """bpy.app.handlers.frame_change_pre: run one DSP frame for the current scene.

    Bootstraps a compile on the first call if no graphs are loaded yet (with
    exponential back-off on repeated failures).  Skips evaluation when a render
    job is running without ``Lock Interface`` to avoid unsafe concurrent bpy
    attribute writes.
    """
    if scene is None:
        scene = getattr(getattr(bpy, "context", None), "scene", None)
    if scene is None:
        return
    engine = ACEAudioEngine.ensure_instance(scene)
    if not engine:
        return
    if engine._undo_recovery_active():
        return
    is_rendering = engine._is_rendering_job()
    if engine._unsafe_for_handler_writes(scene):
        engine._log_runtime(
            "render_lock_interface",
            "Skipping ACE frame writes during render. Enable Render > Lock Interface for safe handler updates.",
            "WARNING",
            min_interval=5.0,
        )
        return
    if not engine.graphs:
        now = time.time()
        if now < getattr(engine, "_bootstrap_retry_at", 0.0):
            engine.execute_next_frame(scene, force=is_rendering)
            return
        try:
            frontends = engine.discover_scene_frontends(scene)
            if frontends:
                engine.compile_scene_frontends(scene)
                engine._bootstrap_fail_count = 0
                engine._bootstrap_retry_at = now + 0.25
            else:
                engine._bootstrap_retry_at = now + 0.25
        except Exception as e:
            fail_count = int(getattr(engine, "_bootstrap_fail_count", 0)) + 1
            engine._bootstrap_fail_count = fail_count
            engine._bootstrap_retry_at = now + min(2.0, 0.2 * (2 ** min(fail_count, 4)))
            engine._log_runtime("bootstrap_compile", f"Frame bootstrap compile failed: {e}", "ERROR", min_interval=2.0)
    engine.execute_next_frame(scene, force=is_rendering)


@persistent
def frame_change_post_handler(scene):
    """bpy.app.handlers.frame_change_post: secondary frame evaluation during rendering.

    Only fires ``frame_change_handler`` when an active render job is in progress,
    ensuring that every rendered frame receives up-to-date DSP values even when
    ``frame_change_pre`` alone is insufficient (e.g. Cycles tile rendering).
    """
    if scene is None:
        scene = getattr(getattr(bpy, "context", None), "scene", None)
    if scene is None:
        return
    engine = ACEAudioEngine.get_instance(scene)
    if engine is not None and engine._undo_recovery_active():
        return
    is_rendering = ACEAudioEngine._is_rendering_job()
    if not is_rendering:
        return
    frame_change_handler(scene)


@persistent
def render_pre_handler(scene):
    """bpy.app.handlers.render_pre: compile and warm the DSP graph before rendering.

    Compiles the ACE tree when no graphs are loaded, forces one evaluation on
    frame 1 to pre-fill caches, then marks the engine as being in a render job
    via ``_mark_render_start()``.
    """
    if scene is None:
        scene = getattr(getattr(bpy, "context", None), "scene", None)
    if scene is None:
        return
    engine = ACEAudioEngine.ensure_instance(scene)
    if engine is None:
        return
    if engine._unsafe_for_handler_writes(scene):
        engine._log_runtime(
            "render_lock_interface_pre",
            "Skipping ACE render bootstrap. Enable Render > Lock Interface for safe handler updates.",
            "WARNING",
            min_interval=5.0,
        )
        engine._mark_render_start(scene)
        return
    if not engine.graphs:
        try:
            if engine.discover_scene_frontends(scene):
                engine.compile_scene_frontends(scene)
        except Exception as e:
            engine._log_runtime("render_pre_compile", f"Render pre compile failed: {e}", "ERROR", min_interval=1.5)
    try:
        engine.execute_next_frame(scene, force=True)
    except Exception as e:
        engine._log_runtime("render_pre_execute", f"Render pre execute failed: {e}", "ERROR", min_interval=1.5)
    engine._mark_render_start(scene)


@persistent
def render_post_handler(scene):
    """bpy.app.handlers.render_post: finalise engine state after a successful render.

    Clears the render-in-progress flag via ``_mark_render_complete()`` and
    silences the preview monitor so audio does not start playing back after the
    render completes.
    """
    engine = ACEAudioEngine.get_instance(scene)
    if engine is None:
        return
    engine._mark_render_complete(scene, cancelled=False)
    try:
        ACEAudioEngine._preview_monitor.clear()
    except Exception:
        pass


@persistent
def render_cancel_handler(scene):
    """bpy.app.handlers.render_cancel: clean up engine state after a cancelled render.

    Marks the render job as cancelled (``_mark_render_complete(cancelled=True)``)
    then delegates to ``render_post_handler`` to clear the preview monitor and
    do common post-render teardown.
    """
    engine = ACEAudioEngine.get_instance(scene)
    if engine is not None:
        engine._mark_render_complete(scene, cancelled=True)
    render_post_handler(scene)


def _original_id(update):
    """Return the original datablock of a depsgraph update.

    ``update.id`` is the evaluated copy, whose pointer differs from the one
    dirty tracking and frontend discovery use.
    """
    update_id = getattr(update, "id", None)
    return getattr(update_id, "original", None) or update_id


def _id_type_name(id_block) -> str:
    """Return the RNA type name ("Object", "Material", "GeometryNodeTree", ...).

    ID types such as Object and Material have no ``bl_idname``; only
    registrable types (node trees, nodes, operators) do.
    """
    identifier = getattr(getattr(id_block, "bl_rna", None), "identifier", "")
    return str(identifier or getattr(id_block, "bl_idname", "") or "")


@persistent
def depsgraph_update_handler(scene, depsgraph):
    """bpy.app.handlers.depsgraph_update_post: react to node-tree edits in real time.

    Consumes any pending geometry-eval updates (proxy attribute schema changes).
    Pre-warms file metadata for FILE sources whose sample-rate or duration are
    not yet known, eliminating the cold-start sample-rate race on the first compile.
    When ``auto_compile`` is enabled, detects edits to ACE geometry node trees and
    calls ``request_recompile()`` so changes are picked up within the next timer
    interval.
    """
    engine = ACEAudioEngine.get_instance(scene)
    ace = getattr(scene, "audiocore", None)
    engine_settings = getattr(ace, "engine_settings", None)
    auto_compile_enabled = bool(ace and engine_settings and engine_settings.auto_compile)

    def _update_touches_ace_tree(update_id) -> bool:
        bl_idname = _id_type_name(update_id)
        if bl_idname in {"GeometryNodeTree", "ShaderNodeTree"}:
            return bool(ACEAudioEngine._tree_is_ace_managed(update_id))
        if bl_idname == "Material":
            tree = getattr(update_id, "node_tree", None) if update_id is not None else None
            return bool(ACEAudioEngine._tree_is_ace_managed(tree))
        if bl_idname == "Object":
            obj = update_id
            for mod in getattr(obj, "modifiers", []) or []:
                tree_ref = getattr(mod, "node_group", None)
                if getattr(mod, "type", None) == "NODES" and ACEAudioEngine._tree_is_ace_managed(tree_ref):
                    return True
            for slot in getattr(obj, "material_slots", []) or []:
                material = getattr(slot, "material", None)
                tree = getattr(material, "node_tree", None) if material is not None else None
                if ACEAudioEngine._tree_is_ace_managed(tree):
                    return True
        return False

    if engine is not None and engine._undo_recovery_active():
        if auto_compile_enabled:
            for update in getattr(depsgraph, "updates", []) or []:
                if _update_touches_ace_tree(_original_id(update)):
                    try:
                        engine.request_recompile(None)
                    except Exception:
                        pass
                    break
        return

    if engine is not None:
        engine._consume_geometry_eval_updates(scene, depsgraph)
        # Early-probe: pre-warm file metadata for any FILE source missing sr/duration so
        # that _flush_probe_updates() can populate native_sample_rate before the next
        # compile cycle reads it — eliminating the cold-start sample-rate race.
        _ace_data = ace
        for _src in getattr(_ace_data, "audio_sources", []):
            if getattr(_src, "source_type", "") == "FILE":
                _fp = getattr(_src, "filepath", "")
                if _fp and (
                    int(getattr(_src, "native_sample_rate", 0) or 0) <= 0
                    or float(getattr(_src, "duration", 0.0) or 0.0) <= 0.0
                ):
                    engine._dispatch_async_probe(_fp)
    if not auto_compile_enabled:
        return

    engine = engine or ACEAudioEngine.ensure_instance(scene)
    if engine is None:
        return

    recompile_targets: list = []
    seen_targets: set[int] = set()

    def _queue_recompile_target(tree_obj) -> None:
        if tree_obj is None:
            return
        try:
            key = int(tree_obj.as_pointer())
        except Exception:
            key = id(tree_obj)
        if key in seen_targets:
            return
        seen_targets.add(key)
        recompile_targets.append(tree_obj)

    compiled_frontend_snapshots = dict(getattr(engine, "_compiled_frontend_snapshots", {}) or {})
    discovered_frontends_by_uid = None

    for update in getattr(depsgraph, "updates", []):
        _id = _original_id(update)
        bl_idname = _id_type_name(_id)

        if bl_idname in {"GeometryNodeTree", "ShaderNodeTree"}:
            if ACEAudioEngine._tree_is_ace_managed(_id):
                _queue_recompile_target(_id)
            continue

        if bl_idname == "Material":
            tree = getattr(_id, "node_tree", None) if _id is not None else None
            if ACEAudioEngine._tree_is_ace_managed(tree):
                _queue_recompile_target(tree)
            continue

        if bl_idname == "Object":
            obj = _id
            has_update_flags = any(
                hasattr(update, token)
                for token in ("is_updated_geometry", "is_updated_shading", "is_updated_transform")
            )
            if has_update_flags and not (
                bool(getattr(update, "is_updated_geometry", False))
                or bool(getattr(update, "is_updated_shading", False))
            ):
                continue

            obj_name = str(getattr(obj, "name", "") or "")
            if obj_name and compiled_frontend_snapshots:
                if discovered_frontends_by_uid is None:
                    discovered_frontends_by_uid = {
                        str(rec.frontend_ref.uid or ""): getattr(rec, "tree", None)
                        for rec in (engine.discover_scene_frontends(scene) or [])
                    }
                for frontend_uid, frontend_snapshot in compiled_frontend_snapshots.items():
                    owners = {
                        str(v)
                        for v in tuple(getattr(frontend_snapshot, "owner_keys", ()) or ())
                        if str(v)
                    }
                    if obj_name not in owners:
                        continue
                    tree_ref = discovered_frontends_by_uid.get(str(frontend_uid))
                    if tree_ref is not None:
                        _queue_recompile_target(tree_ref)

            for mod in getattr(obj, "modifiers", []) or []:
                tree_ref = getattr(mod, "node_group", None)
                if getattr(mod, "type", None) == "NODES" and ACEAudioEngine._tree_is_ace_managed(tree_ref):
                    _queue_recompile_target(tree_ref)

            for slot in getattr(obj, "material_slots", []) or []:
                material = getattr(slot, "material", None)
                tree = getattr(material, "node_tree", None) if material is not None else None
                if ACEAudioEngine._tree_is_ace_managed(tree):
                    _queue_recompile_target(tree)

    for target in recompile_targets:
        engine.request_recompile(target)


def auto_compile_timer():
    """bpy.app.timers callback (0.25 s): compile dirty trees when auto_compile is on.

    Runs ``_maybe_compile_dirty()`` and, if a recompile was triggered, forces
    one execution of ``execute_next_frame()`` so attribute values are refreshed
    immediately rather than waiting for the next frame-change event.

    Returns:
        0.25 — reschedule interval in seconds (always).
    """
    try:
        scene = getattr(getattr(bpy, "context", None), "scene", None)
        if scene is None:
            return 0.25
        ace = getattr(scene, "audiocore", None)
        settings = getattr(ace, "engine_settings", None)
        if ace and settings and settings.auto_compile:
            engine = ACEAudioEngine.ensure_instance(scene)
            if engine._undo_recovery_active():
                return 0.25
            if engine._unsafe_for_handler_writes(scene):
                return 0.25
            compiled = engine._maybe_compile_dirty(scene)
            if compiled:
                engine.execute_next_frame(scene, force=True)
    except Exception:
        pass
    return 0.25


def _shm_drain_timer():
    """
    Periodic SHM drain called by bpy.app.timers every 2 seconds.

    _parallel_poll_futures() is normally called inside execute_next_frame(), but
    when the scene is paused indefinitely (no frame change events), completed
    futures hold their SHM handles until the next frame event.  On platforms
    with a small /dev/shm limit (Linux default 64 MB) this can exhaust shared
    memory for multi-source sessions.

    During active playback the per-frame execute_next_frame() already drains
    futures every frame; firing the drain here too causes redundant IPC pipe
    reads (future_obj.result(timeout=0.0) on a ProcessPoolExecutor) which add
    measurable latency on Windows and can produce periodic stutters that match
    the 2 s timer interval.  Skip the drain while the animation is playing.
    """
    try:
        scene = getattr(getattr(bpy, "context", None), "scene", None)
        engine = ACEAudioEngine.get_instance(scene) if scene is not None else None
        if engine is not None and engine._parallel_future_meta:
            if engine._undo_recovery_active():
                return 2.0
            # Only drain when the scene is NOT actively playing — during playback
            # execute_next_frame() handles all future draining every frame.
            if not engine._is_animation_playing():
                engine._parallel_poll_futures(scene, current_frame=int(getattr(engine, "current_frame", 0)))
    except Exception as _exc:
        try:
            utils.log(
                f"[ACE-SHM] Drain timer error — completed futures may not be released "
                f"and shared memory may leak: {_exc!r}",
                "WARNING",
            )
        except Exception:
            pass
    return 2.0


def _ace_handler_registered(handler_list, func) -> bool:
    """
    Check if a handler with the same __qualname__ is already in handler_list.

    Blender's `in` operator uses object identity, which fails under
    'Reload Scripts' because the module is re-imported and each function
    gets a new memory address.  Checking by __qualname__ (a stable string
    tied to the function's definition site) survives reloads correctly.
    """
    target_name = getattr(func, "__qualname__", None)
    if target_name is None:
        return func in handler_list
    return any(getattr(h, "__qualname__", None) == target_name for h in handler_list)


def _ace_handler_remove_all(handler_list, func):
    """
    Remove all entries in handler_list whose __qualname__ matches func.

    On 'Reload Scripts' the old function object is still registered (different
    identity, same qualname), so a plain `remove(func)` silently fails and
    leaves a ghost handler.  This helper removes every matching entry.
    """
    target_name = getattr(func, "__qualname__", None)
    if target_name is None:
        to_remove = [h for h in handler_list if h is func]
    else:
        to_remove = [h for h in handler_list if getattr(h, "__qualname__", None) == target_name]
    for h in to_remove:
        try:
            handler_list.remove(h)
        except Exception:
            pass


def register_handlers():
    """Register all ACE ``bpy.app.handlers`` and ``bpy.app.timers``.

    Called from ``__init__.register()``.  Uses ``_ace_handler_registered()``
    guards so that repeated Reload-Scripts calls do not accumulate duplicate
    handler entries.  Also triggers Numba JIT warm-up and creates the initial
    engine instance for the current scene.
    """
    # Trigger Numba JIT compilation at addon load time so the first preview frame
    # does not pay the cold-start compilation penalty (~50–200 ms).
    if _NUMBA_AVAILABLE:
        try:
            _dummy = np.zeros((4, 2), dtype=np.float32)
            _preview_limiter_block_numba(_dummy, 0.891, 0.0001, 1.0)
        except Exception:
            pass

    if not _ace_handler_registered(bpy.app.handlers.load_post, _load_post_reset_engine):
        bpy.app.handlers.load_post.append(_load_post_reset_engine)
    if not _ace_handler_registered(bpy.app.handlers.undo_post, _undo_redo_invalidate_engine_caches):
        bpy.app.handlers.undo_post.append(_undo_redo_invalidate_engine_caches)
    if not _ace_handler_registered(bpy.app.handlers.redo_post, _undo_redo_invalidate_engine_caches):
        bpy.app.handlers.redo_post.append(_undo_redo_invalidate_engine_caches)
    if not _ace_handler_registered(bpy.app.handlers.frame_change_pre, frame_change_handler):
        bpy.app.handlers.frame_change_pre.append(frame_change_handler)
    if not _ace_handler_registered(bpy.app.handlers.frame_change_post, frame_change_post_handler):
        bpy.app.handlers.frame_change_post.append(frame_change_post_handler)
    if not _ace_handler_registered(bpy.app.handlers.render_pre, render_pre_handler):
        bpy.app.handlers.render_pre.append(render_pre_handler)
    if not _ace_handler_registered(bpy.app.handlers.render_post, render_post_handler):
        bpy.app.handlers.render_post.append(render_post_handler)
    if not _ace_handler_registered(bpy.app.handlers.render_cancel, render_cancel_handler):
        bpy.app.handlers.render_cancel.append(render_cancel_handler)
    if not _ace_handler_registered(bpy.app.handlers.depsgraph_update_post, depsgraph_update_handler):
        bpy.app.handlers.depsgraph_update_post.append(depsgraph_update_handler)
    if not bpy.app.timers.is_registered(auto_compile_timer):
        bpy.app.timers.register(auto_compile_timer, first_interval=0.25, persistent=True)
    if not bpy.app.timers.is_registered(_shm_drain_timer):
        bpy.app.timers.register(_shm_drain_timer, first_interval=2.0, persistent=True)
    ACEAudioEngine.ensure_instance(getattr(getattr(bpy, "context", None), "scene", None))


def unregister_handlers():
    """Unregister all ACE ``bpy.app.handlers`` and ``bpy.app.timers``.

    Called from ``__init__.unregister()``.  Uses ``_ace_handler_remove_all()``
    so that ghost entries left over from a previous Reload-Scripts cycle are also
    removed.  Shuts down all engine instances to release parallel workers and
    shared memory before the addon is unloaded.
    """
    _ace_handler_remove_all(bpy.app.handlers.load_post, _load_post_reset_engine)
    _ace_handler_remove_all(bpy.app.handlers.undo_post, _undo_redo_invalidate_engine_caches)
    _ace_handler_remove_all(bpy.app.handlers.redo_post, _undo_redo_invalidate_engine_caches)
    _ace_handler_remove_all(bpy.app.handlers.frame_change_pre, frame_change_handler)
    _ace_handler_remove_all(bpy.app.handlers.frame_change_post, frame_change_post_handler)
    _ace_handler_remove_all(bpy.app.handlers.render_pre, render_pre_handler)
    _ace_handler_remove_all(bpy.app.handlers.render_post, render_post_handler)
    _ace_handler_remove_all(bpy.app.handlers.render_cancel, render_cancel_handler)
    _ace_handler_remove_all(bpy.app.handlers.depsgraph_update_post, depsgraph_update_handler)
    if bpy.app.timers.is_registered(auto_compile_timer):
        bpy.app.timers.unregister(auto_compile_timer)
    if bpy.app.timers.is_registered(_shm_drain_timer):
        bpy.app.timers.unregister(_shm_drain_timer)

    ACEAudioEngine.shutdown_all_instances()
