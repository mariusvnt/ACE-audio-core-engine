"""Audio preview routing: chunk collection, mixing and the monitor feeder."""

import math
import numpy as np
import threading
import time
from collections import deque
from typing import Any, Dict, Optional
from .. import utils
from .abstract_tree import ACEAbstractNode, ACEAbstractTree
from .tasks.block_size import resolve_preview_block_size
from .tasks.chunk import AudioChunk, SAMPLE_RATE
from .tasks.graph import ACEAudioGraph
from .tasks.preview import PreviewSinkTask
from .tasks.source import SourceTask
from .engine_common import _NUMBA_AVAILABLE, _preview_limiter_block_numba, engine_settings


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
                        with utils.logged_errors("audio_engine._ACEPreviewFeeder._loop"):
                            utils.log(
                                f"[ACE-Feeder] ring_append_fn failed (audio may drop): {_ring_exc!r}",
                                "WARNING",
                            )

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
        except Exception as exc:
            utils.note_swallowed("audio_engine._ACEPreviewAudioMonitor._chunk_frame_count", exc)
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
                with utils.logged_errors("audio_engine._ACEPreviewAudioMonitor._prepare_loop"):
                    import traceback as _tb
                    utils.log(
                        f"[ACE-Preview] _prepare_preview_payload error "
                        f"(#{self._preview_consecutive_errors}): {_exc!r}\n{_tb.format_exc()}",
                        "WARNING",
                    )
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
            with utils.logged_errors("audio_engine._ACEPreviewAudioMonitor._callback"):
                self._feeder.notify_ring_low()

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
            with utils.logged_errors("audio_engine._ACEPreviewAudioMonitor._apply_safety_limiter"):
                out, gain = _preview_limiter_block_numba(arr, ceiling, release_step, gain)
                np.clip(out, -ceiling, ceiling, out=out)
                self._limiter_gain = float(min(max(gain, 0.0), 1.0))
                return out.astype(np.float32, copy=False)

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
            with utils.logged_errors("audio_engine._ACEPreviewAudioMonitor._close_stream_only"):
                self._stream.stop()
            with utils.logged_errors("audio_engine._ACEPreviewAudioMonitor._close_stream_only#2"):
                self._stream.close()
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
            with utils.logged_errors("audio_engine._ACEPreviewAudioMonitor.close"):
                worker.join(timeout=0.5)
        self._close_stream_only()


class PreviewRouter:
    """Audio preview routing: chunk collection, mixing and the monitor feeder."""

    _preview_monitor = _ACEPreviewAudioMonitor()
    _PREVIEW_MIN_INTERVAL_S = 0.003

    def __init__(self, engine):
        self.engine = engine
        self._preview_worker_stop: bool = False
        self._preview_worker_cond: threading.Condition = threading.Condition()
        self._preview_worker_queue: list = []
        self._preview_worker_thread: Optional[threading.Thread] = None

    @classmethod
    def preview_monitor_available(cls) -> bool:
        """Return True if the sounddevice backend loaded successfully."""
        try:
            return bool(cls._preview_monitor._resolve_backend())
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine.preview_monitor_available", exc)
            return False

    @classmethod
    def preview_monitor_consecutive_errors(cls) -> int:
        """Return the current consecutive DSP error count from the prepare worker."""
        try:
            return int(cls._preview_monitor._preview_consecutive_errors)
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine.preview_monitor_consecutive_errors", exc)
            return 0

    @classmethod
    def _preview_block_size(cls, scene_or_engine_settings=None) -> int:
        settings = engine_settings(scene_or_engine_settings)
        raw = 256
        try:
            raw = int(getattr(settings, "ace_preview_block_size", 256) or 256)
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._preview_block_size", exc)
            raw = 256
        return int(resolve_preview_block_size(raw))

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
            except Exception as exc:
                utils.note_swallowed("audio_engine.ACEAudioEngine._chunk_is_unmasked_audio", exc)
                return False
        try:
            if bool(getattr(chunk, "analysis_ops", ())):
                return False
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._chunk_is_unmasked_audio#2", exc)
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
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._task_current_chunk_for_frame", exc)
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
                    self.engine.diagnostics._perf_add("preview", time.perf_counter() - t0)
                except Exception as exc:
                    utils.note_swallowed("audio_engine.ACEAudioEngine._collect_preview_chunk", exc)
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
                        self.engine.diagnostics._perf_add("preview", time.perf_counter() - t0)
                    except Exception as exc:
                        utils.note_swallowed("audio_engine.ACEAudioEngine._collect_preview_chunk#2", exc)
                        direct_chunk = None
                    if direct_chunk is not None and self._chunk_is_unmasked_audio(direct_chunk):
                        chunks.append(self._fit_preview_chunk_block(direct_chunk, preview_block_size))
                        continue
            try:
                t0 = time.perf_counter()
                chunk = task.get_output(frame_index, 0)
                self.engine.diagnostics._perf_add("preview", time.perf_counter() - t0)
            except Exception as exc:
                utils.note_swallowed("audio_engine.ACEAudioEngine._collect_preview_chunk#3", exc)
                continue
            if hasattr(chunk, "mono") and hasattr(chunk, "sample_rate"):
                chunks.append(self._fit_preview_chunk_block(chunk, preview_block_size))
        return PreviewRouter._mix_preview_chunks(chunks)

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
            with utils.logged_errors("audio_engine.ACEAudioEngine._preview_worker_loop"):
                result_chunk = task.process({0: upstream_chunk})
                if result_chunk is not None and hasattr(result_chunk, "mono"):
                    monitor.push_chunk(result_chunk)

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
