"""Paced feeder thread between preview chunks and the output stream."""

import numpy as np
import threading
from collections import deque
from typing import Optional
from .... import utils


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
