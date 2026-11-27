"""Audio monitor: mixes, limits and plays preview audio."""

import math
import numpy as np
import threading
import time
from collections import deque
from typing import Any, Dict, Optional
from .... import deps, utils
from ...tasks.chunk import SAMPLE_RATE
from ..common import _NUMBA_AVAILABLE, _preview_limiter_block_numba
from .feeder import _ACEPreviewFeeder


class _ACEPreviewAudioMonitor:
    """sounddevice-backed real-time audio monitor for branch preview output."""

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
        if deps.sounddevice_enabled():
            import sounddevice as sd  # type: ignore
            self._backend = sd
        else:
            self._backend = False
            if not self._unavailable_logged:
                utils.log("Preview monitor unavailable: SoundDevice is off or failed to load (see the add-on preferences).", "WARNING")
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
        """Inform the monitor of the current analysis block size."""
        self._analysis_block_size_hint = max(int(block_size), 0)

    @staticmethod
    def _chunk_frame_count(chunk) -> int:
        try:
            mono = np.asarray(getattr(chunk, "mono", None), dtype=np.float32).reshape(-1)
            return int(mono.shape[0] or 0)
        except Exception as exc:
            utils.note_swallowed("monitor._ACEPreviewAudioMonitor._chunk_frame_count", exc)
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
                with utils.logged_errors("monitor._ACEPreviewAudioMonitor._prepare_loop"):
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
            with utils.logged_errors("monitor._ACEPreviewAudioMonitor._callback"):
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
        """Protect preview output from clipping spikes."""
        arr = np.asarray(frames, dtype=np.float32)
        if arr.size == 0:
            return arr

        ceiling = float(max(self._limiter_ceiling_lin, 1e-6))
        sr = max(int(sample_rate), 1)
        release_tc = max(float(self._limiter_release_ms), 1.0) / 1000.0
        release_step = 1.0 - math.exp(-1.0 / (float(sr) * release_tc))

        gain = float(min(max(self._limiter_gain, 0.0), 1.0))
        if _NUMBA_AVAILABLE:
            with utils.logged_errors("monitor._ACEPreviewAudioMonitor._apply_safety_limiter"):
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
        # The running maximum of the release potential bounds the gain recovery in this block.
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
        with self._lock:
            self._append_ring_block_locked(block)

    def update_scene_fps(self, fps: float):
        """Update the playback fps used for feeder drift-correction."""
        self._scene_fps = max(float(fps), 1.0)
        self._feeder.update_fps(self._scene_fps)

    def _prepare_preview_payload(self, payload: Dict[str, Any]):
        """Apply the safety limiter to a queued payload and route it through the feeder."""
        chunk = payload.get("chunk")
        sample_rate = int(payload.get("sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
        frames = self._chunk_to_stereo_frames(chunk)
        if frames is None:
            return
        if not self._ensure_stream(sample_rate):
            return
        frames = self._apply_safety_limiter(frames, sample_rate)
        arr = np.asarray(frames, dtype=np.float32)
        # The feeder paces delivery at the device block rate; Blender timers jitter.
        with self._lock:
            self._preview_push_count += 1
            self._preview_push_frames += int(arr.shape[0] or 0)
            self._last_push_ts = time.perf_counter()
            latency_ms = float(getattr(chunk, "latency_ms", 0.0) or 0.0)
            if latency_ms > 0.0:
                self._preview_latency_ms.append(latency_ms)
        self._feeder.push_frames(arr)

    def push_chunk(self, chunk) -> bool:
        """Enqueue an AudioChunk for asynchronous prepare and playback."""
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
        """Return accumulated playback diagnostics as a flat float dict."""
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
        """Return a real-time latency snapshot for the N-panel live display."""
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
            # Report backlog above the steady-state headroom only.
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
        """Stop the feeder and hardware stream without touching the prepare thread."""
        # Stop the feeder before clearing the ring so it cannot push new frames
        # concurrently with the ring teardown.
        self._feeder.stop()
        with self._lock:
            self._ring.clear()
            self._ring_frames = 0
            self._last_callback_ts = 0.0
            self._last_push_ts = 0.0
        if self._stream is not None:
            with utils.logged_errors("monitor._ACEPreviewAudioMonitor._close_stream_only"):
                self._stream.stop()
            with utils.logged_errors("monitor._ACEPreviewAudioMonitor._close_stream_only#2"):
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
            with utils.logged_errors("monitor._ACEPreviewAudioMonitor.close"):
                worker.join(timeout=0.5)
        self._close_stream_only()
