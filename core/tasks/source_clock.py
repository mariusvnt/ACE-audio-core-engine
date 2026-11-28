"""Sample clocks that keep file and live reads aligned with the timeline."""

import time
from typing import Any, Optional, Tuple
import numpy as np
from ... import utils
from .block_size import _resolve_block_size_default
from .chunk import AudioChunk, HISTORY_SAMPLES, MAX_BLOCK_SIZE, MIN_BLOCK_SIZE


class SourceClock:
    @staticmethod
    def _coerce_stereo_block(stereo: Any, frame_count: int) -> Optional[np.ndarray]:
        """A (2, n) float32 view of a stereo block in either layout, or None."""
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

    def _reset_sample_clocks(self):
        self.read_cursor = 0
        self._last_seek_sample = -1
        self._last_frame_request = None
        self._live_clock_anchor_frame = None
        self._live_clock_anchor_sample = None
        self._live_clock_last_target_sample = None
        self._live_block_buffer.clear()
        self._live_history_block = None
        self._live_history_tail = None
        self._live_history_before = None

    def _live_history(self, mono: np.ndarray, stereo: Optional[np.ndarray]):
        """(mono, stereo) heard before this live block, from the blocks played before it."""
        # The same samples again mean no new audio arrived: their history is unchanged.
        if self._live_history_block is None or not np.array_equal(mono, self._live_history_block):
            tail = self._live_history_tail
            if tail is None or (tail[1] is None) != (stereo is None) or (stereo is not None and tail[1].shape[0] != stereo.shape[0]):
                tail = (
                    np.zeros(HISTORY_SAMPLES, dtype=np.float32),
                    None if stereo is None else np.zeros((stereo.shape[0], HISTORY_SAMPLES), dtype=np.float32),
                )
            self._live_history_block = mono
            self._live_history_before = tail
            self._live_history_tail = (
                np.concatenate((tail[0], np.asarray(mono, dtype=np.float32).reshape(-1)))[-HISTORY_SAMPLES:],
                None if stereo is None else np.concatenate((tail[1], stereo), axis=1)[:, -HISTORY_SAMPLES:],
            )
        return self._live_history_before

    def _requested_frame(self) -> int:
        try:
            return int(getattr(self, "_current_frame_request", 0) or 0)
        except Exception as exc:
            utils.note_swallowed("source_clock.SourceClock._requested_frame", exc)
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
            utils.note_swallowed("source_clock.SourceClock._sync_reader_cursor", exc)
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
        """Main-thread entry point for reading the next audio block."""
        view = self._context_view()
        frame_request = getattr(self, "_current_frame_request", None)

        if self._prefetch_enabled and view.source_type == "FILE" and frame_request is not None and not getattr(self, "replaying", False):
            chunk = None
            if (
                self._prefetch_ready.is_set()
                and self._prefetch_for_frame == frame_request
                and view.signature == self._last_config_signature
            ):
                # Cache hit: take pre-read chunk, clear slot, trigger next read.
                self._prefetch_ready.clear()
                chunk = self._prefetch_chunk
                self._prefetch_chunk = None
            if chunk is not None:
                self._prefetch_view = view
                self._prefetch_target_frame = frame_request + 1
                self._prefetch_trigger.set()
                return chunk
            # Cache miss (scrub, first frame, a changed source, a read the thread left to the
            # main thread): stop prefetch, do sync read, then restart prefetch for the next frame.
            self._stop_prefetch_thread()
            chunk = self._read_next_block_sync(view=view)
            if chunk is not None:
                self._prefetch_target_frame = (
                    (frame_request + 1) if frame_request is not None else None
                )
                self._prefetch_view = self._context_view()
                self._start_prefetch_thread()
                self._prefetch_trigger.set()
            return chunk

        return self._read_next_block_sync(view=view)

    def _read_next_block_sync(
        self, track_perf_stage: bool = True, frame_request: Optional[int] = None, view: Optional[Any] = None,
        prefetch: bool = False,
    ) -> Optional["AudioChunk"]:
        """Synchronous block read for *frame_request* (default: the frame being evaluated), one read at a time.

        *view* is what the read needs from the source and the settings (``_context_view``).
        Without one, the read takes it itself: on the main thread, or on a thread whose
        source is a snapshot (the bake). The *prefetch* thread's view comes from the main thread,
        and that thread never opens the source: opening reads and writes Blender data.
        """
        with self._read_lock:
            return self._read_block_locked(track_perf_stage, frame_request, view, prefetch)

    def _read_block_locked(
        self, track_perf_stage: bool, requested_frame: Optional[int], view: Optional[Any] = None, prefetch: bool = False,
    ) -> Optional["AudioChunk"]:
        t0 = time.perf_counter()
        try:
            if view is None:
                view = self._context_view()
            current_signature = view.signature
            if prefetch and (
                not self.initialized
                or (self._last_config_signature is not None and current_signature != self._last_config_signature)
            ):
                return None  # opening the source reads and writes Blender data: the main thread does it
            if self.initialized and self._last_config_signature is not None and current_signature != self._last_config_signature:
                # Source IO params changed at runtime: reopen cleanly so readers/analyzers follow.
                self._reopen_for_config_change()
            if not self.initialized:
                self._load_source()
            if self._last_config_signature is None:
                self._last_config_signature = current_signature

            source_type = view.source_type
            if source_type != "LIVE" and not self.source_ready:
                return None

            # Arm prefetch thread on the first successful FILE source open.
            if source_type == "FILE" and not self._prefetch_enabled and getattr(self, "prefetch_allowed", True):
                self._start_prefetch_thread()

            resolve_fn = self.resolve_block_size_fn or _resolve_block_size_default
            try:
                delta_frames = float(getattr(getattr(self, "engine_ref", None), "_current_delta_frames", 1.0) or 1.0)
            except Exception as exc:
                utils.note_swallowed("source_clock.SourceClock._read_next_block_sync", exc)
                delta_frames = 1.0
            timeslice_mode = view.timeslice_mode
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
            history = None
            frame_request = self._requested_frame() if requested_frame is None else int(requested_frame)

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
                            history = self._live_history(mono, self._coerce_stereo_block(stereo, mono.shape[0]))
                            self.sample_rate = int(sr)
                            self.read_cursor = int(live_target)
                            self.live_ready = True
                        # Stay live_ready while the buffer is empty so process() keeps running until audio arrives.
                else:
                    # The block of a frame is fixed: every frame reads from its own start sample, so
                    # playing, seeking and baking read the same samples (ARCHITECTURE.md §3, Time).
                    desired_sample = self._file_sample_clock_target(frame_request)
                    self._sync_reader_cursor(desired_sample, force=True)
                    history_fn = getattr(self.reception, "history_before", None)
                    if callable(history_fn):
                        history = history_fn(HISTORY_SAMPLES)
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
                            utils.note_swallowed("source_clock.SourceClock._read_next_block_sync#2", exc)
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
            live_latency_ms = float(view.live_latency_ms) if source_type == "LIVE" else 0.0
            sr = max(float(self.sample_rate), 1.0)
            block_latency_ms = (float(self.block_size) / sr) * 1000.0
            total_latency_ms = float(max(0.0, live_latency_ms + block_latency_ms))
            return AudioChunk(
                mono_raw,
                stereo_raw,
                self.sample_rate,
                provider=self,
                lineage=(self.node_id,),
                latency_ms=total_latency_ms,
                history_mono=None if history is None else history[0],
                history_stereo=None if history is None else history[1],
            )
        finally:
            engine = getattr(self, "engine_ref", None)
            elapsed = time.perf_counter() - t0
            if bool(track_perf_stage):
                perf_add = getattr(getattr(engine, "diagnostics", None), "_perf_add", None)
                if callable(perf_add):
                    with utils.logged_errors("source_clock.SourceClock._read_next_block_sync#3"):
                        perf_add("source_read", elapsed)
            else:
                prefetch_add = getattr(getattr(engine, "diagnostics", None), "_record_source_prefetch", None)
                if callable(prefetch_add):
                    with utils.logged_errors("source_clock.SourceClock._read_next_block_sync#4"):
                        prefetch_add(elapsed)
