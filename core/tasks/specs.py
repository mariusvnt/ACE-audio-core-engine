"""Source specs task: tempo, beat phase and long-term analysis."""

import os
import heapq
import threading
import time
from queue import Queue, Empty, Full
from typing import Dict, Any, Optional, Tuple
import numpy as np
from ..audio_influx import AudioReception
from ... import utils
from .analyzer import AudioFluxAnalyzer
from .base import ACEAudioTask
from .block_size import _resolve_block_size_default
from .chunk import AudioChunk, MAX_BLOCK_SIZE, MIN_BLOCK_SIZE, SAMPLE_RATE, _ensure_audio_chunk
from .source import _get_file_open_lock


class SourceSpecsTask(ACEAudioTask):
    """
    Outputs static metadata about the source with lightweight long-term analysis.
    Passes the incoming audio through unchanged on the last output.
    """
    _GLOBAL_LONG_TERM_CACHE: Dict[Tuple[str, float], Dict[str, Any]] = {}
    _GLOBAL_ACTIVE_ANALYSIS: Dict[Tuple[str, float], int] = {}
    _GLOBAL_CACHE_TTL = 120.0  # seconds
    _GLOBAL_CACHE_MAX = 128
    _LONG_TERM_MAX_SECONDS = 15.0
    _LONG_TERM_BLOCKS_PER_TICK = 3
    _LONG_TERM_TICK_INTERVAL_S = 0.05
    _GLOBAL_CACHE_LOCK = threading.Lock()
    _BG_QUEUE_MAX = 16
    _BG_TASK_QUEUE: Queue = Queue(maxsize=_BG_QUEUE_MAX)
    _BG_PENDING: set[Tuple[str, float]] = set()
    _BG_QUEUED: set[Tuple[str, float]] = set()
    _BG_THREAD_LOCK = threading.Lock()
    _BG_STOP_EVENT = threading.Event()
    _BG_THREAD: Optional[threading.Thread] = None
    # Pending ctx writes: (target_key, bpm, beats_per_bar). Drained on main thread.
    _PENDING_CTX_WRITES: "Queue[Tuple[Tuple[str, float], float, int]]" = Queue()

    def __init__(self, node_id, params, block_size):
        super().__init__(node_id, params, block_size)
        self._static_specs: Optional[Tuple[float, int, float, float, float, float]] = None
        self._long_term_cache: Dict[str, Any] = {}
        self._long_term_target: Optional[Tuple[str, float]] = None
        self._long_term_done: bool = False
        self._auto_bpm_key = "ace_bpm_auto"
        self._auto_meter_key = "ace_meter_auto"
        self._analysis_rec: Optional[AudioReception] = None
        self._analysis_analyzer = None
        self._analysis_block: int = 0
        self._analysis_sr: int = SAMPLE_RATE
        self._analysis_flux_vals: list[float] = []
        self._analysis_total_samples: int = 0
        self._analysis_frame_idx: int = 0
        self._analysis_next_tick_ts: float = 0.0

    def process(self, inputs):
        # Flush any detected-tempo results from the background worker.
        # Guarded to main-thread execution only so worker/bake threads never mutate bpy state.
        if threading.current_thread() is threading.main_thread():
            SourceSpecsTask.flush_pending_ctx_writes(
                getattr(getattr(self, "engine_ref", None), "_source_specs_task_registry", {})
            )
        # Preserve the audio path
        chunk_in = _ensure_audio_chunk(inputs.get(0), self.block_size)
        ctx = self.source_context
        if ctx:
            # If the user changed BPM or meter after an automatic estimate, honor
            # that override and stop auto-updates for this source.
            with utils.logged_errors("audio_tasks_registry.SourceSpecsTask.process"):
                auto_val = ctx.get(self._auto_bpm_key, None) if hasattr(ctx, "get") else None
                if auto_val is not None:
                    current = float(getattr(ctx, "bpm", 120.0))
                    if abs(current - float(auto_val)) > 1e-3:
                        ctx["ace_bpm_user_override"] = True
                        self._long_term_done = True
                        self._reset_long_term_analysis_state(clear_target=False)
            with utils.logged_errors("audio_tasks_registry.SourceSpecsTask.process#2"):
                auto_meter = ctx.get(self._auto_meter_key, None) if hasattr(ctx, "get") else None
                if auto_meter is not None:
                    current_meter = int(getattr(ctx, "beats_per_bar", 4) or 4)
                    if current_meter != int(auto_meter):
                        ctx["ace_meter_user_override"] = True
                        self._long_term_done = True
                        self._reset_long_term_analysis_state(clear_target=False)
        fps = getattr(self, "scene_fps", 24.0)
        self._maybe_run_long_term_analysis(ctx)

        bpm = float(self._long_term_cache.get("bpm", getattr(ctx, "bpm", 120.0) if ctx else 120.0) or 120.0)
        beats_per_bar = int(getattr(ctx, "beats_per_bar", 4) or 4) if ctx else 4
        sample_rate = float(getattr(ctx, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE) if ctx else float(SAMPLE_RATE)
        duration = float(getattr(ctx, "duration", 0.0) or 0.0) if ctx else 0.0

        fps_val = float(fps) if float(fps) > 0.0 else 24.0
        if (
            self._static_specs is None
            or self._static_specs[2] != sample_rate
            or self._static_specs[1] != beats_per_bar
            or self._static_specs[4] != bpm
            or self._static_specs[5] != fps_val
        ):
            beats_per_frame = float(bpm / max(60.0 * fps_val, 1e-6))
            self._static_specs = (beats_per_frame, beats_per_bar, sample_rate, duration, bpm, fps_val)
        else:
            beats_per_frame, beats_per_bar, sample_rate, duration, _, _ = self._static_specs

        frames_per_beat = float(1.0 / beats_per_frame) if beats_per_frame > 1e-6 else 0.0
        engine = getattr(self, "engine_ref", None)
        current_frame = getattr(engine, "current_frame", 0) if engine else 0
        measure_len_frames = float(beats_per_bar) / beats_per_frame if beats_per_frame > 0 else 0.0
        if measure_len_frames > 0.0:
            pos_in_measure = (float(current_frame) * beats_per_frame) % float(beats_per_bar)
        else:
            pos_in_measure = 0.0

        beat_phase = pos_in_measure / float(beats_per_bar) if beats_per_bar > 0 else 0.0
        # Output order matches ACE_NODES: Frames per Beat, Beat Phase, Audio passthrough
        return (frames_per_beat, beat_phase, chunk_in)

    def close(self):
        self._reset_long_term_analysis_state(clear_target=True)


    @staticmethod
    def _user_overrode_tempo(ctx: Optional[Any]) -> bool:
        if not ctx:
            return False
        getter = getattr(ctx, "get", None)
        if callable(getter):
            try:
                if getter("ace_bpm_user_override", False) or getter("ace_meter_user_override", False):
                    return True
            except Exception as exc:
                utils.note_swallowed("audio_tasks_registry.SourceSpecsTask._user_overrode_tempo", exc)
                return False
        return False

    def _reset_long_term_analysis_state(self, clear_target: bool = False):
        self._release_global_analysis_slot(self._long_term_target)
        rec = self._analysis_rec
        if rec is not None:
            with utils.logged_errors("audio_tasks_registry.SourceSpecsTask._reset_long_term_analysis_state"):
                rec.close()
        self._analysis_rec = None

        analyzer = self._analysis_analyzer
        if analyzer is not None:
            with utils.logged_errors("audio_tasks_registry.SourceSpecsTask._reset_long_term_analysis_state#2"):
                AudioFluxAnalyzer.release(analyzer)
        self._analysis_analyzer = None

        self._analysis_block = 0
        self._analysis_sr = SAMPLE_RATE
        self._analysis_flux_vals = []
        self._analysis_total_samples = 0
        self._analysis_frame_idx = 0
        self._analysis_next_tick_ts = 0.0
        if clear_target:
            self._long_term_target = None

    def _claim_global_analysis_slot(self, target: Tuple[str, float]) -> bool:
        owner = SourceSpecsTask._GLOBAL_ACTIVE_ANALYSIS.get(target)
        me = int(id(self))
        if owner is None or int(owner) == me:
            SourceSpecsTask._GLOBAL_ACTIVE_ANALYSIS[target] = me
            return True
        return False

    def _release_global_analysis_slot(self, target: Optional[Tuple[str, float]] = None):
        active_target = target if target is not None else self._long_term_target
        if active_target is None:
            return
        owner = SourceSpecsTask._GLOBAL_ACTIVE_ANALYSIS.get(active_target)
        if owner is None:
            return
        if int(owner) == int(id(self)):
            SourceSpecsTask._GLOBAL_ACTIVE_ANALYSIS.pop(active_target, None)

    @staticmethod
    def resolve_analysis_target(ctx: Optional[Any]) -> Optional[Tuple[str, float]]:
        if not ctx or str(getattr(ctx, "source_type", "") or "").upper() != "FILE":
            return None
        filepath = str(getattr(ctx, "filepath", "") or "").strip()
        if not filepath or not os.path.isfile(filepath):
            return None
        try:
            return (filepath, float(os.path.getmtime(filepath)))
        except OSError:
            return None

    @classmethod
    def get_cached_analysis(cls, target: Optional[Tuple[str, float]]) -> Optional[Dict[str, Any]]:
        if target is None:
            return None
        with cls._GLOBAL_CACHE_LOCK:
            raw = cls._GLOBAL_LONG_TERM_CACHE.get(target)
            cached = dict(raw) if isinstance(raw, dict) else None
        if not cached:
            return None
        if (time.time() - float(cached.get("ts", 0.0) or 0.0)) > cls._GLOBAL_CACHE_TTL:
            with cls._GLOBAL_CACHE_LOCK:
                cls._GLOBAL_LONG_TERM_CACHE.pop(target, None)
            return None
        return cached

    @classmethod
    def _store_global_analysis_result(
        cls,
        target: Tuple[str, float],
        bpm_est: float,
        beats_per_bar: int,
    ):
        with cls._GLOBAL_CACHE_LOCK:
            cls._GLOBAL_LONG_TERM_CACHE[target] = {
                "bpm": float(bpm_est),
                "beats_per_bar": int(max(1, beats_per_bar)),
                "ts": time.time(),
            }
            if len(cls._GLOBAL_LONG_TERM_CACHE) > cls._GLOBAL_CACHE_MAX:
                overflow = len(cls._GLOBAL_LONG_TERM_CACHE) - cls._GLOBAL_CACHE_MAX
                stale_items = heapq.nsmallest(
                    overflow,
                    cls._GLOBAL_LONG_TERM_CACHE.items(),
                    key=lambda kv: float(kv[1].get("ts", 0.0)) if isinstance(kv[1], dict) else 0.0,
                )
                for stale_key, _ in stale_items:
                    cls._GLOBAL_LONG_TERM_CACHE.pop(stale_key, None)

    @classmethod
    def _store_global_bpm_cache(cls, target: Tuple[str, float], bpm_est: float, beats_per_bar: int = 4):
        cls._store_global_analysis_result(target, bpm_est, beats_per_bar)

    @classmethod
    def _store_global_analysis_failure(cls, target: Tuple[str, float], error: str = ""):
        with cls._GLOBAL_CACHE_LOCK:
            cls._GLOBAL_LONG_TERM_CACHE[target] = {
                "bpm": None,
                "beats_per_bar": None,
                "error": str(error or "Tempo analysis failed"),
                "ts": time.time(),
            }
            if len(cls._GLOBAL_LONG_TERM_CACHE) > cls._GLOBAL_CACHE_MAX:
                overflow = len(cls._GLOBAL_LONG_TERM_CACHE) - cls._GLOBAL_CACHE_MAX
                stale_items = heapq.nsmallest(
                    overflow,
                    cls._GLOBAL_LONG_TERM_CACHE.items(),
                    key=lambda kv: float(kv[1].get("ts", 0.0)) if isinstance(kv[1], dict) else 0.0,
                )
                for stale_key, _ in stale_items:
                    cls._GLOBAL_LONG_TERM_CACHE.pop(stale_key, None)

    @classmethod
    def _ensure_background_worker(cls):
        thread = cls._BG_THREAD
        if thread is not None and thread.is_alive():
            return
        with cls._BG_THREAD_LOCK:
            thread = cls._BG_THREAD
            if thread is not None and thread.is_alive():
                return
            cls._BG_STOP_EVENT.clear()
            cls._BG_THREAD = threading.Thread(
                target=cls._background_worker_loop,
                name="ACE_SourceSpecsLongTermWorker",
                daemon=True,
            )
            cls._BG_THREAD.start()

    @classmethod
    def shutdown_background_worker(cls):
        cls._BG_STOP_EVENT.set()
        thread = cls._BG_THREAD
        if thread is not None and thread.is_alive():
            with utils.logged_errors("audio_tasks_registry.SourceSpecsTask.shutdown_background_worker"):
                thread.join(timeout=0.25)
        cls._BG_THREAD = None
        with cls._GLOBAL_CACHE_LOCK:
            cls._BG_PENDING.clear()
            cls._BG_QUEUED.clear()
            cls._GLOBAL_ACTIVE_ANALYSIS.clear()
        try:
            while True:
                cls._BG_TASK_QUEUE.get_nowait()
                cls._BG_TASK_QUEUE.task_done()
        except Empty:
            pass
        cls.clear_pending_ctx_writes_queue()

    @classmethod
    def clear_pending_ctx_writes_queue(cls) -> None:
        try:
            while True:
                cls._PENDING_CTX_WRITES.get_nowait()
                cls._PENDING_CTX_WRITES.task_done()
        except Empty:
            pass

    @classmethod
    def flush_pending_ctx_writes(cls, task_registry: dict) -> None:
        """Apply detected-tempo results to bpy PropertyGroup objects.

        Must be called from the main thread.
        """
        utils.assert_main_thread("SourceSpecsTask.flush_pending_ctx_writes")
        try:
            while True:
                payload = cls._PENDING_CTX_WRITES.get_nowait()
                if len(payload) >= 4:
                    target, bpm_val, beats_per_bar, generation_id = payload[:4]
                else:
                    target, bpm_val, beats_per_bar = payload
                    generation_id = None
                task = task_registry.get(target)
                ctx = getattr(task, "source_context", None) if task else None
                if ctx is None:
                    continue
                task_generation = int(getattr(task, "undo_generation", 0) or 0)
                engine_generation = int(
                    getattr(getattr(task, "engine_ref", None), "undo_generation", task_generation) or task_generation
                )
                if generation_id is not None and int(generation_id or 0) != engine_generation:
                    continue
                if task_generation != engine_generation:
                    continue
                if cls._user_overrode_tempo(ctx):
                    continue
                with utils.logged_errors("audio_tasks_registry.SourceSpecsTask.flush_pending_ctx_writes#2"):
                    ctx[task._auto_bpm_key] = float(bpm_val)
                    ctx[task._auto_meter_key] = int(max(1, beats_per_bar))
                    ctx.bpm = float(bpm_val)
                    ctx.beats_per_bar = int(max(1, beats_per_bar))
                    ctx["ace_bpm_user_override"] = False
                    ctx["ace_meter_user_override"] = False
        except Empty:
            pass
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.SourceSpecsTask.flush_pending_ctx_writes", exc)

    @classmethod
    def _queue_background_analysis(cls, target: Tuple[str, float]) -> bool:
        cls._ensure_background_worker()
        with cls._GLOBAL_CACHE_LOCK:
            if target in cls._BG_QUEUED:
                return True
            cls._BG_PENDING.add(target)
        try:
            cls._BG_TASK_QUEUE.put_nowait(target)
            with cls._GLOBAL_CACHE_LOCK:
                cls._BG_QUEUED.add(target)
            return True
        except Full:
            # Queue is full. Keep target in _BG_PENDING so callers avoid flooding
            # duplicate enqueue attempts while the worker drains existing items.
            return False

    @classmethod
    def _background_worker_loop(cls):
        while not cls._BG_STOP_EVENT.is_set():
            try:
                target = cls._BG_TASK_QUEUE.get(timeout=0.25)
            except Empty:
                continue
            with cls._GLOBAL_CACHE_LOCK:
                cls._BG_QUEUED.discard(target)

            try:
                detected = cls._estimate_tempo_for_target(target)
                if not detected:
                    cls._store_global_analysis_failure(target)
                else:
                    cls._store_global_analysis_result(
                        target,
                        float(detected["bpm"]),
                        int(detected["beats_per_bar"]),
                    )
            except Exception as exc:
                utils.note_swallowed("audio_tasks_registry.SourceSpecsTask._background_worker_loop", exc)
                cls._store_global_analysis_failure(target)
            finally:
                with cls._GLOBAL_CACHE_LOCK:
                    cls._BG_PENDING.discard(target)
                    cls._BG_QUEUED.discard(target)
                with utils.logged_errors("audio_tasks_registry.SourceSpecsTask._background_worker_loop#2"):
                    cls._BG_TASK_QUEUE.task_done()

    @staticmethod
    def _flux_peak_indices(flux_arr: np.ndarray) -> np.ndarray:
        flux = np.asarray(flux_arr, dtype=np.float32).reshape(-1)
        if flux.size < 4:
            return np.empty(0, dtype=np.int32)
        mean = float(np.mean(flux))
        std = float(np.std(flux))
        thresh = mean + (0.5 * std)
        mids = flux[1:-1]
        if mids.size <= 0:
            return np.empty(0, dtype=np.int32)
        peaks_mask = (mids > flux[:-2]) & (mids > flux[2:]) & (mids > thresh)
        return (np.nonzero(peaks_mask)[0] + 1).astype(np.int32, copy=False)

    @classmethod
    def _estimate_beats_per_bar_from_flux(
        cls,
        flux_arr: np.ndarray,
        peaks: np.ndarray,
        beat_period_blocks: float,
    ) -> int:
        flux = np.asarray(flux_arr, dtype=np.float32).reshape(-1)
        peak_idx = np.asarray(peaks, dtype=np.int32).reshape(-1)
        if flux.size < 8 or peak_idx.size < 4 or beat_period_blocks <= 1e-6:
            return 4

        strengths = flux[peak_idx].astype(np.float64, copy=False)
        mean_strength = float(np.mean(strengths) or 0.0)
        if mean_strength <= 1e-9:
            return 4

        beat_numbers = np.rint(
            (peak_idx.astype(np.float64) - float(peak_idx[0])) / max(float(beat_period_blocks), 1e-6)
        ).astype(np.int32, copy=False)
        centered_flux = flux.astype(np.float64) - float(np.mean(flux))
        best_candidate = 4
        best_score = -1.0e12
        max_candidate = min(8, max(2, int(peak_idx.size // 2)))

        for candidate in range(2, max_candidate + 1):
            sums = np.zeros(candidate, dtype=np.float64)
            counts = np.zeros(candidate, dtype=np.float64)
            for beat_num, strength in zip(beat_numbers, strengths):
                idx = int(beat_num % candidate)
                sums[idx] += float(strength)
                counts[idx] += 1.0
            if np.count_nonzero(counts) < candidate:
                continue

            means = np.divide(sums, np.maximum(counts, 1.0))
            normalized = means / mean_strength
            variance_score = float(np.var(normalized))
            downbeat_score = float(normalized[0] - np.mean(normalized[1:])) if candidate > 1 else 0.0

            periodicity_score = 0.0
            lag = int(round(float(beat_period_blocks) * float(candidate)))
            if 0 < lag < centered_flux.size:
                lhs = centered_flux[:-lag]
                rhs = centered_flux[lag:]
                denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
                if denom > 1e-9:
                    periodicity_score = float(np.dot(lhs, rhs) / denom)

            score = variance_score + max(downbeat_score, 0.0) + (0.5 * max(periodicity_score, 0.0))
            score -= 0.05 * float(candidate - 2)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        return int(max(best_candidate, 1))

    @classmethod
    def _estimate_tempo_from_flux(
        cls,
        flux_vals: list[float] | np.ndarray,
        block_size: int,
        sample_rate: float,
    ) -> Optional[Dict[str, Any]]:
        flux_arr = np.asarray(flux_vals, dtype=np.float32).reshape(-1)
        if flux_arr.size < 4:
            return None

        peaks = cls._flux_peak_indices(flux_arr)
        if peaks.size < 2:
            return None

        intervals = np.diff(peaks).astype(np.float64)
        intervals = intervals[intervals > 0.5]
        if intervals.size == 0:
            return None

        beat_period_blocks = float(np.median(intervals))
        seconds_per_beat = beat_period_blocks * (float(block_size) / max(float(sample_rate), 1.0))
        if seconds_per_beat <= 1e-6:
            return None

        bpm_est = 60.0 / seconds_per_beat
        if not np.isfinite(bpm_est):
            return None

        beats_per_bar = cls._estimate_beats_per_bar_from_flux(flux_arr, peaks, beat_period_blocks)
        return {
            "bpm": float(np.clip(bpm_est, 20.0, 300.0)),
            "beats_per_bar": int(max(1, beats_per_bar)),
        }

    @classmethod
    def _estimate_tempo_for_target(cls, target: Tuple[str, float]) -> Optional[Dict[str, Any]]:
        filepath, _mtime = target
        if not filepath or not os.path.isfile(filepath):
            return None

        rec = AudioReception("FILE", filepath=filepath)
        with _get_file_open_lock(filepath):
            opened = rec.open()
        if not opened:
            return None

        try:
            sr = int(getattr(rec, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
            block = _resolve_block_size_default(sr, 24.0)
            block = int(max(MIN_BLOCK_SIZE, min(block, MAX_BLOCK_SIZE)))
            max_samples = int(max(float(sr), 1.0) * cls._LONG_TERM_MAX_SECONDS)

            flux_vals: list[float] = []
            prev_mag: Optional[np.ndarray] = None
            total_samples = 0

            while total_samples < max_samples and not cls._BG_STOP_EVENT.is_set():
                payload = rec.read_block(block)
                if payload is None:
                    break
                mono, _stereo, sr_val = payload
                mono_arr = np.asarray(mono, dtype=np.float32).reshape(-1)
                if mono_arr.size <= 0:
                    break
                sr = int(max(int(sr_val or sr), 1))
                total_samples += int(mono_arr.size)

                mag = np.abs(np.fft.rfft(mono_arr, n=block)).astype(np.float32, copy=False)
                if prev_mag is not None and prev_mag.shape == mag.shape:
                    diff = mag - prev_mag
                    flux_vals.append(float(np.sum(np.maximum(diff, 0.0), dtype=np.float64)))
                prev_mag = mag

            return cls._estimate_tempo_from_flux(flux_vals, block, sr)
        finally:
            with utils.logged_errors("audio_tasks_registry.SourceSpecsTask._estimate_tempo_for_target"):
                rec.close()

    def _apply_detected_tempo(
        self,
        ctx: Optional[Any],
        bpm_val: float,
        beats_per_bar: int,
        *,
        respect_override: bool = True,
    ):
        # Update the instance cache (task-local state).
        self._long_term_cache["bpm"] = float(bpm_val)
        self._long_term_cache["beats_per_bar"] = int(max(1, beats_per_bar))
        self._long_term_done = True
        # Do not write bpy properties here: this may be called from background work.
        # Queue the write for main-thread flush instead.
        if ctx and (not respect_override or not SourceSpecsTask._user_overrode_tempo(ctx)):
            target = self._long_term_target
            if target is not None:
                with utils.logged_errors("audio_tasks_registry.SourceSpecsTask._apply_detected_tempo"):
                    SourceSpecsTask._PENDING_CTX_WRITES.put_nowait(
                        (
                            target,
                            float(bpm_val),
                            int(max(1, beats_per_bar)),
                            int(getattr(self, "undo_generation", 0) or 0),
                        )
                    )

    def _start_long_term_analysis(self, target: Tuple[str, float]) -> bool:
        filepath, _ = target
        rec = AudioReception("FILE", filepath=filepath)
        with _get_file_open_lock(filepath):
            opened = rec.open()
        if not opened:
            return False
        sr = int(getattr(rec, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
        block = _resolve_block_size_default(sr, getattr(self, "scene_fps", 24.0))
        block = int(max(MIN_BLOCK_SIZE, min(block, MAX_BLOCK_SIZE)))
        try:
            analyzer = AudioFluxAnalyzer(sr, block)
        except Exception as exc:
            utils.note_swallowed("audio_tasks_registry.SourceSpecsTask._start_long_term_analysis", exc)
            with utils.logged_errors("audio_tasks_registry.SourceSpecsTask._start_long_term_analysis#2"):
                rec.close()
            return False

        self._analysis_rec = rec
        self._analysis_analyzer = analyzer
        self._analysis_block = block
        self._analysis_sr = sr
        self._analysis_flux_vals = []
        self._analysis_total_samples = 0
        self._analysis_frame_idx = 0
        return True

    def _finalize_long_term_analysis(self, ctx: Optional[Any]):
        flux_vals = list(self._analysis_flux_vals)
        sr = max(float(self._analysis_sr), 1.0)
        block = max(int(self._analysis_block), 1)
        detected = self._estimate_tempo_from_flux(flux_vals, block, sr)
        if detected:
            target = self._long_term_target
            if target is not None:
                SourceSpecsTask._store_global_analysis_result(
                    target,
                    float(detected["bpm"]),
                    int(detected["beats_per_bar"]),
                )
            self._apply_detected_tempo(
                ctx,
                float(detected["bpm"]),
                int(detected["beats_per_bar"]),
            )
        else:
            self._long_term_done = True
        self._reset_long_term_analysis_state(clear_target=False)

    def _step_long_term_analysis(self, ctx: Optional[Any]):
        rec = self._analysis_rec
        analyzer = self._analysis_analyzer
        if rec is None or analyzer is None:
            self._long_term_done = True
            return

        max_samples = int(max(float(self._analysis_sr), 1.0) * self._LONG_TERM_MAX_SECONDS)
        reached_end = False
        for _ in range(max(int(self._LONG_TERM_BLOCKS_PER_TICK), 1)):
            if self._analysis_total_samples >= max_samples:
                reached_end = True
                break
            chunk = rec.read_block(self._analysis_block)
            if chunk is None:
                reached_end = True
                break
            mono, stereo, sr_val = chunk
            self._analysis_total_samples += int(np.asarray(mono).shape[-1])
            temp_chunk = AudioChunk(
                mono,
                stereo,
                sr_val,
                frame_index=self._analysis_frame_idx,
                lineage=("long_term",),
            )
            flux = analyzer.compute_feature("flux", temp_chunk)
            if flux is not None:
                self._analysis_flux_vals.append(float(flux))
            self._analysis_frame_idx += 1

        if reached_end or self._analysis_total_samples >= max_samples:
            self._finalize_long_term_analysis(ctx)

    def _maybe_run_long_term_analysis(self, ctx: Optional[Any]):
        target = self.resolve_analysis_target(ctx)
        if target is None:
            self._reset_long_term_analysis_state(clear_target=True)
            return
        if self._user_overrode_tempo(ctx):
            self._long_term_done = True
            self._reset_long_term_analysis_state(clear_target=False)
            return

        if self._long_term_target != target:
            self._reset_long_term_analysis_state(clear_target=False)
            self._long_term_target = target
            self._long_term_cache.clear()
            self._long_term_done = False

        cached = SourceSpecsTask.get_cached_analysis(target)
        if cached:
            bpm_cached = cached.get("bpm", None)
            beats_per_bar_cached = cached.get("beats_per_bar", None)
            if bpm_cached is not None:
                self._apply_detected_tempo(
                    ctx,
                    float(bpm_cached),
                    int(beats_per_bar_cached or 4),
                )
            else:
                self._long_term_done = True
            return

        if self._long_term_done:
            return

        now = time.time()
        if now < self._analysis_next_tick_ts:
            return
        self._analysis_next_tick_ts = now + self._LONG_TERM_TICK_INTERVAL_S

        # Run long-term tempo estimation outside the realtime DSP frame path.
        SourceSpecsTask._queue_background_analysis(target)
