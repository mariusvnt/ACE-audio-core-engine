"""Whole-file tempo analysis: background worker, shared results and deferred writes."""

import os
import heapq
import threading
import time
from queue import Empty, Full
from typing import Dict, Any, Optional, Tuple
import numpy as np
from ..input.reception import AudioReception
from ... import utils
from .analyzer import AudioFluxAnalyzer
from .block_size import _resolve_block_size_default
from .chunk import AudioChunk, MAX_BLOCK_SIZE, MIN_BLOCK_SIZE, SAMPLE_RATE
from .source import _get_file_open_lock


class LongTermAnalysis:
    def _reset_long_term_analysis_state(self, clear_target: bool = False):
        self._release_global_analysis_slot(self._long_term_target)
        rec = self._analysis_rec
        if rec is not None:
            with utils.logged_errors("specs_analysis.LongTermAnalysis._reset_long_term_analysis_state"):
                rec.close()
        self._analysis_rec = None

        analyzer = self._analysis_analyzer
        if analyzer is not None:
            with utils.logged_errors("specs_analysis.LongTermAnalysis._reset_long_term_analysis_state#2"):
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
        owner = self._GLOBAL_ACTIVE_ANALYSIS.get(target)
        me = int(id(self))
        if owner is None or int(owner) == me:
            self._GLOBAL_ACTIVE_ANALYSIS[target] = me
            return True
        return False

    def _release_global_analysis_slot(self, target: Optional[Tuple[str, float]] = None):
        active_target = target if target is not None else self._long_term_target
        if active_target is None:
            return
        owner = self._GLOBAL_ACTIVE_ANALYSIS.get(active_target)
        if owner is None:
            return
        if int(owner) == int(id(self)):
            self._GLOBAL_ACTIVE_ANALYSIS.pop(active_target, None)

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
            with utils.logged_errors("specs_analysis.LongTermAnalysis.shutdown_background_worker"):
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
        """Apply detected-tempo results to bpy PropertyGroup objects."""
        utils.assert_main_thread("cls.flush_pending_ctx_writes")
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
                with utils.logged_errors("specs_analysis.LongTermAnalysis.flush_pending_ctx_writes"):
                    ctx[task._auto_bpm_key] = float(bpm_val)
                    ctx[task._auto_meter_key] = int(max(1, beats_per_bar))
                    ctx.bpm = float(bpm_val)
                    ctx.beats_per_bar = int(max(1, beats_per_bar))
                    ctx["ace_bpm_user_override"] = False
                    ctx["ace_meter_user_override"] = False
        except Empty:
            pass
        except Exception as exc:
            utils.note_swallowed("specs_analysis.LongTermAnalysis.flush_pending_ctx_writes#2", exc)

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
                utils.note_swallowed("specs_analysis.LongTermAnalysis._background_worker_loop", exc)
                cls._store_global_analysis_failure(target)
            finally:
                with cls._GLOBAL_CACHE_LOCK:
                    cls._BG_PENDING.discard(target)
                    cls._BG_QUEUED.discard(target)
                with utils.logged_errors("specs_analysis.LongTermAnalysis._background_worker_loop#2"):
                    cls._BG_TASK_QUEUE.task_done()

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
            utils.note_swallowed("specs_analysis.LongTermAnalysis._start_long_term_analysis", exc)
            with utils.logged_errors("specs_analysis.LongTermAnalysis._start_long_term_analysis#2"):
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
                self._store_global_analysis_result(
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

        cached = self.get_cached_analysis(target)
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
        self._queue_background_analysis(target)
