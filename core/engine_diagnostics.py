"""Engine performance counters, latency metrics and runtime diagnostics."""

import bpy
import threading
import time
from collections import deque
from typing import Any, Dict
from .. import utils
from .proxy_management import ACEProxyManager
from .engine_common import engine_settings


class EngineDiagnostics:
    """Engine performance counters, latency metrics and runtime diagnostics."""

    _PERF_STAGES = ("source_read", "dsp", "preview", "proxy_write", "ui_sync")
    _PERF_HISTORY_FRAMES = 360
    _DSP_NODE_PROFILE_SAMPLES = 1024
    _BLENDER_LATENCY_EVENT_TTL_S = 2.0
    _CURRENT_LATENCY_SAMPLE_FRESHNESS_FRAMES = 2

    def __init__(self, engine):
        self.engine = engine
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

    @classmethod
    def _perf_enabled(cls, scene_or_engine_settings=None) -> bool:
        settings = engine_settings(scene_or_engine_settings)
        try:
            return bool(getattr(settings, "ace_perf_enabled", False))
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._perf_enabled", exc)
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
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._id_metric_token", exc)
            name = ""
        if not name:
            return ""
        try:
            type_name = str(getattr(getattr(datablock, "bl_rna", None), "identifier", "") or type(datablock).__name__)
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._id_metric_token#2", exc)
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
                "scene_key": int(self.engine._scene_key(scene)),
                "frame": int(self.engine.current_frame),
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
        scene_key = int(self.engine._scene_key(scene))
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
            self._geometry_eval_last_frame = int(self.engine.current_frame)

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
        preview_diag = dict(self.engine._preview_monitor.diagnostics_snapshot(reset=reset))
        worker_target = self.engine._parallel_worker_count(None)
        executor = self.engine._parallel_executor
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
            "parallel_queue_pressure_drops": float(self.engine._parallel_queue_pressure_drops),
            "parallel_job_timeouts": float(self.engine._parallel_job_timeouts),
            "parallel_job_exceptions": float(self.engine._parallel_job_exceptions),
            "parallel_pool_restarts": float(self.engine._parallel_restart_count),
            "parallel_stale_job_drops": float(self.engine._parallel_stale_job_drops),
            "parallel_stale_result_drops": float(self.engine._parallel_stale_result_drops),
            "parallel_result_accepts": float(self.engine._parallel_result_accepts),
            "parallel_disabled_sources": float(len(self.engine._parallel_disabled_sources)),
            "parallel_worker_target": float(worker_target),
            "parallel_worker_active": float(active_workers),
            "parallel_inflight_limit": float(self.engine._parallel_inflight_limit(None)),
            "parallel_payload_slots": float(len(self.engine._parallel_payload_slots)),
            "source_prefetch_mean_ms": float(source_prefetch_mean_ms),
            "source_prefetch_p95_ms": float(source_prefetch_p95_ms),
            "blender_geometry_eval_mean_ms": float(geometry_mean_ms),
            "blender_geometry_eval_p95_ms": float(geometry_p95_ms),
            "blender_render_mean_ms": float(render_mean_ms),
            "blender_render_p95_ms": float(render_p95_ms),
        }
        report.update(preview_diag)
        if reset:
            self.engine._parallel_queue_pressure_drops = 0
            self.engine._parallel_job_timeouts = 0
            self.engine._parallel_job_exceptions = 0
            self.engine._parallel_stale_job_drops = 0
            self.engine._parallel_stale_result_drops = 0
            self.engine._parallel_result_accepts = 0
            self._geometry_eval_history.clear()
            self._render_history.clear()
        return report

    def get_runtime_diagnostics_current(self) -> Dict[str, float]:
        now_ts = time.perf_counter()
        preview_diag = dict(self.engine._preview_monitor.current_snapshot())
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
            current_frame = int(self.engine.current_frame)
            last_frame = int(self._geometry_eval_last_frame)
            if (current_frame - last_frame) <= int(max(self._CURRENT_LATENCY_SAMPLE_FRESHNESS_FRAMES, 0)):
                geometry_ms = float(self._geometry_eval_history[-1] * 1000.0)
        render_ms = 0.0
        if self.engine._is_rendering_job() and float(self._render_start_ts or 0.0) > 0.0:
            render_ms = float(max(now_ts - float(self._render_start_ts), 0.0) * 1000.0)
        return {
            "source_prefetch_ms": float(source_prefetch_ms),
            "preview_branch_latency_ms": float(preview_diag.get("preview_branch_latency_ms", 0.0) or 0.0),
            "preview_output_buffer_ms": float(preview_diag.get("preview_output_buffer_ms", 0.0) or 0.0),
            "preview_callback_ms": float(preview_diag.get("preview_callback_ms", 0.0) or 0.0),
            "blender_geometry_eval_ms": float(geometry_ms),
            "blender_render_ms": float(render_ms),
        }
