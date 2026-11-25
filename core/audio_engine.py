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

import bpy
import numpy as np
import queue
import time
from collections import deque
from typing import Any, Dict, List, Optional
from .. import utils
from ..properties import ACEAudioSource
from .abstract_tree import ACEAbstractTree
from .audio_tasks_registry import ACEAudioGraph, ACEAudioTask, AudioChunk, PreviewSinkTask, SAMPLE_RATE, SourceSpecsTask, SourceTask, _clear_file_open_locks, reset_biquad_fallback_warned, resolve_context_block_size
from .proxy_management import ACEProxyManager
from .engine_common import FRONTEND_KIND_SHADER
from .engine_bake import BakeManager
from .engine_compiler import GraphCompiler
from .engine_diagnostics import EngineDiagnostics
from .engine_frontends import FrontendTracker
from .engine_parallel import ParallelOffload
from .engine_preview import PreviewRouter
from .engine_probes import SourceProbes
from .engine_sinks import SinkResolver


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

    # Attribute schema is synced via a dirty flag set on compile, not by frame polling.
    # The fallback interval applies only when the dirty set is missing (defensive).
    _ATTR_RESYNC_INTERVAL_FRAMES = 120

    _BLOCK_SIZE_HYSTERESIS_FRAMES = 2

    _BLOCK_SIZE_IMMEDIATE_DELTA = 1024

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
        self.effective_sample_rate: int = SAMPLE_RATE
        self._last_attr_sync_frame: Dict[str, int] = {}
        self._block_size_state: Dict[str, Dict[str, int]] = {}
        self._bootstrap_retry_at: float = 0.0
        self._bootstrap_fail_count: int = 0
        self._source_specs_task_registry: Dict[tuple[str, float], SourceSpecsTask] = {}
        self._current_delta_frames: float = 1.0
        self._preview_last_chunk: Optional[AudioChunk] = None
        self._preview_last_eval_frame: int = -1
        self._preview_next_eval_ts: float = 0.0
        # Tracks scene FPS so time-based constants can be converted to frame counts.
        self._current_scene_fps: float = 24.0
        # Dirty-flag set for attribute schema resync: populated on compile,
        # cleared per-graph after ensure_attributes_exist() runs.
        self._attr_schema_dirty: set[str] = set()
        self.undo_generation: int = 0
        self.undo_recovery_until: float = 0.0
        # Components own their own state; build them last so they can read the fields above.
        self.diagnostics = EngineDiagnostics(self)
        self.probes = SourceProbes(self)
        self.preview = PreviewRouter(self)
        self.bake = BakeManager(self)
        self.parallel = ParallelOffload(self)
        self.frontends = FrontendTracker(self)
        self.sinks = SinkResolver(self)
        self.compiler = GraphCompiler(self)

    @staticmethod
    def _scene_key(scene) -> int:
        """Return ``scene.as_pointer()`` as an int, or 0 if unavailable."""
        if scene is not None and hasattr(scene, "as_pointer"):
            with utils.logged_errors("audio_engine.ACEAudioEngine._scene_key"):
                return int(scene.as_pointer())
        return 0

    @staticmethod
    def _scene_name(scene) -> str:
        """Return the scene name string, preferring ``name_full`` over ``name``."""
        if scene is None:
            return ""
        try:
            return str(getattr(scene, "name_full", "") or getattr(scene, "name", "") or "")
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._scene_name", exc)
            return ""

    def _clear_blender_runtime_refs(self) -> None:
        self.abstract_tree_ref = None
        self.compiler._compiled_tree_ref = None
        self.compiler._compiled_tree_ptr = 0
        self.compiler._compiled_tree_frontend_uid = ""
        self.compiler._compiled_tree_bl_idname = ""
        self.compiler._compiled_frontend_refs.clear()
        self.compiler._compiled_frontend_snapshots.clear()
        self.compiler._compiled_frontend_ptrs.clear()
        self.compiler._compiled_frontend_uids = ()
        self.frontends._dirty_trees.clear()
        self.frontends._dirty_tree_ptrs.clear()
        self.frontends._dirty_tree_bl_idnames.clear()
        self.frontends._dirty_frontend_uids.clear()
        self.frontends._pending_tree = None
        self.frontends._pending_tree_ptr = 0
        self.frontends._pending_tree_bl_idname = ""
        self.frontends._pending_scene_rebuild = False
        self.compiler._last_frontend_membership_signature = ()

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
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._undo_recovery_active_for_scene", exc)
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
            with utils.logged_errors("audio_engine.ACEAudioEngine.shutdown_all_instances"):
                inst._shutdown_graphs(wait_parallel_pool=True, clear_bake_cache=True)
        cls._instances.clear()
        with utils.logged_errors("audio_engine.ACEAudioEngine.shutdown_all_instances#2"):
            PreviewRouter._preview_monitor.close()
        with utils.logged_errors("audio_engine.ACEAudioEngine.shutdown_all_instances#3"):
            SourceSpecsTask.shutdown_background_worker()
        with utils.logged_errors("audio_engine.ACEAudioEngine.shutdown_all_instances#4"):
            _clear_file_open_locks()
        with utils.logged_errors("audio_engine.ACEAudioEngine.shutdown_all_instances#5"):
            reset_biquad_fallback_warned()

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
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._scheduler_mode", exc)
            return "demand"

    @classmethod
    def _timeslice_mode(cls, scene_or_engine_settings=None) -> str:
        settings = cls._engine_settings(scene_or_engine_settings)
        try:
            val = str(getattr(settings, "ace_timeslice_mode", "fixed") or "fixed").strip().lower()
            return val if val in {"fixed", "adaptive"} else "fixed"
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._timeslice_mode", exc)
            return "fixed"

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
                        with utils.logged_errors("audio_engine.ACEAudioEngine._evaluate_graph_plan_iterative"):
                            input_data[input_idx] = extract_fn(socket_idx)
                            continue
                try:
                    input_data[input_idx] = upstream.get_output(frame_idx, socket_idx)
                except Exception as exc:
                    utils.note_swallowed("audio_engine.ACEAudioEngine._evaluate_graph_plan_iterative#2", exc)
                    input_data[input_idx] = None

            try:
                t0 = time.perf_counter()
                task.cached_output = task.process(input_data)
                self.diagnostics._dsp_profile_add(task, time.perf_counter() - t0)
                tag_fn = getattr(task, "_tag_output_frame", None)
                if callable(tag_fn):
                    tag_fn(task.cached_output, frame_idx)
                task.last_frame_processed = frame_idx
            except Exception as exc:
                utils.note_swallowed("audio_engine.ACEAudioEngine._evaluate_graph_plan_iterative#3", exc)
                continue

    def _collect_graph_results_for_frame(
        self,
        graph: ACEAudioGraph,
        frame_index: int,
        attr_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        output_slots = self.compiler._graph_output_slot_specs(graph, attr_keys=attr_keys)
        target_node_ids = sorted({node_id for _, node_id, _ in output_slots if node_id in graph.tasks})
        plan = self._graph_eval_plan_for_targets(graph, target_node_ids)
        if plan:
            t0_plan = time.perf_counter()
            self._evaluate_graph_plan_iterative(graph, frame_index, plan)
            self.diagnostics._perf_add("dsp", time.perf_counter() - t0_plan)
        for attr_key, node_id, socket_idx in output_slots:
            task = graph.tasks.get(node_id)
            if not task:
                continue
            if int(getattr(task, "last_frame_processed", -1)) == int(frame_index):
                extract_fn = getattr(task, "_extract_result", None)
                if callable(extract_fn):
                    try:
                        val = extract_fn(socket_idx)
                    except Exception as exc:
                        utils.note_swallowed("audio_engine.ACEAudioEngine._collect_graph_results_for_frame", exc)
                        val = task.get_output(frame_index, socket_idx)
                else:
                    val = task.get_output(frame_index, socket_idx)
            else:
                t0 = time.perf_counter()
                val = task.get_output(frame_index, socket_idx)
                self.diagnostics._perf_add("dsp", time.perf_counter() - t0)
            if val is None:
                continue
            results[attr_key] = self._normalize_output_value(attr_key, val)
        return results

    def _shutdown_graphs(self, wait_parallel_pool: bool = False, clear_bake_cache: bool = False):
        self.preview._stop_preview_worker()
        self.parallel._shutdown_parallel_pool(wait=wait_parallel_pool)
        for graph in self.graphs.values():
            for task in graph.tasks.values():
                close_fn = getattr(task, "close", None)
                if callable(close_fn):
                    with utils.logged_errors("audio_engine.ACEAudioEngine._shutdown_graphs"):
                        close_fn()
        self.graphs.clear()
        self._block_size_state.clear()
        self._clear_blender_runtime_refs()
        self.diagnostics._pending_geometry_events.clear()
        self.probes._clear_probe_runtime_state()
        self.compiler._last_tree_signature = None
        self._attr_schema_dirty.clear()
        self._last_attr_sync_frame.clear()
        self._source_specs_task_registry.clear()
        self.writer = None
        self._preview_last_chunk = None
        self._preview_last_eval_frame = -1
        self._preview_next_eval_ts = 0.0
        if clear_bake_cache:
            self.bake._clear_bake_cache()
        self.bake._bake_jobs.clear()
        try:
            while True:
                self.bake._bake_result_queue.get_nowait()
        except queue.Empty:
            pass
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._shutdown_graphs#2", exc)

    def invalidate_runtime_caches(self):
        writer = self.writer
        if writer is None:
            return
        with utils.logged_errors("audio_engine.ACEAudioEngine.invalidate_runtime_caches"):
            writer.clear_runtime_caches()

    def begin_undo_recovery(self, scene=None, cooldown_s: float = 0.75) -> None:
        self.undo_generation = int(getattr(self, "undo_generation", 0) or 0) + 1
        self.undo_recovery_until = time.monotonic() + max(float(cooldown_s), 0.0)
        with utils.logged_errors("audio_engine.ACEAudioEngine.begin_undo_recovery"):
            self.invalidate_runtime_caches()
        with utils.logged_errors("audio_engine.ACEAudioEngine.begin_undo_recovery#2"):
            SourceSpecsTask.shutdown_background_worker()
        self._shutdown_graphs(wait_parallel_pool=False, clear_bake_cache=False)
        now = time.time()
        self.frontends._pending_scene_rebuild = True
        self.frontends._last_dirty_time = now
        self.frontends._first_dirty_time = now
        self.frontends._last_request_time = now
        if scene is not None:
            self.scene_key = self._scene_key(scene)
            self.scene_name = self._scene_name(scene)

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
                except Exception as exc:
                    utils.note_swallowed("audio_engine.ACEAudioEngine.adjust_graph_block_size_for_task", exc)
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
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._is_rendering_job", exc)
            return False

    @staticmethod
    def _unsafe_for_handler_writes(scene) -> bool:
        """Avoid mesh/RNA writes from frame handlers while rendering without UI lock."""
        is_rendering = ACEAudioEngine._is_rendering_job()
        if not is_rendering:
            return False
        try:
            return not bool(getattr(getattr(scene, "render", None), "use_lock_interface", False))
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._unsafe_for_handler_writes", exc)
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
                        except Exception as exc:
                            utils.note_swallowed("audio_engine.ACEAudioEngine._is_shader_viewport_preview_active#2", exc)
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
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._is_shader_viewport_preview_active", exc)
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
        with utils.logged_errors("audio_engine.ACEAudioEngine._is_animation_playing"):
            wm = getattr(getattr(bpy, "context", None), "window_manager", None)
            windows = getattr(wm, "windows", []) if wm else []
            for win in windows:
                screen = getattr(win, "screen", None)
                if screen and bool(getattr(screen, "is_animation_playing", False)):
                    return True
        return False

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
            utils.log_throttled(
                "render_lock_interface_execute",
                "Skipping ACE frame writes during render. Enable Render > Lock Interface for safe handler updates.",
                "WARNING",
                min_interval=5.0,
            )
            return
        # Flush async bake progress/completions and any deferred SourceSpecs ctx writes.
        self.bake.flush_bake_results()
        self._refresh_source_specs_task_registry()
        SourceSpecsTask.flush_pending_ctx_writes(self._source_specs_task_registry)
        if self.bake._bake_jobs:
            # Bake worker evaluates the live graph/task set; suspend frame-runtime
            # evaluation while a bake is active to avoid cross-thread task mutation.
            utils.log_throttled(
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
            and not self.frontends._pending_tree_ptr
            and not self.frontends._pending_scene_rebuild
            and not self.frontends._dirty_frontend_uids
        ):
            return

        try:
            delta_frames = float(int(scene_frame) - int(prev_scene_frame)) if prev_scene_frame >= 0 else 1.0
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine.execute_next_frame", exc)
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
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine.execute_next_frame#2", exc)
            fps = 24.0
        self._current_scene_fps = max(float(fps), 1.0)

        # Keep per-source BPM / beats-per-bar in each graph's parallel IR fresh
        # so _WorkerSourceSpecsPassthroughTask always uses the current values.
        # This is especially important for LIVE sources that have no auto-BPM
        # detection: the user sets BPM manually and the worker must see the update
        # on the very next frame rather than waiting for the next full recompile.
        with utils.logged_errors("audio_engine.ACEAudioEngine.execute_next_frame#3"):
            self.parallel._refresh_specs_ir_bpm()

        # Flush background probe results before the compile so native_sample_rate
        # is populated for any sources whose probe thread has just completed.
        with utils.logged_errors("audio_engine.ACEAudioEngine.execute_next_frame#4"):
            ace = getattr(scene, "audiocore", None)
            sources = list(getattr(ace, "audio_sources", []) or []) if ace else []
            if self.probes._flush_probe_updates(
                sources,
                scene=scene,
                preferred_tree_ptr=int(getattr(self.compiler, "_compiled_tree_ptr", 0) or 0),
            ):
                pass  # recompile already requested inside _flush_probe_updates
            self.probes.flush_source_tempo_detection_updates(sources)

        if self.frontends._maybe_compile_dirty(scene):
            # Freshly compiled this frame — skip DSP writes so Blender's GN
            # evaluator gets one clean cycle with the new graph before we start
            # pushing attribute values into it.  Without this guard the first
            # frame after a recompile can produce partial / stale geometry output.
            return

        if not self.writer:
            return
        scheduler_mode = self._scheduler_mode(scene)
        timeslice_mode = self._timeslice_mode(scene)
        parallel_mode = self.parallel._parallel_mode(scene)
        parallel_pool_ready = False
        if parallel_mode == "processes_dsp":
            parallel_pool_ready = self.parallel._ensure_parallel_pool(scene)
        elif self.parallel._parallel_executor is not None:
            self.parallel._shutdown_parallel_pool()

        # Keep the preview feeder's drift-correction in sync with the scene fps.
        with utils.logged_errors("audio_engine.ACEAudioEngine.execute_next_frame#5"):
            PreviewRouter._preview_monitor.update_scene_fps(fps)

        # Seek detection: flush stale audio from all monitor buffers when the
        # user scrubs to a non-adjacent frame or scrubs backward. Without this,
        # the ring buffer can hold future-relative audio that plays at the wrong
        # timeline position.  Any backward frame (delta < 0) triggers a flush
        # because even going back one frame means buffered audio is now "ahead".
        with utils.logged_errors("audio_engine.ACEAudioEngine.execute_next_frame#6"):
            if abs(delta_frames) > 2 or delta_frames < -0.5:
                PreviewRouter._preview_monitor.clear()

        self.parallel._parallel_poll_futures(scene, current_frame=int(self.current_frame))
        preview_allowed = self._is_animation_playing()
        preview_block_size = int(self.preview._preview_block_size(scene))
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
        with utils.logged_errors("audio_engine.ACEAudioEngine.execute_next_frame#7"):
            PreviewRouter._preview_monitor.hint_block_size(_min_analysis_bs)
        preview_interval_s = max(
            float(preview_block_size) / float(max(preview_sample_rate, 1)),
            float(self.preview._PREVIEW_MIN_INTERVAL_S),
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
                    self.parallel._parallel_poll_futures(scene, current_frame=int(self.current_frame))
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
                        except Exception as exc:
                            utils.note_swallowed("audio_engine.ACEAudioEngine.execute_next_frame#8", exc)
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
                        graph.sink_bindings_by_attr = self.sinks._resolve_sink_bindings_for_graph(
                            graph,
                            self.abstract_tree_ref,
                            source_uid=str(getattr(graph, "source_uid", "") or ""),
                            proxy_name=str(getattr(graph, "proxy_name", "") or ""),
                            single_source_mode=bool(len(self.graphs) <= 1),
                        )
                        self.compiler._rebuild_graph_output_slot_specs(graph)
                        if self.abstract_tree_ref is not None:
                            self.parallel._prepare_parallel_graph_ir(graph, self.abstract_tree_ref)
                        try:
                            t0 = time.perf_counter()
                            touched = self.writer.ensure_sink_layout(graph, active_keys=set(graph.output_attr_keys))
                            self.diagnostics._perf_add("proxy_write", time.perf_counter() - t0)
                            if isinstance(touched, set):
                                geometry_proxy_names.update({str(v) for v in touched if str(v)})
                        except Exception as e:
                            utils.log_throttled(f"attr_ensure_recover:{graph_key}", f"Attribute ensure after recovery failed for '{graph_key}': {e}", "ERROR", min_interval=2.0)
                        utils.log_throttled(
                            f"recover_keys:{graph_key}",
                            f"Recovered {len(graph.output_attr_keys)} output keys for source '{getattr(graph, 'source_uid', '')}'.",
                            "WARNING",
                            min_interval=1.5,
                        )
                    else:
                        if graph.preview_task_ids:
                            continue
                        utils.log_throttled(
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
                        self.diagnostics._perf_add("proxy_write", time.perf_counter() - t0)
                        if isinstance(touched, set):
                            geometry_proxy_names.update({str(v) for v in touched if str(v)})
                    except Exception as e:
                        utils.log_throttled(f"attr_ensure:{graph_key}", f"Attribute ensure failed for '{graph_key}': {e}", "ERROR", min_interval=2.0)
                    self._attr_schema_dirty.discard(graph_key)
                    self._last_attr_sync_frame[graph_key] = int(self.current_frame)

                baked_results = self.bake._get_baked_frame_results(graph, self.current_frame)
                if baked_results is not None:
                    try:
                        t0 = time.perf_counter()
                        touched = self.writer.commit_results(graph, baked_results, active_keys=set(graph.output_attr_keys))
                        self.diagnostics._perf_add("proxy_write", time.perf_counter() - t0)
                        if isinstance(touched, set):
                            geometry_proxy_names.update({str(v) for v in touched if str(v)})
                    except Exception as e:
                        utils.log_throttled(f"attr_write_baked:{graph_key}", f"Baked attribute write failed for '{graph_key}': {e}", "ERROR", min_interval=2.0)
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
                    utils.log_throttled(f"source_unready:{graph_key}", f"Skipping source '{graph_key}': {msg}", "ERROR", min_interval=2.0)
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
                    and parallel_graph_key not in self.parallel._parallel_disabled_sources
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
                        except Exception as exc:
                            utils.note_swallowed("audio_engine.ACEAudioEngine.execute_next_frame#9", exc)
                            src_chunk = None
                        if hasattr(src_chunk, "mono") and hasattr(src_chunk, "sample_rate"):
                            source_chunks_for_worker[str(src_node_id)] = src_chunk
                    if source_chunks_for_worker:
                        shard_specs = list(getattr(graph, "parallel_output_shards", []) or [])
                        if not shard_specs:
                            shard_specs = self.parallel._parallel_shard_attr_keys(
                                list(getattr(graph, "parallel_offload_attr_keys", []) or [])
                            )
                            graph.parallel_output_shards = list(shard_specs)
                        if not shard_specs:
                            parallel_submit_enabled = False
                        else:
                            for shard_idx, shard_keys in enumerate(shard_specs):
                                if not shard_keys:
                                    continue
                                self.parallel._parallel_submit_graph_job(
                                    scene=scene,
                                    graph=graph,
                                    graph_key=parallel_graph_key,
                                    queue_key=f"{parallel_graph_key}:shard{int(shard_idx)}",
                                    owner_graph_key=parallel_graph_key,
                                    frame_index=int(self.current_frame),
                                    source_chunks=source_chunks_for_worker,
                                    output_keys=list(shard_keys),
                                    include_graph_ir=bool(parallel_graph_key in self.parallel._parallel_graph_ir_required),
                                )
                    else:
                        parallel_submit_enabled = False

                if parallel_offload_graph:
                    # Stage B: asynchronous numeric DSP worker path (fresh -> stale -> zero).
                    local_keys = list(getattr(graph, "parallel_local_attr_keys", []) or [])
                    has_parallel_values = self.parallel._parallel_has_usable_values_for_graph_frame(
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
                        parallel_values = self.parallel._parallel_values_for_graph_frame(
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
                    self.preview._start_preview_worker()  # idempotent
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
                            raw_chunk = self.preview._task_current_chunk_for_frame(src_task, self.current_frame)
                            if raw_chunk is None:
                                try:
                                    t0 = time.perf_counter()
                                    raw_chunk = src_task.get_output(self.current_frame, int(src_sock_idx))
                                    self.diagnostics._perf_add("preview", time.perf_counter() - t0)
                                except Exception as exc:
                                    utils.note_swallowed("audio_engine.ACEAudioEngine.execute_next_frame#10", exc)
                                    raw_chunk = None
                            if raw_chunk is not None and self.preview._chunk_is_unmasked_audio(raw_chunk):
                                preview_chunks.append(
                                    self.preview._fit_preview_chunk_block(raw_chunk, preview_block_size)
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
                            upstream_chunk = self.preview._task_current_chunk_for_frame(
                                upstream_task, self.current_frame
                            )
                            if upstream_chunk is None:
                                try:
                                    t0 = time.perf_counter()
                                    upstream_chunk = upstream_task.get_output(
                                        self.current_frame, sock_idx
                                    )
                                    self.diagnostics._perf_add("preview", time.perf_counter() - t0)
                                except Exception as exc:
                                    utils.note_swallowed("audio_engine.ACEAudioEngine.execute_next_frame#11", exc)
                                    upstream_chunk = None
                            if upstream_chunk is not None:
                                if self.preview._submit_preview_work(preview_task, upstream_chunk):
                                    # Worker owns the push; mark preview as active so
                                    # the monitor is not cleared this frame.
                                    preview_pushed = True

                try:
                    # Stage C (main thread only): commit numeric results into resolved sinks.
                    t0 = time.perf_counter()
                    touched = self.writer.commit_results(graph, results, active_keys=set(graph.output_attr_keys))
                    self.diagnostics._perf_add("proxy_write", time.perf_counter() - t0)
                    if isinstance(touched, set):
                        geometry_proxy_names.update({str(v) for v in touched if str(v)})
                except Exception as e:
                    utils.log_throttled(f"attr_write:{graph_key}", f"Attribute write failed for '{graph_key}': {e}", "ERROR", min_interval=2.0)

            if preview_allowed:
                t0 = time.perf_counter()
                if preview_eval_due:
                    mixed_preview = self.preview._mix_preview_chunks(preview_chunks)
                    self._preview_last_chunk = mixed_preview
                    self._preview_last_eval_frame = int(self.current_frame)
                else:
                    mixed_preview = self._preview_last_chunk
                if mixed_preview is not None:
                    with utils.logged_errors("audio_engine.ACEAudioEngine.execute_next_frame#12"):
                        self.preview._preview_monitor.push_chunk(mixed_preview)
                        preview_pushed = True
                self.diagnostics._perf_add("preview", time.perf_counter() - t0)
            else:
                self._preview_last_chunk = None
                self._preview_last_eval_frame = -1
                self._preview_next_eval_ts = 0.0

            if (not preview_allowed) or (not preview_pushed):
                with utils.logged_errors("audio_engine.ACEAudioEngine.execute_next_frame#13"):
                    self.preview._preview_monitor.clear()
        finally:
            if geometry_proxy_names:
                self.diagnostics._mark_geometry_eval_pending(scene, proxy_names=geometry_proxy_names)
            self.diagnostics._perf_finalize_frame()

    @staticmethod
    def _tree_has_ace_nodes(tree: bpy.types.NodeTree | None):
        return FrontendTracker._tree_has_ace_nodes(tree)

    @classmethod
    def _tree_is_ace_managed(cls, tree):
        return FrontendTracker._tree_is_ace_managed(tree)

    def bake_source_to_cache(self, source, tree=None, scene=None):
        return self.bake.bake_source_to_cache(source, tree, scene)

    def compile(self, abstract_tree: ACEAbstractTree, blender_scene):
        return self.compiler.compile(abstract_tree, blender_scene)

    def compile_from_blender_tree(self, bl_tree: bpy.types.NodeTree, scene):
        return self.compiler.compile_from_blender_tree(bl_tree, scene)

    def compile_scene_frontends(self, scene, preferred_tree=None):
        return self.compiler.compile_scene_frontends(scene, preferred_tree)

    def discover_scene_frontends(self, scene):
        return self.frontends.discover_scene_frontends(scene)

    def flush_bake_results(self):
        return self.bake.flush_bake_results()

    def flush_source_tempo_detection_updates(self, sources):
        return self.probes.flush_source_tempo_detection_updates(sources)

    def get_dsp_node_profile_snapshot(self, reset: bool=False):
        return self.diagnostics.get_dsp_node_profile_snapshot(reset)

    def get_perf_current(self):
        return self.diagnostics.get_perf_current()

    def get_perf_snapshot(self, reset: bool=False):
        return self.diagnostics.get_perf_snapshot(reset)

    def get_runtime_diagnostics(self, reset: bool=False):
        return self.diagnostics.get_runtime_diagnostics(reset)

    def get_runtime_diagnostics_current(self):
        return self.diagnostics.get_runtime_diagnostics_current()

    @classmethod
    def preview_monitor_available(cls):
        return PreviewRouter.preview_monitor_available()

    @classmethod
    def preview_monitor_consecutive_errors(cls):
        return PreviewRouter.preview_monitor_consecutive_errors()

    def request_recompile(self, bl_tree: bpy.types.NodeTree | None):
        return self.frontends.request_recompile(bl_tree)

    def request_source_tempo_detection(self, source: Optional[ACEAudioSource]):
        return self.probes.request_source_tempo_detection(source)
