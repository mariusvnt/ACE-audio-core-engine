"""Per-frame execution: block sizing, graph evaluation, preview and sink writes."""

import numpy as np
from dataclasses import dataclass, field
import time
from collections import deque
from typing import Any, Dict, List, Optional
from ... import utils
from ..tasks.base import ACEAudioTask
from ..tasks.block_size import resolve_context_block_size
from ..tasks.chunk import AudioChunk, SAMPLE_RATE
from ..tasks.graph import ACEAudioGraph
from ..tasks.preview import PreviewSinkTask
from ..tasks.source import SourceTask
from ..tasks.specs import SourceSpecsTask
from .common import engine_settings


@dataclass
class _FrameContext:
    """Per-frame values shared by the FrameRunner steps."""

    fps: float
    delta_frames: float
    scheduler_mode: str
    timeslice_mode: str
    parallel_mode: str
    parallel_pool_ready: bool
    preview_allowed: bool
    preview_eval_due: bool
    preview_block_size: int
    preview_chunks: list = field(default_factory=list)
    geometry_proxy_names: set = field(default_factory=set)
    preview_pushed: bool = False


class FrameRunner:
    """Per-frame execution: block sizing, graph evaluation, preview and sink writes."""

    # Attribute schema is synced via a dirty flag set on compile, not by frame polling.
    # The fallback interval applies only when the dirty set is missing (defensive).
    _ATTR_RESYNC_INTERVAL_FRAMES = 120
    _BLOCK_SIZE_HYSTERESIS_FRAMES = 2
    _BLOCK_SIZE_IMMEDIATE_DELTA = 1024

    def __init__(self, engine):
        self.engine = engine
        self._block_size_state: Dict[str, Dict[str, int]] = {}
        self._source_specs_task_registry: Dict[tuple[str, float], SourceSpecsTask] = {}
        self._preview_last_chunk: Optional[AudioChunk] = None
        self._preview_last_eval_frame: int = -1
        self._preview_next_eval_ts: float = 0.0

    def execute_next_frame(self, scene, force: bool = False):
        """Evaluate all graphs for the current scene frame and write results to their sinks.

        Called every frame from ``frame_change_handler`` (and during render).
        ``force`` skips the same-frame guard (render handlers need a write
        even when the frame number has not changed).
        """
        utils.assert_main_thread("execute_next_frame")
        clock = self._start_frame(scene, force)
        if clock is None:
            return
        fps, delta_frames = clock
        if self._sync_runtime_inputs(scene):
            # Freshly compiled this frame: give Geometry Nodes one clean cycle
            # with the new graph before values are pushed into it.
            return
        ctx = self._frame_context(scene, fps, delta_frames)
        if ctx is None:
            return
        try:
            for _, graph in self.engine.graphs.items():
                self._run_graph(scene, graph, ctx)
            self._finish_preview(ctx)
        finally:
            if ctx.geometry_proxy_names:
                self.engine.diagnostics._mark_geometry_eval_pending(scene, proxy_names=ctx.geometry_proxy_names)
            self.engine.diagnostics._perf_finalize_frame()

    def _start_frame(self, scene, force):
        """Apply the frame guards and advance the clock; None means skip this frame."""
        self.engine.scene_key = self.engine._scene_key(scene)
        if scene is not None:
            self.engine.scene_name = self.engine._scene_name(scene)
        if self.engine._undo_recovery_active():
            return
        if self.engine._unsafe_for_handler_writes(scene):
            utils.log_throttled(
                "render_lock_interface_execute",
                "Skipping ACE frame writes during render. Enable Render > Lock Interface for safe handler updates.",
                "WARNING",
                min_interval=5.0,
            )
            return
        # Flush async bake progress/completions and any deferred SourceSpecs ctx writes.
        self.engine.bake.flush_bake_results()
        self._refresh_source_specs_task_registry()
        SourceSpecsTask.flush_pending_ctx_writes(self._source_specs_task_registry)
        if self.engine.bake._bake_jobs:
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
            scene_frame = self.engine.current_frame + 1

        prev_scene_frame = int(self.engine.last_scene_frame)
        if (
            (not force)
            and scene_frame == prev_scene_frame
            and not self.engine.frontends._pending_tree_ptr
            and not self.engine.frontends._pending_scene_rebuild
            and not self.engine.frontends._dirty_frontend_uids
        ):
            return

        try:
            delta_frames = float(int(scene_frame) - int(prev_scene_frame)) if prev_scene_frame >= 0 else 1.0
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine.execute_next_frame", exc)
            delta_frames = 1.0
        if abs(delta_frames) <= 1e-9:
            delta_frames = 1.0
        self.engine._current_delta_frames = float(delta_frames)

        self.engine.current_frame = int(scene_frame)
        self.engine.last_scene_frame = int(scene_frame)

        fps = 24.0
        try:
            raw_fps = float(getattr(getattr(scene, "render", None), "fps", 24.0) or 24.0)
            fps_base = float(getattr(getattr(scene, "render", None), "fps_base", 1.0) or 1.0)
            if fps_base > 0.0:
                fps = raw_fps / fps_base
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine.execute_next_frame#2", exc)
            fps = 24.0
        self.engine._current_scene_fps = max(float(fps), 1.0)
        return float(fps), float(delta_frames)

    def _sync_runtime_inputs(self, scene) -> bool:
        """Refresh BPM in worker IRs, flush probe results, run a pending compile.

        Returns True when a compile ran this frame.
        """
        # Keep per-source BPM / beats-per-bar in each graph's parallel IR fresh
        # so _WorkerSourceSpecsPassthroughTask always uses the current values.
        # This is especially important for LIVE sources that have no auto-BPM
        # detection: the user sets BPM manually and the worker must see the update
        # on the very next frame rather than waiting for the next full recompile.
        with utils.logged_errors("audio_engine.ACEAudioEngine.execute_next_frame#3"):
            self.engine.parallel.refresh_specs_params()

        # Flush background probe results before the compile so native_sample_rate
        # is populated for any sources whose probe thread has just completed.
        with utils.logged_errors("audio_engine.ACEAudioEngine.execute_next_frame#4"):
            ace = getattr(scene, "audiocore", None)
            sources = list(getattr(ace, "audio_sources", []) or []) if ace else []
            if self.engine.probes._flush_probe_updates(
                sources,
                scene=scene,
                preferred_tree_ptr=int(getattr(self.engine.compiler, "_compiled_tree_ptr", 0) or 0),
            ):
                pass  # recompile already requested inside _flush_probe_updates
            self.engine.probes.flush_source_tempo_detection_updates(sources)

        if self.engine.frontends._maybe_compile_dirty(scene):
            # Freshly compiled this frame — skip DSP writes so Blender's GN
            # evaluator gets one clean cycle with the new graph before we start
            # pushing attribute values into it.  Without this guard the first
            # frame after a recompile can produce partial / stale geometry output.
            return True
        return False

    def _frame_context(self, scene, fps, delta_frames):
        """Modes, worker pool, preview timing and monitor state for this frame; None without a writer."""
        if not self.engine.writer:
            return
        scheduler_mode = self._scheduler_mode(scene)
        timeslice_mode = self._timeslice_mode(scene)
        parallel_mode = self.engine.parallel.mode(scene)
        parallel_pool_ready = False
        if parallel_mode == "processes_dsp":
            parallel_pool_ready = self.engine.parallel.ensure_pool(scene)
        elif self.engine.parallel.pool.executor is not None:
            self.engine.parallel.shutdown_pool()

        # Keep the preview feeder's drift-correction in sync with the scene fps.
        with utils.logged_errors("audio_engine.ACEAudioEngine.execute_next_frame#5"):
            self.engine.preview._preview_monitor.update_scene_fps(fps)

        # Seek detection: flush stale audio from all monitor buffers when the
        # user scrubs to a non-adjacent frame or scrubs backward. Without this,
        # the ring buffer can hold future-relative audio that plays at the wrong
        # timeline position.  Any backward frame (delta < 0) triggers a flush
        # because even going back one frame means buffered audio is now "ahead".
        with utils.logged_errors("audio_engine.ACEAudioEngine.execute_next_frame#6"):
            if abs(delta_frames) > 2 or delta_frames < -0.5:
                self.engine.preview._preview_monitor.clear()

        self.engine.parallel.poll(scene, current_frame=int(self.engine.current_frame))
        preview_allowed = self.engine._is_animation_playing()
        preview_block_size = int(self.engine.preview._preview_block_size(scene))
        preview_sample_rate = int(max(self.engine.effective_sample_rate, SAMPLE_RATE, 1))
        _min_analysis_bs = 0
        for graph in self.engine.graphs.values():
            source_ctx = getattr(graph, "source_context", None)
            source_sr = int(getattr(source_ctx, "sample_rate", 0) or 0)
            if source_sr > 0:
                preview_sample_rate = max(preview_sample_rate, source_sr)
            _gbs = int(getattr(graph, "block_size", 0) or 0)
            if _gbs >= 64:
                _min_analysis_bs = _gbs if _min_analysis_bs == 0 else min(_min_analysis_bs, _gbs)
        with utils.logged_errors("audio_engine.ACEAudioEngine.execute_next_frame#7"):
            self.engine.preview._preview_monitor.hint_block_size(_min_analysis_bs)
        preview_interval_s = max(
            float(preview_block_size) / float(max(preview_sample_rate, 1)),
            float(self.engine.preview._PREVIEW_MIN_INTERVAL_S),
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
        geometry_proxy_names: set[str] = set()
        return _FrameContext(
            fps=fps,
            delta_frames=delta_frames,
            scheduler_mode=scheduler_mode,
            timeslice_mode=timeslice_mode,
            parallel_mode=parallel_mode,
            parallel_pool_ready=parallel_pool_ready,
            preview_allowed=preview_allowed,
            preview_eval_due=preview_eval_due,
            preview_block_size=preview_block_size,
            preview_chunks=preview_chunks,
            geometry_proxy_names=geometry_proxy_names,
        )

    def _run_graph(self, scene, graph, ctx):
        """Size, evaluate and commit one source graph for this frame."""
        graph_key, source_tasks = self._prepare_graph(scene, graph, ctx)
        if not self._ensure_graph_outputs(graph, graph_key, ctx):
            return
        self._sync_graph_schema(graph, graph_key, ctx)
        if self._write_baked_results(graph, graph_key, ctx):
            return
        if not self._graph_sources_ready(graph, graph_key):
            return
        results = self._compute_graph_results(scene, graph, graph_key, source_tasks, ctx)
        if ctx.preview_eval_due:
            # After the main evaluation, so FILE-source prefetch has the full
            # frame interval to fill.
            self._collect_graph_preview(graph, ctx)
        self._commit_graph_results(graph, graph_key, results, ctx)

    def _prepare_graph(self, scene, graph, ctx):
        """Set the graph's modes, pick its source tasks and adjust its block size."""
        fps = ctx.fps
        delta_frames = ctx.delta_frames
        scheduler_mode = ctx.scheduler_mode
        timeslice_mode = ctx.timeslice_mode
        parallel_mode = ctx.parallel_mode
        if parallel_mode == "processes_dsp":
            self.engine.parallel.poll(scene, current_frame=int(self.engine.current_frame))
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
                or self.engine.effective_sample_rate
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
                    source_task.get_output(self.engine.current_frame, 0)
                except Exception as exc:
                    utils.note_swallowed("audio_engine.ACEAudioEngine.execute_next_frame#8", exc)
                    continue
        return graph_key, source_tasks

    def _ensure_graph_outputs(self, graph, graph_key, ctx):
        """Resolve the proxy and output keys, recovering them when missing; False skips the graph."""
        geometry_proxy_names = ctx.geometry_proxy_names
        proxy = graph.get_proxy_object()
        if not proxy:
            source_proxy = getattr(getattr(graph, "source_context", None), "proxy_object_name", "")
            if source_proxy and source_proxy != getattr(graph, "proxy_name", ""):
                graph.proxy_name = source_proxy
                proxy = graph.get_proxy_object()
        if not graph.output_attr_keys:
            recovered: list[str] = []
            if self.engine.writer and self.engine.abstract_tree_ref:
                for node_id in graph.tasks.keys():
                    node = self.engine.abstract_tree_ref.nodes.get(node_id)
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
                        if key in self.engine.writer.output_defs:
                            recovered.append(key)
            if recovered:
                graph.output_attr_keys = sorted(set(recovered))
                graph.sink_bindings_by_attr = self.engine.sinks._resolve_sink_bindings_for_graph(
                    graph,
                    self.engine.abstract_tree_ref,
                    source_uid=str(getattr(graph, "source_uid", "") or ""),
                    proxy_name=str(getattr(graph, "proxy_name", "") or ""),
                    single_source_mode=bool(len(self.engine.graphs) <= 1),
                )
                self.engine.compiler._rebuild_graph_output_slot_specs(graph)
                if self.engine.abstract_tree_ref is not None:
                    self.engine.parallel.prepare_graph_ir(graph, self.engine.abstract_tree_ref)
                try:
                    t0 = time.perf_counter()
                    touched = self.engine.writer.ensure_sink_layout(graph, active_keys=set(graph.output_attr_keys))
                    self.engine.diagnostics._perf_add("proxy_write", time.perf_counter() - t0)
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
                    return False
                utils.log_throttled(
                    f"missing_keys:{graph_key}",
                    f"No output keys for source '{getattr(graph, 'source_uid', '')}'. Skipping write this frame.",
                    "WARNING",
                    min_interval=2.0,
                )
                return False
        return True

    def _sync_graph_schema(self, graph, graph_key, ctx):
        """Sync the sink layout after a compile (or on the fallback interval)."""
        geometry_proxy_names = ctx.geometry_proxy_names
        # Dirty-flag resync: sync immediately after compile, then fall back to
        # the interval only if the dirty set is somehow absent (defensive).
        schema_dirty = graph_key in self.engine._attr_schema_dirty
        if not schema_dirty:
            last_sync = self.engine._last_attr_sync_frame.get(graph_key, -10_000_000)
            schema_dirty = (self.engine.current_frame - last_sync) >= self._ATTR_RESYNC_INTERVAL_FRAMES
        if schema_dirty:
            try:
                t0 = time.perf_counter()
                touched = self.engine.writer.ensure_sink_layout(graph, active_keys=set(graph.output_attr_keys))
                self.engine.diagnostics._perf_add("proxy_write", time.perf_counter() - t0)
                if isinstance(touched, set):
                    geometry_proxy_names.update({str(v) for v in touched if str(v)})
            except Exception as e:
                utils.log_throttled(f"attr_ensure:{graph_key}", f"Attribute ensure failed for '{graph_key}': {e}", "ERROR", min_interval=2.0)
            self.engine._attr_schema_dirty.discard(graph_key)
            self.engine._last_attr_sync_frame[graph_key] = int(self.engine.current_frame)

    def _write_baked_results(self, graph, graph_key, ctx):
        """Write baked values when the bake covers this frame; True means done."""
        geometry_proxy_names = ctx.geometry_proxy_names
        baked_results = self.engine.bake._get_baked_frame_results(graph, self.engine.current_frame)
        if baked_results is not None:
            try:
                t0 = time.perf_counter()
                touched = self.engine.writer.commit_results(graph, baked_results, active_keys=set(graph.output_attr_keys))
                self.engine.diagnostics._perf_add("proxy_write", time.perf_counter() - t0)
                if isinstance(touched, set):
                    geometry_proxy_names.update({str(v) for v in touched if str(v)})
            except Exception as e:
                utils.log_throttled(f"attr_write_baked:{graph_key}", f"Baked attribute write failed for '{graph_key}': {e}", "ERROR", min_interval=2.0)
            return True
        return False

    def _graph_sources_ready(self, graph, graph_key):
        """False while a live source is not ready or a file source failed."""
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
            return False
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
            return False
        return True

    def _compute_graph_results(self, scene, graph, graph_key, source_tasks, ctx):
        """Submit worker jobs and collect values: worker results where usable, local DSP otherwise."""
        parallel_mode = ctx.parallel_mode
        parallel_pool_ready = ctx.parallel_pool_ready
        parallel_graph_key = str(getattr(graph, "parallel_graph_uid", "") or graph_key)
        parallel_offload_graph = (
            parallel_mode == "processes_dsp"
            and bool(getattr(graph, "parallel_ir", {}))
            and bool(getattr(graph, "parallel_offload_attr_keys", []))
        )
        parallel_submit_enabled = (
            parallel_offload_graph
            and parallel_pool_ready
            and parallel_graph_key not in self.engine.parallel.disabled_sources
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
                    src_chunk = source_task.get_output(self.engine.current_frame, 0)
                except Exception as exc:
                    utils.note_swallowed("audio_engine.ACEAudioEngine.execute_next_frame#9", exc)
                    src_chunk = None
                if hasattr(src_chunk, "mono") and hasattr(src_chunk, "sample_rate"):
                    source_chunks_for_worker[str(src_node_id)] = src_chunk
            if source_chunks_for_worker:
                shard_specs = list(getattr(graph, "parallel_output_shards", []) or [])
                if not shard_specs:
                    shard_specs = self.engine.parallel.shard_attr_keys(
                        list(getattr(graph, "parallel_offload_attr_keys", []) or [])
                    )
                    graph.parallel_output_shards = list(shard_specs)
                if not shard_specs:
                    parallel_submit_enabled = False
                else:
                    for shard_idx, shard_keys in enumerate(shard_specs):
                        if not shard_keys:
                            continue
                        self.engine.parallel.submit_graph_job(
                            scene=scene,
                            graph=graph,
                            graph_key=parallel_graph_key,
                            queue_key=f"{parallel_graph_key}:shard{int(shard_idx)}",
                            owner_graph_key=parallel_graph_key,
                            frame_index=int(self.engine.current_frame),
                            source_chunks=source_chunks_for_worker,
                            output_keys=list(shard_keys),
                            include_graph_ir=bool(parallel_graph_key in self.engine.parallel.ir_required),
                        )
            else:
                parallel_submit_enabled = False

        if parallel_offload_graph:
            # Stage B: asynchronous numeric DSP worker path (fresh -> stale -> zero).
            local_keys = list(getattr(graph, "parallel_local_attr_keys", []) or [])
            has_parallel_values = self.engine.parallel.has_usable_values(
                parallel_graph_key,
                int(self.engine.current_frame),
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
            results = self._collect_graph_results_for_frame(graph, self.engine.current_frame, attr_keys=local_keys)
            if has_parallel_values and parallel_submit_enabled:
                parallel_values = self.engine.parallel.values_for_frame(
                    graph=graph,
                    graph_key=parallel_graph_key,
                    frame_index=int(self.engine.current_frame),
                )
                results.update(parallel_values)
        else:
            results = self._collect_graph_results_for_frame(graph, self.engine.current_frame)
        return results

    def _collect_graph_preview(self, graph, ctx):
        """Gather this graph's preview audio (passthrough chunks or preview sinks)."""
        preview_block_size = ctx.preview_block_size
        preview_chunks = ctx.preview_chunks
        # Collect preview after the main graph/source evaluation so
        # FILE-source prefetch has the full frame interval to fill.
        self.engine.preview._start_preview_worker()  # idempotent
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
                raw_chunk = self.engine.preview._task_current_chunk_for_frame(src_task, self.engine.current_frame)
                if raw_chunk is None:
                    try:
                        t0 = time.perf_counter()
                        raw_chunk = src_task.get_output(self.engine.current_frame, int(src_sock_idx))
                        self.engine.diagnostics._perf_add("preview", time.perf_counter() - t0)
                    except Exception as exc:
                        utils.note_swallowed("audio_engine.ACEAudioEngine.execute_next_frame#10", exc)
                        raw_chunk = None
                if raw_chunk is not None and self.engine.preview._chunk_is_unmasked_audio(raw_chunk):
                    preview_chunks.append(
                        self.engine.preview._fit_preview_chunk_block(raw_chunk, preview_block_size)
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
                upstream_chunk = self.engine.preview._task_current_chunk_for_frame(
                    upstream_task, self.engine.current_frame
                )
                if upstream_chunk is None:
                    try:
                        t0 = time.perf_counter()
                        upstream_chunk = upstream_task.get_output(
                            self.engine.current_frame, sock_idx
                        )
                        self.engine.diagnostics._perf_add("preview", time.perf_counter() - t0)
                    except Exception as exc:
                        utils.note_swallowed("audio_engine.ACEAudioEngine.execute_next_frame#11", exc)
                        upstream_chunk = None
                if upstream_chunk is not None:
                    if self.engine.preview._submit_preview_work(preview_task, upstream_chunk):
                        # Worker owns the push; mark preview as active so
                        # the monitor is not cleared this frame.
                        ctx.preview_pushed = True

    def _commit_graph_results(self, graph, graph_key, results, ctx):
        """Stage C (main thread only): commit values into the resolved sinks."""
        geometry_proxy_names = ctx.geometry_proxy_names
        try:
            # Stage C (main thread only): commit numeric results into resolved sinks.
            t0 = time.perf_counter()
            touched = self.engine.writer.commit_results(graph, results, active_keys=set(graph.output_attr_keys))
            self.engine.diagnostics._perf_add("proxy_write", time.perf_counter() - t0)
            if isinstance(touched, set):
                geometry_proxy_names.update({str(v) for v in touched if str(v)})
        except Exception as e:
            utils.log_throttled(f"attr_write:{graph_key}", f"Attribute write failed for '{graph_key}': {e}", "ERROR", min_interval=2.0)

    def _finish_preview(self, ctx):
        """Mix and push the frame's preview audio, or clear the monitor."""
        preview_allowed = ctx.preview_allowed
        preview_eval_due = ctx.preview_eval_due
        preview_chunks = ctx.preview_chunks
        if preview_allowed:
            t0 = time.perf_counter()
            if preview_eval_due:
                mixed_preview = self.engine.preview._mix_preview_chunks(preview_chunks)
                self._preview_last_chunk = mixed_preview
                self._preview_last_eval_frame = int(self.engine.current_frame)
            else:
                mixed_preview = self._preview_last_chunk
            if mixed_preview is not None:
                with utils.logged_errors("audio_engine.ACEAudioEngine.execute_next_frame#12"):
                    self.engine.preview._preview_monitor.push_chunk(mixed_preview)
                    ctx.preview_pushed = True
            self.engine.diagnostics._perf_add("preview", time.perf_counter() - t0)
        else:
            self._preview_last_chunk = None
            self._preview_last_eval_frame = -1
            self._preview_next_eval_ts = 0.0

        if (not preview_allowed) or (not ctx.preview_pushed):
            with utils.logged_errors("audio_engine.ACEAudioEngine.execute_next_frame#13"):
                self.engine.preview._preview_monitor.clear()

    def adjust_graph_block_size_for_task(self, task: ACEAudioTask, new_block_size: int):
        """Update the block size for all tasks in the graph that owns *task*.

        SourceTask instances are updated via ``_set_block_size_from_engine``
        (which handles prefetch buffer resize).  All other tasks have their
        ``block_size``, ``silence_buffer``, and cached array attributes reset
        directly so stale workspace allocations are not reused.
        """
        target_graph = None
        for graph in self.engine.graphs.values():
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

    def _zero_value_for_attr(self, attr_key: str):
        info = self.engine.writer.output_defs.get(attr_key) if self.engine.writer else None
        attr_type = info.get("type", "FLOAT") if info else "FLOAT"
        if attr_type == "FLOAT_VECTOR":
            return (0.0, 0.0, 0.0)
        if attr_type == "FLOAT_COLOR":
            return (0.0, 0.0, 0.0, 1.0)
        return 0.0

    def _normalize_output_value(self, attr_key: str, value: Any):
        info = self.engine.writer.output_defs.get(attr_key) if self.engine.writer else None
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
                self.engine.diagnostics._dsp_profile_add(task, time.perf_counter() - t0)
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
        output_slots = self.engine.compiler._graph_output_slot_specs(graph, attr_keys=attr_keys)
        target_node_ids = sorted({node_id for _, node_id, _ in output_slots if node_id in graph.tasks})
        plan = self._graph_eval_plan_for_targets(graph, target_node_ids)
        if plan:
            t0_plan = time.perf_counter()
            self._evaluate_graph_plan_iterative(graph, frame_index, plan)
            self.engine.diagnostics._perf_add("dsp", time.perf_counter() - t0_plan)
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
                self.engine.diagnostics._perf_add("dsp", time.perf_counter() - t0)
            if val is None:
                continue
            results[attr_key] = self._normalize_output_value(attr_key, val)
        return results

    def _refresh_source_specs_task_registry(self) -> None:
        """Rebuild the lookup used to flush SourceSpecsTask main-thread ctx writes."""
        registry: Dict[tuple[str, float], SourceSpecsTask] = {}
        for graph in (self.engine.graphs or {}).values():
            for task in (getattr(graph, "tasks", {}) or {}).values():
                if not isinstance(task, SourceSpecsTask):
                    continue
                target = getattr(task, "_long_term_target", None)
                if target is None:
                    continue
                registry[target] = task
        self._source_specs_task_registry = registry

    @classmethod
    def _scheduler_mode(cls, scene_or_engine_settings=None) -> str:
        settings = engine_settings(scene_or_engine_settings)
        try:
            val = str(getattr(settings, "ace_scheduler_mode", "demand") or "demand").strip().lower()
            return val if val in {"demand", "legacy"} else "demand"
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._scheduler_mode", exc)
            return "demand"

    @classmethod
    def _timeslice_mode(cls, scene_or_engine_settings=None) -> str:
        settings = engine_settings(scene_or_engine_settings)
        try:
            val = str(getattr(settings, "ace_timeslice_mode", "fixed") or "fixed").strip().lower()
            return val if val in {"fixed", "adaptive"} else "fixed"
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._timeslice_mode", exc)
            return "fixed"
