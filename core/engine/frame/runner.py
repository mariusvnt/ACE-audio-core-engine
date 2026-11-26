"""FrameRunner: evaluates every source graph for the current frame and commits the results."""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional
from .... import utils
from ...tasks.chunk import AudioChunk, SAMPLE_RATE
from ...tasks.preview import PreviewSinkTask
from ...tasks.source import SourceTask
from .block_size import BlockSizing
from .evaluation import GraphEvaluation


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
class FrameRunner(BlockSizing, GraphEvaluation):
    """Per-frame execution: block sizing, graph evaluation, preview and sink writes."""

    # Attribute schema is synced via a dirty flag set on compile, not by frame polling.
    # The fallback interval applies only when the dirty set is missing (defensive).
    _ATTR_RESYNC_INTERVAL_FRAMES = 120
    _BLOCK_SIZE_HYSTERESIS_FRAMES = 2
    _BLOCK_SIZE_IMMEDIATE_DELTA = 1024

    def __init__(self, engine):
        self.engine = engine
        self._block_size_state: Dict[str, Dict[str, int]] = {}
        self._preview_last_chunk: Optional[AudioChunk] = None
        self._preview_last_eval_frame: int = -1
        self._preview_next_eval_ts: float = 0.0

    def execute_next_frame(self, scene, force: bool = False):
        """Evaluate all graphs for the current scene frame and write results to their sinks."""
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
        # Flush async bake progress and completions.
        self.engine.bake.flush_bake_results()
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
            utils.note_swallowed("runner.FrameRunner._start_frame", exc)
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
            utils.note_swallowed("runner.FrameRunner._start_frame#2", exc)
            fps = 24.0
        self.engine._current_scene_fps = max(float(fps), 1.0)
        return float(fps), float(delta_frames)

    def _sync_runtime_inputs(self, scene) -> bool:
        """Refresh BPM in worker IRs, flush probe results, run a pending compile."""
        # Tempo and fps reach the workers without a recompile.
        with utils.logged_errors("runner.FrameRunner._sync_runtime_inputs"):
            self.engine.parallel.refresh_specs_params()

        with utils.logged_errors("runner.FrameRunner._sync_runtime_inputs#driven"):
            from ....nodes.split_freqs import refresh_driven_splits
            refresh_driven_splits(self.engine.graphs.values(), scene)

        # Flush background probe results before the compile so native_sample_rate
        # is populated for any sources whose probe thread has just completed.
        with utils.logged_errors("runner.FrameRunner._sync_runtime_inputs#2"):
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
            # Skip writes on the compile frame so Geometry Nodes evaluates the new graph once first.
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
        with utils.logged_errors("runner.FrameRunner._frame_context"):
            self.engine.preview._preview_monitor.update_scene_fps(fps)

        # A jump or a step back invalidates buffered monitor audio.
        with utils.logged_errors("runner.FrameRunner._frame_context#2"):
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
        with utils.logged_errors("runner.FrameRunner._frame_context#3"):
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
                    utils.note_swallowed("runner.FrameRunner._prepare_graph", exc)
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
                self.engine.latest_values.update(baked_results)
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
                    utils.note_swallowed("runner.FrameRunner._compute_graph_results", exc)
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
                # Without usable worker values, offloaded outputs are computed locally.
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
            # Prefer this frame's source chunk; evaluating late gives the prefetcher time to fill.
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
                        utils.note_swallowed("runner.FrameRunner._collect_graph_preview", exc)
                        raw_chunk = None
                if raw_chunk is not None and self.engine.preview._chunk_is_unmasked_audio(raw_chunk):
                    # The whole frame of audio is pushed; the monitor's feeder paces it.
                    preview_chunks.append(raw_chunk)
        else:
            # Preview sink chains run on the preview thread.
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
                        utils.note_swallowed("runner.FrameRunner._collect_graph_preview#2", exc)
                        upstream_chunk = None
                if upstream_chunk is not None:
                    if self.engine.preview._submit_preview_work(preview_task, upstream_chunk):
                        # Worker owns the push; mark preview as active so
                        # the monitor is not cleared this frame.
                        ctx.preview_pushed = True

    def _commit_graph_results(self, graph, graph_key, results, ctx):
        geometry_proxy_names = ctx.geometry_proxy_names
        try:
            # Stage C (main thread only): commit numeric results into resolved sinks.
            t0 = time.perf_counter()
            self.engine.latest_values.update(results)
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
                with utils.logged_errors("runner.FrameRunner._finish_preview"):
                    self.engine.preview._preview_monitor.push_chunk(mixed_preview)
                    ctx.preview_pushed = True
            self.engine.diagnostics._perf_add("preview", time.perf_counter() - t0)
        else:
            self._preview_last_chunk = None
            self._preview_last_eval_frame = -1
            self._preview_next_eval_ts = 0.0

        if (not preview_allowed) or (not ctx.preview_pushed):
            with utils.logged_errors("runner.FrameRunner._finish_preview#2"):
                self.engine.preview._preview_monitor.clear()
