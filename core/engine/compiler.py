"""Compile scene frontends into per-source runtime task graphs."""

from dataclasses import dataclass
import bpy
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set
from ... import utils
from ...nodes.dummy_tree import ACEDummyTreeReader
from ...properties import ACEAudioSource
from ..abstract_tree import ACEAbstractNode, ACEAbstractTree
from ..input.reception import build_live_flow_identifier
from ..tasks.base import ACEAudioTask
from ..tasks.chunk import SAMPLE_RATE
from ..tasks.delta import AttackReleaseSplitterTask, AutoComponentSplitterTask, ChromagramTask, FrequencySplitterTask, MidsideSplitterTask
from ..tasks.graph import ACEAudioGraph
from ..tasks.preview import PreviewSinkTask
from ..tasks.source import SourceTask
from ..tasks.specs import SourceSpecsTask
from ..tasks.stream import LUFSTask, OnsetTask, PeakTask, PitchTask, RMSTask, SpectralCentroidTask, SpectralFluxTask, StereoBalanceTask, StereoVectorTask, StereoWidthTask, ZeroCrossTask
from ..tasks.utility import AudioMathTask, RollingAvgTask, StreamAccumulatorTask
from ..id_management import ACEIdManager
from ..proxy.manager import ACEProxyManager
from .common import FRONTEND_KIND_GEOMETRY, FRONTEND_KIND_SHADER, _ACECompiledFrontendSnapshot, engine_settings


@dataclass
class _CompileContext:
    """Values shared by the per-source compile steps."""

    abstract_tree: Any
    blender_scene: Any
    fps: Any
    engine_settings: Any
    scheduler_mode: Any
    timeslice_mode: Any
    preview_block_size: Any
    single_source_mode: Any
    source_context_lookup: Any
    stream_accumulator_state: Any
    manual_preview_winner_id: Any
    has_manual_preview: Any
    sink_writes_allowed: Any


class GraphCompiler:
    """Compile scene frontends into per-source runtime task graphs."""

    _FILE_WORKING_SAMPLE_RATE_CAP = 48_000

    def __init__(self, engine):
        self.engine = engine
        self._last_compile_time: float = 0.0
        self._compiled_tree_ref: Optional[bpy.types.NodeTree] = None
        self._compiled_frontend_refs: Dict[str, Any] = {}
        self._compiled_frontend_snapshots: Dict[str, _ACECompiledFrontendSnapshot] = {}
        self._compiled_frontend_ptrs: Dict[str, int] = {}
        self._compiled_frontend_uids: tuple[str, ...] = ()
        self._last_frontend_membership_signature: tuple[Any, ...] = ()
        self._last_tree_signature: Optional[Any] = None
        self._compiled_tree_ptr: int = 0
        self._compiled_tree_frontend_uid: str = ""
        self._compiled_tree_bl_idname: str = ""

    def compile(self, abstract_tree: ACEAbstractTree, blender_scene):
        """Build one ACEAudioGraph per audio source from the abstract tree.

        Shuts down any existing graphs, resets the parallel generation counter,
        resolves sample rates and block sizes for each source, creates all task
        objects (SourceTask, stream tasks, PreviewSinkTask, etc.), wires their
        inputs according to the abstract tree links, prepares the parallel IR for
        worker offload, and ensures Blender proxy attributes exist.

        Should only be called from the main thread.  For tree-change-triggered
        recompiles, prefer ``compile_from_blender_tree()`` which also parses the
        GN node tree and syncs attribute IDs.

        Args:
            abstract_tree: Parsed representation of the Geometry Nodes tree.
            blender_scene: The active ``bpy.types.Scene`` object.
        """
        utils.assert_main_thread("compile")
        self.engine.scene_key = self.engine._scene_key(blender_scene)
        self.engine.scene_name = self.engine._scene_name(blender_scene)
        self.engine.frontends._pending_tree = None
        self.engine.frontends._pending_tree_ptr = 0
        self.engine.frontends._pending_tree_bl_idname = ""
        self.engine.frontends._pending_scene_rebuild = False
        stream_accumulator_state = self._export_stream_accumulator_state()
        self.engine._shutdown_graphs()
        self.engine.graphs.clear()
        self.engine._last_attr_sync_frame.clear()
        self.engine._attr_schema_dirty.clear()
        self.engine.frame._block_size_state.clear()
        self.engine.abstract_tree_ref = abstract_tree

        self.engine.writer = ACEProxyManager(abstract_tree)
        sink_writes_allowed = not self.engine._unsafe_for_handler_writes(blender_scene)
        tree_signature = self.engine.frontends._compute_tree_signature(abstract_tree)
        self._last_tree_signature = tree_signature
        if not hasattr(blender_scene, "audiocore"):
            return

        fps = 24.0
        if hasattr(blender_scene, "render"):
            raw_fps = blender_scene.render.fps
            base = blender_scene.render.fps_base
            if base > 0:
                fps = raw_fps / base
        engine_settings = getattr(getattr(blender_scene, "audiocore", None), "engine_settings", None)
        scheduler_mode = self.engine.frame._scheduler_mode(engine_settings)
        timeslice_mode = self.engine.frame._timeslice_mode(engine_settings)
        preview_block_size = self.engine.preview._preview_block_size(engine_settings)
        sources = list(getattr(getattr(blender_scene, "audiocore", None), "audio_sources", []) or [])
        single_source_mode = len(sources) <= 1
        self.engine.bake._prune_bake_cache(sources, tree_signature)

        self.engine.effective_sample_rate = self._collect_sample_rate_floor(
            sources,
            engine_settings,
            fps,
            allow_io=False,
        )
        fallback_sr = self.engine.effective_sample_rate

        source_runtime, source_context_lookup = self._resolve_sources(blender_scene, sources, fallback_sr, engine_settings, fps, timeslice_mode)
        manual_preview_winner_id, has_manual_preview = self._resolve_preview_winner(abstract_tree)
        ctx = _CompileContext(
            abstract_tree=abstract_tree,
            blender_scene=blender_scene,
            fps=fps,
            engine_settings=engine_settings,
            scheduler_mode=scheduler_mode,
            timeslice_mode=timeslice_mode,
            preview_block_size=preview_block_size,
            single_source_mode=single_source_mode,
            source_context_lookup=source_context_lookup,
            stream_accumulator_state=stream_accumulator_state,
            manual_preview_winner_id=manual_preview_winner_id,
            has_manual_preview=has_manual_preview,
            sink_writes_allowed=sink_writes_allowed,
        )
        for runtime in source_runtime:
            self._build_source_graph(ctx, runtime)

        if self.engine.writer is not None and sink_writes_allowed:
            self._prune_stale_consumer_properties()

        with utils.logged_errors("audio_engine.ACEAudioEngine.compile#7"):
            ACEProxyManager.ensure_sonar_setup(blender_scene, getattr(blender_scene.audiocore, "audio_sources", []))
        self._last_compile_time = time.time()

    def _export_stream_accumulator_state(self):
        """Stream Accumulator state of the current graphs, keyed by (source, node), to restore after the rebuild."""
        stream_accumulator_state: Dict[tuple[str, str], Dict[str, Any]] = {}
        for graph_key, graph in dict(getattr(self.engine, "graphs", {}) or {}).items():
            source_key = str(
                getattr(graph, "source_uid", "")
                or graph_key
                or getattr(graph, "proxy_name", "")
                or ""
            )
            if not source_key:
                continue
            for node_id, task in dict(getattr(graph, "tasks", {}) or {}).items():
                if not isinstance(task, StreamAccumulatorTask):
                    continue
                export_fn = getattr(task, "export_state", None)
                if not callable(export_fn):
                    continue
                try:
                    state = dict(export_fn() or {})
                except Exception as exc:
                    utils.note_swallowed("audio_engine.ACEAudioEngine.compile", exc)
                    state = {}
                if state:
                    stream_accumulator_state[(source_key, str(node_id))] = state
        return stream_accumulator_state

    def _resolve_sources(self, blender_scene, sources, fallback_sr, engine_settings, fps, timeslice_mode):
        """Proxy name, sample rate and block size per source, and a lookup by uid and proxy name."""
        existing_ids = [getattr(s, "stable_id", "") for s in getattr(blender_scene.audiocore, "audio_sources", []) if getattr(s, "stable_id", "")]
        source_runtime: list[dict[str, Any]] = []
        source_context_lookup: Dict[str, ACEAudioSource] = {}

        for source in sources:
            # Ensure a proxy exists and resolve the stable proxy name for filtering
            try:
                proxy_obj = ACEProxyManager.ensure_proxy_for_source(source, scene=blender_scene)
            except Exception as exc:
                utils.note_swallowed("audio_engine.ACEAudioEngine.compile#2", exc)
                proxy_obj = None
            src_uid = ACEIdManager.ensure_source_uid(source, existing_ids=existing_ids)
            if src_uid and src_uid not in existing_ids:
                existing_ids.append(src_uid)
            src_proxy_name = getattr(source, "proxy_object_name", "") or (proxy_obj.name if proxy_obj else "")
            if not src_proxy_name:
                src_proxy_name = ACEIdManager.generate_source_proxy_name(
                    getattr(source, "name", "") or src_uid or "Source",
                    stable_seed=src_uid or getattr(source, "name", "") or "source",
                )
                with utils.logged_errors("audio_engine.ACEAudioEngine.compile#3"):
                    source.proxy_object_name = src_proxy_name

            if src_uid:
                source_context_lookup[src_uid] = source
            if src_proxy_name:
                source_context_lookup[src_proxy_name] = source

            self.engine.probes._probe_source_metadata(source, allow_io=False)
            source_sr = self._resolve_source_sample_rate(source, fallback_sr, engine_settings)
            source_type = str(getattr(source, "source_type", "") or "").upper()
            block_size = self.engine.frame._resolve_block_size(
                source_sr,
                fps,
                source_type=source_type,
                timeslice_mode=timeslice_mode,
                delta_frames=1.0,
            )
            source_runtime.append({
                "source": source,
                "src_uid": src_uid,
                "src_proxy_name": src_proxy_name,
                "source_sr": source_sr,
                "source_type": source_type,
                "block_size": block_size,
            })
        return source_runtime, source_context_lookup

    def _resolve_preview_winner(self, abstract_tree):
        """The one manual Audio Preview node that plays, if any."""
        manual_preview_winner_id, manual_preview_candidates = self.engine.preview._resolve_global_manual_preview_winner(
            abstract_tree
        )
        has_manual_preview = bool(manual_preview_winner_id)
        if len(manual_preview_candidates) > 1 and manual_preview_winner_id:
            utils.log_throttled(
                "preview_conflict_resolution",
                (
                    "Multiple Audio Preview nodes are present. "
                    f"Using the most recent preview sink '{manual_preview_winner_id}' and muting the others."
                ),
                "WARNING",
                min_interval=2.0,
            )
        return manual_preview_winner_id, has_manual_preview

    def _build_source_graph(self, ctx, runtime):
        """Build, wire and register the runtime graph of one source."""
        required_node_ids = self._compute_required_node_ids_for_source(
            ctx.abstract_tree,
            runtime["src_uid"],
            runtime["src_proxy_name"],
            ctx.single_source_mode,
            scheduler_mode=ctx.scheduler_mode,
        )
        graph = self._create_graph_tasks(ctx, runtime, required_node_ids)
        self._wire_graph_links(ctx, graph, required_node_ids)
        self._assign_graph_outputs(ctx, runtime, graph)
        self._add_auto_preview_taps(ctx, runtime, graph)
        self._index_source_tasks(graph)
        self._register_graph(ctx, runtime, graph)

    def _create_graph_tasks(self, ctx, runtime, required_node_ids):
        """A graph for the source with one task per required node (accumulator state restored)."""
        abstract_tree = ctx.abstract_tree
        blender_scene = ctx.blender_scene
        fps = ctx.fps
        engine_settings = ctx.engine_settings
        scheduler_mode = ctx.scheduler_mode
        timeslice_mode = ctx.timeslice_mode
        source_context_lookup = ctx.source_context_lookup
        stream_accumulator_state = ctx.stream_accumulator_state
        source = runtime["source"]
        src_uid = runtime["src_uid"]
        src_proxy_name = runtime["src_proxy_name"]
        source_sr = runtime["source_sr"]
        source_type = runtime["source_type"]
        block_size = runtime["block_size"]
        graph = ACEAudioGraph(source, src_proxy_name)
        graph.source_uid = src_uid
        graph.block_size = block_size
        graph.scheduler_mode = scheduler_mode
        graph.timeslice_mode = timeslice_mode
        graph.proxy_lookup_fn = self.engine.sinks._lookup_proxy_object
        for node_id in sorted(required_node_ids):
            node = abstract_tree.nodes.get(node_id)
            if node is None:
                continue
            task = self._create_task_for_node(node, block_size)
            if task:
                graph.tasks[node_id] = task
                if isinstance(task, StreamAccumulatorTask):
                    restore_source_keys = [
                        str(src_uid or ""),
                        str(src_proxy_name or ""),
                    ]
                    for restore_source_key in restore_source_keys:
                        if not restore_source_key:
                            continue
                        state = stream_accumulator_state.get((restore_source_key, str(node_id)))
                        if not state:
                            continue
                        import_fn = getattr(task, "import_state", None)
                        if callable(import_fn):
                            with utils.logged_errors("audio_engine.ACEAudioEngine.compile#4"):
                                import_fn(state)
                        break
                task.engine_ref = self.engine
                task.undo_generation = int(getattr(self.engine, "undo_generation", 0) or 0)
                task.resolve_block_size_fn = self._block_size_resolver(fps, source_sr, source_type, graph)
                if isinstance(task, (SourceTask, SourceSpecsTask)):
                    node_ctx = self._resolve_source_context_for_node(
                        node,
                        source_context_lookup,
                        default_source=source,
                    )
                    task.source_context = node_ctx
                task.scene_fps = fps
                task.scene_frame_start = int(getattr(blender_scene, "frame_start", 1) or 1)
                task.engine_settings = engine_settings
        return graph

    def _wire_graph_links(self, ctx, graph, required_node_ids):
        """Connect the tasks along the abstract tree links."""
        abstract_tree = ctx.abstract_tree
        for link in abstract_tree.links:
            if link.source_node_id not in required_node_ids or link.target_node_id not in required_node_ids:
                continue
            source_task = graph.tasks.get(link.source_node_id)
            target_task = graph.tasks.get(link.target_node_id)

            if source_task and target_task:
                target_task.connect_input(
                    input_idx=link.target_socket_idx,
                    target_task=source_task,
                    target_output_idx=link.source_socket_idx
                )
        self.engine.frame._rebuild_graph_execution_order(graph)

    def _assign_graph_outputs(self, ctx, runtime, graph):
        """Output keys, sink bindings and manual preview of the graph."""
        abstract_tree = ctx.abstract_tree
        single_source_mode = ctx.single_source_mode
        manual_preview_winner_id = ctx.manual_preview_winner_id
        has_manual_preview = ctx.has_manual_preview
        src_uid = runtime["src_uid"]
        src_proxy_name = runtime["src_proxy_name"]
        filtered_keys = []
        for k in self.engine.writer.output_defs.keys():
            info = self.engine.writer.output_defs.get(k)
            node = abstract_tree.nodes.get(info.get("node_id")) if info else None
            if not node:
                continue
            if node.id not in graph.tasks:
                continue
            if not self._is_output_node(node):
                continue
            if not self._node_matches_source(node, src_uid, src_proxy_name, single_source_mode):
                continue
            filtered_keys.append(k)
        graph.output_attr_keys = filtered_keys
        graph.sink_bindings_by_attr = self.engine.sinks._resolve_sink_bindings_for_graph(
            graph,
            abstract_tree,
            source_uid=src_uid,
            proxy_name=src_proxy_name,
            single_source_mode=single_source_mode,
        )
        self._rebuild_graph_output_slot_specs(graph)
        graph.manual_preview_task_ids = []
        if has_manual_preview and manual_preview_winner_id in graph.tasks:
            graph.manual_preview_task_ids = [str(manual_preview_winner_id)]

    def _add_auto_preview_taps(self, ctx, runtime, graph):
        """Without a manual preview, tap Group Output audio links (passthrough or a preview sink)."""
        abstract_tree = ctx.abstract_tree
        blender_scene = ctx.blender_scene
        fps = ctx.fps
        engine_settings = ctx.engine_settings
        preview_block_size = ctx.preview_block_size
        single_source_mode = ctx.single_source_mode
        has_manual_preview = ctx.has_manual_preview
        src_uid = runtime["src_uid"]
        src_proxy_name = runtime["src_proxy_name"]
        source_sr = runtime["source_sr"]
        source_type = runtime["source_type"]
        auto_preview_task_ids: list[str] = []
        auto_preview_passthrough_specs: list[tuple[str, int]] = []
        if not has_manual_preview:
            auto_preview_specs = self._group_output_preview_specs_for_source(
                abstract_tree,
                src_uid,
                src_proxy_name,
                single_source_mode,
            )
            for upstream_node_id, upstream_sock_idx in auto_preview_specs:
                upstream_task = graph.tasks.get(upstream_node_id)
                if upstream_task is None:
                    continue
                upstream_node = abstract_tree.nodes.get(upstream_node_id)
                if (
                    isinstance(upstream_task, SourceTask)
                    and upstream_node is not None
                    and str(getattr(upstream_node, "category", "")) == "source_mngmt"
                    and str(getattr(upstream_node, "type", "")) == "src"
                ):
                    auto_preview_passthrough_specs.append((str(upstream_node_id), int(upstream_sock_idx)))
                    continue
                auto_task_id = f"__ace_auto_preview__{upstream_node_id}_{int(upstream_sock_idx)}"
                if auto_task_id in graph.tasks:
                    continue
                preview_task = PreviewSinkTask(auto_task_id, {"preview_enabled": True}, int(preview_block_size))
                preview_task.engine_ref = self.engine
                preview_task.resolve_block_size_fn = self._block_size_resolver(fps, source_sr, source_type, graph)
                preview_task.scene_fps = fps
                preview_task.scene_frame_start = int(getattr(blender_scene, "frame_start", 1) or 1)
                preview_task.engine_settings = engine_settings
                preview_task.connect_input(
                    input_idx=0,
                    target_task=upstream_task,
                    target_output_idx=int(upstream_sock_idx),
                )
                graph.tasks[auto_task_id] = preview_task
                auto_preview_task_ids.append(auto_task_id)

        graph.auto_preview_task_ids = sorted(auto_preview_task_ids)
        graph.auto_preview_passthrough_specs = sorted(set(auto_preview_passthrough_specs))
        # A single manual Audio Preview node wins globally; otherwise use per-graph auto taps.
        if has_manual_preview:
            graph.preview_task_ids = list(graph.manual_preview_task_ids)
        else:
            graph.preview_task_ids = list(graph.auto_preview_task_ids)

    def _index_source_tasks(self, graph):
        """Cache which tasks are sources, live and file."""
        source_task_ids: list[str] = []
        live_source_task_ids: list[str] = []
        file_source_task_ids: list[str] = []
        for node_id, task in graph.tasks.items():
            if not isinstance(task, SourceTask):
                continue
            source_task_ids.append(node_id)
            source_kind = str(getattr(getattr(task, "source_context", None), "source_type", "") or "").upper()
            if source_kind == "LIVE":
                live_source_task_ids.append(node_id)
            else:
                file_source_task_ids.append(node_id)
        graph.source_task_ids = source_task_ids
        graph.live_source_task_ids = live_source_task_ids
        graph.file_source_task_ids = file_source_task_ids

    def _register_graph(self, ctx, runtime, graph):
        """Build the worker IR, store the graph and lay out its sinks."""
        abstract_tree = ctx.abstract_tree
        sink_writes_allowed = ctx.sink_writes_allowed
        source = runtime["source"]
        src_uid = runtime["src_uid"]
        # prepare_graph_ir() reads BPM/beats-per-bar from this source
        # context, so it must be set before the IR is built.
        graph.source_context = source
        self.engine.parallel.prepare_graph_ir(graph, abstract_tree)
        self.engine.graphs[src_uid] = graph
        self.engine._attr_schema_dirty.add(src_uid)  # mark for first-frame schema sync

        if self.engine.writer is not None and sink_writes_allowed:
            with utils.logged_errors("audio_engine.ACEAudioEngine.compile#5"):
                self.engine.writer.ensure_sink_layout(graph, active_keys=set(graph.output_attr_keys))
                self.engine._last_attr_sync_frame[src_uid] = int(self.engine.current_frame)
        elif self.engine.writer is not None and self.engine._is_rendering_job():
            utils.log_throttled(
                "render_lock_interface_compile_stage_c",
                "Skipping ACE sink layout writes during render. Enable Render > Lock Interface for safe handler updates.",
                "WARNING",
                min_interval=5.0,
            )

    def _prune_stale_consumer_properties(self):
        """Remove ACE_* custom properties that no graph writes any more."""
        # Runs after every graph's sink bindings are resolved; the caller checks
        # that a writer exists and sink writes are allowed.
        with utils.logged_errors("audio_engine.ACEAudioEngine.compile#6"):
            active_props_by_object: Dict[str, Set[str]] = {}
            for g in self.engine.graphs.values():
                for bindings in getattr(g, "sink_bindings_by_attr", {}).values():
                    for binding in (bindings or []):
                        if str(getattr(binding, "sink_kind", "") or "") != "OBJECT_PROPERTY":
                            continue
                        obj_name = str(getattr(binding, "target_object_name", "") or "")
                        prop_name = str(getattr(binding, "target_data_name", "") or "")
                        if obj_name and prop_name:
                            active_props_by_object.setdefault(obj_name, set()).add(prop_name)
            self.engine.writer.prune_consumer_objects(active_props_by_object)

    def _block_size_resolver(self, fps, source_sr, source_type, graph):
        """Block-size callback for a task: follows the graph's timeslice mode and the frame delta.

        Tasks call it as (sample_rate, fps, source_type=..., timeslice_mode=...,
        delta_frames=...); the source type, the graph's timeslice mode and the
        engine's frame delta are the ones that count, so those hints are ignored.
        """
        eng = self.engine
        return lambda sr=None, fps_val=fps, **_caller_hints: eng.frame._resolve_block_size(
            sr or source_sr,
            fps_val,
            source_type=source_type,
            timeslice_mode=getattr(graph, "timeslice_mode", eng.frame._timeslice_mode()),
            delta_frames=float(getattr(eng, "_current_delta_frames", 1.0) or 1.0),
        )

    def compile_scene_frontends(self, scene, preferred_tree=None):
        """Discover, parse, merge, and compile all ACE frontends in the scene."""
        utils.assert_main_thread("compile_scene_frontends")
        if scene is None:
            return
        if self.engine._undo_recovery_active():
            self.engine.frontends.request_recompile(None)
            return

        frontends = self.engine.frontends.discover_scene_frontends(scene)
        membership_signature = self.engine.frontends._frontend_membership_signature(frontends)
        self._last_frontend_membership_signature = membership_signature
        has_shader_frontends = any(
            str(getattr(rec.frontend_ref, "kind", "") or "") == FRONTEND_KIND_SHADER
            for rec in (frontends or [])
        )
        if has_shader_frontends and not self.engine._structural_tree_writes_allowed(
            scene,
            tree=preferred_tree,
            frontend_kind=FRONTEND_KIND_SHADER,
        ):
            target_tree = preferred_tree
            if str(getattr(target_tree, "bl_idname", "") or "") != "ShaderNodeTree":
                target_tree = next(
                    (
                        rec.tree
                        for rec in (frontends or [])
                        if str(getattr(rec.frontend_ref, "kind", "") or "") == FRONTEND_KIND_SHADER
                    ),
                    None,
                )
            self.engine.frontends.request_recompile(target_tree)
            utils.log_throttled(
                "shader_tree_preview_deferred",
                "Deferring ACE shader-tree rebuild while a live material/rendered viewport preview is active.",
                "WARNING",
                min_interval=5.0,
            )
            return
        sources = list(getattr(getattr(scene, "audiocore", None), "audio_sources", []) or [])
        # Shader driver bindings need concrete proxy object names before the
        # frontend sync stage; ensure proxies are materialized up-front.
        for source in sources:
            with utils.logged_errors("audio_engine.ACEAudioEngine.compile_scene_frontends"):
                ACEProxyManager.ensure_proxy_for_source(source, scene=scene)
        reserved_node_ids: set[str] = set()
        merged_tree = ACEAbstractTree()

        for frontend in frontends:
            tree = frontend.tree
            frontend_uid = str(frontend.frontend_ref.uid or "")
            frontend_kind = str(frontend.frontend_ref.kind or "")
            if not tree or not frontend_uid:
                continue

            if frontend_kind == FRONTEND_KIND_GEOMETRY:
                with utils.logged_errors("audio_engine.ACEAudioEngine.compile_scene_frontends#2"):
                    from ...nodes.dummy_tree import ACEDummyNodeFactory
                    removed = ACEDummyNodeFactory.prune_orphaned_ace_nodes(tree, scene)
                    if removed:
                        utils.log(f"ACE: pruned {removed} orphaned source node(s) from tree '{tree.name}'", "INFO")

            with utils.logged_errors("audio_engine.ACEAudioEngine.compile_scene_frontends#3"):
                ACEProxyManager.sync_managed_source_tree(tree, sources, scene=scene)

            if frontend_kind == FRONTEND_KIND_SHADER:
                from ...nodes.shader_tree import ACEShaderTreeReader
                reader = ACEShaderTreeReader(
                    tree,
                    frontend_uid=frontend_uid,
                    frontend_kind=frontend_kind,
                    reserved_node_ids=reserved_node_ids,
                    consumer_keys=frontend.consumer_object_names,
                )
            else:
                reader = ACEDummyTreeReader(
                    tree,
                    frontend_uid=frontend_uid,
                    frontend_kind=frontend_kind,
                    reserved_node_ids=reserved_node_ids,
                    consumer_keys=frontend.consumer_object_names,
                )

            partial_tree = reader.build()
            changed = False
            changed_node_ids: set[str] = set()
            try:
                if frontend_kind == FRONTEND_KIND_SHADER:
                    changed, changed_node_ids = self.engine.sinks._sync_shader_output_ids(tree, partial_tree, scene)
                else:
                    changed, changed_node_ids = self.engine.sinks._sync_auto_attr_ids(tree, partial_tree, scene)
            except Exception as exc:
                utils.note_swallowed("audio_engine.ACEAudioEngine.compile_scene_frontends#4", exc)
                changed = False
                changed_node_ids = set()
            if changed:
                self.engine.sinks._refresh_abstract_attr_params_from_tree(partial_tree, tree, changed_node_ids)
            self._refresh_consumer_intents_from_nodes(partial_tree)
            partial_tree.frontends[frontend_uid] = frontend.frontend_ref
            merged_tree.merge_from(partial_tree)

        self.compile(merged_tree, scene)
        # compile() clears the runtime refs, this signature included; without it
        # every auto-compile poll sees a membership change and rebuilds.
        self._last_frontend_membership_signature = membership_signature
        self._compiled_frontend_refs.clear()
        self._compiled_frontend_snapshots = {
            rec.frontend_ref.uid: self.engine.frontends._compiled_snapshot_from_discovery(rec)
            for rec in frontends
        }
        self._compiled_frontend_ptrs = {
            rec.frontend_ref.uid: int(self.engine.frontends._node_tree_pointer(rec.tree))
            for rec in frontends
            if rec.tree is not None
        }
        self._compiled_frontend_uids = tuple(rec.frontend_ref.uid for rec in frontends)
        resolved_preferred_tree = preferred_tree or (frontends[0].tree if frontends else None)
        self._compiled_tree_ref = None
        self._compiled_tree_ptr = self.engine.frontends._node_tree_pointer(resolved_preferred_tree) if resolved_preferred_tree else 0
        self._compiled_tree_frontend_uid = self.engine.frontends._tree_frontend_uid(resolved_preferred_tree)
        self._compiled_tree_bl_idname = str(getattr(resolved_preferred_tree, "bl_idname", "") or "")
        self.engine.frontends._pending_scene_rebuild = False
        self.engine.frontends._dirty_frontend_uids.clear()

    def compile_from_blender_tree(
        self,
        bl_tree: bpy.types.NodeTree,
        scene,
    ):
        """Compatibility wrapper for scene-scoped frontend compilation.

        Public operators and older call sites still pass a specific Blender
        tree here. The actual compile path is now scene-wide:
          1. Discover all managed Geometry Nodes and Shader frontends.
          2. Parse and merge them into one ``ACEAbstractTree``.
          3. Compile one ``ACEAudioGraph`` per source.

        Must be called from the main thread (asserted via ``utils.assert_main_thread``).

        Args:
            bl_tree: Preferred frontend tree for compatibility bookkeeping.
            scene: The active ``bpy.types.Scene``.
        """
        utils.assert_main_thread("compile_from_blender_tree")
        if not scene:
            return
        self.compile_scene_frontends(scene, preferred_tree=bl_tree)
        self._compiled_tree_ref = None
        self._compiled_tree_ptr = self.engine.frontends._node_tree_pointer(bl_tree) if bl_tree else 0
        self._compiled_tree_frontend_uid = self.engine.frontends._tree_frontend_uid(bl_tree)
        self._compiled_tree_bl_idname = str(getattr(bl_tree, "bl_idname", "") or "")

    def _collect_sample_rate_floor(self, sources, engine_settings, fps: float, allow_io: bool = True) -> int:
        min_sr = None
        for source in sources or []:
            source_type = str(getattr(source, "source_type", "") or "").upper()
            if source_type == "FILE":
                try:
                    sr = int(getattr(source, "native_sample_rate", 0) or 0)
                except Exception as exc:
                    utils.note_swallowed("audio_engine.ACEAudioEngine._collect_sample_rate_floor", exc)
                    sr = 0
            else:
                sr = int(getattr(source, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
            cache_key = ""
            mtime = None
            with utils.logged_errors("audio_engine.ACEAudioEngine._collect_sample_rate_floor#2"):
                if source_type == "LIVE":
                    live_flow_id = build_live_flow_identifier(source)
                    with utils.logged_errors("audio_engine.ACEAudioEngine._collect_sample_rate_floor#3"):
                        source.flow_identifier = live_flow_id
                    # Compile path must not touch live devices/sockets.
                    live_transport = str(getattr(source, "live_transport", "") or "").upper()
                    udp_sr = int(getattr(source, "live_udp_sample_rate", 0) or 0)
                    if live_transport == "UDP" and udp_sr > 0:
                        sr = udp_sr
                    elif sr <= 0:
                        sr = SAMPLE_RATE
                    with utils.logged_errors("audio_engine.ACEAudioEngine._collect_sample_rate_floor#4"):
                        source.sample_rate = int(sr)
                    if min_sr is None or sr < min_sr:
                        min_sr = sr
                    continue

                # Fast path: trust existing metadata when available
                if source_type == "FILE" and sr > 0 and getattr(source, "duration", 0.0) > 0.0:
                    working_sr = self._cap_source_sample_rate(source_type, sr, engine_settings)
                    if min_sr is None or working_sr < min_sr:
                        min_sr = working_sr
                    continue
                if source_type == "FILE":
                    import os
                    cache_key = f"FILE:{getattr(source, 'filepath', '')}"
                    try:
                        mtime = os.path.getmtime(getattr(source, "filepath", ""))
                    except OSError:
                        mtime = None
                    cached = self.engine.probes._meta_cache.get(cache_key)
                    ts_ok = cached and (getattr(__import__("time"), "time")() - cached.get("ts", 0.0) <= self.engine.probes._meta_cache_ttl)
                    if cached and cached.get("mtime") == mtime and ts_ok:
                        sr = int(cached.get("native_sample_rate", cached.get("sample_rate", sr)) or sr)
                        with utils.logged_errors("audio_engine.ACEAudioEngine._collect_sample_rate_floor#5"):
                            source.native_sample_rate = int(sr)
                        if getattr(source, "duration", 0.0) == 0.0:
                            with utils.logged_errors("audio_engine.ACEAudioEngine._collect_sample_rate_floor#6"):
                                source.duration = float(cached.get("duration", 0.0))
                        working_sr = self._cap_source_sample_rate(source_type, sr, engine_settings)
                        if min_sr is None or working_sr < min_sr:
                            min_sr = working_sr
                        continue
                if source_type == "FILE":
                    # Dispatch a background probe instead of blocking the main thread.
                    # Results will be applied by _flush_probe_updates() on the next frame,
                    # which triggers a recompile so the correct sample-rate is picked up.
                    self.engine.probes._dispatch_async_probe(getattr(source, "filepath", ""))
                # Use current sr as fallback; updated on recompile after probe completes.
            working_sr = self._cap_source_sample_rate(source_type, sr, engine_settings)
            if min_sr is None or working_sr < min_sr:
                min_sr = working_sr

        if min_sr is None:
            min_sr = SAMPLE_RATE
        return int(min_sr)

    @classmethod
    def _resolve_source_sample_rate(cls, source, fallback_sr: int, scene_or_engine_settings=None) -> int:
        """
        Choose the best-known sample rate for a source, preferring its metadata and
        falling back to the scene-wide minimum if unavailable.
        """
        sr = cls._source_original_sample_rate(source)
        if sr <= 0:
            sr = int(fallback_sr)
        source_type = str(getattr(source, "source_type", "") or "").strip().upper()
        return cls._cap_source_sample_rate(source_type, sr, scene_or_engine_settings)

    @classmethod
    def _file_working_sample_rate_cap(cls, scene_or_engine_settings=None) -> int:
        settings = engine_settings(scene_or_engine_settings)
        default_cap = int(cls._FILE_WORKING_SAMPLE_RATE_CAP)
        try:
            raw = str(
                getattr(settings, "ace_file_working_sample_rate", str(default_cap)) or str(default_cap)
            ).strip().lower()
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._file_working_sample_rate_cap", exc)
            raw = str(default_cap)
        if raw in {"native", "off", "0"}:
            return 0
        try:
            return max(int(raw), 0)
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._file_working_sample_rate_cap#2", exc)
            return default_cap

    @classmethod
    def _cap_source_sample_rate(
        cls,
        source_type: str,
        sample_rate: int,
        scene_or_engine_settings=None,
    ) -> int:
        sr = max(int(sample_rate or SAMPLE_RATE), 1)
        if str(source_type or "").strip().upper() == "FILE":
            cap_sr = cls._file_working_sample_rate_cap(scene_or_engine_settings)
            if cap_sr > 0:
                sr = min(sr, int(cap_sr))
        return sr

    @staticmethod
    def _source_original_sample_rate(source) -> int:
        source_type = str(getattr(source, "source_type", "") or "").strip().upper()
        if source_type == "FILE":
            try:
                sr = int(getattr(source, "native_sample_rate", 0) or 0)
            except Exception as exc:
                utils.note_swallowed("audio_engine.ACEAudioEngine._source_original_sample_rate", exc)
                sr = 0
            if sr > 0:
                return sr
        try:
            return int(getattr(source, "sample_rate", 0) or 0)
        except Exception as exc:
            utils.note_swallowed("audio_engine.ACEAudioEngine._source_original_sample_rate#2", exc)
            return 0

    def _source_task_target_sample_rate(self, task: Optional[SourceTask]) -> Optional[int]:
        if not isinstance(task, SourceTask):
            return None
        ctx = getattr(task, "source_context", None)
        source_type = str(getattr(ctx, "source_type", "") or "").strip().upper()
        if source_type != "FILE":
            return None
        cap_sr = self._file_working_sample_rate_cap(getattr(task, "engine_settings", None))
        return int(cap_sr) if cap_sr > 0 else None

    @staticmethod
    def _refresh_consumer_intents_from_nodes(abstract_tree: ACEAbstractTree):
        for intent in getattr(abstract_tree, "consumer_intents", []) or []:
            node = abstract_tree.nodes.get(str(getattr(intent, "node_id", "") or ""))
            if node is None:
                continue
            socket_idx = int(getattr(intent, "socket_idx", 0) or 0)
            outputs = list(getattr(node, "outputs", []) or [])
            if 0 <= socket_idx < len(outputs):
                intent.output_name = str(getattr(outputs[socket_idx], "name", "") or "")
            params = dict(getattr(node, "parameters", {}) or {})
            explicit_name = str(params.get(f"attr_id_out_{socket_idx}", "") or "")
            intent.explicit_name = explicit_name

    @staticmethod
    def _is_output_node(node: ACEAbstractNode) -> bool:
        is_specs = bool(
            node
            and node.type in {'src_specs', 'util_rolling_avg', 'stream_accumulator'}
            and node.category == 'utilities'
        )
        return bool(
            node
            and (
                node.category == 'stream'
                or is_specs
                or (node.category == 'utilities' and node.type == 'preview_sink')
            )
        )

    @staticmethod
    def _node_matches_source(node: ACEAbstractNode, source_uid: str, proxy_name: str, single_source_mode: bool) -> bool:
        if single_source_mode:
            return True
        roots = {str(v) for v in set(getattr(node, "root_sources", set()) or set()) if str(v)}
        if not roots:
            return True
        source_aliases = {str(v) for v in (source_uid, proxy_name) if str(v)}
        if not source_aliases:
            return False
        return bool(roots.intersection(source_aliases))

    @staticmethod
    def _prune_ambiguous_required_node_ids(
        abstract_tree: ACEAbstractTree,
        required_node_ids: set[str],
        source_uid: str,
        proxy_name: str,
        single_source_mode: bool,
    ) -> set[str]:
        if single_source_mode:
            return set(required_node_ids)
        aliases = {str(v) for v in (source_uid, proxy_name) if str(v)}
        if not aliases:
            return set()
        reverse_adj: Dict[str, set[str]] = {}
        for link in getattr(abstract_tree, "links", []) or []:
            src_id = str(getattr(link, "source_node_id", "") or "")
            dst_id = str(getattr(link, "target_node_id", "") or "")
            if src_id and dst_id:
                reverse_adj.setdefault(dst_id, set()).add(src_id)

        ambiguous: set[str] = set()
        for node_id in set(required_node_ids):
            node = abstract_tree.nodes.get(node_id)
            if node is None:
                continue
            roots = {str(v) for v in set(getattr(node, "root_sources", set()) or set()) if str(v)}
            if not roots or roots.issubset(aliases) or not roots.intersection(aliases):
                continue
            upstream_roots: set[tuple[str, ...]] = set()
            for upstream_id in reverse_adj.get(str(node_id), set()):
                if upstream_id not in required_node_ids:
                    continue
                upstream_node = abstract_tree.nodes.get(str(upstream_id))
                if upstream_node is None:
                    continue
                uroots = sorted(
                    str(v)
                    for v in set(getattr(upstream_node, "root_sources", set()) or set())
                    if str(v)
                )
                if uroots:
                    upstream_roots.add(tuple(uroots))
            if len(upstream_roots) >= 2:
                ambiguous.add(str(node_id))

        if not ambiguous:
            return set(required_node_ids)

        pruned_adj: Dict[str, set[str]] = {}
        for link in getattr(abstract_tree, "links", []) or []:
            src_id = str(getattr(link, "source_node_id", "") or "")
            dst_id = str(getattr(link, "target_node_id", "") or "")
            if not src_id or not dst_id:
                continue
            if src_id in ambiguous or dst_id in ambiguous:
                continue
            if src_id not in required_node_ids or dst_id not in required_node_ids:
                continue
            pruned_adj.setdefault(dst_id, set()).add(src_id)

        seed_nodes = [
            node_id
            for node_id in required_node_ids
            if node_id not in ambiguous
        ]
        kept: set[str] = set()
        stack = list(seed_nodes)
        while stack:
            current = stack.pop()
            if current in kept:
                continue
            kept.add(current)
            for upstream in pruned_adj.get(current, set()):
                if upstream not in kept:
                    stack.append(upstream)
        return kept

    @staticmethod
    def _is_bundle_output_socket(node: Optional[ACEAbstractNode], socket_idx: int) -> bool:
        if node is None:
            return False
        outputs = getattr(node, "outputs", []) or []
        if socket_idx < 0 or socket_idx >= len(outputs):
            return False
        data_type = str(getattr(outputs[socket_idx], "data_type", "") or "")
        return "Bundle" in data_type

    def _group_output_preview_specs_for_source(
        self,
        abstract_tree: ACEAbstractTree,
        source_uid: str,
        proxy_name: str,
        single_source_mode: bool,
    ) -> list[tuple[str, int]]:
        specs: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for out_link in (getattr(abstract_tree, "group_output_links", []) or []):
            node_id = str(getattr(out_link, "source_node_id", "") or "")
            sock_idx = int(getattr(out_link, "source_socket_idx", 0) or 0)
            if not node_id:
                continue
            target_name = str(getattr(out_link, "target_socket_name", "") or "").strip().lower()
            target_idx = int(getattr(out_link, "target_socket_idx", 0) or 0)
            # Auto-preview only taps the intended GN Group Output "Audio" bundle.
            if target_name:
                if target_name != "audio":
                    continue
            elif target_idx != 1:
                continue
            node = abstract_tree.nodes.get(node_id)
            if node is None:
                continue
            if not self._is_bundle_output_socket(node, sock_idx):
                continue
            if not self._node_matches_source(node, source_uid, proxy_name, single_source_mode):
                continue
            key = (node_id, sock_idx)
            if key in seen:
                continue
            seen.add(key)
            specs.append(key)
        return specs

    def _compute_required_node_ids_for_source(
        self,
        abstract_tree: ACEAbstractTree,
        source_uid: str,
        proxy_name: str,
        single_source_mode: bool,
        scheduler_mode: str = "legacy",
    ) -> set[str]:
        if not abstract_tree:
            return set()

        seed_nodes: set[str] = set()
        for node in abstract_tree.nodes.values():
            if not self._is_output_node(node):
                continue
            if self._node_matches_source(node, source_uid, proxy_name, single_source_mode):
                seed_nodes.add(node.id)
        if str(scheduler_mode or "legacy").strip().lower() == "legacy":
            for node in abstract_tree.nodes.values():
                # Legacy mode keeps source nodes active each frame even when no
                # stream/spec branch requests them.
                if not (node.category == 'source_mngmt' and node.type == 'src'):
                    continue
                if self._node_matches_source(node, source_uid, proxy_name, single_source_mode):
                    seed_nodes.add(node.id)
        for node_id, _sock_idx in self._group_output_preview_specs_for_source(
            abstract_tree,
            source_uid,
            proxy_name,
            single_source_mode,
        ):
            seed_nodes.add(node_id)

        if not seed_nodes:
            return set()

        reverse_adj: Dict[str, set[str]] = {}
        for link in abstract_tree.links:
            reverse_adj.setdefault(link.target_node_id, set()).add(link.source_node_id)

        required: set[str] = set()
        stack = list(seed_nodes)
        while stack:
            current = stack.pop()
            if current in required:
                continue
            required.add(current)
            for upstream in reverse_adj.get(current, set()):
                if upstream not in required:
                    stack.append(upstream)
        return self._prune_ambiguous_required_node_ids(
            abstract_tree,
            required,
            source_uid=source_uid,
            proxy_name=proxy_name,
            single_source_mode=single_source_mode,
        )

    @staticmethod
    def _resolve_source_context_for_node(
        node: Optional[ACEAbstractNode],
        source_lookup: Dict[str, ACEAudioSource],
        default_source: Optional[ACEAudioSource] = None,
    ) -> Optional[ACEAudioSource]:
        if node is None:
            return default_source

        tokens: list[str] = []
        if getattr(node, "id", ""):
            tokens.append(str(node.id))
        for root in sorted(getattr(node, "root_sources", set()) or set()):
            tokens.append(str(root))

        for token in tokens:
            src = source_lookup.get(token)
            if src is not None:
                return src
        return default_source

    def _create_task_for_node(self, node: ACEAbstractNode, block_size: int) -> Optional[ACEAudioTask]:
        cat = node.category
        typ = node.type

        if cat == 'source_mngmt':
            if typ == 'src':
                return SourceTask(node.id, node.parameters, block_size)
        elif cat == 'utilities':
            if typ == 'src_specs':
                return SourceSpecsTask(node.id, node.parameters, block_size)
            if typ == 'preview_sink':
                return PreviewSinkTask(node.id, node.parameters, block_size)
            if typ == 'audio_math':
                return AudioMathTask(node.id, node.parameters, block_size)
            if typ == 'util_rolling_avg':
                return RollingAvgTask(node.id, node.parameters, block_size)
            if typ == 'stream_accumulator':
                return StreamAccumulatorTask(node.id, node.parameters, block_size)
        elif cat == 'delta':
            if typ == 'frequency_splitter':
                return FrequencySplitterTask(node.id, node.parameters, block_size)
            if typ == 'midside_splitter':
                return MidsideSplitterTask(node.id, node.parameters, block_size)
            if typ == 'attackrelease_splitter':
                return AttackReleaseSplitterTask(node.id, node.parameters, block_size)
            if typ == 'chromagram':
                return ChromagramTask(node.id, node.parameters, block_size)
            if typ == 'auto_component_splitter':
                return AutoComponentSplitterTask(node.id, node.parameters, block_size)
        elif cat == 'stream':
            if typ == 'rms':
                return RMSTask(node.id, node.parameters, block_size)
            if typ == 'lufs':
                return LUFSTask(node.id, node.parameters, block_size)
            if typ == 'peak':
                return PeakTask(node.id, node.parameters, block_size)
            if typ == 'pitch':
                return PitchTask(node.id, node.parameters, block_size)
            if typ == 'spectral_centroid':
                return SpectralCentroidTask(node.id, node.parameters, block_size)
            if typ == 'spectral_flux':
                return SpectralFluxTask(node.id, node.parameters, block_size)
            if typ == 'onset':
                return OnsetTask(node.id, node.parameters, block_size)
            if typ == 'zero_crossing_rate':
                return ZeroCrossTask(node.id, node.parameters, block_size)
            if typ == 'stereo_width':
                return StereoWidthTask(node.id, node.parameters, block_size)
            if typ == 'stereo_balance':
                return StereoBalanceTask(node.id, node.parameters, block_size)
            if typ == 'stereo_vector':
                return StereoVectorTask(node.id, node.parameters, block_size)

        return ACEAudioTask(node.id, node.parameters, block_size)

    def _rebuild_graph_output_slot_specs(self, graph: ACEAudioGraph):
        specs: list[tuple[str, str, int]] = []
        for attr_key in list(getattr(graph, "output_attr_keys", []) or []):
            node_id, _, sock_idx_str = str(attr_key).partition(":")
            if node_id not in graph.tasks:
                continue
            try:
                socket_idx = int(sock_idx_str) if sock_idx_str else 0
            except Exception as exc:
                utils.note_swallowed("audio_engine.ACEAudioEngine._rebuild_graph_output_slot_specs", exc)
                socket_idx = 0
            specs.append((str(attr_key), str(node_id), int(socket_idx)))
        graph.output_slot_specs = specs
        graph.output_slot_specs_by_subset = {}

    def _graph_output_slot_specs(
        self,
        graph: ACEAudioGraph,
        attr_keys: Optional[List[str]] = None,
    ) -> List[tuple[str, str, int]]:
        if attr_keys is None:
            specs = list(getattr(graph, "output_slot_specs", []) or [])
            if specs:
                return specs
            self._rebuild_graph_output_slot_specs(graph)
            return list(getattr(graph, "output_slot_specs", []) or [])

        keys = [str(v) for v in list(attr_keys or []) if str(v)]
        if not keys:
            return []
        key_tuple = tuple(keys)
        subset_cache = getattr(graph, "output_slot_specs_by_subset", None)
        if not isinstance(subset_cache, OrderedDict):
            subset_cache = OrderedDict()
            graph.output_slot_specs_by_subset = subset_cache
        cached = subset_cache.get(key_tuple)
        if isinstance(cached, list):
            # Move to end (most-recently-used) for true LRU eviction ordering.
            subset_cache.move_to_end(key_tuple)
            return cached

        full_specs = list(getattr(graph, "output_slot_specs", []) or [])
        if not full_specs:
            self._rebuild_graph_output_slot_specs(graph)
            full_specs = list(getattr(graph, "output_slot_specs", []) or [])
        full_map = {attr_key: (node_id, socket_idx) for attr_key, node_id, socket_idx in full_specs}

        subset: list[tuple[str, str, int]] = []
        for attr_key in keys:
            rec = full_map.get(attr_key)
            if rec is not None:
                subset.append((attr_key, str(rec[0]), int(rec[1])))
                continue
            node_id, _, sock_idx_str = str(attr_key).partition(":")
            if node_id not in graph.tasks:
                continue
            try:
                socket_idx = int(sock_idx_str) if sock_idx_str else 0
            except Exception as exc:
                utils.note_swallowed("audio_engine.ACEAudioEngine._graph_output_slot_specs", exc)
                socket_idx = 0
            subset.append((str(attr_key), str(node_id), int(socket_idx)))
        subset_cache[key_tuple] = subset
        # Evict the least-recently-used entry when the cache exceeds the limit.
        if len(subset_cache) > 32:
            subset_cache.popitem(last=False)
        return subset
