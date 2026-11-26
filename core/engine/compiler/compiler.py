"""GraphCompiler: scene frontends to one runtime graph per source."""

from dataclasses import dataclass
import bpy
import time
from typing import Any, Dict, Optional, Set
from .... import utils
from ....nodes.geometry.reader import ACEDummyTreeReader
from ....properties import ACEAudioSource
from ...abstract_tree import ACEAbstractTree
from ...tasks.utility import StreamAccumulatorTask
from ...id_management import ACEIdManager
from ...proxy.manager import ACEProxyManager
from ....nodes.source_nodes import refresh_source_nodes
from ..common import FRONTEND_KIND_GEOMETRY, FRONTEND_KIND_SHADER, _ACECompiledFrontendSnapshot
from .graph_build import GraphBuild
from .required_nodes import RequiredNodes
from .sample_rates import SampleRates


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
class GraphCompiler(SampleRates, RequiredNodes, GraphBuild):
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
        """Build one ACEAudioGraph per audio source from the abstract tree."""
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
                    utils.note_swallowed("compiler.GraphCompiler._export_stream_accumulator_state", exc)
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
            # Only a source some node reads has a proxy; the others keep a name for filtering.
            proxy_obj = bpy.data.objects.get(str(getattr(source, "proxy_object_name", "") or ""))
            src_uid = ACEIdManager.ensure_source_uid(source, existing_ids=existing_ids)
            if src_uid and src_uid not in existing_ids:
                existing_ids.append(src_uid)
            src_proxy_name = getattr(source, "proxy_object_name", "") or (proxy_obj.name if proxy_obj else "")
            if not src_proxy_name:
                src_proxy_name = ACEIdManager.generate_source_proxy_name(
                    getattr(source, "name", "") or src_uid or "Source",
                    stable_seed=src_uid or getattr(source, "name", "") or "source",
                )

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

    def _prune_stale_consumer_properties(self):
        """Remove ACE_* custom properties that no graph writes any more."""
        # Runs after every graph's sink bindings are resolved; the caller checks
        # that a writer exists and sink writes are allowed.
        with utils.logged_errors("compiler.GraphCompiler._prune_stale_consumer_properties"):
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
        # The sources the user's Source nodes read get a proxy object before the frontends are
        # read (shader driver bindings need its name); the others get nothing.
        with utils.logged_errors("compiler.GraphCompiler.compile_scene_frontends"):
            refresh_source_nodes([record.tree for record in frontends if record.tree], scene)
        reserved_node_ids: set[str] = set()
        merged_tree = ACEAbstractTree()

        for frontend in frontends:
            tree = frontend.tree
            frontend_uid = str(frontend.frontend_ref.uid or "")
            frontend_kind = str(frontend.frontend_ref.kind or "")
            if not tree or not frontend_uid:
                continue

            if frontend_kind == FRONTEND_KIND_GEOMETRY:
                with utils.logged_errors("compiler.GraphCompiler.compile_scene_frontends#2"):
                    from ....nodes.geometry.factory import ACEDummyNodeFactory
                    removed = ACEDummyNodeFactory.prune_orphaned_ace_nodes(tree, scene)
                    if removed:
                        utils.log(f"ACE: pruned {removed} orphaned source node(s) from tree '{tree.name}'", "INFO")

            if frontend_kind == FRONTEND_KIND_SHADER:
                from ....nodes.shader.reader import ACEShaderTreeReader
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
                utils.note_swallowed("compiler.GraphCompiler.compile_scene_frontends#4", exc)
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
        # After this compile's own writes (attribute ids, proxy sockets), so they don't count as edits.
        self.engine.frontends.record_compiled_content(frontends)

    def compile_from_blender_tree(
        self,
        bl_tree: bpy.types.NodeTree,
        scene,
    ):
        """Compatibility wrapper for scene-scoped frontend compilation."""
        utils.assert_main_thread("compile_from_blender_tree")
        if not scene:
            return
        self.compile_scene_frontends(scene, preferred_tree=bl_tree)
        self._compiled_tree_ref = None
        self._compiled_tree_ptr = self.engine.frontends._node_tree_pointer(bl_tree) if bl_tree else 0
        self._compiled_tree_frontend_uid = self.engine.frontends._tree_frontend_uid(bl_tree)
        self._compiled_tree_bl_idname = str(getattr(bl_tree, "bl_idname", "") or "")
