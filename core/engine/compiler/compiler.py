"""GraphCompiler: scene frontends to one runtime graph per source."""

from dataclasses import dataclass
import bpy
import time
from typing import Any, Dict, Optional, Set
from .... import utils
from ....nodes.geometry.reader import ACEDummyTreeReader
from ....properties import ACEAudioSource
from ...abstract_tree import ACEAbstractTree
from ...id_management import ACEIdManager
from ...proxy.manager import ACEProxyManager
from ..common import FRONTEND_KIND_SHADER, _ACECompiledFrontendSnapshot, close_task
from .graph_build import GraphBuild
from .required_nodes import RequiredNodes
from .sample_rates import SampleRates
from .signals import build_signals, resolve_sink_conflicts


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
    previous_tasks: Any  # {(source uid, task id): task} of the graphs being replaced
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
        """Build one ACEAudioGraph per audio source from the instance tree.

        A compile is a difference: the uses are merged into signals (``signals.py``), a task
        whose signal remains is kept with its state, new signals get new tasks, and the
        tasks of signals gone are closed. A source whose signals are unchanged keeps its
        worker graph and its bake.
        """
        utils.assert_main_thread("compile")
        self.engine.scene_key = self.engine._scene_key(blender_scene)
        self.engine.scene_name = self.engine._scene_name(blender_scene)
        self.engine.frontends._pending_tree = None
        self.engine.frontends._pending_tree_ptr = 0
        self.engine.frontends._pending_tree_bl_idname = ""
        self.engine.frontends._pending_scene_rebuild = False
        previous_graphs = self.engine._shutdown_graphs(keep_tasks=True)
        previous_tasks = {
            (str(getattr(graph, "source_uid", "") or key), str(task_id)): task
            for key, graph in previous_graphs.items()
            for task_id, task in graph.tasks.items()
        }
        try:
            self._compile_signals(abstract_tree, blender_scene, previous_tasks)
        finally:
            kept = {id(task) for graph in self.engine.graphs.values() for task in graph.tasks.values()}
            for task in previous_tasks.values():
                if id(task) not in kept:
                    close_task(task)
            self.engine.parallel.forget_graphs({str(g.parallel_graph_uid) for g in self.engine.graphs.values()})

    def _compile_signals(self, instance_tree: ACEAbstractTree, blender_scene, previous_tasks):
        self.engine.graphs.clear()
        self.engine._last_attr_sync_frame.clear()
        self.engine._attr_schema_dirty.clear()
        self.engine.frame._block_size_state.clear()
        self.engine.instance_tree_ref = instance_tree
        self.engine.abstract_tree_ref = instance_tree

        self.engine.writer = ACEProxyManager(instance_tree)
        sink_writes_allowed = not self.engine._unsafe_for_handler_writes(blender_scene)
        tree_signature = self.engine.frontends._compute_tree_signature(instance_tree)
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

        self.engine.effective_sample_rate = self._collect_sample_rate_floor(
            sources,
            engine_settings,
            fps,
            allow_io=False,
        )
        fallback_sr = self.engine.effective_sample_rate

        source_runtime, source_context_lookup = self._resolve_sources(blender_scene, sources, fallback_sr, engine_settings, fps, timeslice_mode)
        source_keys = {
            str(rt["src_uid"]): (int(rt["source_sr"]), int(rt["block_size"]), round(float(fps), 6))
            for rt in source_runtime
            if rt["src_uid"]
        }
        signal_set = build_signals(instance_tree, source_keys)
        abstract_tree = signal_set.tree
        self.engine.signal_of = dict(signal_set.signal_of)
        self.engine.abstract_tree_ref = abstract_tree
        self.engine.writer = ACEProxyManager(abstract_tree)
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
            previous_tasks=previous_tasks,
            manual_preview_winner_id=manual_preview_winner_id,
            has_manual_preview=has_manual_preview,
            sink_writes_allowed=sink_writes_allowed,
        )
        for runtime in source_runtime:
            self._build_source_graph(ctx, runtime)

        self.engine.sink_conflicts = resolve_sink_conflicts(self.engine.graphs)
        for conflict in self.engine.sink_conflicts:
            utils.log_throttled(
                f"sink_conflict:{conflict['sink_name']}",
                f"ACE: sink '{conflict['sink_name']}' is written by uses computing different values; "
                f"'{conflict['kept']}' keeps it, {', '.join(conflict['dropped'])} do not write it.",
                "WARNING",
                min_interval=5.0,
            )
        if self.engine.writer is not None and sink_writes_allowed:
            self._prune_stale_scene_properties(blender_scene, sources)
            # The bakes, laid out onto what the frontends read (core/bake_store.py).
            self.engine.bake.after_compile(sources, blender_scene)

        self._last_compile_time = time.time()

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
            if proxy_obj is not None and src_uid:
                ACEProxyManager._mark_proxy(proxy_obj, src_uid)
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

    def _prune_stale_scene_properties(self, scene, sources):
        """Remove the ACE_* scene properties that no shader sink writes any more."""
        # Runs after every graph's sink bindings are resolved; the caller checks
        # that a writer exists and sink writes are allowed.
        from ....nodes.identity import source_tempo_props

        with utils.logged_errors("compiler.GraphCompiler._prune_stale_scene_properties"):
            active: Set[str] = set()
            for g in self.engine.graphs.values():
                for bindings in getattr(g, "sink_bindings_by_attr", {}).values():
                    for binding in (bindings or []):
                        if str(getattr(binding, "sink_kind", "") or "") == "SCENE_PROPERTY":
                            active.add(str(getattr(binding, "target_data_name", "") or ""))
            keep = {name for s in sources for name in source_tempo_props(str(getattr(s, "stable_id", "") or ""))}
            from ....nodes.shader.mirror import controller_sink_names

            keep |= controller_sink_names(scene)  # written by mirror.read_back, read by shader controllers
            self.engine.writer.prune_scene_properties(scene, active, keep=keep)

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

        # Source nodes find their source through the id their proxy carries: mark the proxies
        # before reading (a proxy made by an earlier version, or renamed, has none yet).
        for source in getattr(getattr(scene, "audiocore", None), "audio_sources", []) or []:
            proxy = bpy.data.objects.get(str(getattr(source, "proxy_object_name", "") or ""))
            if proxy is not None and getattr(source, "stable_id", ""):
                with utils.logged_errors("compiler.GraphCompiler.compile_scene_frontends#mark"):
                    ACEProxyManager._mark_proxy(proxy, source.stable_id)
        frontends = self.engine.frontends.discover_scene_frontends(scene)
        membership_signature = self.engine.frontends._frontend_membership_signature(frontends)
        self._last_frontend_membership_signature = membership_signature
        # Reading writes nothing into the user's trees. The one exception: sink names that
        # are missing or claimed twice are repaired once, before reading.
        with utils.logged_errors("compiler.GraphCompiler.compile_scene_frontends#placement"):
            from ....nodes.placement import place_new_nodes

            for record in frontends:
                place_new_nodes(record.tree, scene)
        with utils.logged_errors("compiler.GraphCompiler.compile_scene_frontends"):
            self.engine.sinks.repair_sink_names([record.tree for record in frontends], scene)
        merged_tree = ACEAbstractTree()
        ignored_links: dict = {}
        # Controllers in shaders: their Geometry Nodes copy, before the sinks are bound (it says
        # which shader measurements it reads from proxies).
        with utils.logged_errors("compiler.GraphCompiler.compile_scene_frontends#mirror"):
            from ....nodes.shader import mirror

            for key, items in mirror.sync(scene, self.engine).items():
                ignored_links.setdefault(key, []).extend(items)

        for frontend in frontends:
            tree = frontend.tree
            frontend_uid = str(frontend.frontend_ref.uid or "")
            frontend_kind = str(frontend.frontend_ref.kind or "")
            if not tree or not frontend_uid:
                continue

            if frontend_kind == FRONTEND_KIND_SHADER:
                from ....nodes.shader.reader import ACEShaderTreeReader
                reader = ACEShaderTreeReader(
                    tree,
                    frontend_uid=frontend_uid,
                    frontend_kind=frontend_kind,
                    consumer_keys=frontend.consumer_object_names,
                    bindings=frontend.bindings,
                    root_key=frontend.root_key,
                    modifier=frontend.modifier,
                )
            else:
                reader = ACEDummyTreeReader(
                    tree,
                    frontend_uid=frontend_uid,
                    frontend_kind=frontend_kind,
                    consumer_keys=frontend.consumer_object_names,
                    bindings=frontend.bindings,
                    root_key=frontend.root_key,
                    modifier=frontend.modifier,
                )

            partial_tree = reader.build()
            for key, items in reader.ignored_links.items():
                ignored_links.setdefault(key, []).extend(items)
            self._refresh_consumer_intents_from_nodes(partial_tree)
            partial_tree.frontends[frontend_uid] = frontend.frontend_ref
            merged_tree.merge_from(partial_tree)

        self.compile(merged_tree, scene)
        # The links ACE ignores are drawn as invalid: a flag, not content, so no edit is seen.
        self.engine.ignored_links = ignored_links
        with utils.logged_errors("compiler.GraphCompiler.compile_scene_frontends#ignored"):
            from ....nodes.link_rules import mark_ignored_links

            mark_ignored_links(ignored_links)
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
        # After the name repair, so its writes don't count as edits.
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
