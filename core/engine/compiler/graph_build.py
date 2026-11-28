"""One source graph: tasks, links, outputs and preview taps."""

from collections import OrderedDict
from types import SimpleNamespace
from typing import List, Optional
from .... import utils
from ...abstract_tree import ACEAbstractNode, audio_links
from ...tasks.base import ACEAudioTask
from ...tasks.factory import create_task
from ...tasks.graph import ACEAudioGraph
from ...tasks.preview import PreviewSinkTask
from ...tasks.source import SourceTask
from ....vars import SOURCE_KIND
from .signals import graph_signature
from ..common import rna_snapshot


class GraphBuild:
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

    @staticmethod
    def _take_previous_task(ctx, src_uid, task_id, block_size):
        """The task the graph being replaced ran for this signal, with its state; None for a new one.

        A task whose block size moved since (adaptive time slices) starts again.
        """
        task = ctx.previous_tasks.pop((str(src_uid or ""), str(task_id)), None)
        if task is None or int(getattr(task, "block_size", 0) or 0) != int(block_size):
            if task is not None:
                ctx.previous_tasks[(str(src_uid or ""), str(task_id))] = task  # closed by the compiler
            return None
        task.inputs.clear()  # wired again from this compile's links
        return task

    def _create_graph_tasks(self, ctx, runtime, required_node_ids):
        """A graph for the source with one task per required signal: the previous task when it remains."""
        abstract_tree = ctx.abstract_tree
        blender_scene = ctx.blender_scene
        fps = ctx.fps
        engine_settings = ctx.engine_settings
        scheduler_mode = ctx.scheduler_mode
        timeslice_mode = ctx.timeslice_mode
        source_context_lookup = ctx.source_context_lookup
        source = runtime["source"]
        src_uid = runtime["src_uid"]
        src_proxy_name = runtime["src_proxy_name"]
        source_sr = runtime["source_sr"]
        source_type = runtime["source_type"]
        block_size = runtime["block_size"]
        graph = ACEAudioGraph(source, src_proxy_name)
        graph.source_uid = src_uid
        graph.block_size = block_size
        graph.sample_rate = int(source_sr or 0)
        graph.scheduler_mode = scheduler_mode
        graph.timeslice_mode = timeslice_mode
        graph.proxy_lookup_fn = self.engine.sinks._lookup_proxy_object
        for node_id in sorted(required_node_ids):
            node = abstract_tree.nodes.get(node_id)
            if node is None:
                continue
            task = self._take_previous_task(ctx, src_uid, node_id, block_size)
            if task is None:
                task = self._create_task_for_node(node, block_size)
            if task:
                graph.tasks[node_id] = task
                task.engine_ref = self.engine
                task.undo_generation = int(getattr(self.engine, "undo_generation", 0) or 0)
                task.resolve_block_size_fn = self._block_size_resolver(fps, source_sr, source_type, graph)
                if isinstance(task, SourceTask):
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
        for link in audio_links(abstract_tree):
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
                    and (str(getattr(upstream_node, "category", "")), str(getattr(upstream_node, "type", ""))) == SOURCE_KIND
                ):
                    auto_preview_passthrough_specs.append((str(upstream_node_id), int(upstream_sock_idx)))
                    continue
                auto_task_id = f"__ace_auto_preview__{upstream_node_id}_{int(upstream_sock_idx)}"
                if auto_task_id in graph.tasks:
                    continue
                preview_task = self._take_previous_task(ctx, src_uid, auto_task_id, int(preview_block_size))
                if preview_task is None:
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
        graph.signal_signature = graph_signature(graph)
        self.engine.parallel.prepare_graph_ir(graph, abstract_tree)
        self.engine.graphs[src_uid] = graph
        self.engine._attr_schema_dirty.add(src_uid)  # mark for first-frame schema sync

        if self.engine.writer is not None and sink_writes_allowed:
            with utils.logged_errors("graph_build.GraphBuild._register_graph"):
                self.engine.writer.ensure_sink_layout(graph, active_keys=set(graph.output_attr_keys))
                self.engine._last_attr_sync_frame[src_uid] = int(self.engine.current_frame)
        elif self.engine.writer is not None and self.engine._is_rendering_job():
            utils.log_throttled(
                "render_lock_interface_compile_stage_c",
                "Skipping ACE sink layout writes during render. Enable Render > Lock Interface for safe handler updates.",
                "WARNING",
                min_interval=5.0,
            )

    def fresh_graph_copy(self, graph: ACEAudioGraph) -> ACEAudioGraph:
        """A copy of *graph* with new tasks, wired the same way, for work off the live tasks.

        The bake streams through a copy: its result is then what a fresh playback computes,
        whatever state live playback left in the graph, and live playback carries on meanwhile.
        Sources and settings are snapshots, so the copy can run on another thread. Preview taps
        are left out.
        """
        settings_snapshot = rna_snapshot(getattr(next(iter(graph.tasks.values()), None), "engine_settings", None))
        source_snapshot = rna_snapshot(getattr(graph, "source_context", None))
        copy = ACEAudioGraph(source_snapshot, str(getattr(graph, "proxy_name", "") or ""))
        for attr in ("source_uid", "block_size", "sample_rate", "scheduler_mode", "timeslice_mode", "signal_signature"):
            setattr(copy, attr, getattr(graph, attr))
        copy.output_attr_keys = list(graph.output_attr_keys)
        node_of_task = {id(task): node_id for node_id, task in graph.tasks.items()}
        abstract_tree = getattr(self.engine, "abstract_tree_ref", None)
        snapshots: dict = {}
        for node_id, old in graph.tasks.items():
            if isinstance(old, PreviewSinkTask):
                continue
            node = abstract_tree.nodes.get(node_id) if abstract_tree is not None else None
            if node is None:
                continue
            spec = SimpleNamespace(
                id=node.id, category=node.category, type=node.type, parameters=dict(getattr(old, "parameters", {}) or {})
            )
            task = self._create_task_for_node(spec, int(getattr(old, "block_size", graph.block_size) or graph.block_size))
            if task is None:
                continue
            for attr in ("engine_ref", "undo_generation", "resolve_block_size_fn", "scene_fps", "scene_frame_start"):
                setattr(task, attr, getattr(old, attr, getattr(task, attr, None)))
            task.engine_settings = settings_snapshot
            if isinstance(task, SourceTask):
                ctx = getattr(old, "source_context", None)
                if ctx is not None and id(ctx) not in snapshots:
                    snapshots[id(ctx)] = source_snapshot if ctx is graph.source_context else rna_snapshot(ctx)
                task.source_context = snapshots.get(id(ctx)) if ctx is not None else None
                task.prefetch_allowed = False
            copy.tasks[node_id] = task
        for node_id, old in graph.tasks.items():
            task = copy.tasks.get(node_id)
            if task is None:
                continue
            for input_idx, conn in old.inputs.items():
                upstream = copy.tasks.get(node_of_task.get(id(conn.task), ""))
                if upstream is not None:
                    task.connect_input(input_idx=input_idx, target_task=upstream, target_output_idx=conn.socket_index)
        self._rebuild_graph_output_slot_specs(copy)
        self._index_source_tasks(copy)
        self.engine.frame._rebuild_graph_execution_order(copy)
        return copy

    def _create_task_for_node(self, node: ACEAbstractNode, block_size: int) -> Optional[ACEAudioTask]:
        return create_task(node.id, node.category, node.type, node.parameters, block_size)

    def _rebuild_graph_output_slot_specs(self, graph: ACEAudioGraph):
        specs: list[tuple[str, str, int]] = []
        for attr_key in list(getattr(graph, "output_attr_keys", []) or []):
            node_id, _, sock_idx_str = str(attr_key).partition(":")
            if node_id not in graph.tasks:
                continue
            try:
                socket_idx = int(sock_idx_str) if sock_idx_str else 0
            except Exception as exc:
                utils.note_swallowed("graph_build.GraphBuild._rebuild_graph_output_slot_specs", exc)
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
                utils.note_swallowed("graph_build.GraphBuild._graph_output_slot_specs", exc)
                socket_idx = 0
            subset.append((str(attr_key), str(node_id), int(socket_idx)))
        subset_cache[key_tuple] = subset
        # Evict the least-recently-used entry when the cache exceeds the limit.
        if len(subset_cache) > 32:
            subset_cache.popitem(last=False)
        return subset
