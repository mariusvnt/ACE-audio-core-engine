"""One source graph: tasks, links, outputs and preview taps."""

from collections import OrderedDict
from typing import List, Optional
from .... import utils
from ...abstract_tree import ACEAbstractNode
from ...tasks.base import ACEAudioTask
from ...tasks.delta import AttackReleaseSplitterTask, AutoComponentSplitterTask, ChromagramTask, FrequencySplitterTask, MidsideSplitterTask
from ...tasks.graph import ACEAudioGraph
from ...tasks.preview import PreviewSinkTask
from ...tasks.source import SourceTask
from ...tasks.specs import SourceSpecsTask
from ...tasks.stream import LUFSTask, OnsetTask, PeakTask, PitchTask, RMSTask, SpectralCentroidTask, SpectralFluxTask, StereoBalanceTask, StereoVectorTask, StereoWidthTask, ZeroCrossTask
from ...tasks.utility import AudioMathTask, RollingAvgTask, StreamAccumulatorTask


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
