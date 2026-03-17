import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Map legacy flat filenames to their new sub-package paths so every
# _read("foo.py") call in this file keeps working without modification.
_SUBPACKAGE_PATHS = {
    "audio_engine.py":            "core/audio_engine.py",
    "audio_influx.py":            "core/audio_influx.py",
    "audio_tasks_registry.py":    "core/audio_tasks_registry.py",
    "abstract_tree.py":           "core/abstract_tree.py",
    "id_management.py":           "core/id_management.py",
    "parallel_dsp.py":            "core/parallel_dsp.py",
    "proxy_management.py":        "core/proxy_management.py",
    "dummy_tree.py":              "nodes/dummy_tree.py",
    "shader_tree.py":             "nodes/shader_tree.py",
    "dynamic_audio_operators.py": "nodes/dynamic_audio_operators.py",
    "gn_menu.py":                 "ui/gn_menu.py",
    "n_panel_operators.py":       "ui/n_panel_operators.py",
    "n_panel_ui.py":              "ui/n_panel_ui.py",
    "prefs_library_manager.py":   "ui/prefs_library_manager.py",
    "prefs_panel_ui.py":          "ui/prefs_panel_ui.py",
}


def _read(name: str) -> str:
    resolved = _SUBPACKAGE_PATHS.get(name, name)
    return (ROOT / resolved).read_text(encoding="utf-8")


def _slice_between(src: str, start_marker: str, end_marker: str, *, label: str = "") -> str:
    start_idx = src.find(start_marker)
    if start_idx < 0:
        raise AssertionError(f"Missing start marker '{start_marker}'" + (f" for {label}" if label else ""))
    start_idx += len(start_marker)
    end_idx = src.find(end_marker, start_idx)
    if end_idx < 0:
        raise AssertionError(f"Missing end marker '{end_marker}'" + (f" for {label}" if label else ""))
    return src[start_idx:end_idx]


class TestRegressionFindings(unittest.TestCase):
    def test_handlers_are_persistent(self):
        src = _read("audio_engine.py")
        self.assertRegex(src, r"@persistent\s+def frame_change_handler\(")
        self.assertRegex(src, r"@persistent\s+def depsgraph_update_handler\(")
        self.assertRegex(src, r"@persistent\s+def _load_post_reset_engine\(")

    def test_scene_scoped_engine_instances(self):
        src = _read("audio_engine.py")
        self.assertIn("_instances", src)
        self.assertIn("def _scene_key(", src)
        self.assertIn("def ensure_instance(cls, scene=None)", src)

    def test_render_lock_interface_safeguard_exists(self):
        src = _read("audio_engine.py")
        self.assertIn("def _unsafe_for_handler_writes(", src)
        self.assertIn("use_lock_interface", src)

    def test_draw_does_not_sync_or_probe(self):
        src = _read("n_panel_ui.py")
        self.assertNotIn("self._refresh_dependencies(REQUIRED_LIBRARIES)", _read("prefs_panel_ui.py"))
        # Draw path should no longer call background sync/probe directly.
        info_streams_body = _slice_between(
            src,
            "class ACE_PT_information_streams",
            "class ACE_PT_information_streams_description",
            label="ACE_PT_information_streams body",
        )
        self.assertIn("bl_label = \"Active Operators\"", info_streams_body)
        self.assertNotIn("layout.label(text=\"Active Operators:\", icon=\"NODETREE\")", info_streams_body)
        self.assertNotIn("        _sync_ui_dummy_tree(context)", info_streams_body)
        self.assertIn("_selected_node_has_operator_settings(context)", info_streams_body)
        self.assertIn("_draw_selected_node_operator_settings(layout, context)", info_streams_body)
        self.assertNotIn("_draw_selected_node_description(", info_streams_body)
        self.assertNotIn("audiocore.bake_active_source", info_streams_body)
        desc_panel_body = _slice_between(
            src,
            "class ACE_PT_information_streams_description",
            "class ACE_PT_capture_influx",
            label="ACE_PT_information_streams_description body",
        )
        self.assertIn("bl_parent_id = \"ACE_PT_information_streams\"", desc_panel_body)
        self.assertIn("_selected_node_description(context)", desc_panel_body)
        self.assertIn("_draw_selected_node_description(self.layout, description)", desc_panel_body)
        capture_body = _slice_between(
            src,
            "class ACE_PT_capture_influx",
            "class ACE_PT_stream_confluence",
            label="ACE_PT_capture_influx body",
        )
        self.assertIn("_draw_bake_controls(layout, context)", capture_body)
        self.assertNotIn("_resolve_capture_node(context)", capture_body)
        self.assertNotIn("engine._probe_source_metadata(active_source)", src)
        self.assertNotIn("ace.gn_target_object = sonar", src)

    def test_proxy_tree_is_modifier(self):
        src = _read("proxy_management.py")
        self.assertIn("tree.is_modifier = True", src)
        self.assertIn("new_socket(\"Audio\", in_out='OUTPUT', socket_type='NodeSocketBundle')", src)

    def test_install_does_not_upgrade_pip_each_time(self):
        src = _read("prefs_library_manager.py")
        self.assertNotIn("\"--upgrade\", \"pip\"", src)
        self.assertIn("--disable-pip-version-check", src)

    def test_registration_uses_qualified_reload_and_rollback(self):
        src = _read("__init__.py")
        self.assertIn("qualified_name = f\"{PACKAGE_NAME}.{module_name}\"", src)
        self.assertIn("Rollback failed for module", src)

    def test_preview_sink_and_splitter_ramp_hooks_exist(self):
        vars_src = _read("vars.py")
        self.assertIn("\"preview_sink\"", vars_src)
        self.assertNotIn("\"label\": \"Latency Tracker\"", vars_src)
        self.assertIn("Frames per Beat", vars_src)

        ops_src = _read("n_panel_operators.py")
        self.assertIn("ace.toggle_preview_sink", ops_src)
        self.assertIn("ace.sync_splitter_ramp", ops_src)
        self.assertIn("ace.apply_splitter_ramp", ops_src)
        self.assertIn("audiocore.detect_source_tempo_meter", ops_src)

        ui_src = _read("n_panel_ui.py")
        self.assertIn("template_color_ramp", ui_src)
        self.assertIn("ace.apply_splitter_ramp", ui_src)
        self.assertNotIn("Default Splitter Bands", ui_src)
        self.assertNotIn("Default Audio Math Inputs", ui_src)
        self.assertIn("(\"add\", \"Add\")", ui_src)
        self.assertIn("(\"subtract\", \"Subtract\")", ui_src)
        self.assertIn("(\"multiply\", \"Multiply\")", ui_src)
        info_streams_body = _slice_between(
            ui_src,
            "class ACE_PT_information_streams",
            "class ACE_PT_information_streams_description",
            label="ACE_PT_information_streams body",
        )
        self.assertIn("bl_label = \"Active Operators\"", info_streams_body)
        self.assertNotIn("layout.label(text=\"Active Operators:\", icon=\"NODETREE\")", info_streams_body)
        self.assertIn("_draw_selected_node_operator_settings(layout, context)", info_streams_body)
        self.assertIn("_selected_node_has_operator_settings(context)", info_streams_body)
        self.assertNotIn("_draw_selected_node_description(", info_streams_body)
        self.assertNotIn("Value:", info_streams_body)
        self.assertNotIn("Type:", info_streams_body)
        self.assertNotIn("Select a node from the active graph list.", info_streams_body)
        desc_panel_body = _slice_between(
            ui_src,
            "class ACE_PT_information_streams_description",
            "class ACE_PT_capture_influx",
            label="ACE_PT_information_streams_description body",
        )
        self.assertIn("bl_label = \"Description\"", desc_panel_body)
        self.assertIn("bl_parent_id = \"ACE_PT_information_streams\"", desc_panel_body)
        self.assertIn("bl_options = {'DEFAULT_CLOSED'}", desc_panel_body)
        self.assertIn("_draw_selected_node_description(self.layout, description)", desc_panel_body)

        dyn_src = _read("dynamic_audio_operators.py")
        self.assertIn("ACEAudioMathInternals_v3", dyn_src)
        self.assertIn("GeometryNodeJoinGeometry", dyn_src)
        self.assertIn("NodeSeparateBundle", dyn_src)
        self.assertIn("ACE_SPLITTER_INTERNALS_V5", dyn_src)
        splitter_body = _slice_between(
            dyn_src,
            "def _build_variant_internals",
            "def _build_variant",
            label="splitter internals",
        )
        self.assertNotIn("GeometryNodeSetPosition", splitter_body)
        self.assertNotIn("combine_xyz", splitter_body)
        self.assertNotIn("clamp_min", splitter_body)

    def test_capture_influx_prefers_active_graph_selection(self):
        ui_src = _read("n_panel_ui.py")
        self.assertIn("def _resolve_ui_list_selected_node(", ui_src)
        self.assertIn("def _build_ui_node_lookup_ids(", ui_src)
        self.assertIn("def _ui_node_lookup_id(", ui_src)
        self.assertIn("def _parsed_node_has_operator_settings(", ui_src)
        self.assertIn("def _selected_node_has_operator_settings(", ui_src)
        self.assertIn("def _selected_node_description(", ui_src)
        desc_helper_body = _slice_between(
            ui_src,
            "def _selected_node_description(",
            "def _draw_selected_node_operator_settings(",
            label="_selected_node_description body",
        )
        self.assertNotIn("_selected_node_has_operator_settings(context)", desc_helper_body)
        resolve_body = _slice_between(
            ui_src,
            "def _resolve_capture_node(",
            "def _splitter_node_signature(",
            label="_resolve_capture_node body",
        )
        self.assertIn("node, parsed = _resolve_ui_list_selected_node(context)", resolve_body)
        self.assertIn("node = getattr(context, \"active_node\", None)", resolve_body)
        self.assertLess(
            resolve_body.index("_resolve_ui_list_selected_node(context)"),
            resolve_body.index("node = getattr(context, \"active_node\", None)"),
        )
        self.assertIn("mapped_id = str(lookup_ids.get(node_id, \"\") or \"\")", ui_src)
        self.assertIn("item.node_id = _ui_node_lookup_id(node, node_lookup_ids)", ui_src)
        self.assertIn("return drew_settings", ui_src)
        self.assertNotIn("Advanced Attributes", ui_src)
        self.assertNotIn("layout.label(text=chunk, icon=\"INFO\")", ui_src)
        self.assertIn("(\"utilities\", \"preview_sink\")", ui_src)

        ops_src = _read("n_panel_operators.py")
        self.assertIn("from .n_panel_ui import _resolve_capture_node", ops_src)

    def test_preview_audio_is_gated_to_timeline_playback(self):
        src = _read("audio_engine.py")
        self.assertIn("preview_allowed = self._is_animation_playing()", src)
        self.assertIn("if (not preview_allowed) or (not preview_pushed):", src)
        self.assertIn("mixed_preview = self._mix_preview_chunks(preview_chunks)", src)
        self.assertNotIn("graph_uid == active_source_uid", src)
        self.assertIn("self._limiter_ceiling_db = -1.0", src)
        self.assertIn("def _apply_safety_limiter(", src)
        self.assertIn("self._ring = deque()", src)
        self.assertIn("def _append_ring_block_locked(", src)

    def test_layer1_specs_and_rolling_avg_use_proxy_object_sampling_internals(self):
        src = _read("dummy_tree.py")
        self.assertIn("\"Proxy Object\",", src)
        self.assertIn("\"Frames per Beat Attribute ID\"", src)
        self.assertIn("\"Beat Phase Attribute ID\"", src)
        self.assertIn("\"Proxy Object\"", src)
        specs_body = _slice_between(
            src,
            "if node_type == 'src_specs':",
            "elif node_type == 'preview_sink':",
            label="src_specs section",
        )
        rolling_avg_body = _slice_between(
            src,
            "elif node_type == 'util_rolling_avg':",
            "\n        # =====================================================================\n        # FALLBACK (unknown node type)",
            label="rolling average section",
        )
        for body in (specs_body, rolling_avg_body):
            self.assertIn('new_socket("Proxy Object"', body)
            self.assertIn("GeometryNodeObjectInfo", body)
            self.assertIn("GeometryNodeInputNamedAttribute", body)
            self.assertIn("GeometryNodeSampleIndex", body)

        menu_src = _read("gn_menu.py")
        init_body = _slice_between(
            menu_src,
            "def _initialize_auto_attr_ids(node, context):",
            "    @staticmethod\n    def _show_attr_id_socket_values(node):",
            label="_initialize_auto_attr_ids body",
        )
        self.assertIn("proxy_obj = bpy.data.objects.get(proxy_name)", init_body)
        self.assertIn("if (is_specs or is_rolling_avg) and proxy_obj:", init_body)
        self.assertIn("proxy_sock.default_value = proxy_obj", init_body)

    def test_runtime_diagnostics_and_parallel_slot_reuse_exist(self):
        src = _read("audio_engine.py")
        self.assertIn("def diagnostics_snapshot(", src)
        self.assertIn("def get_runtime_diagnostics(", src)
        self.assertIn("def get_dsp_node_profile_snapshot(", src)
        self.assertIn("preview_underruns", src)
        self.assertIn("preview_queue_drops", src)
        self.assertIn("parallel_job_timeouts", src)
        self.assertIn("parallel_pool_restarts", src)
        self.assertIn("_PARALLEL_ATTRS_PER_JOB", src)
        self.assertIn("parallel_output_shards", src)
        self.assertIn("_parallel_payload_slots", src)
        self.assertIn("def _parallel_release_payload_slots(", src)
        self.assertIn("def _parallel_ensure_payload_slot(", src)
        self.assertIn("def _parallel_reserve_payload_slot(", src)
        self.assertIn("_parallel_frame_payload_slots", src)
        self.assertIn("_parallel_payload_slot_refs", src)
        self.assertIn("def _parallel_shard_attr_keys(", src)
        self.assertIn("output_keys", src)

    def test_blender_side_latency_metrics_exist(self):
        src = _read("audio_engine.py")
        self.assertIn("def _mark_geometry_eval_pending(", src)
        self.assertIn("def _consume_geometry_eval_updates(", src)
        self.assertIn("def _mark_render_start(", src)
        self.assertIn("def _mark_render_complete(", src)
        self.assertIn("def get_perf_current(", src)
        self.assertIn("def get_runtime_diagnostics_current(", src)
        self.assertIn("def current_snapshot(", src)
        self.assertIn("_CURRENT_LATENCY_SAMPLE_FRESHNESS_FRAMES = 2", src)
        self.assertIn("self._geometry_eval_last_frame", src)
        self.assertIn("self._is_rendering_job()", src)
        self.assertIn("blender_geometry_eval_p95_ms", src)
        self.assertIn("blender_render_p95_ms", src)
        self.assertIn("preview_branch_latency_ms", src)
        self.assertIn("preview_output_buffer_ms", src)
        self.assertIn("preview_callback_ms", src)
        self.assertIn("self._last_push_ts = 0.0", src)
        self.assertIn("latency_ms=latency_ms,", src)
        self.assertIn("engine._consume_geometry_eval_updates(scene, depsgraph)", src)
        self.assertIn("engine._mark_render_start(scene)", src)
        self.assertIn("engine._mark_render_complete(scene, cancelled=False)", src)

    def test_parallel_worker_policy_is_cpu_based_not_source_count_capped(self):
        src = _read("audio_engine.py")
        worker_body = _slice_between(
            src,
            "def _parallel_worker_count(",
            "def _parallel_inflight_limit(",
            label="parallel worker policy",
        )
        self.assertIn("ACE_PARALLEL_WORKERS", worker_body)
        self.assertIn("os.cpu_count()", worker_body)
        self.assertIn("_PARALLEL_WORKER_HEADROOM_CORES", worker_body)
        self.assertNotIn("source_count", worker_body)
        self.assertNotIn("len(self.graphs)", worker_body)

    def test_preview_monitor_preparation_is_offloaded(self):
        src = _read("audio_engine.py")
        self.assertIn("self._prepare_condition = threading.Condition()", src)
        self.assertIn("def _prepare_loop(", src)
        self.assertIn("def _prepare_preview_payload(", src)
        self.assertIn("preview_prepare_queue_pressure", src)
        self.assertRegex(src, r"def push_chunk\(self, chunk\) -> bool:")

    def test_group_output_preview_metadata_and_auto_preview_hooks_exist(self):
        abstract_src = _read("abstract_tree.py")
        self.assertIn("class ACEAbstractOutputLink", abstract_src)
        self.assertIn("group_output_links", abstract_src)

        dummy_src = _read("dummy_tree.py")
        self.assertIn("add_group_output_link", dummy_src)
        self.assertIn("GROUP_OUTPUT", dummy_src)

        engine_src = _read("audio_engine.py")
        self.assertIn("_group_output_preview_specs_for_source", engine_src)
        self.assertIn("__ace_auto_preview__", engine_src)
        self.assertIn("_resolve_global_manual_preview_winner", engine_src)
        self.assertIn("manual_preview_task_ids", engine_src)
        self.assertIn("auto_preview_task_ids", engine_src)

    def test_auto_attr_ids_reinitialize_on_reconnect(self):
        engine_src = _read("audio_engine.py")
        self.assertIn("ace_auto_attr_topology_json", engine_src)
        self.assertIn("incoming_by_target", engine_src)
        self.assertIn("upstream_sig = json.dumps(sorted(incoming_by_target.get(abs_node.id, [])))", engine_src)
        self.assertIn("topology_sig=topology_sig", engine_src)
        self.assertNotIn("mode = \"manual\" if mode_user_locked else \"auto\"", engine_src)

        menu_src = _read("gn_menu.py")
        self.assertNotIn("ace_attr_mode_user_locked", menu_src)
        self.assertIn("sock.hide_value = False", menu_src)

        ops_src = _read("n_panel_operators.py")
        self.assertNotIn("ace.set_attr_mode", ops_src)
        self.assertNotIn("ACE_OT_set_attr_mode", ops_src)

        ui_src = _read("n_panel_ui.py")
        self.assertNotIn("ace.set_attr_mode", ui_src)
        self.assertIn("Auto-Compile", ui_src)

        dummy_src = _read("dummy_tree.py")
        self.assertIn("is_custom = bool(attr_name)", dummy_src)
        self.assertIn("item.hide_value = False", dummy_src)
        self.assertIn("self._ace_id_owner", dummy_src)
        self.assertIn("def _reserve_unique_ace_id(", dummy_src)

    def test_deterministic_node_id_seed_uses_structural_context_only(self):
        dummy_src = _read("dummy_tree.py")
        self.assertIn("def _build_deterministic_node_seed(", dummy_src)
        self.assertIn("\"socket_signature\"", dummy_src)
        self.assertIn("\"topology_context\"", dummy_src)
        self.assertIn("\"source_context\"", dummy_src)
        self.assertIn("\"root_context\"", dummy_src)
        seed_body = _slice_between(
            dummy_src,
            "def _build_deterministic_node_seed(",
            "def _ensure_ace_id(",
            label="deterministic node id seed builder",
        )
        self.assertNotIn("bl_node.name", seed_body)

    def test_file_import_is_limited_to_supported_formats(self):
        ops_src = _read("n_panel_operators.py")
        self.assertIn("def _supported_audio_filter_glob()", ops_src)
        self.assertIn("default=_supported_audio_filter_glob()", ops_src)
        self.assertIn("AudioReception.is_supported_file_path(filepath)", ops_src)
        self.assertIn("Skipped unsupported files", ops_src)

        influx_src = _read("audio_influx.py")
        self.assertIn("def supported_file_extensions()", influx_src)
        self.assertIn("def supported_file_filter_glob()", influx_src)
        self.assertIn("def is_supported_file_path(filepath: str)", influx_src)

    def test_ui_marks_source_connected_nodes_as_active(self):
        ui_src = _read("n_panel_ui.py")
        self.assertIn("def _sync_ui_dummy_tree(", ui_src)
        self.assertIn("def _ui_nodes_match(", ui_src)
        self.assertIn("item.current_value", ui_src)
        self.assertNotIn("show_inactive_nodes", ui_src)
        self.assertNotIn("Show All Source-Connected Nodes", ui_src)

    def test_source_management_taxonomy_and_menu_visibility(self):
        vars_src = _read("vars.py")
        # Source management category only hosts the explicit Source contract node.
        self.assertRegex(vars_src, r"\"source_mngmt\"\s*:\s*\{")
        self.assertRegex(vars_src, r"\"source_mngmt\"[\s\S]*?\"nodes\"\s*:\s*\{\s*\"src\"\s*:")
        self.assertNotIn("\"source_mngmt_src_specs\"", vars_src)
        self.assertNotIn("\"source_mngmt_scanner\"", vars_src)
        # Specs now live under utilities.
        self.assertRegex(vars_src, r"\"utilities\"[\s\S]*?\"src_specs\"\s*:")
        self.assertNotRegex(vars_src, r"\"utilities\"[\s\S]*?\"scanner\"\s*:")

        menu_src = _read("gn_menu.py")
        self.assertIn("def _initialize_source_binding(node, context):", menu_src)
        self.assertIn("node[\"ace_source_id\"]", menu_src)
        self.assertIn("bl_label = \"Input\"", menu_src)
        self.assertIn("layout.menu(\"ACE_MT_source_mngmt\"", menu_src)
        self.assertIn("layout.menu(\"ACE_MT_delta\"", menu_src)
        self.assertIn("layout.menu(\"ACE_MT_stream\"", menu_src)
        self.assertIn("layout.menu(\"ACE_MT_utilities\"", menu_src)
        self.assertNotIn("managed automatically per audio source", menu_src)

    def test_shader_editor_add_menu_uses_shader_group_nodes(self):
        menu_src = _read("gn_menu.py")
        self.assertIn("_ALLOWED_CATEGORIES_BY_TREE_TYPE", menu_src)
        self.assertIn('"ShaderNodeTree": ("source_mngmt", "delta", "stream", "utilities")', menu_src)
        self.assertIn("def _allowed_categories_for_context(context) -> tuple[str, ...]:", menu_src)
        self.assertIn("from ..nodes.shader_tree import ACEShaderNodeFactory", menu_src)
        self.assertIn("def poll(cls, context):", menu_src)
        self.assertIn("def _factory_for_context(cls, context):", menu_src)
        self.assertIn("def _group_node_type_for_context(cls, context) -> str:", menu_src)
        self.assertIn("return ACEShaderNodeFactory", menu_src)
        self.assertIn('return "ShaderNodeGroup"', menu_src)
        self.assertIn("if not self._category_allowed_for_context(self.category_key, context):", menu_src)
        self.assertIn('return "source_mngmt" in _allowed_categories_for_context(context)', menu_src)
        self.assertIn("tree_type not in {\"GeometryNodeTree\", \"ShaderNodeTree\"}", menu_src)
        self.assertIn("ACESplitterManager.get_or_create_variant(splitter_bands, tree)", menu_src)
        self.assertIn("ACEAudioMathManager.get_or_create_variant(audio_math_inputs, tree)", menu_src)
        self.assertIn("node = tree.nodes.new(group_node_type)", menu_src)
        self.assertIn("self._ensure_shader_frontend_uid(tree, context)", menu_src)
        self.assertIn("ACEShaderNodeFactory.sync_node_instance_group_bindings(", menu_src)
        self.assertIn("scene=getattr(context, \"scene\", None)", menu_src)
        self.assertIn("NODE_MT_category_shader_add.append(add_shader_category_menu_func)", menu_src)

    def test_shader_editor_dynamic_variants_are_tree_aware(self):
        dynamic_src = _read("dynamic_audio_operators.py")
        shader_src = _read("shader_tree.py")
        dummy_src = _read("dummy_tree.py")

        self.assertIn("def _tree_bl_idname(tree_or_type) -> str:", dynamic_src)
        self.assertIn("from .shader_tree import ACEShaderNodeFactory", dynamic_src)
        self.assertIn("if _tree_bl_idname(tree_or_type) == \"ShaderNodeTree\":", dynamic_src)
        self.assertNotIn("ShaderNodeMath", dynamic_src)
        self.assertIn("NodeReroute", dynamic_src)
        self.assertIn("get_or_create_variant(band_count, getattr(node, \"id_data\", None))", dynamic_src)
        self.assertIn("get_or_create_variant(count, getattr(node, \"id_data\", None))", dynamic_src)
        self.assertIn("def rebuild_node_group(cls, cat_key: str, type_key: str):", shader_src)
        self.assertIn("ACESplitterManager.refresh_existing_variants(\"ShaderNodeTree\")", shader_src)
        self.assertIn("ACEAudioMathManager.refresh_existing_variants(\"ShaderNodeTree\")", shader_src)
        self.assertIn("ShaderNodeAttribute", shader_src)
        self.assertIn("def sync_node_instance_group_bindings(", shader_src)
        self.assertIn("for enum_token in (\"OBJECT\", \"OBJECT_DATA\")", shader_src)
        self.assertIn("ACEAudioMathManager.refresh_existing_variants(\"GeometryNodeTree\")", dummy_src)

    def test_shader_output_id_sync_updates_shader_instance_bindings(self):
        engine_src = _read("audio_engine.py")
        body = _slice_between(
            engine_src,
            "def _sync_shader_output_ids(",
            "def _create_task_for_node(",
            label="_sync_shader_output_ids body",
        )
        self.assertIn("shader_factory.sync_node_instance_group_bindings(", body)

    def test_shader_sink_bindings_route_to_consumer_objects(self):
        engine_src = _read("audio_engine.py")
        body = _slice_between(
            engine_src,
            "def _resolve_sink_bindings_for_graph(",
            "@staticmethod\n    def _safe_file_mtime(",
            label="_resolve_sink_bindings_for_graph body",
        )
        self.assertIn("OBJECT_PROPERTY", body)
        self.assertIn("owner_keys", body)
        # Shader sinks must NOT force-route to proxy — they write to
        # consumer objects so Attribute nodes in Object mode can read them.
        self.assertNotIn("consumers = (str(proxy_name),)", body)

    def test_shader_sinks_use_attribute_nodes(self):
        shader_src = _read("shader_tree.py")
        self.assertIn("ShaderNodeAttribute", shader_src)
        self.assertIn("_set_attr_node_defaults", shader_src)
        self.assertIn("_HELPER_NODE_OWNER_KEY", shader_src)
        self.assertIn("_HELPER_ATTR_NAME_KEY", shader_src)
        self.assertIn("cleanup_tree_helper_nodes", shader_src)
        self.assertIn("_ensure_helper_link", shader_src)
        self.assertIn("_rebuild_helper_attr_node", shader_src)
        self.assertIn("attribute_type", shader_src)
        self.assertIn("sink_node[cls._OUTPUT_NODE_KIND_KEY]", shader_src)
        # Driver bridge must be fully removed.
        self.assertNotIn("driver_add(", shader_src)
        self.assertNotIn("driver_remove(", shader_src)
        self.assertNotIn("FLOAT_DRIVER", shader_src)
        self.assertNotIn("VECTOR_DRIVER", shader_src)
        self.assertNotIn("COLOR_DRIVER", shader_src)

    def test_ui_active_node_list_follows_shader_or_geometry_editor_context(self):
        ui_src = _read("n_panel_ui.py")
        find_body = _slice_between(
            ui_src,
            "def _find_active_node_tree(context, editor_context_only: bool = False):",
            "def _describe_node(",
            label="_find_active_node_tree body",
        )
        self.assertIn("{'GeometryNodeTree', 'ShaderNodeTree'}", find_body)
        self.assertIn("_iter_visible_node_editor_trees(context)", find_body)
        self.assertIn("if editor_context_only:", find_body)
        self.assertIn("get_from_context", find_body)
        self.assertIn("material_slots", find_body)

        sync_body = _slice_between(
            ui_src,
            "def _sync_ui_dummy_tree(context, force: bool = False, allow_dirty_rebuild: bool = True):",
            "def _refresh_active_source_metadata(",
            label="_sync_ui_dummy_tree body",
        )
        self.assertIn("_find_active_node_tree(context, editor_context_only=True)", sync_body)
        self.assertIn("if str(getattr(tree, \"bl_idname\", \"\") or \"\") == \"ShaderNodeTree\":", sync_body)
        self.assertIn("reader = ACEShaderTreeReader(tree)", sync_body)
        self.assertIn("reader = ACEDummyTreeReader(tree)", sync_body)
        self.assertIn("no active ACE node tree found", sync_body)

        timer_body = _slice_between(
            ui_src,
            "def _ui_sync_timer():",
            "@persistent",
            label="_ui_sync_timer body",
        )
        self.assertIn("_find_active_node_tree(context, editor_context_only=True)", timer_body)

    def test_ui_stream_list_filters_source_management_nodes(self):
        ui_src = _read("n_panel_ui.py")
        sync_body = _slice_between(
            ui_src,
            "def _sync_ui_dummy_tree(",
            "def _refresh_active_source_metadata(",
            label="_sync_ui_dummy_tree body",
        )
        self.assertIn("!= \"source_mngmt\"", sync_body)
        self.assertIn("for node in nodes", sync_body)

    def test_source_processing_is_forced_each_frame_per_graph(self):
        engine_src = _read("audio_engine.py")
        self.assertIn("if scheduler_mode == \"legacy\":", engine_src)
        self.assertIn("source_task.get_output(self.current_frame, 0)", engine_src)
        self.assertIn("if not (node.category == 'source_mngmt' and node.type == 'src')", engine_src)
        self.assertIn("scheduler_mode: str = \"legacy\"", engine_src)
        self.assertIn("graph.scheduler_mode = scheduler_mode", engine_src)
        legacy_body = _slice_between(
            engine_src,
            "if scheduler_mode == \"legacy\":",
            "if preview_allowed:",
            label="legacy scheduler execute section",
        )
        self.assertIn("source_task.get_output(self.current_frame, 0)", legacy_body)

    def test_perf_and_timeslice_controls_exist(self):
        props_src = _read("properties.py")
        self.assertIn("ace_perf_enabled", props_src)
        self.assertIn("ace_scheduler_mode", props_src)
        self.assertIn("ace_timeslice_mode", props_src)
        self.assertIn("default=\"demand\"", props_src)
        self.assertIn("default=\"fixed\"", props_src)
        self.assertIn("ace_parallel_mode", props_src)
        self.assertIn("(\"threads_io\", \"Threads-I/O\"", props_src)
        self.assertIn("(\"processes_dsp\", \"Processes-DSP\"", props_src)
        self.assertIn("default=\"processes_dsp\"", props_src)

        ui_src = _read("n_panel_ui.py")
        self.assertNotIn("perf_row.prop(eng_settings, \"ace_perf_enabled\"", ui_src)
        self.assertNotIn("sched_row.prop(eng_settings, \"ace_scheduler_mode\"", ui_src)
        self.assertNotIn("timeslice_row.prop(eng_settings, \"ace_timeslice_mode\"", ui_src)
        self.assertNotIn("parallel_row.prop(eng_settings, \"ace_parallel_mode\"", ui_src)
        self.assertNotIn("turbo_row.prop(eng_settings, \"preview_numba_turbo\"", ui_src)

        engine_src = _read("audio_engine.py")
        self.assertIn("def _parallel_mode(cls, scene_or_engine_settings=None) -> str:", engine_src)
        self.assertIn("return \"processes_dsp\"", engine_src)

    def test_project_settings_panel_hides_backend_runtime_details(self):
        ui_src = _read("n_panel_ui.py")
        panel_body = _slice_between(
            ui_src,
            "class ACE_PT_project_settings",
            "class ACE_PT_sub_latency_measures",
            label="ACE_PT_project_settings body",
        )
        self.assertIn("bl_label = \"Engine Status\"", panel_body)
        self.assertNotIn("layout.prop(ace, \"mode\", text=\"Mode\")", ui_src)
        self.assertNotIn("Runtime Profile: Processes-DSP (locked)", ui_src)
        self.assertNotIn("Scheduler: Demand | Timeslice: Fixed", ui_src)
        self.assertNotIn("Trigger Engine Recompilation", ui_src)
        self.assertIn("status_row = layout.row(align=True)", panel_body)
        self.assertIn("status_row.label(text=status_text, icon=status_icon)", panel_body)
        self.assertIn("status_text = \"Compiling...\"", panel_body)
        self.assertIn("status_text = \"Active\"", panel_body)
        self.assertIn("status_text = \"Idle\"", panel_body)
        self.assertNotIn("engine_box = layout.box()", panel_body)
        self.assertNotIn("PREFERENCES", panel_body)
        self.assertIn("ACEAudioEngine._is_animation_playing()", panel_body)
        self.assertIn("latency_metrics = _engine_latency_metrics(context)", panel_body)
        self.assertIn("performance_box.label(text=\"Engine Performance\", icon=\"DRIVER\")", panel_body)
        self.assertIn("current_col.label(text=\"Current Latency\")", panel_body)
        self.assertIn("max_col.label(text=\"Latency Budget\")", panel_body)
        self.assertIn("current_col.label(text=f\"{float(latency_metrics['total_latency_ms']):.2f} ms\")", panel_body)
        self.assertIn("fps_base = float(getattr(render, \"fps_base\", 1.0) or 1.0)", panel_body)
        self.assertIn("frame_budget_ms = 1000.0 / max(scene_fps, 1e-6)", panel_body)
        self.assertNotIn("fps\")", panel_body)
        self.assertNotIn("Waiting for data", panel_body)
        self.assertNotIn("Realtime Threshold", panel_body)
        self.assertNotIn("Latency Detail", panel_body)
        self.assertIn("text=\"Refresh Engine\"", panel_body)
        self.assertNotIn("Use manual compile when Auto-Compile is disabled.", panel_body)
        self.assertNotIn("Select a Geometry Nodes graph to compile.", panel_body)
        self.assertNotIn("layout.prop(eng_settings, \"auto_compile\", text=\"Auto-Compile Graph Changes\")", panel_body)
        self.assertNotIn("Engine Settings", panel_body)
        self.assertNotIn("status_row.label(text=\"Ready\"", panel_body)
        self.assertIn("bl_options = {'DEFAULT_CLOSED'}", panel_body)
        self.assertIn("bl_order = -1", panel_body)

    def test_active_ace_instances_panel_uses_compiled_source_instance_ledger(self):
        ui_src = _read("n_panel_ui.py")
        panel_body = _slice_between(
            ui_src,
            "class ACE_PT_stream_confluence",
            "# --- REGISTRATION ---",
            label="ACE_PT_stream_confluence body",
        )
        self.assertIn("bl_label = \"Active ACE Instances\"", panel_body)
        self.assertIn("ledger_rows = _compiled_source_instance_rows(", panel_body)
        self.assertIn("source_box = layout.box()", panel_body)
        self.assertIn("source_icon = _source_list_icon(entry.get(\"source_prop\"))", panel_body)
        self.assertIn("empty_row.label(text=\"No connected instances\")", panel_body)
        self.assertIn("layout.label(text=\"Engine offline or no sources found.\")", panel_body)
        self.assertIn("_frontend_object_name(frontend_ref)", panel_body)
        self.assertIn("_frontend_instance_type(frontend_kind)", panel_body)
        self.assertNotIn("tree_name", panel_body)
        self.assertNotIn("material_name", panel_body)
        self.assertNotIn("bl_options = {'DEFAULT_CLOSED'}", panel_body)
        self.assertNotIn("frontends = _scene_frontends(", panel_body)
        self.assertNotIn("No active ACE instances.", panel_body)
        self.assertNotIn("layout.prop(ace, \"gn_target_object\", text=\"GN Target\")", panel_body)

    def test_latency_measures_panel_is_project_settings_child(self):
        ui_src = _read("n_panel_ui.py")
        latency_body = _slice_between(
            ui_src,
            "class ACE_PT_sub_latency_measures",
            "class ACE_PT_sub_engine_settings",
            label="ACE_PT_sub_latency_measures body",
        )
        self.assertIn("bl_label = \"Latency Detail\"", latency_body)
        self.assertIn("bl_parent_id = \"ACE_PT_project_settings\"", latency_body)
        self.assertIn("bl_options = {'DEFAULT_CLOSED'}", latency_body)
        self.assertIn("latency_metrics = _engine_latency_metrics(context)", latency_body)
        self.assertIn("_UI_LATENCY_UPDATE_INTERVAL_S = 1.0", ui_src)
        self.assertIn("_ui_latency_cache = {", ui_src)
        self.assertIn("\"source_uid\": \"\"", ui_src)
        self.assertIn("\"ts\": 0.0", ui_src)
        self.assertIn("engine.get_perf_current()", ui_src)
        self.assertIn("engine.get_runtime_diagnostics_current()", ui_src)
        self.assertIn("input_total_ms", latency_body)
        self.assertIn("engine_total_ms", latency_body)
        self.assertIn("preview_branch_ms", ui_src)
        self.assertIn("preview_output_buffer_ms", ui_src)
        self.assertIn("preview_device_ms", ui_src)
        self.assertIn("preview_prefetch_ms", ui_src)
        self.assertIn("\"preview_active\": False", ui_src)
        self.assertIn("\"preview_prefetch_ms\": 0.0", ui_src)
        self.assertIn("+ float(metrics[\"file_buffer_ms\"])", ui_src)
        self.assertIn("- float(metrics[\"preview_prefetch_ms\"])", ui_src)
        self.assertIn("+ float(metrics[\"preview_branch_ms\"])", ui_src)
        self.assertIn("+ float(metrics[\"preview_output_buffer_ms\"])", ui_src)
        self.assertIn("+ float(metrics[\"preview_device_ms\"])", ui_src)
        self.assertIn("+ float(metrics[\"preview_prefetch_ms\"])", ui_src)
        self.assertNotIn("+ float(metrics[\"preview_total_ms\"])", ui_src)
        self.assertIn("blender_total_ms", latency_body)
        self.assertIn("Source Input", latency_body)
        self.assertIn("Audio Analysis", latency_body)
        self.assertIn("Blender Side", latency_body)
        self.assertNotIn("preview_total_ms", latency_body)
        self.assertNotIn("Audio Preview", latency_body)
        self.assertNotIn("Total latency", latency_body)
        self.assertNotIn("Play the timeline to collect timing data.", latency_body)
        self.assertNotIn("run a render to measure", latency_body)
        self.assertNotIn("Preview branch p95", latency_body)

    def test_engine_settings_panel_is_project_settings_child(self):
        ui_src = _read("n_panel_ui.py")
        settings_body = _slice_between(
            ui_src,
            "class ACE_PT_sub_engine_settings",
            "class ACE_PT_audio_sources",
            label="ACE_PT_sub_engine_settings body",
        )
        self.assertIn("bl_label = \"Engine Settings\"", settings_body)
        self.assertIn("bl_parent_id = \"ACE_PT_project_settings\"", settings_body)
        self.assertIn("bl_options = {'DEFAULT_CLOSED'}", settings_body)
        self.assertIn("sample_rate_row = layout.row(align=True)", settings_body)
        self.assertIn("sample_rate_row.label(text=\"Sample Rate\")", settings_body)
        self.assertIn("sample_rate_buttons = sample_rate_row.row(align=True)", settings_body)
        self.assertIn("for option in (\"22050\", \"44100\", \"48000\", \"96000\")", settings_body)
        self.assertIn("from ..utils import _get_active_source", settings_body)
        self.assertIn("_, src = _get_active_source(context)", settings_body)
        self.assertIn("layout.prop(src, \"enable_preprocessing\", text=\"Enable Source Preprocessing\")", settings_body)
        self.assertIn("split = layout.split(factor=0.5, align=True)", settings_body)
        self.assertIn("split.enabled = bool(getattr(src, \"enable_preprocessing\", False))", settings_body)
        self.assertIn("left.prop(src, \"freq_min\", text=\"Min Hz\")", settings_body)
        self.assertIn("left.prop(src, \"envelope_attack\", text=\"Attack (ms)\")", settings_body)
        self.assertIn("right.prop(src, \"freq_max\", text=\"Max Hz\")", settings_body)
        self.assertIn("right.prop(src, \"envelope_release\", text=\"Release (ms)\")", settings_body)
        self.assertIn("layout.prop(eng_settings, \"auto_compile\", text=\"Auto-Compile Graph Changes\")", settings_body)
        self.assertNotIn("for option in (\"native\",", settings_body)
        self.assertLess(
            settings_body.index("sample_rate_row = layout.row(align=True)"),
            settings_body.index("layout.prop(src, \"enable_preprocessing\", text=\"Enable Source Preprocessing\")"),
        )
        self.assertLess(
            settings_body.index("layout.prop(src, \"enable_preprocessing\", text=\"Enable Source Preprocessing\")"),
            settings_body.index("layout.prop(eng_settings, \"auto_compile\", text=\"Auto-Compile Graph Changes\")"),
        )
        self.assertNotIn("class ACE_PT_sub_envelope_parameters", ui_src)

    def test_audio_sources_tree_panel_reads_compiled_ir_only(self):
        ui_src = _read("n_panel_ui.py")
        helper_body = _slice_between(
            ui_src,
            "def _compiled_source_instance_rows(scene):",
            "class ACE_PT_audio_sources",
            label="_compiled_source_instance_rows body",
        )
        panel_body = _slice_between(
            ui_src,
            "class ACE_PT_stream_confluence",
            "# --- REGISTRATION ---",
            label="ACE_PT_stream_confluence body",
        )
        self.assertIn("abstract_tree = _compiled_abstract_tree(scene)", helper_body)
        self.assertIn("consumer_intents", helper_body)
        self.assertIn("source_binding_id", helper_body)
        self.assertIn("source_mngmt", helper_body)
        self.assertIn("empty_row.label(text=\"No connected instances\")", panel_body)
        self.assertIn("layout.label(text=\"Engine offline or no sources found.\")", panel_body)
        self.assertNotIn("empty_row.label(text=\"No connected instances\", icon=\"INFO\")", panel_body)
        self.assertNotIn("layout.label(text=\"Engine offline or no sources found.\", icon=\"INFO\")", panel_body)
        self.assertIn("separator(factor=2.0)", panel_body)
        self.assertIn("_frontend_kind_icon(frontend_kind)", panel_body)
        self.assertNotIn("compile_scene_frontends(", panel_body)
        self.assertNotIn("request_recompile(", panel_body)

    def test_audio_sources_panel_is_top_level_manage_sources_panel(self):
        ui_src = _read("n_panel_ui.py")
        manage_body = _slice_between(
            ui_src,
            "class ACE_PT_audio_sources",
            "class ACE_PT_sub_tempo_meter",
            label="ACE_PT_audio_sources body",
        )
        self.assertIn("bl_label = \"Audio Sources\"", manage_body)
        self.assertIn("bl_order = 2", manage_body)
        self.assertNotIn("bl_parent_id", manage_body)
        self.assertIn("row.template_list(", manage_body)
        self.assertIn("layout.prop(active_source, \"source_type\", text=\"Type\")", manage_body)

    def test_file_source_description_uses_original_sample_rate_metadata(self):
        props_src = _read("properties.py")
        self.assertIn("native_sample_rate: bpy.props.IntProperty(", props_src)
        self.assertIn("Original Sample Rate", props_src)

        ui_src = _read("n_panel_ui.py")
        self.assertIn("def _source_display_sample_rate(", ui_src)
        self.assertIn("getattr(source, \"native_sample_rate\", 0)", ui_src)
        audio_sources_body = _slice_between(
            ui_src,
            "class ACE_PT_audio_sources",
            "class ACE_PT_sub_tempo_meter",
            label="ACE_PT_audio_sources body",
        )
        self.assertIn("rate = _source_display_sample_rate(active_source)", audio_sources_body)

        tasks_src = _read("audio_tasks_registry.py")
        self.assertIn("self._pending_ctx_writes[\"native_sample_rate\"] = int(native_sample_rate)", tasks_src)
        self.assertIn("def _flush_pending_ctx_writes(self) -> None:", tasks_src)

        engine_src = _read("audio_engine.py")
        self.assertIn("def _source_original_sample_rate(", engine_src)
        self.assertIn("cached.get(\"native_sample_rate\"", engine_src)

    def test_add_source_operator_uses_explicit_source_type(self):
        ops_src = _read("n_panel_operators.py")
        self.assertIn("source_type: bpy.props.EnumProperty(", ops_src)
        self.assertNotIn("mode = getattr(ace, \"mode\", 'FILE')", ops_src)
        self.assertIn("new_source.source_type = source_type", ops_src)
        self.assertIn("new_source.name = f\"Audio Source {len(ace.audio_sources)}\"", ops_src)

        ui_src = _read("n_panel_ui.py")
        self.assertIn("col.operator(\"audiocore.add_audio_source\", icon='ADD', text=\"\")", ui_src)
        self.assertIn("col.operator(\"audiocore.remove_audio_source\", icon='REMOVE', text=\"\")", ui_src)
        self.assertNotIn("audiocore.add_audio_sources_from_files", ui_src)
        self.assertNotIn("add_live =", ui_src)
        self.assertIn("def _source_list_icon(source_prop) -> str:", ui_src)
        self.assertIn("icon = _source_list_icon(item)", ui_src)
        self.assertNotIn("icon = 'FILE_SOUND'", ui_src)
        self.assertNotIn("Flow token is generated internally from these settings.", ui_src)

    def test_engine_status_panel_shows_latency_metrics(self):
        ui_src = _read("n_panel_ui.py")
        self.assertIn("class ACE_PT_sub_latency_measures", ui_src)
        self.assertNotIn("latency_box.label(text=\"Latency\", icon=\"SORTTIME\")", ui_src)

    def test_tempo_meter_panel_places_auto_detect_above_row(self):
        ui_src = _read("n_panel_ui.py")
        tempo_body = _slice_between(
            ui_src,
            "class ACE_PT_sub_tempo_meter",
            "class ACE_PT_information_streams",
            label="ACE_PT_sub_tempo_meter body",
        )
        self.assertIn("audiocore.detect_source_tempo_meter", tempo_body)
        self.assertIn("row = layout.row(align=True)", tempo_body)
        self.assertIn("row.prop(src, \"bpm\", text=\"BPM\")", tempo_body)
        self.assertIn("row.prop(src, \"beats_per_bar\", text=\"Beats/Bar\")", tempo_body)
        self.assertNotIn("col.prop(src, \"bpm\", text=\"BPM\")", tempo_body)

    def test_n_panel_queues_non_blocking_recompiles_for_edit_operators(self):
        ops_src = _read("n_panel_operators.py")
        self.assertIn("def _queue_engine_recompile(", ops_src)
        self.assertIn("engine.request_recompile(resolved_tree)", ops_src)
        self.assertNotIn("allow_tree_mutation", ops_src)
        self.assertIn("engine.compile_scene_frontends(scene)", ops_src)
        self.assertIn("engine.compile_scene_frontends(context.scene)", ops_src)
        self.assertNotIn("engine.execute_next_frame(context.scene, force=True)", ops_src)

        engine_src = _read("audio_engine.py")
        self.assertNotIn("allow_tree_mutation", engine_src)
        self.assertIn("def request_recompile(self, bl_tree: bpy.types.NodeTree | None):", engine_src)

    def test_realtime_path_remains_single_threaded(self):
        engine_src = _read("audio_engine.py")
        execute_body = _slice_between(
            engine_src,
            "def execute_next_frame",
            "def bake_source_to_cache",
            label="execute_next_frame",
        )
        self.assertNotIn("threading.Thread", execute_body)
        tasks_src = _read("audio_tasks_registry.py")
        # Background metadata analysis may use a dedicated worker thread, but it
        # must stay outside the realtime execute path.
        self.assertIn("def _ensure_background_worker(", tasks_src)
        self.assertIn("threading.Thread(", tasks_src)
        self.assertIn("def _step_long_term_analysis(", tasks_src)
        self.assertIn("def _reset_long_term_analysis_state(", tasks_src)

    def test_parallel_process_dsp_guardrails_exist(self):
        engine_src = _read("audio_engine.py")
        self.assertIn("multiprocessing.get_context(\"spawn\")", engine_src)
        self.assertIn("_parallel_generation_id", engine_src)
        self.assertIn("_parallel_values_for_graph_frame", engine_src)
        self.assertIn("if parallel_mode == \"processes_dsp\":", engine_src)
        self.assertIn("\"spectral_flux\"", engine_src)
        self.assertIn("\"onset\"", engine_src)

        worker_src = _read("parallel_dsp.py")
        self.assertNotIn("import bpy", worker_src)
        self.assertIn("def evaluate_parallel_job(", worker_src)
        self.assertIn("SpectralFluxTask", worker_src)
        self.assertIn("OnsetTask", worker_src)

    def test_parallel_offload_uses_last_valid_without_main_thread_fallback(self):
        engine_src = _read("audio_engine.py")
        self.assertIn("parallel_offload_graph = (", engine_src)
        self.assertIn("if parallel_offload_graph:", engine_src)
        self.assertIn("parallel_values = self._parallel_values_for_graph_frame(", engine_src)
        self.assertIn("reusing last-valid offload values", engine_src)
        self.assertNotIn("falling back to main-thread DSP", engine_src)
        self.assertNotIn("using main-thread DSP", engine_src)

    def test_compile_write_guardrails_only_touch_auto_attr_fields(self):
        engine_src = _read("audio_engine.py")
        sync_body = _slice_between(
            engine_src,
            "def _sync_auto_attr_ids(",
            "def _attribute_socket_to_output_index(",
            label="_sync_auto_attr_ids body",
        )
        self.assertIn("sock.default_value = target", sync_body)
        self.assertIn("bl_node[\"ace_auto_attr_topology_json\"] = topology_sig", sync_body)
        self.assertNotIn("bl_node[\"ace_id\"] =", sync_body)
        self.assertNotIn("bl_node.name =", sync_body)
        self.assertNotIn("root_sources =", sync_body)

    def test_compile_scene_frontends_refreshes_auto_attr_metadata(self):
        engine_src = _read("audio_engine.py")
        compile_body = _slice_between(
            engine_src,
            "def compile_scene_frontends(",
            "def compile_from_blender_tree(",
            label="compile_scene_frontends body",
        )
        self.assertIn("self.discover_scene_frontends(scene)", compile_body)
        self.assertIn("self._sync_auto_attr_ids(", compile_body)
        self.assertIn("self._refresh_abstract_attr_params_from_tree(", compile_body)
        self.assertIn("merged_tree.merge_from(partial_tree)", compile_body)

        wrapper_body = _slice_between(
            engine_src,
            "def compile_from_blender_tree(",
            "def adjust_graph_block_size_for_task(",
            label="compile_from_blender_tree body",
        )
        self.assertIn("self.compile_scene_frontends(scene, preferred_tree=bl_tree)", wrapper_body)

    def test_scene_frontend_sink_dispatch_contract_exists(self):
        engine_src = _read("audio_engine.py")
        proxy_src = _read("proxy_management.py")
        tasks_src = _read("audio_tasks_registry.py")

        self.assertIn("def discover_scene_frontends(self, scene)", engine_src)
        self.assertIn("def compile_scene_frontends(self, scene, preferred_tree=None):", engine_src)
        self.assertIn("graph.sink_bindings_by_attr = self._resolve_sink_bindings_for_graph(", engine_src)
        self.assertIn("self.writer.ensure_sink_layout(graph, active_keys=set(graph.output_attr_keys))", engine_src)
        self.assertIn("self.writer.commit_results(graph, results, active_keys=set(graph.output_attr_keys))", engine_src)
        self.assertIn("def ensure_sink_layout(self, graph, active_keys: Set[str] | None = None):", proxy_src)
        self.assertIn("def commit_results(self, graph, results: Dict[str, Any], active_keys: Set[str] | None = None):", proxy_src)
        self.assertIn("self._object_prop_cache", proxy_src)
        self.assertIn("self._last_object_prop_values", proxy_src)
        self.assertIn("self.sink_bindings_by_attr: Dict[str, List[\"ACESinkBinding\"]] = {}", tasks_src)
        self.assertIn("class ACESinkBinding:", tasks_src)

    def test_preview_conflict_resolution_selects_single_global_manual_preview(self):
        engine_src = _read("audio_engine.py")
        abstract_src = _read("abstract_tree.py")

        self.assertIn("discovery_index: int = 0", abstract_src)
        self.assertIn("def _resolve_global_manual_preview_winner(", engine_src)
        self.assertIn("Multiple Audio Preview nodes are present.", engine_src)
        self.assertIn("manual_preview_winner_id", engine_src)
        self.assertIn("if has_manual_preview:", engine_src)
        self.assertIn("graph.preview_task_ids = list(graph.manual_preview_task_ids)", engine_src)
        self.assertIn("if not has_manual_preview:", engine_src)

    def test_stage_a_and_stage_c_mutators_assert_main_thread(self):
        engine_src = _read("audio_engine.py")
        proxy_src = _read("proxy_management.py")

        compile_body = _slice_between(
            engine_src,
            "def compile(self, abstract_tree: ACEAbstractTree, blender_scene):",
            "def _log_runtime(",
            label="compile body",
        )
        self.assertIn('utils.assert_main_thread("compile")', compile_body)

        ensure_layout_body = _slice_between(
            proxy_src,
            "def ensure_sink_layout(self, graph, active_keys: Set[str] | None = None):",
            "def commit_results(",
            label="ensure_sink_layout body",
        )
        self.assertIn('utils.assert_main_thread("ensure_sink_layout")', ensure_layout_body)

        commit_body = _slice_between(
            proxy_src,
            "def commit_results(self, graph, results: Dict[str, Any], active_keys: Set[str] | None = None):",
            "def update_proxy_object(",
            label="commit_results body",
        )
        self.assertIn('utils.assert_main_thread("commit_results")', commit_body)

    def test_stage_c_runtime_paths_guard_unsafe_render_writes(self):
        engine_src = _read("audio_engine.py")
        compile_body = _slice_between(
            engine_src,
            "def compile(self, abstract_tree: ACEAbstractTree, blender_scene):",
            "def _log_runtime(",
            label="compile body",
        )
        self.assertIn(
            "sink_writes_allowed = not self._unsafe_for_handler_writes(blender_scene)",
            compile_body,
        )
        self.assertIn("if self.writer is not None and sink_writes_allowed:", compile_body)
        self.assertIn("Skipping ACE sink layout writes during render.", compile_body)

        execute_body = _slice_between(
            engine_src,
            "def execute_next_frame",
            "def bake_source_to_cache",
            label="execute_next_frame body",
        )
        self.assertIn("if self._unsafe_for_handler_writes(scene):", execute_body)
        self.assertIn("Skipping ACE frame writes during render.", execute_body)

    def test_object_property_sink_updates_tag_depsgraph(self):
        proxy_src = _read("proxy_management.py")
        self.assertIn("obj.update_tag(refresh={\"OBJECT\", \"DATA\"})", proxy_src)
        self.assertIn("elif frontend_kind == FRONTEND_KIND_SHADER:", proxy_src)

    def test_scene_and_membership_signatures_ignore_discovery_order_metadata(self):
        engine_src = _read("audio_engine.py")

        tree_sig_body = _slice_between(
            engine_src,
            "def _compute_tree_signature(abstract_tree: ACEAbstractTree):",
            "def _build_auto_attr_name(",
            label="_compute_tree_signature body",
        )
        self.assertNotIn("discovery_index", tree_sig_body)

        membership_body = _slice_between(
            engine_src,
            "def _frontend_membership_signature(frontends: list[_ACEFrontendDiscovery]) -> tuple[Any, ...]:",
            "def _refresh_consumer_intents_from_nodes(",
            label="_frontend_membership_signature body",
        )
        self.assertNotIn("discovery_index", membership_body)

    def test_frontend_uid_repair_runs_on_canonical_frontend_order(self):
        engine_src = _read("audio_engine.py")
        discover_body = _slice_between(
            engine_src,
            "def discover_scene_frontends(self, scene) -> list[_ACEFrontendDiscovery]:",
            "@staticmethod\n    def _is_output_node(",
            label="discover_scene_frontends body",
        )
        self.assertIn("def _frontend_record_sort_key(", discover_body)
        self.assertIn("for rec in sorted(by_ptr.values(), key=_frontend_record_sort_key):", discover_body)

    def test_depsgraph_handler_tracks_geometry_shader_tree_and_object_frontends(self):
        engine_src = _read("audio_engine.py")
        handler_body = _slice_between(
            engine_src,
            "def depsgraph_update_handler(",
            "def auto_compile_timer(",
            label="depsgraph_update_handler body",
        )
        self.assertIn("bl_idname in {\"GeometryNodeTree\", \"ShaderNodeTree\"}", handler_body)
        self.assertIn("if bl_idname == \"Object\":", handler_body)
        self.assertIn("if bl_idname == \"Material\":", handler_body)
        self.assertIn("recompile_targets: list = []", handler_body)
        self.assertIn("seen_targets: set[int] = set()", handler_body)
        self.assertIn("def _queue_recompile_target(tree_obj) -> None:", handler_body)
        self.assertIn("if ACEAudioEngine._tree_is_ace_managed(_id):", handler_body)
        self.assertNotIn("recompile_targets.append(None)", handler_body)
        self.assertIn("for target in recompile_targets:", handler_body)
        self.assertIn("engine.request_recompile(target)", handler_body)

    def test_preview_numba_is_forced_when_dependency_is_available(self):
        tasks_src = _read("audio_tasks_registry.py")
        numba_body = _slice_between(
            tasks_src,
            "def _numba_turbo_enabled(self) -> bool:",
            "@staticmethod\n    def _resolve_bounds(",
            label="_numba_turbo_enabled body",
        )
        self.assertIn("return bool(NUMBA_AVAILABLE)", numba_body)
        self.assertNotIn("preview_numba_turbo", numba_body)

    def test_source_specs_outputs_frames_per_beat_and_no_time_signature(self):
        tasks_src = _read("audio_tasks_registry.py")
        self.assertIn("frames_per_beat = float(1.0 / beats_per_frame)", tasks_src)
        self.assertIn("return (frames_per_beat, beat_phase, chunk_in)", tasks_src)
        self.assertNotIn("Time Signature", tasks_src)

    def test_preview_sink_uses_strict_time_domain_path(self):
        tasks_src = _read("audio_tasks_registry.py")
        preview_body = _slice_between(
            tasks_src,
            "class PreviewSinkTask",
            "class AudioMathTask",
            label="PreviewSinkTask definition",
        )
        self.assertNotIn("np.fft.rfft", preview_body)
        self.assertNotIn("np.fft.irfft", preview_body)
        self.assertNotIn("_masked_ifft", preview_body)

    def test_parallel_payload_contract_uses_keyed_source_chunks(self):
        engine_src = _read("audio_engine.py")
        self.assertIn("\"source_chunks\": payload_chunks", engine_src)
        self.assertIn("\"source_chunks_shm\"", engine_src)
        self.assertIn("source_chunks_for_worker", engine_src)
        payload_builder = _slice_between(
            engine_src,
            "def _build_parallel_source_payload",
            "def _parallel_drop_future",
            label="parallel source payload builder",
        )
        self.assertNotIn("\"lineage\"", payload_builder)
        self.assertNotIn("\"analysis_ops\"", payload_builder)

        worker_src = _read("parallel_dsp.py")
        self.assertIn("source_chunks_raw = dict(job.get(\"source_chunks\", {}) or {})", worker_src)
        self.assertIn("def set_source_payloads(", worker_src)
        self.assertIn("source_chunks_shm", worker_src)

    def test_preview_latency_controls_exist(self):
        props_src = _read("properties.py")
        self.assertIn("live_latency_ms", props_src)
        self.assertIn("default=10", props_src)
        self.assertIn("ace_preview_block_size", props_src)
        self.assertNotIn("legacy_preview_fft", props_src)

        engine_src = _read("audio_engine.py")
        self.assertIn("_preview_block_size(", engine_src)
        self.assertIn("preview_interval_s", engine_src)
        self.assertIn("_preview_next_eval_ts", engine_src)

    def test_perf_baseline_report_has_full_required_matrix(self):
        baseline = json.loads(_read("tests/perf_baseline_report.json"))
        scenarios = dict(baseline.get("scenarios", {}) or {})
        required = ("1", "4", "12", "24", "36", "50")
        stages = ("source_read", "dsp", "preview", "proxy_write", "ui_sync")
        for sid in required:
            self.assertIn(sid, scenarios)
            for stage in stages:
                self.assertIn(stage, scenarios[sid])

    def test_engine_perf_harness_exists_for_execute_next_frame_gate(self):
        src = _read("tests/engine_perf_harness.py")
        self.assertIn("generate_engine_perf_report", src)
        self.assertIn("ACEAudioEngine", src)
        self.assertIn("execute_next_frame", src)
        self.assertIn("\"processes_dsp\"", src)
        self.assertIn("parallel_output_shards", src)
        self.assertIn("ENGINE_PERF_STRICT_DSP_RATIO_MAX = 1.10", src)
        self.assertIn("ENGINE_PERF_STRICT_QUEUE_DROP_RATE_MAX = 0.03", src)
        self.assertIn("ENGINE_PERF_STRICT_FRESH_RESULT_RATE_MIN = 0.90", src)
        self.assertIn("ENGINE_PERF_SLA_SAMPLE_RATE_HZ = 96000", src)
        self.assertIn("ENGINE_PERF_SLA_PASS_TOTAL_LATENCY_MS = 15.0", src)
        self.assertIn("ENGINE_PERF_SLA_IDEAL_TOTAL_LATENCY_MS = 10.0", src)
        self.assertIn("\"sla_topology\":", src)
        self.assertIn("AMD Ryzen 7", src)
        self.assertIn("_engine_perf_strict_enabled", src)

        gate_src = _read("tests/test_engine_perf_gate.py")
        self.assertIn("dsp_p95_ratio_processes_vs_threads", gate_src)
        self.assertIn("sla_topology", gate_src)
        self.assertIn("ACE_ENGINE_PERF_GATE_STRICT", gate_src)
        self.assertIn("zero/placeholder value", gate_src)

        perf_src = _read("tests/perf_harness.py")
        self.assertIn("PERF_STRICT_PREVIEW_P95_MS = 10.0", perf_src)
        self.assertIn("_validate_perf_report_strict", perf_src)


    def test_masked_mono_signal_layer2_guard_and_not_reachable_from_preview(self):
        tasks_src = _read("audio_tasks_registry.py")
        # Layer-2-only guard comment must be present on the function.
        self.assertIn("# [LAYER-2-ONLY: DO NOT CALL FROM PREVIEW PATH]", tasks_src)
        # The iFFT is present somewhere in the file (in _masked_mono_signal itself).
        self.assertIn("np.fft.irfft", tasks_src)
        # PreviewSinkTask.process() must not contain _masked_mono_signal or irfft.
        preview_body = _slice_between(
            tasks_src,
            "class PreviewSinkTask",
            "class AudioMathTask",
            label="PreviewSinkTask definition",
        )
        self.assertNotIn("_masked_mono_signal", preview_body)
        self.assertNotIn("np.fft.irfft", preview_body)

    def test_probe_offloaded_from_main_thread(self):
        engine_src = _read("audio_engine.py")
        self.assertIn("def _dispatch_async_probe(", engine_src)
        self.assertIn("def _run_probe_in_bg(", engine_src)
        self.assertIn("def _flush_probe_updates(", engine_src)
        self.assertIn("self._probe_pending_updates", engine_src)
        self.assertIn("self._probe_thread_lock", engine_src)
        self.assertIn("self._probe_in_flight", engine_src)
        # Probe dispatch must not open AudioReception inline (no blocking I/O on main thread).
        probe_body = _slice_between(
            engine_src,
            "def _probe_source_metadata(",
            "def _dispatch_async_probe(",
            label="_probe_source_metadata body",
        )
        self.assertNotIn("AudioReception(", probe_body)
        # The background worker is the only place AudioReception is opened for probing.
        bg_body = _slice_between(
            engine_src,
            "def _run_probe_in_bg(",
            "def _flush_probe_updates(",
            label="_run_probe_in_bg body",
        )
        self.assertIn("AudioReception(", bg_body)
        self.assertIn("daemon=True", engine_src)

    def test_parallel_stale_reuse_is_time_based(self):
        engine_src = _read("audio_engine.py")
        self.assertIn("_PARALLEL_STALE_REUSE_MS", engine_src)
        self.assertNotIn("_PARALLEL_STALE_REUSE_WINDOW_FRAMES", engine_src)
        self.assertIn("_PARALLEL_STALE_REUSE_MS * fps_now / 1000.0", engine_src)

    def test_inflight_per_source_within_spec(self):
        engine_src = _read("audio_engine.py")
        # _PARALLEL_MAX_INFLIGHT_PER_SOURCE must not exceed 2 (spec mandates maxlen 1–2).
        import re
        match = re.search(r"_PARALLEL_MAX_INFLIGHT_PER_SOURCE\s*=\s*(\d+)", engine_src)
        self.assertIsNotNone(match, "_PARALLEL_MAX_INFLIGHT_PER_SOURCE constant not found")
        self.assertLessEqual(int(match.group(1)), 2)

    def test_parallel_worker_processes_terminated_on_shutdown(self):
        engine_src = _read("audio_engine.py")
        shutdown_body = _slice_between(
            engine_src,
            "def _shutdown_parallel_pool(",
            "def _parallel_reset_generation(",
            label="_shutdown_parallel_pool body",
        )
        self.assertIn("proc.terminate()", shutdown_body)
        self.assertIn("_processes", shutdown_body)



    # ------------------------------------------------------------------
    # Gap-fix regression tests (audit 2026-03-01)
    # ------------------------------------------------------------------

    def test_worker_specs_reads_injected_bpm_fps(self):
        """Gap 1: _WorkerSourceSpecsPassthroughTask must read from params, not literals."""
        worker_src = _read("parallel_dsp.py")
        specs_body = _slice_between(
            worker_src,
            "class _WorkerSourceSpecsPassthroughTask",
            "class _WorkerSourceProvider",
            label="_WorkerSourceSpecsPassthroughTask body",
        )
        # Must read dynamic values from self.parameters
        self.assertIn("self.parameters.get(\"scene_bpm\"", specs_body)
        self.assertIn("self.parameters.get(\"scene_fps\"", specs_body)
        self.assertIn("self.parameters.get(\"scene_beats_per_bar\"", specs_body)
        self.assertIn("frames_per_beat = float(1.0 / beats_per_frame)", specs_body)
        # Must NOT have bare hardcoded literals assigned to bpm / fps variables
        self.assertNotIn("bpm = 120.0\n", specs_body)
        self.assertNotIn("fps = 24.0\n", specs_body)

        # Engine must inject scene values into src_specs IR params.
        engine_src = _read("audio_engine.py")
        ir_body = _slice_between(
            engine_src,
            "def _prepare_parallel_graph_ir(",
            "def _build_parallel_source_payload(",
            label="_prepare_parallel_graph_ir body",
        )
        self.assertIn("scene_bpm", ir_body)
        self.assertIn("scene_fps", ir_body)
        self.assertIn("scene_beats_per_bar", ir_body)
        self.assertIn("node.type == \"src_specs\"", ir_body)
        # Engine must capture scene BPM at frame evaluation time.
        execute_body = _slice_between(
            engine_src,
            "def execute_next_frame(",
            "def bake_source_to_cache(",
            label="execute_next_frame body",
        )
        self.assertIn("_current_scene_bpm", execute_body)
        self.assertIn("_current_scene_beats_per_bar", execute_body)

    def test_delta_nodes_set_band_metadata_for_preview_iir(self):
        """Gap 2 (re-audit, already solved): band metadata chain is complete."""
        tasks_src = _read("audio_tasks_registry.py")
        # FrequencySplitterTask must tag outputs with band_hz_bounds
        split_body = _slice_between(
            tasks_src,
            "class FrequencySplitterTask",
            "class MidsideSplitterTask",
            label="FrequencySplitterTask body",
        )
        self.assertIn("band_hz_bounds=(low_cut, high_cut)", split_body)
        # ChromagramTask must tag outputs with band_indices
        chroma_body = _slice_between(
            tasks_src,
            "class ChromagramTask",
            "class AutoComponentSplitterTask",
            label="ChromagramTask body",
        )
        self.assertIn("band_indices=band_indices", chroma_body)
        # PreviewSinkTask must react to both band_hz_bounds and band_indices
        preview_body = _slice_between(
            tasks_src,
            "class PreviewSinkTask",
            "class AudioMathTask",
            label="PreviewSinkTask definition",
        )
        self.assertIn("_resolve_bounds(chunk)", preview_body)
        self.assertIn("_resolve_sparse_mask(chunk", preview_body)
        self.assertIn("_ensure_iir_setup(", preview_body)
        self.assertIn("_mask_to_sections(", preview_body)
        # Ensure no FFT reconstruction slipped in
        self.assertNotIn("np.fft.irfft", preview_body)

    def test_render_guard_uses_blender_job_api(self):
        """Gap 3: _is_rendering_job must use bpy.app.is_job_running and be documented."""
        engine_src = _read("audio_engine.py")
        is_rendering_body = _slice_between(
            engine_src,
            "def _is_rendering_job(",
            "def _unsafe_for_handler_writes(",
            label="_is_rendering_job body",
        )
        self.assertIn("is_job_running", is_rendering_body)
        self.assertIn('"RENDER"', is_rendering_body)
        # Must have a docstring (first non-blank line after def should be a quote)
        stripped = is_rendering_body.strip()
        self.assertTrue(
            stripped.startswith('"""') or '"""' in stripped[:120],
            "Expected docstring in _is_rendering_job",
        )
        # Must actually be called somewhere in the engine
        self.assertIn("_is_rendering_job()", engine_src)

    def test_shader_viewport_preview_guard_defers_structural_shader_writes(self):
        engine_src = _read("audio_engine.py")
        self.assertIn("def _is_shader_viewport_preview_active(", engine_src)
        self.assertIn("def _unsafe_for_shader_tree_mutations(", engine_src)
        self.assertIn('"RENDERED"', engine_src)
        self.assertIn('"MATERIAL"', engine_src)
        maybe_compile_body = _slice_between(
            engine_src,
            "def _maybe_compile_dirty(",
            "def request_recompile(",
            label="_maybe_compile_dirty body",
        )
        self.assertIn("touches_shader_frontends", maybe_compile_body)
        self.assertIn("self._unsafe_for_shader_tree_mutations(scene)", maybe_compile_body)
        compile_body = _slice_between(
            engine_src,
            "def compile_scene_frontends(",
            "def compile_from_blender_tree(",
            label="compile_scene_frontends body",
        )
        self.assertIn("has_shader_frontends", compile_body)
        self.assertIn("self.request_recompile(target_tree)", compile_body)
        sync_body = _slice_between(
            engine_src,
            "def _sync_shader_output_ids(",
            "def _create_task_for_node(",
            label="_sync_shader_output_ids body",
        )
        self.assertIn("shader_tree_mutations_unsafe = self._unsafe_for_shader_tree_mutations(scene)", sync_body)
        self.assertIn("if shader_factory is not None and not shader_tree_mutations_unsafe:", sync_body)

    def test_proxy_and_menu_defer_shader_tree_sync_during_viewport_preview(self):
        proxy_src = _read("proxy_management.py")
        self.assertIn("def _shader_tree_writes_are_unsafe(", proxy_src)
        self.assertIn("def _defer_shader_tree_sync(", proxy_src)
        self.assertIn("def sync_managed_source_tree(tree: bpy.types.NodeTree, sources, scene=None):", proxy_src)
        self.assertIn("ACEProxyManager._shader_tree_writes_are_unsafe(scene)", proxy_src)
        self.assertIn("ACEProxyManager._defer_shader_tree_sync(tree, scene)", proxy_src)
        self.assertIn("ACEProxyManager.sync_managed_source_tree(tree, sources or [], scene=scene)", proxy_src)

        menu_src = _read("gn_menu.py")
        self.assertIn("ACEAudioEngine._unsafe_for_shader_tree_mutations(scene)", menu_src)
        self.assertIn("engine.request_recompile(tree)", menu_src)
        self.assertIn("ACEShaderNodeFactory.sync_node_instance_group_bindings(", menu_src)

    def test_parallel_shard_respects_source_groups(self):
        """Gap 4: shard function must accept source_groups and bucket by source."""
        engine_src = _read("audio_engine.py")
        shard_body = _slice_between(
            engine_src,
            "def _parallel_shard_attr_keys(",
            "def _file_working_sample_rate_cap(",
            label="_parallel_shard_attr_keys body",
        )
        # Signature must accept source_groups
        self.assertIn("source_groups", shard_body)
        # Must bucket by group when source_groups provided
        self.assertIn("buckets", shard_body)
        self.assertIn("ungrouped", shard_body)
        # Single-source fallback must be preserved
        self.assertIn("return [keys]", shard_body)

        # _prepare_parallel_graph_ir must compute and store the groupings
        ir_body = _slice_between(
            engine_src,
            "def _prepare_parallel_graph_ir(",
            "def _build_parallel_source_payload(",
            label="_prepare_parallel_graph_ir body",
        )
        self.assertIn("attr_key_source_group", ir_body)
        self.assertIn("parallel_output_shard_groups", ir_body)
        self.assertIn("source_groups=attr_key_source_group", ir_body)


if __name__ == "__main__":
    unittest.main()
