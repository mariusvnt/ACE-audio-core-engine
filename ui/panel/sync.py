"""N-panel state kept in sync with the active tree and the engine by a timer."""

import bpy  # type: ignore
import time
import hashlib
try:
    from bpy.app.handlers import persistent  # type: ignore
except ImportError:  # pragma: no cover - Blender provides this at runtime
    def persistent(fn):
        return fn
from ...core.id_management import ACEIdManager
from ...core.engine.engine import ACEAudioEngine
from ...nodes.geometry.reader import ACEDummyTreeReader
from ...nodes.shader.reader import ACEShaderTreeReader
from ... import utils
from .helpers import _build_ui_node_lookup_ids, _describe_node, _find_active_node_tree, _find_tree_node_by_ace_id, _node_type_display, _tree_has_ace_nodes, _tree_pointer, _ui_is_blocked, _ui_node_lookup_id


_UI_SYNC_MIN_INTERVAL = 0.25


_UI_SYNC_TIMER_INTERVAL_ACTIVE = 0.25


_UI_SYNC_TIMER_INTERVAL_IDLE = 1.0


_UI_DIRTY_REBUILD_MIN_INTERVAL = 0.6


_UI_METADATA_PROBE_INTERVAL = 5.0


_UI_LATENCY_UPDATE_INTERVAL_S = 1.0


_ui_sync_last_ts = 0.0


_ui_sync_last_key = ""


_ui_timer_last_quick_key = ""


_ui_timer_last_sync_ts = 0.0


_ui_timer_last_dirty_rebuild_ts = 0.0


_ui_source_probe_last: dict[str, float] = {}


_ui_latency_cache = {
    "scene_key": 0,
    "source_uid": "",
    "ts": 0.0,
    "metrics": None,
}


_ui_tree_cache = {
    "tree_ptr": 0,
    "topology_sig": "",
    "sig_ts": 0.0,
    "source_uid": "",
    "source_proxy": "",
    "nodes": (),
    "lookup_ids": {},
}


_has_ace_nodes_in_trees: bool = False


def _refresh_ace_nodes_in_trees_flag(scene) -> None:
    """Update the module-level ``_has_ace_nodes_in_trees`` flag."""
    global _has_ace_nodes_in_trees
    if scene is None:
        _has_ace_nodes_in_trees = False
        return
    for obj in getattr(scene, "objects", []) or []:
        for mod in getattr(obj, "modifiers", []) or []:
            if getattr(mod, "type", None) != "NODES":
                continue
            if _tree_has_ace_nodes(getattr(mod, "node_group", None)):
                _has_ace_nodes_in_trees = True
                return
        for slot in getattr(obj, "material_slots", []) or []:
            mat = getattr(slot, "material", None)
            tree = getattr(mat, "node_tree", None) if mat is not None else None
            if _tree_has_ace_nodes(tree):
                _has_ace_nodes_in_trees = True
                return
    _has_ace_nodes_in_trees = False


def _resolve_ui_list_selected_node(context):
    """Resolve the node selected in the active graph list shown in the N-panel."""
    from ...utils import _get_active_source

    _, active_source = _get_active_source(context)
    if not active_source:
        return None, None
    idx = int(getattr(active_source, "active_ui_node_index", -1))
    if idx < 0 or idx >= len(getattr(active_source, "ui_nodes", [])):
        return None, None

    ui_item = active_source.ui_nodes[idx]
    node_id = str(getattr(ui_item, "node_id", "") or "")
    if not node_id:
        return None, None

    tree = _find_active_node_tree(context, editor_context_only=True)
    node, parsed = _find_tree_node_by_ace_id(tree, node_id)
    if node and parsed:
        return node, parsed

    lookup_ids = {}
    tree_ptr = _tree_pointer(tree)
    if tree and int(_ui_tree_cache.get("tree_ptr", 0) or 0) == int(tree_ptr):
        lookup_ids = dict(_ui_tree_cache.get("lookup_ids", {}) or {})
    if not lookup_ids and tree:
        lookup_ids = _build_ui_node_lookup_ids(tree)
    mapped_id = str(lookup_ids.get(node_id, "") or "")
    if mapped_id:
        node, parsed = _find_tree_node_by_ace_id(tree, mapped_id)
        if node and parsed:
            return node, parsed

    # A node still active in another editor counts as the selection.
    active_node = getattr(context, "active_node", None)
    active_id = str(active_node.get("ace_id", "") or "") if active_node and hasattr(active_node, "get") else ""
    if active_node and active_id == node_id:
        return active_node, parsed or ACEIdManager.parse_node_id(node_id)
    return None, parsed


def _ui_tree_topology_signature(tree) -> str:
    """Compute a short content hash representing the current GN tree topology."""
    if tree is None:
        return ""

    attr_socket_names = {
        "Attribute ID",
        "Frames per Beat Attribute ID",
        "Beat Phase Attribute ID",
    }
    node_tokens = []
    for node in getattr(tree, "nodes", []):
        ptr = _tree_pointer(node)
        node_type = str(getattr(node, "type", "") or "")
        node_name = str(getattr(node, "name", "") or "")
        node_tree_name = str(getattr(getattr(node, "node_tree", None), "name", "") or "")

        ace_id = ""
        source_id = ""
        op_value = ""
        preview_enabled = True
        try:
            ace_id = str(node.get("ace_id", "") or "")
        except Exception as exc:
            utils.note_swallowed("sync._ui_tree_topology_signature", exc)
            ace_id = ""
        if ace_id:
            try:
                source_id = str(node.get("ace_source_id", "") or "")
            except Exception as exc:
                utils.note_swallowed("sync._ui_tree_topology_signature#2", exc)
                source_id = ""
            try:
                op_value = str(node.get("ace_op", "") or "")
            except Exception as exc:
                utils.note_swallowed("sync._ui_tree_topology_signature#3", exc)
                op_value = ""
            try:
                preview_enabled = bool(node.get("ace_preview_enabled", True))
            except Exception as exc:
                utils.note_swallowed("sync._ui_tree_topology_signature#4", exc)
                preview_enabled = True

        proxy_name = ""
        attr_defaults = []
        for sock in getattr(node, "inputs", []):
            sock_name = str(getattr(sock, "name", "") or "")
            if sock_name == "Proxy Object" and not bool(getattr(sock, "is_linked", False)):
                try:
                    proxy_name = str(getattr(getattr(sock, "default_value", None), "name", "") or "")
                except Exception as exc:
                    utils.note_swallowed("sync._ui_tree_topology_signature#5", exc)
                    proxy_name = ""
            if sock_name not in attr_socket_names or bool(getattr(sock, "is_linked", False)):
                continue
            raw_value = getattr(sock, "default_value", "")
            if hasattr(raw_value, "name"):
                raw_value = getattr(raw_value, "name", "")
            attr_defaults.append((sock_name, str(raw_value)))
        attr_defaults.sort()

        node_tokens.append(
            (
                ptr,
                ace_id,
                node_type,
                node_tree_name,
                node_name,
                source_id,
                proxy_name,
                op_value,
                int(preview_enabled),
                tuple(attr_defaults),
                len(getattr(node, "inputs", [])),
                len(getattr(node, "outputs", [])),
            )
        )
    node_tokens.sort()

    link_tokens = []
    for link in getattr(tree, "links", []):
        from_node = getattr(link, "from_node", None)
        to_node = getattr(link, "to_node", None)
        if from_node is None or to_node is None:
            continue

        from_socket = getattr(link, "from_socket", None)
        to_socket = getattr(link, "to_socket", None)
        from_sock_token = str(
            getattr(from_socket, "identifier", "") or getattr(from_socket, "name", "") or ""
        )
        to_sock_token = str(
            getattr(to_socket, "identifier", "") or getattr(to_socket, "name", "") or ""
        )
        try:
            from_sock_idx = int(getattr(from_socket, "index", -1))
        except Exception as exc:
            utils.note_swallowed("sync._ui_tree_topology_signature#6", exc)
            from_sock_idx = -1
        try:
            to_sock_idx = int(getattr(to_socket, "index", -1))
        except Exception as exc:
            utils.note_swallowed("sync._ui_tree_topology_signature#7", exc)
            to_sock_idx = -1

        link_tokens.append(
            (
                _tree_pointer(from_node),
                _tree_pointer(to_node),
                from_sock_idx,
                to_sock_idx,
                from_sock_token,
                to_sock_token,
            )
        )
    link_tokens.sort()

    payload = repr((tuple(node_tokens), tuple(link_tokens))).encode("utf-8", "ignore")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _ui_nodes_match(active_source, node_ids: list[str]) -> bool:
    try:
        ui_nodes = getattr(active_source, "ui_nodes", None)
        if ui_nodes is None or len(ui_nodes) != len(node_ids):
            return False
        for idx, node_id in enumerate(node_ids):
            if str(getattr(ui_nodes[idx], "node_id", "") or "") != str(node_id or ""):
                return False
        return True
    except Exception as exc:
        utils.note_swallowed("sync._ui_nodes_match", exc)
        return False


def _sync_ui_dummy_tree(context, force: bool = False, allow_dirty_rebuild: bool = True):
    global _ui_sync_last_ts, _ui_sync_last_key, _ui_tree_cache

    ace = getattr(context.scene, "audiocore", None)
    if not ace:
        utils.log("UI sync skipped: no audiocore on scene.", "DEBUG")
        return
    _, active_source = utils._get_active_source(context)
    if not active_source:
        utils.log("UI sync skipped: no active source.", "DEBUG")
        return
    with utils.logged_errors("sync._sync_ui_dummy_tree"):
        ACEIdManager.ensure_source_uid(active_source)
    # Abort if collection is not writable in the current context (e.g., during draw in some areas)
    if not hasattr(active_source, "ui_nodes"):
        utils.log("UI sync skipped: active_source has no ui_nodes property.", "DEBUG")
        return

    tree = _find_active_node_tree(context, editor_context_only=True)
    source_uid = getattr(active_source, "stable_id", "") or ""
    source_proxy = getattr(active_source, "proxy_object_name", "") or ""
    tree_ptr = _tree_pointer(tree)
    now = time.monotonic()
    cache_scope_match = (
        bool(tree)
        and int(_ui_tree_cache.get("tree_ptr", 0) or 0) == int(tree_ptr)
        and str(_ui_tree_cache.get("source_uid", "") or "") == source_uid
        and str(_ui_tree_cache.get("source_proxy", "") or "") == source_proxy
    )

    tree_dirty_hint = bool(force)
    engine = ACEAudioEngine.get_instance(context.scene)
    engine_abstract = None
    engine_tree_ready = False
    if tree and engine:
        tree_ptr = 0
        try:
            tree_ptr = int(tree.as_pointer())
        except Exception as exc:
            utils.note_swallowed("sync._sync_ui_dummy_tree#2", exc)
            tree_ptr = 0
        pending_tree_ptr = int(getattr(getattr(engine, "frontends", None), "_pending_tree_ptr", 0) or 0)
        dirty_tree_ptrs = getattr(getattr(engine, "frontends", None), "_dirty_tree_ptrs", set()) or set()
        dirty_tree_match = bool((tree_ptr and pending_tree_ptr == tree_ptr) or (tree_ptr and tree_ptr in dirty_tree_ptrs))
        if allow_dirty_rebuild:
            tree_dirty_hint = tree_dirty_hint or dirty_tree_match
        if not tree_dirty_hint:
            compiled_ptr = int(getattr(getattr(engine, "compiler", None), "_compiled_tree_ptr", 0) or 0)
            # Any compiled frontend tree uses the merged abstract tree.
            compiled_frontend_ptrs = set(
                (getattr(getattr(engine, "compiler", None), "_compiled_frontend_ptrs", {}) or {}).values()
            )
            if compiled_ptr == int(tree_ptr) or int(tree_ptr) in compiled_frontend_ptrs:
                engine_abstract = getattr(engine, "abstract_tree_ref", None)
                engine_tree_ready = bool(engine_abstract)

    rebuild_topology = bool(
        tree
        and (
            force
            or (not cache_scope_match)
            or tree_dirty_hint
        )
    )
    topology_sig = ""
    if tree:
        if rebuild_topology:
            if engine_tree_ready and engine is not None:
                # Avoid Python's process-randomized hash() so UI topology cache
                # keys remain stable across sessions.
                engine_sig_seed = str(getattr(getattr(engine, "compiler", None), "_last_tree_signature", None)).encode("utf-8")
                engine_sig = hashlib.blake2b(engine_sig_seed, digest_size=8).hexdigest()
                topology_sig = f"eng:{engine_sig}"
            else:
                topology_sig = _ui_tree_topology_signature(tree)
        else:
            topology_sig = str(_ui_tree_cache.get("topology_sig", "") or "")
    else:
        rebuild_topology = True
        topology_sig = ""
    sync_key = (
        f"{source_uid}|{source_proxy}|"
        f"{getattr(tree, 'name', '') if tree else ''}|{tree_ptr}"
    )
    if not force and sync_key == _ui_sync_last_key and (now - _ui_sync_last_ts) < _UI_SYNC_MIN_INTERVAL:
        return

    nodes = []
    node_lookup_ids = {}
    if (not rebuild_topology) and cache_scope_match:
        nodes = list(_ui_tree_cache.get("nodes", ()) or ())
        node_lookup_ids = dict(_ui_tree_cache.get("lookup_ids", {}) or {})
    elif tree:
        if engine_tree_ready and engine_abstract is not None:
            abstract = engine_abstract
            node_lookup_ids = _build_ui_node_lookup_ids(tree)
        else:
            if str(getattr(tree, "bl_idname", "") or "") == "ShaderNodeTree":
                reader = ACEShaderTreeReader(tree)
            else:
                reader = ACEDummyTreeReader(tree)
            abstract = reader.build()
            node_lookup_ids = _build_ui_node_lookup_ids(tree, reader=reader)
        nodes = list(abstract.nodes.values())
        if nodes:
            bl_order: dict[str, int] = {}
            if tree:
                for _idx, _bl_node in enumerate(getattr(tree, "nodes", [])):
                    try:
                        _ace_id = str(_bl_node.get("ace_id", "") or "")
                    except Exception as exc:
                        utils.note_swallowed("sync._sync_ui_dummy_tree#3", exc)
                        _ace_id = ""
                    if _ace_id:
                        bl_order[_ace_id] = _idx

            def _creation_sort_key(n, _bl_order=bl_order, _lu=node_lookup_ids):
                resolved = str(_lu.get(n.id, "") or n.id or "")
                return _bl_order.get(resolved, _bl_order.get(n.id, 999999))

            nodes.sort(key=_creation_sort_key)
            nodes = [
                node
                for node in nodes
                if str(getattr(node, "category", "") or "") != "source_mngmt"
            ]
        _ui_tree_cache = {
            "tree_ptr": int(tree_ptr),
            "topology_sig": topology_sig,
            "sig_ts": float(now),
            "source_uid": source_uid,
            "source_proxy": source_proxy,
            "nodes": tuple(nodes),
            "lookup_ids": dict(node_lookup_ids),
        }
    else:
        _ui_tree_cache = {
            "tree_ptr": 0,
            "topology_sig": "",
            "sig_ts": float(now),
            "source_uid": "",
            "source_proxy": "",
            "nodes": (),
            "lookup_ids": {},
        }

    nodes = [
        node
        for node in nodes
        if str(getattr(node, "category", "") or "") != "source_mngmt"
    ]
    if not nodes:
        if not tree:
            utils.log("UI sync skipped: no active ACE node tree found.", "DEBUG")
            try:
                active_source.ui_nodes.clear()
            except Exception as e:
                utils.log(f"UI sync could not clear ui_nodes: {e}", "DEBUG")
            _ui_sync_last_key = sync_key
            _ui_sync_last_ts = now
            return

    utils.log(f"UI sync found {len(nodes)} nodes for active source.", "DEBUG")

    proxy = bpy.data.objects.get(getattr(active_source, "proxy_object_name", "")) if hasattr(active_source, "proxy_object_name") else None

    def _resolve_attr_name(node_obj):
        attr_id = None
        attr_custom = False
        if isinstance(getattr(node_obj, "parameters", None), dict):
            attr_id = node_obj.parameters.get("attr_id_out_0")
            attr_custom = bool(node_obj.parameters.get("attr_id_custom_out_0", False))
        if attr_custom and isinstance(attr_id, str) and attr_id.strip():
            name = attr_id.strip()
            if not name.startswith(ACEIdManager.PREFIX):
                name = ACEIdManager.PREFIX + ACEIdManager._clean_string(name)
            return name[:63]

        if not proxy:
            return None
        if not getattr(node_obj, "outputs", None):
            return None
        out = node_obj.outputs[0]
        return ACEIdManager.generate_compact_attr_name(
            proxy.name,
            node_obj.id or node_obj.type,
            out.name or out.id or "out0",
            0,
        )[:63]

    def _read_attr_value(mesh, name):
        import math
        attr = mesh.attributes.get(name) if mesh else None
        if not attr or not attr.data:
            return None
        try:
            if attr.data_type == 'FLOAT':
                return float(attr.data[0].value)
            if attr.data_type == 'FLOAT_VECTOR':
                vec = attr.data[0].vector
                return float(math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2))
            if attr.data_type == 'FLOAT_COLOR':
                col = attr.data[0].color
                return float(sum(col) / len(col))
        except Exception as exc:
            utils.note_swallowed("sync._sync_ui_dummy_tree._read_attr_value", exc)
            return None
        return None

    def _write_ui_item(item, node):
        item.name = node.name or node.id
        item.node_id = _ui_node_lookup_id(node, node_lookup_ids)
        item.node_type = _node_type_display(node.category, node.type)
        value = None
        if proxy and getattr(proxy, "data", None):
            attr_name = _resolve_attr_name(node)
            if attr_name:
                value = _read_attr_value(proxy.data, attr_name)
        item.current_value = float(value) if value is not None else 0.0
        item.description = _describe_node(node.category, node.type)

    node_ids = [_ui_node_lookup_id(node, node_lookup_ids) for node in nodes]
    reuse_ui_rows = _ui_nodes_match(active_source, node_ids)
    if reuse_ui_rows:
        for idx, node in enumerate(nodes):
            try:
                item = active_source.ui_nodes[idx]
            except Exception as exc:
                utils.note_swallowed("sync._sync_ui_dummy_tree#4", exc)
                continue
            _write_ui_item(item, node)
    else:
        try:
            active_source.ui_nodes.clear()
        except Exception as e:
            utils.log(f"UI sync could not clear ui_nodes before repopulating: {e}", "DEBUG")
            return
        for node in nodes:
            try:
                item = active_source.ui_nodes.add()
            except Exception:
                utils.log("UI sync could not add a ui_nodes item (write restricted).", "DEBUG")
                break
            _write_ui_item(item, node)

    with utils.logged_errors("sync._sync_ui_dummy_tree#5"):
        if active_source.active_ui_node_index >= len(active_source.ui_nodes):
            active_source.active_ui_node_index = max(0, len(active_source.ui_nodes) - 1)
    _ui_sync_last_key = sync_key
    _ui_sync_last_ts = now


def _refresh_active_source_metadata(scene, now: float | None = None):
    """Trigger an async metadata probe for the active file source if needed."""
    global _ui_source_probe_last
    if scene is None:
        return
    ace = getattr(scene, "audiocore", None)
    sources = getattr(ace, "audio_sources", None) if ace else None
    if not sources:
        return
    idx = int(getattr(ace, "active_audio_source_index", 0) or 0)
    if idx < 0 or idx >= len(sources):
        return
    src = sources[idx]
    if getattr(src, "source_type", "") != 'FILE':
        return
    filepath = str(getattr(src, "filepath", "") or "")
    if not filepath:
        return

    source_key = str(getattr(src, "stable_id", "") or filepath)
    ts_now = float(now if now is not None else time.monotonic())
    last_probe = float(_ui_source_probe_last.get(source_key, 0.0) or 0.0)
    if (ts_now - last_probe) < _UI_METADATA_PROBE_INTERVAL:
        return

    dur = float(getattr(src, "duration", 0.0) or 0.0)
    native_rate = int(getattr(src, "native_sample_rate", 0) or 0)
    if dur > 0.0 and native_rate > 0:
        _ui_source_probe_last.pop(source_key, None)
        return

    _ui_source_probe_last[source_key] = ts_now
    with utils.logged_errors("sync._refresh_active_source_metadata"):
        engine = ACEAudioEngine.ensure_instance(scene)
        engine.probes._probe_source_metadata(src)


def _ui_quick_state_key(context, scene, tree) -> str:
    """Build a cheap string key capturing the most volatile UI state."""
    ace = getattr(scene, "audiocore", None)
    src_uid = ""
    src_idx = -1
    if ace is not None:
        try:
            src_idx = utils.as_int(getattr(ace, "active_audio_source_index", None), -1)
        except Exception as exc:
            utils.note_swallowed("sync._ui_quick_state_key", exc)
            src_idx = -1
        sources = getattr(ace, "audio_sources", None) or []
        if 0 <= src_idx < len(sources):
            src_uid = str(getattr(sources[src_idx], "stable_id", "") or "")

    return f"{_tree_pointer(tree)}|{src_idx}|{src_uid}"


def _ui_has_pending_engine_changes(scene, tree) -> bool:
    """Return True when the engine has a pending recompile or dirty tree."""
    engine = ACEAudioEngine.get_instance(scene)
    if not engine:
        return False

    pending_tree_ptr = int(getattr(getattr(engine, "frontends", None), "_pending_tree_ptr", 0) or 0)
    dirty_tree_ptrs = getattr(getattr(engine, "frontends", None), "_dirty_tree_ptrs", set()) or set()

    if pending_tree_ptr:
        if tree is None:
            return True
        with utils.logged_errors("sync._ui_has_pending_engine_changes"):
            if int(tree.as_pointer()) == pending_tree_ptr:
                return True
    if not dirty_tree_ptrs:
        return False
    if tree is None:
        return True
    try:
        return int(tree.as_pointer()) in dirty_tree_ptrs
    except Exception as exc:
        utils.note_swallowed("sync._ui_has_pending_engine_changes#2", exc)
        return False


def _engine_latency_metrics(context) -> dict[str, float | bool]:
    """Collect and cache per-stage latency metrics for the active source."""
    global _ui_latency_cache
    scene = getattr(context, "scene", None)
    engine = ACEAudioEngine.get_instance(scene)
    scene_key = int(ACEAudioEngine._scene_key(scene))
    now = time.monotonic()
    _, active_source = utils._get_active_source(context)
    source_uid = str(getattr(active_source, "stable_id", "") or getattr(active_source, "name", "") or "")
    cached_metrics = _ui_latency_cache.get("metrics")
    cached_scene_key = int(_ui_latency_cache.get("scene_key", 0) or 0)
    cached_source_uid = str(_ui_latency_cache.get("source_uid", "") or "")
    cached_ts = float(_ui_latency_cache.get("ts", 0.0) or 0.0)
    if (
        isinstance(cached_metrics, dict)
        and cached_scene_key == scene_key
        and cached_source_uid == source_uid
        and (now - cached_ts) < _UI_LATENCY_UPDATE_INTERVAL_S
    ):
        return dict(cached_metrics)
    metrics: dict[str, float | bool] = {
        "source_ms": 0.0,
        "dsp_ms": 0.0,
        "preview_ms": 0.0,
        "preview_branch_ms": 0.0,
        "preview_output_buffer_ms": 0.0,
        "preview_device_ms": 0.0,
        "preview_prefetch_ms": 0.0,
        "preview_active": False,
        "proxy_write_ms": 0.0,
        "file_buffer_ms": 0.0,
        "geometry_ms": 0.0,
        "geometry_effective_ms": 0.0,
        "render_ms": 0.0,
        "live_target_ms": 0.0,
        "input_total_ms": 0.0,
        "engine_total_ms": 0.0,
        "preview_total_ms": 0.0,
        "blender_total_ms": 0.0,
        "total_latency_ms": 0.0,
        "has_data": False,
    }
    if active_source is not None and getattr(active_source, "source_type", "") == 'LIVE':
        metrics["live_target_ms"] = float(int(getattr(active_source, "live_latency_ms", 0) or 0))
    if engine is None:
        _ui_latency_cache = {
            "scene_key": scene_key,
            "source_uid": source_uid,
            "ts": now,
            "metrics": dict(metrics),
        }
        return metrics

    perf = engine.get_perf_current()
    diag = engine.get_runtime_diagnostics_current()
    metrics["source_ms"] = float(perf.get("source_read", 0.0) or 0.0)
    metrics["dsp_ms"] = float(perf.get("dsp", 0.0) or 0.0)
    metrics["preview_ms"] = float(perf.get("preview", 0.0) or 0.0)
    metrics["preview_branch_ms"] = float(diag.get("preview_branch_latency_ms", 0.0) or 0.0)
    metrics["preview_output_buffer_ms"] = float(diag.get("preview_output_buffer_ms", 0.0) or 0.0)
    metrics["preview_device_ms"] = float(diag.get("preview_callback_ms", 0.0) or 0.0)
    metrics["proxy_write_ms"] = float(perf.get("proxy_write", 0.0) or 0.0)
    metrics["file_buffer_ms"] = float(diag.get("source_prefetch_ms", 0.0) or 0.0)
    metrics["preview_active"] = bool(
        float(metrics["preview_ms"]) > 0.0
        or float(metrics["preview_branch_ms"]) > 0.0
        or float(metrics["preview_output_buffer_ms"]) > 0.0
        or float(metrics["preview_device_ms"]) > 0.0
    )
    metrics["preview_prefetch_ms"] = (
        float(metrics["file_buffer_ms"])
        if (bool(metrics["preview_active"]) and float(metrics["preview_output_buffer_ms"]) > 0.0)
        else 0.0
    )
    metrics["geometry_ms"] = float(diag.get("blender_geometry_eval_ms", 0.0) or 0.0)
    metrics["geometry_effective_ms"] = (
        float(metrics["geometry_ms"])
        if float(metrics["geometry_ms"]) > 0.0
        else float(metrics["proxy_write_ms"])
    )
    metrics["render_ms"] = float(diag.get("blender_render_ms", 0.0) or 0.0)
    measured_total_ms = (
        float(metrics["source_ms"])
        + float(metrics["dsp_ms"])
        + float(metrics["preview_ms"])
        + float(metrics["preview_branch_ms"])
        + float(metrics["preview_output_buffer_ms"])
        + float(metrics["preview_device_ms"])
        + float(metrics["file_buffer_ms"])
        + float(metrics["proxy_write_ms"])
        + float(metrics["geometry_ms"])
        + float(metrics["render_ms"])
    )
    metrics["has_data"] = bool(measured_total_ms > 0.0)
    if not bool(metrics["has_data"]):
        return metrics

    metrics["input_total_ms"] = (
        float(metrics["live_target_ms"])
        + float(metrics["source_ms"])
        + float(metrics["file_buffer_ms"])
        - float(metrics["preview_prefetch_ms"])
    )
    metrics["engine_total_ms"] = (
        float(metrics["dsp_ms"])
    )
    metrics["preview_total_ms"] = (
        float(metrics["preview_branch_ms"])
        + float(metrics["preview_ms"])
        + float(metrics["preview_output_buffer_ms"])
        + float(metrics["preview_prefetch_ms"])
    )
    metrics["blender_total_ms"] = (
        float(metrics["proxy_write_ms"])
        + float(metrics["geometry_ms"])
        + float(metrics["render_ms"])
    )
    if float(metrics["geometry_ms"]) <= 0.0:
        metrics["blender_total_ms"] = (
            float(metrics["geometry_effective_ms"])
            + float(metrics["render_ms"])
        )
    metrics["total_latency_ms"] = (
        float(metrics["input_total_ms"])
        + float(metrics["engine_total_ms"])
        + float(metrics["blender_total_ms"])
    )
    _ui_latency_cache = {
        "scene_key": scene_key,
        "source_uid": source_uid,
        "ts": now,
        "metrics": dict(metrics),
    }
    return metrics


def _ui_sync_timer():
    """Blender persistent timer callback: drive background UI sync."""
    global _ui_timer_last_quick_key, _ui_timer_last_sync_ts, _ui_timer_last_dirty_rebuild_ts
    with utils.logged_errors("sync._ui_sync_timer"):
        context = getattr(bpy, "context", None)
        scene = getattr(context, "scene", None) if context else None
        if not scene:
            return _UI_SYNC_TIMER_INTERVAL_IDLE
        if _ui_is_blocked(context):
            return _UI_SYNC_TIMER_INTERVAL_IDLE

        _refresh_ace_nodes_in_trees_flag(scene)
        tree = _find_active_node_tree(context, editor_context_only=True)
        quick_key = _ui_quick_state_key(context, scene, tree)
        now = time.monotonic()
        engine_dirty = _ui_has_pending_engine_changes(scene, tree)
        is_playing = ACEAudioEngine._is_animation_playing()
        target_interval = (
            _UI_SYNC_TIMER_INTERVAL_ACTIVE
            if (is_playing or engine_dirty)
            else _UI_SYNC_TIMER_INTERVAL_IDLE
        )
        key_changed = quick_key != _ui_timer_last_quick_key
        dirty_rebuild_due = bool(
            engine_dirty
            and ((now - _ui_timer_last_dirty_rebuild_ts) >= _UI_DIRTY_REBUILD_MIN_INTERVAL)
        )
        force_sync = bool(key_changed or dirty_rebuild_due)
        allow_dirty_rebuild = bool(force_sync)

        _refresh_active_source_metadata(scene, now=now)
        with utils.logged_errors("sync._ui_sync_timer#2"):
            engine = ACEAudioEngine.ensure_instance(scene)
            ace = getattr(scene, "audiocore", None)
            sources = list(getattr(ace, "audio_sources", []) or []) if ace else []
            engine.flush_source_tempo_detection_updates(sources)

        if key_changed or engine_dirty or ((now - _ui_timer_last_sync_ts) >= target_interval):
            t0 = time.perf_counter()
            _sync_ui_dummy_tree(
                context,
                force=force_sync,
                allow_dirty_rebuild=allow_dirty_rebuild,
            )
            engine = ACEAudioEngine.get_instance(scene)
            if engine is not None:
                with utils.logged_errors("sync._ui_sync_timer#3"):
                    engine.diagnostics._perf_add("ui_sync", time.perf_counter() - t0)
            _ui_timer_last_quick_key = quick_key
            _ui_timer_last_sync_ts = now
            if dirty_rebuild_due:
                _ui_timer_last_dirty_rebuild_ts = now
        return target_interval
    return _UI_SYNC_TIMER_INTERVAL_IDLE


@persistent
def _ui_load_post(_dummy):
    """Blender load_post handler: reset all UI state caches after a file load."""
    global _ui_sync_last_ts, _ui_sync_last_key, _ui_timer_last_quick_key, _ui_timer_last_sync_ts, _ui_timer_last_dirty_rebuild_ts, _ui_source_probe_last, _ui_latency_cache, _ui_tree_cache, _has_ace_nodes_in_trees
    _ui_sync_last_ts = 0.0
    _ui_sync_last_key = ""
    _ui_timer_last_quick_key = ""
    _ui_timer_last_sync_ts = 0.0
    _ui_timer_last_dirty_rebuild_ts = 0.0
    _has_ace_nodes_in_trees = False
    _ui_source_probe_last = {}
    _ui_latency_cache = {
        "scene_key": 0,
        "source_uid": "",
        "ts": 0.0,
        "metrics": None,
    }
    _ui_tree_cache = {
        "tree_ptr": 0,
        "topology_sig": "",
        "sig_ts": 0.0,
        "source_uid": "",
        "source_proxy": "",
        "nodes": (),
        "lookup_ids": {},
    }


def reset_state():
    global _ui_timer_last_quick_key, _ui_timer_last_sync_ts, _ui_timer_last_dirty_rebuild_ts, _ui_source_probe_last, _ui_latency_cache, _ui_tree_cache, _has_ace_nodes_in_trees
    _ui_timer_last_quick_key = ""
    _ui_timer_last_sync_ts = 0.0
    _ui_timer_last_dirty_rebuild_ts = 0.0
    _has_ace_nodes_in_trees = False
    _ui_source_probe_last = {}
    _ui_latency_cache = {
        "scene_key": 0,
        "source_uid": "",
        "ts": 0.0,
        "metrics": None,
    }
    _ui_tree_cache = {
        "tree_ptr": 0,
        "topology_sig": "",
        "sig_ts": 0.0,
        "source_uid": "",
        "source_proxy": "",
        "nodes": (),
    }
