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
from ...vars import ACE_NODES
from . import operator_tree
from . import usage
from ...nodes.split_freqs import split_freq_states
from ...nodes.dynamic_audio_operators import is_automatic_count_label
from .helpers import _build_ui_node_lookup_ids, _find_active_node_tree, _find_tree_node_by_ace_id, _tree_pointer, _ui_is_blocked


_UI_SYNC_MIN_INTERVAL = 0.25


_UI_SYNC_TIMER_INTERVAL_ACTIVE = 0.25


_UI_SYNC_TIMER_INTERVAL_IDLE = 0.5



_UI_METADATA_PROBE_INTERVAL = 5.0


_UI_LATENCY_UPDATE_INTERVAL_S = 1.0


_ui_sync_last_ts = 0.0


_ui_sync_last_key = ""


_ui_timer_last_quick_key = ""


_ui_timer_last_sync_ts = 0.0



_ui_source_probe_last: dict[str, float] = {}


_ui_latency_cache = {
    "scene_key": 0,
    "source_uid": "",
    "ts": 0.0,
    "metrics": None,
}


_ui_tree_cache: dict = {
    "tree_ptr": 0,
    "topology_sig": "",
    "sig_ts": 0.0,
    "source_uid": "",
    "source_proxy": "",
    "rows": [],
    "lookup_ids": {},
}


def _resolve_ui_list_selected_node(context):
    """Resolve the operator selected in the Active Operators list (by ace_id)."""
    from ...utils import _get_active_source

    _, active_source = _get_active_source(context)
    if not active_source:
        return None, None
    node_id = str(getattr(active_source, "active_ui_node_id", "") or "")
    if not node_id:
        return None, None
    tree = _find_active_node_tree(context, editor_context_only=True)
    node, parsed = _find_tree_node_by_ace_id(tree, node_id)
    if node and parsed:
        return node, parsed
    return None, parsed


def _ui_tree_topology_signature(tree) -> str:
    """Compute a short content hash representing the current GN tree topology."""
    if tree is None:
        return ""

    attr_socket_names = {"Attribute ID"}
    node_tokens = []
    for node in getattr(tree, "nodes", []):
        ptr = _tree_pointer(node)
        node_type = str(getattr(node, "type", "") or "")
        node_name = f"{getattr(node, 'name', '') or ''}|{getattr(node, 'label', '') or ''}"
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
                op_value = str(node.get('ace_mode', '') or '')
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


def _sync_ui_dummy_tree(context, force: bool = False):
    """Rebuild the active-source operator list when the active tree's topology changes.

    The list follows the tree itself, not the engine: it updates on node, link and
    setting edits even while auto-compile is off or a compile is still pending.
    """
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

    # The tree's own signature is the change detector: cheap (about 10 us per node)
    # and independent of auto-compile and of the engine's dirty tracking.
    topology_sig = _ui_tree_topology_signature(tree) if tree else ""
    rebuild_topology = bool(
        tree
        and (
            force
            or (not cache_scope_match)
            or topology_sig != str(_ui_tree_cache.get("topology_sig", "") or "")
        )
    )
    if not tree:
        rebuild_topology = True
    sync_key = (
        f"{source_uid}|{source_proxy}|"
        f"{getattr(tree, 'name', '') if tree else ''}|{tree_ptr}"
    )
    if (
        not force
        and not rebuild_topology
        and sync_key == _ui_sync_last_key
        and (now - _ui_sync_last_ts) < _UI_SYNC_MIN_INTERVAL
    ):
        return

    if not tree:
        utils.log("UI sync: no active ACE node tree found.", "DEBUG")
        _ui_tree_cache = _empty_tree_cache(now)
    elif rebuild_topology or not cache_scope_match:
        if str(getattr(tree, "bl_idname", "") or "") == "ShaderNodeTree":
            reader = ACEShaderTreeReader(tree)
        else:
            reader = ACEDummyTreeReader(tree)
        abstract = reader.build()
        node_lookup_ids = _build_ui_node_lookup_ids(tree, reader=reader)
        # Reading can assign missing ace_ids; sign the tree as it is now.
        topology_sig = _ui_tree_topology_signature(tree)
        roots_by_node = {
            str(node_lookup_ids.get(node.id, "") or node.id): set(node.root_sources) | (
                {node.source_binding_id} if node.source_binding_id else set()
            )
            for node in abstract.nodes.values()
        }
        _ui_tree_cache = {
            "roots_by_node": roots_by_node,
            "tree_ptr": int(tree_ptr),
            "topology_sig": topology_sig,
            "sig_ts": float(now),
            "source_uid": source_uid,
            "source_proxy": source_proxy,
            "rows": _operator_rows(tree, abstract, node_lookup_ids, source_uid),
            "lookup_ids": dict(node_lookup_ids),
        }

    rows = list(_ui_tree_cache.get("rows", []) or [])
    if not _ui_rows_match(active_source, rows):
        _write_ui_rows(active_source, rows)
    if active_source.active_ui_node_index >= len(active_source.ui_nodes):
        set_active_row(active_source, max(0, len(active_source.ui_nodes) - 1))
    _align_selection(active_source)
    _ui_sync_last_key = sync_key
    _ui_sync_last_ts = now


def _empty_tree_cache(now: float = 0.0) -> dict:
    return {
        "tree_ptr": 0,
        "topology_sig": "",
        "sig_ts": float(now),
        "source_uid": "",
        "source_proxy": "",
        "rows": [],
        "lookup_ids": {},
        "roots_by_node": {},
    }


def _operator_rows(tree, abstract, node_lookup_ids: dict[str, str], source_uid: str) -> list[operator_tree.Row]:
    """Folded rows of the operators under the active source's Source node(s) in *tree*."""
    bl_by_id = {}
    for bl_node in getattr(tree, "nodes", []) or []:
        ace_id = str(bl_node.get("ace_id", "") or "")
        if ace_id:
            bl_by_id[ace_id] = bl_node

    def _resolved(abstract_id: str) -> str:
        return str(node_lookup_ids.get(abstract_id, "") or abstract_id)

    graph_nodes = []
    bound_roots: list[tuple[int, str]] = []
    viewers = {node.id for node in abstract.nodes.values() if node.parameters.get("viewer")}
    for node in abstract.nodes.values():
        if node.id in viewers:
            continue
        node_id = _resolved(node.id)
        bl_node = bl_by_id.get(node_id)
        label = str((bl_node.label or bl_node.name) if bl_node is not None else (node.name or node_id))
        outputs = list(getattr(node, "outputs", []) or [])
        band_names = tuple(str(o.name) for o in outputs) if node.category == "delta" and len(outputs) > 1 else ()
        order = utils.as_int(node.parameters.get("frontend_node_index"), 0)
        prefix = ""
        if band_names:
            # "Split Low", "Split Side", "Split 2"...; a name the user gave replaces "Split".
            custom = str(getattr(bl_node, "label", "") or "")
            type_label = ACE_NODES["delta"]["nodes"].get(node.type, {}).get("label", "")
            automatic = custom == type_label or is_automatic_count_label(custom, ("Splitter", "Frequency Splitter"), "Bands")
            prefix = "Split" if automatic else custom
        graph_nodes.append(operator_tree.GraphNode(node_id, label, order, band_names, prefix))
        if node.category == "source_mngmt" and node.type == "src":
            if source_uid and str(getattr(node, "source_binding_id", "") or "") == source_uid:
                bound_roots.append((order, node_id))
    links = [
        operator_tree.GraphLink(_resolved(l.source_node_id), int(l.source_socket_idx), _resolved(l.target_node_id))
        for l in abstract.links
        if l.target_node_id not in viewers
    ]
    roots = [node_id for _order, node_id in sorted(bound_roots)]  # none: an empty list
    return operator_tree.build_rows(graph_nodes, links, roots)


def _row_name(row: operator_tree.Row) -> str:
    return " \u203a ".join(seg.label for seg in row.segments)


def _ui_rows_match(active_source, rows: list[operator_tree.Row]) -> bool:
    items = active_source.ui_nodes
    if len(items) != len(rows):
        return False
    for item, row in zip(items, rows):
        if (
            item.name != _row_name(row)
            or item.depth != row.depth
            or item.parent_row != row.parent
            or item.branch_key != row.branch_key
            or item.has_children != row.has_children
            or len(item.segments) != len(row.segments)
        ):
            return False
        for seg_item, seg in zip(item.segments, row.segments):
            if (seg_item.node_id, seg_item.kind) != (seg.node_id, seg.kind):
                return False
    return True


def _write_ui_rows(active_source, rows: list[operator_tree.Row]) -> None:
    items = active_source.ui_nodes
    items.clear()
    for row in rows:
        item = items.add()
        item.name = _row_name(row)
        item.node_id = row.segments[-1].node_id
        item.depth = row.depth
        item.has_children = row.has_children
        item.branch_key = row.branch_key
        item.parent_row = row.parent
        for seg in row.segments:
            seg_item = item.segments.add()
            seg_item.node_id = seg.node_id
            seg_item.label = seg.label
            seg_item.kind = seg.kind


# Collapsed branch keys by tree pointer. Not stored on the nodes: writing node
# properties tags the tree for the depsgraph.
_collapsed_by_tree: dict[int, set[str]] = {}

# Set while the add-on itself moves the list's active row, so the row callback
# does not replace the selected operator with the row's last one.
_setting_active_row = False

# (tree pointer, ace_id) of the Node Editor's active node at the last check.
_last_editor_active: tuple[int, str] = (0, "")


def current_rows() -> list[operator_tree.Row]:
    return list(_ui_tree_cache.get("rows", []) or [])


def collapsed_keys() -> set[str]:
    return _collapsed_by_tree.setdefault(int(_ui_tree_cache.get("tree_ptr", 0) or 0), set())


def toggle_branch(branch_key: str) -> None:
    keys = collapsed_keys()
    if branch_key in keys:
        keys.discard(branch_key)
    else:
        keys.add(branch_key)


def set_active_row(active_source, idx: int) -> None:
    global _setting_active_row
    if active_source.active_ui_node_index == idx:
        return
    _setting_active_row = True
    try:
        active_source.active_ui_node_index = idx
    finally:
        _setting_active_row = False


def is_setting_active_row() -> bool:
    return _setting_active_row


def reveal_operator(active_source, node_id: str) -> int:
    """Expand the branches above *node_id* and make its row the list's active row; returns the row."""
    rows = current_rows()
    idx = operator_tree.row_of_node(rows, node_id)
    if idx < 0:
        return idx
    collapsed_keys().difference_update(operator_tree.ancestor_keys(rows, idx))
    set_active_row(active_source, idx)
    return idx


def note_editor_active(tree, node_id: str) -> None:
    """Record a selection the add-on made in the Node Editor, so it is not read back as a new one."""
    global _last_editor_active
    _last_editor_active = (_tree_pointer(tree), str(node_id or ""))


def _sync_editor_active(tree, active_source) -> None:
    """Selecting an operator node in the Node Editor selects it in the list.

    A node of another source's graph makes that source the active one first.
    """
    global _last_editor_active
    nodes = getattr(tree, "nodes", None)
    active = getattr(nodes, "active", None) if nodes is not None else None
    node_id = str(active.get("ace_id", "") or "") if active is not None else ""
    key = (_tree_pointer(tree), node_id)
    if key == _last_editor_active:
        return
    # Only a change within a tree already on screen is a click: showing a tree switches nothing.
    clicked = key[0] == _last_editor_active[0]
    _last_editor_active = key
    if not node_id:
        return
    # The click may land before the periodic sync has read a tree or node that just appeared.
    _sync_ui_dummy_tree(bpy.context)
    switched = _activate_source_of(node_id) if clicked else None
    if switched is not None:
        active_source = switched
    elif node_id == active_source.active_ui_node_id:
        return
    if operator_tree.row_of_node(current_rows(), node_id) < 0:
        return
    active_source.active_ui_node_id = node_id
    reveal_operator(active_source, node_id)


def _activate_source_of(node_id: str):
    """Make the source feeding *node_id* active when the active one does not; returns it, else None."""
    context = bpy.context
    project = context.scene.audiocore
    roots = set(_ui_tree_cache.get("roots_by_node", {}).get(node_id, ()))
    current = int(project.active_audio_source_index)
    if 0 <= current < len(project.audio_sources) and project.audio_sources[current].stable_id in roots:
        return None
    for index, source in enumerate(project.audio_sources):
        if source.stable_id and source.stable_id in roots:
            project.active_audio_source_index = index
            _sync_ui_dummy_tree(context, force=True)
            return project.audio_sources[index]
    return None


def _align_selection(active_source) -> None:
    """Keep the list's active row and the selected operator on the same row."""
    rows = current_rows()
    if not len(active_source.ui_nodes):
        if active_source.active_ui_node_id:
            active_source.active_ui_node_id = ""  # an empty list selects nothing
        return
    if len(rows) != len(active_source.ui_nodes):
        return
    selected = str(active_source.active_ui_node_id or "")
    current = min(max(int(active_source.active_ui_node_index), 0), len(rows) - 1)
    index = operator_tree.row_of_node(rows, selected) if selected else -1
    if index >= 0:
        if not any(seg.kind == "node" and seg.node_id == selected for seg in rows[current].segments):
            set_active_row(active_source, index)
        return
    # The selected operator is not in this source's graph: select what the active row shows.
    set_active_row(active_source, current)
    nodes = [seg for seg in rows[current].segments if seg.kind in operator_tree.NODE_KINDS]
    active_source.active_ui_node_id = nodes[-1].node_id if nodes else ""


# The audio source list folds each source's instances; closed until opened.
_expanded_sources: set[str] = set()
_setting_source_row = False


def expanded_source_keys() -> set[str]:
    return _expanded_sources


def toggle_source_branch(branch_key: str) -> None:
    if branch_key in _expanded_sources:
        _expanded_sources.discard(branch_key)
    else:
        _expanded_sources.add(branch_key)


def _source_row_key(source, index: int) -> str:
    return str(getattr(source, "stable_id", "") or "") or f"#{index}"


def source_for_row(project, item):
    """(index, source) a source row stands for, or (-1, None)."""
    for index, source in enumerate(project.audio_sources):
        if _source_row_key(source, index) == item.target:
            return index, source
    return -1, None


def is_setting_source_row() -> bool:
    return _setting_source_row


def follow_active_source(project) -> None:
    """Highlight the active source's row, unless one of its instances is highlighted."""
    global _setting_source_row
    index = int(project.active_audio_source_index)
    if not 0 <= index < len(project.audio_sources):
        return
    key = _source_row_key(project.audio_sources[index], index)
    rows = project.source_rows
    current = int(project.active_source_row)
    if 0 <= current < len(rows):
        item = rows[current]
        owner = rows[item.parent_row] if item.parent_row >= 0 else item
        if owner.target == key:
            return
    for row_index, item in enumerate(rows):
        if item.kind == "source" and item.target == key:
            _setting_source_row = True
            try:
                project.active_source_row = row_index
            finally:
                _setting_source_row = False
            return


def _source_rows(scene) -> list[tuple]:
    """(kind, target, name, detail, icon, detail_icon, depth, has_children, branch_key, parent) rows."""
    from .helpers import _compiled_source_instance_rows, _frontend_instance_type, _frontend_kind_icon, _frontend_object_name

    instances: dict[str, list] = {}
    for entry in _compiled_source_instance_rows(scene) or []:
        source_prop = entry.get("source_prop")
        token = str(getattr(source_prop, "stable_id", "") or entry.get("source_token", "") or "")
        instances[token] = list(entry.get("frontend_refs", []) or [])

    rows: list[tuple] = []
    for index, source in enumerate(scene.audiocore.audio_sources):
        key = _source_row_key(source, index)
        refs = instances.get(key, [])
        parent = len(rows)
        rows.append(("source", key, str(source.name), "", "NONE", "NONE", 0, bool(refs), f"src:{key}", -1))
        for ref in refs:
            kind = str(getattr(ref, "kind", "") or "")
            rows.append((
                "instance", _frontend_object_name(ref), _frontend_object_name(ref), _frontend_instance_type(kind),
                "OBJECT_DATA", _frontend_kind_icon(kind), 1, False, "", parent,
            ))
    return rows


def _row_tuple(item) -> tuple:
    return (
        item.kind, item.target, item.name, item.detail, item.icon, item.detail_icon,
        item.depth, item.has_children, item.branch_key, item.parent_row,
    )


def _sync_source_rows(scene) -> None:
    ace = getattr(scene, "audiocore", None)
    if ace is None:
        return
    rows = _source_rows(scene)
    if [_row_tuple(item) for item in ace.source_rows] != rows:
        ace.source_rows.clear()
        for row in rows:
            item = ace.source_rows.add()
            (item.kind, item.target, item.name, item.detail, item.icon, item.detail_icon,
             item.depth, item.has_children, item.branch_key, item.parent_row) = row
    follow_active_source(ace)


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
    # Latency describes playback: with the timeline stopped (and no render) there is none.
    if engine is None or not (engine._is_animation_playing() or engine._is_rendering_job()):
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


_last_driven_splits: tuple = ()


def _redraw_on_driven_split_change(context) -> None:
    global _last_driven_splits
    node, parsed = _resolve_ui_list_selected_node(context)
    values: tuple = ()
    if node is not None and parsed and parsed.get("type") == "frequency_splitter":
        values = tuple((value, state) for _sock, value, state in split_freq_states(node, context.scene)
                       if state != "own")
    if values == _last_driven_splits:
        return
    _last_driven_splits = values
    for window in getattr(context.window_manager, "windows", []) or []:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


# How often the Node Editor's active node is checked: short, so a click shows up at once.
_UI_CLICK_INTERVAL = 0.1


_LATENCY_KEYS = ("total_latency_ms", "input_total_ms", "engine_total_ms", "blender_total_ms")


def _redraw_sidebars(context) -> None:
    for window in getattr(context.window_manager, "windows", []) or []:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            for region in area.regions:
                if region.type == "UI":
                    region.tag_redraw()


@persistent
def _ui_click_timer():
    """Persistent timer: follow node clicks quickly (the full sync runs less often)."""
    with utils.logged_errors("sync._ui_click_timer"):
        context = bpy.context
        scene = getattr(context, "scene", None)
        if scene is None or _ui_is_blocked(context):
            return _UI_SYNC_TIMER_INTERVAL_IDLE
        tree = _find_active_node_tree(context, editor_context_only=True)
        _, active_source = utils._get_active_source(context)
        if tree is not None and active_source is not None:
            before = _last_editor_active
            _sync_editor_active(tree, active_source)
            if _last_editor_active != before:
                _redraw_sidebars(context)
    return _UI_CLICK_INTERVAL


def _ui_sync_timer():
    """Blender persistent timer callback: drive background UI sync."""
    global _ui_timer_last_quick_key, _ui_timer_last_sync_ts
    with utils.logged_errors("sync._ui_sync_timer"):
        context = getattr(bpy, "context", None)
        scene = getattr(context, "scene", None) if context else None
        if not scene:
            return _UI_SYNC_TIMER_INTERVAL_IDLE
        if _ui_is_blocked(context):
            return _UI_SYNC_TIMER_INTERVAL_IDLE

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

        _refresh_active_source_metadata(scene, now=now)
        with utils.logged_errors("sync._ui_sync_timer#2"):
            engine = ACEAudioEngine.ensure_instance(scene)
            ace = getattr(scene, "audiocore", None)
            sources = list(getattr(ace, "audio_sources", []) or []) if ace else []
            engine.flush_source_tempo_detection_updates(sources)

        # Every tick: the sync itself only rebuilds when the tree signature changed.
        t0 = time.perf_counter()
        _sync_ui_dummy_tree(context, force=key_changed)
        _, active_source = utils._get_active_source(context)
        if tree is not None and active_source is not None:
            with utils.logged_errors("sync._ui_sync_timer#ramp"):
                from .splitter import remove_legacy_ramp_group
                remove_legacy_ramp_group()
        with utils.logged_errors("sync._ui_sync_timer#usage"):
            latencies = _engine_latency_metrics(context)
            if usage.sample(extra={key: float(latencies[key]) for key in _LATENCY_KEYS}):
                _redraw_sidebars(context)
        with utils.logged_errors("sync._ui_sync_timer#driven"):
            _redraw_on_driven_split_change(context)
        with utils.logged_errors("sync._ui_sync_timer#sources"):
            _sync_source_rows(scene)
        engine = ACEAudioEngine.get_instance(scene)
        if engine is not None:
            with utils.logged_errors("sync._ui_sync_timer#3"):
                engine.diagnostics._perf_add("ui_sync", time.perf_counter() - t0)
        _ui_timer_last_quick_key = quick_key
        _ui_timer_last_sync_ts = now
        return target_interval
    return _UI_SYNC_TIMER_INTERVAL_IDLE


@persistent
def _ui_load_post(_dummy):
    """Blender load_post handler: reset all UI state caches after a file load."""
    global _ui_sync_last_ts, _ui_sync_last_key, _ui_timer_last_quick_key, _ui_timer_last_sync_ts, _ui_source_probe_last, _ui_latency_cache, _ui_tree_cache, _last_editor_active
    _ui_sync_last_ts = 0.0
    _ui_sync_last_key = ""
    _ui_timer_last_quick_key = ""
    _ui_timer_last_sync_ts = 0.0
    _ui_source_probe_last = {}
    _ui_latency_cache = {
        "scene_key": 0,
        "source_uid": "",
        "ts": 0.0,
        "metrics": None,
    }
    _ui_tree_cache = _empty_tree_cache()
    _collapsed_by_tree.clear()
    _expanded_sources.clear()
    _last_editor_active = (0, "")


def reset_state():
    global _ui_timer_last_quick_key, _ui_timer_last_sync_ts, _ui_source_probe_last, _ui_latency_cache, _ui_tree_cache, _last_editor_active
    _ui_timer_last_quick_key = ""
    _ui_timer_last_sync_ts = 0.0
    _ui_source_probe_last = {}
    _ui_latency_cache = {
        "scene_key": 0,
        "source_uid": "",
        "ts": 0.0,
        "metrics": None,
    }
    _ui_tree_cache = _empty_tree_cache()
    _collapsed_by_tree.clear()
    _last_editor_active = (0, "")
