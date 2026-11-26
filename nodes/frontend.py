"""Frontend kinds and frontend identity shared by geometry and shader trees."""

import hashlib
from ..core.id_management import ACEIdManager
from .. import utils


FRONTEND_KIND_GEOMETRY = "GEOMETRY_NODES"


FRONTEND_KIND_SHADER = "SHADER_NODES"


def _tree_pointer(tree) -> int:
    try:
        return int(tree.as_pointer())
    except Exception as exc:
        utils.note_swallowed("frontend._tree_pointer", exc)
        return id(tree)


def ensure_frontend_uid(tree, frontend_kind: str, existing_uids) -> str:
    """Guarantee a persistent, unique frontend UID on a top-level node tree."""
    if tree is None:
        return ""
    existing = {str(v) for v in (existing_uids or []) if str(v)}
    current_uid = ""
    try:
        current_uid = str(tree.get("ace_frontend_uid", "") or "")
    except Exception as exc:
        utils.note_swallowed("frontend.ensure_frontend_uid", exc)
        current_uid = ""
    if current_uid and current_uid not in existing:
        return current_uid

    kind_token = ACEIdManager._clean_string(str(frontend_kind or "frontend")).lower()[:10] or "frontend"
    tree_name = str(getattr(tree, "name", "") or "Tree")
    tree_type = str(getattr(tree, "bl_idname", "") or "")
    max_attempts = max(len(existing) + 64, 256)
    for salt in range(max_attempts):
        suffix = hashlib.blake2b(
            f"{frontend_kind}:{tree_type}:{tree_name}:{salt}".encode("utf-8"),
            digest_size=4,
        ).hexdigest()[:6]
        candidate = f"ACE_FRONTEND_{kind_token}_{suffix}"
        if candidate in existing:
            continue
        with utils.logged_errors("frontend.ensure_frontend_uid#2"):
            tree["ace_frontend_uid"] = candidate
        return candidate
    return current_uid


# Group rebuilds (an ACE node's sockets changed) reset the input values of every node using
# the group. Values are kept by socket name and put back once Blender has refreshed the
# nodes' sockets, which happens on its next update, not during the rebuild.
_pending_input_restores: list[tuple[str, str, str, str, dict]] = []
_RESTORE_ATTEMPTS = 20


def _group_users(ng):
    import bpy  # type: ignore

    for tree in bpy.data.node_groups:
        for node in tree.nodes:
            if getattr(node, "node_tree", None) == ng:
                yield "group", tree.name, node
    for material in bpy.data.materials:
        tree = material.node_tree
        if tree is None:
            continue
        for node in tree.nodes:
            if getattr(node, "node_tree", None) == ng:
                yield "material", material.name, node


def _plain_value(value):
    if value is None or isinstance(value, (str, int, float, bool)) or hasattr(value, "bl_rna"):
        return value
    return tuple(value)


def remember_group_inputs(ng) -> None:
    """Call before rebuilding *ng*: its users' unlinked input values come back afterwards."""
    for owner_kind, owner_name, node in _group_users(ng):
        values = {
            sock.name: _plain_value(sock.default_value)
            for sock in node.inputs
            if not sock.is_linked and hasattr(sock, "default_value")
        }
        if values:
            _pending_input_restores.append((owner_kind, owner_name, node.name, ng.name, values))
    _schedule_input_restore()


def _schedule_input_restore() -> None:
    import bpy  # type: ignore

    if _pending_input_restores and not bpy.app.timers.is_registered(_restore_group_inputs):
        _restore_group_inputs.attempts = 0
        bpy.app.timers.register(_restore_group_inputs, first_interval=0.05)


def _restore_group_inputs():
    """Timer: restore remembered values on nodes whose sockets match their rebuilt group."""
    import bpy  # type: ignore

    with utils.logged_errors("frontend._restore_group_inputs"):
        _restore_group_inputs.attempts += 1
        waiting = []
        for entry in _pending_input_restores:
            owner_kind, owner_name, node_name, group_name, values = entry
            owner = (bpy.data.node_groups if owner_kind == "group" else bpy.data.materials).get(owner_name)
            tree = owner if owner_kind == "group" else getattr(owner, "node_tree", None)
            node = tree.nodes.get(node_name) if tree is not None else None
            group = bpy.data.node_groups.get(group_name)
            if node is None or group is None or node.node_tree != group:
                continue
            expected = [i.name for i in group.interface.items_tree if getattr(i, "in_out", "") == "INPUT"]
            if [sock.name for sock in node.inputs] != expected:
                waiting.append(entry)
                continue
            for sock in node.inputs:
                if sock.name in values and not sock.is_linked and hasattr(sock, "default_value"):
                    with utils.logged_errors("frontend._restore_group_inputs#value"):
                        sock.default_value = values[sock.name]
        _pending_input_restores[:] = waiting
        if waiting and _restore_group_inputs.attempts < _RESTORE_ATTEMPTS:
            return 0.1
    _pending_input_restores.clear()
    return None


_restore_group_inputs.attempts = 0


def convert_rolling_average_groups() -> None:
    """Convert the open file's Rolling Average groups (also needed when no file load follows)."""
    import bpy  # type: ignore

    for ng in list(bpy.data.node_groups):
        if "util_rolling_avg" in ng.name and ng.library is None:
            with utils.logged_errors("frontend.convert_rolling_average_groups"):
                rolling_frames_to_window(ng)


def rolling_frames_to_window(ng) -> None:
    """Rolling Average counted frames; it now takes a time. Done in place, so links survive."""
    import bpy  # type: ignore

    def frames_item():
        return next((i for i in ng.interface.items_tree
                     if getattr(i, "in_out", "") == "INPUT" and i.name == "Number of Frames"), None)

    item = frames_item()
    if item is None:
        return
    scene = getattr(bpy.context, "scene", None)
    fps = scene.render.fps / scene.render.fps_base if scene is not None else 24.0
    kept = [(node, node.inputs["Number of Frames"].default_value)
            for _kind, _owner, node in _group_users(ng) if "Number of Frames" in node.inputs]
    item.name = "Window"
    item.socket_type = "NodeSocketFloat"
    window = next(i for i in ng.interface.items_tree if getattr(i, "in_out", "") == "INPUT" and i.name == "Window")
    window.subtype = "TIME_ABSOLUTE"
    window.min_value, window.max_value, window.default_value = 0.001, 60.0, 0.25
    for node, frames in kept:
        socket = node.inputs.get("Window")
        if socket is not None:
            socket.default_value = max(float(frames), 1.0) / fps
