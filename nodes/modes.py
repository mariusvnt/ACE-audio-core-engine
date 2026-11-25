"""Mode nodes (Loudness, Stereo, Spectral): applying a mode, and migrating the old per-method nodes."""

import bpy  # type: ignore
from ..core.id_management import ACEIdManager
from ..vars import ACE_NODES, LEGACY_STREAM_TYPES, node_mode_label, node_modes, resolve_node_mode
from .. import utils


# Inputs used by one mode only; hidden on the node in the other modes.
_MODE_ONLY_INPUTS: dict[tuple[str, str], dict[str, str]] = {
    ("stream", "loudness"): {"Window (s)": "lufs"},
}

# Labels the old per-method nodes were created with, to tell automatic labels from the user's.
_LEGACY_LABELS = {
    "rms": "RMS",
    "lufs": "Loudness (LUFS)",
    "peak": "Peak",
    "stereo_width": "Stereo Width",
    "stereo_balance": "Stereo Balance",
    "spectral_centroid": "Spectral Centroid",
    "spectral_flux": "Spectral Flux",
    "zero_crossing_rate": "Zero Crossing Rate",
}


def mode_node_label(category: str, node_type: str, mode: str) -> str:
    """Automatic label of a mode node, e.g. "Loudness (LUFS)"."""
    base = str(ACE_NODES.get(category, {}).get("nodes", {}).get(node_type, {}).get("label", "") or node_type)
    mode_label = node_mode_label(category, node_type, mode)
    return f"{base} ({mode_label})" if mode_label else base


def automatic_node_label(node, category: str, node_type: str) -> str:
    """The label the add-on gives *node* when the user has not named it."""
    if node_modes(category, node_type):
        return mode_node_label(category, node_type, resolve_node_mode(category, node_type, node.get("ace_mode")))
    base = str(ACE_NODES.get(category, {}).get("nodes", {}).get(node_type, {}).get("label", "") or node_type)
    if (category, node_type) == ("delta", "frequency_splitter"):
        return f"{base} ({int(node.get('ace_band_count', max(2, len(node.outputs))))} Bands)"
    if (category, node_type) == ("utilities", "audio_math"):
        return f"{base} ({int(node.get('ace_input_count', max(2, len(node.inputs))))} Inputs)"
    return base


def _is_automatic_label(label: str, category: str, node_type: str) -> bool:
    label = str(label or "")
    if not label:
        return True
    automatic = {mode_node_label(category, node_type, m[0]) for m in node_modes(category, node_type)}
    automatic.add(str(ACE_NODES.get(category, {}).get("nodes", {}).get(node_type, {}).get("label", "")))
    automatic.update(_LEGACY_LABELS.values())
    return label in automatic


def apply_node_mode(node, category: str, node_type: str, mode) -> str:
    """Store *mode* on *node* (the default when invalid), refresh its automatic label and mode-only inputs."""
    resolved = resolve_node_mode(category, node_type, mode)
    if not resolved:
        return ""
    node["ace_mode"] = resolved
    if _is_automatic_label(getattr(node, "label", ""), category, node_type):
        node.label = mode_node_label(category, node_type, resolved)
    for sock_name, only_mode in _MODE_ONLY_INPUTS.get((category, node_type), {}).items():
        sock = node.inputs.get(sock_name) if hasattr(node, "inputs") else None
        if sock is not None and not sock.is_linked:
            sock.hide = resolved != only_mode
    return resolved


def _parse_legacy_stream_id(ace_id: str):
    """("ACE_stream_lufs_ab12cd") -> ("lufs", "ab12cd"), or None for any other id."""
    prefix = f"{ACEIdManager.PREFIX}stream_"
    if not ace_id.startswith(prefix):
        return None
    body, _, uid = ace_id[len(prefix):].rpartition("_")
    if body in LEGACY_STREAM_TYPES and uid:
        return body, uid
    return None


def _factory_for_tree(tree):
    from .geometry.factory import ACEDummyNodeFactory
    from .shader.factory import ACEShaderNodeFactory

    if str(getattr(tree, "bl_idname", "") or "") == "ShaderNodeTree":
        return ACEShaderNodeFactory
    return ACEDummyNodeFactory


def _migrate_node(tree, node, legacy_type: str, uid: str) -> bool:
    new_type, mode = LEGACY_STREAM_TYPES[legacy_type]
    factory = _factory_for_tree(tree)
    group_name = factory.get_group_name("stream", new_type)
    group = bpy.data.node_groups.get(group_name)
    if group is None or str(getattr(group, "bl_idname", "")) != str(getattr(tree, "bl_idname", "")):
        factory.rebuild_node_group("stream", new_type)
        group = bpy.data.node_groups.get(group_name)
    if group is None:
        utils.log(f"ACE: cannot migrate '{node.name}': group '{group_name}' is missing.", "WARNING")
        return False

    # Sockets are matched by name: the new group's identifiers differ from the old one's.
    incoming = [(link.from_socket, sock.name) for sock in node.inputs for link in sock.links]
    outgoing = [link.to_socket for sock in node.outputs for link in sock.links]
    values = {}
    for sock in node.inputs:
        if not sock.is_linked and hasattr(sock, "default_value"):
            value = sock.default_value
            values[sock.name] = value if isinstance(value, (str, int, float)) else tuple(value)

    node.node_tree = group

    for name, value in values.items():
        sock = node.inputs.get(name)
        if sock is not None and hasattr(sock, "default_value"):
            with utils.logged_errors("modes._migrate_node"):
                sock.default_value = value
    for from_socket, name in incoming:
        sock = node.inputs.get(name)
        if sock is not None and not any(link.from_socket == from_socket for link in sock.links):
            tree.links.new(from_socket, sock)
    if node.outputs:
        value_out = node.outputs[0]
        for to_socket in outgoing:
            if not any(link.to_socket == to_socket for link in value_out.links):
                tree.links.new(value_out, to_socket)

    old_label = str(node.label or "")
    old_name = str(node.name or "")
    node["ace_id"] = f"{ACEIdManager.PREFIX}stream_{new_type}_{uid}"
    apply_node_mode(node, "stream", new_type, mode)
    legacy_label = _LEGACY_LABELS.get(legacy_type, "")
    name_was_automatic = bool(legacy_label) and (old_name == legacy_label or old_name.startswith(f"{legacy_label}."))
    if name_was_automatic or old_name == old_label:
        node.name = node.label
    return True


def migrate_legacy_stream_nodes() -> int:
    """Turn nodes of the old per-method stream types into mode nodes. Returns the number migrated."""
    # Linked data is read-only; embedded material trees belong to their material's library.
    trees = [ng for ng in bpy.data.node_groups if ng.library is None and not ng.name.startswith(".ACE_")]
    trees += [m.node_tree for m in bpy.data.materials if m.library is None and m.node_tree is not None]
    migrated = 0
    for tree in trees:
        for node in list(getattr(tree, "nodes", [])):
            parsed = _parse_legacy_stream_id(str(node.get("ace_id", "") or ""))
            if parsed is None:
                continue
            with utils.logged_errors("modes.migrate_legacy_stream_nodes"):
                if _migrate_node(tree, node, *parsed):
                    migrated += 1
    if migrated:
        utils.log(f"ACE: migrated {migrated} stream node(s) to Loudness / Stereo / Spectral.", "INFO")
    return migrated
