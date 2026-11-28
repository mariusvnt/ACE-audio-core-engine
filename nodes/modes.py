"""Mode nodes (Loudness, Stereo): labels and applying a mode."""

from ..vars import ACE_NODES, SPLITTER_KIND, mode_inputs, node_mode_label, node_modes, resolve_node_mode


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
    if (category, node_type) == SPLITTER_KIND:
        return f"{base} ({int(node.get('ace_band_count', max(2, len(node.outputs))))} Bands)"
    return base


def _is_automatic_label(label: str, category: str, node_type: str) -> bool:
    label = str(label or "")
    if not label:
        return True
    automatic = {mode_node_label(category, node_type, m[0]) for m in node_modes(category, node_type)}
    automatic.add(str(ACE_NODES.get(category, {}).get("nodes", {}).get(node_type, {}).get("label", "")))
    return label in automatic


def apply_node_mode(node, category: str, node_type: str, mode) -> str:
    """Store *mode* on *node* (the default when invalid), refresh its automatic label and mode-only inputs."""
    resolved = resolve_node_mode(category, node_type, mode)
    if not resolved:
        return ""
    node["ace_mode"] = resolved
    if _is_automatic_label(getattr(node, "label", ""), category, node_type):
        node.label = mode_node_label(category, node_type, resolved)
    for sock_name, only_mode in mode_inputs(category, node_type).items():
        sock = node.inputs.get(sock_name) if hasattr(node, "inputs") else None
        if sock is not None and not sock.is_linked:
            sock.hide = resolved != only_mode
    return resolved
