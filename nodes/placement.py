"""Placing an ACE node: what a node gets when it lands in a tree, whichever way it came.

The add menu places the nodes it adds. Blender's link-drag search and Add > Group add a
kind's template group (nodes/groups.py) as they would any group, so ACE places those nodes
when their tree next changes:

- a Frequency Splitter gets the variant of its band count (the default count when new);
- a shader Source reads the active source through that source's own group; a Geometry Nodes
  Source with no proxy reads the active source;
- the node gets its label (its mode's, for a node with modes), its mode, and its preview flag;
- the inputs its mode doesn't use are hidden (``sync_mode_sockets``).

Sink names are left to the compile's repair (core/engine/sinks.py): a node without names of its
own, a template's placeholders included, gets new ones there. A placed node carries
``ace_placed``, so a copy keeps its settings and is never placed again.
"""

from .. import utils
from ..vars import (
    ACE_NODES, PREVIEW_KIND, SOURCE_KIND, SPLITTER_KIND, mode_inputs, mode_menu, node_def, node_modes,
)
from .identity import is_ace_group, node_kind

PLACED_KEY = "ace_placed"


def _scene_sources(scene):
    ace = getattr(scene, "audiocore", None)
    return ace, list(getattr(ace, "audio_sources", []) or []) if ace is not None else []


def active_source(scene):
    ace, sources = _scene_sources(scene)
    idx = int(getattr(ace, "active_audio_source_index", -1) or 0) if ace is not None else -1
    return sources[idx] if 0 <= idx < len(sources) else None


def menu_mode(node, category: str, node_type: str):
    """The item chosen in the node's mode menu, or None when the menu is linked (any mode may come)."""
    name = mode_menu(category, node_type)
    sock = node.inputs.get(name) if name and hasattr(node, "inputs") else None
    if sock is None or sock.is_linked:
        return None
    return str(getattr(sock, "default_value", "") or "")


def sync_mode_sockets(node, category: str = "", node_type: str = "") -> bool:
    """Hide the inputs the node's menu mode doesn't use (a linked input stays); True when one changed."""
    if not category:
        category, node_type = node_kind(node)
    if not category or not mode_menu(category, node_type):
        return False
    mode = menu_mode(node, category, node_type)
    changed = False
    for sock_name, only_mode in mode_inputs(category, node_type).items():
        sock = node.inputs.get(sock_name)
        if sock is None:
            continue
        hide = mode is not None and mode != only_mode and not sock.is_linked
        if bool(sock.hide) != hide:
            sock.hide = hide
            changed = True
    return changed


def _splitter_bands(node, default: int) -> int:
    count = int(node.get("ace_band_count", 0) or 0)
    group = getattr(node, "node_tree", None)
    if not count and group is not None:
        count = int(group.get("ace_bands", 0) or 0)
    return max(2, min(count or default, 8))


def place_node(node, category: str, node_type: str, scene, source=None) -> None:
    """Give *node* what a node of its kind needs; what it has already (a copy's settings) stays."""
    from .groups import is_template
    from .modes import apply_node_mode, automatic_node_label

    label = str(node_def(category, node_type).get("label", "") or node_type)
    if (category, node_type) == SPLITTER_KIND:
        from .dynamic_audio_operators import ACESplitterManager

        settings = getattr(getattr(scene, "audiocore", None), "engine_settings", None)
        bands = _splitter_bands(node, int(getattr(settings, "splitter_default_bands", 2) or 2))
        if is_template(node.node_tree) or int(node.node_tree.get("ace_bands", 0) or 0) != bands:
            ACESplitterManager.apply_variant_to_node(node, bands)
        node["ace_band_count"] = bands
    if (category, node_type) == PREVIEW_KIND and "ace_preview_enabled" not in node:
        node["ace_preview_enabled"] = True
    if node_modes(category, node_type):
        apply_node_mode(node, category, node_type, node.get("ace_mode"))
    if not node.label:
        node.label = automatic_node_label(node, category, node_type) or label
    if (category, node_type) == SOURCE_KIND:
        from .source_nodes import node_source_id, set_node_source

        source = source if source is not None else active_source(scene)
        proxy_socket = node.inputs.get("Proxy Object")
        fed = proxy_socket is not None and proxy_socket.is_linked  # a group's input chooses its source
        if source is not None and not fed and not node_source_id(node):
            set_node_source(node, source, scene)
    sync_mode_sockets(node, category, node_type)
    node[PLACED_KEY] = True


def place_new_nodes(tree, scene) -> int:
    """Place the ACE nodes of *tree* that came in some other way than the add menu; how many."""
    if tree is None or getattr(tree, "library", None) is not None or is_ace_group(tree):
        return 0
    placed = 0
    for node in list(getattr(tree, "nodes", []) or []):
        category, node_type = node_kind(node)
        if not category or node.get(PLACED_KEY):
            continue
        with utils.logged_errors("placement.place_new_nodes"):
            new = str(node.name).startswith(("Group", "NodeGroup"))
            place_node(node, category, node_type, scene)
            if new:
                node.name = node.label or ACE_NODES[category]["nodes"][node_type]["label"]
            placed += 1
    return placed


def sync_tree_mode_sockets(tree) -> int:
    """Follow every ACE node's menu mode in *tree*: how many nodes changed."""
    changed = 0
    if tree is None or getattr(tree, "library", None) is not None or is_ace_group(tree):
        return 0
    for node in list(getattr(tree, "nodes", []) or []):
        category, node_type = node_kind(node)
        if category and mode_menu(category, node_type) and sync_mode_sockets(node, category, node_type):
            changed += 1
    return changed
