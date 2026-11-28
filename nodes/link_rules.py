"""Links into ACE nodes that ACE ignores: why, and showing them as invalid links.

A link is ignored when ACE can't use what it carries: a linked Attribute ID, a Proxy Object
that doesn't come from a group input, audio reached through a node other than a reroute, a
control (a Window) that isn't a constant, a split frequency computed by a node ACE doesn't
know. The definition of a tree lists them (``nodes/definitions.py``), the reading leaves them
out, and ``mark_ignored_links`` sets ``NodeLink.is_valid`` to False on each, so Blender draws
it red. The N-panel says why.

The flag only shows: Blender still evaluates through the link. Blender also checks links again
when a tree's nodes or links change and when a file is read, which sets the flag back, so the
links are marked again after every compile and every tree update.
"""

from dataclasses import dataclass

from .. import utils
from ..vars import ROLE_SPLITTER, node_role
from .identity import is_ace_group, node_kind, tree_key


@dataclass(frozen=True)
class IgnoredLink:
    node_name: str  # the ACE node the link goes into
    input_index: int
    input_identifier: str
    input_name: str
    from_node: str  # the label of the node it comes from (after reroutes)
    reason: str


def _feeding_link(sock):
    return next((lk for lk in getattr(sock, "links", []) or [] if not getattr(lk, "is_muted", False)), None)


def _origin(sock):
    """(node, link into *sock*) at the start of the chain of reroutes feeding *sock*."""
    link = _feeding_link(sock)
    first = link
    for _ in range(64):
        if link is None:
            return None, first
        node = link.from_node
        if str(getattr(node, "bl_idname", "") or "") != "NodeReroute":
            return node, first
        link = _feeding_link(node.inputs[0])
    return None, first


def _label(node) -> str:
    return str(getattr(node, "label", "") or getattr(node, "name", "") or "a node")


def _passes_audio(node) -> bool:
    """Audio may come straight from an ACE node, the group's input or a group."""
    if node is None:
        return False
    if node_kind(node)[0]:
        return True
    node_type = str(getattr(node, "type", "") or "")
    if node_type == "GROUP_INPUT":
        return True
    tree = getattr(node, "node_tree", None)
    return node_type == "GROUP" and tree is not None and not is_ace_group(tree)


def _number(value) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def judge_input(node, category: str, node_type: str, index: int, sock, data_type: str):
    """What ACE makes of linked input *index* of an ACE node.

    Returns ("ok", None), ("constant", value) when a constant chain feeds it, or
    ("ignored", IgnoredLink).
    """
    from . import driven_values  # it needs bpy, which reading a definition doesn't

    origin, _link = _origin(sock)
    name = str(getattr(sock, "name", "") or "")
    role = node_role(category, node_type)

    def ignored(reason: str):
        return "ignored", IgnoredLink(str(node.name), int(index), str(sock.identifier), name, _label(origin), reason)

    if name.startswith("Attribute ID"):
        return ignored(f"{name} is a name to type in the field, not a link: ACE writes to the name typed there.")
    if name == "Proxy Object":
        if origin is not None and str(getattr(origin, "type", "") or "") == "GROUP_INPUT":
            return "ok", None
        return ignored(f"{name} comes from {_label(origin)}: ACE reads the source from the node itself or from the group's input.")
    if "Bundle" in str(data_type or ""):
        if _passes_audio(origin):
            return "ok", None
        return ignored(f"Audio comes through {_label(origin)}: ACE follows audio only from ACE nodes, groups and the group's input.")
    if role == ROLE_SPLITTER:
        unsupported = driven_values.unsupported_node(sock)
        if unsupported is None:
            return "ok", None
        return ignored(f"{name} comes through {unsupported}, which ACE can't compute: its own value, {_number(sock.default_value)}, is used.")
    # A control of a measurement (a Window): a group input binds it per use, a constant sets it.
    if origin is not None and str(getattr(origin, "type", "") or "") == "GROUP_INPUT":
        return "ok", None
    value, depends_on = driven_values.constant_value(sock)
    if value is not None:
        return "constant", value
    default = getattr(sock, "default_value", None)
    return ignored(f"{name} comes from {depends_on}: ACE needs a constant here, so its own value, {_number(default)}, is used.")


# Marking


def _trees():
    """Every node tree of the file: groups, and the trees of materials, the World and lights."""
    import bpy  # type: ignore

    data = getattr(bpy, "data", None)
    if data is None:
        return
    yield from getattr(data, "node_groups", []) or []
    for collection in ("materials", "worlds", "lights"):
        for owner in getattr(data, collection, []) or []:
            tree = getattr(owner, "node_tree", None)
            if tree is not None:
                yield tree


def find_tree(key: str):
    for tree in _trees():
        if tree_key(tree) == key:
            return tree
    return None


def _link_into(tree, item: IgnoredLink):
    node = tree.nodes.get(item.node_name) if hasattr(tree.nodes, "get") else None
    if node is None or item.input_index >= len(node.inputs):
        return None
    sock = node.inputs[item.input_index]
    if str(sock.identifier) != item.input_identifier:
        sock = next((s for s in node.inputs if str(s.identifier) == item.input_identifier), None)
    return _feeding_link(sock) if sock is not None else None


def mark_ignored_links(ignored) -> int:
    """Set is_valid to False on each ignored link still there; how many links were written.

    *ignored* maps a tree key to its IgnoredLinks. A flag already False is not written again, so
    marking from a depsgraph handler never loops.
    """
    written = 0
    for key, items in (ignored or {}).items():
        if not items:
            continue
        tree = find_tree(key)
        if tree is None or getattr(tree, "library", None) is not None:
            continue
        for item in items:
            link = _link_into(tree, item)
            if link is None or not bool(getattr(link, "is_valid", True)):
                continue
            with utils.logged_errors("link_rules.mark_ignored_links"):
                link.is_valid = False
                written += 1
    return written


def ignored_for_node(ignored, tree, node_name: str) -> list:
    """The IgnoredLinks into one node of *tree*."""
    return [item for item in (ignored or {}).get(tree_key(tree), ()) if item.node_name == node_name]
