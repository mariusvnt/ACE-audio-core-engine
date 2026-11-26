"""Source nodes: which audio source each one reads.

The user adds them; the add-on never creates one. A new node reads the active source, and the
choice can be changed in the node's settings.
"""

import bpy  # type: ignore
from ..core.id_management import ACEIdManager
from ..core.proxy.manager import ACEProxyManager
from .. import utils


def _is_source_node(node) -> bool:
    parsed = ACEIdManager.parse_node_id(str(node.get("ace_id", "") or ""))
    if parsed:
        return parsed.get("category") == "source_mngmt" and parsed.get("type") == "src"
    return bool(node.get("ace_source_id"))


def _source_nodes(trees):
    for tree in trees:
        for node in getattr(tree, "nodes", []) or []:
            if _is_source_node(node):
                yield node


def _set_if_changed(node, key: str, value: str) -> None:
    if str(node.get(key, "") or "") != value:
        node[key] = value


def bind_node(node, source, scene) -> None:
    """Make *node* read *source*: identity, title and Proxy Object input."""
    ACEIdManager.ensure_source_uid(source)
    ACEProxyManager.ensure_proxy_for_source(source, scene=scene)
    name = str(source.name or "Source")
    _set_if_changed(node, "ace_source_id", str(source.stable_id))
    _set_if_changed(node, "ace_source_name", name)
    if node.label != name:
        node.label = node.name = name

    socket = next((s for s in node.inputs if s.name == "Proxy Object"), None)
    proxy_name = str(source.proxy_object_name or "")
    if socket is None or not proxy_name:
        return
    with utils.logged_errors("source_nodes.bind_node"):
        if socket.type == "OBJECT":
            proxy = bpy.data.objects.get(proxy_name)
            if socket.default_value != proxy:
                socket.default_value = proxy
        elif isinstance(socket.default_value, str) and socket.default_value != proxy_name:
            socket.default_value = proxy_name


def refresh_source_nodes(trees, scene) -> set[str]:
    """Bring the Source nodes of *trees* up to date with their sources; returns the ids they read.

    A node without a source (added before any existed) takes the active one. Only sources read
    by a node get a proxy object.
    """
    project = getattr(scene, "audiocore", None)
    sources = list(getattr(project, "audio_sources", []) or [])
    if not sources:
        return set()
    by_id = {str(source.stable_id): source for source in sources if source.stable_id}
    index = int(getattr(project, "active_audio_source_index", 0) or 0)
    active = sources[index] if 0 <= index < len(sources) else sources[0]

    read: set[str] = set()
    for node in _source_nodes(trees):
        source_id = str(node.get("ace_source_id", "") or "")
        source = by_id.get(source_id) if source_id else active
        if source is None:
            continue  # its source was removed: the orphan pruning decides
        with utils.logged_errors("source_nodes.refresh_source_nodes"):
            bind_node(node, source, scene)
            read.add(str(source.stable_id))
    return read
