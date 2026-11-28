"""Source nodes: which audio source each one reads.

The user chooses it; ACE never writes it during a compile.

- **Geometry Nodes:** the node's Proxy Object socket holds the source's proxy
  object, and the proxy carries the source's id (``ace_source_id``).
- **Shaders:** the node uses the source's own group (``.ACE_SHADER_source_<id>``),
  which carries the id and reads the source's BPM and Beats per Bar from the
  scene.

A node whose proxy or group no longer belongs to a source reads nothing.
"""

import bpy  # type: ignore
from ..core.proxy.manager import ACEProxyManager
from .identity import SOURCE_ID_KEY, ace_nodes
from ..vars import SOURCE_KIND
from .. import utils


def is_source_node(node) -> bool:
    from .identity import node_kind

    return node_kind(node) == SOURCE_KIND


def _proxy_socket(node):
    return next((s for s in getattr(node, "inputs", []) or [] if getattr(s, "name", "") == "Proxy Object"), None)


def node_source_id(node) -> str:
    """The stable id of the source *node* reads, "" when it reads none."""
    group = getattr(node, "node_tree", None)
    if str(getattr(group, "bl_idname", "") or "") == "ShaderNodeTree":
        with utils.logged_errors("source_nodes.node_source_id"):
            return str(group.get(SOURCE_ID_KEY, "") or "")
        return ""
    socket = _proxy_socket(node)
    if socket is None or getattr(socket, "is_linked", False):
        return ""
    proxy = getattr(socket, "default_value", None)
    if proxy is None or isinstance(proxy, str):
        return ""
    with utils.logged_errors("source_nodes.node_source_id#proxy"):
        return str(proxy.get(SOURCE_ID_KEY, "") or "")
    return ""


def set_node_source(node, source, scene) -> None:
    """Make *node* read *source*. Only for the user's own actions: adding the node, choosing its source."""
    from ..core.id_management import ACEIdManager

    ACEIdManager.ensure_source_uid(source)
    old_name = ""
    current = node_source_id(node)
    if current:
        project = getattr(scene, "audiocore", None)
        old_name = next((s.name for s in getattr(project, "audio_sources", []) if s.stable_id == current), "")
    group = getattr(node, "node_tree", None)
    if str(getattr(group, "bl_idname", "") or "") == "ShaderNodeTree":
        from .shader.factory import ACEShaderNodeFactory

        source_group = ACEShaderNodeFactory.ensure_source_group(source)
        if source_group is not None and node.node_tree != source_group:
            node.node_tree = source_group
    else:
        proxy = ACEProxyManager.ensure_proxy_for_source(source, scene=scene)
        socket = _proxy_socket(node)
        if socket is not None and proxy is not None and socket.default_value != proxy:
            socket.default_value = proxy
    name = str(source.name or "Source")
    if not node.label or node.label in (old_name, "Source"):
        node.label = name


def source_nodes(trees):
    """(node, source id) of every Source node in *trees*."""
    for tree in trees:
        for node, category, node_type in ace_nodes(tree):
            if (category, node_type) == SOURCE_KIND:
                yield node, node_source_id(node)


def editable_trees():
    from .identity import is_ace_group

    trees = [ng for ng in bpy.data.node_groups if not is_ace_group(ng)]
    trees += [m.node_tree for m in bpy.data.materials if m.node_tree is not None]
    trees += [w.node_tree for w in bpy.data.worlds if w.node_tree is not None]
    return [t for t in trees if getattr(t, "library", None) is None]
