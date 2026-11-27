"""Sink bindings: where each compiled output is written, and the repair of sink names.

Sink names belong to the user's tree: the add menu writes them once when it
creates a node. A compile only reads them, except for one repair: a name that
two definition nodes claim (after Shift+D, a copied material or an append), or
a node with no name at all, gets a fresh name on the later node, in a stable
order. That is one write, after which the names are stable.

- **Geometry Nodes** values go to an attribute of the source's proxy mesh.
- **Shader** values go to a scene property (one global value per sink), which
  the node's sink group reads with View Layer Attribute nodes.
"""

import bpy
from typing import Dict, List, Optional
from ... import utils
from ..abstract_tree import ACEAbstractNode, ACEAbstractTree
from ..tasks.graph import ACEAudioGraph, ACESinkBinding
from .common import FRONTEND_KIND_SHADER


class SinkResolver:
    """Sink bindings for proxy-attribute and scene-property outputs."""

    def __init__(self, engine):
        self.engine = engine

    @staticmethod
    def _build_source_proxy_lookup(scene) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        ace = getattr(scene, "audiocore", None)
        for src in getattr(ace, "audio_sources", []) or []:
            sid = getattr(src, "stable_id", "")
            proxy_name = getattr(src, "proxy_object_name", "")
            if proxy_name:
                lookup[proxy_name] = proxy_name
            if sid and proxy_name:
                lookup[sid] = proxy_name
        return lookup

    @staticmethod
    def _lookup_proxy_object(proxy_name: str):
        if not proxy_name:
            return None
        try:
            return bpy.data.objects.get(str(proxy_name))
        except Exception as exc:
            utils.note_swallowed("sinks.SinkResolver._lookup_proxy_object", exc)
            return None

    @staticmethod
    def _resolve_proxy_name_for_output_node(
        node: Optional[ACEAbstractNode],
        source_lookup: Dict[str, str],
        unique_proxies: list[str] | tuple[str, ...],
    ) -> str:
        if node is None:
            return ""
        for root in sorted(set(getattr(node, "root_sources", set()) or set())):
            proxy_name = str(source_lookup.get(str(root), "") or "")
            if proxy_name:
                return proxy_name
        if len(unique_proxies) == 1:
            return str(unique_proxies[0] or "")
        return ""

    def _resolve_sink_bindings_for_graph(
        self,
        graph: ACEAudioGraph,
        abstract_tree: ACEAbstractTree,
        source_uid: str,
        proxy_name: str,
        single_source_mode: bool,
    ) -> Dict[str, List[ACESinkBinding]]:
        """For each output of *graph*: the sinks its value goes to, named by the user's tree."""
        bindings_by_attr: Dict[str, List[ACESinkBinding]] = {}
        scene_name = str(getattr(self.engine, "scene_name", "") or "")
        for attr_key in list(getattr(graph, "output_attr_keys", []) or []):
            info = self.engine.writer.output_defs.get(attr_key, {}) if self.engine.writer else {}
            node = abstract_tree.nodes.get(str(info.get("node_id", "") or ""))
            sink_name = str(info.get("attr_id", "") or "").strip()
            if node is None or not sink_name:
                bindings_by_attr[attr_key] = []
                continue
            attr_type = str(info.get("type", "FLOAT") or "FLOAT")
            if str(getattr(node, "frontend_kind", "") or "") == FRONTEND_KIND_SHADER:
                binding = ACESinkBinding(
                    sink_kind="SCENE_PROPERTY",
                    target_object_name=scene_name,
                    target_data_name=sink_name,
                    attr_type=attr_type,
                )
            else:
                binding = ACESinkBinding(
                    sink_kind="PROXY_ATTRIBUTE",
                    target_object_name=proxy_name,
                    target_data_name=sink_name,
                    attr_type=attr_type,
                )
            bindings_by_attr[attr_key] = [binding]
        return bindings_by_attr

    def repair_sink_names(self, trees, scene) -> int:
        """Give every output node of *trees* names of its own; returns how many nodes were renamed.

        Names of trees that can't be edited (linked data) are claimed first, so a local copy
        gives way to them. Within that order, trees go by name and nodes by name.
        """
        from ...nodes.identity import ace_nodes, is_output_kind, new_sink_name, sink_names, value_outputs

        utils.assert_main_thread("repair_sink_names")
        shader_writes = self.engine._structural_tree_writes_allowed(scene, frontend_kind=FRONTEND_KIND_SHADER)
        ordered = sorted(
            (t for t in trees if t is not None),
            key=lambda t: (bool(getattr(t, "is_editable", True)), str(getattr(t, "name_full", "") or getattr(t, "name", ""))),
        )
        claimed: set[str] = set()
        pending = []
        for tree in ordered:
            nodes = sorted(ace_nodes(tree), key=lambda item: str(getattr(item[0], "name", "")))
            for node, category, node_type in nodes:
                if not is_output_kind(category, node_type):
                    continue
                names = sink_names(node)
                bad = []
                for out_idx, _output_name in value_outputs(category, node_type):
                    name = names.get(out_idx)
                    if name is None or name in claimed:
                        bad.append(out_idx)
                    else:
                        claimed.add(name)
                if bad:
                    pending.append((tree, node, category, node_type, names, bad))

        renamed = 0
        for tree, node, category, node_type, names, bad in pending:
            is_shader = str(getattr(tree, "bl_idname", "") or "") == "ShaderNodeTree"
            if not bool(getattr(tree, "is_editable", True)) or (is_shader and not shader_writes):
                continue
            outputs = dict(value_outputs(category, node_type))
            for out_idx in bad:
                names[out_idx] = new_sink_name(outputs[out_idx], claimed)
                claimed.add(names[out_idx])
            with utils.logged_errors("sinks.SinkResolver.repair_sink_names"):
                write_sink_names(node, category, node_type, names)
                renamed += 1
                utils.log(f"ACE: '{tree.name}' › '{node.name}' got sink name(s) of its own: "
                          f"{', '.join(names[i] for i in bad)}", "INFO")
        return renamed


def write_sink_names(node, category: str, node_type: str, names: Dict[int, str]) -> None:
    """Store *names* (output index -> name) where *node*'s tree holds them.

    For the add menu, which names a node it creates, and the collision repair: a compile writes nothing else.
    """
    from ...nodes.identity import attribute_socket_name, value_outputs

    slots = value_outputs(category, node_type)
    if str(getattr(getattr(node, "id_data", None), "bl_idname", "") or "") == "ShaderNodeTree":
        from ...nodes.shader.factory import ACEShaderNodeFactory

        ordered = [names.get(out_idx, "") for out_idx, _name in slots]
        group = ACEShaderNodeFactory.ensure_sink_group(category, node_type, ordered)
        if node.node_tree != group:
            node.node_tree = group
        return
    for slot, (out_idx, _name) in enumerate(slots):
        sock = node.inputs.get(attribute_socket_name(slot))
        if sock is not None and out_idx in names and sock.default_value != names[out_idx]:
            sock.default_value = names[out_idx]
