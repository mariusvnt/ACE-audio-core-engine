"""Signals: each distinct computation of the scene, once.

The instance walker (``nodes/geometry/reader.py``) gives one abstract node per use of an ACE
node. Uses that compute the same thing get the same signal key: the same operation, mode and
parameters, fed by signals with equal keys, down to the same source read at the same sample
rate, block size and fps. A key never holds a sink name, a proxy reference or the node's place
in the editor, so the same RMS in two trees, or in two uses of a group, is one task, and that
task writes the sinks of every use.

Keys also carry state across compiles: a task whose key is unchanged is kept, with its state
(``graph_build.py``), and a source whose signals are unchanged keeps its worker graph and bake.
"""

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Tuple

from ...abstract_tree import ACEAbstractLink, ACEAbstractNode, ACEAbstractOutputLink, ACEAbstractTree, audio_links

# Parameters that are not part of the computation: the sink names and the editor order.
_SINK_PARAM_PREFIXES = ("attr_id_out_", "attr_id_custom_out_")
_LAYOUT_PARAMS = frozenset({"frontend_node_index"})


def _plain(value: Any):
    """*value* as JSON data, for keys and digests."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item") and not hasattr(value, "__len__"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_plain(v) for v in value]
        return sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return str(value)


def worker_params(node: ACEAbstractNode) -> Dict[str, Any]:
    """The parameters a task needs: all of them but the sink names (Attribute ID inputs too) and the editor order."""
    sink_inputs = {f"in_{sock.socket_idx}" for sock in node.inputs if "Attribute ID" in str(sock.name)}
    return {
        k: v for k, v in dict(node.parameters or {}).items()
        if k not in _LAYOUT_PARAMS and k not in sink_inputs and not str(k).startswith(_SINK_PARAM_PREFIXES)
    }


def key_params(node: ACEAbstractNode) -> Dict[str, Any]:
    """The parameters that decide what *node* computes.

    Proxy Object inputs name where audio comes from; the inputs say that. A linked split
    frequency is read again every frame, so its compile-time value is left out; the node and
    walk it is read through stay in.
    """
    params = worker_params(node)
    skip = {f"in_{sock.socket_idx}" for sock in node.inputs if str(sock.data_type) == "NodeSocketObject"}
    skip |= {f"in_{int(i)}" for i in params.get("driven_inputs") or ()}
    return {k: v for k, v in params.items() if k not in skip and not _is_object_ref(v)}


def _is_object_ref(value) -> bool:
    """An Object socket's value as definitions store it: ("object", name)."""
    return isinstance(value, (tuple, list)) and len(value) == 2 and value[0] == "object"


def _digest(payload) -> str:
    data = json.dumps(_plain(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(data, digest_size=8).hexdigest()


def _source_root(node: ACEAbstractNode) -> str:
    roots = sorted(str(v) for v in (node.root_sources or ()) if str(v))
    return str(node.source_binding_id or "") or (roots[0] if roots else "")


@dataclass
class SignalSet:
    """The compiled signals: a tree with one node per signal, and which uses each one stands for."""

    tree: ACEAbstractTree
    signal_of: Dict[str, str] = field(default_factory=dict)  # instance node id -> signal id
    instances: Dict[str, List[str]] = field(default_factory=dict)  # signal id -> instance ids, sorted


def signal_keys(instance_tree: ACEAbstractTree, source_keys: Dict[str, Tuple]) -> Dict[str, str]:
    """Signal id of every instance node; *source_keys* gives (sample rate, block size, fps) per source uid."""
    nodes = instance_tree.nodes
    inputs_of: Dict[str, List[Tuple[int, str, int]]] = {}
    for link in audio_links(instance_tree):
        if link.source_node_id in nodes and link.target_node_id in nodes:
            inputs_of.setdefault(link.target_node_id, []).append(
                (int(link.target_socket_idx), str(link.source_node_id), int(link.source_socket_idx))
            )
    keys: Dict[str, str] = {}
    for start in sorted(nodes):
        stack = [(start, False)]
        on_path: set = set()
        while stack:
            node_id, ready = stack.pop()
            if node_id in keys:
                continue
            upstream = [src for _i, src, _o in inputs_of.get(node_id, ())]
            if not ready:
                if node_id in on_path:  # a cycle (not possible in a node editor): cut it here
                    upstream = []
                else:
                    on_path.add(node_id)
                    stack.append((node_id, True))
                    stack.extend((src, False) for src in upstream if src not in keys and src not in on_path)
                    continue
            on_path.discard(node_id)
            keys[node_id] = _signal_id(nodes[node_id], inputs_of.get(node_id, ()), keys, source_keys)
    return keys


def _signal_id(node: ACEAbstractNode, inputs, keys: Dict[str, str], source_keys: Dict[str, Tuple]) -> str:
    if (node.category, node.type) == ("utilities", "preview_sink"):
        return node.id  # what is heard is chosen per node: never shared
    payload: Dict[str, Any] = {
        "op": [node.category, node.type],
        "params": key_params(node),
        "inputs": sorted([idx, keys.get(src, src), out] for idx, src, out in inputs),
    }
    if (node.category, node.type) == ("source_mngmt", "src"):
        root = _source_root(node)
        payload["source"] = [root, *source_keys.get(root, ())]
    return f"S_{node.category}_{node.type}_{_digest(payload)}"


def build_signals(instance_tree: ACEAbstractTree, source_keys: Dict[str, Tuple]) -> SignalSet:
    """One node per signal, standing for every instance with that key (the first one's values)."""
    keys = signal_keys(instance_tree, source_keys)
    instances: Dict[str, List[str]] = {}
    for instance_id in sorted(keys):
        instances.setdefault(keys[instance_id], []).append(instance_id)

    tree = ACEAbstractTree(
        frontends=dict(instance_tree.frontends),
        consumer_intents=list(instance_tree.consumer_intents),
    )
    for sid, instance_ids in sorted(instances.items()):
        uses = [instance_tree.nodes[i] for i in instance_ids]
        first = uses[0]
        sinks: Dict[int, List[Tuple[str, str, str]]] = {}
        for use in uses:
            for sock in use.outputs:
                name = str((use.parameters or {}).get(f"attr_id_out_{sock.socket_idx}", "") or "").strip()
                if name:
                    sinks.setdefault(int(sock.socket_idx), []).append((use.id, name, str(use.frontend_kind or "")))
        tree.add_node(replace(
            first,
            id=sid,
            parameters=dict(first.parameters or {}),
            inputs=[replace(s, id=f"{sid}_in_{s.socket_idx}") for s in first.inputs],
            outputs=[replace(s, id=f"{sid}_out_{s.socket_idx}") for s in first.outputs],
            root_sources=set().union(*(set(u.root_sources or ()) for u in uses)),
            instance_ids=list(instance_ids),
            instance_sinks=sinks,
        ))

    seen_links: set = set()
    for link in instance_tree.links:
        src, dst = keys.get(link.source_node_id), keys.get(link.target_node_id)
        marker = (src, dst, int(link.source_socket_idx), int(link.target_socket_idx))
        if src is None or dst is None or marker in seen_links:
            continue
        seen_links.add(marker)
        tree.add_link(ACEAbstractLink(source_node_id=src, target_node_id=dst,
                                      source_socket_idx=marker[2], target_socket_idx=marker[3]))
    seen_outputs: set = set()
    for out in instance_tree.group_output_links:
        src = keys.get(out.source_node_id)
        marker = (src, int(out.source_socket_idx), int(out.target_socket_idx), str(out.target_socket_name or ""))
        if src is None or marker in seen_outputs:
            continue
        seen_outputs.add(marker)
        tree.add_group_output_link(replace(out, source_node_id=src) if isinstance(out, ACEAbstractOutputLink) else out)
    return SignalSet(tree=tree, signal_of=keys, instances=instances)


def graph_signature(graph) -> str:
    """What a source graph computes and outputs: equal when only sinks or other sources changed."""
    return _digest({
        "tasks": sorted(str(k) for k in getattr(graph, "tasks", {}) or {}),
        "outputs": sorted(str(k) for k in getattr(graph, "output_attr_keys", []) or []),
        "block_size": int(getattr(graph, "block_size", 0) or 0),
    })


def resolve_sink_conflicts(graphs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Two signals writing one sink: the use first in id order keeps it; the others' bindings go.

    Uses of one definition node share its sink names, so uses computing different values
    (different group inputs, say) claim the same proxy attribute or scene property.
    """
    claims: Dict[Tuple[str, str, str], List[Tuple[str, str, str, Any]]] = {}
    for graph_key in sorted(graphs):
        graph = graphs[graph_key]
        for attr_key, bindings in sorted(dict(getattr(graph, "sink_bindings_by_attr", {}) or {}).items()):
            for binding in bindings or []:
                target = (str(binding.sink_kind), str(binding.target_object_name), str(binding.target_data_name))
                claims.setdefault(target, []).append((str(binding.instance_id or ""), str(graph_key), str(attr_key), binding))
    conflicts: List[Dict[str, Any]] = []
    for target, items in sorted(claims.items()):
        if len({(g, a) for _i, g, a, _b in items}) <= 1:
            continue
        items.sort(key=lambda item: item[:3])
        _kept_id, kept_graph, kept_attr, _kept_binding = items[0]
        dropped = []
        for instance_id, graph_key, attr_key, binding in items[1:]:
            if (graph_key, attr_key) == (kept_graph, kept_attr):
                continue
            graph = graphs[graph_key]
            graph.sink_bindings_by_attr[attr_key] = [
                b for b in graph.sink_bindings_by_attr.get(attr_key, []) if b is not binding
            ]
            dropped.append(instance_id)
        conflicts.append({
            "sink_kind": target[0],
            "target": target[1],
            "sink_name": target[2],
            "kept": items[0][0],
            "dropped": sorted(set(dropped)),
        })
    return conflicts
