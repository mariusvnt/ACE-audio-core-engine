"""Graph IR sent to workers: offload policy, IR construction, digest and sharding."""

import hashlib
import json
from typing import Any, Dict, List, Optional

import numpy as np

from .... import utils
from ...abstract_tree import ACEAbstractNode, ACEAbstractTree, audio_links
from ...tasks.chunk import SAMPLE_RATE
from ...tasks.graph import ACEAudioGraph
from ...tasks.source import SourceTask
from ..compiler.signals import worker_params


def is_offload_candidate(node: Optional[ACEAbstractNode]) -> bool:
    if node is None:
        return False
    # Utilities stay on the main thread: a running sum must not jump when execution moves.
    return str(getattr(node, "category", "") or "") == "stream"


def sanitize(value: Any):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    return str(value)


def graph_uid(graph: ACEAudioGraph) -> str:
    key = (
        str(getattr(graph, "source_uid", "") or "")
        or str(getattr(getattr(graph, "source_context", None), "stable_id", "") or "")
        or str(getattr(graph, "proxy_name", "") or "")
    )
    return key or f"graph:{id(graph)}"


def shard_attr_keys(attr_keys: List[str], source_groups: Optional[Dict[str, str]] = None) -> List[List[str]]:
    """One shard per source lineage when several sources feed the graph, else one shard."""
    keys = [str(v) for v in list(attr_keys or []) if str(v)]
    if not keys:
        return []
    if not source_groups or len(set(source_groups.values())) <= 1:
        return [keys]
    buckets: Dict[str, List[str]] = {}
    ungrouped: List[str] = []
    for k in keys:
        group = source_groups.get(k)
        if group:
            buckets.setdefault(str(group), []).append(k)
        else:
            ungrouped.append(k)
    shards = [v for v in buckets.values() if v]
    if ungrouped:
        shards.append(ungrouped)
    return shards or [keys]


def ir_digest(parallel_ir: Dict[str, Any]) -> str:
    try:
        payload = json.dumps(parallel_ir, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        utils.note_swallowed("ir.ir_digest", exc)
        payload = str(parallel_ir).encode("utf-8", "ignore")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def _offload_outputs(graph: ACEAudioGraph, abstract_tree: ACEAbstractTree) -> List[str]:
    keys = []
    for attr_key in list(getattr(graph, "output_attr_keys", []) or []):
        node_id, _, sock_idx = attr_key.partition(":")
        if is_offload_candidate(abstract_tree.nodes.get(node_id)) and (not sock_idx or sock_idx.isdigit()):
            keys.append(attr_key)
    return keys


def _upstream_nodes(start_ids, reverse_adj, graph_task_ids) -> set:
    required: set = set()
    stack = list(start_ids)
    while stack:
        node_id = str(stack.pop())
        if node_id in required:
            continue
        required.add(node_id)
        stack.extend(src for src in reverse_adj.get(node_id, set()) if src in graph_task_ids and src not in required)
    return required


def _source_ir(graph: ACEAudioGraph, node_id: str) -> Dict[str, Any]:
    task = graph.tasks.get(node_id)
    ctx = getattr(task, "source_context", None) if isinstance(task, SourceTask) else None
    # The compile-time rate, not the task's: a task kept from the last compile has opened its
    # file and knows another one, which would send an unchanged graph again. Jobs carry each
    # chunk's own rate; this is the fallback.
    sample_rate = int(
        getattr(graph, "sample_rate", 0)
        or getattr(ctx, "sample_rate", 0)
        or (getattr(task, "sample_rate", 0) if isinstance(task, SourceTask) else 0)
        or SAMPLE_RATE
    )
    return {"sample_rate": max(sample_rate, 1), "lineage": [node_id], "spectrum_token": [node_id]}


def _nodes_ir(graph, abstract_tree, node_ids, fps):
    nodes_ir: Dict[str, Dict[str, Any]] = {}
    source_ir: Dict[str, Dict[str, Any]] = {}
    for node_id in sorted(node_ids):
        node = abstract_tree.nodes.get(node_id)
        if node is None:
            continue
        # No sink names and no editor order: renaming a sink does not change the graph.
        params = {str(k): sanitize(v) for k, v in worker_params(node).items()}
        nodes_ir[node_id] = {
            "category": str(getattr(node, "category", "") or ""),
            "type": str(getattr(node, "type", "") or ""),
            "parameters": params,
        }
        if node.category == "source_mngmt" and node.type == "src":
            source_ir[node_id] = _source_ir(graph, node_id)
    return nodes_ir, source_ir


def _links_ir(abstract_tree, nodes_ir):
    links = []
    for link in audio_links(abstract_tree):
        src_id = str(getattr(link, "source_node_id", "") or "")
        dst_id = str(getattr(link, "target_node_id", "") or "")
        if src_id in nodes_ir and dst_id in nodes_ir:
            links.append((
                src_id,
                int(getattr(link, "source_socket_idx", 0) or 0),
                dst_id,
                int(getattr(link, "target_socket_idx", 0) or 0),
            ))
    return links


def _source_groups(attr_keys, source_ids, reverse_adj) -> Dict[str, str]:
    """Each output's nearest source ancestor."""
    groups: Dict[str, str] = {}
    for attr_key in attr_keys:
        stack, visited = [attr_key.partition(":")[0]], set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            if cur in source_ids:
                groups[attr_key] = cur
                break
            stack.extend(p for p in reverse_adj.get(cur, set()) if p not in visited)
    return groups


def build_graph_ir(graph: ACEAudioGraph, abstract_tree: Optional[ACEAbstractTree], output_defs: Dict[str, Any], fps: float) -> bool:
    """Set the graph's parallel_* fields; True when some outputs are offloaded."""
    graph.parallel_graph_uid = graph_uid(graph)
    graph.parallel_ir = {}
    graph.parallel_ir_digest = ""
    graph.parallel_offload_attr_keys = []
    graph.parallel_output_shards = []
    graph.parallel_local_attr_keys = list(getattr(graph, "output_attr_keys", []) or [])
    if abstract_tree is None:
        return False
    offload_keys = _offload_outputs(graph, abstract_tree)
    if not offload_keys:
        return False

    reverse_adj: Dict[str, set] = {}
    for link in audio_links(abstract_tree):
        reverse_adj.setdefault(str(link.target_node_id), set()).add(str(link.source_node_id))
    required = _upstream_nodes(
        {k.partition(":")[0] for k in offload_keys}, reverse_adj, set(getattr(graph, "tasks", {}).keys())
    )
    nodes_ir, source_ir = _nodes_ir(graph, abstract_tree, required, fps)
    output_specs: Dict[str, Dict[str, Any]] = {}
    for attr_key in offload_keys:
        node_id, _, sock_idx = attr_key.partition(":")
        if node_id in nodes_ir:
            output_specs[attr_key] = {
                "node_id": node_id,
                "socket_idx": int(sock_idx) if sock_idx else 0,
                "attr_type": str(dict(output_defs.get(attr_key, {}) or {}).get("type", "FLOAT") or "FLOAT"),
            }
    if not output_specs:
        return False

    source_ids = sorted(source_ir)
    graph.parallel_ir = {
        "nodes": nodes_ir,
        "links": _links_ir(abstract_tree, nodes_ir),
        "source_node_ids": source_ids,
        "source_ir": source_ir,
        "output_specs": output_specs,
    }
    graph.parallel_offload_attr_keys = sorted(output_specs)
    groups = _source_groups(graph.parallel_offload_attr_keys, set(source_ids), reverse_adj)
    graph.parallel_output_shard_groups = groups
    graph.parallel_output_shards = shard_attr_keys(
        graph.parallel_offload_attr_keys, source_groups=groups if len(source_ids) > 1 else None
    )
    offloaded = set(graph.parallel_offload_attr_keys)
    graph.parallel_local_attr_keys = [k for k in graph.parallel_local_attr_keys if k not in offloaded]
    graph.parallel_ir_digest = ir_digest(graph.parallel_ir)
    return True
