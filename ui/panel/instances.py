"""The compiled model as the UI shows it, read from the engine and never from the trees.

- ``instance_report``: each ACE node (a definition: a tree and a node name), its uses
  (owner, then the group nodes entered), the signal each use computes and the uses it
  shares that signal with, and the sink conflicts it takes part in.
- ``source_uses``: for each source, the owners whose trees read it (the Audio Inputs list).
- ``compiled_tree_view``: one use of a tree, under the ids the sidebar gives the tree's own
  nodes, so a group opened in the editor lists its operators under the source each use
  gives it.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from ...core.abstract_tree import ACEAbstractLink, ACEAbstractTree

_report_cache: dict = {"key": None, "report": []}


@dataclass
class InstanceUse:
    """One use of a definition node."""

    instance_id: str
    owners: List[str]  # readable owner labels ("Cube › GeometryNodes", "World › World", ...)
    chain: tuple  # the group nodes entered from the owner's tree
    signal_id: str
    shared_with: int  # other uses, of any node, computing the same signal
    sources: List[str]  # stable ids of the sources the use reads


@dataclass
class DefinitionReport:
    """An ACE node as the user placed it, and everything compiled for it."""

    tree_name: str
    node_name: str
    category: str
    node_type: str
    uses: List[InstanceUse] = field(default_factory=list)
    conflicts: List[dict] = field(default_factory=list)


def compile_marker(engine) -> float:
    """Changes with every compile: what the UI's caches of the compiled model are keyed by."""
    return float(getattr(getattr(engine, "compiler", None), "_last_compile_time", 0.0) or 0.0)


def owner_label(owner_key: str, material_name: str = "") -> str:
    """A readable owner: "Object › Modifier", "Object › Material", "Light › name", "World › name"."""
    key = str(owner_key or "")
    for prefix, label in (("world:", "World"), ("light:", "Light")):
        if key.startswith(prefix):
            return f"{label} › {key[len(prefix):]}"
    obj, sep, modifier = key.partition(":")
    if sep:
        return f"{obj} › {modifier}"
    return f"{obj} › {material_name}" if material_name else obj


def _owners(frontend) -> List[str]:
    if frontend is None:
        return []
    material = str(getattr(frontend, "material_name", "") or "")
    return [owner_label(key, material) for key in tuple(getattr(frontend, "owner_keys", ()) or ())]


def _uses_by_signal(engine) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for signal in (getattr(engine, "signal_of", {}) or {}).values():
        counts[signal] = counts.get(signal, 0) + 1
    return counts


def instance_report(engine) -> List[DefinitionReport]:
    """Every compiled ACE node with its uses; cached until the next compile."""
    tree = getattr(engine, "instance_tree_ref", None) if engine is not None else None
    key = (id(engine), compile_marker(engine))
    if tree is None:
        return []
    if _report_cache["key"] == key:
        return _report_cache["report"]
    signal_of = dict(getattr(engine, "signal_of", {}) or {})
    counts = _uses_by_signal(engine)
    by_definition: Dict[tuple, DefinitionReport] = {}
    for node in sorted(tree.nodes.values(), key=lambda n: (n.definition_tree, n.name, n.use_chain, n.id)):
        if (node.parameters or {}).get("viewer"):
            continue  # a Viewer previews; it is not an ACE node of the user's
        def_key = (str(node.definition_tree or ""), str(node.name))
        report = by_definition.get(def_key)
        if report is None:
            report = by_definition[def_key] = DefinitionReport(
                tree_name=def_key[0], node_name=def_key[1], category=str(node.category), node_type=str(node.type),
            )
        signal = str(signal_of.get(node.id, "") or "")
        report.uses.append(InstanceUse(
            instance_id=str(node.id),
            owners=_owners(tree.frontends.get(node.frontend_uid)),
            chain=tuple(node.use_chain or ()),
            signal_id=signal,
            shared_with=max(0, counts.get(signal, 1) - 1),
            sources=sorted(str(v) for v in (node.root_sources or ()) if str(v)),
        ))
    for conflict in getattr(engine, "sink_conflicts", []) or []:
        involved = {conflict.get("kept", "")} | set(conflict.get("dropped", []) or [])
        for report in by_definition.values():
            if involved & {use.instance_id for use in report.uses}:
                report.conflicts.append(dict(conflict))
    result = sorted(by_definition.values(), key=lambda r: (r.tree_name.lower(), r.node_name.lower()))
    _report_cache.update(key=key, report=result)
    return result


def source_uses(engine) -> Dict[str, list]:
    """{source stable id: [frontend refs whose trees read it]}, in discovery order."""
    tree = getattr(engine, "instance_tree_ref", None) if engine is not None else None
    if tree is None:
        return {}
    found: Dict[str, Dict[str, object]] = {}
    for node in tree.nodes.values():
        frontend = tree.frontends.get(node.frontend_uid)
        if frontend is None:
            continue
        for source in node.root_sources or ():
            found.setdefault(str(source), {})[node.frontend_uid] = frontend
    return {
        source: sorted(refs.values(), key=lambda f: (int(getattr(f, "discovery_index", 0) or 0), str(f.uid)))
        for source, refs in found.items()
    }


def _compiled_is_current(engine, bl_tree) -> bool:
    """True when *bl_tree* is as the last compile read it (no edit since)."""
    from ...nodes.definitions import tree_content_signature

    recorded = (getattr(getattr(engine, "frontends", None), "_compiled_content_sigs", {}) or {}).get(
        int(bl_tree.as_pointer())
    )
    return recorded is not None and recorded == tree_content_signature(bl_tree)


def compiled_tree_view(engine, bl_tree, source_uid: str = "") -> Optional[ACEAbstractTree]:
    """The compiled nodes of one use of *bl_tree*, under the ids of the tree's own nodes.

    The use shown is the first reading *source_uid*, else the first. None when the tree was
    not compiled, or was edited since (the caller reads the tree itself then).
    """
    from ...nodes.identity import node_id

    tree = getattr(engine, "instance_tree_ref", None) if engine is not None else None
    if tree is None or bl_tree is None or not _compiled_is_current(engine, bl_tree):
        return None
    ptr = int(bl_tree.as_pointer())
    uses: Dict[tuple, list] = {}
    for node in tree.nodes.values():
        if int(node.definition_ptr or 0) == ptr:
            uses.setdefault((str(node.frontend_uid), tuple(node.use_chain or ())), []).append(node)
    if not uses:
        return None

    def order(item):
        (frontend_uid, chain), nodes = item
        frontend = tree.frontends.get(frontend_uid)
        reads = any(source_uid and source_uid in (n.root_sources or ()) for n in nodes)
        return (not reads, int(getattr(frontend, "discovery_index", 0) or 0), frontend_uid, chain)

    (_use, nodes) = sorted(uses.items(), key=order)[0]
    own_ids: Dict[str, str] = {}
    for node in nodes:
        bl_node = bl_tree.nodes.get(node.name)
        own = node_id(bl_node) if bl_node is not None and not (node.parameters or {}).get("viewer") else ""
        own_ids[node.id] = own or node.id
    view = ACEAbstractTree(frontends=dict(tree.frontends))
    for node in nodes:
        view.add_node(replace(node, id=own_ids[node.id], root_sources=set(node.root_sources or ())))
    for link in tree.links:
        if link.source_node_id in own_ids and link.target_node_id in own_ids:
            view.add_link(ACEAbstractLink(
                source_node_id=own_ids[link.source_node_id], target_node_id=own_ids[link.target_node_id],
                source_socket_idx=link.source_socket_idx, target_socket_idx=link.target_socket_idx,
            ))
    return view
