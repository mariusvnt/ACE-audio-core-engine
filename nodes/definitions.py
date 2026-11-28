"""Definitions: each node tree parsed once, into plain data that depends on its contents only.

A definition holds what an instance walk needs from one tree, without following
any group into another tree:

- its ACE nodes: kind, settings, unlinked input values, sink names, and for each
  linked input whether the link comes from the tree's Group Input;
- its other groups (user groups), with their unlinked input values and the
  inputs fed by the tree's own Group Input;
- the flows: from each output that carries data (an ACE node's, the Group
  Input's, a user group's), the inputs it reaches through any number of other
  nodes (reroutes, math, switches...).

Definitions are cached by tree, keyed by the tree's content signature, so a
compile parses again only the trees that changed. They hold no Blender data:
nothing in them outlives an undo.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any

from ..vars import ACE_NODES, SOURCE_KIND, SPLITTER_KIND, resolve_node_mode
from .. import utils
from .identity import is_ace_group, node_kind, sink_names, tree_key

GROUP_INPUT = "gi"


@dataclass
class AceEntry:
    name: str
    category: str
    node_type: str
    order: int
    inputs: list = field(default_factory=list)  # (name, data type, default value)
    outputs: list = field(default_factory=list)  # (name, data type)
    params: dict = field(default_factory=dict)
    # Input index -> Group Input socket identifier, for inputs linked from the Group Input.
    bound_inputs: dict = field(default_factory=dict)
    linked_inputs: tuple = ()
    source_id: str = ""  # a Source node's own source (its proxy or its shader group)
    source_input: str = ""  # a Source node whose Proxy Object comes from this Group Input socket
    ignored: list = field(default_factory=list)  # link_rules.IgnoredLink per input ACE ignores


@dataclass
class GroupEntry:
    name: str
    tree_name: str
    defaults: dict = field(default_factory=dict)  # socket identifier -> plain value (unlinked inputs)
    bound_inputs: dict = field(default_factory=dict)  # socket identifier -> Group Input socket identifier
    linked_inputs: tuple = ()


@dataclass
class Definition:
    key: str
    signature: str
    ace: dict = field(default_factory=dict)  # node name -> AceEntry
    groups: dict = field(default_factory=dict)  # node name -> GroupEntry
    # ("ace", name, out_idx) | ("gi", identifier) | ("grp", name, identifier)
    #   -> [("ace", name, in_idx) | ("go", identifier, idx, socket name) | ("grp", name, identifier)
    #       | ("viewer", name, identifier)]
    flows: dict = field(default_factory=dict)
    viewers: dict = field(default_factory=dict)  # viewer name -> identifier of the input it plays
    ignored: list = field(default_factory=list)  # link_rules.IgnoredLink: links into ACE nodes that ACE ignores

    @property
    def uses_ace(self) -> bool:
        return bool(self.ace)


_cache: dict[str, Definition] = {}


def clear_cache() -> None:
    _cache.clear()


def plain_value(value) -> Any:
    """A socket value as plain data: an object by name, vectors as tuples."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "bl_rna") or hasattr(value, "name") and not hasattr(value, "__len__"):
        return ("object", str(getattr(value, "name", "") or ""))
    try:
        return tuple(float(v) for v in value)
    except TypeError:
        return repr(value)


def _socket_value(sock):
    value = getattr(sock, "default_value", None)
    if hasattr(value, "name"):
        return value.name
    if hasattr(value, "__len__") and not isinstance(value, str):
        return tuple(value)
    return value


def _node_pointer(item) -> int:
    try:
        return int(item.as_pointer())
    except Exception as exc:
        utils.note_swallowed("definitions._node_pointer", exc)
        return id(item)


def tree_content_signature(tree) -> str:
    """Hash of what a definition reads from *tree*: nodes, their settings and inputs, links, interface.

    Node positions, labels and selection are left out: changing them needs no recompile. An
    ACE node's name is in: it is the node's identity.
    """
    tokens: list = []
    for item in getattr(getattr(tree, "interface", None), "items_tree", []) or []:
        tokens.append(("if", str(getattr(item, "in_out", "")), str(getattr(item, "name", "")),
                       str(getattr(item, "socket_type", "")), str(getattr(item, "identifier", ""))))
    for node in getattr(tree, "nodes", []) or []:
        ace_props: tuple = ()
        values: tuple = ()
        name = ""
        get_keys = getattr(node, "keys", None)
        keys = list(get_keys()) if callable(get_keys) else []
        is_group = str(getattr(node, "type", "") or "") == "GROUP"
        if node_kind(node)[0] or is_group:
            name = str(getattr(node, "name", "") or "")
            ace_props = tuple(sorted((k, repr(node[k])) for k in keys if str(k).startswith("ace_")))
            values = tuple(
                (str(getattr(sock, "identifier", "")), bool(getattr(sock, "is_linked", False)), repr(_socket_value(sock)))
                for sock in getattr(node, "inputs", []) or []
            )
        tokens.append((
            "n", _node_pointer(node), str(getattr(node, "bl_idname", "")),
            str(getattr(getattr(node, "node_tree", None), "name", "") or ""),
            bool(getattr(node, "mute", False)), name, ace_props, values,
            len(getattr(node, "outputs", []) or []),
        ))
    for link in getattr(tree, "links", []) or []:
        tokens.append((
            "l", _node_pointer(link.from_node), str(link.from_socket.identifier),
            _node_pointer(link.to_node), str(link.to_socket.identifier),
            # Not is_valid: ACE marks the links it ignores, and that must not read as an edit.
            bool(getattr(link, "is_muted", False)),
        ))
    payload = repr(tokens).encode("utf-8", "ignore")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def definition(tree) -> Definition:
    """The definition of *tree*, parsed again only when its content changed."""
    key = tree_key(tree)
    signature = tree_content_signature(tree)
    cached = _cache.get(key)
    if cached is not None and cached.signature == signature:
        return cached
    parsed = _parse(tree, key, signature)
    _cache[key] = parsed
    return parsed


def is_user_group_node(node) -> bool:
    if str(getattr(node, "type", "") or "") != "GROUP":
        return False
    group = getattr(node, "node_tree", None)
    return group is not None and not is_ace_group(group)


def tree_uses_ace(tree, _seen=None) -> bool:
    """True when *tree* or a group it uses (at any depth) holds an ACE node."""
    if tree is None:
        return False
    seen = _seen if _seen is not None else {}
    key = _node_pointer(tree)
    if key in seen:
        return seen[key]
    seen[key] = False
    found = False
    for node in getattr(tree, "nodes", []) or []:
        if node_kind(node)[0]:
            found = True
            break
        if is_user_group_node(node) and tree_uses_ace(node.node_tree, seen):
            found = True
            break
    seen[key] = found
    return found


def hierarchy_trees(tree) -> list:
    """*tree* and every user group it uses, at any depth, each once."""
    found: list = []
    seen: set = set()
    stack = [tree] if tree is not None else []
    while stack and len(found) < 4096:
        current = stack.pop()
        marker = _node_pointer(current)
        if marker in seen:
            continue
        seen.add(marker)
        found.append(current)
        for node in getattr(current, "nodes", []) or []:
            if is_user_group_node(node):
                stack.append(node.node_tree)
    return found


def relevant_inputs(tree, _depth: int = 0) -> set[str]:
    """The Group Input socket identifiers of *tree* whose value reaches an ACE node's setting or source."""
    if tree is None or _depth > 32:
        return set()
    parsed = definition(tree)
    found = {ident for entry in parsed.ace.values() for ident in entry.bound_inputs.values()}
    found |= {entry.source_input for entry in parsed.ace.values() if entry.source_input}
    for name, group in parsed.groups.items():
        if not group.bound_inputs:
            continue
        node = tree.nodes.get(name) if hasattr(tree.nodes, "get") else None
        inner = relevant_inputs(getattr(node, "node_tree", None), _depth + 1)
        found |= {outer for inner_ident, outer in group.bound_inputs.items() if inner_ident in inner}
    return found


def _feed(sock) -> str | None:
    """The Group Input socket identifier feeding *sock* (through reroutes), or None."""
    for _ in range(64):
        link = next((lk for lk in getattr(sock, "links", []) or [] if not getattr(lk, "is_muted", False)), None)
        if link is None:
            return None
        node = link.from_node
        if str(getattr(node, "type", "") or "") == "GROUP_INPUT":
            return str(link.from_socket.identifier)
        if str(getattr(node, "bl_idname", "") or "") != "NodeReroute":
            return None
        sock = node.inputs[0]
    return None


def _input_index(node, to_socket) -> int:
    idx = getattr(to_socket, "index", None)
    if idx is not None:
        return int(idx)
    identifier = getattr(to_socket, "identifier", None)
    for i, sock in enumerate(getattr(node, "inputs", []) or []):
        if sock == to_socket or (identifier is not None and getattr(sock, "identifier", None) == identifier):
            return i
    return 0


def _targets(start_socket) -> list:
    """The inputs *start_socket* reaches, through any nodes that are neither ACE nor groups."""
    found: list = []
    seen: set = set()
    stack = [start_socket]
    while stack:
        sock = stack.pop()
        for link in getattr(sock, "links", []) or []:
            node, to_socket = getattr(link, "to_node", None), getattr(link, "to_socket", None)
            if node is None or to_socket is None:
                continue
            marker = (_node_pointer(node), str(getattr(to_socket, "identifier", "")))
            if marker in seen:
                continue
            seen.add(marker)
            node_type = str(getattr(node, "type", "") or "")
            if node_kind(node)[0]:
                found.append(("ace", node.name, _input_index(node, to_socket)))
            elif node_type == "GROUP_OUTPUT":
                found.append(("go", str(to_socket.identifier), _input_index(node, to_socket), str(to_socket.name)))
            elif is_user_group_node(node):
                found.append(("grp", node.name, str(to_socket.identifier)))
            elif str(getattr(node, "bl_idname", "") or "") == "GeometryNodeViewer":
                found.append(("viewer", node.name, str(to_socket.identifier)))
            else:
                stack.extend(out for out in getattr(node, "outputs", []) or [] if getattr(out, "is_linked", False))
    return found


def _parse_ace(node, category: str, node_type: str, order: int) -> AceEntry:
    if (category, node_type) == SPLITTER_KIND:
        input_defs = [("NodeSocketBundle", "Audio In")] + [
            ("NodeSocketFloat", f"Split Freq {k + 1}") for k in range(len(node.inputs) - 1)
        ]
        output_defs = [("NodeSocketBundle", "Band")] * len(node.outputs)
    else:
        node_def = ACE_NODES[category]["nodes"][node_type]
        input_defs = node_def.get("inputs", [])
        output_defs = node_def.get("outputs", [])
    from .link_rules import judge_input

    entry = AceEntry(name=str(node.name), category=category, node_type=node_type, order=order)
    linked = []
    for i, sock in enumerate(node.inputs):
        data_type = input_defs[i][0] if i < len(input_defs) else (
            "NodeSocketString" if "Attribute ID" in str(sock.name) else "NodeSocketFloat")
        value = getattr(sock, "default_value", None)
        entry.inputs.append((str(sock.name), data_type, plain_value(value)))
        verdict, detail = judge_input(node, category, node_type, i, sock, data_type) if sock.is_linked else ("ok", None)
        if verdict == "constant":
            entry.params[f"in_{i}"] = detail  # a constant chain: its value, as if typed in
            continue
        if verdict == "ignored":
            entry.ignored.append(detail)  # read as the socket's own value
            if value is not None and not hasattr(value, "bl_rna") and "Bundle" not in str(data_type):
                entry.params[f"in_{i}"] = value.to_tuple() if hasattr(value, "to_tuple") else value
            continue
        if sock.is_linked:
            linked.append(i)
            ident = _feed(sock)
            if ident is not None:
                entry.bound_inputs[i] = ident
        else:
            if hasattr(value, "to_tuple"):
                value = value.to_tuple()
            elif hasattr(value, "to_list"):
                value = value.to_list()
            if value is not None and not hasattr(value, "bl_rna"):
                entry.params[f"in_{i}"] = value
    entry.linked_inputs = tuple(linked)
    for out_idx, attr_name in sink_names(node).items():
        entry.params[f"attr_id_out_{out_idx}"] = attr_name
        entry.params[f"attr_id_custom_out_{out_idx}"] = True
    mode = resolve_node_mode(category, node_type, node.get("ace_mode"))
    if mode:
        entry.params["mode"] = mode
    entry.params["preview_enabled"] = bool(node.get("ace_preview_enabled", True))
    entry.params["frontend_node_index"] = order
    for i, sock in enumerate(node.outputs):
        entry.outputs.append((str(sock.name), output_defs[i][0] if i < len(output_defs) else "NodeSocketFloat"))
    if (category, node_type) == SOURCE_KIND:
        from .source_nodes import node_source_id

        entry.source_id = node_source_id(node)
        proxy = next((s for s in node.inputs if s.name == "Proxy Object"), None)
        if proxy is not None and proxy.is_linked:
            entry.source_input = _feed(proxy) or ""
    return entry


def _parse(tree, key: str, signature: str) -> Definition:
    parsed = Definition(key=key, signature=signature)
    for order, node in enumerate(getattr(tree, "nodes", []) or []):
        category, node_type = node_kind(node)
        node_type_name = str(getattr(node, "type", "") or "")
        if category:
            with utils.logged_errors("definitions._parse#ace"):
                parsed.ace[node.name] = _parse_ace(node, category, node_type, order)
                parsed.ignored.extend(parsed.ace[node.name].ignored)
                for out_idx, sock in enumerate(node.outputs):
                    if sock.is_linked:
                        parsed.flows[("ace", node.name, out_idx)] = _targets(sock)
        elif is_user_group_node(node):
            entry = GroupEntry(name=str(node.name), tree_name=str(node.node_tree.name))
            linked = []
            for sock in node.inputs:
                if sock.is_linked:
                    linked.append(str(sock.identifier))
                    ident = _feed(sock)
                    if ident is not None:
                        entry.bound_inputs[str(sock.identifier)] = ident
                elif hasattr(sock, "default_value"):
                    entry.defaults[str(sock.identifier)] = plain_value(sock.default_value)
            entry.linked_inputs = tuple(linked)
            parsed.groups[node.name] = entry
            for sock in node.outputs:
                if sock.is_linked:
                    parsed.flows[("grp", node.name, str(sock.identifier))] = _targets(sock)
        elif node_type_name == "GROUP_INPUT":
            for sock in node.outputs:
                if sock.is_linked:
                    port = (GROUP_INPUT, str(sock.identifier))
                    parsed.flows.setdefault(port, []).extend(_targets(sock))
        elif str(getattr(node, "bl_idname", "") or "") == "GeometryNodeViewer":
            heard = next((s for s in node.inputs if s.is_linked and str(getattr(s, "type", "")) == "BUNDLE"), None)
            if heard is not None:
                parsed.viewers[node.name] = str(heard.identifier)
    # What ACE ignores reaches no node: the reading never follows it.
    dropped = {("ace", item.node_name, item.input_index) for item in parsed.ignored}
    if dropped:
        for port, targets in parsed.flows.items():
            parsed.flows[port] = [t for t in targets if tuple(t) not in dropped]
    return parsed
