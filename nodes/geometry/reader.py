"""Reads one use of a node tree into an abstract graph: an instance walk over cached definitions.

The walk starts at the owner's tree (a modifier's, a material's, the World's, a
light's) and enters every user group on the way, so ACE nodes nested at any
depth are read. Along the path it binds what each group receives: the
modifier's input values at the root, a group node's own input values below,
or a value bound one level up. Each use of an ACE node gets an id of its own,
from the path: the same group used twice gives two sets of nodes.

The reader writes nothing to the trees.
"""

import hashlib
from collections import deque
from typing import Optional
from ...core.abstract_tree import (
    ACEAbstractTree,
    ACEAbstractNode,
    ACEAbstractSocket,
    ACEAbstractLink,
    ACEAbstractOutputLink,
    ACEFrontendRef,
    ACEConsumerIntent,
)
from ...core.id_management import ACEIdManager
from ... import utils
from ..definitions import GROUP_INPUT, definition
from ..frontend import FRONTEND_KIND_GEOMETRY, FRONTEND_KIND_SHADER, _tree_pointer
from ..identity import SOURCE_ID_KEY, instance_node_id, tree_key
from ...vars import PREVIEW_KIND, SOURCE_KIND, SPLITTER_KIND, writes_values

_MAX_DEPTH = 32
UNBOUND = object()


def _bound_value(value):
    """A bound value as a node input takes it: objects by reference, the rest as is."""
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "object":
        import bpy  # type: ignore

        return bpy.data.objects.get(value[1])
    return value


class _Scope:
    """One tree on the walk: the root, or a group entered from its parent scope."""

    def __init__(self, tree, chain: tuple, chain_nodes: tuple, bindings: dict, parent=None, group_name: str = ""):
        self.tree = tree
        self.definition = definition(tree)
        self.chain = chain  # names of the group nodes entered, from the root
        self.chain_nodes = chain_nodes  # the group nodes themselves, for the driven split frequencies
        self.bindings = bindings  # Group Input socket identifier -> value, or UNBOUND
        self.parent = parent
        self.group_name = group_name
        self.children: dict[str, "_Scope"] = {}


class ACEDummyTreeReader:
    """Read one use of a tree into an ACEAbstractTree. Reads only: nothing is written to the tree."""

    def __init__(
        self,
        blender_node_tree,
        frontend_uid: str = "",
        frontend_kind: str = FRONTEND_KIND_GEOMETRY,
        consumer_keys: tuple[str, ...] = (),
        bindings: Optional[dict] = None,
        root_key: str = "",
        modifier=None,
    ):
        self.bl_tree = blender_node_tree
        self.abstract_tree = ACEAbstractTree()
        self.frontend_uid = str(frontend_uid or "")
        self.frontend_kind = str(frontend_kind or FRONTEND_KIND_GEOMETRY)
        self.consumer_keys = tuple(str(v) for v in (consumer_keys or ()) if str(v))
        self.root_bindings = dict(bindings or {})
        self.root_key = str(root_key or (tree_key(blender_node_tree) if blender_node_tree is not None else ""))
        self.modifier = modifier
        self.visited_trees: list = []  # every tree of the walk, root first
        # (node id, input index) of inputs linked to a value the walk cannot bind (a computed field).
        self.unbound_inputs: list[tuple[str, int]] = []
        if self.frontend_uid:
            self.abstract_tree.add_frontend(
                ACEFrontendRef(
                    uid=self.frontend_uid,
                    kind=self.frontend_kind,
                    tree_name=str(getattr(blender_node_tree, "name", "") or ""),
                    tree_bl_idname=str(getattr(blender_node_tree, "bl_idname", "") or ""),
                    tree_ptr=_tree_pointer(blender_node_tree),
                )
            )

    def build(self) -> ACEAbstractTree:
        """Walk the tree and the groups it uses; return the populated ACEAbstractTree."""
        if not self.bl_tree:
            return self.abstract_tree
        root = _Scope(self.bl_tree, (), (), self.root_bindings)
        self._scopes: list[_Scope] = []
        self._open_scopes(root, 0)
        self._link_keys: set = set()
        self._output_link_keys: set = set()
        self._traverse(root)
        self._validate_cross_source_links()
        self._emit_consumer_intents()
        return self.abstract_tree

    # Scopes

    def _open_scopes(self, scope: _Scope, depth: int) -> None:
        self._scopes.append(scope)
        if scope.tree not in self.visited_trees:
            self.visited_trees.append(scope.tree)
        if depth >= _MAX_DEPTH:
            return
        for name, entry in scope.definition.groups.items():
            node = scope.tree.nodes.get(name) if hasattr(scope.tree.nodes, "get") else None
            child_tree = getattr(node, "node_tree", None)
            if child_tree is None:
                continue
            bindings = {ident: value for ident, value in entry.defaults.items()}
            for ident in entry.linked_inputs:
                outer = entry.bound_inputs.get(ident)
                bindings[ident] = scope.bindings.get(outer, UNBOUND) if outer is not None else UNBOUND
            child = _Scope(child_tree, scope.chain + (name,), scope.chain_nodes + (node,), bindings, scope, name)
            scope.children[name] = child
            self._open_scopes(child, depth + 1)

    def _source_of(self, scope: _Scope, entry) -> str:
        if entry.source_input:
            proxy = _bound_value(scope.bindings.get(entry.source_input, UNBOUND))
            if proxy is UNBOUND or proxy is None or isinstance(proxy, str):
                return ""
            with utils.logged_errors("reader.ACEDummyTreeReader._source_of"):
                return str(proxy.get(SOURCE_ID_KEY, "") or "")
            return ""
        # Read now, not from the cached definition: the id is on the proxy object, which can be
        # marked (or another source's) without any change to the tree the cache follows.
        node = scope.tree.nodes.get(entry.name) if hasattr(scope.tree.nodes, "get") else None
        if node is None:
            return entry.source_id
        from ..source_nodes import node_source_id

        with utils.logged_errors("reader.ACEDummyTreeReader._source_of#own"):
            return node_source_id(node)
        return entry.source_id

    def _node_id(self, scope: _Scope, entry) -> str:
        return instance_node_id(entry.category, entry.node_type, self.root_key, scope.chain, entry.name)

    # Nodes

    def _abstract_node(self, scope: _Scope, entry, root_id: str) -> ACEAbstractNode:
        node_id = self._node_id(scope, entry)
        existing = self.abstract_tree.nodes.get(node_id)
        if existing is not None:
            if root_id:
                existing.root_sources.add(root_id)
            return existing
        params = dict(entry.params)
        is_splitter = (entry.category, entry.node_type) == SPLITTER_KIND
        for idx in entry.linked_inputs:
            if is_splitter and idx > 0:
                continue  # a linked split frequency keeps its per-frame driven value
            ident = entry.bound_inputs.get(idx)
            value = scope.bindings.get(ident, UNBOUND) if ident is not None else UNBOUND
            if value is UNBOUND or (isinstance(value, tuple) and value[:1] == ("object",)):
                if ident is not None:
                    self.unbound_inputs.append((node_id, idx))
                continue
            params[f"in_{idx}"] = value
        if is_splitter and any(idx > 0 for idx in entry.linked_inputs):
            from ..split_freqs import capture_driven_splits

            node = scope.tree.nodes.get(entry.name)
            if node is not None:
                with utils.logged_errors("reader.ACEDummyTreeReader._abstract_node#driven"):
                    capture_driven_splits(node, params, chain=scope.chain_nodes, modifier=self.modifier,
                                          root_key=self.root_key)
        inputs = [
            ACEAbstractSocket(id=f"{node_id}_in_{i}", name=name, socket_idx=i, is_output=False,
                              data_type=data_type, default_value=default)
            for i, (name, data_type, default) in enumerate(entry.inputs)
        ]
        outputs = [
            ACEAbstractSocket(id=f"{node_id}_out_{i}", name=name, socket_idx=i, is_output=True, data_type=data_type)
            for i, (name, data_type) in enumerate(entry.outputs)
        ]
        node = ACEAbstractNode(
            id=node_id,
            name=entry.name,
            type=entry.node_type,
            category=entry.category,
            inputs=inputs,
            outputs=outputs,
            parameters=params,
            root_sources={root_id} if root_id else set(),
            frontend_uid=self.frontend_uid,
            frontend_kind=self.frontend_kind,
            source_binding_id=root_id if (entry.category, entry.node_type) == SOURCE_KIND else "",
            definition_ptr=_tree_pointer(scope.tree),
            definition_tree=str(getattr(scope.tree, "name_full", "") or getattr(scope.tree, "name", "") or ""),
            use_chain=tuple(scope.chain),
        )
        self.abstract_tree.add_node(node)
        return node

    # Traversal

    def _traverse(self, root: _Scope) -> None:
        queue: deque = deque()
        for scope in self._scopes:
            for entry in scope.definition.ace.values():
                if (entry.category, entry.node_type) != SOURCE_KIND:
                    continue
                root_id = self._source_of(scope, entry)
                parsed = ACEIdManager.parse_node_id(root_id)
                if not (parsed and (parsed["category"], parsed["type"]) == SOURCE_KIND):
                    continue
                node = self._abstract_node(scope, entry, root_id)
                for out_idx in range(len(entry.outputs)):
                    queue.append((scope, ("ace", entry.name, out_idx), root_id, node.id, out_idx))

        visited: set = set()
        while queue:
            scope, port, root_id, carry, carry_idx = queue.popleft()
            state = (scope.chain, port, root_id, carry, carry_idx)
            if state in visited:
                continue
            visited.add(state)
            for target in scope.definition.flows.get(port, []):
                kind = target[0]
                if kind == "ace":
                    entry = scope.definition.ace.get(target[1])
                    if entry is None:
                        continue
                    node = self._abstract_node(scope, entry, root_id)
                    if node.id != carry:
                        key = (carry, node.id, int(carry_idx), int(target[2]))
                        if key not in self._link_keys:
                            self._link_keys.add(key)
                            self.abstract_tree.links.append(ACEAbstractLink(
                                source_node_id=carry, target_node_id=node.id,
                                source_socket_idx=int(carry_idx), target_socket_idx=int(target[2]),
                            ))
                    for out_idx in range(len(entry.outputs)):
                        queue.append((scope, ("ace", entry.name, out_idx), root_id, node.id, out_idx))
                elif kind == "grp":
                    child = scope.children.get(target[1])
                    if child is not None:
                        queue.append((child, (GROUP_INPUT, target[2]), root_id, carry, carry_idx))
                elif kind == "go":
                    if scope.parent is not None:
                        queue.append((scope.parent, ("grp", scope.group_name, target[1]), root_id, carry, carry_idx))
                    else:
                        self._add_group_output_link(carry, carry_idx, target[2], target[3])
                elif kind == "viewer":
                    if scope.definition.viewers.get(target[1]) == target[2]:
                        self._add_viewer_preview(scope, target[1], root_id, carry, carry_idx)

    def _add_group_output_link(self, source_id: str, source_idx: int, target_idx: int, target_name: str) -> None:
        """An audio output reaching the root tree's Group Output: the runtime builds default preview sinks."""
        key = (str(source_id), int(source_idx), int(target_idx))
        if key in self._output_link_keys:
            return
        self._output_link_keys.add(key)
        self.abstract_tree.add_group_output_link(ACEAbstractOutputLink(
            source_node_id=str(source_id),
            source_socket_idx=int(source_idx),
            target_socket_idx=int(target_idx),
            target_socket_name=str(target_name or ""),
            frontend_uid=self.frontend_uid,
            frontend_kind=self.frontend_kind,
        ))

    def _add_viewer_preview(self, scope: _Scope, viewer_name: str, root_id: str, source_id: str, source_idx: int) -> None:
        """A Viewer node fed by an audio output behaves as an Audio Preview node.

        With several audio inputs on one Viewer, the first one is heard.
        """
        source = self.abstract_tree.nodes.get(source_id)
        outputs = getattr(source, "outputs", []) or []
        if not (0 <= source_idx < len(outputs)) or "Bundle" not in str(outputs[source_idx].data_type):
            return
        path = "/".join([*scope.chain, viewer_name])
        digest = hashlib.blake2b(f"{self.frontend_uid}:{path}".encode("utf-8"), digest_size=4).hexdigest()
        sink_id = f"{ACEIdManager.PREFIX}{PREVIEW_KIND[0]}_{PREVIEW_KIND[1]}_viewer{digest}"
        roots = {root_id} if root_id else set()
        existing = self.abstract_tree.nodes.get(sink_id)
        if existing is not None:
            existing.root_sources.update(roots)
        else:
            order = next((i for i, n in enumerate(scope.tree.nodes) if n.name == viewer_name), 0)
            self.abstract_tree.add_node(ACEAbstractNode(
                id=sink_id,
                name=viewer_name,
                type=PREVIEW_KIND[1],
                category=PREVIEW_KIND[0],
                inputs=[ACEAbstractSocket(
                    id=f"{sink_id}_in_0", name="Audio", socket_idx=0, is_output=False,
                    data_type="NodeSocketBundle",
                )],
                outputs=[],
                parameters={
                    "preview_enabled": True,
                    "frontend_node_index": int(order),
                    "viewer": True,  # not an operator: left out of the Active Operators list
                },
                root_sources=roots,
                frontend_uid=self.frontend_uid,
                frontend_kind=self.frontend_kind,
            ))
        if not any(l.source_node_id == source_id and l.target_node_id == sink_id for l in self.abstract_tree.links):
            self.abstract_tree.links.append(ACEAbstractLink(
                source_node_id=source_id,
                target_node_id=sink_id,
                source_socket_idx=source_idx,
                target_socket_idx=0,
            ))

    # Checks and sinks

    def _validate_cross_source_links(self):
        """Warn when a node receives inputs from distinct root sources."""
        for link in self.abstract_tree.links:
            src_node = self.abstract_tree.nodes.get(link.source_node_id)
            tgt_node = self.abstract_tree.nodes.get(link.target_node_id)
            if src_node is None or tgt_node is None:
                continue
            src_roots = getattr(src_node, "root_sources", None) or set()
            tgt_roots = getattr(tgt_node, "root_sources", None) or set()
            if not src_roots or not tgt_roots:
                continue
            if src_roots.issubset(tgt_roots):
                continue
            utils.log(
                f"ACE: Cross-source link '{src_node.name}' → '{tgt_node.name}': "
                f"a node takes the audio of one source. "
                f"Node '{tgt_node.name}' will produce undefined output.",
                "WARNING",
            )

    @staticmethod
    def _is_writeable_output_node(node: Optional[ACEAbstractNode]) -> bool:
        if node is None:
            return False
        return writes_values(str(getattr(node, "category", "")), str(getattr(node, "type", "")))

    def _emit_consumer_intents(self):
        sink_kind = "SCENE_PROPERTY" if self.frontend_kind == FRONTEND_KIND_SHADER else "PROXY_ATTRIBUTE"
        for node in self.abstract_tree.nodes.values():
            if not self._is_writeable_output_node(node):
                continue
            root_tokens = sorted({str(v) for v in (getattr(node, "root_sources", set()) or set()) if str(v)}) or [""]
            params = dict(getattr(node, "parameters", {}) or {})
            for sock in getattr(node, "outputs", []) or []:
                if str(getattr(sock, "data_type", "") or "") in ("NodeSocketBundle", "BUNDLE", "BUNDLE_SOCKET"):
                    continue
                socket_idx = int(getattr(sock, "socket_idx", 0) or 0)
                for source_token in root_tokens:
                    self.abstract_tree.add_consumer_intent(ACEConsumerIntent(
                        frontend_uid=self.frontend_uid,
                        node_id=str(getattr(node, "id", "") or ""),
                        socket_idx=socket_idx,
                        sink_kind=sink_kind,
                        source_token=str(source_token or ""),
                        consumer_keys=self.consumer_keys if sink_kind == "SCENE_PROPERTY" else (),
                        output_name=str(getattr(sock, "name", "") or ""),
                        explicit_name=str(params.get(f"attr_id_out_{socket_idx}", "") or ""),
                    ))
