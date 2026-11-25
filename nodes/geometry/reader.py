"""Reads a Geometry Nodes frontend into an abstract graph."""

import hashlib
from collections import deque
from typing import Optional, Any
from ...vars import ACE_NODES, resolve_node_mode
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
from ..split_freqs import capture_driven_splits
from ..frontend import FRONTEND_KIND_GEOMETRY, FRONTEND_KIND_SHADER, _tree_pointer
from .factory import ACEDummyNodeFactory
from .identity import NodeIdentity


class ACEDummyTreeReader(NodeIdentity):
    """Parse an active Blender GeometryNodeTree into an ACEAbstractTree."""

    # Group name -> (category, type), rebuilt when the node registry changes.
    _GROUP_NAME_LOOKUP: dict[str, tuple[str, str]] = {}
    _ACE_NODES_LOOKUP_HASH: str = ""
    FACTORY_CLS = ACEDummyNodeFactory

    def __init__(
        self,
        blender_node_tree,
        frontend_uid: str = "",
        frontend_kind: str = FRONTEND_KIND_GEOMETRY,
        reserved_node_ids: set[str] | None = None,
        consumer_keys: tuple[str, ...] = (),
    ):
        self.bl_tree = blender_node_tree
        self.abstract_tree = ACEAbstractTree()
        self.frontend_uid = str(frontend_uid or "")
        self.frontend_kind = str(frontend_kind or FRONTEND_KIND_GEOMETRY)
        self.consumer_keys = tuple(str(v) for v in (consumer_keys or ()) if str(v))
        self._node_order_by_ptr = {
            self._node_pointer(node): idx
            for idx, node in enumerate(getattr(blender_node_tree, "nodes", []) or [])
        }
        # Track which source roots have already visited a Blender node
        self.visited_blender_nodes: dict[Any, set[str]] = {}
        # Track ACE ID ownership inside this read pass so duplicated Blender
        # nodes (Shift+D) cannot collapse to one abstract/task node.
        self._ace_id_owner: dict[str, int] = {}
        self._node_id_by_ptr: dict[int, str] = {}
        self._reserved_node_ids = reserved_node_ids if reserved_node_ids is not None else set()
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
        """Execute the full parse pass and return the populated ACEAbstractTree."""
        if not self.bl_tree: return self.abstract_tree
        sources = self._find_source_nodes()
        if sources:
            self._traverse_from_sources(sources)
        self._validate_cross_source_links()
        self._emit_consumer_intents()
        return self.abstract_tree

    def _validate_cross_source_links(self):
        """Warn when a non-AudioMath node receives inputs from distinct root sources."""
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
            # Cross-source link detected.
            if getattr(tgt_node, "type", "") == "audio_math":
                continue
            utils.log(
                f"ACE: Cross-source link '{src_node.name}' → '{tgt_node.name}': "
                f"only AudioMath can receive mixed sources. "
                f"Node '{tgt_node.name}' will produce undefined output.",
                "WARNING",
            )

    @staticmethod
    def _is_writeable_output_node(node: Optional[ACEAbstractNode]) -> bool:
        if node is None:
            return False
        is_stream = str(getattr(node, "category", "")) == "stream"
        is_specs = (
            str(getattr(node, "category", "")) == "utilities"
            and str(getattr(node, "type", "")) == "src_specs"
        )
        is_rolling_avg = (
            str(getattr(node, "category", "")) == "utilities"
            and str(getattr(node, "type", "")) == "util_rolling_avg"
        )
        is_stream_accumulator = (
            str(getattr(node, "category", "")) == "utilities"
            and str(getattr(node, "type", "")) == "stream_accumulator"
        )
        return bool(is_stream or is_specs or is_rolling_avg or is_stream_accumulator)

    def _emit_consumer_intents(self):
        sink_kind = (
            "OBJECT_PROPERTY"
            if self.frontend_kind == FRONTEND_KIND_SHADER
            else "PROXY_ATTRIBUTE"
        )
        for node in self.abstract_tree.nodes.values():
            if not self._is_writeable_output_node(node):
                continue
            root_tokens = sorted({str(v) for v in (getattr(node, "root_sources", set()) or set()) if str(v)})
            if not root_tokens:
                root_tokens = [""]
            params = dict(getattr(node, "parameters", {}) or {})
            for sock in getattr(node, "outputs", []) or []:
                data_type = str(getattr(sock, "data_type", "") or "")
                if data_type in ("NodeSocketBundle", "BUNDLE", "BUNDLE_SOCKET"):
                    continue
                explicit_name = str(params.get(f"attr_id_out_{int(getattr(sock, 'socket_idx', 0) or 0)}", "") or "")
                for source_token in root_tokens:
                    self.abstract_tree.add_consumer_intent(
                        ACEConsumerIntent(
                            frontend_uid=self.frontend_uid,
                            node_id=str(getattr(node, "id", "") or ""),
                            socket_idx=int(getattr(sock, "socket_idx", 0) or 0),
                            sink_kind=sink_kind,
                            source_token=str(source_token or ""),
                            consumer_keys=self.consumer_keys if sink_kind == "OBJECT_PROPERTY" else (),
                            output_name=str(getattr(sock, "name", "") or ""),
                            explicit_name=explicit_name,
                        )
                    )

    def _find_source_nodes(self):
        """Return all managed source placeholder nodes in the Blender tree."""
        sources = []
        for node in self.bl_tree.nodes:
            root_id = self._resolve_root_id(node)
            ace_id = self._ensure_ace_id(node, root_id=root_id)
            if not ace_id: continue
            
            parsed = ACEIdManager.parse_node_id(ace_id)
            if parsed and parsed['category'] == 'source_mngmt' and parsed['type'] == 'src':
                # Ignore ad-hoc source nodes; only managed placeholders carry a source binding.
                source_id = ""
                try:
                    source_id = str(node.get("ace_source_id", "") or "")
                except Exception as exc:
                    utils.note_swallowed("reader.ACEDummyTreeReader._find_source_nodes", exc)
                    source_id = ""
                if not source_id or not ACEIdManager.parse_node_id(source_id):
                    continue
                sources.append(node)
        return sources

    def _traverse_from_sources(self, start_nodes):
        queue = deque()
        for src_node in start_nodes:
            root_id = self._resolve_root_id(src_node)
            queue.append((src_node, root_id, "", 0, None))

        visited_states: set[tuple[int, str, str, int, str]] = set()
        link_keys: set[tuple[str, str, int, int]] = {
            (l.source_node_id, l.target_node_id, int(l.source_socket_idx), int(l.target_socket_idx))
            for l in self.abstract_tree.links
        }
        output_link_keys: set[tuple[str, int, int]] = {
            (
                str(l.source_node_id),
                int(l.source_socket_idx),
                int(l.target_socket_idx),
            )
            for l in getattr(self.abstract_tree, "group_output_links", []) or []
        }

        while queue:
            current_bl_node, root_id, prev_ace_id, prev_output_idx, incoming_socket = queue.popleft()

            state_key = (
                id(current_bl_node),
                root_id or "",
                prev_ace_id or "",
                int(prev_output_idx) if prev_output_idx is not None else 0,
                self._socket_state_key(incoming_socket),
            )
            if state_key in visited_states:
                continue
            visited_states.add(state_key)

            abstract_node = self._parse_node(current_bl_node, root_id)
            current_ace_id = abstract_node.id if abstract_node else ""

            if abstract_node:
                # Merge roots if node already exists
                if abstract_node.id in self.abstract_tree.nodes:
                    self.abstract_tree.nodes[abstract_node.id].root_sources.update(abstract_node.root_sources)
                else:
                    self.abstract_tree.add_node(abstract_node)

            # Capture ACE outputs connected to Group Output so the runtime can
            # build default preview sinks without requiring manual nodes.
            if (
                getattr(current_bl_node, "type", "") == "GROUP_OUTPUT"
                and prev_ace_id
                and incoming_socket is not None
            ):
                target_socket_idx = self._resolve_target_socket_idx(current_bl_node, incoming_socket)
                if target_socket_idx is None:
                    target_socket_idx = 0
                link_key = (str(prev_ace_id), int(prev_output_idx), int(target_socket_idx))
                if link_key not in output_link_keys:
                    output_link_keys.add(link_key)
                    target_socket_name = str(getattr(incoming_socket, "name", "") or "")
                    self.abstract_tree.add_group_output_link(
                        ACEAbstractOutputLink(
                            source_node_id=str(prev_ace_id),
                            source_socket_idx=int(prev_output_idx),
                            target_socket_idx=int(target_socket_idx),
                            target_socket_name=target_socket_name,
                            frontend_uid=self.frontend_uid,
                            frontend_kind=self.frontend_kind,
                        )
                    )

            if (
                str(getattr(current_bl_node, "bl_idname", "")) == "GeometryNodeViewer"
                and prev_ace_id
                and incoming_socket is not None
            ):
                self._add_viewer_preview(current_bl_node, root_id, str(prev_ace_id), int(prev_output_idx), incoming_socket)

            # If we reached an ACE node after traversing non-ACE nodes, bridge the link.
            if current_ace_id and prev_ace_id and incoming_socket is not None and current_ace_id != prev_ace_id:
                target_socket_idx = self._resolve_target_socket_idx(current_bl_node, incoming_socket)
                if target_socket_idx is None:
                    try:
                        if len(getattr(current_bl_node, "inputs", [])) > 0:
                            target_socket_idx = 0
                    except Exception as exc:
                        utils.note_swallowed("reader.ACEDummyTreeReader._traverse_from_sources", exc)
                        target_socket_idx = None

                if target_socket_idx is not None:
                    link_key = (prev_ace_id, current_ace_id, int(prev_output_idx), int(target_socket_idx))
                    if link_key not in link_keys:
                        link_keys.add(link_key)
                        self.abstract_tree.links.append(ACEAbstractLink(
                            source_node_id=prev_ace_id,
                            target_node_id=current_ace_id,
                            source_socket_idx=int(prev_output_idx),
                            target_socket_idx=int(target_socket_idx),
                        ))

            carry_ace_id = current_ace_id or prev_ace_id
            for output_idx, socket in enumerate(getattr(current_bl_node, "outputs", [])):
                if not getattr(socket, "is_linked", False):
                    continue
                next_output_idx = output_idx if current_ace_id else prev_output_idx
                for link in getattr(socket, "links", []):
                    target_node = getattr(link, "to_node", None)
                    if not target_node:
                        continue
                    queue.append((target_node, root_id, carry_ace_id, next_output_idx, getattr(link, "to_socket", None)))

    def _add_viewer_preview(self, viewer, root_id: str, source_id: str, source_idx: int, to_socket) -> None:
        """A Viewer node fed by an audio output behaves as an Audio Preview node.

        With several audio inputs on one Viewer, the first one is heard.
        """
        heard = next((s for s in viewer.inputs if s.is_linked and str(getattr(s, "type", "")) == "BUNDLE"), None)
        if heard is None or heard.identifier != getattr(to_socket, "identifier", None):
            return
        source = self.abstract_tree.nodes.get(source_id)
        outputs = getattr(source, "outputs", []) or []
        if not (0 <= source_idx < len(outputs)) or "Bundle" not in str(outputs[source_idx].data_type):
            return

        digest = hashlib.blake2b(f"{self.frontend_uid}:{viewer.name}".encode("utf-8"), digest_size=4).hexdigest()
        sink_id = f"{ACEIdManager.PREFIX}utilities_preview_sink_viewer{digest}"
        roots = {root_id} if root_id else set()
        existing = self.abstract_tree.nodes.get(sink_id)
        if existing is not None:
            existing.root_sources.update(roots)
        else:
            self.abstract_tree.add_node(ACEAbstractNode(
                id=sink_id,
                name=viewer.name,
                type="preview_sink",
                category="utilities",
                inputs=[ACEAbstractSocket(
                    id=f"{sink_id}_in_0", name="Audio", socket_idx=0, is_output=False,
                    data_type="NodeSocketBundle",
                )],
                outputs=[],
                parameters={
                    "preview_enabled": True,
                    "frontend_node_index": int(self._node_order_by_ptr.get(self._node_pointer(viewer), 0)),
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

    @staticmethod
    def _socket_state_key(sock) -> str:
        if sock is None:
            return "NONE"
        identifier = str(getattr(sock, "identifier", "") or "")
        if identifier:
            return f"id:{identifier}"
        name = str(getattr(sock, "name", "") or "")
        if name:
            return f"name:{name}"
        try:
            return f"idx:{int(getattr(sock, 'index'))}"
        except Exception as exc:
            utils.note_swallowed("reader.ACEDummyTreeReader._socket_state_key", exc)
            return f"obj:{id(sock)}"

    @staticmethod
    def _resolve_target_socket_idx(target_node, to_socket) -> Optional[int]:
        # Not every socket type exposes .index (NodeSocketBundle does not).
        target_socket_idx = getattr(to_socket, "index", None)
        if target_socket_idx is not None:
            return target_socket_idx

        try:
            target_socket_idx = next((i for i, s in enumerate(target_node.inputs) if s == to_socket), None)
        except Exception as exc:
            utils.note_swallowed("reader.ACEDummyTreeReader._resolve_target_socket_idx", exc)
            target_socket_idx = None
        if target_socket_idx is not None:
            return target_socket_idx

        sock_name = ""
        try:
            sock_name = getattr(to_socket, "name", "") or ""
        except Exception as exc:
            utils.note_swallowed("reader.ACEDummyTreeReader._resolve_target_socket_idx#2", exc)
            sock_name = ""
        if sock_name:
            try:
                target_socket_idx = next(
                    (i for i, s in enumerate(target_node.inputs) if getattr(s, "name", "") == sock_name),
                    None,
                )
            except Exception as exc:
                utils.note_swallowed("reader.ACEDummyTreeReader._resolve_target_socket_idx#3", exc)
                target_socket_idx = None
        if target_socket_idx is not None:
            return target_socket_idx

        sock_identifier = ""
        try:
            sock_identifier = getattr(to_socket, "identifier", "") or ""
        except Exception as exc:
            utils.note_swallowed("reader.ACEDummyTreeReader._resolve_target_socket_idx#4", exc)
            sock_identifier = ""
        if sock_identifier:
            try:
                target_socket_idx = next(
                    (i for i, s in enumerate(target_node.inputs) if getattr(s, "identifier", "") == sock_identifier),
                    None,
                )
            except Exception as exc:
                utils.note_swallowed("reader.ACEDummyTreeReader._resolve_target_socket_idx#5", exc)
                target_socket_idx = None
        return target_socket_idx

    def _resolve_root_id(self, bl_node) -> str:
        """Try to derive a stable source identifier from the source node: prefer explicit ace_source_id, then the proxy object name, else the node ace_id."""
        root_id = ""
        try:
            root_id = bl_node.get("ace_source_id", "")
        except Exception as exc:
            utils.note_swallowed("reader.ACEDummyTreeReader._resolve_root_id", exc)
            root_id = ""
        if not root_id:
            try:
                proxy_sock = next((s for s in bl_node.inputs if getattr(s, "name", "") == "Proxy Object"), None)
                if proxy_sock and getattr(proxy_sock, "default_value", None):
                    proxy_value = proxy_sock.default_value
                    if isinstance(proxy_value, str):
                        root_id = str(proxy_value or "").strip()
                    else:
                        root_id = getattr(proxy_value, "name", "") or ""
            except Exception as exc:
                utils.note_swallowed("reader.ACEDummyTreeReader._resolve_root_id#2", exc)
                root_id = ""
        return root_id

    @staticmethod
    def _attribute_socket_to_output_index(sock_name: str) -> Optional[int]:
        name = str(sock_name or "").strip()
        if name == "Attribute ID":
            return 0
        if name.startswith("Attribute ID "):
            suffix = name[len("Attribute ID "):].strip()
            if suffix.isdigit():
                idx = int(suffix) - 1
                if idx >= 0:
                    return int(idx)
        mapping = {
            "Frames per Beat Attribute ID": 0,
            "Beat Phase Attribute ID": 1,
        }
        return mapping.get(name)

    def _parse_node(self, bl_node, root_id: str) -> Optional[ACEAbstractNode]:
        ace_id = self._ensure_ace_id(bl_node, root_id=root_id)
        if not ace_id:
            return None
        parsed = ACEIdManager.parse_node_id(ace_id)
        if not parsed: return None
        
        cat, typ = parsed['category'], parsed['type']
        
        # Splitter Handling
        if cat == 'delta' and typ == 'frequency_splitter':
            input_count = len(bl_node.inputs)
            output_count = len(bl_node.outputs)
            input_defs = [('NodeSocketBundle', 'Audio In')]
            for k in range(input_count - 1):
                input_defs.append(('NodeSocketFloat', f"Split Freq {k+1}"))
            output_defs = [('NodeSocketBundle', 'Band')] * output_count
        elif cat == 'utilities' and typ == 'audio_math':
            input_count = len(bl_node.inputs)
            output_count = len(bl_node.outputs)
            input_defs = [('NodeSocketBundle', f"Audio {k+1}") for k in range(input_count)]
            output_defs = [('NodeSocketBundle', 'Audio')] * max(output_count, 1)
        else:
            try:
                node_def = ACE_NODES[cat]['nodes'][typ]
                input_defs = node_def.get('inputs', [])
                output_defs = node_def.get('outputs', [])
            except KeyError:
                return None

        # Build Inputs
        inputs = []
        params = {}
        
        for i, sock in enumerate(bl_node.inputs):
            # Safe type check
            bl_type_str = "NodeSocketFloat"
            if i < len(input_defs):
                bl_type_str = input_defs[i][0]
            elif "Attribute ID" in str(getattr(sock, "name", "") or ""):
                bl_type_str = "NodeSocketString"
            
            inputs.append(ACEAbstractSocket(
                id=f"{ace_id}_in_{i}",
                name=sock.name,
                socket_idx=i,
                is_output=False,
                data_type=bl_type_str,
                default_value=getattr(sock, 'default_value', None)
            ))
            
            # Param Capture
            if not sock.is_linked:
                val = getattr(sock, 'default_value', None)
                if hasattr(val, "to_tuple"): val = val.to_tuple()
                elif hasattr(val, "to_list"): val = val.to_list()
                if val is not None:
                     params[f"in_{i}"] = val

            attr_out_idx = self._attribute_socket_to_output_index(sock.name)
            if attr_out_idx is not None:
                val = getattr(sock, 'default_value', None)
                if isinstance(val, str):
                    attr_name = val.strip()
                    if attr_name:
                        params[f"attr_id_out_{attr_out_idx}"] = attr_name
                    is_custom = bool(attr_name)
                    params[f"attr_id_custom_out_{attr_out_idx}"] = bool(is_custom)

        # Custom parameters (e.g., Audio Math operation)
        op_value = bl_node.get("ace_op")
        if isinstance(op_value, str):
            params["op"] = op_value
        mode = resolve_node_mode(cat, typ, bl_node.get("ace_mode"))
        if mode:
            params["mode"] = mode
        preview_enabled = bl_node.get("ace_preview_enabled", True)
        params["preview_enabled"] = bool(preview_enabled)
        params["frontend_node_index"] = int(
            self._node_order_by_ptr.get(self._node_pointer(bl_node), 0)
        )

        if cat == 'delta' and typ == 'frequency_splitter':
            with utils.logged_errors("reader.ACEDummyTreeReader._parse_node#driven"):
                capture_driven_splits(bl_node, params)

        # Build Outputs
        outputs = []
        for i, sock in enumerate(bl_node.outputs):
            bl_type_str = "NodeSocketFloat"
            if i < len(output_defs):
                bl_type_str = output_defs[i][0]
                
            outputs.append(ACEAbstractSocket(
                id=f"{ace_id}_out_{i}",
                name=sock.name,
                socket_idx=i,
                is_output=True,
                data_type=bl_type_str
            ))
            
        return ACEAbstractNode(
            id=ace_id,
            name=bl_node.name,
            type=typ,
            category=cat,
            inputs=inputs,
            outputs=outputs,
            parameters=params,
            root_sources={root_id} if root_id else set(),
            frontend_uid=self.frontend_uid,
            frontend_kind=self.frontend_kind,
            source_binding_id=(
                str(getattr(bl_node, "get", lambda *_: "")("ace_source_id", "") or "")
                if cat == "source_mngmt" and typ == "src"
                else ""
            ),
        )
