"""Stable node identity: node kinds, topology signatures and ace_id assignment."""

import json
from ...vars import ACE_NODES
from ...core.id_management import ACEIdManager
from ... import utils


class NodeIdentity:
    @classmethod
    def _ensure_group_name_lookup(cls):
        """Populate (or invalidate and rebuild) _GROUP_NAME_LOOKUP from ACE_NODES.

        Uses the same blake2b hash as ACEDummyNodeFactory to detect registry
        changes.  The lookup is rebuilt on first use and whenever the registry
        hash differs from the cached value.
        """
        current_hash = cls.FACTORY_CLS._ace_nodes_hash()
        if cls._GROUP_NAME_LOOKUP and current_hash and current_hash == cls._ACE_NODES_LOOKUP_HASH:
            return
        cls._GROUP_NAME_LOOKUP.clear()
        for cat_key, category in ACE_NODES.items():
            for type_key in category.get("nodes", {}).keys():
                canonical = cls.FACTORY_CLS.get_group_name(cat_key, type_key)
                cls._GROUP_NAME_LOOKUP[canonical] = (cat_key, type_key)
                norm = canonical.lstrip(".")
                if norm != canonical:
                    cls._GROUP_NAME_LOOKUP[norm] = (cat_key, type_key)
        if current_hash:
            cls._ACE_NODES_LOOKUP_HASH = current_hash

    @staticmethod
    def _node_pointer(node) -> int:
        """Return a stable integer identity for a Blender node.

        Uses as_pointer() when available (Blender C-level pointer) and falls
        back to Python id() so the result is always an int regardless of node
        type.
        """
        try:
            return int(node.as_pointer())
        except Exception as exc:
            utils.note_swallowed("dummy_tree.ACEDummyTreeReader._node_pointer", exc)
            return id(node)

    def _reserve_unique_ace_id(
        self,
        bl_node,
        ace_id: str,
        category: str,
        node_type: str,
        deterministic_seed: str = "",
    ) -> str:
        """Guarantee that *ace_id* is used by at most one Blender node per read pass.

        If the ID is already claimed by a different node pointer (i.e. the user
        duplicated a node with Shift+D), a fresh deterministic ID is generated
        and persisted back into bl_node["ace_id"] so the duplicate gets a
        stable unique identity on subsequent sessions.

        Args:
            bl_node: The Blender node whose ace_id is being resolved.
            ace_id: Candidate ID string from the node's custom property.
            category: ACE node category key.
            node_type: ACE node type key.
            deterministic_seed: Optional seed payload for collision resolution.

        Returns:
            A unique ACE ID string assigned to *bl_node* for this read pass.
        """
        if not ace_id:
            return ""
        node_ptr = self._node_pointer(bl_node)
        cached = self._node_id_by_ptr.get(node_ptr)
        if cached:
            return cached
        owner_ptr = self._ace_id_owner.get(ace_id)
        reserved_elsewhere = ace_id in self._reserved_node_ids and owner_ptr != node_ptr
        if (owner_ptr is None or owner_ptr == node_ptr) and not reserved_elsewhere:
            self._ace_id_owner[ace_id] = node_ptr
            self._node_id_by_ptr[node_ptr] = ace_id
            self._reserved_node_ids.add(ace_id)
            return ace_id

        existing_ids = list(set(self._ace_id_owner.keys()).union(self._reserved_node_ids))
        seed_payload = str(deterministic_seed or f"{category}:{node_type}")

        try:
            new_id = ACEIdManager.generate_node_id(
                category,
                node_type,
                existing_ids=existing_ids,
                deterministic_seed=seed_payload,
            )
        except Exception as exc:
            utils.note_swallowed("dummy_tree.ACEDummyTreeReader._reserve_unique_ace_id", exc)
            return ace_id

        self._ace_id_owner[new_id] = node_ptr
        self._node_id_by_ptr[node_ptr] = new_id
        self._reserved_node_ids.add(new_id)
        # Persist the disambiguated ID back into the node's custom property so
        # that duplicate nodes have stable unique IDs on the next session/reload.
        with utils.logged_errors("dummy_tree.ACEDummyTreeReader._reserve_unique_ace_id#2"):
            bl_node["ace_id"] = new_id
        return new_id

    def _resolve_node_kind(self, bl_node) -> tuple[str, str]:
        if getattr(bl_node, "type", "") != 'GROUP':
            return "", ""
        node_tree = getattr(bl_node, "node_tree", None)
        group_name = getattr(node_tree, "name", "") if node_tree else ""
        if not group_name:
            return "", ""

        # O(1) exact lookup via pre-built reverse dict — avoids O(categories×types)
        # scan on every node during compilation of large GN trees.
        self._ensure_group_name_lookup()
        result = self._GROUP_NAME_LOOKUP.get(group_name)
        if result:
            return result
        norm_group_name = group_name.lstrip(".")
        result = self._GROUP_NAME_LOOKUP.get(norm_group_name)
        if result:
            return result

        # Fallback: prefix-based scan for dynamic/specialized variants.
        # This covers splitter/audio_math dynamic arity groups and shader
        # per-instance groups such as ".ACE_SHADER_stream_rms_<hash>".
        resolved_cat = ""
        resolved_type = ""
        clean_group_name = ACEIdManager._clean_string(norm_group_name).lower()
        for cat_key, category in ACE_NODES.items():
            for type_key in category.get("nodes", {}).keys():
                canonical = self.FACTORY_CLS.get_group_name(cat_key, type_key)
                norm_canonical = canonical.lstrip(".")
                clean_canonical = ACEIdManager._clean_string(norm_canonical).lower()
                if (
                    group_name.startswith(canonical + "_")
                    or group_name.startswith(canonical + ".")
                    or norm_group_name.startswith(norm_canonical + "_")
                    or norm_group_name.startswith(norm_canonical + ".")
                    or (clean_group_name and clean_group_name.startswith(clean_canonical))
                ):
                    resolved_cat = cat_key
                    resolved_type = type_key
                    break
            if resolved_cat:
                break
        return resolved_cat, resolved_type

    def _node_kind_token(self, bl_node) -> str:
        cat, typ = self._resolve_node_kind(bl_node)
        if cat and typ:
            return f"{cat}:{typ}"
        return str(getattr(bl_node, "type", "") or "node")

    @staticmethod
    def _socket_signature(sockets) -> list[tuple[int, str, str, str, int]]:
        sig: list[tuple[int, str, str, str, int]] = []
        for idx, sock in enumerate(getattr(sockets, "__iter__", lambda: [])()):
            sock_identifier = str(getattr(sock, "identifier", "") or "")
            sock_name = str(getattr(sock, "name", "") or "")
            sock_type = str(
                getattr(sock, "bl_socket_idname", "")
                or getattr(sock, "type", "")
                or type(sock).__name__
            )
            linked = int(bool(getattr(sock, "is_linked", False)))
            sig.append((int(idx), sock_identifier, sock_name, sock_type, linked))
        return sig

    @staticmethod
    def _socket_index_for(node, sock, is_input: bool) -> int:
        with utils.logged_errors("dummy_tree.ACEDummyTreeReader._socket_index_for"):
            idx_val = getattr(sock, "index")
            if idx_val is not None:
                return int(idx_val)
        with utils.logged_errors("dummy_tree.ACEDummyTreeReader._socket_index_for#2"):
            seq = getattr(node, "inputs" if is_input else "outputs", [])
            for idx, ref in enumerate(seq):
                if ref == sock:
                    return int(idx)
        return 0

    def _topology_signature(self, bl_node) -> dict[str, list[tuple[int, int, str]]]:
        incoming: list[tuple[int, int, str]] = []
        outgoing: list[tuple[int, int, str]] = []
        with utils.logged_errors("dummy_tree.ACEDummyTreeReader._topology_signature"):
            for in_sock in getattr(bl_node, "inputs", []):
                in_idx = self._socket_index_for(bl_node, in_sock, is_input=True)
                for link in getattr(in_sock, "links", []):
                    src_node = getattr(link, "from_node", None)
                    src_sock = getattr(link, "from_socket", None)
                    if src_node is None or src_sock is None:
                        continue
                    src_idx = self._socket_index_for(src_node, src_sock, is_input=False)
                    incoming.append((int(in_idx), int(src_idx), self._node_kind_token(src_node)))
        with utils.logged_errors("dummy_tree.ACEDummyTreeReader._topology_signature#2"):
            for out_sock in getattr(bl_node, "outputs", []):
                out_idx = self._socket_index_for(bl_node, out_sock, is_input=False)
                for link in getattr(out_sock, "links", []):
                    dst_node = getattr(link, "to_node", None)
                    dst_sock = getattr(link, "to_socket", None)
                    if dst_node is None or dst_sock is None:
                        continue
                    dst_idx = self._socket_index_for(dst_node, dst_sock, is_input=True)
                    outgoing.append((int(out_idx), int(dst_idx), self._node_kind_token(dst_node)))
        return {
            "incoming": sorted(set(incoming)),
            "outgoing": sorted(set(outgoing)),
        }

    def _build_deterministic_node_seed(
        self,
        bl_node,
        category: str,
        node_type: str,
        root_id: str = "",
    ) -> str:
        source_context = ""
        if category == "source_mngmt" and node_type == "src":
            try:
                source_context = str(bl_node.get("ace_source_id", "") or "")
            except Exception as exc:
                utils.note_swallowed("dummy_tree.ACEDummyTreeReader._build_deterministic_node_seed", exc)
                source_context = ""
        payload = {
            "frontend_uid": str(self.frontend_uid or ""),
            "category": str(category),
            "type": str(node_type),
            "socket_signature": {
                "inputs": self._socket_signature(getattr(bl_node, "inputs", [])),
                "outputs": self._socket_signature(getattr(bl_node, "outputs", [])),
            },
            "topology_context": self._topology_signature(bl_node),
            "source_context": str(source_context),
            "root_context": str(root_id or ""),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _ensure_ace_id(self, bl_node, root_id: str = "") -> str:
        """Return the ACE ID for *bl_node*, generating one deterministically if absent.

        All nodes derive their ID from a content hash of their socket
        signature and topology context so that the same physical wiring always
        yields the same ID without persisting state. Source management nodes
        keep the stable source token separately in ``ace_source_id``.

        Args:
            bl_node: The Blender node to identify.
            root_id: The source root identifier for deterministic seed scoping.

        Returns:
            A non-empty ACE ID string, or an empty string if the node is not an
            ACE node group.
        """
        node_ptr = self._node_pointer(bl_node)
        cached = self._node_id_by_ptr.get(node_ptr)
        if cached:
            return cached

        resolved_cat, resolved_type = self._resolve_node_kind(bl_node)
        if not resolved_cat:
            return ""

        # Source nodes: use the stored ace_id as the primary identity (stable across topology
        # changes) so that _sync_shader_output_ids / _sync_auto_attr_ids can look them up by
        # bl_node["ace_id"] without the id drifting when connections change.
        try:
            stored_ace_id = str(bl_node.get("ace_id", "") or "")
        except Exception as exc:
            utils.note_swallowed("dummy_tree.ACEDummyTreeReader._ensure_ace_id", exc)
            stored_ace_id = ""
        parsed_stored = ACEIdManager.parse_node_id(stored_ace_id) if stored_ace_id else None
        if parsed_stored and parsed_stored.get("category") == resolved_cat and parsed_stored.get("type") == resolved_type:
            if resolved_cat == "source_mngmt" and resolved_type == "src":
                try:
                    source_id = str(bl_node.get("ace_source_id", "") or "")
                except Exception as exc:
                    utils.note_swallowed("dummy_tree.ACEDummyTreeReader._ensure_ace_id#2", exc)
                    source_id = ""
                # Must not accidentally use the source's own stable_id as the node id.
                if stored_ace_id == source_id:
                    stored_ace_id = ""
                    parsed_stored = None

        if stored_ace_id and parsed_stored:
            det_seed = (
                f"{self.frontend_uid}:source-node:{bl_node.get('ace_source_id', '') or ''}"
                if resolved_cat == "source_mngmt" and resolved_type == "src"
                else stored_ace_id
            )
            resolved_id = self._reserve_unique_ace_id(
                bl_node, stored_ace_id, resolved_cat, resolved_type, deterministic_seed=det_seed,
            )
            # Persist back so duplicated nodes (which get a new id from _reserve_unique_ace_id)
            # are also stable on the next compile.
            with utils.logged_errors("dummy_tree.ACEDummyTreeReader._ensure_ace_id#3"):
                if str(bl_node.get("ace_id", "") or "") != resolved_id:
                    bl_node["ace_id"] = resolved_id
            return resolved_id

        seed = self._build_deterministic_node_seed(
            bl_node,
            resolved_cat,
            resolved_type,
            root_id=str(root_id or ""),
        )
        try:
            generated = ACEIdManager.generate_node_id(
                resolved_cat,
                resolved_type,
                existing_ids=list(self._ace_id_owner.keys()),
                deterministic_seed=seed,
            )
        except Exception as exc:
            utils.note_swallowed("dummy_tree.ACEDummyTreeReader._ensure_ace_id#4", exc)
            return ""
        resolved_id = self._reserve_unique_ace_id(
            bl_node,
            generated,
            resolved_cat,
            resolved_type,
            deterministic_seed=seed,
        )
        # Persist the generated id so the next compile reuses it instead of re-generating
        # from topology (which changes whenever the user adds/removes links).
        with utils.logged_errors("dummy_tree.ACEDummyTreeReader._ensure_ace_id#5"):
            if str(bl_node.get("ace_id", "") or "") != resolved_id:
                bl_node["ace_id"] = resolved_id
        return resolved_id
