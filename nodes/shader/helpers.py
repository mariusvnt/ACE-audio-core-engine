"""Helper Attribute nodes that read each ACE shader output's object property."""

import bpy  # type: ignore
from ...vars import ACE_NODES
from ... import utils
from ...core.id_management import ACEIdManager
from ..frontend import FRONTEND_KIND_SHADER


class HelperAttributeNodes:
    @classmethod
    def _find_helper_attr_nodes(cls, tree, ace_id: str) -> dict[int, object]:
        found: dict[int, object] = {}
        target_ace_id = str(ace_id or "")
        # Pre-compute expected node names so we can match by name when
        # _HELPER_NODE_OWNER_KEY is stale or unset (e.g. after an ace_id
        # scheme change).  Name-based matching prevents _ensure_helper_attr_node
        # from creating a duplicate alongside the existing node.
        expected_name_to_idx: dict[str, int] = {
            cls._helper_node_name(target_ace_id, i): i for i in range(16)
        }
        for helper_node in getattr(tree, "nodes", []) or []:
            if str(getattr(helper_node, "bl_idname", "") or "") != "ShaderNodeAttribute":
                continue
            try:
                owner = str(helper_node.get(cls._HELPER_NODE_OWNER_KEY, "") or "")
            except Exception as exc:
                utils.note_swallowed("shader_tree.cls._find_helper_attr_nodes", exc)
                owner = ""
            matched_by_owner = bool(target_ace_id and owner == target_ace_id)
            if not matched_by_owner:
                # Fallback: match by deterministic node name.  This handles nodes
                # whose owner key was written with a stale id or was never set.
                node_name = str(getattr(helper_node, "name", "") or "")
                if node_name not in expected_name_to_idx:
                    continue
            try:
                out_idx = utils.as_int(helper_node.get(cls._ATTR_NODE_OUTPUT_IDX_KEY), -1)
            except Exception as exc:
                utils.note_swallowed("shader_tree.cls._find_helper_attr_nodes#2", exc)
                out_idx = -1
            if out_idx < 0:
                # Derive index from the node name if the custom property is missing.
                node_name = str(getattr(helper_node, "name", "") or "")
                out_idx = expected_name_to_idx.get(node_name, -1)
            if out_idx < 0:
                continue
            found[int(out_idx)] = helper_node
        return found

    @classmethod
    def _ensure_helper_attr_node(
        cls,
        tree,
        owner_node,
        ace_id: str,
        output_idx: int,
        output_name: str,
        location: tuple[float, float] | None = None,
    ):
        changed = False
        helper_nodes = cls._find_helper_attr_nodes(tree, ace_id)
        helper_node = helper_nodes.get(int(output_idx))
        if helper_node is None:
            try:
                helper_node = tree.nodes.new("ShaderNodeAttribute")
            except Exception as exc:
                utils.note_swallowed("shader_tree.cls._ensure_helper_attr_node", exc)
                return None, False
            changed = True
        with utils.logged_errors("shader_tree.cls._ensure_helper_attr_node#2"):
            if str(helper_node.get(cls._HELPER_NODE_OWNER_KEY, "") or "") != str(ace_id or ""):
                helper_node[cls._HELPER_NODE_OWNER_KEY] = str(ace_id or "")
                changed = True
        with utils.logged_errors("shader_tree.cls._ensure_helper_attr_node#3"):
            if utils.as_int(helper_node.get(cls._ATTR_NODE_OUTPUT_IDX_KEY), -1) != int(output_idx):
                helper_node[cls._ATTR_NODE_OUTPUT_IDX_KEY] = int(output_idx)
                changed = True
        with utils.logged_errors("shader_tree.cls._ensure_helper_attr_node#4"):
            if str(helper_node.get(cls._OUTPUT_NODE_KIND_KEY, "") or "") != "TOP_LEVEL_ATTRIBUTE":
                helper_node[cls._OUTPUT_NODE_KIND_KEY] = "TOP_LEVEL_ATTRIBUTE"
                changed = True
        helper_name = cls._helper_node_name(ace_id, output_idx)
        with utils.logged_errors("shader_tree.cls._ensure_helper_attr_node#5"):
            if str(getattr(helper_node, "name", "") or "") != helper_name:
                helper_node.name = helper_name
                changed = True
        with utils.logged_errors("shader_tree.cls._ensure_helper_attr_node#6"):
            label = f"ACE {str(output_name or f'Out {int(output_idx + 1)}')}"
            if str(getattr(helper_node, "label", "") or "") != label:
                helper_node.label = label
                changed = True
        if cls._ensure_helper_node_compact_display(helper_node):
            changed = True
        with utils.logged_errors("shader_tree.cls._ensure_helper_attr_node#7"):
            if location is not None:
                helper_node.location = (float(location[0]), float(location[1]))
        return helper_node, changed

    @classmethod
    def _hide_helper_inputs(cls, node, node_def) -> bool:
        changed = False
        helper_names = {helper_name for _out_idx, _sock_type, helper_name in cls._helper_input_specs(node_def)}
        for sock in getattr(node, "inputs", []) or []:
            if str(getattr(sock, "name", "") or "") not in helper_names:
                continue
            if hasattr(sock, "hide"):
                with utils.logged_errors("shader_tree.cls._hide_helper_inputs"):
                    if not bool(getattr(sock, "hide", False)):
                        sock.hide = True
                        changed = True
            if hasattr(sock, "hide_value"):
                with utils.logged_errors("shader_tree.cls._hide_helper_inputs#2"):
                    if not bool(getattr(sock, "hide_value", False)):
                        sock.hide_value = True
                        changed = True
        return changed

    @classmethod
    def _ensure_helper_link(cls, tree, helper_node, helper_output, group_node, target_input) -> bool:
        if helper_node is None or helper_output is None or target_input is None:
            return False
        changed = False
        for link in list(getattr(target_input, "links", []) or []):
            if getattr(link, "from_node", None) == helper_node and getattr(link, "from_socket", None) == helper_output:
                return changed
            with utils.logged_errors("shader_tree.cls._ensure_helper_link"):
                tree.links.remove(link)
                changed = True
        with utils.logged_errors("shader_tree.cls._ensure_helper_link#2"):
            tree.links.new(helper_output, target_input)
            changed = True
        return changed

    @classmethod
    def _store_helper_attr_name(cls, helper_node, attr_name: str) -> bool:
        if helper_node is None:
            return False
        target_name = str(attr_name or "")
        try:
            current_name = str(helper_node.get(cls._HELPER_ATTR_NAME_KEY, "") or "")
        except Exception as exc:
            utils.note_swallowed("shader_tree.cls._store_helper_attr_name", exc)
            current_name = ""
        if current_name == target_name:
            return False
        try:
            helper_node[cls._HELPER_ATTR_NAME_KEY] = target_name
            return True
        except Exception as exc:
            utils.note_swallowed("shader_tree.cls._store_helper_attr_name#2", exc)
            return False

    @classmethod
    def _rebuild_helper_attr_node(
        cls,
        tree,
        helper_node,
        owner_node,
        ace_id: str,
        output_idx: int,
        output_name: str,
        socket_type: str,
        target_name: str,
    ):
        if tree is None or helper_node is None:
            return helper_node, False
        try:
            prev_location = cls._node_local_location(helper_node)
        except Exception as exc:
            utils.note_swallowed("shader_tree.cls._rebuild_helper_attr_node", exc)
            prev_location = None

        preserved_links: list[tuple[str, object]] = []
        for out_sock in list(getattr(helper_node, "outputs", []) or []):
            out_name = str(getattr(out_sock, "name", "") or "")
            for link in list(getattr(out_sock, "links", []) or []):
                target_socket = getattr(link, "to_socket", None)
                if target_socket is None:
                    continue
                preserved_links.append((out_name, target_socket))

        try:
            tree.nodes.remove(helper_node)
        except Exception as exc:
            utils.note_swallowed("shader_tree.cls._rebuild_helper_attr_node#2", exc)
            return helper_node, False

        rebuilt_node, changed = cls._ensure_helper_attr_node(
            tree,
            owner_node,
            ace_id,
            output_idx,
            output_name,
            location=prev_location,
        )
        if rebuilt_node is None:
            return None, True
        if cls._set_attr_node_defaults(rebuilt_node, target_name):
            changed = True
        if cls._store_helper_attr_name(rebuilt_node, target_name):
            changed = True

        fallback_output = cls._shader_attr_output_socket(rebuilt_node, socket_type)
        for output_name_token, target_socket in preserved_links:
            source_socket = next(
                (
                    sock
                    for sock in getattr(rebuilt_node, "outputs", []) or []
                    if str(getattr(sock, "name", "") or "") == str(output_name_token or "")
                ),
                None,
            )
            if source_socket is None:
                source_socket = fallback_output
            if source_socket is None or target_socket is None:
                continue
            with utils.logged_errors("shader_tree.cls._rebuild_helper_attr_node#3"):
                tree.links.new(source_socket, target_socket)
                changed = True
        return rebuilt_node, changed

    @classmethod
    def cleanup_tree_helper_nodes(cls, tree) -> bool:
        if tree is None:
            return False
        live_owner_ids: set[str] = set()
        live_node_names: set[str] = set()
        for node in getattr(tree, "nodes", []) or []:
            try:
                ace_id = str(node.get("ace_id", "") or "")
            except Exception as exc:
                utils.note_swallowed("shader_tree.cls.cleanup_tree_helper_nodes", exc)
                ace_id = ""
            if ace_id:
                live_owner_ids.add(ace_id)
                # Pre-compute the expected helper/frame names for this live node so
                # we can catch orphans whose owner key is gone but whose name still
                # matches the pattern (e.g. left over from a previous ace_id scheme).
                for out_idx in range(16):
                    live_node_names.add(cls._helper_node_name(ace_id, out_idx))
                live_node_names.add(cls._helper_frame_name(ace_id))

        changed = False
        for helper_node in list(getattr(tree, "nodes", []) or []):
            bl_idname = str(getattr(helper_node, "bl_idname", "") or "")
            owner_id = ""
            if bl_idname == "ShaderNodeAttribute":
                try:
                    owner_id = str(helper_node.get(cls._HELPER_NODE_OWNER_KEY, "") or "")
                except Exception as exc:
                    utils.note_swallowed("shader_tree.cls.cleanup_tree_helper_nodes#2", exc)
                    owner_id = ""
            elif cls._is_frame_node(helper_node):
                try:
                    owner_id = str(helper_node.get(cls._HELPER_FRAME_OWNER_KEY, "") or "")
                except Exception as exc:
                    utils.note_swallowed("shader_tree.cls.cleanup_tree_helper_nodes#3", exc)
                    owner_id = ""
                if owner_id:
                    has_children = any(getattr(node, "parent", None) == helper_node for node in getattr(tree, "nodes", []) or [])
                    if has_children and owner_id in live_owner_ids:
                        continue
                    with utils.logged_errors("shader_tree.cls.cleanup_tree_helper_nodes#4"):
                        tree.nodes.remove(helper_node)
                        changed = True
                    continue
            else:
                continue
            node_name = str(getattr(helper_node, "name", "") or "")
            # Primary check: owner key points at a live ace_id.
            if owner_id and owner_id in live_owner_ids:
                continue
            # Fallback: the node's name is one of the expected helper names — keep it.
            # This handles nodes whose owner key was written with an old id scheme but
            # whose name still matches the current live node.
            if node_name and node_name in live_node_names:
                continue
            # Neither check passed — orphan, remove it.
            with utils.logged_errors("shader_tree.cls.cleanup_tree_helper_nodes#5"):
                tree.nodes.remove(helper_node)
                changed = True
        return changed

    @classmethod
    def sync_node_instance_group_bindings(
        cls, node, proxy_name: str = "", scene=None,
        attr_names_by_idx: dict[int, str] | None = None,
    ) -> bool:
        """Ensure shader node instance outputs read the intended object properties."""
        if node is None or str(getattr(node, "type", "") or "") != "GROUP":
            return False
        if scene is None:
            try:
                scene = getattr(getattr(bpy, "context", None), "scene", None)
            except Exception as exc:
                utils.note_swallowed("shader_tree.cls.sync_node_instance_group_bindings", exc)
                scene = None
        with utils.logged_errors("shader_tree.cls.sync_node_instance_group_bindings#2"):
            from ...core.engine.engine import ACEAudioEngine
            if not ACEAudioEngine._structural_tree_writes_allowed(
                scene,
                tree=getattr(node, "id_data", None),
                frontend_kind=FRONTEND_KIND_SHADER,
            ):
                return False
        try:
            ace_id = str(node.get("ace_id", "") or "")
        except Exception as exc:
            utils.note_swallowed("shader_tree.cls.sync_node_instance_group_bindings#3", exc)
            ace_id = ""
        parsed = ACEIdManager.parse_node_id(ace_id) if ace_id else None
        if not parsed:
            return False
        category = str(parsed.get("category", "") or "")
        node_type = str(parsed.get("type", "") or "")
        if not cls._is_property_output_node(category, node_type):
            return False

        node_def = ACE_NODES.get(category, {}).get("nodes", {}).get(node_type, {})
        output_defs = list(node_def.get("outputs", [])) if isinstance(node_def, dict) else []
        if not output_defs:
            return False

        expected_indices = cls._expected_attribute_output_indices({"outputs": output_defs})
        if not expected_indices:
            return False

        # Use caller-supplied names first, then fill any gaps from the visible
        # Attribute ID sockets on the node instance itself.
        explicit_by_idx: dict[int, str] = {}
        for raw_idx, raw_name in dict(attr_names_by_idx or {}).items():
            try:
                out_idx = int(raw_idx)
            except Exception as exc:
                utils.note_swallowed("shader_tree.cls.sync_node_instance_group_bindings#4", exc)
                continue
            sanitized = cls._sanitize_explicit_name(str(raw_name or ""))
            if sanitized:
                explicit_by_idx[out_idx] = sanitized

        socket_to_idx = cls._socket_name_to_output_idx(category, node_type)
        for sock in getattr(node, "inputs", []) or []:
            out_idx = socket_to_idx.get(str(getattr(sock, "name", "") or ""))
            if out_idx is None or int(out_idx) in explicit_by_idx:
                continue
            raw_val = getattr(sock, "default_value", None)
            if isinstance(raw_val, str):
                sanitized = cls._sanitize_explicit_name(raw_val)
                if sanitized:
                    explicit_by_idx[int(out_idx)] = sanitized

        group, changed = cls._ensure_shared_group_for_node(node, category, node_type)
        if group is None:
            return changed
        if cls._hide_helper_inputs(node, node_def):
            changed = True

        tree = getattr(node, "id_data", None)
        if tree is None:
            return changed
        helper_nodes_by_idx = cls._find_helper_attr_nodes(tree, ace_id)
        expected_idx_set = {int(idx) for idx in expected_indices}
        for stale_idx, stale_node in list(helper_nodes_by_idx.items()):
            if int(stale_idx) in expected_idx_set:
                continue
            with utils.logged_errors("shader_tree.cls.sync_node_instance_group_bindings#5"):
                tree.nodes.remove(stale_node)
                changed = True
            helper_nodes_by_idx.pop(int(stale_idx), None)

        for out_idx in expected_indices:
            if out_idx < 0 or out_idx >= len(output_defs):
                continue
            sock_type, default_output_name = output_defs[out_idx]
            if "Bundle" in str(sock_type or ""):
                continue

            output_name = str(default_output_name or f"out_{int(out_idx)}")
            node_outputs = list(getattr(node, "outputs", []) or [])
            if out_idx < len(node_outputs):
                output_name = str(getattr(node_outputs[out_idx], "name", "") or output_name)

            target_name = explicit_by_idx.get(int(out_idx), "")
            if not target_name:
                target_name = ACEIdManager.generate_compact_attr_name(
                    str(proxy_name or ""),
                    str(ace_id or ""),
                    output_name,
                    int(out_idx),
                )[:63]
            if not target_name:
                target_name = cls._placeholder_attr_name(int(out_idx), output_name)

            helper_node, helper_changed = cls._ensure_helper_attr_node(
                tree,
                node,
                ace_id,
                int(out_idx),
                output_name,
            )
            if helper_node is None:
                continue
            if helper_changed:
                changed = True
                helper_nodes_by_idx[int(out_idx)] = helper_node
            # Update attribute_name in place — never rebuild (remove + recreate) the node.
            # Rebuilding is both unnecessary (only attribute_name needs to change) and
            # dangerous: if the removal is interrupted the old node stays while a new one
            # is also inserted, causing accumulation.
            if cls._set_attr_node_defaults(helper_node, target_name):
                changed = True
            if cls._store_helper_attr_name(helper_node, target_name):
                changed = True

            helper_input_name = cls._helper_input_name(int(out_idx))
            helper_input = next(
                (
                    sock
                    for sock in getattr(node, "inputs", []) or []
                    if str(getattr(sock, "name", "") or "") == helper_input_name
                ),
                None,
            )
            helper_output = cls._shader_attr_output_socket(helper_node, str(sock_type or ""))
            if cls._ensure_helper_link(tree, helper_node, helper_output, node, helper_input):
                changed = True
            if cls._sync_helper_node_layout(tree, node, helper_node, ace_id, int(out_idx)):
                changed = True

        return changed
