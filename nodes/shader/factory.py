"""Shader node groups for ACE nodes: creation, content and refresh."""

import hashlib
import json
import bpy  # type: ignore
from ...vars import ACE_NODES
from ... import utils
from ...core.id_management import ACEIdManager
from .helpers import HelperAttributeNodes
from .interface import ShaderGroupInterface
from .layout import HelperNodeLayout


class ACEShaderNodeFactory(ShaderGroupInterface, HelperNodeLayout, HelperAttributeNodes):
    """Create hidden ShaderNodeTree ACE groups for shader frontends."""

    _NODES_CONTENT_HASH: str = ""
    _ATTR_NODE_OUTPUT_IDX_KEY = "ace_shader_output_idx"
    _OUTPUT_NODE_KIND_KEY = "ace_shader_sink_kind"
    _HELPER_NODE_OWNER_KEY = "ace_shader_helper_owner"
    _HELPER_ATTR_NAME_KEY = "ace_shader_attr_name"
    _HELPER_FRAME_OWNER_KEY = "ace_shader_helper_frame_owner"
    _HELPER_FRAME_KIND_KEY = "ace_shader_helper_frame_kind"
    _HELPER_INPUT_PREFIX = "ACE Property"
    _HELPER_FALLBACK_OWNER_HEIGHT = 128.0
    _HELPER_VERTICAL_GAP = 18.0
    _HELPER_VERTICAL_STEP = 62.0
    _HELPER_HIDDEN_WIDTH = 150.0

    @staticmethod
    def get_group_name(category: str, node_type: str) -> str:
        return f".ACE_SHADER_{category}_{node_type}"

    @classmethod
    def _ace_nodes_hash(cls) -> str:
        try:
            serialized = json.dumps(
                {"kind": "shader", "nodes": ACE_NODES},
                sort_keys=True,
                default=str,
            ).encode("utf-8")
            return hashlib.blake2b(serialized, digest_size=8).hexdigest()
        except Exception as exc:
            utils.note_swallowed("shader_tree.cls._ace_nodes_hash", exc)
            return ""

    @staticmethod
    def _shader_attr_output_socket(attr_node, socket_type: str):
        st = str(socket_type or "")
        if "Vector" in st:
            preferred = ("Vector", "Color", "Factor", "Fac", "Alpha")
        elif "Color" in st or "RGBA" in st:
            preferred = ("Color", "Vector", "Factor", "Fac", "Alpha")
        else:
            preferred = ("Factor", "Fac", "Alpha", "Color", "Vector")
        outputs = list(getattr(attr_node, "outputs", []) or [])
        for wanted in preferred:
            for sock in outputs:
                if str(getattr(sock, "name", "") or "") == wanted:
                    return sock
        return outputs[0] if outputs else None

    @classmethod
    def _set_attr_node_defaults(cls, attr_node, attr_name: str) -> bool:
        changed = False
        if hasattr(attr_node, "attribute_type"):
            for enum_token in ("OBJECT", "OBJECT_DATA"):
                try:
                    if str(getattr(attr_node, "attribute_type", "") or "") == enum_token:
                        break
                    attr_node.attribute_type = enum_token
                    changed = True
                    break
                except Exception as exc:
                    utils.note_swallowed("shader_tree.cls._set_attr_node_defaults", exc)
                    continue
        name = str(attr_name or "").strip()
        if not name:
            return changed
        prop_names = []
        if hasattr(attr_node, "attribute_name"):
            prop_names.append("attribute_name")
        elif hasattr(attr_node, "name"):
            prop_names.append("name")
        for prop_name in prop_names:
            if not hasattr(attr_node, prop_name):
                continue
            try:
                current_name = str(getattr(attr_node, prop_name, "") or "")
            except Exception as exc:
                utils.note_swallowed("shader_tree.cls._set_attr_node_defaults#2", exc)
                current_name = ""
            if current_name == name:
                continue
            with utils.logged_errors("shader_tree.cls._set_attr_node_defaults#3"):
                setattr(attr_node, prop_name, name)
                changed = True
        return changed

    @staticmethod
    def _placeholder_attr_name(output_idx: int, output_name: str) -> str:
        token = ACEIdManager._clean_string(str(output_name or f"out_{int(output_idx)}")).lower() or f"out{int(output_idx)}"
        return f"{ACEIdManager.PREFIX}{token[:20]}_{int(output_idx)}"[:63]

    @classmethod
    def _find_group_output_node(cls, ng):
        for node in getattr(ng, "nodes", []) or []:
            if str(getattr(node, "type", "") or "") == "GROUP_OUTPUT":
                return node
        return None

    @classmethod
    def _create_output_sink_node(
        cls,
        ng,
        output_node,
        output_idx: int,
        socket_type: str,
        socket_name: str,
    ):
        if output_node is None or output_idx < 0 or output_idx >= len(getattr(output_node, "inputs", []) or []):
            return None
        try:
            sink_node = ng.nodes.new("ShaderNodeAttribute")
        except Exception as exc:
            utils.note_swallowed("shader_tree.cls._create_output_sink_node", exc)
            return None
        sink_kind = "ATTRIBUTE"
        cls._set_attr_node_defaults(
            sink_node,
            cls._placeholder_attr_name(output_idx, socket_name),
        )
        out_sock = cls._shader_attr_output_socket(sink_node, socket_type)

        sink_node.location = (300, -200 - (int(output_idx) * 150))
        with utils.logged_errors("shader_tree.cls._create_output_sink_node#2"):
            sink_node[cls._ATTR_NODE_OUTPUT_IDX_KEY] = int(output_idx)
        with utils.logged_errors("shader_tree.cls._create_output_sink_node#3"):
            sink_node[cls._OUTPUT_NODE_KIND_KEY] = str(sink_kind)
        with utils.logged_errors("shader_tree.cls._create_output_sink_node#4"):
            sink_node["ace_shader_output_name"] = str(socket_name or "")
        if out_sock is not None:
            with utils.logged_errors("shader_tree.cls._create_output_sink_node#5"):
                ng.links.new(out_sock, output_node.inputs[int(output_idx)])
        return sink_node

    @classmethod
    def _set_socket_default(cls, ng, socket_name: str, default_value: str):
        if default_value == "":
            return
        try:
            item = next(
                (
                    it
                    for it in getattr(ng.interface, "items_tree", [])
                    if getattr(it, "in_out", None) == "INPUT" and getattr(it, "name", "") == socket_name
                ),
                None,
            )
        except Exception as exc:
            utils.note_swallowed("shader_tree.cls._set_socket_default", exc)
            item = None
        if item is None:
            return
        if not hasattr(item, "default_value"):
            return
        with utils.logged_errors("shader_tree.cls._set_socket_default#2"):
            item.default_value = default_value

    @classmethod
    def _ensure_attr_interface_visible(cls, ng, category: str, node_type: str):
        for attr_name in cls._attr_socket_names(category, node_type):
            item = next(
                (
                    it
                    for it in getattr(ng.interface, "items_tree", [])
                    if getattr(it, "in_out", None) == "INPUT" and getattr(it, "name", "") == attr_name
                ),
                None,
            )
            if item is not None and hasattr(item, "hide_value"):
                with utils.logged_errors("shader_tree.cls._ensure_attr_interface_visible"):
                    item.hide_value = False

    @staticmethod
    def _bundle_item_candidates_for_socket(socket_type: str) -> tuple[str, ...]:
        st = str(socket_type or "")
        if "Bundle" in st:
            return ("BUNDLE", "VALUE", "FLOAT")
        if "String" in st:
            return ("STRING", "TEXT", "VALUE")
        if "Vector" in st:
            return ("VECTOR", "FLOAT_VECTOR", "VALUE")
        if "Color" in st or "RGBA" in st:
            return ("RGBA", "COLOR", "FLOAT", "VALUE")
        if "Int" in st:
            return ("INT", "FLOAT", "VALUE")
        if "Float" in st:
            return ("FLOAT", "VALUE")
        return ("VALUE", "FLOAT")

    @classmethod
    def _safe_bundle_item(cls, bundle_node, item_name: str, candidates: tuple[str, ...]):
        for item_type in candidates:
            try:
                return bundle_node.bundle_items.new(item_type, item_name)
            except Exception as exc:
                utils.note_swallowed("shader_tree.cls._safe_bundle_item", exc)
                continue
        return None

    @classmethod
    def _link_or_consume_reroute(cls, ng, source_socket, location_x: int, location_y: int):
        try:
            reroute = ng.nodes.new("NodeReroute")
            reroute.location = (int(location_x), int(location_y))
            ng.links.new(source_socket, reroute.inputs[0])
            return True
        except Exception as exc:
            utils.note_swallowed("shader_tree.cls._link_or_consume_reroute", exc)
            return False

    @classmethod
    def _build_group_content(cls, ng, node_def, category: str, node_type: str):
        base_inputs = list(node_def.get("inputs", []))
        base_outputs = list(node_def.get("outputs", []))
        extra_inputs = [spec for spec in cls._extra_inputs(category, node_type) if cls._shader_interface_supports(spec[0])]
        helper_inputs = list(cls._helper_input_specs(node_def))

        for sock_type_str, sock_name in base_inputs:
            cls._new_socket(ng, str(sock_name), "INPUT", str(sock_type_str))
        for sock_type_str, sock_name in base_outputs:
            cls._new_socket(ng, str(sock_name), "OUTPUT", str(sock_type_str))

        if str(category or "") == "stream" and str(node_type or "") == "lufs":
            for item in getattr(getattr(ng, "interface", None), "items_tree", []):
                if getattr(item, "in_out", None) != "INPUT":
                    continue
                name = str(getattr(item, "name", "") or "")
                if name == "Window (s)":
                    with utils.logged_errors("shader_tree.cls._build_group_content"):
                        item.default_value = 0.4
                    for attr_name, value in (
                        ("min_value", 0.05),
                        ("max_value", 3.0),
                    ):
                        with utils.logged_errors("shader_tree.cls._build_group_content#2"):
                            setattr(item, attr_name, value)

        for sock_type_str, sock_name, default_value in extra_inputs:
            cls._new_socket(ng, sock_name, "INPUT", sock_type_str)
            cls._set_socket_default(ng, sock_name, default_value)
        for _out_idx, sock_type_str, helper_name in helper_inputs:
            cls._new_socket(ng, helper_name, "INPUT", sock_type_str)

        # Shader groups are strict facades: Group Input/Output + direct wiring.
        # NO NodeSeparateBundle / NodeCombineBundle inside shader groups —
        # Cycles' ShaderNodesInliner cannot handle them and crashes (null deref).
        # Bundle sockets pass through directly; helper inputs wire to outputs.
        # Companion Attribute nodes live in the parent material tree.
        input_node = ng.nodes.new("NodeGroupInput")
        output_node = ng.nodes.new("NodeGroupOutput")
        input_node.location = (-760, 0)
        output_node.location = (680, 0)

        all_inputs = list(base_inputs) + [
            (sock_type_str, sock_name)
            for sock_type_str, sock_name, _default_value in extra_inputs
        ] + [
            (sock_type_str, helper_name)
            for _out_idx, sock_type_str, helper_name in helper_inputs
        ]
        bundle_output_indices = [
            idx for idx, (sock_type, _name) in enumerate(base_outputs) if "Bundle" in str(sock_type or "")
        ]

        # GroupInput.outputs may omit NodeSocketBundle entries in ShaderNodeTree —
        # Blender's shader backend does not always expose bundle sockets on GroupInput.
        # Using index-based access (all_inputs[i] == input_node.outputs[i]) is
        # therefore unreliable.  Look up every socket by NAME instead.
        input_sock_by_name: dict[str, object] = {}
        for _s in getattr(input_node, "outputs", []) or []:
            _sn = str(getattr(_s, "name", "") or "")
            if _sn and _sn not in input_sock_by_name:
                input_sock_by_name[_sn] = _s

        # Helper inputs: map output_idx → the named GroupInput socket
        helper_source_by_idx: dict[int, object] = {}
        for _out_idx, _sock_type, _helper_name in helper_inputs:
            _s = input_sock_by_name.get(str(_helper_name))
            if _s is not None:
                helper_source_by_idx[int(_out_idx)] = _s
        helper_socket_names = {str(_hn) for _oi, _st, _hn in helper_inputs}

        # First bundle input name (for passthrough and consume-skip logic)
        first_bundle_name: str | None = next(
            (str(_sn) for _st, _sn in all_inputs if "Bundle" in str(_st or "")),
            None,
        )

        # Direct bundle passthrough: link first bundle input straight to bundle outputs.
        if bundle_output_indices and first_bundle_name:
            _bundle_in = input_sock_by_name.get(first_bundle_name)
            if _bundle_in is not None:
                for output_idx in bundle_output_indices:
                    if output_idx < len(getattr(output_node, "inputs", [])):
                        with utils.logged_errors("shader_tree.cls._build_group_content#3"):
                            ng.links.new(_bundle_in, output_node.inputs[int(output_idx)])

        # Consume every non-helper, non-primary-bundle input via reroute.
        for _ci, (socket_type, socket_name) in enumerate(all_inputs):
            _s = input_sock_by_name.get(str(socket_name))
            if _s is None:
                continue
            if str(socket_name) in helper_socket_names:
                continue
            if "Bundle" in str(socket_type or ""):
                # Primary bundle already handled above; others get a reroute.
                if str(socket_name) != first_bundle_name:
                    cls._link_or_consume_reroute(ng, _s, -120, -160 - (_ci * 90))
                continue
            cls._link_or_consume_reroute(ng, _s, -120, -160 - (_ci * 90))

        for output_idx, (socket_type, _socket_name) in enumerate(base_outputs):
            if "Bundle" in str(socket_type or ""):
                continue
            if str(category or "") == "stream" and str(node_type or "") == "lufs":
                continue
            if output_idx >= len(getattr(output_node, "inputs", [])):
                continue
            source_out = helper_source_by_idx.get(int(output_idx))
            if source_out is None:
                continue
            with utils.logged_errors("shader_tree.cls._build_group_content#4"):
                ng.links.new(source_out, output_node.inputs[int(output_idx)])

        # LUFS-specific shader equivalent of the GN index-driver trick:
        # Window * 0 + 1 multiplies each scalar property output by 1, preserving
        # output values while keeping the control consumed (not greyed out).
        if str(category or "") == "stream" and str(node_type or "") == "lufs" and helper_source_by_idx:
            # Use name-based lookup — bundle sockets may shift numeric indices.
            window_sock = input_sock_by_name.get("Window (s)")
            if window_sock is not None:
                mul_zero = ng.nodes.new("ShaderNodeMath")
                mul_zero.operation = "MULTIPLY"
                mul_zero.inputs[1].default_value = 0.0
                mul_zero.location = (220, -320)
                ng.links.new(window_sock, mul_zero.inputs[0])

                add_one = ng.nodes.new("ShaderNodeMath")
                add_one.operation = "ADD"
                add_one.inputs[1].default_value = 1.0
                add_one.location = (420, -320)
                ng.links.new(mul_zero.outputs[0], add_one.inputs[0])

                for output_idx, source_out in helper_source_by_idx.items():
                    if output_idx < 0 or output_idx >= len(base_outputs):
                        continue
                    if output_idx >= len(getattr(output_node, "inputs", [])):
                        continue

                    socket_type = str(base_outputs[output_idx][0] or "")
                    target_input = output_node.inputs[int(output_idx)]
                    for link in list(getattr(target_input, "links", []) or []):
                        with utils.logged_errors("shader_tree.cls._build_group_content#5"):
                            ng.links.remove(link)
                    if source_out is None:
                        continue

                    if "Vector" in socket_type:
                        vec_mul = ng.nodes.new("ShaderNodeVectorMath")
                        vec_mul.operation = "SCALE"
                        vec_mul.location = (460, 180 - (int(output_idx) * 150))
                        ng.links.new(source_out, vec_mul.inputs[0])
                        scale_input = None
                        try:
                            scale_input = vec_mul.inputs.get("Scale")
                        except Exception as exc:
                            utils.note_swallowed("shader_tree.cls._build_group_content#6", exc)
                            scale_input = None
                        if scale_input is None and len(getattr(vec_mul, "inputs", [])) > 1:
                            scale_input = vec_mul.inputs[1]
                        if scale_input is not None:
                            ng.links.new(add_one.outputs[0], scale_input)
                        ng.links.new(vec_mul.outputs[0], target_input)
                    else:
                        scalar_mul = ng.nodes.new("ShaderNodeMath")
                        scalar_mul.operation = "MULTIPLY"
                        scalar_mul.location = (460, 180 - (int(output_idx) * 150))
                        ng.links.new(source_out, scalar_mul.inputs[0])
                        ng.links.new(add_one.outputs[0], scalar_mul.inputs[1])
                        ng.links.new(scalar_mul.outputs[0], target_input)

        cls._ensure_attr_interface_visible(ng, category, node_type)

    @staticmethod
    def _sanitize_explicit_name(raw_name: str) -> str:
        name = str(raw_name or "").strip()
        if not name:
            return ""
        if not name.startswith(ACEIdManager.PREFIX):
            name = ACEIdManager.PREFIX + ACEIdManager._clean_string(name)
        return name[:63]

    @staticmethod
    def _socket_name_to_output_idx(category: str, node_type: str) -> dict[str, int]:
        cat = str(category or "")
        typ = str(node_type or "")
        if cat == "stream":
            mapping: dict[str, int] = {"Attribute ID": 0}
            stream_nodes = (
                ACE_NODES.get("stream", {}).get("nodes", {})
                if isinstance(ACE_NODES.get("stream", {}), dict)
                else {}
            )
            node_def = stream_nodes.get(typ, {}) if isinstance(stream_nodes, dict) else {}
            outputs = list(node_def.get("outputs", [])) if isinstance(node_def, dict) else []
            value_output_count = max(
                1,
                sum(1 for sock_type, _sock_name in outputs if "Bundle" not in str(sock_type or "")),
            )
            for idx in range(1, int(value_output_count)):
                mapping[f"Attribute ID {int(idx + 1)}"] = int(idx)
            return mapping
        if cat == "utilities" and typ in {"util_rolling_avg", "stream_accumulator"}:
            return {"Attribute ID": 0}
        if cat == "utilities" and typ == "src_specs":
            return {
                "Frames per Beat Attribute ID": 0,
                "Beat Phase Attribute ID": 1,
            }
        return {}

    @classmethod
    def _ensure_shared_group_for_node(cls, node, category: str, node_type: str):
        desired_name = cls.get_group_name(category, node_type)
        current_group = getattr(node, "node_tree", None)
        current_name = str(getattr(current_group, "name", "") or "")
        if current_group is not None and current_name == desired_name:
            return current_group, False

        group = bpy.data.node_groups.get(desired_name)
        if group is None or str(getattr(group, "bl_idname", "") or "") != "ShaderNodeTree":
            cls.rebuild_node_group(category, node_type)
            group = bpy.data.node_groups.get(desired_name)
        if group is None:
            return None, False

        changed = False
        if current_group != group:
            with utils.logged_errors("shader_tree.cls._ensure_shared_group_for_node"):
                node.node_tree = group
                changed = True
        return group, changed

    @classmethod
    def rebuild_node_group(cls, cat_key: str, type_key: str):
        """Force-rebuild a single shader ACE node group from ACE_NODES."""
        try:
            node_def = ACE_NODES[cat_key]["nodes"][type_key]
        except KeyError:
            return

        group_name = cls.get_group_name(cat_key, type_key)
        ng = bpy.data.node_groups.get(group_name)
        if ng is not None and getattr(ng, "bl_idname", "") != "ShaderNodeTree":
            try:
                bpy.data.node_groups.remove(ng)
            except Exception as exc:
                utils.note_swallowed("shader_tree.cls.rebuild_node_group", exc)
                ng = None
            else:
                ng = None
        if ng is None:
            ng = bpy.data.node_groups.new(group_name, "ShaderNodeTree")
        else:
            ng.nodes.clear()
            ng.interface.clear()
        cls._build_group_content(ng, node_def, cat_key, type_key)

    @classmethod
    def _cleanup_legacy_drivers(cls):
        """Strip stale driver FCurves from ACE shader node groups.

        Legacy scenes may contain driver-based sink nodes (ShaderNodeValue,
        ShaderNodeCombineXYZ, ShaderNodeRGB with driver FCurves).  This method
        removes those FCurves so they do not cause evaluation overhead.
        """
        try:
            node_groups = bpy.data.node_groups
        except AttributeError:
            return
        for ng in node_groups:
            name = str(getattr(ng, "name", "") or "")
            if not name.startswith(".ACE_SHADER_"):
                continue
            anim_data = getattr(ng, "animation_data", None)
            if anim_data is None:
                continue
            drivers = getattr(anim_data, "drivers", None)
            if drivers is None or len(drivers) == 0:
                continue
            for fc in list(drivers):
                with utils.logged_errors("shader_tree.cls._cleanup_legacy_drivers"):
                    drivers.remove(fc)

    @classmethod
    def ensure_node_groups_exist(cls):
        try:
            _ = bpy.data.node_groups
        except AttributeError:
            utils.log("Shader node groups unavailable in current context; skipping build.", "WARNING")
            return
        current_hash = cls._ace_nodes_hash()
        if current_hash and current_hash == cls._NODES_CONTENT_HASH:
            return
        cls._cleanup_legacy_drivers()
        for cat_key, category in ACE_NODES.items():
            for type_key, node_def in category.get("nodes", {}).items():
                group_name = cls.get_group_name(cat_key, type_key)
                ng = bpy.data.node_groups.get(group_name)
                if ng is not None and getattr(ng, "bl_idname", "") != "ShaderNodeTree":
                    try:
                        bpy.data.node_groups.remove(ng)
                    except Exception as exc:
                        utils.note_swallowed("shader_tree.cls.ensure_node_groups_exist", exc)
                        ng = None
                    else:
                        ng = None
                if ng is None:
                    ng = bpy.data.node_groups.new(group_name, "ShaderNodeTree")
                elif cls._node_group_needs_refresh(ng, node_def, cat_key, type_key):
                    ng.nodes.clear()
                    ng.interface.clear()
                else:
                    cls._ensure_attr_interface_visible(ng, cat_key, type_key)
                    continue
                cls._build_group_content(ng, node_def, cat_key, type_key)
        with utils.logged_errors("shader_tree.cls.ensure_node_groups_exist#2"):
            from ..dynamic_audio_operators import ACESplitterManager, ACEAudioMathManager

            ACESplitterManager.refresh_existing_variants("ShaderNodeTree")
            ACEAudioMathManager.refresh_existing_variants("ShaderNodeTree")
        if current_hash:
            from .reader import ACEShaderTreeReader

            cls._NODES_CONTENT_HASH = current_hash
            ACEShaderTreeReader._GROUP_NAME_LOOKUP.clear()
            ACEShaderTreeReader._ACE_NODES_LOOKUP_HASH = ""
def register():
    try:
        ACEShaderNodeFactory.ensure_node_groups_exist()
    except AttributeError:
        utils.log("Skipping shader node group creation due to restricted data access.", "WARNING")
def unregister():
    pass
