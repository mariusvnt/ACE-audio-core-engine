"""Blender Shader node group management for ACE frontends."""

import hashlib
import json

import bpy  # type: ignore

from ..vars import ACE_NODES
from .. import utils
from ..core.id_management import ACEIdManager
from .dummy_tree import ACEDummyTreeReader, FRONTEND_KIND_SHADER, ACEDummyNodeFactory


class ACEShaderNodeFactory:
    """Create hidden ShaderNodeTree ACE groups for shader frontends."""

    _NODES_CONTENT_HASH: str = ""
    _ATTR_NODE_OUTPUT_IDX_KEY = "ace_shader_output_idx"
    _OUTPUT_NODE_KIND_KEY = "ace_shader_sink_kind"
    _HELPER_NODE_OWNER_KEY = "ace_shader_helper_owner"
    _HELPER_ATTR_NAME_KEY = "ace_shader_attr_name"
    _HELPER_INPUT_PREFIX = "ACE Property"
    _HELPER_FALLBACK_OWNER_HEIGHT = 128.0
    _HELPER_VERTICAL_GAP = 18.0
    _HELPER_VERTICAL_STEP = 62.0
    _HELPER_HIDDEN_WIDTH = 150.0

    @staticmethod
    def get_group_name(category: str, node_type: str) -> str:
        return f".ACE_SHADER_{category}_{node_type}"

    @staticmethod
    def _ace_nodes_hash() -> str:
        try:
            serialized = json.dumps(
                {"kind": "shader", "nodes": ACE_NODES},
                sort_keys=True,
                default=str,
            ).encode("utf-8")
            return hashlib.blake2b(serialized, digest_size=8).hexdigest()
        except Exception:
            return ""

    @staticmethod
    def _shader_socket_type(sock_type: str) -> str:
        mapping = {
            # Keep bundle sockets native in shader groups (Blender 5.0+),
            # only remap types that are unreliable in shader interfaces.
            "NodeSocketInt": "NodeSocketFloat",
        }
        return mapping.get(str(sock_type or ""), str(sock_type or "NodeSocketFloat"))

    @classmethod
    def _new_socket(cls, ng, sock_name: str, in_out: str, socket_type: str):
        requested = str(socket_type or "NodeSocketFloat")
        mapped = cls._shader_socket_type(requested)
        candidates: list[str] = []
        for candidate in (requested, mapped):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        # Do not silently downgrade string/bundle interfaces to float.
        needs_strict_type = any(token in mapped for token in ("String", "Bundle"))
        if (not needs_strict_type) and "NodeSocketFloat" not in candidates:
            candidates.append("NodeSocketFloat")
        for candidate in candidates:
            try:
                return ng.interface.new_socket(sock_name, in_out=in_out, socket_type=candidate)
            except Exception:
                continue
        return None

    @staticmethod
    def _extra_inputs(category: str, node_type: str) -> tuple[tuple[str, str, str], ...]:
        if category == "stream":
            stream_nodes = (
                ACE_NODES.get("stream", {}).get("nodes", {})
                if isinstance(ACE_NODES.get("stream", {}), dict)
                else {}
            )
            node_def = stream_nodes.get(str(node_type or ""), {}) if isinstance(stream_nodes, dict) else {}
            outputs = list(node_def.get("outputs", [])) if isinstance(node_def, dict) else []
            value_output_count = max(
                1,
                sum(1 for sock_type, _sock_name in outputs if "Bundle" not in str(sock_type or "")),
            )
            specs = [("NodeSocketString", "Attribute ID", "ACE_undefined")]
            for idx in range(1, int(value_output_count)):
                specs.append(
                    (
                        "NodeSocketString",
                        f"Attribute ID {int(idx + 1)}",
                        f"ACE_undefined_{int(idx + 1)}",
                    )
                )
            return tuple(specs)
        if category == "source_mngmt" and node_type == "src":
            return (("NodeSocketString", "Proxy Object", ""),)
        if category == "utilities" and node_type == "src_specs":
            return (
                ("NodeSocketString", "Frames per Beat Attribute ID", "ACE_undefined_frames_per_beat"),
                ("NodeSocketString", "Beat Phase Attribute ID", "ACE_undefined_beatphase"),
                ("NodeSocketString", "Proxy Object", ""),
            )
        if category == "utilities" and node_type == "util_rolling_avg":
            return (
                ("NodeSocketString", "Attribute ID", "ACE_undefined"),
                ("NodeSocketString", "Proxy Object", ""),
            )
        if category == "utilities" and node_type == "stream_accumulator":
            return (
                ("NodeSocketString", "Attribute ID", "ACE_undefined"),
                ("NodeSocketString", "Proxy Object", ""),
            )
        if category == "utilities" and node_type == "logic":
            return (
                ("NodeSocketString", "Attribute ID", "ACE_undefined"),
                ("NodeSocketString", "Proxy Object", ""),
            )
        return ()

    @staticmethod
    def _attr_socket_names(category: str, node_type: str) -> tuple[str, ...]:
        if category == "stream":
            return tuple(
                str(sock_name)
                for _sock_type, sock_name, _default in ACEShaderNodeFactory._extra_inputs(category, node_type)
            )
        return ACEDummyNodeFactory._attr_socket_names(category, node_type)

    @staticmethod
    def _is_property_output_node(category: str, node_type: str) -> bool:
        cat = str(category or "")
        typ = str(node_type or "")
        return bool(
            cat == "stream"
            or (cat == "utilities" and typ in {"src_specs", "util_rolling_avg", "util_smooth_operator", "stream_accumulator", "logic"})
        )

    @staticmethod
    def _expected_attribute_output_indices(node_def) -> tuple[int, ...]:
        indices: list[int] = []
        for idx, (sock_type, _sock_name) in enumerate(node_def.get("outputs", [])):
            if "Bundle" in str(sock_type or ""):
                continue
            indices.append(int(idx))
        return tuple(indices)

    @classmethod
    def _helper_input_name(cls, output_idx: int) -> str:
        if int(output_idx) <= 0:
            return cls._HELPER_INPUT_PREFIX
        return f"{cls._HELPER_INPUT_PREFIX} {int(output_idx + 1)}"

    @classmethod
    def _helper_input_specs(cls, node_def) -> tuple[tuple[int, str, str], ...]:
        specs: list[tuple[int, str, str]] = []
        for output_idx, (sock_type, _sock_name) in enumerate(node_def.get("outputs", [])):
            if "Bundle" in str(sock_type or ""):
                continue
            specs.append((int(output_idx), str(sock_type or "NodeSocketFloat"), cls._helper_input_name(int(output_idx))))
        return tuple(specs)

    @classmethod
    def _expected_interface_names(cls, node_def, category: str, node_type: str) -> tuple[set[str], set[str]]:
        input_names = {str(name) for _sock_type, name in node_def.get("inputs", [])}
        output_names = {str(name) for _sock_type, name in node_def.get("outputs", [])}
        for _sock_type, extra_name, _default_value in cls._extra_inputs(category, node_type):
            input_names.add(str(extra_name))
        for _output_idx, _sock_type, helper_name in cls._helper_input_specs(node_def):
            input_names.add(str(helper_name))
        return input_names, output_names

    @classmethod
    def _expected_interface_types(
        cls,
        node_def,
        category: str,
        node_type: str,
    ) -> tuple[dict[str, str], dict[str, str]]:
        input_types: dict[str, str] = {}
        output_types: dict[str, str] = {}
        for sock_type, sock_name in node_def.get("inputs", []):
            input_types[str(sock_name)] = cls._shader_socket_type(str(sock_type or "NodeSocketFloat"))
        for sock_type, sock_name in node_def.get("outputs", []):
            output_types[str(sock_name)] = cls._shader_socket_type(str(sock_type or "NodeSocketFloat"))
        for sock_type, sock_name, _default_value in cls._extra_inputs(category, node_type):
            input_types[str(sock_name)] = cls._shader_socket_type(str(sock_type or "NodeSocketFloat"))
        for _output_idx, sock_type, helper_name in cls._helper_input_specs(node_def):
            input_types[str(helper_name)] = cls._shader_socket_type(str(sock_type or "NodeSocketFloat"))
        return input_types, output_types

    @staticmethod
    def _interface_socket_type(item) -> str:
        for attr_name in ("socket_type", "bl_socket_idname", "type"):
            try:
                raw = getattr(item, attr_name, "")
            except Exception:
                raw = ""
            token = str(raw or "")
            if token:
                return token
        return ""

    @staticmethod
    def _socket_type_matches(expected: str, actual: str) -> bool:
        exp = str(expected or "").lower()
        act = str(actual or "").lower()
        if not exp:
            return True
        if not act:
            return False
        if "bundle" in exp:
            return "bundle" in act
        if "string" in exp:
            return "string" in act
        if "vector" in exp:
            return "vector" in act
        if "color" in exp or "rgba" in exp:
            return ("color" in act) or ("rgba" in act)
        if "float" in exp:
            return ("float" in act) or ("value" in act)
        if "int" in exp:
            return ("int" in act) or ("float" in act) or ("value" in act)
        return exp == act

    @classmethod
    def _node_group_needs_refresh(cls, ng, node_def, category: str, node_type: str) -> bool:
        expected_inputs, expected_outputs = cls._expected_interface_names(node_def, category, node_type)
        expected_input_types, expected_output_types = cls._expected_interface_types(node_def, category, node_type)
        try:
            inputs = {
                str(item.name): cls._interface_socket_type(item)
                for item in ng.interface.items_tree
                if getattr(item, "in_out", None) == "INPUT"
            }
            outputs = {
                str(item.name): cls._interface_socket_type(item)
                for item in ng.interface.items_tree
                if getattr(item, "in_out", None) == "OUTPUT"
            }
        except Exception:
            return True
        if not expected_inputs.issubset(set(inputs.keys())) or not expected_outputs.issubset(set(outputs.keys())):
            return True
        for sock_name, expected_type in expected_input_types.items():
            if not cls._socket_type_matches(expected_type, inputs.get(str(sock_name), "")):
                return True
        for sock_name, expected_type in expected_output_types.items():
            if not cls._socket_type_matches(expected_type, outputs.get(str(sock_name), "")):
                return True
        try:
            has_input = any(getattr(node, "type", "") == "GROUP_INPUT" for node in ng.nodes)
            has_output = any(getattr(node, "type", "") == "GROUP_OUTPUT" for node in ng.nodes)
            if not has_input or not has_output:
                return True
            for node in ng.nodes:
                _ = node
            # Detect wrong wiring: a bundle-type socket feeding a non-bundle output.
            for node in ng.nodes:
                if str(getattr(node, "type", "") or "") != "GROUP_OUTPUT":
                    continue
                for inp in getattr(node, "inputs", []) or []:
                    inp_type = str(getattr(inp, "type", "") or "").upper()
                    if inp_type in ("BUNDLE", "CUSTOM", ""):
                        continue
                    for link in getattr(inp, "links", []) or []:
                        from_sock = getattr(link, "from_socket", None)
                        if from_sock is None:
                            continue
                        from_type = str(getattr(from_sock, "type", "") or "").upper()
                        if from_type in ("BUNDLE", "CUSTOM"):
                            return True

            # Detect missing wiring: non-bundle GROUP_OUTPUT inputs with no links.
            # For stream / utility property-output nodes every non-bundle output
            # must be driven by a helper (ACE Property) GroupInput socket.
            # A missing link means the old build failed silently and the node
            # will output zero even when a source is connected.
            is_property_node = cls._is_property_output_node(category, node_type)
            if is_property_node:
                for node in ng.nodes:
                    if str(getattr(node, "type", "") or "") != "GROUP_OUTPUT":
                        continue
                    for inp in getattr(node, "inputs", []) or []:
                        inp_type = str(getattr(inp, "type", "") or "").upper()
                        if inp_type in ("BUNDLE", "CUSTOM", ""):
                            continue
                        if not list(getattr(inp, "links", []) or []):
                            return True
        except Exception:
            return True
        return False

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
                except Exception:
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
            except Exception:
                current_name = ""
            if current_name == name:
                continue
            try:
                setattr(attr_node, prop_name, name)
                changed = True
            except Exception:
                pass
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
        except Exception:
            return None
        sink_kind = "ATTRIBUTE"
        cls._set_attr_node_defaults(
            sink_node,
            cls._placeholder_attr_name(output_idx, socket_name),
        )
        out_sock = cls._shader_attr_output_socket(sink_node, socket_type)

        sink_node.location = (300, -200 - (int(output_idx) * 150))
        try:
            sink_node[cls._ATTR_NODE_OUTPUT_IDX_KEY] = int(output_idx)
        except Exception:
            pass
        try:
            sink_node[cls._OUTPUT_NODE_KIND_KEY] = str(sink_kind)
        except Exception:
            pass
        try:
            sink_node["ace_shader_output_name"] = str(socket_name or "")
        except Exception:
            pass
        if out_sock is not None:
            try:
                ng.links.new(out_sock, output_node.inputs[int(output_idx)])
            except Exception:
                pass
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
        except Exception:
            item = None
        if item is None:
            return
        if not hasattr(item, "default_value"):
            return
        try:
            item.default_value = default_value
        except Exception:
            pass

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
                try:
                    item.hide_value = False
                except Exception:
                    pass

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

    @staticmethod
    def _safe_bundle_item(bundle_node, item_name: str, candidates: tuple[str, ...]):
        for item_type in candidates:
            try:
                return bundle_node.bundle_items.new(item_type, item_name)
            except Exception:
                continue
        return None

    @staticmethod
    def _link_or_consume_reroute(ng, source_socket, location_x: int, location_y: int):
        try:
            reroute = ng.nodes.new("NodeReroute")
            reroute.location = (int(location_x), int(location_y))
            ng.links.new(source_socket, reroute.inputs[0])
            return True
        except Exception:
            return False

    @classmethod
    def _build_group_content(cls, ng, node_def, category: str, node_type: str):
        base_inputs = list(node_def.get("inputs", []))
        base_outputs = list(node_def.get("outputs", []))
        extra_inputs = list(cls._extra_inputs(category, node_type))
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
                    try:
                        item.default_value = 0.4
                    except Exception:
                        pass
                    for attr_name, value in (
                        ("min_value", 0.05),
                        ("max_value", 3.0),
                        ("soft_min", 0.05),
                        ("soft_max", 3.0),
                    ):
                        try:
                            setattr(item, attr_name, value)
                        except Exception:
                            pass
                elif name == "Overlap Ratio":
                    try:
                        item.default_value = 0.25
                    except Exception:
                        pass
                    for attr_name, value in (
                        ("min_value", 0.0),
                        ("max_value", 1.0),
                        ("soft_min", 0.0),
                        ("soft_max", 1.0),
                    ):
                        try:
                            setattr(item, attr_name, value)
                        except Exception:
                            pass

        for sock_type_str, sock_name, default_value in extra_inputs:
            cls._new_socket(ng, sock_name, "INPUT", sock_type_str)
            cls._set_socket_default(ng, sock_name, default_value)
        for _out_idx, sock_type_str, helper_name in helper_inputs:
            cls._new_socket(ng, helper_name, "INPUT", sock_type_str)

        # Shader groups are still facades, but they now actively consume audio
        # bundle inputs so sockets stay live in the editor without helper frames.
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

        bundle_input_names = [
            str(_sn) for _st, _sn in all_inputs
            if "Bundle" in str(_st or "")
        ]
        first_bundle_name = bundle_input_names[0] if bundle_input_names else None
        first_bundle_socket = input_sock_by_name.get(first_bundle_name) if first_bundle_name else None

        scalar_inputs: list[tuple[str, object]] = []
        passthrough_inputs: list[tuple[int, object]] = []
        for _ci, (socket_type, socket_name) in enumerate(all_inputs):
            _s = input_sock_by_name.get(str(socket_name))
            if _s is None:
                continue
            if str(socket_name) in helper_socket_names:
                continue
            socket_type_str = str(socket_type or "")
            if "Bundle" in socket_type_str:
                passthrough_inputs.append((_ci, _s))
                continue
            if any(token in socket_type_str for token in ("Float", "Int", "Value")):
                scalar_inputs.append((str(socket_name), _s))
                continue
            cls._link_or_consume_reroute(ng, _s, -140, -160 - (_ci * 90))

        neutral_scalar = None
        if scalar_inputs:
            add_out = None
            for input_idx, (_socket_name, source_socket) in enumerate(scalar_inputs):
                if add_out is None:
                    add_out = source_socket
                    continue
                add_node = ng.nodes.new("ShaderNodeMath")
                add_node.operation = "ADD"
                add_node.location = (-360 + (input_idx * 180), -320)
                ng.links.new(add_out, add_node.inputs[0])
                ng.links.new(source_socket, add_node.inputs[1])
                add_out = add_node.outputs[0]
            mul_zero = ng.nodes.new("ShaderNodeMath")
            mul_zero.operation = "MULTIPLY"
            mul_zero.inputs[1].default_value = 0.0
            mul_zero.location = (40, -320)
            ng.links.new(add_out, mul_zero.inputs[0])
            neutral_scalar = mul_zero.outputs[0]

        bundle_zero_scalar = None
        if first_bundle_socket is not None:
            try:
                separate_bundle = ng.nodes.new("NodeSeparateBundle")
                separate_bundle.location = (-500, 60)
                zero_item = cls._safe_bundle_item(separate_bundle, "ACE Zero", ("FLOAT", "VALUE"))
                ng.links.new(first_bundle_socket, separate_bundle.inputs[0])
                if zero_item is not None:
                    try:
                        bundle_zero_scalar = separate_bundle.outputs[zero_item.name]
                    except Exception:
                        outputs = list(getattr(separate_bundle, "outputs", []) or [])
                        bundle_zero_scalar = outputs[-1] if outputs else None
            except Exception:
                bundle_zero_scalar = None

        combined_neutral = neutral_scalar or bundle_zero_scalar
        if neutral_scalar is not None and bundle_zero_scalar is not None:
            neutral_add = ng.nodes.new("ShaderNodeMath")
            neutral_add.operation = "ADD"
            neutral_add.location = (220, -320)
            ng.links.new(neutral_scalar, neutral_add.inputs[0])
            ng.links.new(bundle_zero_scalar, neutral_add.inputs[1])
            combined_neutral = neutral_add.outputs[0]

        if str(category or "") == "source_mngmt" and str(node_type or "") == "src":
            zero_value = ng.nodes.new("ShaderNodeValue")
            zero_value.location = (-420, 40)
            try:
                zero_value.outputs[0].default_value = 0.0
            except Exception:
                pass
            combine_bundle = ng.nodes.new("NodeCombineBundle")
            combine_bundle.location = (-120, 40)
            zero_item = cls._safe_bundle_item(combine_bundle, "ACE Zero", ("FLOAT", "VALUE"))
            if zero_item is not None:
                try:
                    ng.links.new(zero_value.outputs[0], combine_bundle.inputs[zero_item.name])
                except Exception:
                    ng.links.new(zero_value.outputs[0], combine_bundle.inputs[0])
            for output_idx in bundle_output_indices:
                if output_idx >= len(getattr(output_node, "inputs", [])):
                    continue
                try:
                    ng.links.new(combine_bundle.outputs[0], output_node.inputs[int(output_idx)])
                except Exception:
                    pass
        elif str(category or "") == "utilities" and str(node_type or "") == "audio_math":
            combine_bundle = ng.nodes.new("NodeCombineBundle")
            combine_bundle.location = (-80, 40)
            item_count = 0
            for input_idx, bundle_socket in passthrough_inputs:
                item = cls._safe_bundle_item(combine_bundle, f"Audio {int(item_count + 1)}", ("BUNDLE", "FLOAT", "VALUE"))
                if item is None:
                    continue
                try:
                    ng.links.new(bundle_socket, combine_bundle.inputs[item.name])
                except Exception:
                    try:
                        ng.links.new(bundle_socket, combine_bundle.inputs[item_count])
                    except Exception:
                        continue
                item_count += 1
            for output_idx in bundle_output_indices:
                if output_idx >= len(getattr(output_node, "inputs", [])):
                    continue
                try:
                    ng.links.new(combine_bundle.outputs[0], output_node.inputs[int(output_idx)])
                except Exception:
                    pass
        elif first_bundle_socket is not None:
            for output_idx in bundle_output_indices:
                if output_idx >= len(getattr(output_node, "inputs", [])):
                    continue
                try:
                    ng.links.new(first_bundle_socket, output_node.inputs[int(output_idx)])
                except Exception:
                    pass

        if not bundle_output_indices and first_bundle_socket is not None and str(category or "") == "utilities" and str(node_type or "") == "preview_sink":
            cls._link_or_consume_reroute(ng, first_bundle_socket, -120, -60)

        for output_idx, (socket_type, _socket_name) in enumerate(base_outputs):
            if "Bundle" in str(socket_type or ""):
                continue
            if output_idx >= len(getattr(output_node, "inputs", [])):
                continue
            source_out = helper_source_by_idx.get(int(output_idx))
            if source_out is None:
                continue
            target_socket = output_node.inputs[int(output_idx)]
            if combined_neutral is None:
                try:
                    ng.links.new(source_out, target_socket)
                except Exception:
                    pass
                continue
            add_node = ng.nodes.new("ShaderNodeMath")
            add_node.operation = "ADD"
            add_node.location = (420, 180 - (int(output_idx) * 150))
            try:
                ng.links.new(source_out, add_node.inputs[0])
                ng.links.new(combined_neutral, add_node.inputs[1])
                ng.links.new(add_node.outputs[0], target_socket)
            except Exception:
                pass

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
        if cat == "utilities" and typ in {"util_rolling_avg", "util_smooth_operator", "stream_accumulator", "logic"}:
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
            try:
                node.node_tree = group
                changed = True
            except Exception:
                pass
        return group, changed

    @classmethod
    def _helper_node_name(cls, ace_id: str, output_idx: int) -> str:
        return f".ACE_HELPER_{ACEIdManager._clean_string(str(ace_id or 'node'))[:24]}_{int(output_idx)}"

    @staticmethod
    def _coerce_xy(raw_value) -> tuple[float, float]:
        if raw_value is None:
            return 0.0, 0.0
        try:
            return (
                float(getattr(raw_value, "x")),
                float(getattr(raw_value, "y")),
            )
        except Exception:
            pass
        try:
            return float(raw_value[0]), float(raw_value[1])
        except Exception:
            return 0.0, 0.0

    @classmethod
    def _node_local_location(cls, node) -> tuple[float, float]:
        return cls._coerce_xy(getattr(node, "location", None))

    @classmethod
    def _node_dimensions(cls, node) -> tuple[float, float]:
        return cls._coerce_xy(getattr(node, "dimensions", None))

    @classmethod
    def _node_height(cls, node, fallback: float = 0.0) -> float:
        _width, height = cls._node_dimensions(node)
        resolved = abs(float(height))
        if resolved > 1.0:
            return resolved
        return float(fallback)

    @classmethod
    def _node_absolute_location(cls, node) -> tuple[float, float]:
        if node is None:
            return 0.0, 0.0
        abs_loc = getattr(node, "location_absolute", None)
        if abs_loc is not None:
            abs_x, abs_y = cls._coerce_xy(abs_loc)
            if abs_x or abs_y or getattr(node, "parent", None) is not None:
                return abs_x, abs_y
        x, y = cls._node_local_location(node)
        visited: set[int] = set()
        parent = getattr(node, "parent", None)
        while parent is not None:
            ptr = id(parent)
            if ptr in visited:
                break
            visited.add(ptr)
            parent_x, parent_y = cls._node_local_location(parent)
            x += parent_x
            y += parent_y
            parent = getattr(parent, "parent", None)
        return x, y

    @classmethod
    def _set_node_location(cls, node, location: tuple[float, float]) -> bool:
        if node is None:
            return False
        target_x, target_y = float(location[0]), float(location[1])
        current_x, current_y = cls._node_local_location(node)
        if abs(current_x - target_x) <= 0.5 and abs(current_y - target_y) <= 0.5:
            return False
        try:
            node.location = (target_x, target_y)
            return True
        except Exception:
            return False

    @staticmethod
    def _is_frame_node(node) -> bool:
        if node is None:
            return False
        return bool(
            str(getattr(node, "bl_idname", "") or "") == "NodeFrame"
            or str(getattr(node, "type", "") or "") == "FRAME"
        )

    @classmethod
    def _ensure_helper_node_compact_display(cls, helper_node) -> bool:
        if helper_node is None:
            return False
        changed = False
        if hasattr(helper_node, "hide"):
            try:
                if not bool(getattr(helper_node, "hide", False)):
                    helper_node.hide = True
                    changed = True
            except Exception:
                pass
        if hasattr(helper_node, "width_hidden"):
            try:
                current = float(getattr(helper_node, "width_hidden", 0.0) or 0.0)
                target = float(cls._HELPER_HIDDEN_WIDTH)
                if abs(current - target) > 0.5:
                    helper_node.width_hidden = target
                    changed = True
            except Exception:
                pass
        return changed

    @classmethod
    def _parent_node_preserve_absolute(cls, node, parent) -> bool:
        if node is None:
            return False
        if getattr(node, "parent", None) == parent:
            return False
        abs_x, abs_y = cls._node_absolute_location(node)
        try:
            node.parent = parent
        except Exception:
            return False
        if parent is None:
            cls._set_node_location(node, (abs_x, abs_y))
            return True
        parent_abs_x, parent_abs_y = cls._node_absolute_location(parent)
        cls._set_node_location(node, (abs_x - parent_abs_x, abs_y - parent_abs_y))
        return True

    @classmethod
    def _sync_helper_node_layout(cls, tree, owner_node, helper_node, ace_id: str, output_idx: int) -> bool:
        if tree is None or owner_node is None or helper_node is None:
            return False
        changed = False
        container = getattr(owner_node, "parent", None)
        if cls._is_frame_node(container):
            if cls._parent_node_preserve_absolute(owner_node, None):
                changed = True
            container = getattr(owner_node, "parent", None)

        if cls._parent_node_preserve_absolute(helper_node, container):
            changed = True

        if cls._ensure_helper_node_compact_display(helper_node):
            changed = True

        if container is not None and getattr(owner_node, "parent", None) == container:
            owner_x, owner_y = cls._node_local_location(owner_node)
        else:
            owner_x, owner_y = cls._node_absolute_location(owner_node)
        owner_height = cls._node_height(owner_node, cls._HELPER_FALLBACK_OWNER_HEIGHT)

        helper_target = (
            float(owner_x),
            float(owner_y) - owner_height - cls._HELPER_VERTICAL_GAP - (int(output_idx) * cls._HELPER_VERTICAL_STEP),
        )
        if cls._set_node_location(helper_node, helper_target):
            changed = True
        return changed

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
            except Exception:
                owner = ""
            matched_by_owner = bool(target_ace_id and owner == target_ace_id)
            if not matched_by_owner:
                # Fallback: match by deterministic node name.  This handles nodes
                # whose owner key was written with a stale id or was never set.
                node_name = str(getattr(helper_node, "name", "") or "")
                if node_name not in expected_name_to_idx:
                    continue
            try:
                out_idx = int(helper_node.get(cls._ATTR_NODE_OUTPUT_IDX_KEY, -1) or -1)
            except Exception:
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
        created = False
        if helper_node is None:
            try:
                helper_node = tree.nodes.new("ShaderNodeAttribute")
            except Exception:
                return None, False
            changed = True
            created = True
        try:
            if str(helper_node.get(cls._HELPER_NODE_OWNER_KEY, "") or "") != str(ace_id or ""):
                helper_node[cls._HELPER_NODE_OWNER_KEY] = str(ace_id or "")
                changed = True
        except Exception:
            pass
        try:
            if int(helper_node.get(cls._ATTR_NODE_OUTPUT_IDX_KEY, -1) or -1) != int(output_idx):
                helper_node[cls._ATTR_NODE_OUTPUT_IDX_KEY] = int(output_idx)
                changed = True
        except Exception:
            pass
        try:
            if str(helper_node.get(cls._OUTPUT_NODE_KIND_KEY, "") or "") != "TOP_LEVEL_ATTRIBUTE":
                helper_node[cls._OUTPUT_NODE_KIND_KEY] = "TOP_LEVEL_ATTRIBUTE"
                changed = True
        except Exception:
            pass
        helper_name = cls._helper_node_name(ace_id, output_idx)
        try:
            if str(getattr(helper_node, "name", "") or "") != helper_name:
                helper_node.name = helper_name
                changed = True
        except Exception:
            pass
        try:
            label = f"ACE {str(output_name or f'Out {int(output_idx + 1)}')}"
            if str(getattr(helper_node, "label", "") or "") != label:
                helper_node.label = label
                changed = True
        except Exception:
            pass
        if cls._ensure_helper_node_compact_display(helper_node):
            changed = True
        try:
            if location is not None:
                helper_node.location = (float(location[0]), float(location[1]))
        except Exception:
            pass
        return helper_node, changed

    @classmethod
    def _hide_helper_inputs(cls, node, node_def) -> bool:
        changed = False
        helper_names = {helper_name for _out_idx, _sock_type, helper_name in cls._helper_input_specs(node_def)}
        for sock in getattr(node, "inputs", []) or []:
            if str(getattr(sock, "name", "") or "") not in helper_names:
                continue
            if hasattr(sock, "hide"):
                try:
                    if not bool(getattr(sock, "hide", False)):
                        sock.hide = True
                        changed = True
                except Exception:
                    pass
            if hasattr(sock, "hide_value"):
                try:
                    if not bool(getattr(sock, "hide_value", False)):
                        sock.hide_value = True
                        changed = True
                except Exception:
                    pass
        return changed

    @classmethod
    def _ensure_helper_link(cls, tree, helper_node, helper_output, group_node, target_input) -> bool:
        if helper_node is None or helper_output is None or target_input is None:
            return False
        changed = False
        for link in list(getattr(target_input, "links", []) or []):
            if getattr(link, "from_node", None) == helper_node and getattr(link, "from_socket", None) == helper_output:
                return changed
            try:
                tree.links.remove(link)
                changed = True
            except Exception:
                pass
        try:
            tree.links.new(helper_output, target_input)
            changed = True
        except Exception:
            pass
        return changed

    @classmethod
    def _store_helper_attr_name(cls, helper_node, attr_name: str) -> bool:
        if helper_node is None:
            return False
        target_name = str(attr_name or "")
        try:
            current_name = str(helper_node.get(cls._HELPER_ATTR_NAME_KEY, "") or "")
        except Exception:
            current_name = ""
        if current_name == target_name:
            return False
        try:
            helper_node[cls._HELPER_ATTR_NAME_KEY] = target_name
            return True
        except Exception:
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
        except Exception:
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
        except Exception:
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
            try:
                tree.links.new(source_socket, target_socket)
                changed = True
            except Exception:
                pass
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
            except Exception:
                ace_id = ""
            if ace_id:
                live_owner_ids.add(ace_id)
                # Pre-compute the expected helper names for this live node so we can
                # catch orphans whose owner key is gone but whose name still matches
                # the pattern (e.g. left over from a previous ace_id scheme).
                for out_idx in range(16):
                    live_node_names.add(cls._helper_node_name(ace_id, out_idx))

        changed = False
        for helper_node in list(getattr(tree, "nodes", []) or []):
            bl_idname = str(getattr(helper_node, "bl_idname", "") or "")
            owner_id = ""
            if bl_idname == "ShaderNodeAttribute":
                try:
                    owner_id = str(helper_node.get(cls._HELPER_NODE_OWNER_KEY, "") or "")
                except Exception:
                    owner_id = ""
            elif cls._is_frame_node(helper_node):
                for child in list(getattr(tree, "nodes", []) or []):
                    if getattr(child, "parent", None) != helper_node:
                        continue
                    if cls._parent_node_preserve_absolute(child, None):
                        changed = True
                try:
                    tree.nodes.remove(helper_node)
                    changed = True
                except Exception:
                    pass
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
            try:
                tree.nodes.remove(helper_node)
                changed = True
            except Exception:
                pass
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
            except Exception:
                scene = None
        try:
            from ..core.audio_engine import ACEAudioEngine
            if not ACEAudioEngine._structural_tree_writes_allowed(
                scene,
                tree=getattr(node, "id_data", None),
                frontend_kind=FRONTEND_KIND_SHADER,
            ):
                return False
        except Exception:
            pass
        try:
            ace_id = str(node.get("ace_id", "") or "")
        except Exception:
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
            except Exception:
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
            try:
                tree.nodes.remove(stale_node)
                changed = True
            except Exception:
                pass
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
            except Exception:
                ng = None
            else:
                ng = None
        if ng is None:
            ng = bpy.data.node_groups.new(group_name, "ShaderNodeTree")
        else:
            ng.clear()
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
                try:
                    drivers.remove(fc)
                except Exception:
                    pass

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
                    except Exception:
                        ng = None
                    else:
                        ng = None
                if ng is None:
                    ng = bpy.data.node_groups.new(group_name, "ShaderNodeTree")
                elif cls._node_group_needs_refresh(ng, node_def, cat_key, type_key):
                    ng.clear()
                    ng.interface.clear()
                else:
                    cls._ensure_attr_interface_visible(ng, cat_key, type_key)
                    continue
                cls._build_group_content(ng, node_def, cat_key, type_key)
        try:
            from .dynamic_audio_operators import ACESplitterManager, ACEAudioMathManager

            ACESplitterManager.refresh_existing_variants("ShaderNodeTree")
            ACEAudioMathManager.refresh_existing_variants("ShaderNodeTree")
        except Exception:
            pass
        if current_hash:
            cls._NODES_CONTENT_HASH = current_hash
            ACEShaderTreeReader._GROUP_NAME_LOOKUP.clear()
            ACEShaderTreeReader._ACE_NODES_LOOKUP_HASH = ""


class ACEShaderTreeReader(ACEDummyTreeReader):
    """Shader frontend reader sharing the GN reader contract."""

    FACTORY_CLS = ACEShaderNodeFactory
    _GROUP_NAME_LOOKUP: dict[str, tuple[str, str]] = {}
    _ACE_NODES_LOOKUP_HASH: str = ""

    def __init__(
        self,
        blender_node_tree,
        frontend_uid: str = "",
        frontend_kind: str = FRONTEND_KIND_SHADER,
        reserved_node_ids: set[str] | None = None,
        consumer_keys: tuple[str, ...] = (),
    ):
        super().__init__(
            blender_node_tree,
            frontend_uid=frontend_uid,
            frontend_kind=frontend_kind,
            reserved_node_ids=reserved_node_ids,
            consumer_keys=consumer_keys,
        )

    @staticmethod
    def is_managed_tree(tree) -> bool:
        """Return True when *tree* is a shader frontend containing ACE nodes."""
        if tree is None or str(getattr(tree, "bl_idname", "") or "") != "ShaderNodeTree":
            return False
        try:
            if bool(tree.get("ace_frontend_uid", "")):
                return True
        except Exception:
            pass
        for node in getattr(tree, "nodes", []) or []:
            try:
                if bool(node.get("ace_id", "")):
                    return True
            except Exception:
                pass
            try:
                if bool(node.get("ace_source_id", "")):
                    return True
            except Exception:
                pass
            if str(getattr(node, "type", "") or "") != "GROUP":
                continue
            group = getattr(node, "node_tree", None)
            group_name = str(getattr(group, "name", "") or "")
            norm_group_name = group_name.lstrip(".")
            if (
                group_name.startswith(".ACE_SHADER_")
                or norm_group_name.startswith("ACE_SHADER_")
                or group_name.startswith(".ACE_")
                or norm_group_name.startswith("ACE_")
            ):
                return True
        return False


def register():
    try:
        ACEShaderNodeFactory.ensure_node_groups_exist()
    except AttributeError:
        utils.log("Skipping shader node group creation due to restricted data access.", "WARNING")


def unregister():
    pass
