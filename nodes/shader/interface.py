"""Shader group interface: sockets ACE nodes expose and the checks that trigger a rebuild."""

from ...vars import ACE_NODES
from ... import utils
from ..geometry.factory import ACEDummyNodeFactory
from ..identity import VALUE_NODE_KINDS


class ShaderGroupInterface:
    @staticmethod
    def _shader_socket_type(sock_type: str) -> str:
        mapping = {
            # Keep bundle sockets native in shader groups (Blender 5.0+),
            # only remap types that are unreliable in shader interfaces.
            "NodeSocketInt": "NodeSocketFloat",
        }
        return mapping.get(str(sock_type or ""), str(sock_type or "NodeSocketFloat"))

    @staticmethod
    def _shader_interface_supports(socket_type: str) -> bool:
        return not any(token in str(socket_type or "") for token in ("String", "Object"))

    @classmethod
    def _new_socket(cls, ng, sock_name: str, in_out: str, socket_type: str):
        requested = str(socket_type or "NodeSocketFloat")
        mapped = cls._shader_socket_type(requested)
        candidates: list[str] = []
        for candidate in (mapped, requested):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        # Do not silently downgrade string/bundle interfaces to float.
        needs_strict_type = any(token in mapped for token in ("String", "Bundle"))
        if (not needs_strict_type) and "NodeSocketFloat" not in candidates:
            candidates.append("NodeSocketFloat")
        last_error = None
        for candidate in candidates:
            try:
                return ng.interface.new_socket(sock_name, in_out=in_out, socket_type=candidate)
            except TypeError as exc:  # this tree type rejects the socket type; try the next
                last_error = exc
        utils.note_swallowed("interface.ShaderGroupInterface._new_socket", last_error or TypeError(sock_name))
        return None

    @staticmethod
    def _numeric_control_names(category: str, node_type: str) -> tuple[str, ...]:
        """Numeric control inputs of a stream node (e.g. Window), in declaration order."""
        if str(category or "") != "stream":
            return ()
        node_def = ACE_NODES.get("stream", {}).get("nodes", {}).get(str(node_type or ""), {})
        names = []
        for idx, (sock_type, sock_name) in enumerate(list(node_def.get("inputs", []) or [])):
            st = str(sock_type or "")
            if idx == 0 or any(token in st for token in ("Bundle", "String", "Object")):
                continue
            names.append(str(sock_name))
        return tuple(names)

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
        if (category, node_type) in VALUE_NODE_KINDS:
            return (("NodeSocketString", "Attribute ID", "ACE_undefined"),)
        return ()

    @classmethod
    def _attr_socket_names(cls, category: str, node_type: str) -> tuple[str, ...]:
        if category == "stream":
            return tuple(
                str(sock_name)
                for _sock_type, sock_name, _default in cls._extra_inputs(category, node_type)
            )
        return ACEDummyNodeFactory._attr_socket_names(category, node_type)

    @staticmethod
    def _is_property_output_node(category: str, node_type: str) -> bool:
        cat = str(category or "")
        typ = str(node_type or "")
        return bool(
            cat == "stream"
            or (cat == "utilities" and typ in {"util_rolling_avg", "stream_accumulator"})
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
    def _expected_interface_names(cls, node_def, category: str, node_type: str) -> tuple[set[str], set[str]]:
        input_names = {str(name) for _sock_type, name in node_def.get("inputs", [])}
        output_names = {str(name) for _sock_type, name in node_def.get("outputs", [])}
        for sock_type, extra_name, _default_value in cls._extra_inputs(category, node_type):
            if cls._shader_interface_supports(sock_type):
                input_names.add(str(extra_name))
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
            if cls._shader_interface_supports(sock_type):
                input_types[str(sock_name)] = cls._shader_socket_type(str(sock_type or "NodeSocketFloat"))
        return input_types, output_types

    @classmethod
    def _interface_socket_type(cls, item) -> str:
        for attr_name in ("socket_type", "bl_socket_idname", "type"):
            try:
                raw = getattr(item, attr_name, "")
            except Exception as exc:
                utils.note_swallowed("interface.ShaderGroupInterface._interface_socket_type", exc)
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
    def _expected_value_names(cls, ng, category: str, node_type: str) -> dict[int, str]:
        from ..identity import SINK_NAMES_KEY, SOURCE_ID_KEY

        return cls._value_names(
            category, node_type,
            sink_names=[str(v) for v in ng.get(SINK_NAMES_KEY, [])],
            source_id=str(ng.get(SOURCE_ID_KEY, "") or ""),
        )

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
        except Exception as exc:
            utils.note_swallowed("interface.ShaderGroupInterface._node_group_needs_refresh", exc)
            return True
        if not expected_inputs.issubset(set(inputs.keys())) or not expected_outputs.issubset(set(outputs.keys())):
            return True
        if str(category or "") == "stream" and set(inputs.keys()) - expected_inputs:
            # An input the definition no longer declares (e.g. a removed control).
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
            allow_math_nodes = bool(cls._numeric_control_names(category, node_type))
            for node in ng.nodes:
                bl_idname = str(getattr(node, "bl_idname", "") or "")
                node_type_name = str(getattr(node, "type", "") or "")
                # Bundle manipulation nodes crash Cycles' ShaderNodesInliner.
                if bl_idname in {"NodeSeparateBundle", "NodeCombineBundle"}:
                    return True
                if node_type_name in {"SEPARATE_BUNDLE", "COMBINE_BUNDLE"}:
                    return True
                if (not allow_math_nodes) and bl_idname in {"ShaderNodeMath", "ShaderNodeVectorMath"}:
                    return True
                if (not allow_math_nodes) and node_type_name in {"MATH", "VECTOR_MATH"}:
                    return True
            # A bundle socket wired to a non-bundle output means a bad build.
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

            # Values come from scene properties: View Layer Attribute nodes reading the group's names.
            readers = sorted(
                str(getattr(node, "attribute_name", "") or "")
                for node in ng.nodes
                if str(getattr(node, "bl_idname", "") or "") == "ShaderNodeAttribute"
            )
            if readers != sorted(cls._expected_value_names(ng, category, node_type).values()):
                return True
            if any(
                str(getattr(node, "attribute_type", "")) != "VIEW_LAYER"
                for node in ng.nodes
                if str(getattr(node, "bl_idname", "") or "") == "ShaderNodeAttribute"
            ):
                return True
            # Every non-bundle output of a property node must be driven by its value.
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
        except Exception as exc:
            utils.note_swallowed("interface.ShaderGroupInterface._node_group_needs_refresh#2", exc)
            return True
        return False
