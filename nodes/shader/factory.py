"""Shader node groups for ACE nodes: creation, content and refresh.

A shader can't name an attribute at run time, so each shader node that writes
values uses a group of its own, ``.ACE_sink_<name>``, whose Attribute nodes (View
Layer mode) read the scene properties ACE writes. A Source node uses its
source's group, ``.ACE_SHADER_source_<id>``, which reads that source's BPM and
Beats per Bar the same way. Nothing is added to the user's material.
"""

import hashlib
import json
import bpy  # type: ignore
from ...vars import ACE_NODES, SOURCE_KIND, apply_socket_ranges, node_def as registry_node_def
from ... import utils
from ..frontend import remember_group_inputs
from ..identity import KIND_KEY, SINK_NAMES_KEY, SOURCE_ID_KEY, group_kind, node_value_outputs, source_tempo_props
from .interface import ShaderGroupInterface


class ACEShaderNodeFactory(ShaderGroupInterface):
    """Create hidden ShaderNodeTree ACE groups for shader frontends."""

    _NODES_CONTENT_HASH: str = ""
    SINK_GROUP_PREFIX = ".ACE_sink_"
    SOURCE_GROUP_PREFIX = ".ACE_SHADER_source_"

    @staticmethod
    def get_group_name(category: str, node_type: str) -> str:
        return f".ACE_SHADER_{category}_{node_type}"

    @classmethod
    def sink_group_name(cls, names: list[str]) -> str:
        return f"{cls.SINK_GROUP_PREFIX}{names[0] if names else 'unnamed'}"[:63]

    @classmethod
    def source_group_name(cls, source_id: str) -> str:
        return f"{cls.SOURCE_GROUP_PREFIX}{str(source_id).rsplit('_', 1)[-1]}"[:63]

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
            utils.note_swallowed("factory.ACEShaderNodeFactory._ace_nodes_hash", exc)
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

    @staticmethod
    def _new_value_reader(ng, attr_name: str, location):
        """An Attribute node reading the scene property *attr_name* (View Layer mode)."""
        attr = ng.nodes.new("ShaderNodeAttribute")
        attr.attribute_type = "VIEW_LAYER"
        attr.attribute_name = str(attr_name or "")
        attr.location = location
        return attr

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
            utils.note_swallowed("factory.ACEShaderNodeFactory._set_socket_default", exc)
            item = None
        if item is None:
            return
        if not hasattr(item, "default_value"):
            return
        with utils.logged_errors("factory.ACEShaderNodeFactory._set_socket_default#2"):
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
                with utils.logged_errors("factory.ACEShaderNodeFactory._ensure_attr_interface_visible"):
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
                utils.note_swallowed("factory.ACEShaderNodeFactory._safe_bundle_item", exc)
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
            utils.note_swallowed("factory.ACEShaderNodeFactory._link_or_consume_reroute", exc)
            return False

    @classmethod
    def _value_names(cls, category: str, node_type: str, sink_names=None, source_id: str = "") -> dict[int, str]:
        """Output index -> the scene property its Attribute node reads ("" reads nothing)."""
        if (category, node_type) == SOURCE_KIND:
            bpm, bpb, downbeat = source_tempo_props(source_id) if source_id else ("", "", "")
            return {1: bpm, 2: bpb, 3: downbeat}
        names = list(sink_names or [])
        return {
            out_idx: (names[slot] if slot < len(names) else "")
            for slot, (out_idx, _name) in enumerate(node_value_outputs(category, node_type, "ShaderNodeTree"))
        }

    @classmethod
    def _build_group_content(cls, ng, node_def, category: str, node_type: str, sink_names=None, source_id: str = ""):
        base_inputs = list(node_def.get("inputs", []))
        base_outputs = list(node_def.get("outputs", []))
        extra_inputs = [spec for spec in cls._extra_inputs(category, node_type) if cls._shader_interface_supports(spec[0])]
        value_names = cls._value_names(category, node_type, sink_names, source_id)

        for sock_type_str, sock_name in base_inputs:
            cls._new_socket(ng, str(sock_name), "INPUT", str(sock_type_str))
        for sock_type_str, sock_name in base_outputs:
            cls._new_socket(ng, str(sock_name), "OUTPUT", str(sock_type_str))

        with utils.logged_errors("factory.ACEShaderNodeFactory._build_group_content"):
            apply_socket_ranges(ng, category, node_type)
        control_names = cls._numeric_control_names(category, node_type)

        for sock_type_str, sock_name, default_value in extra_inputs:
            cls._new_socket(ng, sock_name, "INPUT", sock_type_str)
            cls._set_socket_default(ng, sock_name, default_value)

        # No bundle nodes inside shader groups: Cycles' shader inliner crashes on them.
        # Bundles pass straight through.
        input_node = ng.nodes.new("NodeGroupInput")
        output_node = ng.nodes.new("NodeGroupOutput")
        input_node.location = (-760, 0)
        output_node.location = (680, 0)

        all_inputs = list(base_inputs) + [
            (sock_type_str, sock_name)
            for sock_type_str, sock_name, _default_value in extra_inputs
        ]
        bundle_output_indices = [
            idx for idx, (sock_type, _name) in enumerate(base_outputs) if "Bundle" in str(sock_type or "")
        ]

        # Shader GroupInput may omit bundle sockets, so sockets are matched by name, not index.
        input_sock_by_name: dict[str, object] = {}
        for _s in getattr(input_node, "outputs", []) or []:
            _sn = str(getattr(_s, "name", "") or "")
            if _sn and _sn not in input_sock_by_name:
                input_sock_by_name[_sn] = _s

        # Each value output reads its scene property.
        helper_source_by_idx: dict[int, object] = {}
        for out_idx, attr_name in sorted(value_names.items()):
            if out_idx >= len(base_outputs):
                continue
            reader = cls._new_value_reader(ng, attr_name, (180, 200 - out_idx * 150))
            helper_source_by_idx[int(out_idx)] = cls._shader_attr_output_socket(reader, str(base_outputs[out_idx][0]))

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
                        with utils.logged_errors("factory.ACEShaderNodeFactory._build_group_content#3"):
                            ng.links.new(_bundle_in, output_node.inputs[int(output_idx)])

        # Consume every non-helper, non-primary-bundle input via reroute.
        for _ci, (socket_type, socket_name) in enumerate(all_inputs):
            _s = input_sock_by_name.get(str(socket_name))
            if _s is None:
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
            if control_names:
                continue
            if output_idx >= len(getattr(output_node, "inputs", [])):
                continue
            source_out = helper_source_by_idx.get(int(output_idx))
            if source_out is None:
                continue
            with utils.logged_errors("factory.ACEShaderNodeFactory._build_group_content#4"):
                ng.links.new(source_out, output_node.inputs[int(output_idx)])

        # (controls) * 0 + 1 keeps numeric controls consumed without changing the output.
        if control_names and helper_source_by_idx:
            # Use name-based lookup — bundle sockets may shift numeric indices.
            control_socks = [input_sock_by_name.get(name) for name in control_names]
            control_socks = [sock for sock in control_socks if sock is not None]
            if control_socks:
                control_sum = control_socks[0]
                for extra_idx, extra_sock in enumerate(control_socks[1:], start=1):
                    add_ctrl = ng.nodes.new("ShaderNodeMath")
                    add_ctrl.operation = "ADD"
                    add_ctrl.location = (20, -320 - (extra_idx * 60))
                    ng.links.new(control_sum, add_ctrl.inputs[0])
                    ng.links.new(extra_sock, add_ctrl.inputs[1])
                    control_sum = add_ctrl.outputs[0]

                mul_zero = ng.nodes.new("ShaderNodeMath")
                mul_zero.operation = "MULTIPLY"
                mul_zero.inputs[1].default_value = 0.0
                mul_zero.location = (220, -320)
                ng.links.new(control_sum, mul_zero.inputs[0])

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
                        with utils.logged_errors("factory.ACEShaderNodeFactory._build_group_content#5"):
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
                            utils.note_swallowed("factory.ACEShaderNodeFactory._build_group_content#6", exc)
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

        # A menu input gets its items from the Menu Switch it feeds; its default is set after the link.
        for menu_name, items in dict(node_def.get("menus", {}) or {}).items():
            menu_in = input_sock_by_name.get(str(menu_name))
            if menu_in is None or not items:
                continue
            with utils.logged_errors("factory.ACEShaderNodeFactory._build_group_content#menu"):
                switch = ng.nodes.new("GeometryNodeMenuSwitch")
                switch.data_type = "FLOAT"
                switch.location = (-120, 260)
                while len(switch.enum_items) < len(items):
                    switch.enum_items.new(str(items[len(switch.enum_items)]))
                for item, item_name in zip(switch.enum_items, items):
                    item.name = str(item_name)
                ng.links.new(menu_in, switch.inputs["Menu"])
                interface_item = next(
                    (i for i in ng.interface.items_tree if getattr(i, "name", "") == menu_name and i.in_out == "INPUT"), None)
                if interface_item is not None:
                    interface_item.default_value = str(items[0])

        cls._ensure_attr_interface_visible(ng, category, node_type)

    @classmethod
    def ensure_sink_group(cls, category: str, node_type: str, names: list[str]):
        """The group a shader node writing to *names* (one per value output) uses."""
        names = [str(n) for n in names]
        name = cls.sink_group_name(names)
        ng = bpy.data.node_groups.get(name)
        if ng is not None and (
            str(getattr(ng, "bl_idname", "")) != "ShaderNodeTree"
            or group_kind(ng) != (category, node_type)
            or [str(v) for v in ng.get(SINK_NAMES_KEY, [])] != names
        ):
            bpy.data.node_groups.remove(ng)
            ng = None
        if ng is None:
            ng = bpy.data.node_groups.new(name, "ShaderNodeTree")
            ng[KIND_KEY] = f"{category}:{node_type}"
            ng[SINK_NAMES_KEY] = names
            cls._build_group_content(ng, ACE_NODES[category]["nodes"][node_type], category, node_type, sink_names=names)
        return ng

    @classmethod
    def ensure_source_group(cls, source):
        """The group a shader Source node reading *source* uses."""
        source_id = str(getattr(source, "stable_id", "") or "")
        if not source_id:
            return None
        name = cls.source_group_name(source_id)
        ng = bpy.data.node_groups.get(name)
        if ng is not None and str(ng.get(SOURCE_ID_KEY, "")) != source_id:
            bpy.data.node_groups.remove(ng)
            ng = None
        if ng is None:
            ng = bpy.data.node_groups.new(name, "ShaderNodeTree")
            ng[KIND_KEY] = ":".join(SOURCE_KIND)
            ng[SOURCE_ID_KEY] = source_id
            cls._build_group_content(ng, registry_node_def(*SOURCE_KIND), *SOURCE_KIND, source_id=source_id)
        return ng

    @classmethod
    def _rebuild_owned_group(cls, ng) -> None:
        """Rebuild a sink or source group in place, keeping its names and its users' input values."""
        category, node_type = group_kind(ng)
        node_def = ACE_NODES[category]["nodes"][node_type]
        remember_group_inputs(ng)
        ng.nodes.clear()
        ng.interface.clear()
        cls._build_group_content(
            ng, node_def, category, node_type,
            sink_names=[str(v) for v in ng.get(SINK_NAMES_KEY, [])],
            source_id=str(ng.get(SOURCE_ID_KEY, "") or ""),
        )

    @classmethod
    def owned_groups(cls):
        """The sink and source groups of the file: ACE groups with names of their own."""
        for ng in list(bpy.data.node_groups):
            if str(getattr(ng, "bl_idname", "")) == "ShaderNodeTree" and ng.get(KIND_KEY) and group_kind(ng)[0]:
                yield ng

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
                utils.note_swallowed("factory.ACEShaderNodeFactory.rebuild_node_group", exc)
                ng = None
            else:
                ng = None
        if ng is None:
            ng = bpy.data.node_groups.new(group_name, "ShaderNodeTree")
        else:
            remember_group_inputs(ng)
            ng.nodes.clear()
            ng.interface.clear()
        cls._build_group_content(ng, node_def, cat_key, type_key)

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
        for cat_key, category in ACE_NODES.items():
            for type_key, node_def in category.get("nodes", {}).items():
                group_name = cls.get_group_name(cat_key, type_key)
                ng = bpy.data.node_groups.get(group_name)
                if ng is not None and getattr(ng, "bl_idname", "") != "ShaderNodeTree":
                    try:
                        bpy.data.node_groups.remove(ng)
                    except Exception as exc:
                        utils.note_swallowed("factory.ACEShaderNodeFactory.ensure_node_groups_exist", exc)
                        ng = None
                    else:
                        ng = None
                if ng is None:
                    ng = bpy.data.node_groups.new(group_name, "ShaderNodeTree")
                elif cls._node_group_needs_refresh(ng, node_def, cat_key, type_key):
                    remember_group_inputs(ng)
                    ng.nodes.clear()
                    ng.interface.clear()
                else:
                    cls._ensure_attr_interface_visible(ng, cat_key, type_key)
                    continue
                cls._build_group_content(ng, node_def, cat_key, type_key)
        for ng in list(cls.owned_groups()):
            if ng.library is not None:
                continue
            category, node_type = group_kind(ng)
            if cls._node_group_needs_refresh(ng, ACE_NODES[category]["nodes"][node_type], category, node_type):
                cls._rebuild_owned_group(ng)
        with utils.logged_errors("factory.ACEShaderNodeFactory.ensure_node_groups_exist#2"):
            from ..dynamic_audio_operators import ACESplitterManager

            ACESplitterManager.refresh_existing_variants("ShaderNodeTree")
        if current_hash:
            cls._NODES_CONTENT_HASH = current_hash
def register():
    try:
        ACEShaderNodeFactory.ensure_node_groups_exist()
    except AttributeError:
        utils.log("Skipping shader node group creation due to restricted data access.", "WARNING")
def unregister():
    pass
