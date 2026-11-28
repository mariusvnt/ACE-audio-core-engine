"""Shader node groups for ACE nodes: creation, content and refresh.

A shader can't name an attribute at run time, so each shader node that writes
values uses a group of its own, ``.ACE_sink_<name>``, whose Attribute nodes (View
Layer mode) read the scene properties ACE writes. A Source node uses its
source's group, ``.ACE_SHADER_source_<id>``, which reads that source's BPM,
Beats per Bar and Downbeat the same way. Nothing is added to the user's material.

Every input of a group reaches one of its outputs, or Blender greys it out as unused: a value
output is its scene property plus 0 x the inputs (an Audio bundle through a Separate Bundle, a
menu through a Menu Switch). The engine reads the inputs from the node, never from the shader.
"""

import hashlib
import json
import bpy  # type: ignore
from ...vars import ACE_NODES, SOURCE_KIND, apply_socket_ranges, node_def as registry_node_def
from ... import utils
from ..frontend import remember_group_inputs
from ..groups import SHADER, find_template, is_template, legacy_name, new_template, tag_template
from ..identity import KIND_KEY, SINK_NAMES_KEY, SOURCE_ID_KEY, group_kind, node_value_outputs, source_tempo_props
from ..layout import arrange
from .interface import ShaderGroupInterface

# Version of what the shader groups hold inside; a group built by another one is rebuilt
# (internals only). 2: every input reaches an output, laid out along the flow.
SHADER_INTERNALS_VERSION = 2


class ACEShaderNodeFactory(ShaderGroupInterface):
    """Create hidden ShaderNodeTree ACE groups for shader frontends."""

    _NODES_CONTENT_HASH: str = ""
    SINK_GROUP_PREFIX = ".ACE_sink_"
    SOURCE_GROUP_PREFIX = ".ACE_SHADER_source_"

    @staticmethod
    def get_group_name(category: str, node_type: str) -> str:
        """The hidden name the kind's groups are built from: its splitter variants' prefix.

        The template group itself is named after the node and found by its kind (nodes/groups.py).
        """
        return legacy_name(category, node_type, SHADER)

    @classmethod
    def template_group(cls, category: str, node_type: str):
        """The kind's shared shader group, built when missing."""
        ng = find_template(category, node_type, SHADER)
        if ng is None:
            cls.rebuild_node_group(category, node_type)
            ng = find_template(category, node_type, SHADER)
        return ng

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
                {"kind": "shader", "nodes": ACE_NODES, "version": SHADER_INTERNALS_VERSION},
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
    def _new_value_reader(ng, attr_name: str):
        """An Attribute node reading the scene property *attr_name* (View Layer mode)."""
        attr = ng.nodes.new("ShaderNodeAttribute")
        attr.attribute_type = "VIEW_LAYER"
        attr.attribute_name = str(attr_name or "")
        return attr

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
        for sock_type_str, sock_name in list(node_def.get("inputs", [])):
            cls._new_socket(ng, str(sock_name), "INPUT", str(sock_type_str))
        for sock_type_str, sock_name in list(node_def.get("outputs", [])):
            cls._new_socket(ng, str(sock_name), "OUTPUT", str(sock_type_str))
        with utils.logged_errors("factory.ACEShaderNodeFactory._build_group_content"):
            apply_socket_ranges(ng, category, node_type)
        cls._build_internals(ng, node_def, category, node_type, sink_names=sink_names, source_id=source_id)
        cls._ensure_attr_interface_visible(ng, category, node_type)

    @classmethod
    def _build_internals(cls, ng, node_def, category: str, node_type: str, sink_names=None, source_id: str = ""):
        """Group Input to Group Output: bundles pass through, values read their scene properties.

        Each input reaches an output, so Blender never greys it out (it greys an input that
        only ends in a Reroute): with no bundle output to pass an Audio input to, the bundle
        goes through a Separate Bundle; a menu goes through a Menu Switch; then 0 x their sum
        is added to every value output.
        """
        inputs = list(node_def.get("inputs", []))
        outputs = list(node_def.get("outputs", []))
        value_names = cls._value_names(category, node_type, sink_names, source_id)
        group_in = ng.nodes.new("NodeGroupInput")
        group_out = ng.nodes.new("NodeGroupOutput")
        in_socks = {str(s.name): s for s in group_in.outputs if s.name}
        out_socks = list(group_out.inputs)
        bundle_outputs = [i for i, (t, _n) in enumerate(outputs) if "Bundle" in str(t) and i < len(out_socks)]
        value_outputs = [(i, str(t)) for i, (t, _n) in enumerate(outputs) if "Bundle" not in str(t) and i < len(out_socks)]
        menus = dict(node_def.get("menus", {}) or {})

        consumed = []  # inputs (or what they become) that 0 x their sum keeps in use
        for sock_type, sock_name in inputs:
            sock = in_socks.get(str(sock_name))
            if sock is None:
                continue
            if "Bundle" in str(sock_type):
                if bundle_outputs:
                    for out_idx in bundle_outputs:
                        ng.links.new(sock, out_socks[out_idx])
                    continue
                separate = ng.nodes.new("NodeSeparateBundle")
                separate.label = f"{sock_name}: in use"
                separate.bundle_items.new("FLOAT", "Value")
                ng.links.new(sock, separate.inputs[0])
                consumed.append(separate.outputs["Value"])
            elif "Menu" in str(sock_type):
                items = [str(i) for i in menus.get(str(sock_name), [])]
                if not items:
                    continue
                switch = ng.nodes.new("GeometryNodeMenuSwitch")
                switch.data_type = "FLOAT"
                switch.label = str(sock_name)
                while len(switch.enum_items) < len(items):
                    switch.enum_items.new(items[len(switch.enum_items)])
                for item, name in zip(switch.enum_items, items):
                    item.name = name
                ng.links.new(sock, switch.inputs["Menu"])
                consumed.append(switch.outputs[0])
                # A menu input gets its items from the Menu Switch it feeds: its default is set after the link.
                interface_item = next(
                    (i for i in ng.interface.items_tree if getattr(i, "name", "") == sock_name and i.in_out == "INPUT"), None)
                if interface_item is not None:
                    interface_item.default_value = items[0]
            else:
                consumed.append(sock)

        zero = None
        if consumed and value_outputs:
            total = consumed[0]
            for extra in consumed[1:]:
                add = ng.nodes.new("ShaderNodeMath")
                add.operation = "ADD"
                ng.links.new(total, add.inputs[0])
                ng.links.new(extra, add.inputs[1])
                total = add.outputs[0]
            times_zero = ng.nodes.new("ShaderNodeMath")
            times_zero.operation = "MULTIPLY"
            times_zero.label = "x 0: inputs in use"
            times_zero.inputs[1].default_value = 0.0
            ng.links.new(total, times_zero.inputs[0])
            zero = times_zero.outputs[0]

        for out_idx, sock_type in value_outputs:
            reader = cls._new_value_reader(ng, value_names.get(out_idx, ""))
            value = cls._shader_attr_output_socket(reader, sock_type)
            if value is None:
                continue
            if zero is not None:
                add = ng.nodes.new("ShaderNodeMath")
                add.operation = "ADD"
                add.label = str(outputs[out_idx][1])
                ng.links.new(value, add.inputs[0])
                ng.links.new(zero, add.inputs[1])
                value = add.outputs[0]
            ng.links.new(value, out_socks[out_idx])
        arrange(ng)
        ng["ace_internals"] = SHADER_INTERNALS_VERSION

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
    def _owned_names(cls, ng) -> dict:
        return {
            "sink_names": [str(v) for v in ng.get(SINK_NAMES_KEY, [])],
            "source_id": str(ng.get(SOURCE_ID_KEY, "") or ""),
        }

    @classmethod
    def _refresh_group(cls, ng, category: str, node_type: str) -> None:
        """Rebuild a group built by another version: all of it when its sockets changed, else only its inside.

        Keeping the interface keeps the links of the nodes that use the group.
        """
        node_def = ACE_NODES[category]["nodes"][node_type]
        names = cls._owned_names(ng)
        if cls._node_group_needs_refresh(ng, node_def, category, node_type):
            remember_group_inputs(ng)
            ng.nodes.clear()
            ng.interface.clear()
            cls._build_group_content(ng, node_def, category, node_type, **names)
        elif ng.get("ace_internals") != SHADER_INTERNALS_VERSION:
            ng.nodes.clear()
            cls._build_internals(ng, node_def, category, node_type, **names)

    @classmethod
    def owned_groups(cls):
        """The sink and source groups of the file: the groups placed shader nodes use, named after what they read."""
        for ng in list(bpy.data.node_groups):
            if str(getattr(ng, "bl_idname", "")) != "ShaderNodeTree" or not group_kind(ng)[0] or is_template(ng):
                continue
            if ng.get(SINK_NAMES_KEY) is not None or ng.get(SOURCE_ID_KEY):
                yield ng

    @classmethod
    def rebuild_node_group(cls, cat_key: str, type_key: str):
        """Force-rebuild a single shader ACE node group from ACE_NODES."""
        try:
            node_def = ACE_NODES[cat_key]["nodes"][type_key]
        except KeyError:
            return
        ng = find_template(cat_key, type_key, SHADER)
        if ng is None:
            ng = new_template(cat_key, type_key, SHADER)
        else:
            tag_template(ng, cat_key, type_key, SHADER)
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
                ng = find_template(cat_key, type_key, SHADER)
                if ng is None:
                    ng = new_template(cat_key, type_key, SHADER)
                    cls._build_group_content(ng, node_def, cat_key, type_key)
                    continue
                tag_template(ng, cat_key, type_key, SHADER)
                cls._refresh_group(ng, cat_key, type_key)
                cls._ensure_attr_interface_visible(ng, cat_key, type_key)
        for ng in list(cls.owned_groups()):
            if ng.library is not None:
                continue
            cls._refresh_group(ng, *group_kind(ng))
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
