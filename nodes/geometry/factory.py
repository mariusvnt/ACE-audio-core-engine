"""Geometry Nodes groups for ACE nodes: interface and internals."""

import bpy  # type: ignore
import hashlib
import json
from ...vars import (
    ACE_NODES, ROLE_MEASURE, ROLE_PREVIEW, ROLE_SOURCE, ROLE_SPLITTER, apply_socket_ranges, node_def as registry_node_def,
    node_is_native, node_role, value_outputs,
)
from ... import utils
from ..frontend import remember_group_inputs


class ACEDummyNodeFactory:
    """Creates and manages Blender node groups based on the ACE_NODES registry."""

    @staticmethod
    def get_group_name(category: str, node_type: str) -> str:
        """Return the canonical hidden datablock name for an ACE node group."""
        # Internal datablocks are hidden (leading ".") so the default Group list
        # does not duplicate the curated ACE add menu entries.
        return f".ACE_{category}_{node_type}"

    _NODES_CONTENT_HASH: str = ""

    @staticmethod
    def _ace_nodes_hash() -> str:
        try:
            serialized = json.dumps([ACE_NODES, INTERNALS_VERSION], sort_keys=True, default=str).encode("utf-8")
            return hashlib.blake2b(serialized, digest_size=8).hexdigest()
        except Exception as exc:
            utils.note_swallowed("factory.ACEDummyNodeFactory._ace_nodes_hash", exc)
            return ""

    @staticmethod
    def ensure_node_groups_exist():
        """Iterates ACE_NODES and creates missing node groups."""
        try:
            _ = bpy.data.node_groups
        except AttributeError:
            utils.log("Node groups unavailable in current context; skipping ACE node group creation.", "WARNING")
            return
        current_hash = ACEDummyNodeFactory._ace_nodes_hash()
        if current_hash and current_hash == ACEDummyNodeFactory._NODES_CONTENT_HASH:
            return  # Registry unchanged — skip the full O(N) group scan.
        for cat_key, category in ACE_NODES.items():
            for type_key, node_def in category['nodes'].items():
                group_name = ACEDummyNodeFactory.get_group_name(cat_key, type_key)
                ng = bpy.data.node_groups.get(group_name)

                if ng:
                    if ACEDummyNodeFactory._node_group_needs_refresh(ng, cat_key, type_key):
                        remember_group_inputs(ng)
                        ng.nodes.clear()
                        ng.interface.clear()
                        ACEDummyNodeFactory._build_group_content(ng, node_def, cat_key, type_key)
                    elif ACEDummyNodeFactory._internals_outdated(ng, cat_key, type_key):
                        # Only the inside changed: the interface stays, so the links of the
                        # nodes using the group in the user's trees stay too.
                        ng.nodes.clear()
                        ACEDummyNodeFactory._create_dummy_internals(
                            ng, node_def.get('inputs', []), node_def.get('outputs', []), cat_key, type_key
                        )
                    ACEDummyNodeFactory._ensure_attr_interface_visible(ng, cat_key, type_key)
                    continue
                else:
                    ng = bpy.data.node_groups.new(group_name, 'GeometryNodeTree')
                    ACEDummyNodeFactory._build_group_content(ng, node_def, cat_key, type_key)

        with utils.logged_errors("factory.ACEDummyNodeFactory.ensure_node_groups_exist"):
            from ..dynamic_audio_operators import ACESplitterManager
            ACESplitterManager.refresh_existing_variants("GeometryNodeTree")

        if current_hash:
            ACEDummyNodeFactory._NODES_CONTENT_HASH = current_hash

    @staticmethod
    def rebuild_node_group(cat_key: str, type_key: str):
        """Force-rebuild a single node group from the current ACE_NODES definition."""
        group_name = ACEDummyNodeFactory.get_group_name(cat_key, type_key)
        
        try:
            node_def = ACE_NODES[cat_key]['nodes'][type_key]
        except KeyError:
            return

        ng = bpy.data.node_groups.get(group_name)
        if ng:
            remember_group_inputs(ng)
            ng.nodes.clear()
            ng.interface.clear()
        else:
            ng = bpy.data.node_groups.new(group_name, 'GeometryNodeTree')
        
        ACEDummyNodeFactory._build_group_content(ng, node_def, cat_key, type_key)

    @staticmethod
    def _internals_outdated(ng, category: str, node_type: str) -> bool:
        """True when a group with real internals (a measure's readers, a controller's maths) was built by an older version."""
        has_internals = node_role(category, node_type) == ROLE_MEASURE or node_is_native(category, node_type)
        return has_internals and ng.get("ace_internals") != INTERNALS_VERSION

    @staticmethod
    def _node_group_needs_refresh(ng, category: str, node_type: str) -> bool:
        try:
            inputs = {item.name for item in ng.interface.items_tree if getattr(item, "in_out", None) == 'INPUT'}
            outputs = {item.name for item in ng.interface.items_tree if getattr(item, "in_out", None) == 'OUTPUT'}
        except Exception as exc:
            utils.note_swallowed("factory.ACEDummyNodeFactory._node_group_needs_refresh", exc)
            return True

        role = node_role(category, node_type)
        nd = registry_node_def(category, node_type)
        declared_inputs = {str(name) for _sock_type, name in list(nd.get("inputs", []))}
        declared_outputs = {str(name) for _sock_type, name in list(nd.get("outputs", []))}
        if role == ROLE_SOURCE:
            required_inputs = {"Proxy Object"}
            return not required_inputs.issubset(inputs) or not declared_outputs.issubset(outputs)
        if role == ROLE_PREVIEW:
            # Sink: one Audio input, no outputs required.
            return "Audio" not in inputs
        if role == ROLE_SPLITTER:
            # Dynamic output count; require at least the template "Audio" output.
            return "Audio" not in inputs or not any(
                n == "Audio" or n.startswith("Audio ") for n in outputs
            )
        if node_is_native(category, node_type):
            # Exactly the declared sockets: a controller computes from all of them.
            return inputs != declared_inputs or outputs != declared_outputs
        attr_inputs = set(ACEDummyNodeFactory._attr_socket_names(category, node_type))
        if role == ROLE_MEASURE:
            required_inputs = declared_inputs | attr_inputs
            # Inputs the definition no longer declares (e.g. a removed control)
            # also make the group stale.
            return (
                not required_inputs.issubset(inputs)
                or not declared_outputs.issubset(outputs)
                or bool(inputs - required_inputs)
            )
        return False

    @staticmethod
    def _attr_socket_names(category: str, node_type: str) -> tuple[str, ...]:
        """Return the Attribute-ID socket names that must remain visible for a node."""
        from ..identity import attribute_socket_name

        return tuple(attribute_socket_name(slot) for slot in range(len(value_outputs(category, node_type))))

    @staticmethod
    def _ensure_attr_interface_visible(ng, category: str, node_type: str):
        for attr_name in ACEDummyNodeFactory._attr_socket_names(category, node_type):
            item = next(
                (
                    it for it in getattr(ng.interface, "items_tree", [])
                    if getattr(it, "in_out", None) == 'INPUT' and getattr(it, "name", "") == attr_name
                ),
                None,
            )
            if item is not None and hasattr(item, "hide_value"):
                with utils.logged_errors("factory.ACEDummyNodeFactory._ensure_attr_interface_visible"):
                    item.hide_value = False

    @staticmethod
    def _build_group_content(ng, node_def, category, node_type):
        """Populate the interface sockets and internal wiring of a node group."""
        inputs = node_def.get('inputs', [])
        for sock_type_str, sock_name in inputs:
            ng.interface.new_socket(sock_name, in_out='INPUT', socket_type=sock_type_str)

        outputs = node_def.get('outputs', [])
        for sock_type_str, sock_name in outputs:
            ng.interface.new_socket(sock_name, in_out='OUTPUT', socket_type=sock_type_str)

        if node_is_native(category, node_type):
            from .controllers import prepare_interface

            prepare_interface(ng, node_def)
        with utils.logged_errors("factory.ACEDummyNodeFactory._build_group_content"):
            apply_socket_ranges(ng, category, node_type)

        ACEDummyNodeFactory._create_dummy_internals(ng, inputs, outputs, category, node_type)
        ACEDummyNodeFactory._ensure_attr_interface_visible(ng, category, node_type)

    @staticmethod
    def _create_dummy_internals(ng, inputs_def, outputs_def, category, node_type):
        """Constructs the internal node graph logic using Bundle nodes."""
        if node_is_native(category, node_type):
            from .controllers import build_internals

            build_internals(ng, registry_node_def(category, node_type), node_type)
            ng["ace_internals"] = INTERNALS_VERSION
            return
        input_node = ng.nodes.new('NodeGroupInput')
        output_node = ng.nodes.new('NodeGroupOutput')
        input_node.location = (-600, 0)
        output_node.location = (600, 0)

        role = node_role(category, node_type)

        # CASE A: SOURCE
        if role == ROLE_SOURCE:
            # Logic: Object Info (Proxy) -> Combine Bundle -> Group Output

            if not any(s.name == "Proxy Object" for s in ng.interface.items_tree):
                ng.interface.new_socket("Proxy Object", in_out='INPUT', socket_type='NodeSocketObject')

            obj_info = ng.nodes.new('GeometryNodeObjectInfo')
            obj_info.location = (-200, 0)
            obj_info.transform_space = 'ORIGINAL'
            obj_info.inputs['As Instance'].default_value = False

            # Link Input (Proxy Object) -> Object Info
            proxy_sock_idx = len(inputs_def)
            ng.links.new(input_node.outputs[proxy_sock_idx], obj_info.inputs[0])

            bundle_node = ng.nodes.new('NodeCombineBundle')
            bundle_node.bundle_items.new('GEOMETRY', "Proxy Geometry")
            bundle_node.location = (200, 0)

            ng.links.new(obj_info.outputs['Geometry'], bundle_node.inputs["Proxy Geometry"])
            ng.links.new(bundle_node.outputs[0], output_node.inputs[0])

            # BPM and Beats per Bar: what the source writes on point 0 of its proxy.
            from ...core.proxy.attributes import SOURCE_TEMPO_ATTRS
            for out_idx, (attr_name, data_type) in enumerate(SOURCE_TEMPO_ATTRS.items(), start=1):
                named_attr = ng.nodes.new('GeometryNodeInputNamedAttribute')
                named_attr.data_type = data_type
                named_attr.inputs['Name'].default_value = attr_name
                named_attr.location = (-200, -200 * out_idx)
                sample_idx = ng.nodes.new('GeometryNodeSampleIndex')
                sample_idx.data_type = data_type
                sample_idx.domain = 'POINT'
                sample_idx.location = (200, -200 * out_idx)
                ng.links.new(obj_info.outputs['Geometry'], sample_idx.inputs['Geometry'])
                ng.links.new(named_attr.outputs[0], sample_idx.inputs['Value'])
                ng.links.new(sample_idx.outputs['Value'], output_node.inputs[out_idx])

        # CASE B: SPLITTER (Audio -> Audio)
        elif role == ROLE_SPLITTER:
            # Logic: Use Join Bundle to drive outputs
            
            if len(inputs_def) > 0:
                source_bundle = input_node.outputs[0] 
                
                for i in range(len(outputs_def)):
                    # Connect Input -> Join Bundle
                    ng.links.new(source_bundle, output_node.inputs[i])

        # CASE C: MEASURE (Audio -> Float/Vector)
        elif role == ROLE_MEASURE:
            # One Sample Index per non-bundle output; extra numeric controls feed (sum * 0) into the index
            # so their sockets stay active without changing the sampled value.

            value_output_count = sum(1 for sock_type, _sock_name in outputs_def if "Bundle" not in str(sock_type or ""))
            if value_output_count <= 0:
                value_output_count = len(outputs_def)
            value_output_count = max(int(value_output_count), 0)
            attr_socket_names: list[str] = []
            for out_idx in range(value_output_count):
                sock_name = "Attribute ID" if out_idx == 0 else f"Attribute ID {int(out_idx + 1)}"
                if not any(s.name == sock_name for s in ng.interface.items_tree):
                    ng.interface.new_socket(sock_name, in_out='INPUT', socket_type='NodeSocketString')
                with utils.logged_errors("factory.ACEDummyNodeFactory._create_dummy_internals"):
                    ng.interface.items_tree[sock_name].default_value = (
                        "ACE_undefined" if out_idx == 0 else f"ACE_undefined_{int(out_idx + 1)}"
                    )
                attr_socket_names.append(sock_name)

            attr_id_start_idx = len(inputs_def)

            sep_bundle = ng.nodes.new('NodeSeparateBundle')
            sep_bundle.bundle_items.new('GEOMETRY', "Proxy Geometry")
            sep_bundle.location = (-220, 0)

            # Optional index driver for numeric controls.
            index_driver_socket = None
            numeric_ctrl_indices = []
            for idx, (sock_type, _sock_name) in enumerate(inputs_def):
                if idx == 0:
                    continue
                st = str(sock_type or "")
                if "Bundle" in st or "String" in st or "Object" in st:
                    continue
                numeric_ctrl_indices.append(idx)
            if len(numeric_ctrl_indices) >= 2:
                math_add = ng.nodes.new('ShaderNodeMath')
                math_add.operation = 'ADD'
                math_add.location = (-560, -180)
                ng.links.new(input_node.outputs[int(numeric_ctrl_indices[0])], math_add.inputs[0])
                ng.links.new(input_node.outputs[int(numeric_ctrl_indices[1])], math_add.inputs[1])
                add_out = math_add.outputs[0]
                for extra_idx in numeric_ctrl_indices[2:]:
                    add_next = ng.nodes.new('ShaderNodeMath')
                    add_next.operation = 'ADD'
                    add_next.location = (int(math_add.location.x) + 180, int(math_add.location.y))
                    ng.links.new(add_out, add_next.inputs[0])
                    ng.links.new(input_node.outputs[int(extra_idx)], add_next.inputs[1])
                    add_out = add_next.outputs[0]
                math_mul = ng.nodes.new('ShaderNodeMath')
                math_mul.operation = 'MULTIPLY'
                math_mul.inputs[1].default_value = 0.0
                math_mul.location = (-340, -180)
                ng.links.new(add_out, math_mul.inputs[0])
                index_driver_socket = math_mul.outputs[0]
            elif len(numeric_ctrl_indices) == 1:
                math_mul = ng.nodes.new('ShaderNodeMath')
                math_mul.operation = 'MULTIPLY'
                math_mul.inputs[1].default_value = 0.0
                math_mul.location = (-340, -180)
                ng.links.new(input_node.outputs[int(numeric_ctrl_indices[0])], math_mul.inputs[0])
                index_driver_socket = math_mul.outputs[0]

            if len(inputs_def) > 0:
                ng.links.new(input_node.outputs[0], sep_bundle.inputs[0])
            frame_index = _frame_index_socket(ng, sep_bundle.outputs[0], index_driver_socket, (-220, -420))

            for out_idx in range(min(len(outputs_def), len(output_node.inputs))):
                sock_type = str(outputs_def[out_idx][0] or "") if out_idx < len(outputs_def) else ""
                if "Bundle" in sock_type:
                    continue

                named_attr = ng.nodes.new('GeometryNodeInputNamedAttribute')
                named_attr.location = (0, 210 - (out_idx * 220))
                target_type = 'FLOAT'
                if 'Vector' in sock_type:
                    target_type = 'FLOAT_VECTOR'
                elif 'Color' in sock_type or 'RGBA' in sock_type:
                    target_type = 'FLOAT_COLOR'
                named_attr.data_type = target_type

                sample_idx = ng.nodes.new('GeometryNodeSampleIndex')
                sample_idx.location = (220, 200 - (out_idx * 220))
                sample_idx.data_type = target_type
                sample_idx.domain = 'POINT'
                sample_idx.clamp = True

                ng.links.new(sep_bundle.outputs[0], sample_idx.inputs['Geometry'])
                attr_socket_idx = attr_id_start_idx + out_idx
                if attr_socket_idx < len(input_node.outputs):
                    ng.links.new(input_node.outputs[attr_socket_idx], named_attr.inputs['Name'])
                ng.links.new(named_attr.outputs[0], sample_idx.inputs['Value'])
                ng.links.new(frame_index, sample_idx.inputs['Index'])
                ng.links.new(sample_idx.outputs['Value'], output_node.inputs[out_idx])

        # CASE D: PREVIEW (pure sink: one bundle input, no outputs)
        elif role == ROLE_PREVIEW:
            pass

        # FALLBACK (unknown node type)
        else:
            for i, _ in enumerate(inputs_def):
                reroute = ng.nodes.new('NodeReroute')
                reroute.location = (0, -i * 100)
                ng.links.new(input_node.outputs[i], reroute.inputs[0])
                if i < len(outputs_def):
                    ng.links.new(reroute.outputs[0], output_node.inputs[i])
        ng["ace_internals"] = INTERNALS_VERSION


# Version of what the groups with real internals hold inside; a group built by another one is
# rebuilt (internals only). 2: measures sample the frame's point of a baked proxy. 3: the
# controllers' Simulation Zones.
INTERNALS_VERSION = 3


def _frame_index_socket(ng, geometry_socket, driver_socket, location):
    """The point to sample: floor(scene frame) - ACE_frame0 of the geometry (+ driver, which is 0).

    A live proxy has one point and the Sample Index nodes clamp, so they read it at any frame;
    a baked proxy has one point per frame from ACE_frame0 on.
    """
    from ...core.proxy.attributes import FRAME0_ATTR

    x, y = location
    scene_time = ng.nodes.new('GeometryNodeInputSceneTime')
    scene_time.location = (x - 400, y)
    floor = ng.nodes.new('ShaderNodeMath')
    floor.operation = 'FLOOR'
    floor.location = (x - 200, y)
    frame0_attr = ng.nodes.new('GeometryNodeInputNamedAttribute')
    frame0_attr.data_type = 'INT'
    frame0_attr.inputs['Name'].default_value = FRAME0_ATTR
    frame0_attr.location = (x - 400, y - 160)
    frame0 = ng.nodes.new('GeometryNodeSampleIndex')
    frame0.data_type = 'INT'
    frame0.domain = 'POINT'
    frame0.location = (x - 200, y - 160)
    offset = ng.nodes.new('ShaderNodeMath')
    offset.operation = 'SUBTRACT'
    offset.location = (x, y)
    ng.links.new(scene_time.outputs['Frame'], floor.inputs[0])
    ng.links.new(geometry_socket, frame0.inputs['Geometry'])
    ng.links.new(frame0_attr.outputs[0], frame0.inputs['Value'])
    ng.links.new(floor.outputs[0], offset.inputs[0])
    ng.links.new(frame0.outputs['Value'], offset.inputs[1])
    if driver_socket is None:
        return offset.outputs[0]
    plus_driver = ng.nodes.new('ShaderNodeMath')
    plus_driver.operation = 'ADD'
    plus_driver.location = (x + 180, y)
    ng.links.new(offset.outputs[0], plus_driver.inputs[0])
    ng.links.new(driver_socket, plus_driver.inputs[1])
    return plus_driver.outputs[0]


def register():
    try:
        ACEDummyNodeFactory.ensure_node_groups_exist()
    except AttributeError:
        # Restricted context (e.g., background handler). Skip building for now.
        utils.log("Skipping node group creation due to restricted data access.", "WARNING")


def unregister():
    pass
