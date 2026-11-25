"""Geometry Nodes groups for ACE nodes: interface and internals."""

import bpy  # type: ignore
import hashlib
import json
from ...vars import ACE_NODES
from ... import utils


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
            serialized = json.dumps(ACE_NODES, sort_keys=True, default=str).encode("utf-8")
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
                    if ACEDummyNodeFactory._node_group_needs_refresh(ng, type_key):
                        ng.nodes.clear()
                        ng.interface.clear()
                        ACEDummyNodeFactory._build_group_content(ng, node_def, cat_key, type_key)
                    ACEDummyNodeFactory._ensure_attr_interface_visible(ng, cat_key, type_key)
                    continue
                else:
                    ng = bpy.data.node_groups.new(group_name, 'GeometryNodeTree')
                    ACEDummyNodeFactory._build_group_content(ng, node_def, cat_key, type_key)

        with utils.logged_errors("factory.ACEDummyNodeFactory.ensure_node_groups_exist"):
            from ..dynamic_audio_operators import ACESplitterManager, ACEAudioMathManager
            ACESplitterManager.refresh_existing_variants("GeometryNodeTree")
            ACEAudioMathManager.refresh_existing_variants("GeometryNodeTree")

        # A registry change also invalidates the reader's group-name lookup.
        if current_hash:
            from .reader import ACEDummyTreeReader

            ACEDummyNodeFactory._NODES_CONTENT_HASH = current_hash
            ACEDummyTreeReader._GROUP_NAME_LOOKUP.clear()
            ACEDummyTreeReader._ACE_NODES_LOOKUP_HASH = ""

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
            ng.nodes.clear()
            ng.interface.clear()
        else:
            ng = bpy.data.node_groups.new(group_name, 'GeometryNodeTree')
        
        ACEDummyNodeFactory._build_group_content(ng, node_def, cat_key, type_key)

    @staticmethod
    def prune_orphaned_ace_nodes(tree, scene) -> int:
        """Remove nodes from *tree* whose 'ace_source_id' property no longer corresponds to any active audio source in the scene."""
        if tree is None:
            return 0
        ace = getattr(scene, "audiocore", None)
        if ace is None:
            return 0
        live_source_ids: set[str] = set()
        for src in getattr(ace, "audio_sources", []):
            sid = getattr(src, "stable_id", "")
            if sid:
                live_source_ids.add(sid)
        removed = 0
        try:
            for node in list(getattr(tree, "nodes", [])):
                source_id = getattr(node, "get", lambda k, d=None: None)("ace_source_id")
                if not source_id:
                    continue
                from ...core.id_management import ACEIdManager
                ace_id = getattr(node, "get", lambda k, d=None: None)("ace_id")
                parsed = ACEIdManager.parse_node_id(str(ace_id or ""))
                if parsed and parsed.get("category") == "source_mngmt" and parsed.get("type") == "src":
                    if source_id not in live_source_ids:
                        tree.nodes.remove(node)
                        removed += 1
        except Exception as exc:
            utils.log(f"ACE: orphan pruning failed: {exc}", "WARNING")
        return removed

    @staticmethod
    def _node_group_needs_refresh(ng, node_type: str) -> bool:
        try:
            inputs = {item.name for item in ng.interface.items_tree if getattr(item, "in_out", None) == 'INPUT'}
            outputs = {item.name for item in ng.interface.items_tree if getattr(item, "in_out", None) == 'OUTPUT'}
        except Exception as exc:
            utils.note_swallowed("factory.ACEDummyNodeFactory._node_group_needs_refresh", exc)
            return True

        if node_type == "src":
            required_inputs = {"Proxy Object"}
            required_outputs = {"Audio"}
            return not required_inputs.issubset(inputs) or not required_outputs.issubset(outputs)
        if node_type == "src_specs":
            required_inputs = {
                "Audio",
                "Frames per Beat Attribute ID",
                "Beat Phase Attribute ID",
                "Proxy Object",
            }
            required_outputs = {"Frames per Beat", "Beat Phase", "Audio"}
            return not required_inputs.issubset(inputs) or not required_outputs.issubset(outputs)
        if node_type == "preview_sink":
            # Sink: one Audio input, no outputs required.
            return "Audio" not in inputs
        if node_type == "audio_math":
            # Renamed sockets break the audio math wiring, so check names, not counts.
            has_audio_in = any(n == "Audio" or n.startswith("Audio ") for n in inputs)
            has_audio_out = any(n == "Audio" or n.startswith("Audio ") for n in outputs)
            return not has_audio_in or not has_audio_out
        if node_type == "util_rolling_avg":
            required_inputs = {"Value", "Number of Frames", "Attribute ID", "Proxy Object"}
            required_outputs = {"Rolling Avg"}
            return not required_inputs.issubset(inputs) or not required_outputs.issubset(outputs)
        if node_type == "stream_accumulator":
            required_inputs = {"Value", "Attribute ID", "Proxy Object"}
            required_outputs = {"Accumulated"}
            return not required_inputs.issubset(inputs) or not required_outputs.issubset(outputs)
        stream_nodes = (
            ACE_NODES.get("stream", {}).get("nodes", {})
            if isinstance(ACE_NODES.get("stream", {}), dict)
            else {}
        )
        if node_type in stream_nodes:
            node_def = stream_nodes.get(node_type, {}) if isinstance(stream_nodes, dict) else {}
            required_inputs = {str(name) for _sock_type, name in list(node_def.get("inputs", []))}
            required_outputs = {str(name) for _sock_type, name in list(node_def.get("outputs", []))}
            value_output_count = 0
            for sock_type, _sock_name in list(node_def.get("outputs", [])):
                if "Bundle" in str(sock_type or ""):
                    continue
                value_output_count += 1
            if value_output_count > 0:
                required_inputs.add("Attribute ID")
            for idx in range(1, int(value_output_count)):
                required_inputs.add(f"Attribute ID {int(idx + 1)}")
            # Inputs the definition no longer declares (e.g. a removed control)
            # also make the group stale.
            return (
                not required_inputs.issubset(inputs)
                or not required_outputs.issubset(outputs)
                or bool(inputs - required_inputs)
            )

        # Delta nodes: one Audio input and type-specific bundle outputs.
        if node_type == "midside_splitter":
            return "Audio" not in inputs or not {"Mid", "Side"}.issubset(outputs)
        if node_type == "attackrelease_splitter":
            return "Audio" not in inputs or not {"Attack", "Release"}.issubset(outputs)
        if node_type == "auto_component_splitter":
            return "Audio" not in inputs or not {"Percussive", "Harmonics", "Residual"}.issubset(outputs)
        if node_type == "chromagram":
            # 12 chroma outputs; verify at least the first (C) is present and
            # the input is wired correctly.
            return "Audio" not in inputs or "C" not in outputs
        if node_type == "frequency_splitter":
            # Dynamic output count; require at least the template "Audio" output.
            return "Audio" not in inputs or not any(
                n == "Audio" or n.startswith("Audio ") for n in outputs
            )

        return False

    @staticmethod
    def _attr_socket_names(category: str, node_type: str) -> tuple[str, ...]:
        """Return the Attribute-ID socket names that must remain visible for a node."""
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
            names = ["Attribute ID"]
            for idx in range(1, int(value_output_count)):
                names.append(f"Attribute ID {int(idx + 1)}")
            return tuple(names)
        if category == "utilities" and node_type == "src_specs":
            return ("Frames per Beat Attribute ID", "Beat Phase Attribute ID")
        if category == "utilities" and node_type == "util_rolling_avg":
            return ("Attribute ID",)
        if category == "utilities" and node_type == "stream_accumulator":
            return ("Attribute ID",)
        return ()

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

        if category == "stream" and node_type == "lufs":
            for item in getattr(getattr(ng, "interface", None), "items_tree", []):
                if getattr(item, "in_out", None) != "INPUT":
                    continue
                name = str(getattr(item, "name", "") or "")
                if name == "Window (s)":
                    with utils.logged_errors("factory.ACEDummyNodeFactory._build_group_content"):
                        item.default_value = 0.4
                    for attr_name, value in (
                        ("min_value", 0.05),
                        ("max_value", 3.0),
                    ):
                        with utils.logged_errors("factory.ACEDummyNodeFactory._build_group_content#2"):
                            setattr(item, attr_name, value)
        
        ACEDummyNodeFactory._create_dummy_internals(ng, inputs, outputs, category, node_type)
        ACEDummyNodeFactory._ensure_attr_interface_visible(ng, category, node_type)

    @staticmethod
    def _create_dummy_internals(ng, inputs_def, outputs_def, category, node_type):
        """Constructs the internal node graph logic using Bundle nodes."""
        input_node = ng.nodes.new('NodeGroupInput')
        output_node = ng.nodes.new('NodeGroupOutput')
        input_node.location = (-600, 0)
        output_node.location = (600, 0)

        # CASE A: SOURCE MANAGEMENT
        if category == 'source_mngmt':

            if node_type == 'src':
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

        # CASE B: DELTA (Audio -> Audio)
        elif category == 'delta':
            # Logic: Use Join Bundle to drive outputs
            
            if len(inputs_def) > 0:
                source_bundle = input_node.outputs[0] 
                
                for i in range(len(outputs_def)):
                    # Connect Input -> Join Bundle
                    ng.links.new(source_bundle, output_node.inputs[i])

        # CASE C: STREAM (Audio -> Float/Vector)
        elif category == 'stream':
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

                ng.links.new(sep_bundle.outputs[0], sample_idx.inputs['Geometry'])
                attr_socket_idx = attr_id_start_idx + out_idx
                if attr_socket_idx < len(input_node.outputs):
                    ng.links.new(input_node.outputs[attr_socket_idx], named_attr.inputs['Name'])
                ng.links.new(named_attr.outputs[0], sample_idx.inputs['Value'])
                if index_driver_socket is not None:
                    ng.links.new(index_driver_socket, sample_idx.inputs['Index'])
                ng.links.new(sample_idx.outputs['Value'], output_node.inputs[out_idx])

        # CASE D: UTILITIES
        elif category == 'utilities':

            if node_type == 'src_specs':
                attr_inputs = (
                    ("Frames per Beat Attribute ID", "ACE_undefined_frames_per_beat"),
                    ("Beat Phase Attribute ID", "ACE_undefined_beatphase"),
                )
                for sock_name, default_name in attr_inputs:
                    if not any(s.name == sock_name for s in ng.interface.items_tree):
                        ng.interface.new_socket(sock_name, in_out='INPUT', socket_type='NodeSocketString')
                        ng.interface.items_tree[sock_name].default_value = default_name
                if not any(s.name == "Proxy Object" for s in ng.interface.items_tree):
                    ng.interface.new_socket("Proxy Object", in_out='INPUT', socket_type='NodeSocketObject')

                attr_id_start_idx = len(inputs_def)
                proxy_obj_idx = attr_id_start_idx + len(attr_inputs)

                obj_info = ng.nodes.new('GeometryNodeObjectInfo')
                obj_info.transform_space = 'ORIGINAL'
                obj_info.location = (-320, 0)
                if 'As Instance' in obj_info.inputs:
                    obj_info.inputs['As Instance'].default_value = False

                ng.links.new(input_node.outputs[proxy_obj_idx], obj_info.inputs['Object'])

                sampler_specs = (
                    ('FLOAT', 0, (-120, 220), (140, 220)),
                    ('FLOAT', 1, (-120, -20), (140, -20)),
                )
                for offset, (data_type, out_idx, attr_loc, sample_loc) in enumerate(sampler_specs):
                    named_attr = ng.nodes.new('GeometryNodeInputNamedAttribute')
                    named_attr.data_type = data_type
                    named_attr.location = attr_loc

                    sample_idx = ng.nodes.new('GeometryNodeSampleIndex')
                    sample_idx.data_type = data_type
                    sample_idx.domain = 'POINT'
                    sample_idx.location = sample_loc

                    ng.links.new(obj_info.outputs['Geometry'], sample_idx.inputs['Geometry'])
                    ng.links.new(input_node.outputs[attr_id_start_idx + offset], named_attr.inputs['Name'])
                    ng.links.new(named_attr.outputs[0], sample_idx.inputs['Value'])
                    if out_idx < len(output_node.inputs):
                        ng.links.new(sample_idx.outputs['Value'], output_node.inputs[out_idx])

                if len(inputs_def) > 0 and len(outputs_def) > 0:
                    ng.links.new(input_node.outputs[0], output_node.inputs[len(outputs_def) - 1])

            elif node_type == 'preview_sink':
                # Pure sink: one bundle input, no outputs.
                pass

            elif node_type == 'audio_math':
                # Pass-through facade; the math runs in the backend.
                for i in range(min(len(inputs_def), len(outputs_def))):
                    ng.links.new(input_node.outputs[i], output_node.inputs[i])

            elif node_type == 'util_rolling_avg':
                # Attribute ID input (string) — the auto-generated proxy attribute name.
                if not any(s.name == "Attribute ID" for s in ng.interface.items_tree):
                    ng.interface.new_socket("Attribute ID", in_out='INPUT', socket_type='NodeSocketString')
                    ng.interface.items_tree["Attribute ID"].default_value = "ACE_undefined"

                attr_id_idx = len(inputs_def)  # 2: after Value (0) and Number of Frames (1)

                # Explicit proxy Object input, filled at creation, so the node does not depend on Self Object.
                if not any(s.name == "Proxy Object" for s in ng.interface.items_tree):
                    ng.interface.new_socket("Proxy Object", in_out='INPUT', socket_type='NodeSocketObject')

                proxy_obj_idx = len(inputs_def) + 1  # 3: after Attribute ID (2)

                # (Value + Number of Frames) * 0: both inputs are consumed and the index stays 0.
                math_add = ng.nodes.new('ShaderNodeMath')
                math_add.operation = 'ADD'
                math_add.location = (-560, -160)

                math_mul = ng.nodes.new('ShaderNodeMath')
                math_mul.operation = 'MULTIPLY'
                math_mul.inputs[1].default_value = 0.0
                math_mul.location = (-380, -160)

                ng.links.new(input_node.outputs[0], math_add.inputs[0])  # Value  → Add A
                ng.links.new(input_node.outputs[1], math_add.inputs[1])  # Frames → Add B
                ng.links.new(math_add.outputs[0], math_mul.inputs[0])    # sum → × 0

                obj_info = ng.nodes.new('GeometryNodeObjectInfo')
                obj_info.transform_space = 'ORIGINAL'
                obj_info.location = (-300, 0)
                if 'As Instance' in obj_info.inputs:
                    obj_info.inputs['As Instance'].default_value = False

                named_attr = ng.nodes.new('GeometryNodeInputNamedAttribute')
                named_attr.data_type = 'FLOAT'
                named_attr.location = (-100, 150)

                sample_idx = ng.nodes.new('GeometryNodeSampleIndex')
                sample_idx.data_type = 'FLOAT'
                sample_idx.domain = 'POINT'
                sample_idx.location = (150, 0)

                # A. Group Input (Proxy Object) → Object Info (Object)
                ng.links.new(input_node.outputs[proxy_obj_idx], obj_info.inputs['Object'])
                # B. Object Info (Geometry) → Sample Index (Geometry)
                ng.links.new(obj_info.outputs['Geometry'], sample_idx.inputs['Geometry'])
                # C. Group Input (Attribute ID) → Named Attribute (Name)
                ng.links.new(input_node.outputs[attr_id_idx], named_attr.inputs['Name'])
                # D. Named Attribute → Sample Index (Value)
                ng.links.new(named_attr.outputs[0], sample_idx.inputs['Value'])
                # E. (Value + Number of Frames) × 0 → Sample Index (Index) — always 0
                ng.links.new(math_mul.outputs[0], sample_idx.inputs['Index'])
                # F. Sample Index (Value) → Group Output (Rolling Avg)
                if len(outputs_def) > 0:
                    ng.links.new(sample_idx.outputs['Value'], output_node.inputs[0])

            elif node_type == 'stream_accumulator':
                # Attribute ID input (string) — the auto-generated proxy attribute name.
                if not any(s.name == "Attribute ID" for s in ng.interface.items_tree):
                    ng.interface.new_socket("Attribute ID", in_out='INPUT', socket_type='NodeSocketString')
                    ng.interface.items_tree["Attribute ID"].default_value = "ACE_undefined"

                attr_id_idx = len(inputs_def)  # 1: after Value (0)

                # Proxy Object input — the source proxy mesh to read the attribute from.
                if not any(s.name == "Proxy Object" for s in ng.interface.items_tree):
                    ng.interface.new_socket("Proxy Object", in_out='INPUT', socket_type='NodeSocketObject')

                proxy_obj_idx = len(inputs_def) + 1  # 2: after Attribute ID (1)

                # Consume Value so the input is active, then force index 0 sampling.
                math_mul = ng.nodes.new('ShaderNodeMath')
                math_mul.operation = 'MULTIPLY'
                math_mul.inputs[1].default_value = 0.0
                math_mul.location = (-420, -160)
                ng.links.new(input_node.outputs[0], math_mul.inputs[0])  # Value → × 0

                obj_info = ng.nodes.new('GeometryNodeObjectInfo')
                obj_info.transform_space = 'ORIGINAL'
                obj_info.location = (-300, 0)
                if 'As Instance' in obj_info.inputs:
                    obj_info.inputs['As Instance'].default_value = False

                named_attr = ng.nodes.new('GeometryNodeInputNamedAttribute')
                named_attr.data_type = 'FLOAT'
                named_attr.location = (-100, 150)

                sample_idx = ng.nodes.new('GeometryNodeSampleIndex')
                sample_idx.data_type = 'FLOAT'
                sample_idx.domain = 'POINT'
                sample_idx.location = (150, 0)

                # A. Group Input (Proxy Object) → Object Info (Object)
                ng.links.new(input_node.outputs[proxy_obj_idx], obj_info.inputs['Object'])
                # B. Object Info (Geometry) → Sample Index (Geometry)
                ng.links.new(obj_info.outputs['Geometry'], sample_idx.inputs['Geometry'])
                # C. Group Input (Attribute ID) → Named Attribute (Name)
                ng.links.new(input_node.outputs[attr_id_idx], named_attr.inputs['Name'])
                # D. Named Attribute → Sample Index (Value)
                ng.links.new(named_attr.outputs[0], sample_idx.inputs['Value'])
                # E. Value × 0 → Sample Index (Index) — always 0
                ng.links.new(math_mul.outputs[0], sample_idx.inputs['Index'])
                # F. Sample Index (Value) → Group Output (Accumulated)
                if len(outputs_def) > 0:
                    ng.links.new(sample_idx.outputs['Value'], output_node.inputs[0])

        # FALLBACK (unknown node type)
        else:
            for i, _ in enumerate(inputs_def):
                reroute = ng.nodes.new('NodeReroute')
                reroute.location = (0, -i * 100)
                ng.links.new(input_node.outputs[i], reroute.inputs[0])
                if i < len(outputs_def):
                    ng.links.new(reroute.outputs[0], output_node.inputs[i])


def register():
    try:
        ACEDummyNodeFactory.ensure_node_groups_exist()
    except AttributeError:
        # Restricted context (e.g., background handler). Skip building for now.
        utils.log("Skipping node group creation due to restricted data access.", "WARNING")


def unregister():
    pass
