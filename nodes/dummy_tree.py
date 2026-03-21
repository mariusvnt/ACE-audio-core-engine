"""
Blender Geometry Node group management for the AudioCore Engine add-on.

``ACEDummyNodeFactory`` creates and maintains the hidden ``.ACE_*`` node
groups that appear in the GN editor as the user-facing ACE nodes.  It uses a
content hash of the ``ACE_NODES`` registry to skip the expensive O(N) scan
on frames where the registry has not changed.

``ACEDummyTreeReader`` parses an active Blender NodeTree into the engine's
``ACEAbstractTree`` IR, performing an O(1) group-name → (category, type) reverse
lookup via a hash-invalidated class-level dictionary.
"""
import bpy  # type: ignore
import hashlib
import json
from collections import deque
from typing import Optional, Any

from ..vars import ACE_NODES
from ..core.abstract_tree import (
    ACEAbstractTree,
    ACEAbstractNode,
    ACEAbstractSocket,
    ACEAbstractLink,
    ACEAbstractOutputLink,
    ACEFrontendRef,
    ACEConsumerIntent,
    ace_node_is_writeable_output,
)
from ..core.id_management import ACEIdManager
from .. import utils

FRONTEND_KIND_GEOMETRY = "GEOMETRY_NODES"
FRONTEND_KIND_SHADER = "SHADER_NODES"


def _tree_pointer(tree) -> int:
    try:
        return int(tree.as_pointer())
    except Exception:
        return id(tree)


def ensure_frontend_uid(tree, frontend_kind: str, existing_uids) -> str:
    """Guarantee a persistent, unique frontend UID on a top-level node tree."""
    if tree is None:
        return ""
    existing = {str(v) for v in (existing_uids or []) if str(v)}
    current_uid = ""
    try:
        current_uid = str(tree.get("ace_frontend_uid", "") or "")
    except Exception:
        current_uid = ""
    if current_uid and current_uid not in existing:
        return current_uid

    kind_token = ACEIdManager._clean_string(str(frontend_kind or "frontend")).lower()[:10] or "frontend"
    tree_name = str(getattr(tree, "name", "") or "Tree")
    tree_type = str(getattr(tree, "bl_idname", "") or "")
    max_attempts = max(len(existing) + 64, 256)
    for salt in range(max_attempts):
        suffix = hashlib.blake2b(
            f"{frontend_kind}:{tree_type}:{tree_name}:{salt}".encode("utf-8"),
            digest_size=4,
        ).hexdigest()[:6]
        candidate = f"ACE_FRONTEND_{kind_token}_{suffix}"
        if candidate in existing:
            continue
        try:
            tree["ace_frontend_uid"] = candidate
        except Exception:
            pass
        return candidate
    return current_uid

class ACEDummyNodeFactory:
    """Creates and manages Blender node groups based on the ACE_NODES registry.

    All groups are GeometryNodeTree datablocks with a leading "." in their
    name so they are hidden from the default user-facing Add menu but remain
    accessible for programmatic use.  Internal wiring uses Blender 5.0+
    Bundle/Separate-Bundle nodes for audio passing.
    """

    @staticmethod
    def get_group_name(category: str, node_type: str) -> str:
        """Return the canonical hidden datablock name for an ACE node group.

        The leading "." prefix keeps the datablock out of the user-visible
        Group list so it does not duplicate entries in the curated ACE add
        menu.

        Args:
            category: ACE node category key (e.g. "stream", "delta").
            node_type: ACE node type key within that category (e.g. "rms").

        Returns:
            The dot-prefixed name string, e.g. ".ACE_stream_rms".
        """
        # Internal datablocks are hidden (leading ".") so the default Group list
        # does not duplicate the curated ACE add menu entries.
        return f".ACE_{category}_{node_type}"

    _NODES_CONTENT_HASH: str = ""

    @staticmethod
    def _ace_nodes_hash() -> str:
        """Compute a stable hash of ACE_NODES to detect registry changes."""
        try:
            serialized = json.dumps(ACE_NODES, sort_keys=True, default=str).encode("utf-8")
            return hashlib.blake2b(serialized, digest_size=8).hexdigest()
        except Exception:
            return ""

    @staticmethod
    def ensure_node_groups_exist():
        """
        Iterates ACE_NODES and creates missing node groups.
        A content hash of ACE_NODES short-circuits the O(N) scan when the
        registry hasn't changed since the last invocation.
        """
        try:
            _ = bpy.data.node_groups
        except AttributeError:
            utils.log("Node groups unavailable in current context; skipping ACE node group creation.", "WARNING")
            return
        current_hash = ACEDummyNodeFactory._ace_nodes_hash()
        if current_hash and current_hash == ACEDummyNodeFactory._NODES_CONTENT_HASH:
            return  # Registry unchanged — skip the full O(N) group scan.
        # 1. Standard Nodes
        for cat_key, category in ACE_NODES.items():
            for type_key, node_def in category['nodes'].items():
                group_name = ACEDummyNodeFactory.get_group_name(cat_key, type_key)
                ng = bpy.data.node_groups.get(group_name)

                if ng:
                    if ACEDummyNodeFactory._node_group_needs_refresh(ng, type_key):
                        ng.clear()
                        ng.interface.clear()
                        ACEDummyNodeFactory._build_group_content(ng, node_def, cat_key, type_key)
                    ACEDummyNodeFactory._ensure_attr_interface_visible(ng, cat_key, type_key)
                    continue
                else:
                    ng = bpy.data.node_groups.new(group_name, 'GeometryNodeTree')
                    ACEDummyNodeFactory._build_group_content(ng, node_def, cat_key, type_key)

        # 2. Dynamic variants are ensured separately to keep sockets consistent.
        try:
            from .dynamic_audio_operators import ACESplitterManager, ACEAudioMathManager
            ACESplitterManager.refresh_existing_variants("GeometryNodeTree")
            ACEAudioMathManager.refresh_existing_variants("GeometryNodeTree")
        except Exception:
            pass

        # Cache the hash so subsequent calls with an unchanged registry skip the scan.
        # Also invalidate the ACEDummyTreeReader reverse-lookup so it is rebuilt on
        # next compilation — prevents stale entries after a registry hot-reload.
        if current_hash:
            ACEDummyNodeFactory._NODES_CONTENT_HASH = current_hash
            ACEDummyTreeReader._GROUP_NAME_LOOKUP.clear()
            ACEDummyTreeReader._ACE_NODES_LOOKUP_HASH = ""

    @staticmethod
    def rebuild_node_group(cat_key: str, type_key: str):
        """Force-rebuild a single node group from the current ACE_NODES definition.

        Clears the existing group interface and internal wiring before
        rebuilding, so stale sockets are removed even if the group already
        exists in bpy.data.node_groups.

        Args:
            cat_key: ACE node category key (e.g. "stream").
            type_key: ACE node type key within that category (e.g. "rms").
        """
        group_name = ACEDummyNodeFactory.get_group_name(cat_key, type_key)
        
        try:
            node_def = ACE_NODES[cat_key]['nodes'][type_key]
        except KeyError:
            return

        ng = bpy.data.node_groups.get(group_name)
        if ng:
            ng.clear()             
            ng.interface.clear()   
        else:
            ng = bpy.data.node_groups.new(group_name, 'GeometryNodeTree')
        
        ACEDummyNodeFactory._build_group_content(ng, node_def, cat_key, type_key)

    @staticmethod
    def prune_orphaned_ace_nodes(tree, scene) -> int:
        """
        Remove nodes from *tree* whose 'ace_source_id' property no longer corresponds
        to any active audio source in the scene.  Called once per compile so
        the node graph stays clean when sources are deleted.

        Returns the number of nodes removed.
        """
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
                from ..core.id_management import ACEIdManager
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
        except Exception:
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
            # Verify sockets follow the "Audio" / "Audio N" naming convention.
            # Checking merely for non-empty sets misses cases where sockets were
            # manually renamed, which silently breaks the audio math task wiring.
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
        if node_type == "logic":
            required_inputs = {"Value", "Threshold", "Attribute ID", "Proxy Object"}
            required_outputs = {"Logic"}
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
            return not required_inputs.issubset(inputs) or not required_outputs.issubset(outputs)

        # ── Delta nodes ──────────────────────────────────────────────────────
        # All delta nodes share a single "Audio" input; outputs are fixed named
        # bundles specific to each type.  Without these branches, socket-schema
        # changes to delta nodes were silently ignored at startup, leaving stale
        # sockets until the user forced a manual reload.
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
        """Return the Attribute-ID socket names that must remain visible for a node.

        The Attribute ID sockets are string inputs that let GN expressions read
        named mesh attributes written by the ACE runtime.  They must not be
        hidden (hide_value=False) so users can set them in the node properties.

        Args:
            category: ACE node category key.
            node_type: ACE node type key.

        Returns:
            A tuple of socket name strings.  Empty tuple for node types that
            carry no attribute-ID sockets.
        """
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
        if category == "utilities" and node_type == "logic":
            return ("Attribute ID",)
        return ()

    @staticmethod
    def _ensure_attr_interface_visible(ng, category: str, node_type: str):
        """Set hide_value=False on all Attribute-ID interface sockets of *ng*.

        Blender may silently re-hide sockets after certain node group edits, so
        this is called both during initial build and on refresh checks.

        Args:
            ng: The Blender GeometryNodeTree node group to patch.
            category: ACE node category key.
            node_type: ACE node type key.
        """
        for attr_name in ACEDummyNodeFactory._attr_socket_names(category, node_type):
            item = next(
                (
                    it for it in getattr(ng.interface, "items_tree", [])
                    if getattr(it, "in_out", None) == 'INPUT' and getattr(it, "name", "") == attr_name
                ),
                None,
            )
            if item is not None and hasattr(item, "hide_value"):
                try:
                    item.hide_value = False
                except Exception:
                    pass

    @staticmethod
    def _build_group_content(ng, node_def, category, node_type):
        """Populate the interface sockets and internal wiring of a node group.

        Delegates internal wiring to _create_dummy_internals() after building
        the socket interface from *node_def*.

        Args:
            ng: The target GeometryNodeTree to populate.
            node_def: The ACE_NODES node definition dict for this node type.
            category: ACE node category key.
            node_type: ACE node type key.
        """
        # --- 1. INTERFACE ---
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
        
        # --- 2. INTERNALS SWITCH ---
        ACEDummyNodeFactory._create_dummy_internals(ng, inputs, outputs, category, node_type)
        ACEDummyNodeFactory._ensure_attr_interface_visible(ng, category, node_type)

    @staticmethod
    def _create_dummy_internals(ng, inputs_def, outputs_def, category, node_type):
        """
        Constructs the internal node graph logic using Bundle nodes.
        """
        input_node = ng.nodes.new('NodeGroupInput')
        output_node = ng.nodes.new('NodeGroupOutput')
        input_node.location = (-600, 0)
        output_node.location = (600, 0)

        # =====================================================================
        # CASE A: SOURCE MANAGEMENT
        # =====================================================================
        if category == 'source_mngmt':

            if node_type == 'src':
                # Logic: Object Info (Proxy) -> Combine Bundle -> Group Output

                # 1. Expose Object Socket on Group Input
                if not any(s.name == "Proxy Object" for s in ng.interface.items_tree):
                    ng.interface.new_socket("Proxy Object", in_out='INPUT', socket_type='NodeSocketObject')

                # 2. Object Info
                obj_info = ng.nodes.new('GeometryNodeObjectInfo')
                obj_info.location = (-200, 0)
                obj_info.transform_space = 'ORIGINAL'
                obj_info.inputs['As Instance'].default_value = False

                # Link Input (Proxy Object) -> Object Info
                proxy_sock_idx = len(inputs_def)
                ng.links.new(input_node.outputs[proxy_sock_idx], obj_info.inputs[0])

                # 3. Combine Bundle (Blender 5.0)
                bundle_node = ng.nodes.new('NodeCombineBundle')
                bundle_node.bundle_items.new('GEOMETRY', "Proxy Geometry")
                bundle_node.location = (200, 0)

                ng.links.new(obj_info.outputs['Geometry'], bundle_node.inputs["Proxy Geometry"])
                ng.links.new(bundle_node.outputs[0], output_node.inputs[0])

        # =====================================================================
        # CASE B: DELTA (Audio -> Audio)
        # =====================================================================
        elif category == 'delta':
            # Logic: Use Join Bundle to drive outputs
            
            if len(inputs_def) > 0:
                source_bundle = input_node.outputs[0] 
                
                for i in range(len(outputs_def)):
                    # Connect Input -> Join Bundle
                    ng.links.new(source_bundle, output_node.inputs[i])

        # =====================================================================
        # CASE C: STREAM (Audio -> Float/Vector)
        # =====================================================================
        elif category == 'stream':
            # Logic: Separate Bundle -> one Sample Index per non-bundle output.
            # Stream nodes with extra numeric controls (e.g. LUFS window/overlap ratio)
            # consume those inputs via (sum * 0) fed into Sample Index[Index] so
            # sockets stay active and non-grey without altering sampled values.

            # 1. Attribute-ID sockets (one per output)
            value_output_count = sum(1 for sock_type, _sock_name in outputs_def if "Bundle" not in str(sock_type or ""))
            if value_output_count <= 0:
                value_output_count = len(outputs_def)
            value_output_count = max(int(value_output_count), 0)
            attr_socket_names: list[str] = []
            for out_idx in range(value_output_count):
                sock_name = "Attribute ID" if out_idx == 0 else f"Attribute ID {int(out_idx + 1)}"
                if not any(s.name == sock_name for s in ng.interface.items_tree):
                    ng.interface.new_socket(sock_name, in_out='INPUT', socket_type='NodeSocketString')
                try:
                    ng.interface.items_tree[sock_name].default_value = (
                        "ACE_undefined" if out_idx == 0 else f"ACE_undefined_{int(out_idx + 1)}"
                    )
                except Exception:
                    pass
                attr_socket_names.append(sock_name)

            attr_id_start_idx = len(inputs_def)

            # 2. Nodes
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

            # 3. Wiring
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

        # =====================================================================
        # CASE D: UTILITIES
        # =====================================================================
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
                # Monitoring sink: one Bundle Audio input, zero outputs.
                # No internal wiring needed — the socket interface alone defines the contract.
                # The facade must accept but not forward the bundle (it is a pure sink).
                pass

            elif node_type == 'audio_math':
                # Transparent passthrough facade: input bundle i → output bundle i.
                # The actual add/subtract/multiply logic runs in the Layer 2 backend.
                # This wiring lets the user see the audio signal flow visually.
                for i in range(min(len(inputs_def), len(outputs_def))):
                    ng.links.new(input_node.outputs[i], output_node.inputs[i])

            elif node_type == 'util_rolling_avg':
                # Attribute ID input (string) — the auto-generated proxy attribute name.
                if not any(s.name == "Attribute ID" for s in ng.interface.items_tree):
                    ng.interface.new_socket("Attribute ID", in_out='INPUT', socket_type='NodeSocketString')
                    ng.interface.items_tree["Attribute ID"].default_value = "ACE_undefined"

                attr_id_idx = len(inputs_def)  # 2: after Value (0) and Number of Frames (1)

                # Proxy Object input — the source proxy mesh to read the attribute from.
                # Auto-populated at node creation time (gn_menu._initialize_auto_attr_ids).
                # Using an explicit Object socket (mirroring the src node) rather than
                # Self Object keeps the node context-independent and lets the user
                # inspect or override the target object.
                if not any(s.name == "Proxy Object" for s in ng.interface.items_tree):
                    ng.interface.new_socket("Proxy Object", in_out='INPUT', socket_type='NodeSocketObject')

                proxy_obj_idx = len(inputs_def) + 1  # 3: after Attribute ID (2)

                # ── Nodes ────────────────────────────────────────────────────
                # (Value + Number of Frames) × 0 → always evaluates to 0.
                # Feeds into Sample Index[Index] so sampling is always at element 0,
                # while both inputs are genuinely consumed by compute nodes (no grey sockets).
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

                # ── Wiring ───────────────────────────────────────────────────
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

            elif node_type == 'logic':
                if not any(s.name == "Attribute ID" for s in ng.interface.items_tree):
                    ng.interface.new_socket("Attribute ID", in_out='INPUT', socket_type='NodeSocketString')
                    ng.interface.items_tree["Attribute ID"].default_value = "ACE_undefined"

                attr_id_idx = len(inputs_def)

                if not any(s.name == "Proxy Object" for s in ng.interface.items_tree):
                    ng.interface.new_socket("Proxy Object", in_out='INPUT', socket_type='NodeSocketObject')

                proxy_obj_idx = len(inputs_def) + 1

                math_add = ng.nodes.new('ShaderNodeMath')
                math_add.operation = 'ADD'
                math_add.location = (-560, -160)

                math_mul = ng.nodes.new('ShaderNodeMath')
                math_mul.operation = 'MULTIPLY'
                math_mul.inputs[1].default_value = 0.0
                math_mul.location = (-380, -160)

                ng.links.new(input_node.outputs[0], math_add.inputs[0])
                ng.links.new(input_node.outputs[1], math_add.inputs[1])
                ng.links.new(math_add.outputs[0], math_mul.inputs[0])

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

                ng.links.new(input_node.outputs[proxy_obj_idx], obj_info.inputs['Object'])
                ng.links.new(obj_info.outputs['Geometry'], sample_idx.inputs['Geometry'])
                ng.links.new(input_node.outputs[attr_id_idx], named_attr.inputs['Name'])
                ng.links.new(named_attr.outputs[0], sample_idx.inputs['Value'])
                ng.links.new(math_mul.outputs[0], sample_idx.inputs['Index'])
                if len(outputs_def) > 0:
                    ng.links.new(sample_idx.outputs['Value'], output_node.inputs[0])

        # =====================================================================
        # FALLBACK (unknown node type)
        # =====================================================================
        else:
            for i, _ in enumerate(inputs_def):
                reroute = ng.nodes.new('NodeReroute')
                reroute.location = (0, -i * 100)
                ng.links.new(input_node.outputs[i], reroute.inputs[0])
                if i < len(outputs_def):
                    ng.links.new(reroute.outputs[0], output_node.inputs[i])


class ACEDummyTreeReader:
    """Parse an active Blender GeometryNodeTree into an ACEAbstractTree.

    Performs a breadth-first traversal from all managed source nodes,
    assigning or verifying ACE IDs for each ACE node group encountered,
    wiring abstract links, and capturing socket-level parameters.

    The resulting ACEAbstractTree is consumed by ACEAudioEngine.compile()
    to build the parallel DSP graph IR.
    """

    # Reverse lookup: group_name (and its stripped variant) → (cat_key, type_key).
    # Rebuilt automatically when ACE_NODES content hash changes so add-on reloads
    # and registry extensions are reflected without a full Blender restart.
    _GROUP_NAME_LOOKUP: dict[str, tuple[str, str]] = {}
    _ACE_NODES_LOOKUP_HASH: str = ""
    FACTORY_CLS = ACEDummyNodeFactory

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

    @staticmethod
    def _node_pointer(node) -> int:
        """Return a stable integer identity for a Blender node.

        Uses as_pointer() when available (Blender C-level pointer) and falls
        back to Python id() so the result is always an int regardless of node
        type.
        """
        try:
            return int(node.as_pointer())
        except Exception:
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
        except Exception:
            return ace_id

        self._ace_id_owner[new_id] = node_ptr
        self._node_id_by_ptr[node_ptr] = new_id
        self._reserved_node_ids.add(new_id)
        # Persist the disambiguated ID back into the node's custom property so
        # that duplicate nodes have stable unique IDs on the next session/reload.
        try:
            bl_node["ace_id"] = new_id
        except Exception:
            pass
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
        try:
            idx_val = getattr(sock, "index")
            if idx_val is not None:
                return int(idx_val)
        except Exception:
            pass
        try:
            seq = getattr(node, "inputs" if is_input else "outputs", [])
            for idx, ref in enumerate(seq):
                if ref == sock:
                    return int(idx)
        except Exception:
            pass
        return 0

    def _topology_signature(self, bl_node) -> dict[str, list[tuple[int, int, str]]]:
        incoming: list[tuple[int, int, str]] = []
        outgoing: list[tuple[int, int, str]] = []
        try:
            for in_sock in getattr(bl_node, "inputs", []):
                in_idx = self._socket_index_for(bl_node, in_sock, is_input=True)
                for link in getattr(in_sock, "links", []):
                    src_node = getattr(link, "from_node", None)
                    src_sock = getattr(link, "from_socket", None)
                    if src_node is None or src_sock is None:
                        continue
                    src_idx = self._socket_index_for(src_node, src_sock, is_input=False)
                    incoming.append((int(in_idx), int(src_idx), self._node_kind_token(src_node)))
        except Exception:
            pass
        try:
            for out_sock in getattr(bl_node, "outputs", []):
                out_idx = self._socket_index_for(bl_node, out_sock, is_input=False)
                for link in getattr(out_sock, "links", []):
                    dst_node = getattr(link, "to_node", None)
                    dst_sock = getattr(link, "to_socket", None)
                    if dst_node is None or dst_sock is None:
                        continue
                    dst_idx = self._socket_index_for(dst_node, dst_sock, is_input=True)
                    outgoing.append((int(out_idx), int(dst_idx), self._node_kind_token(dst_node)))
        except Exception:
            pass
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
            except Exception:
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

    def build(self) -> ACEAbstractTree:
        """Execute the full parse pass and return the populated ACEAbstractTree.

        Returns:
            The ACEAbstractTree built from the Blender node tree supplied at
            construction time.  Returns an empty tree if bl_tree is None.
        """
        if not self.bl_tree: return self.abstract_tree
        sources = self._find_source_nodes()
        if sources:
            self._traverse_from_sources(sources)
        self._validate_cross_source_links()
        self._emit_consumer_intents()
        return self.abstract_tree

    def _validate_cross_source_links(self):
        """
        Warn when a non-AudioMath node receives inputs from distinct root sources.
        Cross-source mixing is only valid through AudioMathTask; any other target
        produces silently undefined analysis output because the audio chains are
        never synchronized.
        """
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
        return ace_node_is_writeable_output(node)

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
        except Exception:
            stored_ace_id = ""
        parsed_stored = ACEIdManager.parse_node_id(stored_ace_id) if stored_ace_id else None
        if parsed_stored and parsed_stored.get("category") == resolved_cat and parsed_stored.get("type") == resolved_type:
            if resolved_cat == "source_mngmt" and resolved_type == "src":
                try:
                    source_id = str(bl_node.get("ace_source_id", "") or "")
                except Exception:
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
            try:
                if str(bl_node.get("ace_id", "") or "") != resolved_id:
                    bl_node["ace_id"] = resolved_id
            except Exception:
                pass
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
        except Exception:
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
        try:
            if str(bl_node.get("ace_id", "") or "") != resolved_id:
                bl_node["ace_id"] = resolved_id
        except Exception:
            pass
        return resolved_id

    def _find_source_nodes(self):
        """Return all managed source placeholder nodes in the Blender tree.

        Only nodes that carry a valid ace_source_id (written by
        ACEProxyManager._sync_sonar_tree) are included.  Ad-hoc group nodes
        that look like source nodes but lack a bound source are excluded so
        unlinked placeholder copies do not generate phantom graph sources.

        Returns:
            A list of Blender node objects of type 'GROUP' with category
            "source_mngmt" and type "src".
        """
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
                except Exception:
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

            # If we reached an ACE node after traversing non-ACE nodes, bridge the link.
            if current_ace_id and prev_ace_id and incoming_socket is not None and current_ace_id != prev_ace_id:
                target_socket_idx = self._resolve_target_socket_idx(current_bl_node, incoming_socket)
                if target_socket_idx is None:
                    try:
                        if len(getattr(current_bl_node, "inputs", [])) > 0:
                            target_socket_idx = 0
                    except Exception:
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
        except Exception:
            return f"obj:{id(sock)}"

    @staticmethod
    def _resolve_target_socket_idx(target_node, to_socket) -> Optional[int]:
        target_socket_idx = None
        try:
            target_socket_idx = to_socket.index  # type: ignore[attr-defined]
        except Exception:
            target_socket_idx = None
        if target_socket_idx is not None:
            return target_socket_idx

        try:
            target_socket_idx = next((i for i, s in enumerate(target_node.inputs) if s == to_socket), None)
        except Exception:
            target_socket_idx = None
        if target_socket_idx is not None:
            return target_socket_idx

        sock_name = ""
        try:
            sock_name = getattr(to_socket, "name", "") or ""
        except Exception:
            sock_name = ""
        if sock_name:
            try:
                target_socket_idx = next(
                    (i for i, s in enumerate(target_node.inputs) if getattr(s, "name", "") == sock_name),
                    None,
                )
            except Exception:
                target_socket_idx = None
        if target_socket_idx is not None:
            return target_socket_idx

        sock_identifier = ""
        try:
            sock_identifier = getattr(to_socket, "identifier", "") or ""
        except Exception:
            sock_identifier = ""
        if sock_identifier:
            try:
                target_socket_idx = next(
                    (i for i, s in enumerate(target_node.inputs) if getattr(s, "identifier", "") == sock_identifier),
                    None,
                )
            except Exception:
                target_socket_idx = None
        return target_socket_idx

    def _resolve_root_id(self, bl_node) -> str:
        """
        Try to derive a stable source identifier from the source node:
        prefer explicit ace_source_id, then the proxy object name, else the node ace_id.
        """
        root_id = ""
        try:
            root_id = bl_node.get("ace_source_id", "")
        except Exception:
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
            except Exception:
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

        # Custom parameters (e.g., Audio Math operation)
        op_value = bl_node.get("ace_op")
        if isinstance(op_value, str):
            params["op"] = op_value
        preview_enabled = bl_node.get("ace_preview_enabled", True)
        params["preview_enabled"] = bool(preview_enabled)
        params["frontend_node_index"] = int(
            self._node_order_by_ptr.get(self._node_pointer(bl_node), 0)
        )

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

# --- MODULE REGISTRATION ---

def register():
    try:
        ACEDummyNodeFactory.ensure_node_groups_exist()
    except AttributeError:
        # Restricted context (e.g., background handler). Skip building for now.
        utils.log("Skipping node group creation due to restricted data access.", "WARNING")

def unregister():
    pass
