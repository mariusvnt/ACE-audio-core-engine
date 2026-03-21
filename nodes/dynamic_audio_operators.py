import math

import bpy # type: ignore
from .dummy_tree import ACEDummyNodeFactory


def _tree_bl_idname(tree_or_type) -> str:
    if isinstance(tree_or_type, str):
        tree_type = tree_or_type
    else:
        tree_type = getattr(tree_or_type, "bl_idname", "")
    tree_type = str(tree_type or "")
    return tree_type or "GeometryNodeTree"


def _factory_for_tree(tree_or_type):
    if _tree_bl_idname(tree_or_type) == "ShaderNodeTree":
        from .shader_tree import ACEShaderNodeFactory
        return ACEShaderNodeFactory
    return ACEDummyNodeFactory


def _mapped_socket_type(tree_or_type, socket_type: str) -> str:
    socket_type = str(socket_type or "")
    if _tree_bl_idname(tree_or_type) != "ShaderNodeTree":
        return socket_type
    mapping = {
        "NodeSocketInt": "NodeSocketFloat",
        "NodeSocketObject": "NodeSocketString",
    }
    return mapping.get(socket_type, socket_type or "NodeSocketFloat")


def _ensure_variant_group(name: str, tree_or_type) -> bpy.types.NodeTree | None:
    tree_type = _tree_bl_idname(tree_or_type)
    ng = bpy.data.node_groups.get(name)
    if ng is not None and getattr(ng, "bl_idname", "") != tree_type:
        try:
            bpy.data.node_groups.remove(ng)
        except Exception:
            pass
        ng = None
    if ng is None:
        ng = bpy.data.node_groups.new(name, tree_type)
    return ng

class ACESplitterManager:
    """
    Manages independent variants of the Frequency Splitter.
    Generates hidden Node Groups like '.ACE_delta_frequency_splitter_3_bands'.
    """
    _INTERNAL_VERSION_TAG = "ACE_SPLITTER_INTERNALS_V5"

    @staticmethod
    def get_or_create_variant(band_count: int, tree_or_type="GeometryNodeTree") -> bpy.types.NodeTree:
        """
        Retrieves (or builds) the Node Group data block for a specific band count.
        Creates it if it doesn't exist.
        """
        band_count = max(2, min(int(band_count), 8))
        # 1. Construct the unique name for this variant
        factory_cls = _factory_for_tree(tree_or_type)
        tree_type = _tree_bl_idname(tree_or_type)
        base_name = factory_cls.get_group_name('delta', 'frequency_splitter')
        variant_name = f"{base_name}_{band_count}_bands"

        # 2. Return if exists (after migration checks)
        if variant_name in bpy.data.node_groups:
            ng = bpy.data.node_groups[variant_name]
            if getattr(ng, "bl_idname", "") == tree_type:
                ACESplitterManager._ensure_split_socket_limits(ng)
                ACESplitterManager._ensure_variant_internals(ng, band_count)
                return ng
            try:
                bpy.data.node_groups.remove(ng)
            except Exception:
                pass
            
        # 3. Create New Blueprint
        return ACESplitterManager._build_variant(variant_name, band_count, tree_or_type)

    @staticmethod
    def refresh_existing_variants(tree_or_type="GeometryNodeTree"):
        """
        Migrate already-existing splitter variants in-place so older projects
        immediately benefit from range constraints and non-greyed controls.
        """
        tree_type = _tree_bl_idname(tree_or_type)
        base_name = _factory_for_tree(tree_or_type).get_group_name('delta', 'frequency_splitter')
        prefix = f"{base_name}_"
        for ng in list(getattr(bpy.data, "node_groups", [])):
            if str(getattr(ng, "bl_idname", "") or "") != tree_type:
                continue
            name = getattr(ng, "name", "")
            if not (name.startswith(prefix) and name.endswith("_bands")):
                continue
            mid = name[len(prefix):-len("_bands")]
            if not mid.isdigit():
                continue
            try:
                ACESplitterManager.get_or_create_variant(int(mid), tree_or_type)
            except Exception:
                pass
        # Ensure baseline variant exists for fresh sessions.
        ACESplitterManager.get_or_create_variant(2, tree_or_type)

    @staticmethod
    def apply_variant_to_node(node: bpy.types.Node, band_count: int):
        """
        Swaps the node_tree of a specific node instance to the requested variant.
        """
        if not node or node.type != 'GROUP': return
        
        # Get the blueprint
        variant_group = ACESplitterManager.get_or_create_variant(band_count, getattr(node, "id_data", None))
        
        # Swap it
        if node.node_tree != variant_group:
            node.node_tree = variant_group
        
        # Update Label for clarity
        node.label = f"Splitter ({band_count} Bands)"
        
        # Store state in custom property so UI knows what to show
        node['ace_band_count'] = band_count

        # Ensure instance sockets expose the intended Hz range in the GN editor.
        ACESplitterManager._ensure_node_socket_limits(node)
        
        # Force refresh
        node.update()

    @staticmethod
    def _default_split_freq(split_idx: int, split_count: int, low_hz: float = 20.0, high_hz: float = 20000.0) -> float:
        """Log-spaced split defaults across the audible range."""
        if split_count <= 0:
            return 1000.0
        low = max(float(low_hz), 1e-3)
        high = max(float(high_hz), low + 1e-3)
        ratio = math.pow(high / low, 1.0 / float(split_count + 1))
        return float(low * math.pow(ratio, float(split_idx + 1)))

    @staticmethod
    def _set_interface_socket_limits(sock):
        try:
            sock.min_value = 0.0
        except Exception:
            pass
        try:
            sock.max_value = 20000.0
        except Exception:
            pass
        try:
            sock.soft_min = 20.0
        except Exception:
            pass
        try:
            sock.soft_max = 20000.0
        except Exception:
            pass
        try:
            sock.default_value = float(min(max(float(sock.default_value), 20.0), 20000.0))
        except Exception:
            pass

    @staticmethod
    def _ensure_split_socket_limits(ng: bpy.types.NodeTree):
        for item in getattr(getattr(ng, "interface", None), "items_tree", []):
            try:
                if getattr(item, "in_out", "") == 'INPUT' and str(getattr(item, "name", "")).startswith("Split Freq"):
                    ACESplitterManager._set_interface_socket_limits(item)
            except Exception:
                continue

    @staticmethod
    def _ensure_node_socket_limits(node: bpy.types.Node):
        for idx, sock in enumerate(getattr(node, "inputs", [])):
            if idx == 0:
                continue
            try:
                sock.min_value = 0.0
            except Exception:
                pass
            try:
                sock.max_value = 20000.0
            except Exception:
                pass
            try:
                sock.soft_min = 20.0
            except Exception:
                pass
            try:
                sock.soft_max = 20000.0
            except Exception:
                pass

    @staticmethod
    def _variant_has_current_internals(ng: bpy.types.NodeTree) -> bool:
        for n in getattr(ng, "nodes", []):
            if getattr(n, "name", "") == ACESplitterManager._INTERNAL_VERSION_TAG:
                return True
        return False

    @staticmethod
    def _ensure_variant_internals(ng: bpy.types.NodeTree, band_count: int):
        if ACESplitterManager._variant_has_current_internals(ng):
            return
        try:
            ng.nodes.clear()
        except Exception:
            pass
        ACESplitterManager._build_variant_internals(ng, band_count)

    @staticmethod
    def _build_variant_internals(ng: bpy.types.NodeTree, band_count: int):
        if str(getattr(ng, "bl_idname", "") or "") == "ShaderNodeTree":
            ACESplitterManager._build_shader_variant_internals(ng, band_count)
            return

        split_count = max(0, int(band_count) - 1)

        input_node = ng.nodes.new('NodeGroupInput')
        output_node = ng.nodes.new('NodeGroupOutput')
        input_node.location = (-700, 0)
        output_node.location = (700, 0)

        marker = ng.nodes.new('NodeFrame')
        marker.name = ACESplitterManager._INTERNAL_VERSION_TAG
        marker.label = "ACE Splitter Internals"
        marker.location = (-520, 260)

        # Minimal dummy internals:
        # - keep proxy geometry in each output bundle
        # - add all split frequencies into each output bundle as float items
        # Backend reads split values from node sockets; this just keeps UI sockets active.
        sep_bundle = ng.nodes.new('NodeSeparateBundle')
        sep_bundle.bundle_items.new('GEOMETRY', "Proxy Geometry")
        sep_bundle.location = (-500, 0)
        sep_bundle.parent = marker
        if len(getattr(input_node, "outputs", [])) > 0:
            ng.links.new(input_node.outputs[0], sep_bundle.inputs[0])

        for out_idx in range(band_count):
            if out_idx >= len(getattr(output_node, "inputs", [])):
                continue

            combine_bundle = ng.nodes.new('NodeCombineBundle')
            combine_bundle.location = (120, -120 - out_idx * 120)
            combine_bundle.parent = marker
            geo_item = combine_bundle.bundle_items.new('GEOMETRY', "Proxy Geometry")
            try:
                ng.links.new(sep_bundle.outputs[0], combine_bundle.inputs[geo_item.name])
            except Exception:
                ng.links.new(sep_bundle.outputs[0], combine_bundle.inputs[0])

            for split_idx in range(split_count):
                input_idx = split_idx + 1
                if input_idx >= len(getattr(input_node, "outputs", [])):
                    continue
                freq_item = None
                for data_kind in ("FLOAT", "VALUE"):
                    try:
                        freq_item = combine_bundle.bundle_items.new(data_kind, f"Split Freq {split_idx + 1}")
                        break
                    except Exception:
                        freq_item = None
                if freq_item is None:
                    continue
                try:
                    ng.links.new(input_node.outputs[input_idx], combine_bundle.inputs[freq_item.name])
                except Exception:
                    try:
                        last_idx = len(getattr(combine_bundle, "inputs", [])) - 1
                        if last_idx >= 0:
                            ng.links.new(input_node.outputs[input_idx], combine_bundle.inputs[last_idx])
                    except Exception:
                        pass

            ng.links.new(combine_bundle.outputs[0], output_node.inputs[out_idx])

    @staticmethod
    def _build_shader_variant_internals(ng: bpy.types.NodeTree, band_count: int):
        split_count = max(0, int(band_count) - 1)

        input_node = ng.nodes.new('NodeGroupInput')
        output_node = ng.nodes.new('NodeGroupOutput')
        input_node.location = (-700, 0)
        output_node.location = (520, 0)
        marker = ng.nodes.new('NodeReroute')
        marker.name = ACESplitterManager._INTERNAL_VERSION_TAG
        marker.location = (-860, 260)

        for split_idx in range(split_count):
            input_idx = split_idx + 1
            if input_idx >= len(getattr(input_node, "outputs", [])):
                continue
            splitter_tap = ng.nodes.new('NodeReroute')
            splitter_tap.location = (-220, -120 * split_idx)
            ng.links.new(input_node.outputs[input_idx], splitter_tap.inputs[0])

        if len(getattr(input_node, "outputs", [])) <= 0:
            return
        for out_idx in range(band_count):
            if out_idx >= len(getattr(output_node, "inputs", [])):
                continue
            reroute = ng.nodes.new('NodeReroute')
            reroute.location = (120, -120 * out_idx)
            ng.links.new(input_node.outputs[0], reroute.inputs[0])
            ng.links.new(reroute.outputs[0], output_node.inputs[out_idx])

    @staticmethod
    def _build_variant(name: str, band_count: int, tree_or_type="GeometryNodeTree") -> bpy.types.NodeTree:
        """
        Constructs the internal node graph for a splitter variant.
        Wires inputs/outputs to internal dummy nodes to prevent UI 'grey-out'.
        """
        ng = _ensure_variant_group(name, tree_or_type)
        if ng is None:
            raise RuntimeError(f"Could not create splitter variant '{name}'")
        
        # --- DYNAMIC SOCKET GENERATION ---
        
        # Input 0: Main Audio
        ng.interface.new_socket("Audio", in_out='INPUT', socket_type=_mapped_socket_type(tree_or_type, 'NodeSocketBundle'))
        
        # Inputs 1..N: Split Frequencies (We need Count-1 split points)
        # e.g. 3 Bands = 2 Split Points (Low/Mid boundary, Mid/High boundary)
        split_count = max(0, band_count - 1)
        for i in range(band_count - 1):
            sock = ng.interface.new_socket(f"Split Freq {i+1}", in_out='INPUT', socket_type=_mapped_socket_type(tree_or_type, 'NodeSocketFloat'))
            sock.default_value = ACESplitterManager._default_split_freq(i, split_count)
            ACESplitterManager._set_interface_socket_limits(sock)
            
        # Outputs: N Bands
        for i in range(band_count):
            name = f"Band {i+1}"
            if i == 0:
                name = "Low"
            elif i == band_count - 1:
                name = "High"
            elif band_count == 3 and i == 1:
                name = "Mid"
            
            ng.interface.new_socket(name, in_out='OUTPUT', socket_type=_mapped_socket_type(tree_or_type, 'NodeSocketBundle'))

        ACESplitterManager._build_variant_internals(ng, band_count)
        
        return ng


class ACEAudioMathManager:
    """
    Manages independent variants of the Audio Math node.
    Generates hidden Node Groups like '.ACE_utilities_audio_math_3_inputs'.
    """
    _INTERNAL_VERSION_TAG = "ACEAudioMathInternals_v3"

    @staticmethod
    def get_or_create_variant(input_count: int, tree_or_type="GeometryNodeTree") -> bpy.types.NodeTree:
        count = max(2, min(int(input_count), 8))
        factory_cls = _factory_for_tree(tree_or_type)
        tree_type = _tree_bl_idname(tree_or_type)
        base_name = factory_cls.get_group_name('utilities', 'audio_math')
        variant_name = f"{base_name}_{count}_inputs"

        if variant_name in bpy.data.node_groups:
            ng = bpy.data.node_groups[variant_name]
            if getattr(ng, "bl_idname", "") == tree_type:
                ACEAudioMathManager._ensure_variant_internals(ng, count)
                return ng
            try:
                bpy.data.node_groups.remove(ng)
            except Exception:
                pass

        return ACEAudioMathManager._build_variant(variant_name, count, tree_or_type)

    @staticmethod
    def refresh_existing_variants(tree_or_type="GeometryNodeTree"):
        tree_type = _tree_bl_idname(tree_or_type)
        base_name = _factory_for_tree(tree_or_type).get_group_name('utilities', 'audio_math')
        prefix = f"{base_name}_"
        for ng in list(getattr(bpy.data, "node_groups", [])):
            if str(getattr(ng, "bl_idname", "") or "") != tree_type:
                continue
            name = getattr(ng, "name", "")
            if not (name.startswith(prefix) and name.endswith("_inputs")):
                continue
            mid = name[len(prefix):-len("_inputs")]
            if not mid.isdigit():
                continue
            try:
                ACEAudioMathManager.get_or_create_variant(int(mid), tree_or_type)
            except Exception:
                pass
        ACEAudioMathManager.get_or_create_variant(2, tree_or_type)

    @staticmethod
    def _variant_has_current_internals(ng: bpy.types.NodeTree) -> bool:
        for n in getattr(ng, "nodes", []):
            if getattr(n, "name", "") == ACEAudioMathManager._INTERNAL_VERSION_TAG:
                return True
        return False

    @staticmethod
    def _ensure_variant_internals(ng: bpy.types.NodeTree, input_count: int):
        if ACEAudioMathManager._variant_has_current_internals(ng):
            return
        try:
            ng.nodes.clear()
        except Exception:
            pass
        ACEAudioMathManager._build_variant_internals(ng, input_count)

    @staticmethod
    def apply_variant_to_node(node: bpy.types.Node, input_count: int):
        if not node or node.type != 'GROUP':
            return

        count = max(2, min(int(input_count), 8))
        variant_group = ACEAudioMathManager.get_or_create_variant(count, getattr(node, "id_data", None))
        if node.node_tree != variant_group:
            node.node_tree = variant_group

        node.label = f"Audio Math ({count} Inputs)"
        node['ace_input_count'] = count
        node.update()

    @staticmethod
    def _build_variant(name: str, input_count: int, tree_or_type="GeometryNodeTree") -> bpy.types.NodeTree:
        ng = _ensure_variant_group(name, tree_or_type)
        if ng is None:
            raise RuntimeError(f"Could not create audio math variant '{name}'")

        for i in range(input_count):
            ng.interface.new_socket(
                f"Audio {i + 1}",
                in_out='INPUT',
                socket_type=_mapped_socket_type(tree_or_type, 'NodeSocketBundle'),
            )
        ng.interface.new_socket("Audio", in_out='OUTPUT', socket_type=_mapped_socket_type(tree_or_type, 'NodeSocketBundle'))

        ACEAudioMathManager._build_variant_internals(ng, input_count)
        return ng

    @staticmethod
    def _build_variant_internals(ng: bpy.types.NodeTree, input_count: int):
        if str(getattr(ng, "bl_idname", "") or "") == "ShaderNodeTree":
            ACEAudioMathManager._build_shader_variant_internals(ng, input_count)
            return

        input_node = ng.nodes.new('NodeGroupInput')
        output_node = ng.nodes.new('NodeGroupOutput')
        input_node.location = (-900, 0)
        output_node.location = (380, 0)

        marker = ng.nodes.new('NodeFrame')
        marker.name = ACEAudioMathManager._INTERNAL_VERSION_TAG
        marker.label = "ACE Audio Math Internals"
        marker.location = (-700, 260)

        join_geo = ng.nodes.new('GeometryNodeJoinGeometry')
        join_geo.location = (-120, 0)
        join_geo.parent = marker

        # Consume every input bundle and merge proxy geometry into one output bundle.
        for i in range(max(int(input_count), 1)):
            if i >= len(getattr(input_node, "outputs", [])):
                continue
            sep_bundle = ng.nodes.new('NodeSeparateBundle')
            sep_bundle.bundle_items.new('GEOMETRY', "Proxy Geometry")
            sep_bundle.location = (-620, -120 * i)
            sep_bundle.parent = marker
            ng.links.new(input_node.outputs[i], sep_bundle.inputs[0])

            try:
                ng.links.new(sep_bundle.outputs[0], join_geo.inputs['Geometry'])
            except Exception:
                ng.links.new(sep_bundle.outputs[0], join_geo.inputs[0])

        combine_bundle = ng.nodes.new('NodeCombineBundle')
        combine_bundle.location = (140, 0)
        combine_bundle.parent = marker
        geo_item = combine_bundle.bundle_items.new('GEOMETRY', "Proxy Geometry")
        try:
            ng.links.new(join_geo.outputs[0], combine_bundle.inputs[geo_item.name])
        except Exception:
            ng.links.new(join_geo.outputs[0], combine_bundle.inputs[0])

        if len(getattr(output_node, "inputs", [])) > 0:
            ng.links.new(combine_bundle.outputs[0], output_node.inputs[0])

    @staticmethod
    def _build_shader_variant_internals(ng: bpy.types.NodeTree, input_count: int):
        input_node = ng.nodes.new('NodeGroupInput')
        output_node = ng.nodes.new('NodeGroupOutput')
        input_node.location = (-900, 0)
        output_node.location = (380, 0)
        marker = ng.nodes.new('NodeReroute')
        marker.name = ACEAudioMathManager._INTERNAL_VERSION_TAG
        marker.location = (-1020, 260)
        outputs = list(getattr(input_node, "outputs", []) or [])
        if len(outputs) > 0 and len(getattr(output_node, "inputs", [])) > 0:
            combine_bundle = ng.nodes.new('NodeCombineBundle')
            combine_bundle.location = (-180, 0)
            item_count = 0
            for bundle_socket in outputs[:max(int(input_count), 1)]:
                try:
                    bundle_item = combine_bundle.bundle_items.new('BUNDLE', f"Audio {int(item_count + 1)}")
                except Exception:
                    try:
                        bundle_item = combine_bundle.bundle_items.new('FLOAT', f"Audio {int(item_count + 1)}")
                    except Exception:
                        continue
                try:
                    ng.links.new(bundle_socket, combine_bundle.inputs[bundle_item.name])
                except Exception:
                    try:
                        ng.links.new(bundle_socket, combine_bundle.inputs[item_count])
                    except Exception:
                        continue
                item_count += 1
            try:
                ng.links.new(combine_bundle.outputs[0], output_node.inputs[0])
            except Exception:
                pass
    
