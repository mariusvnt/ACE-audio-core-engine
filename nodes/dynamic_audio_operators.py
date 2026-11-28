import math

import bpy # type: ignore
from .. import utils
from .geometry.factory import ACEDummyNodeFactory
from ..vars import SPLITTER_KIND


def is_automatic_count_label(label: str, bases: tuple[str, ...], unit: str) -> bool:
    """True for an empty label or one like "Splitter (3 Bands)" that the add-on wrote."""
    label = str(label or "")
    if not label or label in bases:
        return True
    for base in bases:
        prefix, suffix = f"{base} (", f" {unit})"
        if label.startswith(prefix) and label.endswith(suffix) and label[len(prefix):-len(suffix)].isdigit():
            return True
    return False


def _tree_bl_idname(tree_or_type) -> str:
    if isinstance(tree_or_type, str):
        tree_type = tree_or_type
    else:
        tree_type = getattr(tree_or_type, "bl_idname", "")
    tree_type = str(tree_type or "")
    return tree_type or "GeometryNodeTree"


def _factory_for_tree(tree_or_type):
    if _tree_bl_idname(tree_or_type) == "ShaderNodeTree":
        from .shader.factory import ACEShaderNodeFactory
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
        with utils.logged_errors("dynamic_audio_operators._ensure_variant_group"):
            bpy.data.node_groups.remove(ng)
        ng = None
    if ng is None:
        ng = bpy.data.node_groups.new(name, tree_type)
    return ng

class ACESplitterManager:
    """Manages independent variants of the Frequency Splitter."""
    _INTERNAL_VERSION_TAG = "ACE_SPLITTER_INTERNALS_V6"

    @staticmethod
    def get_or_create_variant(band_count: int, tree_or_type="GeometryNodeTree") -> bpy.types.NodeTree:
        """Retrieves (or builds) the Node Group data block for a specific band count."""
        band_count = max(2, min(int(band_count), 8))
        factory_cls = _factory_for_tree(tree_or_type)
        tree_type = _tree_bl_idname(tree_or_type)
        base_name = factory_cls.get_group_name(*SPLITTER_KIND)
        variant_name = f"{base_name}_{band_count}_bands"

        if variant_name in bpy.data.node_groups:
            ng = bpy.data.node_groups[variant_name]
            if getattr(ng, "bl_idname", "") == tree_type:
                if int(ng.get("ace_bands", 0) or 0) != band_count:
                    from .groups import tag_owned

                    tag_owned(ng, *SPLITTER_KIND)
                    ng["ace_bands"] = band_count
                ACESplitterManager._ensure_split_socket_limits(ng)
                ACESplitterManager._ensure_variant_internals(ng, band_count)
                return ng
            with utils.logged_errors("dynamic_audio_operators.ACESplitterManager.get_or_create_variant"):
                bpy.data.node_groups.remove(ng)
            
        return ACESplitterManager._build_variant(variant_name, band_count, tree_or_type)

    @staticmethod
    def refresh_existing_variants(tree_or_type="GeometryNodeTree"):
        """Migrate already-existing splitter variants in-place so older projects immediately benefit from range constraints and non-greyed controls."""
        tree_type = _tree_bl_idname(tree_or_type)
        base_name = _factory_for_tree(tree_or_type).get_group_name(*SPLITTER_KIND)
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
            with utils.logged_errors("dynamic_audio_operators.ACESplitterManager.refresh_existing_variants"):
                ACESplitterManager.get_or_create_variant(int(mid), tree_or_type)
        # Ensure baseline variant exists for fresh sessions.
        ACESplitterManager.get_or_create_variant(2, tree_or_type)

    @staticmethod
    def apply_variant_to_node(node: bpy.types.Node, band_count: int):
        """Swaps the node_tree of a specific node instance to the requested variant."""
        if not node or node.type != 'GROUP': return
        
        # Get the blueprint
        variant_group = ACESplitterManager.get_or_create_variant(band_count, getattr(node, "id_data", None))
        
        # Swap it
        if node.node_tree != variant_group:
            node.node_tree = variant_group
        
        # Automatic labels follow the band count; a name the user gave is kept.
        if is_automatic_count_label(node.label, ("Splitter", "Frequency Splitter"), "Bands"):
            node.label = f"Splitter ({band_count} Bands)"
        
        # Store state in custom property so UI knows what to show
        node['ace_band_count'] = band_count

        # Force refresh
        node.update()

    @staticmethod
    def _default_split_freq(split_idx: int, split_count: int, low_hz: float = 20.0, high_hz: float = 20000.0) -> float:
        if split_count <= 0:
            return 1000.0
        low = max(float(low_hz), 1e-3)
        high = max(float(high_hz), low + 1e-3)
        ratio = math.pow(high / low, 1.0 / float(split_count + 1))
        return float(low * math.pow(ratio, float(split_idx + 1)))

    @staticmethod
    def _set_interface_socket_limits(sock):
        with utils.logged_errors("dynamic_audio_operators.ACESplitterManager._set_interface_socket_limits"):
            sock.min_value = 0.0
        with utils.logged_errors("dynamic_audio_operators.ACESplitterManager._set_interface_socket_limits#2"):
            sock.max_value = 20000.0
        with utils.logged_errors("dynamic_audio_operators.ACESplitterManager._set_interface_socket_limits#3"):
            sock.default_value = float(min(max(float(sock.default_value), 20.0), 20000.0))

    @staticmethod
    def _ensure_split_socket_limits(ng: bpy.types.NodeTree):
        for item in getattr(getattr(ng, "interface", None), "items_tree", []):
            try:
                if getattr(item, "in_out", "") == 'INPUT' and str(getattr(item, "name", "")).startswith("Split Freq"):
                    ACESplitterManager._set_interface_socket_limits(item)
            except Exception as exc:
                utils.note_swallowed("dynamic_audio_operators.ACESplitterManager._ensure_split_socket_limits", exc)
                continue

    @staticmethod
    def _variant_has_current_internals(ng: bpy.types.NodeTree) -> bool:
        return ng.get("ace_internals") == ACESplitterManager._INTERNAL_VERSION_TAG

    @staticmethod
    def _ensure_variant_internals(ng: bpy.types.NodeTree, band_count: int):
        if ACESplitterManager._variant_has_current_internals(ng):
            return
        with utils.logged_errors("dynamic_audio_operators.ACESplitterManager._ensure_variant_internals"):
            ng.nodes.clear()
        ACESplitterManager._build_variant_internals(ng, band_count)

    @staticmethod
    def _build_variant_internals(ng: bpy.types.NodeTree, band_count: int):
        """Each band output is a bundle of the input and the split frequencies.

        A facade: the engine reads the split frequencies from the node, never from the bundle.
        Packing them in is what makes Blender count them as used, so it doesn't grey them out.
        In Geometry Nodes the input's Proxy Geometry is carried along; a shader bundle carries
        nothing, so the input goes in whole.
        """
        from .layout import arrange

        is_shader = str(getattr(ng, "bl_idname", "") or "") == "ShaderNodeTree"
        split_count = max(0, int(band_count) - 1)
        input_node = ng.nodes.new('NodeGroupInput')
        output_node = ng.nodes.new('NodeGroupOutput')
        if is_shader:
            carried, item_type = input_node.outputs[0], 'BUNDLE'
            item_name = "Audio"
        else:
            sep_bundle = ng.nodes.new('NodeSeparateBundle')
            sep_bundle.bundle_items.new('GEOMETRY', "Proxy Geometry")
            ng.links.new(input_node.outputs[0], sep_bundle.inputs[0])
            carried, item_type = sep_bundle.outputs[0], 'GEOMETRY'
            item_name = "Proxy Geometry"
        for out_idx in range(min(band_count, len(output_node.inputs))):
            combine = ng.nodes.new('NodeCombineBundle')
            combine.label = f"{output_node.inputs[out_idx].name} Band"
            combine.bundle_items.new(item_type, item_name)
            for split_idx in range(split_count):
                combine.bundle_items.new('FLOAT', f"Split Freq {split_idx + 1}")
            # Sockets by name: an item returned by bundle_items.new is stale after the next new.
            ng.links.new(carried, combine.inputs[item_name])
            for split_idx in range(split_count):
                name = f"Split Freq {split_idx + 1}"
                ng.links.new(input_node.outputs[name], combine.inputs[name])
            ng.links.new(combine.outputs[0], output_node.inputs[out_idx])
        arrange(ng)
        ng["ace_internals"] = ACESplitterManager._INTERNAL_VERSION_TAG

    @staticmethod
    def _build_variant(name: str, band_count: int, tree_or_type="GeometryNodeTree") -> bpy.types.NodeTree:
        """Constructs the internal node graph for a splitter variant."""
        ng = _ensure_variant_group(name, tree_or_type)
        if ng is None:
            raise RuntimeError(f"Could not create splitter variant '{name}'")
        from .groups import tag_owned

        tag_owned(ng, *SPLITTER_KIND)
        ng["ace_bands"] = int(band_count)
        
        
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
