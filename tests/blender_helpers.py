"""Helpers shared by the scripts that run inside Blender (smoke test, reference curves)."""

import importlib

import bpy  # type: ignore


def find_addon_key() -> str:
    """The module name ACE is enabled under in this session."""
    keys = [k for k in bpy.context.preferences.addons.keys() if "AudioCore" in k or k.endswith("audio_core_engine")]
    if not keys:
        raise AssertionError("The ACE add-on is not enabled in this Blender session")
    return keys[0]


def _write_test_wav(path, seconds=4.0, sample_rate=48000):
    """Write a mono 16-bit WAV: quiet 220 Hz for the first half, loud 440 Hz after.

    Played over frames 1-72 at 24 fps (3 s), so every sampled frame is inside the file.
    """
    import array
    import math
    import wave

    frames = int(seconds * sample_rate)
    half = frames // 2
    samples = array.array("h")
    for i in range(frames):
        amp, freq = (0.05, 220.0) if i < half else (0.6, 440.0)
        samples.append(int(32767 * amp * math.sin(2.0 * math.pi * freq * i / sample_rate)))
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())


def _add_ace_node(addon_key, tree, category, node_type, label, mode=None):
    """Add an ACE group node the way the Add menu does (that operator needs a node editor)."""
    gn_menu = importlib.import_module(f"{addon_key}.ui.gn_menu")
    dynamic = importlib.import_module(f"{addon_key}.nodes.dynamic_audio_operators")
    shader = tree.bl_idname == "ShaderNodeTree"
    if (category, node_type) == ("delta", "frequency_splitter"):
        group = dynamic.ACESplitterManager.get_or_create_variant(2, tree)
    else:
        if shader:
            factory = importlib.import_module(f"{addon_key}.nodes.shader.factory").ACEShaderNodeFactory
        else:
            factory = importlib.import_module(f"{addon_key}.nodes.geometry.factory").ACEDummyNodeFactory
        if factory.get_group_name(category, node_type) not in bpy.data.node_groups:
            factory.rebuild_node_group(category, node_type)  # built on first use, as by the menu
        group = bpy.data.node_groups[factory.get_group_name(category, node_type)]
    node = tree.nodes.new("ShaderNodeGroup" if shader else "GeometryNodeGroup")
    node.node_tree = group
    node.label = node.name = label
    if (category, node_type) == ("delta", "frequency_splitter"):
        node["ace_band_count"] = 2
    op = gn_menu.ACE_OT_add_node_group
    if (category, node_type) == ("source_mngmt", "src"):
        op._initialize_source_binding(node, bpy.context)
    modes = importlib.import_module(f"{addon_key}.nodes.modes")
    if importlib.import_module(f"{addon_key}.vars").node_modes(category, node_type):
        modes.apply_node_mode(node, category, node_type, mode)
        node.name = label
    op._initialize_sink_names(node, category, node_type, bpy.context)
    op._show_attr_id_socket_values(node)
    return node


def node_id(addon_key, node) -> str:
    """The node's id, as ACE computes it (nothing is stored on the node)."""
    return importlib.import_module(f"{addon_key}.nodes.identity").node_id(node)


def node_source(addon_key, node) -> str:
    """The stable id of the source a Source node reads."""
    return importlib.import_module(f"{addon_key}.nodes.source_nodes").node_source_id(node)


def sink_names(addon_key, node) -> dict:
    return importlib.import_module(f"{addon_key}.nodes.identity").sink_names(node)


def _new_gn_tree(scene, name):
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    obj = bpy.data.objects.new(f"{name}_Object", mesh)
    scene.collection.objects.link(obj)
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.is_modifier = True
    obj.modifiers.new(name, "NODES").node_group = tree
    return obj, tree


def _proxy_values(engine):
    """Float ACE attributes on each source proxy (names are generated, so read them all)."""
    values = {}
    for graph in engine.graphs.values():
        proxy = graph.get_proxy_object()
        mesh = getattr(proxy, "data", None)
        for attr in getattr(mesh, "attributes", []):
            if attr.name.startswith("ACE_") and attr.data_type == "FLOAT" and len(attr.data) > 0:
                values[attr.name] = float(attr.data[0].value)
    return values
