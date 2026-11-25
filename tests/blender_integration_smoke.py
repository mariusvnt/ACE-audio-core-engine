"""
Run inside Blender (background or UI) after enabling the add-on.

Headless, from the repo root (enables the add-on from this checkout):
  python tests/run_in_blender.py
"""

import types
import bpy  # type: ignore
import importlib
import os
import tempfile
import wave


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _check_object_and_material_edits_queue_recompile(engine, scene, obj, modifier, gn_tree, shader_tree):
    """Depsgraph updates on an object or material reach request_recompile with the original trees."""
    settings = scene.audiocore.engine_settings
    requested = []
    original_request = engine.frontends.request_recompile
    engine.frontends.request_recompile = lambda tree, **_kw: requested.append(tree)
    try:
        bpy.context.view_layer.update()
        requested.clear()
        modifier.show_viewport = not modifier.show_viewport
        bpy.context.view_layer.update()
        obj.data.materials[0].diffuse_color = (0.2, 0.4, 0.6, 1.0)
        bpy.context.view_layer.update()
    finally:
        engine.frontends.request_recompile = original_request
    pointers = {t.as_pointer() for t in requested if t is not None}
    _assert(gn_tree.as_pointer() in pointers, "Object edit did not queue its Geometry Nodes frontend")
    _assert(shader_tree.as_pointer() in pointers, "Material edit did not queue its shader frontend")
    _assert(all(not t.is_evaluated for t in requested if t is not None), "Recompile queued an evaluated copy")


def _stale_ace_groups(addon_key):
    dummy_tree = importlib.import_module(f"{addon_key}.nodes.geometry.factory")
    shader_tree = importlib.import_module(f"{addon_key}.nodes.shader.factory")
    ace_nodes = importlib.import_module(f"{addon_key}.vars").ACE_NODES
    gn, sh = dummy_tree.ACEDummyNodeFactory, shader_tree.ACEShaderNodeFactory
    stale = []
    for cat_key, category in ace_nodes.items():
        for type_key, node_def in category["nodes"].items():
            group = bpy.data.node_groups.get(gn.get_group_name(cat_key, type_key))
            if group is not None and gn._node_group_needs_refresh(group, type_key):
                stale.append(group.name)
            group = bpy.data.node_groups.get(sh.get_group_name(cat_key, type_key))
            if group is not None and sh._node_group_needs_refresh(group, node_def, cat_key, type_key):
                stale.append(group.name)
    return stale


def _check_loaded_files_get_current_node_groups(addon_key):
    """Fresh groups are never stale, and a file's outdated groups are rebuilt on load."""
    stale = _stale_ace_groups(addon_key)
    _assert(not stale, f"Freshly built ACE groups count as stale (rebuilt on every load): {stale}")
    dummy_tree = importlib.import_module(f"{addon_key}.nodes.geometry.factory")
    lufs_name = dummy_tree.ACEDummyNodeFactory.get_group_name("stream", "loudness")
    bpy.data.node_groups[lufs_name].interface.new_socket("Leftover", in_out="INPUT", socket_type="NodeSocketFloat")
    path = os.path.join(tempfile.mkdtemp(prefix="ace_smoke_"), "outdated_groups.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)
    names = [i.name for i in bpy.data.node_groups[lufs_name].interface.items_tree if i.in_out == "INPUT"]
    _assert("Leftover" not in names, f"Outdated ACE group kept after load: {names}")
    _assert(not _stale_ace_groups(addon_key), "ACE groups still stale after load")


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
    dummy_tree = importlib.import_module(f"{addon_key}.nodes.geometry.factory")
    dynamic = importlib.import_module(f"{addon_key}.nodes.dynamic_audio_operators")
    id_management = importlib.import_module(f"{addon_key}.core.id_management")
    if (category, node_type) == ("delta", "frequency_splitter"):
        group = dynamic.ACESplitterManager.get_or_create_variant(2, tree)
    else:
        group = bpy.data.node_groups[dummy_tree.ACEDummyNodeFactory.get_group_name(category, node_type)]
    node = tree.nodes.new("GeometryNodeGroup")
    node.node_tree = group
    existing = [n.get("ace_id") for n in tree.nodes if n.get("ace_id")]
    node["ace_id"] = id_management.ACEIdManager.generate_node_id(category, node_type, existing_ids=existing)
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
    op._initialize_auto_attr_ids(node, bpy.context)
    op._show_attr_id_socket_values(node)
    return node


def _new_gn_tree(scene, name):
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    obj = bpy.data.objects.new(f"{name}_Object", mesh)
    scene.collection.objects.link(obj)
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.is_modifier = True
    obj.modifiers.new(name, "NODES").node_group = tree
    return obj, tree


def _check_old_stream_nodes_migrate(addon_key, scene):
    """A saved per-method node (here LUFS) becomes a Loudness node in LUFS mode, keeping links and values."""
    modes = importlib.import_module(f"{addon_key}.nodes.modes")
    reader_mod = importlib.import_module(f"{addon_key}.nodes.geometry.reader")
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_Migrate")
    try:
        source = _add_ace_node(addon_key, tree, "source_mngmt", "src", "Source")
        old_group = bpy.data.node_groups.new(".ACE_stream_lufs_smoke", "GeometryNodeTree")
        old_group.interface.new_socket("Audio", in_out="INPUT", socket_type="NodeSocketBundle")
        old_group.interface.new_socket("Window (s)", in_out="INPUT", socket_type="NodeSocketFloat")
        old_group.interface.new_socket("Loudness", in_out="OUTPUT", socket_type="NodeSocketFloat")
        old_group.interface.new_socket("Attribute ID", in_out="INPUT", socket_type="NodeSocketString")
        old = tree.nodes.new("GeometryNodeGroup")
        old.node_tree = old_group
        old["ace_id"] = "ACE_stream_lufs_abc123"
        old.label = old.name = "Loudness (LUFS)"
        old.inputs["Window (s)"].default_value = 1.25
        old.inputs["Attribute ID"].default_value = "ACE_smoke_attr"
        math = tree.nodes.new("ShaderNodeMath")
        tree.links.new(source.outputs["Audio"], old.inputs["Audio"])
        tree.links.new(old.outputs["Loudness"], math.inputs[0])

        _assert(modes.migrate_legacy_stream_nodes() >= 1, "No old stream node was migrated")
        node = next(n for n in tree.nodes if str(n.get("ace_id", "")).endswith("_abc123"))
        _assert(node.get("ace_id") == "ACE_stream_loudness_abc123", f"Migrated id: {node.get('ace_id')}")
        _assert(node.node_tree.name == ".ACE_stream_loudness", f"Migrated group: {node.node_tree.name}")
        _assert(node.get("ace_mode") == "lufs", f"Migrated mode: {node.get('ace_mode')}")
        _assert(node.label == "Loudness (LUFS)", f"Migrated label: {node.label}")
        _assert(abs(node.inputs["Window (s)"].default_value - 1.25) < 1e-6, "Window value lost")
        _assert(node.inputs["Attribute ID"].default_value == "ACE_smoke_attr", "Attribute ID lost")
        _assert(node.inputs["Audio"].is_linked, "Audio input link lost")
        _assert(any(l.to_node == math for l in node.outputs[0].links), "Output link lost")
        abstract = reader_mod.ACEDummyTreeReader(tree).build()
        read = abstract.nodes.get("ACE_stream_loudness_abc123")
        _assert(read is not None and read.parameters.get("mode") == "lufs", "Reader misses the migrated node")
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)
        group = bpy.data.node_groups.get(".ACE_stream_lufs_smoke")
        if group is not None:
            bpy.data.node_groups.remove(group)


def _check_operator_list_follows_tree(addon_key, scene):
    """Active Operators: a folded tree that follows edits (without compiling), selects both ways, renames."""
    sync = importlib.import_module(f"{addon_key}.ui.panel.sync")
    selection = importlib.import_module(f"{addon_key}.ui.panel.selection")
    operator_tree = importlib.import_module(f"{addon_key}.ui.panel.operator_tree")
    modes = importlib.import_module(f"{addon_key}.nodes.modes")
    settings = scene.audiocore.engine_settings
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_List")
    originals = (sync._find_active_node_tree, selection._find_active_node_tree)
    sync._find_active_node_tree = selection._find_active_node_tree = lambda *_a, **_k: tree
    sep = " \u203a "
    try:
        src = scene.audiocore.audio_sources[scene.audiocore.active_audio_source_index]

        def rows():
            sync._sync_ui_dummy_tree(bpy.context)
            return [("  " * item.depth) + item.name for item in src.ui_nodes]

        source = _add_ace_node(addon_key, tree, "source_mngmt", "src", "Source")
        loud = _add_ace_node(addon_key, tree, "stream", "loudness", "Loudness")
        root = source.label or source.name
        _assert(rows() == [root], f"An unconnected operator is listed: {rows()}")
        tree.links.new(source.outputs["Audio"], loud.inputs["Audio"])
        _assert(rows() == [f"{root}{sep}Loudness (LUFS)"], f"A connected chain is not one row: {rows()}")

        split = _add_ace_node(addon_key, tree, "delta", "frequency_splitter", "Splitter")
        rms = _add_ace_node(addon_key, tree, "stream", "loudness", "Loudness", mode="rms")
        stereo = _add_ace_node(addon_key, tree, "stream", "stereo", "Stereo")
        tree.links.new(source.outputs["Audio"], split.inputs[0])
        tree.links.new(split.outputs[0], rms.inputs["Audio"])
        tree.links.new(split.outputs[1], stereo.inputs["Audio"])
        low, high = split.outputs[0].name, split.outputs[1].name
        expected = [
            root,
            "  Loudness (LUFS)",
            "  Splitter",
            f"    {low}{sep}Loudness (RMS)",
            f"    {high}{sep}Stereo (Width)",
        ]
        _assert(rows() == expected, f"Folded tree: {rows()}")

        # List -> editor.
        _assert("FINISHED" in bpy.ops.ace.select_operator(node_id=stereo["ace_id"]), "select_operator failed")
        _assert(tree.nodes.active == stereo and stereo.select and not loud.select, "List click did not select the node")
        _assert(src.active_ui_node_id == stereo["ace_id"], "List click did not select the operator")
        _assert(src.active_ui_node_index == 4, f"Active row: {src.active_ui_node_index}")

        # Editor -> list, unfolding a folded branch.
        sync.toggle_branch(split["ace_id"])
        shown = operator_tree.visible_rows(sync.current_rows(), sync.collapsed_keys())
        _assert(sum(shown) == 3, f"Folding the splitter left {sum(shown)} rows")
        tree.nodes.active = rms
        sync._sync_editor_active(tree, src)
        _assert(src.active_ui_node_id == rms["ace_id"], "Editor selection did not reach the list")
        _assert(split["ace_id"] not in sync.collapsed_keys(), "Selecting a folded operator did not unfold it")
        _assert(src.active_ui_node_index == 3, f"Active row: {src.active_ui_node_index}")

        # Rename box (double-click): list and node title follow; a custom name survives a mode switch;
        # an empty name restores the automatic one.
        wm = bpy.context.window_manager

        def rename_rms(value):
            selection.start_rename(rms["ace_id"])
            wm.ace_rename_name = value

        selection.start_rename(rms["ace_id"])
        _assert(wm.ace_rename_name == rms.label, f"Rename box shows {wm.ace_rename_name!r}")
        rename_rms("Kick Level")
        _assert(rms.label == "Kick Level" and rms.name == "Kick Level", f"Rename: {rms.label!r} / {rms.name!r}")
        _assert(f"    {low}{sep}Kick Level" in rows(), f"Renamed row: {rows()}")
        modes.apply_node_mode(rms, "stream", "loudness", "peak")
        _assert(rms.label == "Kick Level", "A mode switch replaced the custom name")
        rename_rms("")
        _assert(rms.label == "Loudness (Peak)", f"Empty name gave {rms.label!r}")
        rename_rms("Bass")
        rows()

        # A click on the row of the selected name keeps it; another row selects its last name.
        row_idx = operator_tree.row_of_node(sync.current_rows(), rms["ace_id"])
        src.active_ui_node_index = row_idx
        _assert(src.active_ui_node_id == rms["ace_id"], "Clicking the selected name's row lost the selection")
        lufs_row = operator_tree.row_of_node(sync.current_rows(), loud["ace_id"])
        src.active_ui_node_index = lufs_row
        _assert(src.active_ui_node_id == loud["ace_id"], "Clicking another row did not select it")

        # Splitter band strip: band widths on the log axis; the old ramp editor group is removed.
        splitter_mod = importlib.import_module(f"{addon_key}.ui.panel.splitter")
        split.inputs[1].default_value = 632.46  # geometric middle of 20 Hz .. 20 kHz
        widths = splitter_mod.splitter_band_widths(split, scene)
        _assert(len(widths) == 2 and abs(sum(widths) - 1.0) < 1e-6, f"Band widths: {widths}")
        _assert(abs(widths[0] - 0.5) < 0.01, f"Band widths do not follow the log axis: {widths}")
        old_ramp = bpy.data.node_groups.new(splitter_mod._SPLITTER_RAMP_NODETREE_NAME, "ShaderNodeTree")
        old_ramp.nodes.new("ShaderNodeValToRGB")
        splitter_mod.remove_legacy_ramp_group()
        _assert(bpy.data.node_groups.get(splitter_mod._SPLITTER_RAMP_NODETREE_NAME) is None, "Old ramp group kept")
        socket_color = tuple(split.inputs[0].draw_color(bpy.context, split))[:3]
        _assert(all(abs(a - b) < 1e-4 for a, b in zip(socket_color, selection._BUNDLE_SOCKET_COLOR)),
                f"Bundle socket colour is {socket_color}")
        selection._match_band_fill(bpy.context, socket_color)
        ui = bpy.context.preferences.themes[0].user_interface
        under = [ui.panel_back[i] * (1 - ui.wcol_box.inner[3]) + ui.wcol_box.inner[i] * ui.wcol_box.inner[3] for i in range(3)]
        seen = [0.5 * wm.ace_band_fill[i] + 0.5 * under[i] for i in range(3)]
        _assert(all(abs(a - b) < 0.01 for a, b in zip(seen, socket_color)), f"Strip looks {seen}, sockets {socket_color}")

        tree.links.remove(stereo.inputs["Audio"].links[0])
        # One band left: the splitter has no branch any more and folds into a single row.
        _assert(rows()[-1] == f"  Splitter{sep}{low}{sep}Bass", f"After disconnecting: {rows()}")
    finally:
        sync._find_active_node_tree, selection._find_active_node_tree = originals
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)


def _check_viewer_previews_audio(addon_key, scene, engine):
    """A Viewer node fed by an audio output compiles to the same preview as an Audio Preview node."""
    sync = importlib.import_module(f"{addon_key}.ui.panel.sync")
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_Viewer")
    original_tree = sync._find_active_node_tree
    try:
        source = _add_ace_node(addon_key, tree, "source_mngmt", "src", "Source")
        split = _add_ace_node(addon_key, tree, "delta", "frequency_splitter", "Splitter")
        loud = _add_ace_node(addon_key, tree, "stream", "loudness", "Loudness")
        tree.links.new(source.outputs["Audio"], split.inputs[0])
        tree.links.new(split.outputs[1], loud.inputs["Audio"])
        viewer = tree.nodes.new("GeometryNodeViewer")
        viewer.viewer_items.new("BUNDLE", "Audio")  # what linking in the editor adds

        def previews():
            engine.compiler.compile_scene_frontends(scene)
            nodes = engine.abstract_tree_ref.nodes
            sinks = {n.id for n in nodes.values() if n.type == "preview_sink" and n.parameters.get("viewer")}
            fed = [(l.source_node_id, l.source_socket_idx) for l in engine.abstract_tree_ref.links if l.target_node_id in sinks]
            manual = {t for g in engine.graphs.values() for t in g.manual_preview_task_ids}
            _assert(manual == sinks, f"Viewer sinks {sinks}, manual previews {manual}")
            return fed

        _assert(previews() == [], "An unconnected Viewer previews")
        tree.links.new(split.outputs[0], viewer.inputs[0])
        _assert(viewer.inputs[0].type == "BUNDLE", f"Viewer socket type: {viewer.inputs[0].type}")
        _assert(previews() == [(split["ace_id"], 0)], f"Viewer on the low band: {previews()}")

        sync._find_active_node_tree = lambda _context, **_kw: tree
        sync._sync_ui_dummy_tree(bpy.context, force=True)
        labels = [seg.label for row in sync.current_rows() for seg in row.segments]
        _assert(viewer.name not in labels, f"Viewer listed as an operator: {labels}")

        while viewer.inputs[0].is_linked:
            tree.links.remove(viewer.inputs[0].links[0])
        viewer.viewer_items.new("FLOAT", "Value")
        tree.links.new(loud.outputs[0], viewer.inputs["Value"])  # a float, not audio
        _assert(previews() == [], "A Viewer on a value previews")
    finally:
        sync._find_active_node_tree = original_tree
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)


def _check_computed_link_values(addon_key, scene):
    """Values computed for a linked input match what Blender's own evaluation gives."""
    driven = importlib.import_module(f"{addon_key}.nodes.driven_values")
    tree = bpy.data.node_groups.new("ACE_Smoke_Driven", "GeometryNodeTree")
    mesh = bpy.data.meshes.new("ACE_Smoke_Driven")
    mesh.from_pydata([(0.0, 0.0, 0.0)], [], [])
    obj = bpy.data.objects.new("ACE_Smoke_Driven", mesh)
    scene.collection.objects.link(obj)
    try:
        tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        cut = tree.interface.new_socket("Cut", in_out="INPUT", socket_type="NodeSocketFloat")
        modifier = obj.modifiers.new("ACE_Smoke_Driven", "NODES")
        modifier.node_group = tree
        group_in = tree.nodes.new("NodeGroupInput")
        group_out = tree.nodes.new("NodeGroupOutput")
        store = tree.nodes.new("GeometryNodeStoreNamedAttribute")
        store.data_type = "FLOAT"
        store.inputs["Name"].default_value = "probe"
        tree.links.new(group_in.outputs["Geometry"], store.inputs["Geometry"])
        tree.links.new(store.outputs["Geometry"], group_out.inputs["Geometry"])
        target = store.inputs["Value"]

        def blender_value():
            obj.update_tag()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            return obj.evaluated_get(depsgraph).data.attributes["probe"].data[0].value

        def check(what):
            expected, got = blender_value(), driven.linked_value(target, scene)
            _assert(got is not None and abs(got - expected) <= 1e-4 * max(1.0, abs(expected)),
                    f"{what}: computed {got}, Blender {expected}")

        value = tree.nodes.new("ShaderNodeValue")
        math_node = tree.nodes.new("ShaderNodeMath")
        reroute = tree.nodes.new("NodeReroute")
        tree.links.new(value.outputs[0], reroute.inputs[0])
        tree.links.new(reroute.outputs[0], math_node.inputs[0])
        tree.links.new(math_node.outputs[0], target)
        operands = ((7.3, 2.0, 0.5), (-7.3, 2.5, 1.5), (0.4, 0.0, 0.0), (-2.0, -3.0, 0.2), (5.5, 4.0, -1.0))
        operations = [item.identifier for item in math_node.bl_rna.properties["operation"].enum_items]
        _assert(set(operations) == set(driven._MATH), f"Math operations: {set(operations) ^ set(driven._MATH)}")
        for operation in operations:
            math_node.operation = operation
            for a, b, c in operands:
                value.outputs[0].default_value = a
                math_node.inputs[1].default_value = b
                math_node.inputs[2].default_value = c
                check(f"{operation} {a} {b} {c}")
        math_node.operation, math_node.use_clamp = "ADD", True
        check("clamped add")

        remap = tree.nodes.new("ShaderNodeMapRange")
        tree.links.new(value.outputs[0], remap.inputs[0])
        tree.links.new(remap.outputs[0], target)
        for mode in ("LINEAR", "STEPPED", "SMOOTHSTEP", "SMOOTHERSTEP"):
            remap.interpolation_type = mode
            for clamp in (False, True):
                remap.clamp = clamp
                for a, bounds in ((0.3, (0.0, 1.0, 20.0, 2000.0)), (1.7, (0.0, 1.0, 900.0, 100.0)),
                                  (0.5, (2.0, 2.0, 0.0, 10.0))):
                    value.outputs[0].default_value = a
                    for index, bound in enumerate(bounds, start=1):
                        remap.inputs[index].default_value = bound
                    remap.inputs[5].default_value = 4.0
                    check(f"map range {mode} clamp={clamp} {a} {bounds}")

        clamp_node = tree.nodes.new("ShaderNodeClamp")
        tree.links.new(value.outputs[0], clamp_node.inputs[0])
        tree.links.new(clamp_node.outputs[0], target)
        for clamp_type in ("MINMAX", "RANGE"):
            clamp_node.clamp_type = clamp_type
            for low, high in ((1.0, 3.0), (3.0, 1.0)):
                clamp_node.inputs[1].default_value, clamp_node.inputs[2].default_value = low, high
                value.outputs[0].default_value = 2.5
                check(f"clamp {clamp_type} {low} {high}")

        integer = tree.nodes.new("FunctionNodeInputInt")
        integer.integer = 42
        tree.links.new(integer.outputs[0], target)
        check("integer")
        tree.links.new(group_in.outputs["Cut"], target)
        modifier.properties.inputs.__getattribute__(cut.identifier).value = 1234.0
        check("modifier input")
        clock = tree.nodes.new("GeometryNodeInputSceneTime")
        for output in ("Frame", "Seconds"):
            tree.links.new(clock.outputs[output], target)
            check(f"scene time {output}")

        noise = tree.nodes.new("ShaderNodeTexNoise")
        tree.links.new(noise.outputs[0], target)
        _assert(driven.linked_value(target, scene) is None, "A noise texture was computed")
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)
        bpy.data.node_groups.remove(tree)


def _check_driven_splits(addon_key, scene, engine):
    """A linked split frequency reaches the engine and the strip, and follows without recompiling."""
    split_freqs = importlib.import_module(f"{addon_key}.nodes.split_freqs")
    splitter = importlib.import_module(f"{addon_key}.ui.panel.splitter")
    dynamic = importlib.import_module(f"{addon_key}.nodes.dynamic_audio_operators")
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_DrivenSplit")
    try:
        source = _add_ace_node(addon_key, tree, "source_mngmt", "src", "Source")
        split = _add_ace_node(addon_key, tree, "delta", "frequency_splitter", "Splitter")
        dynamic.ACESplitterManager.apply_variant_to_node(split, 3)
        loud = _add_ace_node(addon_key, tree, "stream", "loudness", "Loudness")
        tree.links.new(source.outputs["Audio"], split.inputs[0])
        tree.links.new(split.outputs[1], loud.inputs["Audio"])
        split.inputs[1].default_value, split.inputs[2].default_value = 200.0, 5000.0
        value = tree.nodes.new("ShaderNodeValue")
        value.outputs[0].default_value = 5.0
        times = tree.nodes.new("ShaderNodeMath")
        times.operation = "MULTIPLY"
        times.inputs[1].default_value = 10.0
        tree.links.new(value.outputs[0], times.inputs[0])
        tree.links.new(times.outputs[0], split.inputs[1])

        def task_params():
            return next(task.parameters for graph in engine.graphs.values() for node_id, task in graph.tasks.items()
                        if node_id == split["ace_id"])

        def states():
            return [(None if v is None else round(v), s) for _sock, v, s in split_freqs.split_freq_states(split, scene)]

        engine.compiler.compile_scene_frontends(scene)
        _assert(task_params().get("in_1") == 50.0 and task_params().get("in_2") == 5000.0,
                f"Compiled split frequencies: {task_params()}")
        _assert(states() == [(50, "linked"), (5000, "own")], f"States: {states()}")
        narrow = splitter.splitter_band_widths(split, scene)

        value.outputs[0].default_value = 100.0
        split_freqs.refresh_driven_splits(engine.graphs.values(), scene)
        _assert(task_params()["in_1"] == 1000.0, f"Engine did not follow: {task_params()['in_1']}")
        _assert(any(g.param_overrides.get(split["ace_id"], {}).get("in_1") == 1000.0 for g in engine.graphs.values()),
                "Workers are not told about the new value")
        _assert(splitter.splitter_band_widths(split, scene)[0] > narrow[0] + 0.3, "The strip did not follow")
        engine.frame.execute_next_frame(scene, force=True)

        noise = tree.nodes.new("ShaderNodeTexNoise")
        tree.links.new(noise.outputs[0], split.inputs[1])
        _assert(states()[0] == (200, "unreadable"), f"Unreadable link: {states()}")
        engine.compiler.compile_scene_frontends(scene)
        _assert(task_params().get("in_1") == 200.0, f"Unreadable link left the cut out: {task_params()}")
        tree.links.new(loud.outputs[0], split.inputs[2])
        _assert(states()[1] == (None, "audio"), f"Audio-driven link: {states()}")
        _assert(len(splitter.splitter_band_widths(split, scene)) == 2, "An unknown cut was drawn")
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)


def _check_selection_follows_sources(addon_key, scene):
    """Clicking another source's node activates it; the list's active row holds the selected name."""
    sync = importlib.import_module(f"{addon_key}.ui.panel.sync")
    operator_tree = importlib.import_module(f"{addon_key}.ui.panel.operator_tree")
    ace = scene.audiocore
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_TwoSources")
    original_tree = sync._find_active_node_tree
    first_index = ace.active_audio_source_index
    count = len(ace.audio_sources)
    try:
        bpy.ops.audiocore.add_audio_source()
        bpy.ops.audiocore.add_audio_source()
        a_index, b_index = count, count + 1
        pairs = []
        for index in (a_index, b_index):
            ace.active_audio_source_index = index
            source = _add_ace_node(addon_key, tree, "source_mngmt", "src", f"Source {index}")
            loud = _add_ace_node(addon_key, tree, "stream", "loudness", f"Loud {index}")
            tree.links.new(source.outputs["Audio"], loud.inputs["Audio"])
            pairs.append((source, loud))
        sync._find_active_node_tree = lambda _context, **_kw: tree

        def active():
            return ace.audio_sources[ace.active_audio_source_index]

        def consistent(what):
            src = active()
            row = sync.current_rows()[src.active_ui_node_index]
            _assert(any(seg.node_id == src.active_ui_node_id for seg in row.segments),
                    f"{what}: selected {src.active_ui_node_id} is not on row {src.active_ui_node_index}")

        ace.active_audio_source_index = a_index
        tree.nodes.active = pairs[0][1]
        sync._sync_ui_dummy_tree(bpy.context, force=True)
        sync._sync_editor_active(tree, active())
        tree.nodes.active = pairs[1][1]  # click on the other source's graph
        sync._sync_editor_active(tree, active())
        _assert(ace.active_audio_source_index == b_index, f"Source not switched: {ace.active_audio_source_index}")
        _assert(active().active_ui_node_id == pairs[1][1]["ace_id"], "Clicked node not selected")
        consistent("after a click")

        # Switching sources in the list, with stale rows and selections on both sides.
        for source_index, loud in ((a_index, pairs[0][1]), (b_index, pairs[1][1]), (a_index, pairs[0][1])):
            src = ace.audio_sources[source_index]
            src.active_ui_node_id = pairs[0][1]["ace_id"] if source_index == b_index else loud["ace_id"]
            ace.active_audio_source_index = source_index
            consistent(f"after switching to {source_index}")
        _assert(operator_tree.row_of_node(sync.current_rows(), pairs[0][1]["ace_id"]) >= 0, "Rows not rebuilt on switch")
    finally:
        sync._find_active_node_tree = original_tree
        while len(ace.audio_sources) > count:
            ace.active_audio_source_index = len(ace.audio_sources) - 1
            bpy.ops.audiocore.remove_audio_source()
        ace.active_audio_source_index = first_index
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)


def _check_new_source_takes_input_name(addon_key, scene):
    """An unnamed source takes the name of its first file or live input; a chosen name is kept."""
    properties = importlib.import_module(f"{addon_key}.properties")
    ace = scene.audiocore
    count = len(ace.audio_sources)
    first_index = ace.active_audio_source_index
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_Names")
    folder = tempfile.mkdtemp(prefix="ace_names_")
    try:
        paths = []
        for stem in ("Jump stance", "Other take"):
            paths.append(os.path.join(folder, f"{stem}.wav"))
            with wave.open(paths[-1], "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(48000)
                handle.writeframes(b"\x00\x00" * 4800)

        bpy.ops.audiocore.add_audio_source()
        source = ace.audio_sources[-1]
        placed = _add_ace_node(addon_key, tree, "source_mngmt", "src", source.name)
        source.filepath = paths[0]
        _assert(source.name == "Jump stance", f"Name from the file: {source.name!r}")
        _assert(placed.label == "Jump stance", f"Placed Source node keeps {placed.label!r}")
        source.filepath = paths[1]
        _assert(source.name == "Jump stance", f"A second file renamed the source: {source.name!r}")

        bpy.ops.audiocore.add_audio_source()
        other = ace.audio_sources[-1]
        other.filepath = paths[0]
        _assert(other.name == "Jump stance 2", f"Taken name: {other.name!r}")

        bpy.ops.audiocore.add_audio_source()
        live = ace.audio_sources[-1]
        live.name = "Audio Source 9"
        live.source_type = "LIVE"
        _assert(live.name == "Audio Source 9", f"Auto input renamed the source: {live.name!r}")
        live.live_transport = "UDP"
        _assert(live.name == f"UDP {live.live_udp_port}", f"Name from UDP: {live.name!r}")

        bpy.ops.audiocore.add_audio_source()
        device = ace.audio_sources[-1]
        device.source_type = "LIVE"
        items = [("__AUTO__", "Auto", ""), ("3", "3: Microphone (MOTU M2) (MME)", "")]
        saved = (properties._LIVE_DEVICE_CACHE, dict(properties._LIVE_DEVICE_NAMES))
        properties._LIVE_DEVICE_CACHE = items
        properties._LIVE_DEVICE_NAMES["3"] = "Microphone (MOTU M2)"
        try:
            device.live_device_selector = "3"
            _assert(device.name == "Microphone (MOTU M2)", f"Name from the device: {device.name!r}")
        finally:
            properties._LIVE_DEVICE_CACHE = saved[0]
            properties._LIVE_DEVICE_NAMES.clear()
            properties._LIVE_DEVICE_NAMES.update(saved[1])
    finally:
        while len(ace.audio_sources) > count:
            ace.active_audio_source_index = len(ace.audio_sources) - 1
            bpy.ops.audiocore.remove_audio_source()
        ace.active_audio_source_index = first_index
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)


def _check_splitter_sliders(addon_key, scene):
    """Split frequency sliders write their own socket; the release sorts the sockets."""
    splitter = importlib.import_module(f"{addon_key}.ui.panel.splitter")
    split_freqs = importlib.import_module(f"{addon_key}.nodes.split_freqs")
    dynamic = importlib.import_module(f"{addon_key}.nodes.dynamic_audio_operators")
    ops = importlib.import_module(f"{addon_key}.ui.n_panel_operators")
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_Splitter")
    original_selected = ops._resolve_panel_selected_node
    try:
        split = _add_ace_node(addon_key, tree, "delta", "frequency_splitter", "Splitter")
        dynamic.ACESplitterManager.apply_variant_to_node(split, 4)

        def sockets():
            return [round(sock.default_value) for sock in list(split.inputs)[1:]]

        def sliders():
            return [round(getattr(split, f"ace_split_freq_{i + 1}")) for i in range(len(split.inputs) - 1)]

        split.ace_split_freq_1, split.ace_split_freq_2, split.ace_split_freq_3 = 100.0, 500.0, 2000.0
        _assert(sockets() == [100, 500, 2000], f"Slider writes: {sockets()}")
        _assert(hasattr(bpy.types, "ACE_OT_sort_split_freqs_on_release"), "Release watcher not registered")

        # While dragging, nothing moves: each slider stays on its socket until the release sorts.
        for value in (600.0, 3000.0, 650.0):
            split.ace_split_freq_1 = value
            _assert(sockets() == [value, 500, 2000] and sliders() == sockets(), f"Drag to {value}: {sockets()}")
        split_freqs.sort_all_split_freqs()  # what the release (and a file load) does
        _assert(sockets() == [500, 650, 2000] and sliders() == sockets(), f"After release: {sockets()}")

        # The slider is logarithmic from 20 Hz to 20 kHz; the Hz field and the slider share a socket.
        split.ace_split_pos_1 = 0.5
        _assert(abs(split.inputs[1].default_value - 632.46) < 0.1, f"Mid slider: {split.inputs[1].default_value}")
        _assert(abs(split.ace_split_pos_1 - 0.5) < 1e-4, f"Slider reads back {split.ace_split_pos_1}")
        for position, hz in ((0.0, 20.0), (1.0, 20000.0)):
            split.ace_split_pos_1 = position
            _assert(abs(split.inputs[1].default_value - hz) < 0.01, f"Slider {position}: {split.inputs[1].default_value}")
        split.ace_split_freq_1 = 200.0
        _assert(abs(split.ace_split_pos_1 - 1.0 / 3.0) < 1e-4, f"200 Hz sits at {split.ace_split_pos_1}")
        split.ace_split_freq_1 = 650.0

        split.ace_split_freq_3 = 99999.0
        split.ace_split_freq_1 = -5.0
        _assert(sockets() == [0, 650, 20000], f"Limits: {sockets()}")
        _assert([s.name for s in list(split.inputs)[1:]] == ["Split Freq 1", "Split Freq 2", "Split Freq 3"], "Socket names")

        # A linked split frequency is left where it is.
        value_node = tree.nodes.new("ShaderNodeValue")
        tree.links.new(value_node.outputs[0], split.inputs[2])
        split.ace_split_freq_1 = 12000.0
        split.ace_split_freq_3 = 9000.0
        split_freqs.sort_split_freqs(split)
        _assert([sockets()[0], sockets()[2]] == [9000, 12000], f"Sort around a link: {sockets()}")
        tree.nodes.remove(value_node)

        ops._resolve_panel_selected_node = lambda _context: (split, {"category": "delta", "type": "frequency_splitter"})
        for direction, count in (("UP", 4), ("UP", 5), ("DOWN", 4)):
            bpy.ops.ace.update_splitter_bands(direction=direction)
            _assert(len(sockets()) == count and sockets() == sorted(sockets()), f"{direction}: {sockets()}")
        bpy.ops.ace.even_splitter_freqs()
        widths = splitter.splitter_band_widths(split, scene)
        _assert(max(widths) - min(widths) < 0.01, f"Distribute horizontally: {widths}")
    finally:
        ops._resolve_panel_selected_node = original_selected
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)


def _check_cosmetic_edits_do_not_recompile(addon_key, scene, engine):
    """Auto-compile rebuilds on content edits only: not idle, nor on selection, rename or moves."""
    import time
    handlers = importlib.import_module(f"{addon_key}.core.engine.handlers")
    modes = importlib.import_module(f"{addon_key}.nodes.modes")
    settings = scene.audiocore.engine_settings
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_Recompile")
    compiles = []
    original = engine.compiler.compile_scene_frontends
    engine.compiler.compile_scene_frontends = lambda *a, **k: (compiles.append(1), original(*a, **k))[1]

    def window(seconds=1.5):
        before = len(compiles)
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            bpy.context.view_layer.update()
            handlers.auto_compile_timer()
            time.sleep(0.05)
        return len(compiles) - before

    def switch_mode():
        modes.apply_node_mode(loud, "stream", "loudness", "rms")
        tree.update_tag()

    try:
        source = _add_ace_node(addon_key, tree, "source_mngmt", "src", "Source")
        loud = _add_ace_node(addon_key, tree, "stream", "loudness", "Loudness")
        split = _add_ace_node(addon_key, tree, "delta", "frequency_splitter", "Splitter")
        tree.links.new(source.outputs["Audio"], split.inputs[0])
        tree.links.new(split.outputs[0], loud.inputs["Audio"])
        window(2.5)
        counts = {"idle": window()}
        for name, edit in (
            ("select", lambda: setattr(loud, "select", True)),
            ("active", lambda: setattr(tree.nodes, "active", loud)),
            ("rename", lambda: (setattr(loud, "label", "Kick"), setattr(loud, "name", "Kick"))),
            ("move", lambda: setattr(loud, "location", (240.0, 0.0))),
            ("mode", switch_mode),
            ("split", lambda: setattr(split, "ace_split_freq_1", 300.0)),
        ):
            edit()
            counts[name] = window()
        expected = {"idle": 0, "select": 0, "active": 0, "rename": 0, "move": 0, "mode": 1, "split": 1}
        _assert(counts == expected, f"Recompiles: {counts}")
    finally:
        engine.compiler.compile_scene_frontends = original
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)


def _check_source_list_instances(addon_key, scene, engine):
    """Audio source list: each source folds open (closed at first) onto the objects it drives."""
    sync = importlib.import_module(f"{addon_key}.ui.panel.sync")
    operator_tree = importlib.import_module(f"{addon_key}.ui.panel.operator_tree")
    ace = scene.audiocore
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_Instances")

    def shown():
        flat = [
            operator_tree.Row(depth=it.depth, segments=[operator_tree.Segment(it.target, it.name)],
                              has_children=it.has_children, branch_key=it.branch_key, parent=it.parent_row)
            for it in ace.source_rows
        ]
        closed = {row.branch_key for row in flat if row.has_children} - sync.expanded_source_keys()
        return operator_tree.visible_rows(flat, closed)

    def active_key():
        return ace.audio_sources[ace.active_audio_source_index].stable_id

    try:
        source = _add_ace_node(addon_key, tree, "source_mngmt", "src", "Source")
        loud = _add_ace_node(addon_key, tree, "stream", "loudness", "Loudness")
        tree.links.new(source.outputs["Audio"], loud.inputs["Audio"])
        engine.compiler.compile_scene_frontends(scene)
        sync._sync_source_rows(scene)
        rows = [(it.kind, it.name, it.depth, it.parent_row) for it in ace.source_rows]
        _assert(sum(r[0] == "source" for r in rows) == len(ace.audio_sources), f"Not every source listed: {rows}")
        src_row = next(i for i, it in enumerate(ace.source_rows) if it.kind == "source"
                       and it.target == source.get("ace_source_id"))
        inst_row = next(i for i, it in enumerate(ace.source_rows) if it.kind == "instance" and it.target == obj.name)
        _assert(ace.source_rows[inst_row].parent_row == src_row, f"Instance not under its source: {rows}")
        _assert(ace.source_rows[inst_row].detail == "Geometry", f"Instance type: {ace.source_rows[inst_row].detail}")
        _assert(shown()[src_row] and not shown()[inst_row], "Instances are not folded at first")
        sync.toggle_source_branch(ace.source_rows[src_row].branch_key)
        _assert(shown()[inst_row], "Opening a source did not show its instances")

        # Adding and removing a source updates the list at once and moves the highlight.
        before = len(ace.audio_sources)
        bpy.ops.audiocore.add_audio_source()
        added = ace.source_rows[ace.active_source_row]
        _assert(added.kind == "source" and added.target == active_key(), "New source not listed and highlighted")
        for other in scene.objects:
            other.select_set(False)
        ace.active_source_row = inst_row
        _assert(bpy.context.view_layer.objects.active == obj and obj.select_get(), "Clicking an instance did not select it")
        _assert(active_key() == source.get("ace_source_id"), "Clicking an instance did not activate its source")
        _assert(ace.active_source_row == inst_row, "The highlight left the clicked instance")
        ace.active_audio_source_index = before
        _assert(ace.source_rows[ace.active_source_row].target == active_key(), "Highlight does not follow the active source")
        bpy.ops.audiocore.remove_audio_source()
        _assert(len([it for it in ace.source_rows if it.kind == "source"]) == before, "Removed source still listed")
        ace.active_source_row = src_row
        _assert(active_key() == source.get("ace_source_id"), "Clicking a source did not make it active")
        sync.toggle_source_branch(ace.source_rows[src_row].branch_key)
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)


class _CheckedLayout:
    """Stand-in UILayout: fails on a property, operator, list or icon that does not exist."""

    def __init__(self, seen):
        self.seen = seen

    def _icon(self, icon):
        if icon not in {None, "", "NONE"}:
            items = bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items
            _assert(icon in items, f"Unknown icon {icon!r}")

    def row(self, **_kw):
        return _CheckedLayout(self.seen)

    column = box = row

    def split(self, **_kw):
        return _CheckedLayout(self.seen)

    def prop(self, data, name, icon="NONE", **_kw):
        _assert(name in data.bl_rna.properties, f"{type(data).__name__} has no property {name!r}")
        self._icon(icon)
        self.seen.append(("prop", name))

    def prop_enum(self, data, name, value, **_kw):
        self.prop(data, name)

    def operator(self, idname, text="", icon="NONE", **_kw):
        module, op = idname.split(".")
        _assert(hasattr(bpy.types, f"{module.upper()}_OT_{op}"), f"No operator {idname!r}")
        self._icon(icon)
        self.seen.append(("operator", idname))
        return types.SimpleNamespace()

    def label(self, text="", icon="NONE", **_kw):
        self._icon(icon)
        self.seen.append(("label", text))

    def template_list(self, list_type, _list_id, data, propname, active_data, active_propname, **_kw):
        _assert(hasattr(bpy.types, list_type), f"No list {list_type!r}")
        _assert(propname in data.bl_rna.properties and active_propname in active_data.bl_rna.properties, "List data")
        self.seen.append(("list", list_type))

    def separator(self, **_kw):
        pass

    template_icon = separator


def _check_panels_draw(addon_key, scene):
    """Every ACE panel and list row draws without naming anything that does not exist."""
    panels = importlib.import_module(f"{addon_key}.ui.panel.panels")
    sync = importlib.import_module(f"{addon_key}.ui.panel.sync")
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_Panels")
    original_tree = sync._find_active_node_tree
    try:
        source = _add_ace_node(addon_key, tree, "source_mngmt", "src", "Source")
        split = _add_ace_node(addon_key, tree, "delta", "frequency_splitter", "Splitter")
        tree.links.new(source.outputs["Audio"], split.inputs[0])
        sync._find_active_node_tree = lambda _context, **_kw: tree
        sync._sync_ui_dummy_tree(bpy.context, force=True)
        sync._sync_source_rows(scene)
        ace = scene.audiocore
        _, active = importlib.import_module(f"{addon_key}.utils")._get_active_source(bpy.context)
        active.active_ui_node_id = split["ace_id"]
        context = types.SimpleNamespace(
            scene=scene, window_manager=bpy.context.window_manager, preferences=bpy.context.preferences,
            view_layer=bpy.context.view_layer, region=types.SimpleNamespace(width=320), active_node=None,
        )
        drawn: dict[str, list] = {}
        for cls in panels.classes:
            if not issubclass(cls, bpy.types.Panel) or cls is panels.ACE_PT_rename_operator:
                continue
            seen = drawn[cls.__name__] = []
            panel = types.SimpleNamespace(layout=_CheckedLayout(seen), bl_label=cls.bl_label)
            if hasattr(cls, "draw_header"):
                cls.draw_header(panel, context)
            cls.draw(panel, context)
        seen = []
        list_self = types.SimpleNamespace(layout_type="DEFAULT")
        for list_cls, data, rows in ((panels.ACE_UL_source_list, ace, ace.source_rows),
                                     (panels.ACE_UL_dummy_tree, active, active.ui_nodes)):
            for index, item in enumerate(rows):
                list_cls.draw_item(list_self, context, _CheckedLayout(seen), data, item, 0, data, "", index)

        def props(panel_name):
            return {value for kind, value in drawn[panel_name] if kind == "prop"}

        _assert({"enable_preprocessing", "envelope_attack", "envelope_release"} <= props("ACE_PT_sub_envelope"),
                f"Envelope panel: {props('ACE_PT_sub_envelope')}")
        _assert({"bpm", "beats_per_bar"} <= props("ACE_PT_sub_tempo_meter"), "Tempo panel")
        _assert(("operator", "audiocore.reset_engine") in drawn["ACE_PT_sub_engine_settings"], "No reset button")
        _assert({"ace_split_pos_1", "ace_split_freq_1", "ace_band_fill"} <= props("ACE_PT_information_streams"),
                f"Splitter settings: {props('ACE_PT_information_streams')}")
        for name in ("ACE_PT_sub_source_info", "ACE_PT_sub_tempo_meter", "ACE_PT_sub_envelope"):
            _assert("DEFAULT_CLOSED" in getattr(panels, name).bl_options, f"{name} is not closed by default")
        order = [panels.ACE_PT_sub_source_info.bl_order, panels.ACE_PT_sub_tempo_meter.bl_order,
                 panels.ACE_PT_sub_envelope.bl_order]
        _assert(order == sorted(order), f"Source sub-panel order: {order}")
    finally:
        sync._find_active_node_tree = original_tree
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)


def _check_manual_engine_reset(addon_key, scene):
    """Manual Engine Reset replaces the engine, empties the shared caches and compiles again."""
    engine_cls = importlib.import_module(f"{addon_key}.core.engine.engine").ACEAudioEngine
    analyzer = importlib.import_module(f"{addon_key}.core.tasks.analyzer").AudioFluxAnalyzer
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_Reset")
    try:
        source = _add_ace_node(addon_key, tree, "source_mngmt", "src", "Source")
        loud = _add_ace_node(addon_key, tree, "stream", "loudness", "Loudness")
        tree.links.new(source.outputs["Audio"], loud.inputs["Audio"])
        before = engine_cls.ensure_instance(scene)
        before.compiler.compile_scene_frontends(scene)
        analyzer._GLOBAL_SPECTRUM_CACHE["stale"] = None
        _assert(bpy.ops.audiocore.reset_engine() == {'FINISHED'}, "Reset failed")
        after = engine_cls.get_instance(scene)
        _assert(after is not None and after is not before, "Reset kept the old engine")
        _assert(not before.graphs and after.graphs, "Reset did not rebuild the graphs")
        _assert("stale" not in analyzer._GLOBAL_SPECTRUM_CACHE, "Reset kept a stale cache")
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)


def _check_group_rebuild_keeps_input_values(addon_key, scene):
    """Rebuilding an outdated ACE group keeps the values typed on the nodes that use it."""
    import time
    factory = importlib.import_module(f"{addon_key}.nodes.geometry.factory").ACEDummyNodeFactory
    frontend = importlib.import_module(f"{addon_key}.nodes.frontend")
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_Rebuild")
    try:
        loud = _add_ace_node(addon_key, tree, "stream", "loudness", "Loudness")
        loud.inputs["Window (s)"].default_value = 1.5
        loud.inputs["Attribute ID"].default_value = "ACE_smoke_kept"
        group = loud.node_tree
        group.interface.new_socket("Leftover", in_out="INPUT", socket_type="NodeSocketFloat")
        bpy.context.view_layer.update()
        factory._NODES_CONTENT_HASH = ""
        factory.ensure_node_groups_exist()
        t0 = time.monotonic()
        while frontend._pending_input_restores and time.monotonic() - t0 < 3.0:
            bpy.context.view_layer.update()
            frontend._restore_group_inputs()
            time.sleep(0.05)
        names = [s.name for s in loud.inputs]
        _assert("Leftover" not in names, f"Outdated socket kept: {names}")
        _assert(abs(loud.inputs["Window (s)"].default_value - 1.5) < 1e-6, f"Window reset to {loud.inputs['Window (s)'].default_value}")
        _assert(loud.inputs["Attribute ID"].default_value == "ACE_smoke_kept", "Attribute ID reset by the rebuild")
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)


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


def _check_file_source_drives_proxy(addon_key, engine, scene, src, parallel_mode="threads_io"):
    """A file source wired to RMS, LUFS and a splitter writes values that follow the audio, and bakes."""
    scene.audiocore.engine_settings.ace_parallel_mode = parallel_mode
    tmp_dir = tempfile.mkdtemp(prefix="ace_smoke_audio_")
    wav_path = os.path.join(tmp_dir, "tone.wav")
    _write_test_wav(wav_path)
    src.source_type = "FILE"
    src.filepath = wav_path

    mesh = bpy.data.meshes.new("ACE_Smoke_Audio_Mesh")
    obj = bpy.data.objects.new("ACE_Smoke_Audio_Object", mesh)
    scene.collection.objects.link(obj)
    tree = bpy.data.node_groups.new("ACE_Smoke_Audio_GN", "GeometryNodeTree")
    tree.is_modifier = True
    obj.modifiers.new("ACE Smoke Audio", "NODES").node_group = tree

    source_node = _add_ace_node(addon_key, tree, "source_mngmt", "src", "Source")
    rms = _add_ace_node(addon_key, tree, "stream", "loudness", "RMS", mode="rms")
    lufs = _add_ace_node(addon_key, tree, "stream", "loudness", "LUFS", mode="lufs")
    split = _add_ace_node(addon_key, tree, "delta", "frequency_splitter", "Splitter")
    band_rms = _add_ace_node(addon_key, tree, "stream", "loudness", "Band RMS", mode="rms")
    audio_out = source_node.outputs["Audio"]
    tree.links.new(audio_out, rms.inputs["Audio"])
    tree.links.new(audio_out, lufs.inputs["Audio"])
    tree.links.new(audio_out, split.inputs[0])
    tree.links.new(split.outputs[0], band_rms.inputs["Audio"])

    scene.frame_start, scene.frame_end = 1, 72
    engine.compiler.compile_scene_frontends(scene)
    _assert(bool(engine.graphs), "No runtime graph compiled for the file source")
    import time
    engine_mod = importlib.import_module(f"{addon_key}.core.engine.engine")
    handlers_mod = importlib.import_module(f"{addon_key}.core.engine.handlers")
    # Let the one legitimate recompile happen first: the file's sample-rate
    # probe finishes in the background and asks for it.
    settle_compiles = []
    original_compile = engine.compiler.compile_scene_frontends
    engine.compiler.compile_scene_frontends = lambda *a, **k: (settle_compiles.append(time.monotonic()), original_compile(*a, **k))[1]
    t_settle = last_change = time.monotonic()
    while time.monotonic() - last_change < 0.5 and time.monotonic() - t_settle < 10.0:
        before = len(settle_compiles)
        scene.frame_set(1 + int((time.monotonic() - t_settle) * 24) % 24)
        handlers_mod.auto_compile_timer()
        if len(settle_compiles) != before:
            last_change = time.monotonic()
        time.sleep(1.0 / 24.0)
    engine.compiler.compile_scene_frontends = original_compile
    _assert(len(settle_compiles) <= 1, f"Scene recompiled {len(settle_compiles)} times while settling")
    cold_start_s = None
    if parallel_mode == "processes_dsp":
        # Workers spawn on first use and take seconds to import; play until one answers.
        t_warm = time.monotonic()
        frame = 1
        while int(engine.parallel.result_accepts) == 0 and time.monotonic() - t_warm < 60.0:
            scene.frame_set(1 + frame % 2)
            frame += 1
            time.sleep(1.0 / 24.0)
        cold_start_s = round(time.monotonic() - t_warm, 2)
        scene.frame_set(72)
        scene.frame_set(1)
    quiet = loud = None
    first_accept_s = None
    parallel = engine.parallel
    counts_before = (parallel.result_accepts, parallel.stale_job_drops, parallel.queue_pressure_drops)
    compiles = []
    original_compile = engine.compiler.compile_scene_frontends
    engine.compiler.compile_scene_frontends = lambda *a, **k: (compiles.append(1), original_compile(*a, **k))[1]
    t0 = time.monotonic()
    for frame in range(1, 73):
        # Real-time pace (24 fps): worker results arrive frames later.
        target = t0 + (frame - 1) / 24.0
        time.sleep(max(0.0, target - time.monotonic()))
        scene.frame_set(frame)
        handlers_mod.auto_compile_timer()  # timers do not run in background mode
        if first_accept_s is None and int(getattr(getattr(engine, "parallel", None), "result_accepts", 0) or 0) > 0:
            first_accept_s = time.monotonic() - t0
        if frame == 30:
            quiet = _proxy_values(engine)
        if frame == 66:  # 2.75 s, in the loud half
            loud = _proxy_values(engine)
    engine.compiler.compile_scene_frontends = original_compile
    _assert(not compiles, f"Playback without edits recompiled the scene {len(compiles)} time(s)")
    if parallel_mode == "processes_dsp":
        accepted, stale, pressure = (
            now - before
            for now, before in zip(
                (parallel.result_accepts, parallel.stale_job_drops, parallel.queue_pressure_drops),
                counts_before,
            )
        )
        print(f"[ACE smoke] processes_dsp: cold start {cold_start_s} s; first worker result after {first_accept_s} s; "
              f"during playback: accepted={accepted}, stale_job_drops={stale}, queue_pressure_drops={pressure}")
    if not loud:
        details = []
        for uid, graph in engine.graphs.items():
            proxy = graph.get_proxy_object()
            mesh = getattr(proxy, "data", None)
            details.append(
                f"graph={uid} proxy={getattr(proxy, 'name', None)} keys={list(graph.output_attr_keys)} "
                f"tasks={sorted(graph.tasks)} mesh_attrs={[a.name for a in mesh.attributes] if mesh else None}"
            )
        swallowed = importlib.import_module(f"{addon_key}.utils").swallowed_errors()
        _assert(False, "No proxy attribute values were written: " + " | ".join(details) + f" swallowed={swallowed}")
    rose = [k for k in loud if quiet.get(k) is not None and loud[k] > quiet[k] + 1e-4]
    _assert(rose, f"No output followed the level change: quiet={quiet} loud={loud}")
    if parallel_mode == "processes_dsp":
        # Values can also come from the main-thread fallback; require real worker results.
        parallel = engine.parallel
        state = {
            name: getattr(parallel, name)
            for name in (
                "result_accepts", "job_exceptions", "disabled_sources", "failures",
                "stale_result_drops", "stale_job_drops", "queue_pressure_drops",
            )
        }
        state.update(pool_failed=parallel.pool.failed, restarts=parallel.pool.restart_count, executor=parallel.pool.executor)
        state["offload_keys"] = {uid: list(getattr(g, "parallel_offload_attr_keys", []) or []) for uid, g in engine.graphs.items()}
        _assert(int(engine.parallel.result_accepts) > 0, f"No worker-process DSP result was accepted: {state}")

    _assert(engine.bake.bake_source_to_cache(src, tree=tree, scene=scene), "Bake did not start")
    import time
    deadline = time.monotonic() + 60.0
    while str(src.get("ace_bake_status", "")) == "baking" and time.monotonic() < deadline:
        engine.bake.flush_bake_results()
        time.sleep(0.05)
    engine.bake.flush_bake_results()
    _assert(str(src.get("ace_bake_status", "")) == "done", f"Bake ended as {src.get('ace_bake_status')!r}: {src.get('ace_bake_error')!r}")

    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.node_groups.remove(tree)


def main():
    print("[ACE smoke] Blender-side smoke script loaded.")

    # Basic sanity checks users can expand:
    _assert(bpy.app.version >= (5, 0, 0), "Blender 5.0+ is required")
    addon_keys = [k for k in bpy.context.preferences.addons.keys() if "AudioCore" in k]
    _assert(bool(addon_keys), "AudioCore add-on is not enabled in current Blender session")
    addon_key = addon_keys[0]
    engine_mod = importlib.import_module(f"{addon_key}.core.engine.engine")
    handlers_mod = importlib.import_module(f"{addon_key}.core.engine.handlers")
    _assert(
        handlers_mod.frame_change_handler in bpy.app.handlers.frame_change_pre,
        "frame_change_handler is not registered",
    )
    _assert(
        handlers_mod.depsgraph_update_handler in bpy.app.handlers.depsgraph_update_post,
        "depsgraph_update_handler is not registered",
    )
    _assert(
        handlers_mod._load_post_reset_engine in bpy.app.handlers.load_post,
        "load_post reset handler is not registered",
    )
    scene = bpy.context.scene
    ace = getattr(scene, "audiocore", None)
    _assert(ace is not None, "Scene.audiocore pointer is missing")
    eng_settings = getattr(ace, "engine_settings", None)
    _assert(eng_settings is not None, "Engine settings pointer is missing")
    _assert(str(getattr(eng_settings, "ace_scheduler_mode", "")) == "demand", "Default scheduler mode must be demand-driven")
    _assert(str(getattr(eng_settings, "ace_timeslice_mode", "")) == "fixed", "Default timeslice mode must be fixed")
    _assert(str(getattr(eng_settings, "ace_parallel_mode", "")) == "processes_dsp", "Default parallel mode must be processes_dsp")
    _assert(bool(getattr(eng_settings, "ace_perf_enabled", False)) is True, "Performance metrics must be enabled by default")

    engine = engine_mod.ACEAudioEngine.ensure_instance(scene)
    engine.frame.execute_next_frame(scene, force=True)

    before_count = len(ace.audio_sources)
    add_result = bpy.ops.audiocore.add_audio_source()
    _assert('FINISHED' in add_result, "audiocore.add_audio_source failed")
    _assert(len(ace.audio_sources) == before_count + 1, "Audio source count did not increase")

    active_idx = int(getattr(ace, "active_audio_source_index", -1))
    _assert(0 <= active_idx < len(ace.audio_sources), "Active source index invalid after add")
    src = ace.audio_sources[active_idx]
    _assert(bool(getattr(src, "stable_id", "")), "Added source has no stable_id")
    proxy_name = getattr(src, "proxy_object_name", "")
    _assert(bool(proxy_name), "Added source has no proxy object name")
    _assert(bpy.data.objects.get(proxy_name) is not None, "Proxy object was not created")

    sonar = bpy.data.objects.get("ACE_Sonar")
    _assert(sonar is not None, "ACE_Sonar object was not created")
    nodes_mod = next((m for m in sonar.modifiers if m.type == 'NODES'), None)
    _assert(nodes_mod is not None, "ACE_Sonar has no Geometry Nodes modifier")
    _assert(nodes_mod.node_group is not None, "ACE_Sonar modifier has no node tree")
    _assert(bool(getattr(nodes_mod.node_group, "is_modifier", False)), "ACE_Sonar node tree is not marked as modifier")
    source_nodes = [
        n for n in nodes_mod.node_group.nodes
        if getattr(n, "type", "") == 'GROUP' and getattr(n, "get", lambda *_: "")("ace_source_id", "") == src.stable_id
    ]
    _assert(bool(source_nodes), "No managed source placeholder found in ACE_Sonar tree")
    sonar_mat = bpy.data.materials.get("ACE_Sonar_Material")
    _assert(sonar_mat is not None, "ACE_Sonar shader material was not created")
    _assert(sonar.active_material == sonar_mat, "ACE_Sonar shader material is not assigned")
    sonar_shader_tree = getattr(sonar_mat, "node_tree", None)
    _assert(sonar_shader_tree is not None, "ACE_Sonar shader material has no node tree")
    shader_source_nodes = [
        n for n in sonar_shader_tree.nodes
        if getattr(n, "type", "") == 'GROUP' and getattr(n, "get", lambda *_: "")("ace_source_id", "") == src.stable_id
    ]
    _assert(bool(shader_source_nodes), "No managed source placeholder found in ACE_Sonar shader tree")

    frontend_obj = None
    frontend_mesh = None
    frontend_gn_tree = None
    frontend_mat = None
    frontend_shader_tree = None
    try:
        frontend_mesh = bpy.data.meshes.new("ACE_Smoke_Frontend_Mesh")
        frontend_obj = bpy.data.objects.new("ACE_Smoke_Frontend_Object", frontend_mesh)
        scene.collection.objects.link(frontend_obj)

        frontend_gn_tree = bpy.data.node_groups.new("ACE_Smoke_GN", "GeometryNodeTree")
        frontend_gn_tree["ace_frontend_uid"] = "ACE_SMOKE_FRONTEND_GN"
        try:
            frontend_gn_tree.is_modifier = True
        except Exception:
            pass
        frontend_mod = frontend_obj.modifiers.new("ACE Smoke GN", "NODES")
        frontend_mod.node_group = frontend_gn_tree

        frontend_mat = bpy.data.materials.new("ACE_Smoke_Material")
        frontend_shader_tree = frontend_mat.node_tree
        _assert(frontend_shader_tree is not None, "Material node tree was not created")
        frontend_shader_tree["ace_frontend_uid"] = "ACE_SMOKE_FRONTEND_SHADER"
        frontend_obj.data.materials.append(frontend_mat)

        _check_object_and_material_edits_queue_recompile(
            engine, scene, frontend_obj, frontend_mod, frontend_gn_tree, frontend_shader_tree
        )

        discovered = engine.frontends.discover_scene_frontends(scene)
        kinds = sorted(rec.frontend_ref.kind for rec in discovered)
        _assert("GEOMETRY_NODES" in kinds, "Scene frontend discovery missed Geometry Nodes frontend")
        _assert("SHADER_NODES" in kinds, "Scene frontend discovery missed Shader frontend")

        engine.compiler.compile_scene_frontends(scene)
        _assert(
            "ACE_SMOKE_FRONTEND_GN" in getattr(getattr(engine, "compiler", None), "_compiled_frontend_snapshots", {}),
            "Scene compile missed the GN frontend",
        )
        _assert(
            "ACE_SMOKE_FRONTEND_SHADER" in getattr(getattr(engine, "compiler", None), "_compiled_frontend_snapshots", {}),
            "Scene compile missed the shader frontend",
        )
    finally:
        if frontend_obj is not None:
            bpy.data.objects.remove(frontend_obj, do_unlink=True)
        if frontend_mat is not None:
            bpy.data.materials.remove(frontend_mat, do_unlink=True)
        if frontend_gn_tree is not None and frontend_gn_tree.users == 0:
            bpy.data.node_groups.remove(frontend_gn_tree)

    _check_file_source_drives_proxy(addon_key, engine, scene, src, parallel_mode="threads_io")
    _check_file_source_drives_proxy(addon_key, engine, scene, src, parallel_mode="processes_dsp")
    _check_old_stream_nodes_migrate(addon_key, scene)
    _check_operator_list_follows_tree(addon_key, scene)
    _check_splitter_sliders(addon_key, scene)
    _check_computed_link_values(addon_key, scene)
    _check_driven_splits(addon_key, scene, engine)
    _check_selection_follows_sources(addon_key, scene)
    _check_new_source_takes_input_name(addon_key, scene)
    _check_viewer_previews_audio(addon_key, scene, engine)
    _check_cosmetic_edits_do_not_recompile(addon_key, scene, engine)
    _check_source_list_instances(addon_key, scene, engine)
    _check_group_rebuild_keeps_input_values(addon_key, scene)
    _check_panels_draw(addon_key, scene)
    _check_manual_engine_reset(addon_key, scene)

    remove_result = bpy.ops.audiocore.remove_audio_source()
    _assert('FINISHED' in remove_result, "audiocore.remove_audio_source failed")
    _assert(len(ace.audio_sources) == before_count, "Audio source count did not return to initial value")

    _check_loaded_files_get_current_node_groups(addon_key)

    swallowed = importlib.import_module(f"{addon_key}.utils").swallowed_errors()
    for key, (count, last_error) in sorted(swallowed.items()):
        print(f"[ACE smoke] swallowed x{count} {key}: {last_error}")
    _assert(not swallowed, f"{len(swallowed)} code path(s) absorbed errors during a normal session")

    print("[ACE smoke] Version, lifecycle, and scene frontend checks passed.")
    print(f"[ACE smoke] Enabled addon keys: {addon_keys}")


if __name__ == "__main__":
    main()
