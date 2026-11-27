"""
Run inside Blender (background or UI) after enabling the add-on.

Headless, from the repo root (enables the add-on from this checkout):
  python tests/run_in_blender.py
"""

import types
import bpy  # type: ignore
import importlib
import os
import shutil
import sys
import tempfile
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blender_helpers import _add_ace_node, _new_gn_tree, _proxy_values, _write_test_wav, find_addon_key  # noqa: E402
import blender_helpers  # noqa: E402


def _nid(node):
    return blender_helpers.node_id(find_addon_key(), node)


def _sid(node):
    return blender_helpers.node_source(find_addon_key(), node)


def _signal(engine, node):
    """The signal (task id) computing the node's use: uses that compute the same share one."""
    return engine.signal_id(_nid(node))


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _check_object_and_material_edits_queue_recompile(engine, scene, obj, modifier, gn_tree, shader_tree):
    """A material edit marks its tree (the original, never an evaluated copy); an object edit marks nothing.

    Which trees an object uses is owner membership, compared by every compile poll: a new
    modifier is a membership change, not a dirty tree.
    """
    requested = []
    original_request = engine.frontends.request_recompile
    engine.frontends.request_recompile = lambda tree, **_kw: requested.append(tree)
    try:
        bpy.context.view_layer.update()
        requested.clear()
        modifier.show_viewport = not modifier.show_viewport
        bpy.context.view_layer.update()
        _assert(not requested, f"An object edit marked trees: {[getattr(t, 'name', t) for t in requested]}")
        obj.data.materials[0].diffuse_color = (0.2, 0.4, 0.6, 1.0)
        bpy.context.view_layer.update()
    finally:
        engine.frontends.request_recompile = original_request
    pointers = {t.as_pointer() for t in requested if t is not None}
    _assert(shader_tree.as_pointer() in pointers, "Material edit did not queue its shader frontend")
    _assert(gn_tree.as_pointer() not in pointers, "A material edit marked an unrelated tree")
    _assert(all(not t.is_evaluated for t in requested if t is not None), "Recompile queued an evaluated copy")
    before = engine.frontends._frontend_membership_signature(engine.frontends.discover_scene_frontends(scene))
    extra = obj.modifiers.new("ACE Smoke GN 2", "NODES")
    extra.node_group = gn_tree
    after = engine.frontends._frontend_membership_signature(engine.frontends.discover_scene_frontends(scene))
    obj.modifiers.remove(extra)
    _assert(before != after, "A new modifier using an ACE tree did not change the owner membership")


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
        loud = _add_ace_node(addon_key, tree, "stream", "loudness", "Loudness", mode="lufs")
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
            f"  Split {low}{sep}Loudness (RMS)",
            f"  Split {high}{sep}Stereo (Width)",
        ]
        _assert(rows() == expected, f"Folded tree: {rows()}")

        # List -> editor.
        _assert("FINISHED" in bpy.ops.ace.select_operator(node_id=_nid(stereo)), "select_operator failed")
        _assert(tree.nodes.active == stereo and stereo.select and not loud.select, "List click did not select the node")
        _assert(src.active_ui_node_id == _nid(stereo), "List click did not select the operator")
        _assert(src.active_ui_node_index == 3, f"Active row: {src.active_ui_node_index}")
        _assert("FINISHED" in bpy.ops.ace.select_operator(node_id=_nid(split), row=3), "band click")
        _assert(tree.nodes.active == split and src.active_ui_node_index == 3, "A band click moved the row")
        bpy.ops.ace.select_operator(node_id=_nid(stereo))

        # Editor -> list, unfolding a folded branch.
        sync.toggle_branch(_nid(source))
        shown = operator_tree.visible_rows(sync.current_rows(), sync.collapsed_keys())
        _assert(sum(shown) == 1, f"Folding the source left {sum(shown)} rows")
        tree.nodes.active = rms
        sync._sync_editor_active(tree, src)
        _assert(src.active_ui_node_id == _nid(rms), "Editor selection did not reach the list")
        _assert(_nid(source) not in sync.collapsed_keys(), "Selecting a folded operator did not unfold it")
        _assert(src.active_ui_node_index == 2, f"Active row: {src.active_ui_node_index}")

        # Rename box (double-click): list and node title follow; a custom name survives a mode switch;
        # an empty name restores the automatic one.
        wm = bpy.context.window_manager

        def rename_rms(value):
            selection.start_rename(_nid(rms))
            wm.ace_rename_name = value

        selection.start_rename(_nid(rms))
        _assert(wm.ace_rename_name == rms.label, f"Rename box shows {wm.ace_rename_name!r}")
        rename_rms("Kick Level")
        _assert(rms.label == "Kick Level" and rms.name == "Kick Level", f"Rename: {rms.label!r} / {rms.name!r}")
        _assert(f"  Split {low}{sep}Kick Level" in rows(), f"Renamed row: {rows()}")
        split.label = "Drums"
        _assert(f"  Drums {low}{sep}Kick Level" in rows(), f"Named splitter: {rows()}")
        split.label = "Splitter (2 Bands)"
        modes.apply_node_mode(rms, "stream", "loudness", "peak")
        _assert(rms.label == "Kick Level", "A mode switch replaced the custom name")
        rename_rms("")
        _assert(rms.label == "Loudness (Peak)", f"Empty name gave {rms.label!r}")
        rename_rms("Bass")
        rows()

        # A click on the row of the selected name keeps it; another row selects its last name.
        row_idx = operator_tree.row_of_node(sync.current_rows(), _nid(rms))
        src.active_ui_node_index = row_idx
        _assert(src.active_ui_node_id == _nid(rms), "Clicking the selected name's row lost the selection")
        lufs_row = operator_tree.row_of_node(sync.current_rows(), _nid(loud))
        src.active_ui_node_index = lufs_row
        _assert(src.active_ui_node_id == _nid(loud), "Clicking another row did not select it")

        # Splitter band strip: band widths on the log axis.
        splitter_mod = importlib.import_module(f"{addon_key}.ui.panel.splitter")
        split.inputs[1].default_value = 632.46  # geometric middle of 20 Hz .. 20 kHz
        widths = splitter_mod.splitter_band_widths(split, scene)
        _assert(len(widths) == 2 and abs(sum(widths) - 1.0) < 1e-6, f"Band widths: {widths}")
        _assert(abs(widths[0] - 0.5) < 0.01, f"Band widths do not follow the log axis: {widths}")
        socket_color = tuple(split.inputs[0].draw_color(bpy.context, split))[:3]
        _assert(all(abs(a - b) < 1e-4 for a, b in zip(socket_color, selection._BUNDLE_SOCKET_COLOR)),
                f"Bundle socket colour is {socket_color}")
        selection._match_band_fill(bpy.context, socket_color)
        ui = bpy.context.preferences.themes[0].user_interface
        under = [ui.panel_back[i] * (1 - ui.wcol_box.inner[3]) + ui.wcol_box.inner[i] * ui.wcol_box.inner[3] for i in range(3)]
        seen = [0.5 * wm.ace_band_fill[i] + 0.5 * under[i] for i in range(3)]
        _assert(all(abs(a - b) < 0.01 for a, b in zip(seen, socket_color)), f"Strip looks {seen}, sockets {socket_color}")

        tree.links.remove(stereo.inputs["Audio"].links[0])
        _assert(rows()[-1] == f"  Split {low}{sep}Bass", f"After disconnecting: {rows()}")
    finally:
        sync._find_active_node_tree, selection._find_active_node_tree = originals
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)


def _check_every_splitter_lists_its_bands(addon_key, scene):
    """Every band splitter shows as "Split <band>"; a name the user gave replaces "Split"."""
    sync = importlib.import_module(f"{addon_key}.ui.panel.sync")
    nodes = importlib.import_module(f"{addon_key}.vars").ACE_NODES["delta"]["nodes"]
    original_tree = sync._find_active_node_tree
    for node_type, definition in nodes.items():
        if node_type == "frequency_splitter":
            continue  # covered with the folded tree
        obj, tree = _new_gn_tree(scene, "ACE_Smoke_Split")
        try:
            source = _add_ace_node(addon_key, tree, "source_mngmt", "src", "Source")
            split = _add_ace_node(addon_key, tree, "delta", node_type, definition["label"])
            tree.links.new(source.outputs["Audio"], split.inputs[0])
            for output in list(split.outputs)[:2]:
                loud = _add_ace_node(addon_key, tree, "stream", "loudness", "Loudness")
                tree.links.new(output, loud.inputs["Audio"])
            sync._find_active_node_tree = lambda _context, **_kw: tree
            first, second = split.outputs[0].name, split.outputs[1].name
            for prefix, title in (("Split", definition["label"]), ("Drums", "Drums")):
                split.label = title
                sync._sync_ui_dummy_tree(bpy.context, force=True)
                heads = [row.segments[0].label for row in sync.current_rows()[1:]]
                _assert(heads == [f"{prefix} {first}", f"{prefix} {second}"], f"{node_type} rows: {heads}")
        finally:
            sync._find_active_node_tree = original_tree
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
        _assert(previews() == [(_signal(engine, split), 0)], f"Viewer on the low band: {previews()}")

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

        mix = tree.nodes.new("ShaderNodeMix")
        mix.data_type = "FLOAT"
        mix_a, mix_b, factor = (next(i for i in mix.inputs if i.identifier == n) for n in ("A_Float", "B_Float", "Factor_Float"))
        tree.links.new(value.outputs[0], factor)
        tree.links.new(next(o for o in mix.outputs if o.identifier == "Result_Float"), target)
        mix_a.default_value, mix_b.default_value = 100.0, 900.0
        for clamp in (True, False):
            mix.clamp_factor = clamp
            for a in (0.25, 1.5):
                value.outputs[0].default_value = a
                check(f"mix clamp={clamp} {a}")

        switch = tree.nodes.new("GeometryNodeSwitch")
        switch.input_type = "FLOAT"
        flag = tree.nodes.new("FunctionNodeInputBool")
        tree.links.new(flag.outputs[0], switch.inputs["Switch"])
        switch.inputs["False"].default_value, switch.inputs["True"].default_value = 11.0, 22.0
        tree.links.new(switch.outputs[0], target)
        _assert(driven.linked_value(target, scene) is None, "A boolean input node is not followed")
        tree.links.remove(switch.inputs["Switch"].links[0])
        for state in (False, True):
            switch.inputs["Switch"].default_value = state
            check(f"switch {state}")

        curve = tree.nodes.new("ShaderNodeFloatCurve")
        curve.mapping.curves[0].points.new(0.5, 0.9)
        curve.mapping.update()
        tree.links.new(value.outputs[0], curve.inputs["Value"])
        tree.links.new(curve.outputs[0], target)
        for a in (0.1, 0.5, 0.8):
            value.outputs[0].default_value = a
            check(f"float curve {a}")

        to_int = tree.nodes.new("FunctionNodeFloatToInt")
        tree.links.new(value.outputs[0], to_int.inputs[0])
        tree.links.new(to_int.outputs[0], target)
        for mode in ("ROUND", "FLOOR", "CEILING", "TRUNCATE"):
            to_int.rounding_mode = mode
            for a in (2.5, -2.5, 3.2):
                value.outputs[0].default_value = a
                check(f"to int {mode} {a}")

        # A user's node group, with its input fed from outside; and a muted node.
        inner = bpy.data.node_groups.new("ACE_Smoke_Inner", "GeometryNodeTree")
        inner.interface.new_socket("In", in_out="INPUT", socket_type="NodeSocketFloat")
        inner.interface.new_socket("Out", in_out="OUTPUT", socket_type="NodeSocketFloat")
        inner_in, inner_out = inner.nodes.new("NodeGroupInput"), inner.nodes.new("NodeGroupOutput")
        double = inner.nodes.new("ShaderNodeMath")
        double.operation = "MULTIPLY"
        double.inputs[1].default_value = 2.0
        inner.links.new(inner_in.outputs[0], double.inputs[0])
        inner.links.new(double.outputs[0], inner_out.inputs[0])
        group = tree.nodes.new("GeometryNodeGroup")
        group.node_tree = inner
        tree.links.new(value.outputs[0], group.inputs[0])
        tree.links.new(group.outputs[0], target)
        value.outputs[0].default_value = 21.0
        check("node group")
        muted = tree.nodes.new("ShaderNodeMath")
        tree.links.new(group.outputs[0], muted.inputs[0])
        tree.links.new(muted.outputs[0], target)
        muted.mute = True
        check("muted node")
        group.node_tree = None
        bpy.data.node_groups.remove(inner)

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
                        if node_id == _signal(engine, split))

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
        _assert(any(g.param_overrides.get(_signal(engine, split), {}).get("in_1") == 1000.0 for g in engine.graphs.values()),
                "Workers are not told about the new value")
        _assert(splitter.splitter_band_widths(split, scene)[0] > narrow[0] + 0.3, "The strip did not follow")
        engine.frame.execute_next_frame(scene, force=True)

        noise = tree.nodes.new("ShaderNodeTexNoise")
        tree.links.new(noise.outputs[0], split.inputs[1])
        _assert(states()[0] == (None, "unreadable"), f"Unreadable link: {states()}")
        engine.compiler.compile_scene_frontends(scene)
        _assert("in_1" not in task_params() and task_params().get("unknown_inputs") == [1],
                f"The socket's own value overrode the link: {task_params()}")
        engine.frame.execute_next_frame(scene, force=True)
        # An audio value through maths: the noodle's value wins over the operator's raw value.
        scale = tree.nodes.new("ShaderNodeMath")
        scale.operation = "MULTIPLY"
        scale.inputs[1].default_value = 4000.0
        tree.links.new(times.outputs[0], split.inputs[1])
        tree.links.new(loud.outputs[0], scale.inputs[0])
        tree.links.new(scale.outputs[0], split.inputs[2])
        engine.compiler.compile_scene_frontends(scene)
        engine.latest_values[f"{_signal(engine, loud)}:0"] = 0.5
        _assert(states()[1] == (2000, "linked"), f"Audio-driven link: {states()}")
        split_freqs.refresh_driven_splits(engine.graphs.values(), scene)
        _assert(task_params().get("in_2") == 2000.0, f"Engine audio-driven value: {task_params()}")
        engine.latest_values.clear()
        engine.frame.execute_next_frame(scene, force=True)
        _assert(f"{_signal(engine, loud)}:0" in engine.latest_values, "The frame did not record operator values")
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
        _assert(active().active_ui_node_id == _nid(pairs[1][1]), "Clicked node not selected")
        consistent("after a click")

        # Switching sources in the list, with stale rows and selections on both sides.
        for source_index, loud in ((a_index, pairs[0][1]), (b_index, pairs[1][1]), (a_index, pairs[0][1])):
            src = ace.audio_sources[source_index]
            src.active_ui_node_id = _nid(pairs[0][1]) if source_index == b_index else _nid(loud)
            ace.active_audio_source_index = source_index
            consistent(f"after switching to {source_index}")
        _assert(operator_tree.row_of_node(sync.current_rows(), _nid(pairs[0][1])) >= 0, "Rows not rebuilt on switch")

        bpy.ops.audiocore.add_audio_source()  # no Source node reads this one
        _assert(not len(active().ui_nodes) and not active().active_ui_node_id,
                f"Unused input lists: {[r.name for r in active().ui_nodes]}")
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


def _user_tree_snapshot():
    """Everything a user tree holds that ACE could write: nodes, their properties, socket values, links."""
    def value(sock):
        v = getattr(sock, "default_value", None)
        if hasattr(v, "name"):
            return v.name
        if hasattr(v, "__len__") and not isinstance(v, str):
            return tuple(v)
        return v

    trees = [t for t in bpy.data.node_groups if not t.name.startswith(".ACE_")]
    trees += [m.node_tree for m in bpy.data.materials if m.node_tree is not None]
    snapshot = {}
    for tree in trees:
        nodes = []
        for node in tree.nodes:
            nodes.append((
                node.name, node.label, node.bl_idname,
                getattr(getattr(node, "node_tree", None), "name", ""),
                tuple(sorted((k, repr(node[k])) for k in node.keys())),
                tuple((s.identifier, repr(value(s))) for s in node.inputs),
            ))
        links = sorted((l.from_node.name, l.from_socket.identifier, l.to_node.name, l.to_socket.identifier) for l in tree.links)
        snapshot[tree.name_full + "|" + tree.bl_idname] = (tuple(sorted(nodes)), tuple(links))
    return snapshot


def _check_compile_writes_nothing(addon_key, scene, engine):
    """Nodes added by the menu have their names: compiles and playback leave every user tree as it was."""
    ace = scene.audiocore
    count, first_index = len(ace.audio_sources), ace.active_audio_source_index
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_NoWrites")
    material = bpy.data.materials.new("ACE_Smoke_NoWrites_Mat")
    obj.data.materials.append(material)
    try:
        bpy.ops.audiocore.add_audio_source()
        ace.active_audio_source_index = count
        source = _add_ace_node(addon_key, tree, "source_mngmt", "src", "Source")
        split = _add_ace_node(addon_key, tree, "delta", "frequency_splitter", "Splitter")
        loud = _add_ace_node(addon_key, tree, "stream", "loudness", "Loudness")
        acc = _add_ace_node(addon_key, tree, "utilities", "stream_accumulator", "Accumulator")
        tree.links.new(source.outputs["Audio"], split.inputs[0])
        tree.links.new(split.outputs[1], loud.inputs["Audio"])
        tree.links.new(loud.outputs[0], acc.inputs["Value"])
        shader_tree = material.node_tree
        s_source = _add_ace_node(addon_key, shader_tree, "source_mngmt", "src", "Source")
        s_loud = _add_ace_node(addon_key, shader_tree, "stream", "loudness", "Loudness")
        shader_tree.links.new(s_source.outputs["Audio"], s_loud.inputs["Audio"])
        _assert(_sid(s_source) == ace.audio_sources[count].stable_id, "The shader Source node does not read the active input")
        before = _user_tree_snapshot()
        engine.compiler.compile_scene_frontends(scene)
        for frame in (1, 2, 3, 2):
            scene.frame_set(frame)
        engine.compiler.compile_scene_frontends(scene)
        after = _user_tree_snapshot()
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        _assert(not changed, f"A compile or playback wrote into user trees: {changed}")
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)
        bpy.data.materials.remove(material)
        while len(ace.audio_sources) > count:
            ace.active_audio_source_index = len(ace.audio_sources) - 1
            bpy.ops.audiocore.remove_audio_source()
        ace.active_audio_source_index = first_index


def _check_sink_name_collisions_are_repaired_once(addon_key, scene, engine):
    """A name two nodes claim (Shift+D, a copied material) or a missing one is repaired once, then left alone."""
    identity = importlib.import_module(f"{addon_key}.nodes.identity")
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_Collisions")
    material = bpy.data.materials.new("ACE_Smoke_Collisions_Mat")
    obj.data.materials.append(material)
    try:
        first = _add_ace_node(addon_key, tree, "stream", "loudness", "Loudness")
        copy = _add_ace_node(addon_key, tree, "stream", "loudness", "Loudness Copy")
        unnamed = _add_ace_node(addon_key, tree, "stream", "pitch", "Pitch")
        kept = blender_helpers.sink_names(addon_key, first)[0]
        copy.inputs["Attribute ID"].default_value = kept  # what Shift+D gives
        unnamed.inputs["Attribute ID"].default_value = "ACE_undefined"  # a group added outside the menu
        s_first = _add_ace_node(addon_key, material.node_tree, "stream", "loudness", "Loudness")
        s_copy = material.node_tree.nodes.new("ShaderNodeGroup")
        s_copy.node_tree = s_first.node_tree
        s_copy.name = "Loudness Copy"
        engine.compiler.compile_scene_frontends(scene)
        names = [blender_helpers.sink_names(addon_key, n).get(0) for n in (first, copy, unnamed, s_first, s_copy)]
        _assert(names[0] == kept, f"The first claim was renamed: {names}")
        _assert(len(set(names)) == 5 and all(identity.usable_sink_name(n) for n in names), f"Names not repaired: {names}")
        _assert(s_copy.node_tree != s_first.node_tree, "The shader copy still shares its sink group")
        snapshot = _user_tree_snapshot()
        engine.compiler.compile_scene_frontends(scene)
        _assert(_user_tree_snapshot() == snapshot, "The repaired names changed again")
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)
        bpy.data.materials.remove(material)


def _check_shader_values_reach_scene(addon_key, scene, engine, src):
    """A shader Loudness writes one scene property, read by its sink group and seen by the evaluated scene."""
    import time
    handlers_mod = importlib.import_module(f"{addon_key}.core.engine.handlers")
    ace = scene.audiocore
    scene.audiocore.engine_settings.ace_parallel_mode = "threads_io"
    ace.active_audio_source_index = list(ace.audio_sources).index(src)
    mesh = bpy.data.meshes.new("ACE_Smoke_Shader_Mesh")
    obj = bpy.data.objects.new("ACE_Smoke_Shader_Object", mesh)
    scene.collection.objects.link(obj)
    material = bpy.data.materials.new("ACE_Smoke_Shader_Mat")
    obj.data.materials.append(material)
    tree = material.node_tree
    try:
        source = _add_ace_node(addon_key, tree, "source_mngmt", "src", "Source")
        loud = _add_ace_node(addon_key, tree, "stream", "loudness", "Loudness", mode="rms")
        tree.links.new(source.outputs["Audio"], loud.inputs["Audio"])
        name = blender_helpers.sink_names(addon_key, loud)[0]
        readers = [n for n in loud.node_tree.nodes if n.bl_idname == "ShaderNodeAttribute"]
        _assert([(n.attribute_type, n.attribute_name) for n in readers] == [("VIEW_LAYER", name)],
                f"Sink group readers: {[(n.attribute_type, n.attribute_name) for n in readers]}")
        engine.compiler.compile_scene_frontends(scene)
        values = {}
        t0 = time.monotonic()
        for frame in range(1, 73):
            time.sleep(max(0.0, t0 + (frame - 1) / 24.0 - time.monotonic()))
            scene.frame_set(frame)
            handlers_mod.auto_compile_timer()
            if frame in (30, 66):
                values[frame] = float(scene.get(name, -1.0))
        _assert(values[66] > values[30] + 1e-4 >= 0.0, f"Scene property did not follow the audio: {values}")
        evaluated = bpy.context.evaluated_depsgraph_get().scene_eval.get(name)
        _assert(evaluated is not None and abs(float(evaluated) - float(scene[name])) < 1e-6,
                f"The evaluated scene does not see the value: {evaluated} vs {scene.get(name)}")
        bpm_name, _bpb_name = importlib.import_module(f"{addon_key}.nodes.identity").source_tempo_props(src.stable_id)
        bpm_readers = [n.attribute_name for n in source.node_tree.nodes if n.bl_idname == "ShaderNodeAttribute"]
        _assert(bpm_name in bpm_readers, f"The shader Source node does not read its BPM: {bpm_readers}")
        engine.probes.flush_source_tempo_detection_updates(list(ace.audio_sources))
        _assert(abs(float(scene.get(bpm_name, -1.0)) - float(src.bpm)) < 1e-6, "The source's BPM is not on the scene")
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.materials.remove(material)


def _check_linked_tree_compiles(addon_key, scene, engine):
    """A tree linked from another file is read; its names are claimed first and never written."""
    import tempfile
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_Library")
    path = os.path.join(tempfile.mkdtemp(prefix="ace_smoke_lib_"), "library.blend")
    try:
        source = _add_ace_node(addon_key, tree, "source_mngmt", "src", "Source")
        loud = _add_ace_node(addon_key, tree, "stream", "loudness", "Loudness")
        tree.links.new(source.outputs["Audio"], loud.inputs["Audio"])
        tree.use_fake_user = True
        bpy.data.libraries.write(path, {tree}, fake_user=True)
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)
    with bpy.data.libraries.load(path, link=True) as (data_from, data_to):
        data_to.node_groups = ["ACE_Smoke_Library"]
    linked = data_to.node_groups[0]
    mesh = bpy.data.meshes.new("ACE_Smoke_Linked_Mesh")
    user = bpy.data.objects.new("ACE_Smoke_Linked_Object", mesh)
    scene.collection.objects.link(user)
    try:
        user.modifiers.new("Linked", "NODES").node_group = linked
        _assert(linked.library is not None and not linked.is_editable, "The tree is not linked")
        engine.compiler.compile_scene_frontends(scene)
        _assert(engine.frontends._tree_frontend_uid(linked) in engine.compiler._compiled_frontend_snapshots,
                "The linked tree was not compiled")
    finally:
        bpy.data.objects.remove(user, do_unlink=True)
        library = linked.library
        bpy.data.libraries.remove(library)


def _user_group(name, inputs):
    group = bpy.data.node_groups.new(name, "GeometryNodeTree")
    for sock_name, sock_type in inputs:
        group.interface.new_socket(sock_name, in_out="INPUT", socket_type=sock_type)
    return group, group.nodes.new("NodeGroupInput")


def _abstract_nodes(engine, node_type, name):
    tree = engine.abstract_tree_ref
    return [n for n in tree.nodes.values() if n.type == node_type and n.name == name] if tree is not None else []


def _check_instances_panel_model(addon_key, scene, engine, obj, tree, window_group, source_group, inner_loud,
                                 sources, count):
    """The Active ACE Instances report and the sidebar's list of a group come from the compiled model."""
    instances = importlib.import_module(f"{addon_key}.ui.panel.instances")
    sync = importlib.import_module(f"{addon_key}.ui.panel.sync")
    report = {(d.tree_name, d.node_name): d for d in instances.instance_report(engine)}
    window = report.get((window_group.name, "Loudness"))
    owner = f"{obj.name} › {obj.modifiers[0].name}"
    _assert(window is not None and len(window.uses) == 2 and window.conflicts, f"Window group report: {window}")
    _assert(all(use.owners == [owner] and len(use.chain) == 1 for use in window.uses),
            f"Uses not named by owner and group node: {[(u.owners, u.chain) for u in window.uses]}")
    inner = report.get((source_group.name, "Inner Loudness"))
    _assert(inner is not None and sorted(s for u in inner.uses for s in u.sources) == sorted(s.stable_id for s in sources),
            f"Each use's source is not reported: {inner}")
    uses = instances.source_uses(engine)
    _assert(all(any(f.tree_name == tree.name for f in uses.get(s.stable_id, [])) for s in sources),
            f"The Audio Inputs list does not show the tree under both sources: {uses}")

    # A group open in the editor: its operators under the source the use gives them (the group
    # read on its own has no source: its Source node's proxy comes from outside).
    ace = scene.audiocore
    original_tree = sync._find_active_node_tree
    try:
        ace.active_audio_source_index = count + 1
        sync._find_active_node_tree = lambda _context, **_kw: source_group
        sync._sync_ui_dummy_tree(bpy.context, force=True)
        labels = [seg.label for row in sync.current_rows() for seg in row.segments]
        _assert("Inner Loudness" in labels, f"The group's operators are not listed under its source: {labels}")
        view = instances.compiled_tree_view(engine, source_group, sources[1].stable_id)
        node = view.nodes.get(_nid(inner_loud)) if view is not None else None
        _assert(node is not None and sources[1].stable_id in node.root_sources, "The view does not use the tree's own ids")
    finally:
        sync._find_active_node_tree = original_tree
        ace.active_audio_source_index = count


def _check_instances_through_groups(addon_key, scene, engine):
    """ACE nodes in user groups are read once per use, with what each use binds: values and sources."""
    ace = scene.audiocore
    count, first_index = len(ace.audio_sources), ace.active_audio_source_index
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_Groups")
    made = []
    try:
        bpy.ops.audiocore.add_audio_source()
        bpy.ops.audiocore.add_audio_source()
        first, second = ace.audio_sources[count], ace.audio_sources[count + 1]
        ace.active_audio_source_index = count
        source = _add_ace_node(addon_key, tree, "source_mngmt", "src", "Source")

        # A group taking the audio and the LUFS window from outside, used twice with two windows.
        window_group, gi = _user_group("ACE_Smoke_WindowGroup", [("Audio", "NodeSocketBundle"), ("Window", "NodeSocketFloat")])
        made.append(window_group)
        loud = _add_ace_node(addon_key, window_group, "stream", "loudness", "Loudness", mode="lufs")
        window_idx = [s.name for s in loud.inputs].index("Window (s)")
        window_group.links.new(gi.outputs["Audio"], loud.inputs["Audio"])
        window_group.links.new(gi.outputs["Window"], loud.inputs[window_idx])
        for value in (0.8, 1.6):
            use = tree.nodes.new("GeometryNodeGroup")
            use.node_tree = window_group
            use.inputs["Window"].default_value = value
            tree.links.new(source.outputs["Audio"], use.inputs["Audio"])

        # A group with its own Source node, reading the proxy each use gives it.
        source_group, gi2 = _user_group("ACE_Smoke_SourceGroup", [("Proxy", "NodeSocketObject")])
        made.append(source_group)
        inner_source = _add_ace_node(addon_key, source_group, "source_mngmt", "src", "Inner Source")
        source_group.links.new(gi2.outputs["Proxy"], inner_source.inputs["Proxy Object"])
        inner_loud = _add_ace_node(addon_key, source_group, "stream", "loudness", "Inner Loudness", mode="rms")
        source_group.links.new(inner_source.outputs["Audio"], inner_loud.inputs["Audio"])
        inner_acc = _add_ace_node(addon_key, source_group, "utilities", "stream_accumulator", "Inner Accumulator")
        source_group.links.new(inner_loud.outputs[0], inner_acc.inputs["Value"])
        proxy_mod = importlib.import_module(f"{addon_key}.core.proxy.manager").ACEProxyManager
        for src in (first, second):
            use = tree.nodes.new("GeometryNodeGroup")
            use.node_tree = source_group
            use.inputs["Proxy"].default_value = proxy_mod.ensure_proxy_for_source(src, scene=scene)

        engine.compiler.compile_scene_frontends(scene)
        windows = sorted(round(float(n.parameters.get(f"in_{window_idx}", -1.0)), 3)
                         for n in _abstract_nodes(engine, "loudness", "Loudness"))
        _assert(windows == [0.8, 1.6], f"The group's two uses did not get their windows: {windows}")
        # Both uses write the group node's one sink name with different values: listed, one kept.
        window_sink = blender_helpers.sink_names(addon_key, loud)[0]
        clash = [c for c in engine.sink_conflicts if c["sink_name"] == window_sink]
        _assert(len(clash) == 1 and len(clash[0]["dropped"]) == 1, f"The sink conflict is not listed: {engine.sink_conflicts}")
        # A value node reached from two sources writes one name on ACE_Values: listed too.
        acc_sink = blender_helpers.sink_names(addon_key, inner_acc)[0]
        clash = [c for c in engine.sink_conflicts if c["sink_name"] == acc_sink]
        _assert(len(clash) == 1 and clash[0]["target"] == "ACE_Values",
                f"The value node conflict is not listed: {engine.sink_conflicts}")
        roots = sorted(sorted(n.root_sources) for n in _abstract_nodes(engine, "loudness", "Inner Loudness"))
        _assert(roots == sorted([[first.stable_id], [second.stable_id]]), f"Each use did not read its source: {roots}")
        for src in (first, second):
            graph = next((g for g in engine.graphs.values() if g.source_uid == src.stable_id), None)
            _assert(graph is not None and any(n.name == "Inner Loudness" for n in
                    (engine.abstract_tree_ref.nodes.get(t) for t in graph.tasks) if n is not None),
                    f"No graph computes the group's Loudness for {src.name}")

        _check_instances_panel_model(addon_key, scene, engine, obj, tree, window_group, source_group,
                                     inner_loud, (first, second), count)

        # An edit inside a group is an edit of its uses.
        before = engine.frontends._hierarchy_trees(engine.frontends.discover_scene_frontends(scene))
        _assert(window_group.as_pointer() in before and source_group.as_pointer() in before,
                "The groups are not part of the compiled trees")

        # Definitions are cached: an edit inside one group parses that group again, nothing else.
        definitions = importlib.import_module(f"{addon_key}.nodes.definitions")
        kept = {name: definitions.definition(t) for name, t in (("tree", tree), ("source", source_group), ("window", window_group))}
        importlib.import_module(f"{addon_key}.nodes.modes").apply_node_mode(loud, "stream", "loudness", "peak")
        engine.compiler.compile_scene_frontends(scene)
        now = {name: definitions.definition(t) for name, t in (("tree", tree), ("source", source_group), ("window", window_group))}
        _assert(now["tree"] is kept["tree"] and now["source"] is kept["source"], "Unchanged trees were parsed again")
        _assert(now["window"] is not kept["window"], "The edited group was not parsed again")
        modes = sorted(n.parameters.get("mode") for n in _abstract_nodes(engine, "loudness", "Loudness"))
        _assert(modes == ["peak", "peak"], f"The group's edit did not reach both uses: {modes}")
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)
        for group in made:
            bpy.data.node_groups.remove(group)
        while len(ace.audio_sources) > count:
            ace.active_audio_source_index = len(ace.audio_sources) - 1
            bpy.ops.audiocore.remove_audio_source()
        ace.active_audio_source_index = first_index


def _check_world_and_instanced_trees_are_found(addon_key, scene, engine):
    """The World's shader and the tree of an object instanced from another scene's collection are discovered."""
    import warnings

    own_world = scene.world
    world = bpy.data.worlds.new("ACE_Smoke_World")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        world.use_nodes = True
    other = bpy.data.scenes.new("ACE_Smoke_Other_Scene")
    collection = bpy.data.collections.new("ACE_Smoke_Instanced")
    other.collection.children.link(collection)
    mesh = bpy.data.meshes.new("ACE_Smoke_Instanced_Mesh")
    inner = bpy.data.objects.new("ACE_Smoke_Instanced_Object", mesh)
    collection.objects.link(inner)
    inner_tree = bpy.data.node_groups.new("ACE_Smoke_Instanced_GN", "GeometryNodeTree")
    inner_tree.is_modifier = True
    blender_helpers.add_geometry_passthrough(inner_tree)
    inner.modifiers.new("ACE", "NODES").node_group = inner_tree
    empty = bpy.data.objects.new("ACE_Smoke_Instancer", None)
    empty.instance_type = "COLLECTION"
    empty.instance_collection = collection
    scene.collection.objects.link(empty)
    try:
        scene.world = world
        _add_ace_node(addon_key, world.node_tree, "stream", "loudness", "World Loudness")
        _add_ace_node(addon_key, inner_tree, "stream", "loudness", "Instanced Loudness")
        bpy.context.view_layer.update()
        found = {rec.frontend_ref.tree_name: rec.frontend_ref.owner_keys
                 for rec in engine.frontends.discover_scene_frontends(scene)}
        _assert(any(f"world:{world.name}" in owners for owners in found.values()), f"The World was not found: {found}")
        _assert(inner_tree.name in found, f"The instanced object's tree was not found: {found}")
        engine.compiler.compile_scene_frontends(scene)
    finally:
        scene.world = own_world
        bpy.data.objects.remove(empty, do_unlink=True)
        bpy.data.objects.remove(inner, do_unlink=True)
        bpy.data.node_groups.remove(inner_tree)
        bpy.data.collections.remove(collection)
        bpy.data.scenes.remove(other)
        bpy.data.worlds.remove(world)


def _check_signals_are_shared_and_kept(addon_key, scene, engine):
    """The same RMS in two trees is one task writing both sinks; a recompile keeps the tasks whose signals remain."""
    ace = scene.audiocore
    count, first_index = len(ace.audio_sources), ace.active_audio_source_index
    settings = ace.engine_settings
    parallel_mode = settings.ace_parallel_mode
    settings.ace_parallel_mode = "threads_io"  # values computed on this thread, frame by frame
    tmp_dir = tempfile.mkdtemp(prefix="ace_smoke_signals_")
    wav_path = os.path.join(tmp_dir, "tone.wav")
    _write_test_wav(wav_path)
    obj_a, tree_a = _new_gn_tree(scene, "ACE_Smoke_Signals_A")
    obj_b, tree_b = _new_gn_tree(scene, "ACE_Smoke_Signals_B")
    try:
        bpy.ops.audiocore.add_audio_source()
        ace.active_audio_source_index = count
        src = ace.audio_sources[count]
        src.source_type = "FILE"
        src.filepath = wav_path
        sources, uses = [], []
        for tree in (tree_a, tree_b):
            source = _add_ace_node(addon_key, tree, "source_mngmt", "src", "Source")
            rms = _add_ace_node(addon_key, tree, "stream", "loudness", "RMS", mode="rms")
            tree.links.new(source.outputs["Audio"], rms.inputs["Audio"])
            sources.append(source)
            uses.append(rms)
        acc = _add_ace_node(addon_key, tree_a, "utilities", "stream_accumulator", "Accumulator")
        tree_a.links.new(uses[0].outputs[0], acc.inputs["Value"])
        # A proxy made by an earlier version carries no source id: the first compile still reads it.
        del bpy.data.objects[src.proxy_object_name]["ace_source_id"]

        engine.compiler.compile_scene_frontends(scene)
        signals = {_signal(engine, rms) for rms in uses}
        _assert(len(signals) == 1 and "" not in signals, f"The two RMS nodes are not one signal: {signals}")
        rms_signal = signals.pop()
        graph = engine.graphs[src.stable_id]
        loudness_tasks = [t for t in graph.tasks if t.startswith("S_stream_loudness_")]
        _assert(loudness_tasks == [rms_signal], f"Loudness tasks: {loudness_tasks}")
        _assert(len([t for t in graph.tasks if t.startswith("S_source_mngmt_src_")]) == 1, "The source is read twice")
        names = [blender_helpers.sink_names(addon_key, rms)[0] for rms in uses]
        for frame in (1, 2, 3, 4, 5):
            scene.frame_set(frame)
        values = _proxy_values(engine)
        _assert(names[0] in values and names[1] in values and values[names[0]] == values[names[1]] > 0.0,
                f"Both sinks do not get the shared value: {names} {values}")

        # The first compile ran before the file was probed: its sample rate makes new signals once.
        engine.compiler.compile_scene_frontends(scene)
        for frame in (6, 7, 8):
            scene.frame_set(frame)
        graph = engine.graphs[src.stable_id]
        acc_signal, rms_signal = _signal(engine, acc), _signal(engine, uses[0])
        acc_task, rms_task = graph.tasks[acc_signal], graph.tasks[rms_signal]
        total = float(acc_task._accumulator.sum())
        _assert(total > 0.0, f"The accumulator did not run: {total}")
        # A value node has no source to carry a proxy: its values go to ACE_Values, which its group reads.
        _assert("Proxy Object" not in acc.inputs, "The accumulator still has a Proxy Object socket")
        values_obj = bpy.data.objects.get("ACE_Values")
        reader = acc.node_tree.nodes.get("ACE Values")
        _assert(values_obj is not None and reader is not None and reader.inputs["Object"].default_value == values_obj,
                "The accumulator's group does not read ACE_Values")
        acc_name = blender_helpers.sink_names(addon_key, acc)[0]
        acc_attr = values_obj.data.attributes.get(acc_name)
        _assert(acc_attr is not None and abs(float(acc_attr.data[0].value) - total) < 1e-3 * max(1.0, total),
                f"ACE_Values does not hold the accumulated value: {acc_attr and acc_attr.data[0].value} vs {total}")
        _assert(acc_name not in graph.get_proxy_object().data.attributes, "The accumulator wrote the source's proxy")
        digest = engine.parallel.ir_digests.get(graph.parallel_graph_uid)
        signature = graph.signal_signature

        # A renamed sink changes no signal: same tasks, same state, same worker graph.
        uses[1].inputs["Attribute ID"].default_value = "ACE_Smoke_Renamed"
        engine.compiler.compile_scene_frontends(scene)
        graph = engine.graphs[src.stable_id]
        _assert(graph.tasks.get(acc_signal) is acc_task and graph.tasks.get(rms_signal) is rms_task,
                f"A sink rename replaced the tasks: signals {(acc_signal, rms_signal)} -> "
                f"{(_signal(engine, acc), _signal(engine, uses[0]))}, block sizes "
                f"{(acc_task.block_size, rms_task.block_size)} -> {graph.block_size}")
        _assert(float(acc_task._accumulator.sum()) == total, "The accumulator lost its sum")
        _assert(engine.parallel.ir_digests.get(graph.parallel_graph_uid) == digest and graph.signal_signature == signature,
                f"A sink rename changed the worker graph or the bake key: "
                f"{digest} -> {engine.parallel.ir_digests.get(graph.parallel_graph_uid)}, "
                f"{signature} -> {graph.signal_signature}")
        attrs = {a.name for a in graph.get_proxy_object().data.attributes}
        _assert("ACE_Smoke_Renamed" in attrs and names[1] not in attrs and names[0] in attrs,
                f"The proxy attributes did not follow the rename: {sorted(attrs)}")

        # A new signal keeps the others running.
        peak = _add_ace_node(addon_key, tree_b, "stream", "loudness", "Peak", mode="peak")
        tree_b.links.new(sources[1].outputs["Audio"], peak.inputs["Audio"])
        engine.compiler.compile_scene_frontends(scene)
        graph = engine.graphs[src.stable_id]
        _assert(graph.tasks.get(acc_signal) is acc_task and graph.tasks.get(rms_signal) is rms_task,
                "A new node elsewhere replaced the tasks")
        _assert(_signal(engine, peak) in graph.tasks and graph.signal_signature != signature, "The new node is not compiled")
        _assert(float(acc_task._accumulator.sum()) == total, "The accumulator lost its sum")
    finally:
        settings.ace_parallel_mode = parallel_mode
        for obj, tree in ((obj_a, tree_a), (obj_b, tree_b)):
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.node_groups.remove(tree)
        while len(ace.audio_sources) > count:
            ace.active_audio_source_index = len(ace.audio_sources) - 1
            bpy.ops.audiocore.remove_audio_source()
        ace.active_audio_source_index = first_index
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _check_source_nodes(addon_key, scene, engine):
    """Source nodes are the user's: none is added or removed by a compile; each picks its source."""
    ops = importlib.import_module(f"{addon_key}.ui.n_panel_operators")
    selection = importlib.import_module(f"{addon_key}.ui.panel.selection")
    ace = scene.audiocore
    count, first_index = len(ace.audio_sources), ace.active_audio_source_index
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_SourceNodes")
    original_selected = ops._resolve_panel_selected_node
    try:
        bpy.ops.audiocore.add_audio_source()
        bpy.ops.audiocore.add_audio_source()
        first, second = ace.audio_sources[count], ace.audio_sources[count + 1]
        ace.active_audio_source_index = count
        _assert(not first.proxy_object_name and not second.proxy_object_name, "An unused input has a proxy")

        node = _add_ace_node(addon_key, tree, "source_mngmt", "src", "Source")
        loud = _add_ace_node(addon_key, tree, "stream", "loudness", "Loudness")
        tree.links.new(node.outputs["Audio"], loud.inputs["Audio"])
        _assert(_sid(node) == first.stable_id and node.label == first.name,
                f"A new Source node does not read the active input: {node.label!r}")
        node_count = len(tree.nodes)
        engine.compiler.compile_scene_frontends(scene)
        _assert(len(tree.nodes) == node_count, f"A compile added or removed nodes: {[n.name for n in tree.nodes]}")
        _assert(bpy.data.objects.get(first.proxy_object_name) is not None, "The read input has no proxy")
        _assert(not second.proxy_object_name, "The unread input got a proxy")

        # Two Source nodes may read the same input; their identities stay put across compiles.
        twin = _add_ace_node(addon_key, tree, "source_mngmt", "src", "Source")
        engine.compiler.compile_scene_frontends(scene)
        ids = (_nid(node), _nid(twin))
        engine.compiler.compile_scene_frontends(scene)
        _assert((_nid(node), _nid(twin)) == ids and ids[0] != ids[1], f"Source node ids moved: {ids}")
        tree.nodes.remove(twin)

        parsed = {"category": "source_mngmt", "type": "src"}
        _assert(selection._parsed_node_has_operator_settings(parsed), "No source menu with several inputs")
        ops._resolve_panel_selected_node = lambda _context: (node, parsed)
        _assert(bpy.ops.ace.set_node_source(source=second.stable_id) == {'FINISHED'}, "Choosing a source failed")
        _assert(_sid(node) == second.stable_id and node.label == second.name, "The node did not switch")
        _assert(ace.active_audio_source_index == count + 1, "The list did not follow the node")
        engine.compiler.compile_scene_frontends(scene)
        _assert(any(_signal(engine, loud) in graph.tasks and graph.source_uid == second.stable_id
                    for graph in engine.graphs.values()), "The branch is not computed for the chosen input")
    finally:
        ops._resolve_panel_selected_node = original_selected
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)
        while len(ace.audio_sources) > count:
            ace.active_audio_source_index = len(ace.audio_sources) - 1
            bpy.ops.audiocore.remove_audio_source()
        ace.active_audio_source_index = first_index


def _check_source_node_outputs_tempo(addon_key, scene, engine):
    """The GN Source node's BPM and Beats per Bar are its source's, as Blender evaluates them."""
    ace = scene.audiocore
    count, first_index = len(ace.audio_sources), ace.active_audio_source_index
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_SourceTempo")
    try:
        obj.data.from_pydata([(0.0, 0.0, 0.0)], [], [])
        bpy.ops.audiocore.add_audio_source()
        source = ace.audio_sources[count]
        ace.active_audio_source_index = count
        group_in, group_out = tree.nodes["Group Input"], tree.nodes["Group Output"]
        node = _add_ace_node(addon_key, tree, "source_mngmt", "src", "Source")
        loud = _add_ace_node(addon_key, tree, "stream", "loudness", "Loudness")
        tree.links.new(node.outputs["Audio"], loud.inputs["Audio"])
        geometry = group_in.outputs["Geometry"]
        for output, data_type, name in (("BPM", "FLOAT", "bpm_probe"), ("Beats per Bar", "INT", "meter_probe")):
            store = tree.nodes.new("GeometryNodeStoreNamedAttribute")
            store.data_type = data_type
            store.inputs["Name"].default_value = name
            tree.links.new(geometry, store.inputs["Geometry"])
            tree.links.new(node.outputs[output], store.inputs["Value"])
            geometry = store.outputs["Geometry"]
        tree.links.new(geometry, group_out.inputs["Geometry"])
        engine.compiler.compile_scene_frontends(scene)
        _assert(bpy.data.objects.get(source.proxy_object_name) is not None, "The read input has no proxy")

        def evaluated():
            engine.probes.flush_source_tempo_detection_updates(list(ace.audio_sources))
            obj.update_tag()
            attrs = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.attributes
            return attrs["bpm_probe"].data[0].value, attrs["meter_probe"].data[0].value

        source.bpm, source.beats_per_bar = 97.5, 7
        _assert(evaluated() == (97.5, 7), f"Source node tempo: {evaluated()}")
        source.bpm, source.beats_per_bar = 140.0, 3
        _assert(evaluated() == (140.0, 3), f"Source node tempo after a change: {evaluated()}")
        # The sink layout keeps the tempo attributes: they belong to no sink.
        engine.execute_next_frame(scene, force=True)
        proxy_attrs = bpy.data.objects[source.proxy_object_name].data.attributes
        _assert(proxy_attrs.get("ACE_bpm") is not None and proxy_attrs.get("ACE_beats_per_bar") is not None,
                f"Tempo attributes pruned: {[a.name for a in proxy_attrs]}")
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)
        while len(ace.audio_sources) > count:
            ace.active_audio_source_index = len(ace.audio_sources) - 1
            bpy.ops.audiocore.remove_audio_source()
        ace.active_audio_source_index = first_index


def _check_status_and_inputs(addon_key, scene):
    """Engine Status meters, the click timer and the live buffer default."""
    panels = importlib.import_module(f"{addon_key}.ui.panel.panels")
    sync = importlib.import_module(f"{addon_key}.ui.panel.sync")
    usage = importlib.import_module(f"{addon_key}.ui.panel.usage")
    import sys
    import time
    usage._history.clear()
    usage.sample(1000.0)
    time.sleep(0.3)
    usage.sample(1000.5)
    if sys.platform == "win32":
        _assert(usage.average("cpu") is not None and 0.0 <= usage.average("cpu") <= 1.0, f"CPU: {usage.average('cpu')}")
    _assert(usage.average("gpu") is None or 0.0 <= usage.average("gpu") <= 1.0, f"GPU: {usage.average('gpu')}")
    reads = (usage._cpu_load, usage._gpu_load)
    try:
        usage._history.clear()
        values = iter((0.2, 0.4, 0.9))
        usage._cpu_load, usage._gpu_load = lambda: next(values), lambda: None
        usage.sample(2000.0, extra={"total_latency_ms": 2.0})
        usage.sample(2000.5, extra={"total_latency_ms": 4.0})
        _assert(abs(usage.average("cpu") - 0.3) < 1e-9 and usage.average("total_latency_ms") == 3.0, "Not averaged")
        _assert(usage.average("gpu") is None, "An unmeasured GPU has a value")
        usage.sample(2010.0, extra={"total_latency_ms": 6.0})  # the older measures have aged out
        _assert(abs(usage.average("cpu") - 0.9) < 1e-9 and usage.average("total_latency_ms") == 6.0, "Window not applied")
    finally:
        usage._cpu_load, usage._gpu_load = reads

    def status_bars(metrics=None):
        seen: list = []
        context = types.SimpleNamespace(scene=scene)
        original = panels._engine_latency_metrics
        if metrics is not None:
            panels._engine_latency_metrics = lambda _context: dict(metrics)
        try:
            panels.ACE_PT_project_settings.draw(types.SimpleNamespace(layout=_CheckedLayout(seen)), context)
        finally:
            panels._engine_latency_metrics = original
        labels = [value for kind, value in seen if kind == "label"]
        _assert(labels == ["Latency", "CPU", "GPU"], f"Engine status rows: {labels}")
        return [value for kind, value in seen if kind == "progress"]

    # The timeline is stopped (background mode plays nothing): no latency, whatever was averaged before.
    bars = status_bars()
    _assert(len(bars) == 3 and bars[0] == "0.00 ms", f"Engine status bars while stopped: {bars}")
    _assert(sync._engine_latency_metrics(types.SimpleNamespace(scene=scene))["total_latency_ms"] == 0.0,
            "Latency measured while the timeline is stopped")
    # During playback the meter shows the recent average.
    bars = status_bars({"has_data": True, "total_latency_ms": 1.0})
    _assert(len(bars) == 3 and bars[0] == "6.00 ms", f"Engine status bars during playback: {bars}")
    _assert(bpy.app.timers.is_registered(sync._ui_click_timer), "Click timer not registered")
    _assert(sync._UI_CLICK_INTERVAL <= 0.1, "Click timer too slow")
    count = len(scene.audiocore.audio_sources)
    bpy.ops.audiocore.add_audio_source()
    _assert(scene.audiocore.audio_sources[-1].live_latency_ms == 5, "Live buffer default is not 5 ms")
    bpy.ops.audiocore.remove_audio_source()
    _assert(len(scene.audiocore.audio_sources) == count, "Input not removed")
    names = (panels.ACE_PT_audio_sources.bl_label, panels.ACE_PT_information_streams.bl_label,
             panels.ACE_PT_project_settings.bl_category)
    _assert(names == ("Audio Inputs", "Operators", "Audio Core Engine"), f"Panel names: {names}")


def _check_titles_units_and_info(addon_key, scene):
    """The buffer field's unit, the Info panel and the Rolling Average window."""
    panels = importlib.import_module(f"{addon_key}.ui.panel.panels")
    ace = scene.audiocore
    count, first_index = len(ace.audio_sources), ace.active_audio_source_index
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_Titles")
    try:
        bpy.ops.audiocore.add_audio_source()
        source = ace.audio_sources[-1]
        _assert(abs(source.ace_buffer_latency - 0.005) < 1e-9, f"Buffer field: {source.ace_buffer_latency}")
        source.ace_buffer_latency = 0.02
        _assert(source.live_latency_ms == 20, f"Buffer latency: {source.live_latency_ms}")
        unit = type(source).bl_rna.properties["ace_buffer_latency"].unit
        _assert(unit == "TIME_ABSOLUTE", f"Buffer unit: {unit}")

        source.source_type = "LIVE"
        source.live_transport = "UDP"
        source.duration = 125.4
        render = scene.render
        fps = render.fps / render.fps_base
        seen: list = []
        context = types.SimpleNamespace(scene=scene)
        panels.ACE_PT_sub_source_info.draw(types.SimpleNamespace(layout=_CheckedLayout(seen)), context)
        labels = [value for kind, value in seen if kind == "label"]
        frames = int(-(-125.4 * fps // 1))
        span = f"Frame Span: {scene.frame_start} - {scene.frame_start + frames - 1} ({frames} frames)"
        _assert("Duration: 2 min 5 s" in labels and span in labels, f"Info: {labels}")

        # A new Rolling Average takes its window as a time.
        node = _add_ace_node(addon_key, tree, "utilities", "util_rolling_avg", "Rolling")
        item = next(i for i in node.node_tree.interface.items_tree if i.name == "Window")
        _assert(abs(node.inputs["Window"].default_value - 0.25) < 1e-6 and item.subtype == "TIME_ABSOLUTE",
                f"New Rolling Average window: {node.inputs['Window'].default_value}")
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)
        while len(ace.audio_sources) > count:
            ace.active_audio_source_index = len(ace.audio_sources) - 1
            bpy.ops.audiocore.remove_audio_source()
        ace.active_audio_source_index = first_index


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
            return [round(split_freqs.get_split_freq(split, i)) for i in range(len(split.inputs) - 1)]

        def put(number, hz):
            split_freqs.set_split_freq(split, number - 1, hz)

        for number, hz in ((1, 100.0), (2, 500.0), (3, 2000.0)):
            put(number, hz)
        _assert(sockets() == [100, 500, 2000], f"Slider writes: {sockets()}")
        _assert(hasattr(bpy.types, "ACE_OT_sort_split_freqs_on_release"), "Release watcher not registered")

        # While dragging, nothing moves: each slider stays on its socket until the release sorts.
        for value in (600.0, 3000.0, 650.0):
            put(1, value)
            _assert(sockets() == [value, 500, 2000] and sliders() == sockets(), f"Drag to {value}: {sockets()}")
        split_freqs.sort_all_split_freqs()  # what the release (and a file load) does
        _assert(sockets() == [500, 650, 2000] and sliders() == sockets(), f"After release: {sockets()}")

        slider = bpy.types.Node.bl_rna.properties["ace_split_freq_1"]
        _assert(slider.unit == "FREQUENCY", f"Slider unit: {slider.unit}")
        split.ace_split_freq_1 = 200.0
        _assert(abs(split.inputs[1].default_value - 200.0) < 1e-3 and split.ace_split_freq_1 == 200.0, "Hz slider")
        put(1, 650.0)

        put(3, 99999.0)
        put(1, -5.0)
        _assert(sockets() == [0, 650, 20000], f"Limits: {sockets()}")
        _assert([s.name for s in list(split.inputs)[1:]] == ["Split Freq 1", "Split Freq 2", "Split Freq 3"], "Socket names")

        # A linked split frequency is left where it is.
        value_node = tree.nodes.new("ShaderNodeValue")
        tree.links.new(value_node.outputs[0], split.inputs[2])
        put(1, 12000.0)
        put(3, 9000.0)
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
    """Auto-compile rebuilds on content edits only: not idle, nor on selection, a new label or moves.

    Renaming an ACE node recompiles once: its name is its identity.
    """
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
        modes.apply_node_mode(loud, "stream", "loudness", "lufs")
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
            ("label", lambda: setattr(loud, "label", "Kick")),
            ("rename", lambda: setattr(loud, "name", "Kick")),
            ("move", lambda: setattr(loud, "location", (240.0, 0.0))),
            ("mode", switch_mode),
            ("split", lambda: setattr(split, "ace_split_freq_1", 300.0)),
        ):
            edit()
            counts[name] = window()
        expected = {"idle": 0, "select": 0, "active": 0, "label": 0, "rename": 1, "move": 0, "mode": 1, "split": 1}
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
                       and it.target == _sid(source))
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
        _assert(active_key() == _sid(source), "Clicking an instance did not activate its source")
        _assert(ace.active_source_row == inst_row, "The highlight left the clicked instance")
        ace.active_audio_source_index = before
        _assert(ace.source_rows[ace.active_source_row].target == active_key(), "Highlight does not follow the active source")
        bpy.ops.audiocore.remove_audio_source()
        _assert(len([it for it in ace.source_rows if it.kind == "source"]) == before, "Removed source still listed")
        ace.active_source_row = src_row
        _assert(active_key() == _sid(source), "Clicking a source did not make it active")
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

    def progress(self, text="", factor=0.0, **_kw):
        _assert(0.0 <= factor <= 1.0, f"Progress factor {factor}")
        self.seen.append(("progress", text))

    def operator_menu_enum(self, idname, prop, text="", **_kw):
        self.operator(idname)

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
        active.active_ui_node_id = _nid(split)
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

        _assert({"bpm", "beats_per_bar"} <= props("ACE_PT_sub_tempo_meter"), "Tempo panel")
        _assert(("operator", "audiocore.reset_engine") in drawn["ACE_PT_sub_engine_settings"], "No reset button")
        _assert({"ace_split_freq_1", "ace_band_fill"} <= props("ACE_PT_information_streams"),
                f"Splitter settings: {props('ACE_PT_information_streams')}")
        for name in ("ACE_PT_sub_source_info", "ACE_PT_sub_tempo_meter"):
            _assert("DEFAULT_CLOSED" in getattr(panels, name).bl_options, f"{name} is not closed by default")
        order = [panels.ACE_PT_sub_source_info.bl_order, panels.ACE_PT_sub_tempo_meter.bl_order]
        _assert(order == sorted(order), f"Source sub-panel order: {order}")
    finally:
        sync._find_active_node_tree = original_tree
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)


def _check_manual_engine_reset(addon_key, scene):
    """Manual Engine Reset replaces the engine, empties the shared caches and compiles again."""
    engine_cls = importlib.import_module(f"{addon_key}.core.engine.engine").ACEAudioEngine
    spectral = importlib.import_module(f"{addon_key}.core.tasks.spectral")
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_Reset")
    try:
        source = _add_ace_node(addon_key, tree, "source_mngmt", "src", "Source")
        loud = _add_ace_node(addon_key, tree, "stream", "loudness", "Loudness")
        tree.links.new(source.outputs["Audio"], loud.inputs["Audio"])
        before = engine_cls.ensure_instance(scene)
        before.compiler.compile_scene_frontends(scene)
        spectral._SPECTRUM_CACHE["stale"] = None
        _assert(bpy.ops.audiocore.reset_engine() == {'FINISHED'}, "Reset failed")
        after = engine_cls.get_instance(scene)
        _assert(after is not None and after is not before, "Reset kept the old engine")
        _assert(not before.graphs and after.graphs, "Reset did not rebuild the graphs")
        _assert("stale" not in spectral._SPECTRUM_CACHE, "Reset kept a stale cache")
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


def _check_file_formats_decode(addon_key):
    """Blender's audio module decodes compressed files for ACE: FLAC exactly like the WAV, MP3 and Ogg to the same length."""
    import aud  # type: ignore
    import numpy as np
    decode = importlib.import_module(f"{addon_key}.core.input.decode")
    reception = importlib.import_module(f"{addon_key}.core.input.reception").AudioReception
    _assert(decode.AUD_AVAILABLE, "Blender's audio module did not load: only WAV files can be read")
    for ext in (".mp3", ".flac", ".ogg", ".wav"):
        _assert(ext in reception.supported_file_extensions(), f"{ext} is not offered")
    tmp_dir = tempfile.mkdtemp(prefix="ace_smoke_formats_")
    wav_path = os.path.join(tmp_dir, "tone.wav")
    _write_test_wav(wav_path, seconds=2.0)
    reference = decode.decode(wav_path)
    for name, container, codec, exact in (
        ("tone.flac", aud.CONTAINER_FLAC, aud.CODEC_FLAC, True),
        ("tone.mp3", aud.CONTAINER_MP3, aud.CODEC_MP3, False),
        ("tone.ogg", aud.CONTAINER_OGG, aud.CODEC_VORBIS, False),
    ):
        path = os.path.join(tmp_dir, name)
        aud.Sound(wav_path).write(path, 48000, aud.CHANNELS_MONO, aud.FORMAT_S16, container, codec)
        _assert(reception.is_supported_file_path(path), f"{name} is not readable")
        decoded = decode.decode(path)
        _assert(decoded.sample_rate == 48000, f"{name}: {decoded.sample_rate} Hz")
        if exact:
            _assert(np.array_equal(decoded.samples, reference.samples), f"{name} does not decode like the WAV")
        else:
            frames = min(decoded.samples.shape[0], reference.samples.shape[0])
            _assert(abs(decoded.samples.shape[0] - reference.samples.shape[0]) <= 4800, f"{name}: {decoded.samples.shape[0]} frames")
            rms = float(np.sqrt(np.mean(decoded.samples[:frames] ** 2)))
            ref_rms = float(np.sqrt(np.mean(reference.samples[:frames] ** 2)))
            _assert(abs(rms - ref_rms) < 0.05 * ref_rms, f"{name}: level {rms:.4f} vs {ref_rms:.4f}")
        rec = reception("FILE", filepath=path)
        _assert(rec.open(block_size=2000), f"{name} did not open: {rec.last_error}")
        block = rec.read_block(2000)
        _assert(block is not None and block[0].shape == (2000,), f"{name}: no block")
        rec.close()


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
    blender_helpers.add_geometry_passthrough(tree)
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
    played: dict = {}  # frame -> {output key: value written}
    for frame in range(1, 73):
        # Real-time pace (24 fps): worker results arrive frames later.
        target = t0 + (frame - 1) / 24.0
        time.sleep(max(0.0, target - time.monotonic()))
        engine.latest_values.clear()
        scene.frame_set(frame)
        played[frame] = dict(engine.latest_values)
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
    # The bake streams on its own fresh tasks, so it equals what playback from the first frame
    # computed. With worker processes, only main-thread values are exact (worker values may be
    # up to 50 ms old).
    graph = next(iter(engine.graphs.values()))
    entry = engine.bake._bake_cache.get(str(getattr(graph, "source_uid", "")))
    _assert(entry is not None, "The bake left no cache entry")
    keys = list(graph.output_attr_keys)
    if parallel_mode == "processes_dsp":
        keys = [k for k in keys if k not in set(getattr(graph, "parallel_offload_attr_keys", []) or [])]
    _assert(keys, "No output to compare the bake with")
    import numpy as np
    mismatches = []
    compared = 0
    for key in keys:
        series = entry["values"].get(key)
        _assert(series is not None, f"The bake has no curve for {key}")
        for frame, values in played.items():
            if key not in values or not isinstance(values[key], float):
                continue
            compared += 1
            baked = float(series[frame - entry["frame_start"]])
            if abs(baked - float(np.float32(values[key]))) > 1e-6:
                mismatches.append((key, frame, baked, values[key]))
    _assert(compared > 0, "No played value to compare the bake with")
    _assert(not mismatches, f"The bake differs from playback ({len(mismatches)} values), first: {mismatches[:3]}")
    print(f"[ACE smoke] {parallel_mode}: bake matches playback on {compared} values")

    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.node_groups.remove(tree)


def main():
    print("[ACE smoke] Blender-side smoke script loaded.")

    # Basic sanity checks users can expand:
    _assert(bpy.app.version >= (5, 2, 0), "Blender 5.2+ is required")
    addon_key = find_addon_key()
    addon_keys = [addon_key]
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

    def datablocks():
        return (len(bpy.data.objects), len(bpy.data.node_groups), len(bpy.data.materials), len(bpy.data.meshes))

    made = datablocks()
    add_result = bpy.ops.audiocore.add_audio_source()
    _assert('FINISHED' in add_result, "audiocore.add_audio_source failed")
    _assert(len(ace.audio_sources) == before_count + 1, "Audio source count did not increase")
    _assert(datablocks() == made, f"Adding an input created data: {made} -> {datablocks()}")

    active_idx = int(getattr(ace, "active_audio_source_index", -1))
    _assert(0 <= active_idx < len(ace.audio_sources), "Active source index invalid after add")
    src = ace.audio_sources[active_idx]
    _assert(bool(getattr(src, "stable_id", "")), "Added source has no stable_id")
    engine.compiler.compile_scene_frontends(scene)
    _assert(datablocks() == made, f"Compiling an unused input created data: {made} -> {datablocks()}")

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
        blender_helpers.add_geometry_passthrough(frontend_gn_tree)
        _add_ace_node(addon_key, frontend_gn_tree, "stream", "loudness", "Loudness")
        try:
            frontend_gn_tree.is_modifier = True
        except Exception:
            pass
        frontend_mod = frontend_obj.modifiers.new("ACE Smoke GN", "NODES")
        frontend_mod.node_group = frontend_gn_tree

        frontend_mat = bpy.data.materials.new("ACE_Smoke_Material")
        frontend_shader_tree = frontend_mat.node_tree
        _assert(frontend_shader_tree is not None, "Material node tree was not created")
        _add_ace_node(addon_key, frontend_shader_tree, "stream", "loudness", "Loudness")
        frontend_obj.data.materials.append(frontend_mat)

        _check_object_and_material_edits_queue_recompile(
            engine, scene, frontend_obj, frontend_mod, frontend_gn_tree, frontend_shader_tree
        )

        discovered = engine.frontends.discover_scene_frontends(scene)
        kinds = sorted(rec.frontend_ref.kind for rec in discovered)
        _assert("GEOMETRY_NODES" in kinds, "Scene frontend discovery missed Geometry Nodes frontend")
        _assert("SHADER_NODES" in kinds, "Scene frontend discovery missed Shader frontend")

        engine.compiler.compile_scene_frontends(scene)
        compiled = engine.compiler._compiled_frontend_snapshots
        for tree, kind in ((frontend_gn_tree, "GN"), (frontend_shader_tree, "shader")):
            _assert(engine.frontends._tree_frontend_uid(tree) in compiled, f"Scene compile missed the {kind} frontend")
    finally:
        if frontend_obj is not None:
            bpy.data.objects.remove(frontend_obj, do_unlink=True)
        if frontend_mat is not None:
            bpy.data.materials.remove(frontend_mat, do_unlink=True)
        if frontend_gn_tree is not None and frontend_gn_tree.users == 0:
            bpy.data.node_groups.remove(frontend_gn_tree)

    _check_file_source_drives_proxy(addon_key, engine, scene, src, parallel_mode="threads_io")
    _check_file_source_drives_proxy(addon_key, engine, scene, src, parallel_mode="processes_dsp")
    _check_shader_values_reach_scene(addon_key, scene, engine, src)
    _check_compile_writes_nothing(addon_key, scene, engine)
    _check_sink_name_collisions_are_repaired_once(addon_key, scene, engine)
    _check_linked_tree_compiles(addon_key, scene, engine)
    _check_instances_through_groups(addon_key, scene, engine)
    _check_signals_are_shared_and_kept(addon_key, scene, engine)
    _check_world_and_instanced_trees_are_found(addon_key, scene, engine)
    _check_file_formats_decode(addon_key)
    _check_operator_list_follows_tree(addon_key, scene)
    _check_splitter_sliders(addon_key, scene)
    _check_every_splitter_lists_its_bands(addon_key, scene)
    _check_computed_link_values(addon_key, scene)
    _check_driven_splits(addon_key, scene, engine)
    _check_selection_follows_sources(addon_key, scene)
    _check_new_source_takes_input_name(addon_key, scene)
    _check_viewer_previews_audio(addon_key, scene, engine)
    _check_cosmetic_edits_do_not_recompile(addon_key, scene, engine)
    _check_source_list_instances(addon_key, scene, engine)
    _check_group_rebuild_keeps_input_values(addon_key, scene)
    _check_panels_draw(addon_key, scene)
    _check_status_and_inputs(addon_key, scene)
    _check_titles_units_and_info(addon_key, scene)
    _check_source_nodes(addon_key, engine=engine, scene=scene)
    _check_source_node_outputs_tempo(addon_key, scene, engine)
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
