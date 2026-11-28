"""Smoke checks of ACE nodes as the Node Editor shows them: group names, nodes added by Blender's
search, inputs in use, mode inputs, cross-tree paste, the internals' layout, shader-only sources.

Called from blender_integration_smoke.py; each check cleans up what it made.
"""

import importlib
import os
import tempfile

import bpy  # type: ignore

import blender_helpers
from blender_helpers import _add_ace_node, _new_gn_tree


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _mod(addon_key, name):
    return importlib.import_module(f"{addon_key}.{name}")


def _kinds(addon_key):
    ace_nodes = _mod(addon_key, "vars").ACE_NODES
    return [(cat, typ) for cat, cat_def in ace_nodes.items() for typ in cat_def["nodes"]]


def check_template_names(addon_key):
    """Each kind's shared group is named after its node, " (Shader)" in shaders, and found by its kind."""
    groups = _mod(addon_key, "nodes.groups")
    vars_mod = _mod(addon_key, "vars")
    factories = {
        "GeometryNodeTree": _mod(addon_key, "nodes.geometry.factory").ACEDummyNodeFactory,
        "ShaderNodeTree": _mod(addon_key, "nodes.shader.factory").ACEShaderNodeFactory,
    }
    for kind in _kinds(addon_key):
        label = vars_mod.node_def(*kind)["label"]
        for tree_type, name in (("GeometryNodeTree", label), ("ShaderNodeTree", f"{label} (Shader)")):
            factories[tree_type].template_group(*kind)  # built on first use, as by the menu
            group = groups.find_template(*kind, tree_type)
            _assert(group is not None, f"No {tree_type} template for {kind}")
            _assert(group.name == name and not group.name.startswith("."), f"{kind} {tree_type} template is named {group.name!r}")
    # A user's own group with a template's name is never taken for it.
    users_group = bpy.data.node_groups.new("Envelope", "GeometryNodeTree")
    try:
        _assert(users_group.name == "Envelope.001", f"The user's group got {users_group.name!r}")
        found = groups.find_template("controllers", "envelope", "GeometryNodeTree")
        _assert(found is not users_group and found.name == "Envelope", f"find_template found {found.name!r}")
    finally:
        bpy.data.node_groups.remove(users_group)
    print("[ACE smoke] templates: named after their nodes, found by kind")


def _search_add(tree, group):
    """What Blender's link-drag search and Add > Group do: a group node, nothing else."""
    node = tree.nodes.new("ShaderNodeGroup" if tree.bl_idname == "ShaderNodeTree" else "GeometryNodeGroup")
    node.node_tree = group
    return node


def check_search_added_nodes_are_placed(addon_key, scene, engine, src):
    """A node added by search is placed when its tree changes: group, label, mode, source, sink names."""
    groups = _mod(addon_key, "nodes.groups")
    identity = _mod(addon_key, "nodes.identity")
    source_nodes = _mod(addon_key, "nodes.source_nodes")
    scene.audiocore.active_audio_source_index = list(scene.audiocore.audio_sources).index(src)
    obj, gn_tree = _new_gn_tree(scene, "ACE_Smoke_Search")
    material = bpy.data.materials.new("ACE_Smoke_Search_Mat")
    obj.data.materials.append(material)
    try:
        for tree in (gn_tree, material.node_tree):
            tree_type = tree.bl_idname
            made = {kind: _search_add(tree, groups.find_template(*kind, tree_type)) for kind in (
                ("input", "src"), ("audio", "frequency_splitter"), ("audio", "loudness"), ("controllers", "envelope"))}
            bpy.context.view_layer.update()  # the depsgraph handler places them
            source, split, loud, env = (made[k] for k in made)
            _assert(source_nodes.node_source_id(source) == src.stable_id, f"{tree_type}: the searched Source reads no source")
            _assert(not groups.is_template(split.node_tree) and len(split.outputs) == 2,
                    f"{tree_type}: the searched splitter kept {split.node_tree.name}")
            _assert(loud.label == "Loudness (RMS)" and loud.get("ace_mode") == "rms", f"{tree_type}: Loudness {loud.label!r}")
            _assert(env.label == "Envelope" and env.name == "Envelope", f"{tree_type}: Envelope {env.label!r} / {env.name!r}")
            _assert(env.inputs["Bounce"].hide and not env.inputs["Attack"].hide, f"{tree_type}: Follow shows Spring's inputs")
            engine.compiler.compile_scene_frontends(scene)
            names = identity.sink_names(loud)
            _assert(names.get(0) and not names[0].startswith("ACE_undefined"), f"{tree_type}: searched Loudness writes {names}")
            if tree_type == "ShaderNodeTree":
                _assert(loud.node_tree.name.startswith(".ACE_sink_"), f"Shader Loudness uses {loud.node_tree.name}")
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.materials.remove(material)
        bpy.data.node_groups.remove(gn_tree)
        engine.compiler.compile_scene_frontends(scene)
    print("[ACE smoke] nodes added by search are placed in both editors")


def check_inputs_in_use(addon_key, scene):
    """Blender greys an input its group doesn't use: no ACE node shows one, in either editor.

    Inputs hidden by the node's mode are left out: they are hidden, not greyed.
    """
    obj, gn_tree = _new_gn_tree(scene, "ACE_Smoke_InUse")
    material = bpy.data.materials.new("ACE_Smoke_InUse_Mat")
    greyed = []
    try:
        for tree in (gn_tree, material.node_tree):
            nodes = []
            for kind in _kinds(addon_key):
                node = _add_ace_node(addon_key, tree, *kind, kind[1])
                nodes.append((kind, node))
            three = _add_ace_node(addon_key, tree, "audio", "frequency_splitter", "Three")
            _mod(addon_key, "nodes.dynamic_audio_operators").ACESplitterManager.apply_variant_to_node(three, 3)
            nodes.append((("audio", "frequency_splitter"), three))
            bpy.context.view_layer.update()
            for kind, node in nodes:
                for sock in node.inputs:
                    if sock.enabled and not sock.hide and sock.is_inactive:
                        greyed.append(f"{tree.bl_idname} {node.name}.{sock.name}")
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.materials.remove(material)
        bpy.data.node_groups.remove(gn_tree)
    # A group without outputs (Audio Preview) has nothing to use its input for.
    greyed = [g for g in greyed if "preview_sink.Audio" not in g]
    _assert(not greyed, f"Inputs drawn unused: {greyed}")
    print("[ACE smoke] every ACE input is in use, in both editors")


def check_mode_hides_inputs(addon_key, scene):
    """Envelope's Mode shows only the inputs it uses; a linked input always shows."""
    obj, gn_tree = _new_gn_tree(scene, "ACE_Smoke_Modes")
    material = bpy.data.materials.new("ACE_Smoke_Modes_Mat")
    obj.data.materials.append(material)  # an unused material isn't evaluated: its edits send no update
    try:
        for tree in (gn_tree, material.node_tree):
            env = _add_ace_node(addon_key, tree, "controllers", "envelope", "Envelope")
            shown = lambda: sorted(s.name for s in env.inputs if not s.hide)  # noqa: E731
            _assert(shown() == sorted(["Value", "Mode", "Attack", "Hold", "Release"]), f"{tree.bl_idname} Follow: {shown()}")
            env.inputs["Mode"].default_value = "Spring"
            bpy.context.view_layer.update()
            _assert(shown() == sorted(["Value", "Mode", "Speed", "Bounce"]), f"{tree.bl_idname} Spring: {shown()}")
            value = tree.nodes.new("ShaderNodeValue")
            tree.links.new(value.outputs[0], env.inputs["Bounce"])
            env.inputs["Mode"].default_value = "Follow"
            bpy.context.view_layer.update()
            _assert("Bounce" in shown(), f"{tree.bl_idname}: a linked Bounce was hidden")
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.materials.remove(material)
        bpy.data.node_groups.remove(gn_tree)
    print("[ACE smoke] Envelope's Mode shows only its inputs")


def _node_editor(context):
    window = context.window_manager.windows[0]
    area = max(window.screen.areas, key=lambda a: a.width * a.height)
    old_type = area.type
    area.type = "NODE_EDITOR"
    region = next(r for r in area.regions if r.type == "WINDOW")
    return window, area, region, old_type


def check_cross_tree_paste(addon_key, scene, src):
    """ACE nodes copied in Geometry Nodes paste into a material with shader groups, and back."""
    obj, gn_tree = _new_gn_tree(scene, "ACE_Smoke_Paste")
    material = bpy.data.materials.new("ACE_Smoke_Paste_Mat")
    groups = _mod(addon_key, "nodes.groups")
    source_nodes = _mod(addon_key, "nodes.source_nodes")
    scene.audiocore.active_audio_source_index = list(scene.audiocore.audio_sources).index(src)
    window, area, region, old_type = _node_editor(bpy.context)
    space = area.spaces.active
    try:
        source = _add_ace_node(addon_key, gn_tree, "input", "src", "Source")
        loud = _add_ace_node(addon_key, gn_tree, "audio", "loudness", "Peak", mode="peak")
        env = _add_ace_node(addon_key, gn_tree, "controllers", "envelope", "Envelope")
        double = gn_tree.nodes.new("ShaderNodeMath")
        double.name = "Double"
        double.operation = "MULTIPLY"
        gn_tree.links.new(source.outputs["Audio"], loud.inputs["Audio"])
        gn_tree.links.new(loud.outputs[0], double.inputs[0])
        gn_tree.links.new(double.outputs[0], env.inputs["Value"])
        env.inputs["Release"].default_value = 1.25
        for node in gn_tree.nodes:
            node.select = node in (source, loud, env, double)
        for tree_from, tree_to in ((gn_tree, material.node_tree), (material.node_tree, None)):
            space.tree_type = tree_from.bl_idname
            space.node_tree = tree_from
            with bpy.context.temp_override(window=window, area=area, region=region, space_data=space):
                _assert(bpy.ops.ace.clipboard_copy() == {"FINISHED"}, "ACE copy failed")
            target = tree_to
            if target is None:
                target = bpy.data.node_groups.new("ACE_Smoke_Paste_Back", "GeometryNodeTree")
            space.tree_type = target.bl_idname
            space.node_tree = target
            with bpy.context.temp_override(window=window, area=area, region=region, space_data=space):
                _assert(bpy.ops.ace.clipboard_paste() == {"FINISHED"}, "ACE paste failed")
            node_kind = _mod(addon_key, "nodes.identity").node_kind
            by_label = {n.label or n.name: n for n in target.nodes}
            by_label.update({"Source": n for n in target.nodes if node_kind(n) == ("input", "src")})
            for label in ("Source", "Peak", "Envelope"):
                node = by_label.get(label)
                _assert(node is not None and node.node_tree is not None and node.node_tree.bl_idname == target.bl_idname,
                        f"{label} was not pasted into {target.bl_idname}: {sorted(by_label)}")
            _assert(by_label["Peak"].get("ace_mode") == "peak", "The pasted Loudness lost its mode")
            _assert(abs(by_label["Envelope"].inputs["Release"].default_value - 1.25) < 1e-6, "The pasted Envelope lost Release")
            _assert(source_nodes.node_source_id(by_label["Source"]) == src.stable_id, "The pasted Source reads no source")
            _assert(not groups.is_template(by_label["Source"].node_tree) or target.bl_idname == "GeometryNodeTree",
                    "The pasted shader Source kept the template")
            def _name(n):
                return "Source" if node_kind(n) == ("input", "src") else (n.label or n.name)

            linked = {(_name(l.from_node), _name(l.to_node)) for l in target.links}
            for pair in (("Source", "Peak"), ("Peak", "Double"), ("Double", "Envelope")):
                _assert(any(a == pair[0] and b.startswith(pair[1]) for a, b in linked), f"Pasted links {linked} miss {pair}")
            for node in target.nodes:
                node.select = True
            if tree_to is None:
                bpy.data.node_groups.remove(target)
    finally:
        area.type = old_type
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.materials.remove(material)
        bpy.data.node_groups.remove(gn_tree)
    print("[ACE smoke] ACE nodes paste across Geometry Nodes and shaders, links and settings kept")


def _rect(layout, node):
    width, height = layout.node_size(node)
    x, y = node.location
    return x, y - height, x + width, y


def check_internals_layout(addon_key, scene):
    """Inside every ACE group, nodes don't overlap and every link runs left to right."""
    layout = _mod(addon_key, "nodes.layout")
    _mod(addon_key, "nodes.dynamic_audio_operators").ACESplitterManager.get_or_create_variant(4, "ShaderNodeTree")
    _mod(addon_key, "nodes.dynamic_audio_operators").ACESplitterManager.get_or_create_variant(4, "GeometryNodeTree")
    _mod(addon_key, "nodes.shader.factory").ACEShaderNodeFactory.ensure_sink_group("controllers", "envelope", ["ACE_layout_env"])
    problems = []
    checked = 0
    for group in bpy.data.node_groups:
        if not _mod(addon_key, "nodes.identity").is_ace_group(group) or group.library is not None:
            continue
        nodes = [n for n in group.nodes if n.bl_idname != "NodeFrame"]
        rects = {n: _rect(layout, n) for n in nodes}
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                ra, rb = rects[a], rects[b]
                if ra[0] < rb[2] and rb[0] < ra[2] and ra[1] < rb[3] and rb[1] < ra[3]:
                    problems.append(f"{group.name}: {a.name} overlaps {b.name}")
        for link in group.links:
            if rects[link.from_node][2] > rects[link.to_node][0] + 1e-3:
                problems.append(f"{group.name}: {link.from_node.name} -> {link.to_node.name} runs backwards")
        checked += 1
    sink = bpy.data.node_groups.get(".ACE_sink_ACE_layout_env")
    if sink is not None:
        bpy.data.node_groups.remove(sink)
    _assert(checked > 20 and not problems, f"{checked} groups, layout problems: {problems[:12]}")
    print(f"[ACE smoke] internals layout: {checked} groups, no overlap, links left to right")


def check_shader_only_source_controller(addon_key, scene, engine):
    """A controller in a material whose source no Geometry Nodes tree reads still follows it.

    The copy that computes it reads the measurement from the source's proxy, made for it.
    """
    mirror = _mod(addon_key, "nodes.shader.mirror")
    ace = scene.audiocore
    bpy.ops.audiocore.add_audio_source()
    src = ace.audio_sources[ace.active_audio_source_index]
    wav = os.path.join(tempfile.mkdtemp(prefix="ace_smoke_"), "shader_only.wav")
    blender_helpers._write_test_wav(wav)
    src.filepath = wav
    mesh = bpy.data.meshes.new("ACE_Smoke_ShaderOnly_Mesh")
    obj = bpy.data.objects.new("ACE_Smoke_ShaderOnly", mesh)
    scene.collection.objects.link(obj)
    material = bpy.data.materials.new("ACE_Smoke_ShaderOnly_Mat")
    obj.data.materials.append(material)
    tree = material.node_tree
    try:
        _assert(not src.proxy_object_name, "A new source has a proxy already")
        source = _add_ace_node(addon_key, tree, "input", "src", "Source")
        split = _add_ace_node(addon_key, tree, "audio", "frequency_splitter", "Splitter")
        loud = _add_ace_node(addon_key, tree, "audio", "loudness", "Loudness", mode="rms")
        env = _add_ace_node(addon_key, tree, "controllers", "envelope", "Envelope")
        env.inputs["Attack"].default_value = 0.0
        env.inputs["Release"].default_value = 0.0
        tree.links.new(source.outputs["Audio"], split.inputs["Audio"])
        tree.links.new(split.outputs[0], loud.inputs["Audio"])
        tree.links.new(loud.outputs[0], env.inputs["Value"])
        engine.compiler.compile_scene_frontends(scene)
        env_name = blender_helpers.sink_names(addon_key, env)[0]
        loud_name = blender_helpers.sink_names(addon_key, loud)[0]
        followed = 0
        for frame in range(1, 30):
            scene.frame_set(frame)
            loud_value, env_value = scene.get(loud_name), scene.get(env_name)
            if loud_value is not None and env_value is not None and float(loud_value) > 0.01:
                _assert(abs(float(env_value) - float(loud_value)) < 1e-5, f"Frame {frame}: Envelope {env_value}, Loudness {loud_value}")
                followed += 1
        _assert(followed >= 20, f"The Envelope followed {followed} frames (mirror reads {sorted(engine.mirror_reads)})")
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.materials.remove(material)
        bpy.ops.audiocore.remove_audio_source()
        engine.compiler.compile_scene_frontends(scene)
        _assert(bpy.data.objects.get(mirror.MIRROR_OBJECT) is None, "The mirror stayed")
    print(f"[ACE smoke] a shader-only source drives a shader Envelope on {followed} frames")


def check_controllers_in_operator_list(addon_key, scene):
    """A controller is listed under the measurement it reads, through a Math node."""
    sync = _mod(addon_key, "ui.panel.sync")
    identity = _mod(addon_key, "nodes.identity")
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_CtrlRows")
    try:
        loud = _add_ace_node(addon_key, tree, "audio", "loudness", "Loudness")
        math = tree.nodes.new("ShaderNodeMath")
        env = _add_ace_node(addon_key, tree, "controllers", "envelope", "Envelope")
        tree.links.new(loud.outputs[0], math.inputs[0])
        tree.links.new(math.outputs[0], env.inputs["Value"])
        nodes, links = sync._controller_rows(tree)
        env_id, loud_id = identity.node_id(env, tree), identity.node_id(loud, tree)
        _assert([n.id for n in nodes] == [env_id], f"Controller rows: {[n.id for n in nodes]}")
        _assert([(l.source_id, l.target_id) for l in links] == [(loud_id, env_id)], f"Controller links: {links}")
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)
    print("[ACE smoke] controllers are listed under what they read")


def check_unlisted_node_selected(addon_key, scene, src):
    """A controller that reads no source has no row: selected in the editor, the panel shows it, not the last row."""
    sync = _mod(addon_key, "ui.panel.sync")
    utils = _mod(addon_key, "utils")
    scene.audiocore.active_audio_source_index = list(scene.audiocore.audio_sources).index(src)
    obj, tree = _new_gn_tree(scene, "ACE_Smoke_Unlisted")
    finders = [(m, m._find_active_node_tree) for m in (sync, _mod(addon_key, "ui.panel.helpers"))]
    try:
        source = _add_ace_node(addon_key, tree, "input", "src", "Source")
        loud = _add_ace_node(addon_key, tree, "audio", "loudness", "Loudness")
        lone = _add_ace_node(addon_key, tree, "controllers", "running_total", "Lone Total")
        tree.links.new(source.outputs["Audio"], loud.inputs["Audio"])
        for module, _finder in finders:
            module._find_active_node_tree = lambda _context, **_kw: tree
        sync._sync_ui_dummy_tree(bpy.context, force=True)
        _, active = utils._get_active_source(bpy.context)
        for node in (loud, lone):
            tree.nodes.active = node
            sync._sync_editor_active(tree, active)
            shown, _parsed = sync._resolve_ui_list_selected_node(bpy.context)
            _assert(shown == node, f"Selecting {node.name} shows {getattr(shown, 'name', None)}")
        _assert(active.active_ui_node_id == _mod(addon_key, "nodes.identity").node_id(loud, tree),
                "Selecting a node without a row changed the list's selection")
        tree.nodes.active = loud
        sync._sync_editor_active(tree, active)
        shown, _parsed = sync._resolve_ui_list_selected_node(bpy.context)
        _assert(shown == loud, "A listed node selected again does not show")
    finally:
        for module, finder in finders:
            module._find_active_node_tree = finder
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.node_groups.remove(tree)
        sync.reset_state()
    print("[ACE smoke] a node without a row shows in the panel when selected in the editor")


def check_templates_built_after_enable(addon_key):
    """bpy.data is restricted while an add-on registers: the templates are built by a timer just after."""
    handlers = _mod(addon_key, "core.engine.handlers")
    _assert(hasattr(handlers, "_build_groups_after_enable"), "No deferred template build")
    for group in [g for g in bpy.data.node_groups if _mod(addon_key, "nodes.groups").is_template(g) and g.users == 0]:
        bpy.data.node_groups.remove(group)
    handlers._build_groups_after_enable()
    groups = _mod(addon_key, "nodes.groups")
    missing = [kind for kind in _kinds(addon_key) for tree_type in ("GeometryNodeTree", "ShaderNodeTree")
               if groups.find_template(*kind, tree_type) is None]
    _assert(not missing, f"Templates missing after the deferred build: {missing}")
    print("[ACE smoke] the templates are built once the add-on is enabled")


def run_all(addon_key, scene, engine, src):
    check_templates_built_after_enable(addon_key)
    check_template_names(addon_key)
    check_search_added_nodes_are_placed(addon_key, scene, engine, src)
    check_inputs_in_use(addon_key, scene)
    check_mode_hides_inputs(addon_key, scene)
    check_cross_tree_paste(addon_key, scene, src)
    check_internals_layout(addon_key, scene)
    check_shader_only_source_controller(addon_key, scene, engine)
    check_controllers_in_operator_list(addon_key, scene)
    check_unlisted_node_selected(addon_key, scene, src)
