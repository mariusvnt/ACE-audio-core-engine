"""
Run inside Blender (background or UI) after enabling the add-on.

Headless, from the repo root (enables the add-on from this checkout):
  python tests/run_in_blender.py
"""

import bpy  # type: ignore
import importlib
import os
import tempfile


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _check_object_and_material_edits_queue_recompile(engine, scene, obj, modifier, gn_tree, shader_tree):
    """Depsgraph updates on an object or material reach request_recompile with the original trees."""
    settings = scene.audiocore.engine_settings
    previous_auto = bool(settings.auto_compile)
    settings.auto_compile = True
    requested = []
    original_request = engine.request_recompile
    engine.request_recompile = lambda tree: requested.append(tree)
    try:
        bpy.context.view_layer.update()
        requested.clear()
        modifier.show_viewport = not modifier.show_viewport
        bpy.context.view_layer.update()
        obj.data.materials[0].diffuse_color = (0.2, 0.4, 0.6, 1.0)
        bpy.context.view_layer.update()
    finally:
        engine.request_recompile = original_request
        settings.auto_compile = previous_auto
    pointers = {t.as_pointer() for t in requested if t is not None}
    _assert(gn_tree.as_pointer() in pointers, "Object edit did not queue its Geometry Nodes frontend")
    _assert(shader_tree.as_pointer() in pointers, "Material edit did not queue its shader frontend")
    _assert(all(not t.is_evaluated for t in requested if t is not None), "Recompile queued an evaluated copy")


def _stale_ace_groups(addon_key):
    dummy_tree = importlib.import_module(f"{addon_key}.nodes.dummy_tree")
    shader_tree = importlib.import_module(f"{addon_key}.nodes.shader_tree")
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
    dummy_tree = importlib.import_module(f"{addon_key}.nodes.dummy_tree")
    lufs_name = dummy_tree.ACEDummyNodeFactory.get_group_name("stream", "lufs")
    bpy.data.node_groups[lufs_name].interface.new_socket("Leftover", in_out="INPUT", socket_type="NodeSocketFloat")
    path = os.path.join(tempfile.mkdtemp(prefix="ace_smoke_"), "outdated_groups.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)
    names = [i.name for i in bpy.data.node_groups[lufs_name].interface.items_tree if i.in_out == "INPUT"]
    _assert("Leftover" not in names, f"Outdated ACE group kept after load: {names}")
    _assert(not _stale_ace_groups(addon_key), "ACE groups still stale after load")


def main():
    print("[ACE smoke] Blender-side smoke script loaded.")

    # Basic sanity checks users can expand:
    _assert(bpy.app.version >= (5, 0, 0), "Blender 5.0+ is required")
    addon_keys = [k for k in bpy.context.preferences.addons.keys() if "AudioCore" in k]
    _assert(bool(addon_keys), "AudioCore add-on is not enabled in current Blender session")
    addon_key = addon_keys[0]
    engine_mod = importlib.import_module(f"{addon_key}.core.audio_engine")
    _assert(
        engine_mod.frame_change_handler in bpy.app.handlers.frame_change_pre,
        "frame_change_handler is not registered",
    )
    _assert(
        engine_mod.depsgraph_update_handler in bpy.app.handlers.depsgraph_update_post,
        "depsgraph_update_handler is not registered",
    )
    _assert(
        engine_mod._load_post_reset_engine in bpy.app.handlers.load_post,
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
    engine.execute_next_frame(scene, force=True)

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
    _assert(bool(getattr(sonar_mat, "use_nodes", False)), "ACE_Sonar shader material is not node-based")
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
        frontend_mat.use_nodes = True
        frontend_shader_tree = frontend_mat.node_tree
        _assert(frontend_shader_tree is not None, "Material node tree was not created")
        frontend_shader_tree["ace_frontend_uid"] = "ACE_SMOKE_FRONTEND_SHADER"
        frontend_obj.data.materials.append(frontend_mat)

        _check_object_and_material_edits_queue_recompile(
            engine, scene, frontend_obj, frontend_mod, frontend_gn_tree, frontend_shader_tree
        )

        discovered = engine.discover_scene_frontends(scene)
        kinds = sorted(rec.frontend_ref.kind for rec in discovered)
        _assert("GEOMETRY_NODES" in kinds, "Scene frontend discovery missed Geometry Nodes frontend")
        _assert("SHADER_NODES" in kinds, "Scene frontend discovery missed Shader frontend")

        engine.compile_scene_frontends(scene)
        _assert(
            "ACE_SMOKE_FRONTEND_GN" in getattr(engine, "_compiled_frontend_snapshots", {}),
            "Scene compile missed the GN frontend",
        )
        _assert(
            "ACE_SMOKE_FRONTEND_SHADER" in getattr(engine, "_compiled_frontend_snapshots", {}),
            "Scene compile missed the shader frontend",
        )
    finally:
        if frontend_obj is not None:
            bpy.data.objects.remove(frontend_obj, do_unlink=True)
        if frontend_mat is not None:
            bpy.data.materials.remove(frontend_mat, do_unlink=True)
        if frontend_gn_tree is not None and frontend_gn_tree.users == 0:
            bpy.data.node_groups.remove(frontend_gn_tree)

    remove_result = bpy.ops.audiocore.remove_audio_source()
    _assert('FINISHED' in remove_result, "audiocore.remove_audio_source failed")
    _assert(len(ace.audio_sources) == before_count, "Audio source count did not return to initial value")

    _check_loaded_files_get_current_node_groups(addon_key)

    print("[ACE smoke] Version, lifecycle, and scene frontend checks passed.")
    print(f"[ACE smoke] Enabled addon keys: {addon_keys}")


if __name__ == "__main__":
    main()
