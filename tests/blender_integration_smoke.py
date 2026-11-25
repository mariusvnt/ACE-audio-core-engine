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


def _write_test_wav(path, seconds=3.0, sample_rate=48000):
    """Write a mono 16-bit WAV: quiet 220 Hz for the first half, loud 440 Hz after."""
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


def _add_ace_node(addon_key, tree, category, node_type, label):
    """Add an ACE group node the way the Add menu does (that operator needs a node editor)."""
    gn_menu = importlib.import_module(f"{addon_key}.ui.gn_menu")
    dummy_tree = importlib.import_module(f"{addon_key}.nodes.dummy_tree")
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
    op._initialize_auto_attr_ids(node, bpy.context)
    op._show_attr_id_socket_values(node)
    return node


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
    rms = _add_ace_node(addon_key, tree, "stream", "rms", "RMS")
    lufs = _add_ace_node(addon_key, tree, "stream", "lufs", "LUFS")
    split = _add_ace_node(addon_key, tree, "delta", "frequency_splitter", "Splitter")
    band_rms = _add_ace_node(addon_key, tree, "stream", "rms", "Band RMS")
    audio_out = source_node.outputs["Audio"]
    tree.links.new(audio_out, rms.inputs["Audio"])
    tree.links.new(audio_out, lufs.inputs["Audio"])
    tree.links.new(audio_out, split.inputs[0])
    tree.links.new(split.outputs[0], band_rms.inputs["Audio"])

    scene.frame_start, scene.frame_end = 1, 72
    scene.audiocore.engine_settings.auto_compile = True
    engine.compile_scene_frontends(scene)
    _assert(bool(engine.graphs), "No runtime graph compiled for the file source")
    import time
    engine_mod = importlib.import_module(f"{addon_key}.core.audio_engine")
    handlers_mod = importlib.import_module(f"{addon_key}.core.engine_handlers")
    # Let the one legitimate recompile happen first: the file's sample-rate
    # probe finishes in the background and asks for it.
    settle_compiles = []
    original_compile = engine.compile_scene_frontends
    engine.compile_scene_frontends = lambda *a, **k: (settle_compiles.append(time.monotonic()), original_compile(*a, **k))[1]
    t_settle = last_change = time.monotonic()
    while time.monotonic() - last_change < 0.5 and time.monotonic() - t_settle < 10.0:
        before = len(settle_compiles)
        scene.frame_set(1 + int((time.monotonic() - t_settle) * 24) % 24)
        handlers_mod.auto_compile_timer()
        if len(settle_compiles) != before:
            last_change = time.monotonic()
        time.sleep(1.0 / 24.0)
    engine.compile_scene_frontends = original_compile
    _assert(len(settle_compiles) <= 1, f"Scene recompiled {len(settle_compiles)} times while settling")
    cold_start_s = None
    if parallel_mode == "processes_dsp":
        # Workers spawn on first use and take seconds to import; play until one answers.
        t_warm = time.monotonic()
        frame = 1
        while int(engine.parallel._parallel_result_accepts) == 0 and time.monotonic() - t_warm < 60.0:
            scene.frame_set(1 + frame % 2)
            frame += 1
            time.sleep(1.0 / 24.0)
        cold_start_s = round(time.monotonic() - t_warm, 2)
        scene.frame_set(72)
        scene.frame_set(1)
    quiet = loud = None
    first_accept_s = None
    compiles = []
    original_compile = engine.compile_scene_frontends
    engine.compile_scene_frontends = lambda *a, **k: (compiles.append(1), original_compile(*a, **k))[1]
    t0 = time.monotonic()
    for frame in range(1, 73):
        # Real-time pace (24 fps): worker results arrive frames later.
        target = t0 + (frame - 1) / 24.0
        time.sleep(max(0.0, target - time.monotonic()))
        scene.frame_set(frame)
        handlers_mod.auto_compile_timer()  # timers do not run in background mode
        if first_accept_s is None and int(getattr(getattr(engine, "parallel", None), "_parallel_result_accepts", 0) or 0) > 0:
            first_accept_s = time.monotonic() - t0
        if frame == 30:
            quiet = _proxy_values(engine)
        if frame == 72:
            loud = _proxy_values(engine)
    engine.compile_scene_frontends = original_compile
    _assert(not compiles, f"Playback without edits recompiled the scene {len(compiles)} time(s)")
    if parallel_mode == "processes_dsp":
        print(f"[ACE smoke] processes_dsp: cold start {cold_start_s} s; first worker result after {first_accept_s} s, "
              f"accepted={engine.parallel._parallel_result_accepts}, stale_job_drops={engine.parallel._parallel_stale_job_drops}, "
              f"queue_pressure_drops={engine.parallel._parallel_queue_pressure_drops}")
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
        state = {
            name: getattr(engine, name, None)
            for name in (
                "_parallel_result_accepts", "_parallel_job_exceptions", "_parallel_pool_failed",
                "_parallel_disabled_sources", "_parallel_failures", "_parallel_stale_result_drops",
                "_parallel_stale_job_drops", "_parallel_queue_pressure_drops", "_parallel_restart_count",
                "_parallel_executor",
            )
        }
        state["offload_keys"] = {uid: list(getattr(g, "parallel_offload_attr_keys", []) or []) for uid, g in engine.graphs.items()}
        _assert(int(engine.parallel._parallel_result_accepts) > 0, f"No worker-process DSP result was accepted: {state}")

    _assert(engine.bake_source_to_cache(src, tree=tree, scene=scene), "Bake did not start")
    import time
    deadline = time.monotonic() + 60.0
    while str(src.get("ace_bake_status", "")) == "baking" and time.monotonic() < deadline:
        engine.flush_bake_results()
        time.sleep(0.05)
    engine.flush_bake_results()
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
    engine_mod = importlib.import_module(f"{addon_key}.core.audio_engine")
    handlers_mod = importlib.import_module(f"{addon_key}.core.engine_handlers")
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

    _check_file_source_drives_proxy(addon_key, engine, scene, src, parallel_mode="threads_io")
    _check_file_source_drives_proxy(addon_key, engine, scene, src, parallel_mode="processes_dsp")

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
