"""
Run inside Blender (background or UI) after enabling the add-on.

Example:
  blender --background --python tests/blender_integration_smoke.py
"""

import bpy  # type: ignore
import importlib


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    print("[ACE smoke] Blender-side smoke script loaded.")

    # Basic sanity checks users can expand:
    _assert(bpy.app.version >= (5, 0, 0), "Blender 5.0+ is required")
    addon_keys = [k for k in bpy.context.preferences.addons.keys() if "AudioCore" in k]
    _assert(bool(addon_keys), "AudioCore add-on is not enabled in current Blender session")
    addon_key = addon_keys[0]
    engine_mod = importlib.import_module(f"{addon_key}.audio_engine")
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

    remove_result = bpy.ops.audiocore.remove_audio_source()
    _assert('FINISHED' in remove_result, "audiocore.remove_audio_source failed")
    _assert(len(ace.audio_sources) == before_count, "Audio source count did not return to initial value")

    print("[ACE smoke] Version and lifecycle checks passed.")
    print(f"[ACE smoke] Enabled addon keys: {addon_keys}")


if __name__ == "__main__":
    main()
