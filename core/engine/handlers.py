"""Blender handlers and timers that drive the ACE engine."""

import bpy
import numpy as np
import time
from ... import utils
from ..tasks.analyzer import AudioFluxAnalyzer
from ..tasks.specs import SourceSpecsTask
from .common import _NUMBA_AVAILABLE, _preview_limiter_block_numba, persistent
from .engine import ACEAudioEngine
from .frontends import FrontendTracker
from .preview.router import PreviewRouter


@persistent
def _load_post_reset_engine(_dummy):
    ACEAudioEngine.shutdown_all_instances()
    _refresh_loaded_node_groups()
    _migrate_legacy_nodes()
    ACEAudioEngine.ensure_instance(getattr(getattr(bpy, "context", None), "scene", None))


def hard_reset_engine(scene):
    """Manual reset: drop every engine, worker and cache, bring the nodes up to date, compile anew."""
    _load_post_reset_engine(None)
    AudioFluxAnalyzer._GLOBAL_SPECTRUM_CACHE.clear()
    SourceSpecsTask._GLOBAL_LONG_TERM_CACHE.clear()
    engine = ACEAudioEngine.ensure_instance(scene)
    if engine.discover_scene_frontends(scene):
        engine.compile_scene_frontends(scene)
        engine.execute_next_frame(scene, force=True)
    return engine


def _refresh_loaded_node_groups():
    from ...nodes.geometry.factory import ACEDummyNodeFactory
    from ...nodes.shader.factory import ACEShaderNodeFactory

    for factory in (ACEDummyNodeFactory, ACEShaderNodeFactory):
        factory._NODES_CONTENT_HASH = ""
        try:
            factory.ensure_node_groups_exist()
        except Exception as exc:  # noqa: BLE001 boundary: a load must not fail on group refresh
            utils.log(f"ACE node group refresh after load failed ({factory.__name__}): {exc}", "ERROR")


def _migrate_legacy_nodes():
    """Bring the open file's nodes up to date: old stream nodes become mode nodes, split frequencies get ordered."""
    from ...nodes.modes import migrate_legacy_stream_nodes
    from ...nodes.split_freqs import sort_all_split_freqs
    from ...nodes.dynamic_audio_operators import drop_input_counts_from_math_titles
    from ...nodes.frontend import convert_rolling_average_groups

    try:
        convert_rolling_average_groups()
        migrate_legacy_stream_nodes()
        sort_all_split_freqs()
        drop_input_counts_from_math_titles()
    except Exception as exc:  # noqa: BLE001 boundary: a load must not fail on node migration
        utils.log(f"ACE node migration failed: {exc}", "ERROR")


def _migrate_after_register():
    """One-shot timer: bpy.data is restricted while the add-on registers."""
    _migrate_legacy_nodes()
    return None


@persistent
def _undo_redo_invalidate_engine_caches(_dummy):
    for engine in ACEAudioEngine.iter_instances():
        with utils.logged_errors("handlers._undo_redo_invalidate_engine_caches"):
            engine.begin_undo_recovery(getattr(getattr(bpy, "context", None), "scene", None))


@persistent
def frame_change_handler(scene):
    """bpy.app.handlers.frame_change_pre: run one DSP frame for the current scene."""
    if scene is None:
        scene = getattr(getattr(bpy, "context", None), "scene", None)
    if scene is None:
        return
    engine = ACEAudioEngine.ensure_instance(scene)
    if not engine:
        return
    if engine._undo_recovery_active():
        return
    is_rendering = engine._is_rendering_job()
    if engine._unsafe_for_handler_writes(scene):
        utils.log_throttled(
            "render_lock_interface",
            "Skipping ACE frame writes during render. Enable Render > Lock Interface for safe handler updates.",
            "WARNING",
            min_interval=5.0,
        )
        return
    if not engine.graphs:
        now = time.time()
        if now < getattr(engine, "_bootstrap_retry_at", 0.0):
            engine.frame.execute_next_frame(scene, force=is_rendering)
            return
        try:
            frontends = engine.frontends.discover_scene_frontends(scene)
            if frontends:
                engine.compiler.compile_scene_frontends(scene)
                engine._bootstrap_fail_count = 0
                engine._bootstrap_retry_at = now + 0.25
            else:
                engine._bootstrap_retry_at = now + 0.25
        except Exception as e:
            fail_count = int(getattr(engine, "_bootstrap_fail_count", 0)) + 1
            engine._bootstrap_fail_count = fail_count
            engine._bootstrap_retry_at = now + min(2.0, 0.2 * (2 ** min(fail_count, 4)))
            utils.log_throttled("bootstrap_compile", f"Frame bootstrap compile failed: {e}", "ERROR", min_interval=2.0)
    engine.frame.execute_next_frame(scene, force=is_rendering)


@persistent
def frame_change_post_handler(scene):
    """bpy.app.handlers.frame_change_post: secondary frame evaluation during rendering."""
    if scene is None:
        scene = getattr(getattr(bpy, "context", None), "scene", None)
    if scene is None:
        return
    engine = ACEAudioEngine.get_instance(scene)
    if engine is not None and engine._undo_recovery_active():
        return
    is_rendering = ACEAudioEngine._is_rendering_job()
    if not is_rendering:
        return
    frame_change_handler(scene)


@persistent
def render_pre_handler(scene):
    """bpy.app.handlers.render_pre: compile and warm the DSP graph before rendering."""
    if scene is None:
        scene = getattr(getattr(bpy, "context", None), "scene", None)
    if scene is None:
        return
    engine = ACEAudioEngine.ensure_instance(scene)
    if engine is None:
        return
    if engine._unsafe_for_handler_writes(scene):
        utils.log_throttled(
            "render_lock_interface_pre",
            "Skipping ACE render bootstrap. Enable Render > Lock Interface for safe handler updates.",
            "WARNING",
            min_interval=5.0,
        )
        engine.diagnostics._mark_render_start(scene)
        return
    if not engine.graphs:
        try:
            if engine.frontends.discover_scene_frontends(scene):
                engine.compiler.compile_scene_frontends(scene)
        except Exception as e:
            utils.log_throttled("render_pre_compile", f"Render pre compile failed: {e}", "ERROR", min_interval=1.5)
    try:
        engine.frame.execute_next_frame(scene, force=True)
    except Exception as e:
        utils.log_throttled("render_pre_execute", f"Render pre execute failed: {e}", "ERROR", min_interval=1.5)
    engine.diagnostics._mark_render_start(scene)


@persistent
def render_post_handler(scene):
    """bpy.app.handlers.render_post: finalise engine state after a successful render."""
    engine = ACEAudioEngine.get_instance(scene)
    if engine is None:
        return
    engine.diagnostics._mark_render_complete(scene, cancelled=False)
    with utils.logged_errors("handlers.render_post_handler"):
        PreviewRouter._preview_monitor.clear()


@persistent
def render_cancel_handler(scene):
    """bpy.app.handlers.render_cancel: clean up engine state after a cancelled render."""
    engine = ACEAudioEngine.get_instance(scene)
    if engine is not None:
        engine.diagnostics._mark_render_complete(scene, cancelled=True)
    render_post_handler(scene)


def _original_id(update):
    update_id = getattr(update, "id", None)
    return getattr(update_id, "original", None) or update_id


def _id_type_name(id_block) -> str:
    identifier = getattr(getattr(id_block, "bl_rna", None), "identifier", "")
    return str(identifier or getattr(id_block, "bl_idname", "") or "")


@persistent
def depsgraph_update_handler(scene, depsgraph):
    """bpy.app.handlers.depsgraph_update_post: react to node-tree edits in real time."""
    engine = ACEAudioEngine.get_instance(scene)
    ace = getattr(scene, "audiocore", None)

    def _update_touches_ace_tree(update_id) -> bool:
        bl_idname = _id_type_name(update_id)
        if bl_idname in {"GeometryNodeTree", "ShaderNodeTree"}:
            return bool(FrontendTracker._tree_is_ace_managed(update_id))
        if bl_idname == "Material":
            tree = getattr(update_id, "node_tree", None) if update_id is not None else None
            return bool(FrontendTracker._tree_is_ace_managed(tree))
        if bl_idname == "Object":
            obj = update_id
            for mod in getattr(obj, "modifiers", []) or []:
                tree_ref = getattr(mod, "node_group", None)
                if getattr(mod, "type", None) == "NODES" and FrontendTracker._tree_is_ace_managed(tree_ref):
                    return True
            for slot in getattr(obj, "material_slots", []) or []:
                material = getattr(slot, "material", None)
                tree = getattr(material, "node_tree", None) if material is not None else None
                if FrontendTracker._tree_is_ace_managed(tree):
                    return True
        return False

    if engine is not None and engine._undo_recovery_active():
        for update in getattr(depsgraph, "updates", []) or []:
            if _update_touches_ace_tree(_original_id(update)):
                with utils.logged_errors("handlers.depsgraph_update_handler"):
                    engine.frontends.request_recompile(None)
                break
        return

    if engine is not None:
        engine.diagnostics._consume_geometry_eval_updates(scene, depsgraph)
        # Probe file metadata early so the next compile has the native sample rate.
        _ace_data = ace
        for _src in getattr(_ace_data, "audio_sources", []):
            if getattr(_src, "source_type", "") == "FILE":
                _fp = getattr(_src, "filepath", "")
                if _fp and (
                    int(getattr(_src, "native_sample_rate", 0) or 0) <= 0
                    or float(getattr(_src, "duration", 0.0) or 0.0) <= 0.0
                ):
                    engine.probes._dispatch_async_probe(_fp)
    if ace is None:
        return

    engine = engine or ACEAudioEngine.ensure_instance(scene)
    if engine is None:
        return

    recompile_targets: list = []
    seen_targets: set[int] = set()

    def _queue_recompile_target(tree_obj) -> None:
        if tree_obj is None:
            return
        try:
            key = int(tree_obj.as_pointer())
        except Exception as exc:
            utils.note_swallowed("handlers.depsgraph_update_handler._queue_recompile_target", exc)
            key = id(tree_obj)
        if key in seen_targets:
            return
        seen_targets.add(key)
        recompile_targets.append(tree_obj)

    compiled_frontend_snapshots = dict(getattr(getattr(engine, "compiler", None), "_compiled_frontend_snapshots", {}) or {})
    discovered_frontends_by_uid = None

    for update in getattr(depsgraph, "updates", []):
        _id = _original_id(update)
        bl_idname = _id_type_name(_id)

        if bl_idname in {"GeometryNodeTree", "ShaderNodeTree"}:
            if FrontendTracker._tree_is_ace_managed(_id):
                _queue_recompile_target(_id)
            continue

        if bl_idname == "Material":
            tree = getattr(_id, "node_tree", None) if _id is not None else None
            if FrontendTracker._tree_is_ace_managed(tree):
                _queue_recompile_target(tree)
            continue

        if bl_idname == "Object":
            obj = _id
            has_update_flags = any(
                hasattr(update, token)
                for token in ("is_updated_geometry", "is_updated_shading", "is_updated_transform")
            )
            if has_update_flags and not (
                bool(getattr(update, "is_updated_geometry", False))
                or bool(getattr(update, "is_updated_shading", False))
            ):
                continue

            obj_name = str(getattr(obj, "name", "") or "")
            if obj_name and compiled_frontend_snapshots:
                if discovered_frontends_by_uid is None:
                    discovered_frontends_by_uid = {
                        str(rec.frontend_ref.uid or ""): getattr(rec, "tree", None)
                        for rec in (engine.frontends.discover_scene_frontends(scene) or [])
                    }
                for frontend_uid, frontend_snapshot in compiled_frontend_snapshots.items():
                    owners = {
                        str(v)
                        for v in tuple(getattr(frontend_snapshot, "owner_keys", ()) or ())
                        if str(v)
                    }
                    if obj_name not in owners:
                        continue
                    tree_ref = discovered_frontends_by_uid.get(str(frontend_uid))
                    if tree_ref is not None:
                        _queue_recompile_target(tree_ref)

            for mod in getattr(obj, "modifiers", []) or []:
                tree_ref = getattr(mod, "node_group", None)
                if getattr(mod, "type", None) == "NODES" and FrontendTracker._tree_is_ace_managed(tree_ref):
                    _queue_recompile_target(tree_ref)

            for slot in getattr(obj, "material_slots", []) or []:
                material = getattr(slot, "material", None)
                tree = getattr(material, "node_tree", None) if material is not None else None
                if FrontendTracker._tree_is_ace_managed(tree):
                    _queue_recompile_target(tree)

    for target in recompile_targets:
        engine.frontends.request_recompile(target, verify_content=True)


def auto_compile_timer():
    """bpy.app.timers callback (0.25 s): compile dirty trees."""
    with utils.logged_errors("handlers.auto_compile_timer"):
        scene = getattr(getattr(bpy, "context", None), "scene", None)
        if scene is None:
            return 0.25
        ace = getattr(scene, "audiocore", None)
        if ace:
            engine = ACEAudioEngine.ensure_instance(scene)
            if engine._undo_recovery_active():
                return 0.25
            if engine._unsafe_for_handler_writes(scene):
                return 0.25
            compiled = engine.frontends._maybe_compile_dirty(scene)
            if compiled:
                engine.frame.execute_next_frame(scene, force=True)
    return 0.25


def _shm_drain_timer():
    """Periodic SHM drain called by bpy.app.timers every 2 seconds."""
    try:
        scene = getattr(getattr(bpy, "context", None), "scene", None)
        engine = ACEAudioEngine.get_instance(scene) if scene is not None else None
        if engine is not None and engine.parallel.future_meta:
            if engine._undo_recovery_active():
                return 2.0
            # Only drain when the scene is NOT actively playing — during playback
            # execute_next_frame() handles all future draining every frame.
            if not engine._is_animation_playing():
                engine.parallel.poll(scene, current_frame=int(getattr(engine, "current_frame", 0)))
    except Exception as _exc:
        with utils.logged_errors("handlers._shm_drain_timer"):
            utils.log(
                f"[ACE-SHM] Drain timer error — completed futures may not be released "
                f"and shared memory may leak: {_exc!r}",
                "WARNING",
            )
    return 2.0


def _ace_handler_registered(handler_list, func) -> bool:
    target_name = getattr(func, "__qualname__", None)
    if target_name is None:
        return func in handler_list
    return any(getattr(h, "__qualname__", None) == target_name for h in handler_list)


def _ace_handler_remove_all(handler_list, func):
    target_name = getattr(func, "__qualname__", None)
    if target_name is None:
        to_remove = [h for h in handler_list if h is func]
    else:
        to_remove = [h for h in handler_list if getattr(h, "__qualname__", None) == target_name]
    for h in to_remove:
        with utils.logged_errors("handlers._ace_handler_remove_all"):
            handler_list.remove(h)


def register_handlers():
    """Register all ACE ``bpy.app.handlers`` and ``bpy.app.timers``."""
    # Trigger Numba JIT compilation at addon load time so the first preview frame
    # does not pay the cold-start compilation penalty (~50–200 ms).
    if _NUMBA_AVAILABLE:
        with utils.logged_errors("handlers.register_handlers"):
            _dummy = np.zeros((4, 2), dtype=np.float32)
            _preview_limiter_block_numba(_dummy, 0.891, 0.0001, 1.0)

    if not _ace_handler_registered(bpy.app.handlers.load_post, _load_post_reset_engine):
        bpy.app.handlers.load_post.append(_load_post_reset_engine)
    if not _ace_handler_registered(bpy.app.handlers.undo_post, _undo_redo_invalidate_engine_caches):
        bpy.app.handlers.undo_post.append(_undo_redo_invalidate_engine_caches)
    if not _ace_handler_registered(bpy.app.handlers.redo_post, _undo_redo_invalidate_engine_caches):
        bpy.app.handlers.redo_post.append(_undo_redo_invalidate_engine_caches)
    if not _ace_handler_registered(bpy.app.handlers.frame_change_pre, frame_change_handler):
        bpy.app.handlers.frame_change_pre.append(frame_change_handler)
    if not _ace_handler_registered(bpy.app.handlers.frame_change_post, frame_change_post_handler):
        bpy.app.handlers.frame_change_post.append(frame_change_post_handler)
    if not _ace_handler_registered(bpy.app.handlers.render_pre, render_pre_handler):
        bpy.app.handlers.render_pre.append(render_pre_handler)
    if not _ace_handler_registered(bpy.app.handlers.render_post, render_post_handler):
        bpy.app.handlers.render_post.append(render_post_handler)
    if not _ace_handler_registered(bpy.app.handlers.render_cancel, render_cancel_handler):
        bpy.app.handlers.render_cancel.append(render_cancel_handler)
    if not _ace_handler_registered(bpy.app.handlers.depsgraph_update_post, depsgraph_update_handler):
        bpy.app.handlers.depsgraph_update_post.append(depsgraph_update_handler)
    if not bpy.app.timers.is_registered(auto_compile_timer):
        bpy.app.timers.register(auto_compile_timer, first_interval=0.25, persistent=True)
    if not bpy.app.timers.is_registered(_shm_drain_timer):
        bpy.app.timers.register(_shm_drain_timer, first_interval=2.0, persistent=True)
    if not bpy.app.timers.is_registered(_migrate_after_register):
        bpy.app.timers.register(_migrate_after_register, first_interval=0.1, persistent=True)
    ACEAudioEngine.ensure_instance(getattr(getattr(bpy, "context", None), "scene", None))


def unregister_handlers():
    """Unregister all ACE ``bpy.app.handlers`` and ``bpy.app.timers``."""
    _ace_handler_remove_all(bpy.app.handlers.load_post, _load_post_reset_engine)
    _ace_handler_remove_all(bpy.app.handlers.undo_post, _undo_redo_invalidate_engine_caches)
    _ace_handler_remove_all(bpy.app.handlers.redo_post, _undo_redo_invalidate_engine_caches)
    _ace_handler_remove_all(bpy.app.handlers.frame_change_pre, frame_change_handler)
    _ace_handler_remove_all(bpy.app.handlers.frame_change_post, frame_change_post_handler)
    _ace_handler_remove_all(bpy.app.handlers.render_pre, render_pre_handler)
    _ace_handler_remove_all(bpy.app.handlers.render_post, render_post_handler)
    _ace_handler_remove_all(bpy.app.handlers.render_cancel, render_cancel_handler)
    _ace_handler_remove_all(bpy.app.handlers.depsgraph_update_post, depsgraph_update_handler)
    if bpy.app.timers.is_registered(auto_compile_timer):
        bpy.app.timers.unregister(auto_compile_timer)
    if bpy.app.timers.is_registered(_shm_drain_timer):
        bpy.app.timers.unregister(_shm_drain_timer)

    ACEAudioEngine.shutdown_all_instances()
